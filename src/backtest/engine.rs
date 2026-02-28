use super::portfolio::{Portfolio, TradeRecord};
use crate::domain::*;
use crate::engine::learner::LearningEngine;
use crate::engine::pattern_library::{CandleSlices, PatternLibrary};
use crate::engine::trade_manager::{ExitReason, TradeAction, TradeManager};
use crate::evaluation::metrics;
use crate::strategy::features::MarketFeatures;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;

/// Backtest engine with strict anti-look-ahead enforcement.
/// Trading logic lives in `TradeManager` and `LearningEngine`.
/// This struct is responsible only for simulation orchestration:
/// feeding candles, managing portfolio, and collecting results.
pub struct BacktestEngine {
    pub portfolio: Portfolio,
    pub learning: LearningEngine,
    pub library: PatternLibrary,
    rng: StdRng,
    trades: TradeManager,
    /// ANTI-LOOKAHEAD: tracks the furthest candle index we've seen
    max_seen_index: usize,
    /// Funding rates indexed by timestamp for O(1) lookup
    funding_rates: HashMap<i64, f64>,
}

impl BacktestEngine {
    pub fn new(config: FuturesConfig, learning: LearningEngine, seed: u64) -> Self {
        Self {
            portfolio: Portfolio::new(config),
            learning,
            library: PatternLibrary::with_defaults(),
            rng: StdRng::seed_from_u64(seed),
            trades: TradeManager::new(),
            max_seen_index: 0,
            funding_rates: HashMap::new(),
        }
    }

    /// Set funding rates for this backtest run.
    pub fn set_funding_rates(&mut self, rates: &[FundingRate]) {
        self.funding_rates = rates
            .iter()
            .map(|r| (r.funding_time, r.funding_rate))
            .collect();
    }

    /// Run backtest on a single symbol's candle data.
    ///
    /// ANTI-LOOK-AHEAD GUARANTEES:
    /// 1. Candles processed strictly in temporal order
    /// 2. MarketState at index i only contains candles [0..=i]
    /// 3. Decision made AFTER candle i closes, BEFORE candle i+1
    /// 4. max_seen_index enforces monotonic progression
    pub fn run(&mut self, symbol: &Symbol, candles: &[Candle], warmup: usize) -> BacktestResult {
        let lookback = 60;

        if candles.len() < warmup + lookback {
            return BacktestResult::empty();
        }

        // Verify temporal order
        for i in 1..candles.len() {
            assert!(
                candles[i].open_time >= candles[i - 1].open_time,
                "LOOK-AHEAD VIOLATION: candles not in temporal order at index {}",
                i
            );
        }

        for i in warmup..candles.len() {
            assert!(
                i >= self.max_seen_index,
                "LOOK-AHEAD VIOLATION: processing index {} after seeing {}",
                i,
                self.max_seen_index
            );
            self.max_seen_index = i;

            let candle = &candles[i];

            // Tick adaptive parameter engine
            self.learning.adaptive.tick_candle();

            // Update portfolio mark-to-market
            self.portfolio.update_mark(candle.close);

            // Apply funding fee if position is open
            if self.portfolio.has_position() {
                self.apply_funding_fee(candle);
            }

            // Check liquidation at worst-case price for current position side
            let liq_price = match self.portfolio.position.as_ref().map(|p| p.side) {
                Some(PositionSide::Long) => candle.low,
                Some(PositionSide::Short) => candle.high,
                _ => candle.close,
            };
            if self.portfolio.check_liquidation(liq_price) {
                self.trades.exit_reason = "liquidation".to_string();
                let close_price = match self.portfolio.position.as_ref().map(|p| p.side) {
                    Some(PositionSide::Long) => candle.low,
                    Some(PositionSide::Short) => candle.high,
                    _ => candle.close,
                };
                if let Some(mut record) = self.portfolio.close_position(
                    close_price,
                    candle.close_time,
                    self.trades.candles_held,
                    self.trades.max_adverse,
                    &self.trades.current_strategy,
                    self.trades.accumulated_funding,
                ) {
                    self.trades.on_exit(symbol, &mut record, &mut self.learning);
                    *self.portfolio.trade_log.last_mut().unwrap() = record;
                }
                self.trades.reset(&self.learning);
                continue;
            }

            // Position management (SL/TP check, excursion tracking, trailing, max hold)
            if self.portfolio.has_position() {
                let pos = self.portfolio.position.as_ref().unwrap();

                // Check SL/TP first (respects min hold)
                if let Some((exit_price, reason)) =
                    self.trades.check_sl_tp(candle, pos, &self.learning)
                {
                    self.trades.exit_reason = reason.to_string();
                    if let Some(mut record) = self.portfolio.close_position(
                        exit_price,
                        candle.close_time,
                        self.trades.candles_held,
                        self.trades.max_adverse,
                        &self.trades.current_strategy,
                        self.trades.accumulated_funding,
                    ) {
                        self.trades.on_exit(symbol, &mut record, &mut self.learning);
                        *self.portfolio.trade_log.last_mut().unwrap() = record;
                    }
                    self.trades.reset(&self.learning);
                }
            }

            // Trailing stops, excursion tracking, max hold check
            if self.portfolio.has_position() {
                let pos = self.portfolio.position.as_ref().unwrap().clone();
                let action = self.trades.on_candle(candle, &pos, &self.learning);

                if let TradeAction::Exit { price, reason } = action {
                    self.trades.exit_reason = reason.to_string();
                    if let Some(mut record) = self.portfolio.close_position(
                        price,
                        candle.close_time,
                        self.trades.candles_held,
                        self.trades.max_adverse,
                        &self.trades.current_strategy,
                        self.trades.accumulated_funding,
                    ) {
                        self.trades.on_exit(symbol, &mut record, &mut self.learning);
                        *self.portfolio.trade_log.last_mut().unwrap() = record;
                    }
                    let cooldown = match reason {
                        ExitReason::MaxHold => (self.learning.adaptive.cooldown() * 2).min(48),
                        _ => self.learning.adaptive.cooldown(),
                    };
                    self.trades.reset_with_cooldown(&self.learning, cooldown);
                    continue;
                }
            }

            // Cooldown
            if self.trades.tick_cooldown() {
                continue;
            }

            // Build market state from ONLY past data
            let state = match build_market_state(symbol, candles, i, lookback) {
                Some(s) => s,
                None => continue,
            };

            let current_atr = state.indicators.atr_14;

            // Feed pattern library
            if let Some(features) = MarketFeatures::extract(&state) {
                let closes: Vec<f64> = candles[..=i].iter().map(|c| c.close).collect();
                let highs: Vec<f64> = candles[..=i].iter().map(|c| c.high).collect();
                let lows: Vec<f64> = candles[..=i].iter().map(|c| c.low).collect();
                let slices = CandleSlices {
                    closes: &closes,
                    highs: &highs,
                    lows: &lows,
                };
                self.library
                    .on_candle(&symbol.0, &features.as_array(), i, &slices);
            }

            // Make decision
            let decision = self.learning.decide(&state, &mut self.rng, &self.library);

            // Record hold observation periodically when not in position
            if !self.portfolio.has_position() && decision.signal == TradeSignal::Hold {
                if i % 10 == 0 {
                    if let Some(pattern) = self.learning.last_pattern.clone() {
                        let lookback_start = if i >= 10 { i - 10 } else { 0 };
                        let recent_return = (candle.close - candles[lookback_start].close)
                            / candles[lookback_start].close
                            * 100.0;
                        self.learning
                            .record_hold(&symbol.0, &pattern, recent_return);
                    }
                }
            }

            // Execute decision
            // When in a position, only allow signal flips after adaptive min_hold.
            // This prevents noisy Thompson samples from causing premature exits.
            // SL still handles risk during min_hold period (checked above).
            if self.portfolio.has_position() {
                let min_hold = self.learning.adaptive.min_hold();
                if self.trades.candles_held >= min_hold {
                    self.execute_decision(symbol, &decision, candle, current_atr);
                }
            } else {
                self.execute_decision(symbol, &decision, candle, current_atr);
            }
        }

        // Close any remaining position
        if self.portfolio.has_position() {
            let last_price = candles.last().unwrap().close;
            let last_time = candles.last().unwrap().close_time;
            self.trades.exit_reason = "end_of_data".to_string();
            if let Some(mut record) = self.portfolio.close_position(
                last_price,
                last_time,
                self.trades.candles_held,
                self.trades.max_adverse,
                &self.trades.current_strategy,
                self.trades.accumulated_funding,
            ) {
                self.trades.on_exit(symbol, &mut record, &mut self.learning);
                *self.portfolio.trade_log.last_mut().unwrap() = record;
            }
        }

        self.build_result()
    }

    fn execute_decision(
        &mut self,
        symbol: &Symbol,
        decision: &TradingDecision,
        candle: &Candle,
        atr: f64,
    ) {
        match decision.signal {
            TradeSignal::Hold => {}
            TradeSignal::Close => {
                if self.portfolio.has_position() {
                    self.trades.exit_reason = "signal_close".to_string();
                    if let Some(mut record) = self.portfolio.close_position(
                        candle.close,
                        candle.close_time,
                        self.trades.candles_held,
                        self.trades.max_adverse,
                        &decision.strategy_name,
                        self.trades.accumulated_funding,
                    ) {
                        self.trades.on_exit(symbol, &mut record, &mut self.learning);
                        *self.portfolio.trade_log.last_mut().unwrap() = record;
                    }
                    self.trades.reset(&self.learning);
                }
            }
            TradeSignal::Long => {
                if self.portfolio.has_position() {
                    let pos_side = self.portfolio.position.as_ref().unwrap().side;
                    if pos_side == PositionSide::Short {
                        self.trades.exit_reason = "signal_flip".to_string();
                        if let Some(mut record) = self.portfolio.close_position(
                            candle.close,
                            candle.close_time,
                            self.trades.candles_held,
                            self.trades.max_adverse,
                            &self.trades.current_strategy,
                            self.trades.accumulated_funding,
                        ) {
                            self.trades.on_exit(symbol, &mut record, &mut self.learning);
                            *self.portfolio.trade_log.last_mut().unwrap() = record;
                        }
                        self.trades.reset(&self.learning);
                    } else {
                        return; // already long
                    }
                }
                if self.portfolio.open_position(
                    symbol,
                    PositionSide::Long,
                    candle.close,
                    decision.size,
                    &decision.strategy_name,
                    candle.close_time,
                ) {
                    self.trades.on_entry(
                        decision,
                        atr,
                        self.learning.last_pattern.clone(),
                        self.portfolio.equity,
                        self.learning.last_prediction,
                    );
                }
            }
            TradeSignal::Short => {
                if self.portfolio.has_position() {
                    let pos_side = self.portfolio.position.as_ref().unwrap().side;
                    if pos_side == PositionSide::Long {
                        self.trades.exit_reason = "signal_flip".to_string();
                        if let Some(mut record) = self.portfolio.close_position(
                            candle.close,
                            candle.close_time,
                            self.trades.candles_held,
                            self.trades.max_adverse,
                            &self.trades.current_strategy,
                            self.trades.accumulated_funding,
                        ) {
                            self.trades.on_exit(symbol, &mut record, &mut self.learning);
                            *self.portfolio.trade_log.last_mut().unwrap() = record;
                        }
                        self.trades.reset(&self.learning);
                    } else {
                        return; // already short
                    }
                }
                if self.portfolio.open_position(
                    symbol,
                    PositionSide::Short,
                    candle.close,
                    decision.size,
                    &decision.strategy_name,
                    candle.close_time,
                ) {
                    self.trades.on_entry(
                        decision,
                        atr,
                        self.learning.last_pattern.clone(),
                        self.portfolio.equity,
                        self.learning.last_prediction,
                    );
                }
            }
        }
    }

    /// Apply funding fee if a funding event occurs during this candle.
    fn apply_funding_fee(&mut self, candle: &Candle) {
        let pos = match &self.portfolio.position {
            Some(p) => p,
            None => return,
        };

        let funding_interval_ms: i64 = 8 * 3600 * 1000;
        let first_funding = (candle.open_time / funding_interval_ms) * funding_interval_ms;

        let mut funding_time = first_funding;
        while funding_time <= candle.close_time {
            if funding_time >= candle.open_time && funding_time <= candle.close_time {
                let rate = self
                    .funding_rates
                    .get(&funding_time)
                    .copied()
                    .unwrap_or(0.0); // no data = no fee

                let notional = pos.size * pos.entry_price;
                let fee = match pos.side {
                    PositionSide::Long => notional * rate,
                    PositionSide::Short => -notional * rate,
                    PositionSide::Flat => 0.0,
                };

                self.trades.add_funding(fee);
                self.portfolio.equity -= fee;
            }
            funding_time += funding_interval_ms;
        }
    }

    fn build_result(&self) -> BacktestResult {
        let days = if self.portfolio.equity_curve.len() > 1 {
            (self.portfolio.equity_curve.len() - 1) as f64 / 96.0
        } else {
            1.0
        };

        let performance = metrics::calculate_metrics(
            &self.portfolio.equity_curve,
            &self.portfolio.trade_log,
            days,
        );

        BacktestResult {
            performance,
            trades: self.portfolio.trade_log.clone(),
            final_equity: self.portfolio.current_equity(),
            health: self.learning.health_report(),
        }
    }
}

/// Complete backtest result
pub struct BacktestResult {
    pub performance: metrics::PerformanceMetrics,
    pub trades: Vec<TradeRecord>,
    pub final_equity: f64,
    pub health: std::collections::HashMap<String, crate::engine::learner::SymbolHealth>,
}

impl BacktestResult {
    pub fn empty() -> Self {
        Self {
            performance: metrics::PerformanceMetrics {
                total_return_pct: 0.0,
                annualized_return_pct: 0.0,
                sharpe_ratio: 0.0,
                sortino_ratio: 0.0,
                max_drawdown_pct: 0.0,
                calmar_ratio: 0.0,
                recovery_factor: 0.0,
                win_rate: 0.0,
                profit_factor: 0.0,
                total_trades: 0,
                winning_trades: 0,
                losing_trades: 0,
                avg_win_pct: 0.0,
                avg_loss_pct: 0.0,
                expectancy: 0.0,
                median_trade_pct: 0.0,
                best_trade_pct: 0.0,
                worst_trade_pct: 0.0,
                avg_holding_periods: 0.0,
                max_consecutive_wins: 0,
                max_consecutive_losses: 0,
                trade_frequency: 0.0,
                long_trades: 0,
                short_trades: 0,
                long_win_rate: 0.0,
                short_win_rate: 0.0,
                long_avg_pnl: 0.0,
                short_avg_pnl: 0.0,
                avg_mae_pct: 0.0,
                avg_mfe_pct: 0.0,
                equity_curve: vec![],
            },
            trades: vec![],
            final_equity: 0.0,
            health: std::collections::HashMap::new(),
        }
    }

    pub fn print_summary(&self) {
        let p = &self.performance;
        let total_tx_fees: f64 = self.trades.iter().map(|t| t.fees_paid).sum();
        let total_funding: f64 = self.trades.iter().map(|t| t.funding_fees_paid).sum();
        println!("\n{}", "=".repeat(70));
        println!("  BACKTEST RESULTS");
        println!("{}", "=".repeat(70));
        println!("  Total Return:       {:>10.2}%", p.total_return_pct);
        println!("  Annualized Return:  {:>10.2}%", p.annualized_return_pct);
        println!("  Sharpe Ratio:       {:>10.2}", p.sharpe_ratio);
        println!("  Sortino Ratio:      {:>10.2}", p.sortino_ratio);
        println!("  Max Drawdown:       {:>10.2}%", p.max_drawdown_pct);
        println!("  Calmar Ratio:       {:>10.2}", p.calmar_ratio);
        println!("  Recovery Factor:    {:>10.2}", p.recovery_factor);
        println!("  Win Rate:           {:>10.2}%", p.win_rate * 100.0);
        println!("  Profit Factor:      {:>10.2}", p.profit_factor);
        println!("  Expectancy:         {:>10.3}%", p.expectancy);
        println!("  Total Trades:       {:>10}", p.total_trades);
        println!("  Winning Trades:     {:>10}", p.winning_trades);
        println!("  Losing Trades:      {:>10}", p.losing_trades);
        println!("  Avg Win:            {:>10.2}%", p.avg_win_pct);
        println!("  Avg Loss:           {:>10.2}%", p.avg_loss_pct);
        println!("  Median Trade:       {:>10.3}%", p.median_trade_pct);
        println!("  Best Trade:         {:>10.2}%", p.best_trade_pct);
        println!("  Worst Trade:        {:>10.2}%", p.worst_trade_pct);
        println!(
            "  Avg Holding:        {:>10.1} candles",
            p.avg_holding_periods
        );
        println!("  Trade Frequency:    {:>10.2}/day", p.trade_frequency);
        println!("  Max Consec. Wins:   {:>10}", p.max_consecutive_wins);
        println!("  Max Consec. Losses: {:>10}", p.max_consecutive_losses);

        println!("\n  --- Long / Short Breakdown ---");
        println!(
            "  Long:  {:>4} trades | win rate {:>5.1}% | avg {:>+7.3}%",
            p.long_trades,
            p.long_win_rate * 100.0,
            p.long_avg_pnl
        );
        println!(
            "  Short: {:>4} trades | win rate {:>5.1}% | avg {:>+7.3}%",
            p.short_trades,
            p.short_win_rate * 100.0,
            p.short_avg_pnl
        );

        println!("\n  --- Excursion Analysis ---");
        println!("  Avg MAE:            {:>10.3}%", p.avg_mae_pct);
        println!("  Avg MFE:            {:>10.3}%", p.avg_mfe_pct);

        // Exit reason distribution
        if !self.trades.is_empty() {
            let mut exit_counts: HashMap<String, usize> = HashMap::new();
            for t in &self.trades {
                *exit_counts.entry(t.exit_reason.clone()).or_insert(0) += 1;
            }
            let mut exits: Vec<_> = exit_counts.into_iter().collect();
            exits.sort_by(|a, b| b.1.cmp(&a.1));
            println!("\n  --- Exit Reasons ---");
            for (reason, count) in &exits {
                let pct = *count as f64 / self.trades.len() as f64 * 100.0;
                let avg_pnl: f64 = self
                    .trades
                    .iter()
                    .filter(|t| &t.exit_reason == reason)
                    .map(|t| t.pnl_pct)
                    .sum::<f64>()
                    / *count as f64;
                println!(
                    "  {:16} {:>4} ({:>5.1}%) avg {:>+7.3}%",
                    reason, count, pct, avg_pnl
                );
            }
        }

        // Comprehensive strategy performance
        if !self.trades.is_empty() {
            struct StratStats {
                pnls: Vec<f64>,
                wins: usize,
                total_pnl: f64,
                total_hold: usize,
                total_mae: f64,
                total_mfe: f64,
                total_fees: f64,
                total_funding: f64,
            }
            let mut strat_map: HashMap<String, StratStats> = HashMap::new();
            for t in &self.trades {
                if t.pattern.is_empty() {
                    continue;
                }
                let key = format!(
                    "{}_{}",
                    t.pattern,
                    match t.side {
                        crate::domain::PositionSide::Long => "L",
                        crate::domain::PositionSide::Short => "S",
                        _ => "?",
                    }
                );
                let entry = strat_map.entry(key).or_insert_with(|| StratStats {
                    pnls: Vec::new(),
                    wins: 0,
                    total_pnl: 0.0,
                    total_hold: 0,
                    total_mae: 0.0,
                    total_mfe: 0.0,
                    total_fees: 0.0,
                    total_funding: 0.0,
                });
                entry.pnls.push(t.pnl_pct);
                if t.pnl_pct > 0.0 {
                    entry.wins += 1;
                }
                entry.total_pnl += t.pnl_pct;
                entry.total_hold += t.holding_periods;
                entry.total_mae += t.max_adverse_excursion;
                entry.total_mfe += t.max_favorable_excursion;
                entry.total_fees += t.fees_paid;
                entry.total_funding += t.funding_fees_paid;
            }
            let mut strats: Vec<_> = strat_map.into_iter().collect();
            strats.sort_by(|a, b| b.1.total_pnl.partial_cmp(&a.1.total_pnl).unwrap());

            println!("\n  --- Strategy Performance (Primary) ---");
            println!(
                "  {:20} {:>5} {:>6} {:>8} {:>8} {:>6} {:>5} {:>5} {:>6} {:>8} {:>8}",
                "Strategy",
                "Trd",
                "Win%",
                "AvgPnL%",
                "TotPnL%",
                "PF",
                "Hold",
                "MAE%",
                "MFE%",
                "TxFees",
                "FundFee"
            );
            for (pat, s) in &strats {
                let n = s.pnls.len();
                let wr = if n > 0 {
                    s.wins as f64 / n as f64 * 100.0
                } else {
                    0.0
                };
                let avg_pnl = if n > 0 { s.total_pnl / n as f64 } else { 0.0 };
                let avg_hold = if n > 0 {
                    s.total_hold as f64 / n as f64
                } else {
                    0.0
                };
                let avg_mae = if n > 0 { s.total_mae / n as f64 } else { 0.0 };
                let avg_mfe = if n > 0 { s.total_mfe / n as f64 } else { 0.0 };
                let gross_wins: f64 = s.pnls.iter().filter(|&&p| p > 0.0).sum();
                let gross_losses: f64 = s.pnls.iter().filter(|&&p| p < 0.0).map(|p| p.abs()).sum();
                let pf = if gross_losses > 0.0 {
                    gross_wins / gross_losses
                } else if gross_wins > 0.0 {
                    99.0
                } else {
                    0.0
                };
                let pat_display = if pat.len() > 20 {
                    &pat[..20]
                } else {
                    pat.as_str()
                };
                println!("  {:20} {:>5} {:>5.1}% {:>+7.3}% {:>+7.02}% {:>5.2} {:>5.1} {:>5.2} {:>5.2} ${:>7.2} ${:>7.2}",
                    pat_display, n, wr, avg_pnl, s.total_pnl, pf,
                    avg_hold, avg_mae, avg_mfe, s.total_fees, s.total_funding);
            }

            // Extended strategy metrics
            println!("\n  --- Strategy Performance (Extended) ---");
            println!(
                "  {:20} {:>7} {:>7} {:>8} {:>6} {:>6} {:>8} {:>7}",
                "Strategy", "Sharpe", "Sortno", "MaxDD%", "CWin", "CLoss", "Expect%", "Ret/DD"
            );
            for (pat, s) in &strats {
                let n = s.pnls.len();
                if n == 0 {
                    continue;
                }

                // Sharpe: mean / std of trade returns
                let mean = s.total_pnl / n as f64;
                let variance = s.pnls.iter().map(|p| (p - mean).powi(2)).sum::<f64>() / n as f64;
                let std = variance.sqrt();
                let sharpe = if std > 1e-10 {
                    mean / std
                } else if mean > 0.0 {
                    99.0
                } else {
                    0.0
                };

                // Sortino: mean / downside std
                let downside_sq: f64 = s.pnls.iter().filter(|&&p| p < 0.0).map(|p| p.powi(2)).sum();
                let downside_std = (downside_sq / n as f64).sqrt();
                let sortino = if downside_std > 1e-10 {
                    mean / downside_std
                } else if mean > 0.0 {
                    99.0
                } else {
                    0.0
                };

                // Max drawdown of strategy equity curve
                let mut peak = 0.0_f64;
                let mut max_dd = 0.0_f64;
                let mut cumulative = 0.0;
                for &pnl in &s.pnls {
                    cumulative += pnl;
                    if cumulative > peak {
                        peak = cumulative;
                    }
                    let dd = peak - cumulative;
                    if dd > max_dd {
                        max_dd = dd;
                    }
                }

                // Max consecutive wins / losses
                let (mut cw, mut cl, mut max_cw, mut max_cl) = (0u32, 0u32, 0u32, 0u32);
                for &pnl in &s.pnls {
                    if pnl > 0.0 {
                        cw += 1;
                        cl = 0;
                        if cw > max_cw {
                            max_cw = cw;
                        }
                    } else {
                        cl += 1;
                        cw = 0;
                        if cl > max_cl {
                            max_cl = cl;
                        }
                    }
                }

                // Expectancy: win_rate * avg_win - loss_rate * avg_loss
                let avg_win = if s.wins > 0 {
                    s.pnls.iter().filter(|&&p| p > 0.0).sum::<f64>() / s.wins as f64
                } else {
                    0.0
                };
                let losses = n - s.wins;
                let avg_loss = if losses > 0 {
                    s.pnls
                        .iter()
                        .filter(|&&p| p <= 0.0)
                        .map(|p| p.abs())
                        .sum::<f64>()
                        / losses as f64
                } else {
                    0.0
                };
                let wr = s.wins as f64 / n as f64;
                let expectancy = wr * avg_win - (1.0 - wr) * avg_loss;

                // Risk-adjusted return: total return / max drawdown
                let ret_dd = if max_dd > 1e-10 {
                    s.total_pnl / max_dd
                } else if s.total_pnl > 0.0 {
                    99.0
                } else {
                    0.0
                };

                let pat_display = if pat.len() > 20 {
                    &pat[..20]
                } else {
                    pat.as_str()
                };
                println!(
                    "  {:20} {:>+6.2} {:>+6.2} {:>+7.2}% {:>5} {:>5} {:>+7.3}% {:>+6.2}",
                    pat_display, sharpe, sortino, max_dd, max_cw, max_cl, expectancy, ret_dd
                );
            }
        }

        println!("\n  --- Fees ---");
        println!("  TX Fees Paid:       ${:>10.2}", total_tx_fees);
        println!("  Funding Fees Paid:  ${:>10.2}", total_funding);
        println!("  Final Equity:       ${:>10.2}", self.final_equity);

        // === EXTENDED REPORTING FOR STRATEGY IMPROVEMENT CONTEXT ===

        // Drawdown analysis
        if p.equity_curve.len() > 1 {
            println!("\n  --- Drawdown Analysis ---");
            let mut peak = p.equity_curve[0];
            let mut max_dd_start = 0usize;
            let mut max_dd_end = 0usize;
            let mut current_dd_start = 0usize;
            let mut max_dd_val = 0.0_f64;
            let mut in_drawdown = false;
            let mut dd_count = 0usize;
            let mut dd_durations = Vec::new();
            let mut current_dd_depth = 0.0_f64;

            for (i, &eq) in p.equity_curve.iter().enumerate() {
                if eq > peak {
                    if in_drawdown {
                        dd_durations.push(i - current_dd_start);
                        in_drawdown = false;
                    }
                    peak = eq;
                } else {
                    let dd = (peak - eq) / peak * 100.0;
                    if dd > 0.01 && !in_drawdown {
                        in_drawdown = true;
                        current_dd_start = i;
                        dd_count += 1;
                    }
                    if dd > current_dd_depth {
                        current_dd_depth = dd;
                    }
                    if dd > max_dd_val {
                        max_dd_val = dd;
                        max_dd_start = current_dd_start;
                        max_dd_end = i;
                    }
                }
            }
            if in_drawdown {
                dd_durations.push(p.equity_curve.len() - 1 - current_dd_start);
            }

            let max_dd_duration = max_dd_end.saturating_sub(max_dd_start);
            let avg_dd_duration = if dd_durations.is_empty() {
                0.0
            } else {
                dd_durations.iter().sum::<usize>() as f64 / dd_durations.len() as f64
            };
            let longest_dd = dd_durations.iter().copied().max().unwrap_or(0);
            // Time underwater = % of candles in drawdown
            let underwater_candles: usize = dd_durations.iter().sum();
            let underwater_pct =
                underwater_candles as f64 / p.equity_curve.len().max(1) as f64 * 100.0;

            println!(
                "  Max DD Duration:    {:>10} candles ({:.1} days)",
                max_dd_duration,
                max_dd_duration as f64 / 96.0
            );
            println!(
                "  Longest Underwater: {:>10} candles ({:.1} days)",
                longest_dd,
                longest_dd as f64 / 96.0
            );
            println!(
                "  Avg DD Duration:    {:>10.0} candles ({:.1} days)",
                avg_dd_duration,
                avg_dd_duration / 96.0
            );
            println!("  Drawdown Periods:   {:>10}", dd_count);
            println!("  Time Underwater:    {:>9.1}%", underwater_pct);
        }

        // Confidence calibration — bucket trades by confidence and show performance
        if !self.trades.is_empty() {
            let conf_trades: Vec<_> = self.trades.iter().filter(|t| t.confidence > 0.0).collect();
            if !conf_trades.is_empty() {
                println!("\n  --- Confidence Calibration ---");
                let buckets = [
                    ("Low  (0.0-0.45)", 0.0, 0.45),
                    ("Med  (0.45-0.55)", 0.45, 0.55),
                    ("High (0.55-0.65)", 0.55, 0.65),
                    ("VHi  (0.65-1.0)", 0.65, 1.01),
                ];
                println!(
                    "  {:20} {:>5} {:>6} {:>8} {:>8} {:>6}",
                    "Bucket", "Trd", "Win%", "AvgPnL%", "TotPnL%", "PF"
                );
                for (label, lo, hi) in &buckets {
                    let bt: Vec<_> = conf_trades
                        .iter()
                        .filter(|t| t.confidence >= *lo && t.confidence < *hi)
                        .collect();
                    if bt.is_empty() {
                        continue;
                    }
                    let n = bt.len();
                    let wins = bt.iter().filter(|t| t.pnl_pct > 0.0).count();
                    let wr = wins as f64 / n as f64 * 100.0;
                    let tot: f64 = bt.iter().map(|t| t.pnl_pct).sum();
                    let avg = tot / n as f64;
                    let gw: f64 = bt.iter().filter(|t| t.pnl_pct > 0.0).map(|t| t.pnl_pct).sum();
                    let gl: f64 = bt
                        .iter()
                        .filter(|t| t.pnl_pct < 0.0)
                        .map(|t| t.pnl_pct.abs())
                        .sum();
                    let pf = if gl > 0.0 { gw / gl } else if gw > 0.0 { 99.0 } else { 0.0 };
                    println!(
                        "  {:20} {:>5} {:>5.1}% {:>+7.3}% {:>+7.2}% {:>5.2}",
                        label, n, wr, avg, tot, pf
                    );
                }
            }
        }

        // Holding period distribution — wins vs losses
        if !self.trades.is_empty() {
            println!("\n  --- Holding Period Analysis ---");
            let buckets = [
                ("1-4 candles (≤1h)", 1, 4),
                ("5-12 (1-3h)", 5, 12),
                ("13-24 (3-6h)", 13, 24),
                ("25-48 (6-12h)", 25, 48),
                ("49-96 (12-24h)", 49, 96),
                ("97+ (>24h)", 97, 9999),
            ];
            println!(
                "  {:20} {:>5} {:>6} {:>8} {:>5} {:>6}",
                "Duration", "Trd", "Win%", "AvgPnL%", "Wins", "Losses"
            );
            for (label, lo, hi) in &buckets {
                let bt: Vec<_> = self
                    .trades
                    .iter()
                    .filter(|t| t.holding_periods >= *lo && t.holding_periods <= *hi)
                    .collect();
                if bt.is_empty() {
                    continue;
                }
                let n = bt.len();
                let wins = bt.iter().filter(|t| t.pnl_pct > 0.0).count();
                let losses = n - wins;
                let wr = wins as f64 / n as f64 * 100.0;
                let avg: f64 = bt.iter().map(|t| t.pnl_pct).sum::<f64>() / n as f64;
                println!(
                    "  {:20} {:>5} {:>5.1}% {:>+7.3}% {:>5} {:>5}",
                    label, n, wr, avg, wins, losses
                );
            }

            // Win vs loss average holding
            let win_holds: Vec<_> = self
                .trades
                .iter()
                .filter(|t| t.pnl_pct > 0.0)
                .map(|t| t.holding_periods)
                .collect();
            let loss_holds: Vec<_> = self
                .trades
                .iter()
                .filter(|t| t.pnl_pct <= 0.0)
                .map(|t| t.holding_periods)
                .collect();
            let avg_win_hold = if win_holds.is_empty() {
                0.0
            } else {
                win_holds.iter().sum::<usize>() as f64 / win_holds.len() as f64
            };
            let avg_loss_hold = if loss_holds.is_empty() {
                0.0
            } else {
                loss_holds.iter().sum::<usize>() as f64 / loss_holds.len() as f64
            };
            println!(
                "  Avg Win Hold:  {:>5.1} candles ({:.1}h)  |  Avg Loss Hold: {:>5.1} candles ({:.1}h)",
                avg_win_hold,
                avg_win_hold * 0.25,
                avg_loss_hold,
                avg_loss_hold * 0.25
            );
        }

        // Position sizing analysis
        if !self.trades.is_empty() {
            let sizes: Vec<f64> = self.trades.iter().map(|t| t.position_size_frac).collect();
            let avg_size = sizes.iter().sum::<f64>() / sizes.len() as f64;
            let min_size = sizes.iter().copied().fold(f64::INFINITY, f64::min);
            let max_size = sizes.iter().copied().fold(0.0_f64, f64::max);

            // Correlation between size and outcome
            let pnls: Vec<f64> = self.trades.iter().map(|t| t.pnl_pct).collect();
            let avg_pnl = pnls.iter().sum::<f64>() / pnls.len() as f64;
            let mut cov = 0.0_f64;
            let mut var_s = 0.0_f64;
            let mut var_p = 0.0_f64;
            for (s, p) in sizes.iter().zip(pnls.iter()) {
                cov += (s - avg_size) * (p - avg_pnl);
                var_s += (s - avg_size).powi(2);
                var_p += (p - avg_pnl).powi(2);
            }
            let corr = if var_s > 1e-10 && var_p > 1e-10 {
                cov / (var_s.sqrt() * var_p.sqrt())
            } else {
                0.0
            };

            println!("\n  --- Position Sizing ---");
            println!("  Avg Size:           {:>10.3} (fraction of equity)", avg_size);
            println!("  Min/Max Size:       {:>10.3} / {:.3}", min_size, max_size);
            println!(
                "  Size-PnL Corr:      {:>10.3} {}",
                corr,
                if corr > 0.1 {
                    "(larger positions = better outcomes)"
                } else if corr < -0.1 {
                    "(larger positions = worse outcomes!)"
                } else {
                    "(no significant correlation)"
                }
            );
        }

        // Fee impact analysis
        if !self.trades.is_empty() {
            let gross_profit: f64 = self
                .trades
                .iter()
                .filter(|t| t.pnl_pct > 0.0)
                .map(|t| t.pnl)
                .sum();
            let gross_loss: f64 = self
                .trades
                .iter()
                .filter(|t| t.pnl_pct < 0.0)
                .map(|t| t.pnl.abs())
                .sum();
            let fee_pct_of_profit = if gross_profit > 0.0 {
                (total_tx_fees + total_funding) / gross_profit * 100.0
            } else {
                0.0
            };
            let fee_pct_of_volume = if !self.trades.is_empty() {
                let total_volume: f64 = self.trades.iter().map(|t| t.size_usd).sum();
                (total_tx_fees + total_funding) / total_volume * 100.0
            } else {
                0.0
            };
            let avg_fee_per_trade = (total_tx_fees + total_funding) / self.trades.len() as f64;

            println!("\n  --- Fee Impact ---");
            println!("  Gross Profit:       ${:>10.2}", gross_profit);
            println!("  Gross Loss:         ${:>10.2}", gross_loss);
            println!("  Total Fees:         ${:>10.2}", total_tx_fees + total_funding);
            println!("  Fees / Gross Profit: {:>9.1}%", fee_pct_of_profit);
            println!("  Fees / Volume:      {:>10.3}%", fee_pct_of_volume);
            println!("  Avg Fee/Trade:      ${:>10.2}", avg_fee_per_trade);
        }

        // Win/Loss streak distribution
        if !self.trades.is_empty() {
            let mut streaks_win: Vec<usize> = Vec::new();
            let mut streaks_loss: Vec<usize> = Vec::new();
            let mut current_win = 0usize;
            let mut current_loss = 0usize;
            for t in &self.trades {
                if t.pnl_pct > 0.0 {
                    current_win += 1;
                    if current_loss > 0 {
                        streaks_loss.push(current_loss);
                        current_loss = 0;
                    }
                } else {
                    current_loss += 1;
                    if current_win > 0 {
                        streaks_win.push(current_win);
                        current_win = 0;
                    }
                }
            }
            if current_win > 0 {
                streaks_win.push(current_win);
            }
            if current_loss > 0 {
                streaks_loss.push(current_loss);
            }

            println!("\n  --- Streak Analysis ---");
            let avg_win_streak = if streaks_win.is_empty() {
                0.0
            } else {
                streaks_win.iter().sum::<usize>() as f64 / streaks_win.len() as f64
            };
            let avg_loss_streak = if streaks_loss.is_empty() {
                0.0
            } else {
                streaks_loss.iter().sum::<usize>() as f64 / streaks_loss.len() as f64
            };
            println!(
                "  Win Streaks:   avg {:.1}, max {}, count {}",
                avg_win_streak,
                streaks_win.iter().max().unwrap_or(&0),
                streaks_win.len()
            );
            println!(
                "  Loss Streaks:  avg {:.1}, max {}, count {}",
                avg_loss_streak,
                streaks_loss.iter().max().unwrap_or(&0),
                streaks_loss.len()
            );
        }

        // MAE/MFE efficiency — how much of MFE is captured vs how much MAE is suffered
        if !self.trades.is_empty() {
            let trades_with_mfe: Vec<_> = self
                .trades
                .iter()
                .filter(|t| t.max_favorable_excursion > 0.01)
                .collect();
            if !trades_with_mfe.is_empty() {
                println!("\n  --- Trade Efficiency (MAE/MFE) ---");
                // For winners: PnL / MFE = how much of the run was captured
                let winners: Vec<_> = trades_with_mfe
                    .iter()
                    .filter(|t| t.pnl_pct > 0.0)
                    .collect();
                let losers: Vec<_> = trades_with_mfe
                    .iter()
                    .filter(|t| t.pnl_pct <= 0.0)
                    .collect();

                if !winners.is_empty() {
                    let avg_capture: f64 = winners
                        .iter()
                        .map(|t| t.pnl_pct / t.max_favorable_excursion.max(0.01))
                        .sum::<f64>()
                        / winners.len() as f64;
                    let avg_win_mae: f64 = winners
                        .iter()
                        .map(|t| t.max_adverse_excursion)
                        .sum::<f64>()
                        / winners.len() as f64;
                    println!(
                        "  Winners: avg capture {:.0}% of MFE, avg MAE {:.3}%",
                        avg_capture * 100.0,
                        avg_win_mae
                    );
                }
                if !losers.is_empty() {
                    let avg_loss_mfe: f64 = losers
                        .iter()
                        .map(|t| t.max_favorable_excursion)
                        .sum::<f64>()
                        / losers.len() as f64;
                    let avg_loss_mae: f64 = losers
                        .iter()
                        .map(|t| t.max_adverse_excursion)
                        .sum::<f64>()
                        / losers.len() as f64;
                    println!(
                        "  Losers:  avg MFE {:.3}% (unrealized), avg MAE {:.3}%",
                        avg_loss_mfe, avg_loss_mae
                    );
                    // How many losers actually had MFE > their loss
                    let could_have_won = losers
                        .iter()
                        .filter(|t| t.max_favorable_excursion > t.pnl_pct.abs())
                        .count();
                    println!(
                        "  Losers with MFE > |loss|: {} ({:.0}%) — were winners at some point",
                        could_have_won,
                        could_have_won as f64 / losers.len() as f64 * 100.0
                    );
                }
            }
        }

        println!("{}\n", "=".repeat(70));
    }

    /// Print the full trade log with all attributes.
    pub fn print_trade_log(&self) {
        if self.trades.is_empty() {
            println!("  No trades.");
            return;
        }
        println!("\n{}", "=".repeat(185));
        println!("  TRADE LOG ({} trades)", self.trades.len());
        println!("{}", "=".repeat(185));
        println!("  {:>3} | {:8} | {:5} | {:>10} | {:>10} | {:>8} | {:>4} | {:6} | {:>5} | {:>5} | {:14} | {:>6} | {:>8} | {:>8} | {:>7} | {:>7}",
            "#", "Symbol", "Side", "Entry", "Exit", "PnL%", "Hold", "Ptrn", "MAE%", "MFE%", "Exit Reason", "Conf", "EqEntry", "EqExit", "TxFee", "FundFee");
        println!("  {}", "-".repeat(182));
        for (i, t) in self.trades.iter().enumerate() {
            let side_str = match t.side {
                crate::domain::PositionSide::Long => "Long",
                crate::domain::PositionSide::Short => "Short",
                _ => "Flat",
            };
            let sym = if t.symbol.len() > 8 {
                &t.symbol[..8]
            } else {
                &t.symbol
            };
            let pat = if t.pattern.len() > 6 {
                &t.pattern[..6]
            } else {
                &t.pattern
            };
            let exit_r = if t.exit_reason.len() > 14 {
                &t.exit_reason[..14]
            } else {
                &t.exit_reason
            };
            println!("  {:>3} | {:8} | {:5} | {:>10.2} | {:>10.2} | {:>+7.3}% | {:>4} | {:6} | {:>5.2} | {:>5.2} | {:14} | {:>5.2} | {:>8.0} | {:>8.0} | {:>7.2} | {:>7.2}",
                i + 1, sym, side_str,
                t.entry_price, t.exit_price, t.pnl_pct,
                t.holding_periods, pat,
                t.max_adverse_excursion, t.max_favorable_excursion,
                exit_r, t.confidence,
                t.equity_at_entry, t.equity_at_exit,
                t.fees_paid, t.funding_fees_paid);
        }
        println!("{}\n", "=".repeat(185));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::learner::{LearnerConfig, LearningEngine};

    fn make_trending_candles(n: usize, start_price: f64, trend: f64) -> Vec<Candle> {
        (0..n)
            .map(|i| {
                let price = start_price + i as f64 * trend;
                let noise = (i as f64 * 0.1).sin() * 0.5;
                Candle {
                    open_time: i as i64 * 900_000,
                    open: price - 0.3 + noise,
                    high: price + 1.0 + noise.abs(),
                    low: price - 1.0 - noise.abs(),
                    close: price + noise,
                    volume: 1000.0 + (i as f64 * 10.0),
                    close_time: (i as i64 + 1) * 900_000 - 1,
                    quote_volume: price * 1000.0,
                    trades: 500,
                }
            })
            .collect()
    }

    #[test]
    fn test_backtest_runs_without_panic() {
        let learning = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let mut bt = BacktestEngine::new(FuturesConfig::default(), learning, 42);
        let candles = make_trending_candles(200, 50000.0, 10.0);
        let result = bt.run(&Symbol("BTCUSDT".into()), &candles, 60);
        assert!(result.final_equity > 0.0);
        assert!(!result.performance.equity_curve.is_empty());
    }

    #[test]
    fn test_temporal_order_enforced() {
        let learning = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let mut bt = BacktestEngine::new(FuturesConfig::default(), learning, 42);
        let candles = make_trending_candles(200, 50000.0, 10.0);
        let _result = bt.run(&Symbol("BTCUSDT".into()), &candles, 60);
    }

    #[test]
    fn test_backtest_with_pattern_library() {
        let learning = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let mut bt = BacktestEngine::new(FuturesConfig::default(), learning, 42);
        let candles = make_trending_candles(300, 50000.0, 5.0);
        let result = bt.run(&Symbol("BTCUSDT".into()), &candles, 60);
        assert!(
            bt.library.completed_global_count() > 0,
            "Library should have completed observations after 300 candles"
        );
        assert!(result.final_equity > 0.0);
        assert!(!result.performance.equity_curve.is_empty());
    }

    #[test]
    #[should_panic(expected = "LOOK-AHEAD VIOLATION")]
    fn test_unordered_candles_rejected() {
        let learning = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let mut bt = BacktestEngine::new(FuturesConfig::default(), learning, 42);
        let mut candles = make_trending_candles(200, 50000.0, 10.0);
        candles.swap(100, 50);
        let _result = bt.run(&Symbol("BTCUSDT".into()), &candles, 60);
    }
}
