use super::learner::LearningEngine;
use crate::backtest::portfolio::TradeRecord;
use crate::domain::*;
use crate::evaluation::scorer;
use crate::strategy::patterns::PatternDiscretizer;

/// What the trade manager tells the backtest/live runner to do.
#[derive(Debug, Clone, PartialEq)]
pub enum TradeAction {
    /// Do nothing this candle.
    None,
    /// Close the current position at this price.
    Exit { price: f64, reason: ExitReason },
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExitReason {
    StopLoss,
    TakeProfit,
    TrailingStop,
    Breakeven,
    MaxHold,
}

impl std::fmt::Display for ExitReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ExitReason::StopLoss => write!(f, "stop_loss"),
            ExitReason::TakeProfit => write!(f, "take_profit"),
            ExitReason::TrailingStop => write!(f, "trailing_stop"),
            ExitReason::Breakeven => write!(f, "breakeven"),
            ExitReason::MaxHold => write!(f, "max_hold"),
        }
    }
}

/// Manages all open-position state and trading logic.
/// Owns: SL/TP, trailing stops, breakeven, excursion tracking,
/// min/max hold, cooldown, and outcome recording.
///
/// The backtest/live runner only needs to:
/// 1. Call `on_candle` each bar to get exit signals
/// 2. Call `on_entry` when opening a position
/// 3. Call `on_exit` when a position closes (for learning feedback)
/// 4. Check `in_cooldown` before requesting decisions
pub struct TradeManager {
    pub current_sl: Option<f64>,
    pub current_tp: Option<f64>,
    pub current_strategy: String,
    pub candles_held: usize,
    pub max_adverse: f64,
    pub max_favorable: f64,
    pub cooldown_remaining: usize,
    pub current_pattern: Option<MarketContext>,
    pub entry_atr: f64,
    pub accumulated_funding: f64,
    pub entry_confidence: f64,
    pub equity_at_entry: f64,
    pub position_size_frac: f64,
    pub exit_reason: String,
    /// KNN directional probability at entry (None if no prediction).
    pub last_prediction_direction: Option<f64>,
}

impl TradeManager {
    pub fn new() -> Self {
        Self {
            current_sl: None,
            current_tp: None,
            current_strategy: String::new(),
            candles_held: 0,
            max_adverse: 0.0,
            max_favorable: 0.0,
            cooldown_remaining: 0,
            current_pattern: None,
            entry_atr: 0.0,
            accumulated_funding: 0.0,
            entry_confidence: 0.0,
            equity_at_entry: 0.0,
            position_size_frac: 0.0,
            exit_reason: String::new(),
            last_prediction_direction: None,
        }
    }

    /// Tick cooldown. Returns true if still in cooldown (caller should skip decisions).
    pub fn tick_cooldown(&mut self) -> bool {
        if self.cooldown_remaining > 0 {
            self.cooldown_remaining -= 1;
            true
        } else {
            false
        }
    }

    /// Process an open position on a new candle.
    /// Handles: excursion tracking, trailing stop, breakeven, min hold SL/TP check, max hold.
    /// Returns an exit action if the position should be closed.
    pub fn on_candle(
        &mut self,
        candle: &Candle,
        position: &Position,
        learning: &LearningEngine,
    ) -> TradeAction {
        self.candles_held += 1;

        // Track excursions
        let adverse = match position.side {
            PositionSide::Long => {
                (position.entry_price - candle.low) / position.entry_price * 100.0
            }
            PositionSide::Short => {
                (candle.high - position.entry_price) / position.entry_price * 100.0
            }
            PositionSide::Flat => 0.0,
        };
        self.max_adverse = self.max_adverse.max(adverse);

        let favorable = match position.side {
            PositionSide::Long => {
                (candle.high - position.entry_price) / position.entry_price * 100.0
            }
            PositionSide::Short => {
                (position.entry_price - candle.low) / position.entry_price * 100.0
            }
            PositionSide::Flat => 0.0,
        };
        self.max_favorable = self.max_favorable.max(favorable.max(0.0));

        // Adaptive trailing stop
        if self.entry_atr > 1e-10 {
            self.update_trailing_stop(candle, position, learning);
        }

        // Forced exit after adaptive max hold
        let max_hold = learning.adaptive.max_hold();
        if self.candles_held >= max_hold {
            return TradeAction::Exit {
                price: candle.close,
                reason: ExitReason::MaxHold,
            };
        }

        TradeAction::None
    }

    /// Check SL/TP exits. Only valid after adaptive min hold.
    /// Returns the exit price and reason if SL or TP was hit.
    pub fn check_sl_tp(
        &self,
        candle: &Candle,
        position: &Position,
        learning: &LearningEngine,
    ) -> Option<(f64, ExitReason)> {
        let min_hold = learning.adaptive.min_hold();
        if self.candles_held < min_hold {
            return None;
        }

        match position.side {
            PositionSide::Long => {
                if let Some(sl) = self.current_sl {
                    if candle.low <= sl {
                        return Some((sl, ExitReason::StopLoss));
                    }
                }
                if let Some(tp) = self.current_tp {
                    if candle.high >= tp {
                        return Some((tp, ExitReason::TakeProfit));
                    }
                }
            }
            PositionSide::Short => {
                if let Some(sl) = self.current_sl {
                    if candle.high >= sl {
                        return Some((sl, ExitReason::StopLoss));
                    }
                }
                if let Some(tp) = self.current_tp {
                    if candle.low <= tp {
                        return Some((tp, ExitReason::TakeProfit));
                    }
                }
            }
            PositionSide::Flat => {}
        }

        None
    }

    /// Initialize state when entering a new position.
    pub fn on_entry(
        &mut self,
        decision: &TradingDecision,
        atr: f64,
        pattern: Option<MarketContext>,
        equity: f64,
        prediction_direction: Option<f64>,
    ) {
        self.current_sl = decision.stop_loss;
        self.current_tp = decision.take_profit;
        self.current_strategy = decision.strategy_name.clone();
        self.current_pattern = pattern;
        self.entry_atr = atr;
        self.entry_confidence = decision.confidence;
        self.equity_at_entry = equity;
        self.position_size_frac = decision.size;
        self.last_prediction_direction = prediction_direction;
    }

    /// Record trade outcome into the learning engine after a position closes.
    /// Also fills extended fields on the TradeRecord for analysis.
    pub fn on_exit(
        &mut self,
        symbol: &Symbol,
        record: &mut TradeRecord,
        learning: &mut LearningEngine,
    ) {
        let adaptive_k = learning.adaptive.reward_k();
        let base_reward = scorer::pnl_to_reward(record.pnl_pct, adaptive_k);

        // Track KNN prediction accuracy separately for adaptive params
        let prediction_correct = match self.last_prediction_direction {
            Some(dir) if dir > 0.6 && record.pnl > 0.0 => true,
            Some(dir) if dir < 0.4 && record.pnl > 0.0 => true,
            Some(_) => false,
            None => false,
        };
        if self.last_prediction_direction.is_some() {
            learning.adaptive.record_prediction_accuracy(prediction_correct);
        }
        // Pure PnL reward — blending prediction accuracy was tested and degraded clarity
        let reward = base_reward;

        // Compute ATR-normalized excursions
        let atr = self.entry_atr.max(1e-10);
        let entry = record.entry_price;
        let favorable_atr = (self.max_favorable / 100.0 * entry) / atr;
        let adverse_atr = (self.max_adverse / 100.0 * entry) / atr;

        // Feed adaptive parameter engine
        learning.adaptive.record_trade(
            record.pnl_pct,
            record.holding_periods,
            favorable_atr,
            adverse_atr,
        );

        if let Some(ref pattern) = self.current_pattern {
            let action = match record.side {
                PositionSide::Long => "long",
                PositionSide::Short => "short",
                _ => {
                    self.fill_record_fields(record, reward);
                    return;
                }
            };

            learning.record_outcome(&symbol.0, pattern, &StrategyId(action.to_string()), reward);

            let pattern_key = PatternDiscretizer::pattern_key(pattern);
            learning.record_excursion(
                &pattern_key,
                action,
                favorable_atr,
                adverse_atr,
                record.pnl > 0.0,
            );
        }

        // Fill extended fields on record
        self.fill_record_fields(record, reward);
    }

    /// Fill the extended analysis fields on a TradeRecord.
    fn fill_record_fields(&self, record: &mut TradeRecord, reward: f64) {
        record.max_favorable_excursion = self.max_favorable;
        record.pattern = self
            .current_pattern
            .as_ref()
            .map(|p| PatternDiscretizer::pattern_key(p))
            .unwrap_or_default();
        record.confidence = self.entry_confidence;
        record.thompson_reward = reward;
        record.entry_atr = self.entry_atr;
        record.exit_reason = self.exit_reason.clone();
        record.equity_at_entry = self.equity_at_entry;
        record.position_size_frac = self.position_size_frac;
    }

    /// Reset after position close. Sets cooldown from adaptive params.
    pub fn reset(&mut self, learning: &LearningEngine) {
        self.current_sl = None;
        self.current_tp = None;
        self.current_strategy.clear();
        self.candles_held = 0;
        self.max_adverse = 0.0;
        self.max_favorable = 0.0;
        self.current_pattern = None;
        self.entry_atr = 0.0;
        self.accumulated_funding = 0.0;
        self.cooldown_remaining = learning.adaptive.cooldown();
        self.last_prediction_direction = None;
    }

    /// Reset with a custom cooldown (e.g. forced exit uses longer cooldown).
    pub fn reset_with_cooldown(&mut self, learning: &LearningEngine, cooldown: usize) {
        self.reset(learning);
        self.cooldown_remaining = cooldown;
    }

    /// Add funding fee to accumulated total.
    pub fn add_funding(&mut self, fee: f64) {
        self.accumulated_funding += fee;
    }

    // --- private ---

    fn update_trailing_stop(
        &mut self,
        candle: &Candle,
        position: &Position,
        learning: &LearningEngine,
    ) {
        let entry = position.entry_price;
        let atr = self.entry_atr;
        let profit_atr = match position.side {
            PositionSide::Long => (candle.close - entry) / atr,
            PositionSide::Short => (entry - candle.close) / atr,
            PositionSide::Flat => 0.0,
        };

        let trail_activation = learning.adaptive.trail_activation_atr();
        let trail_dist = learning.adaptive.trail_distance_atr();
        let breakeven_activation = learning.adaptive.breakeven_activation_atr();

        if profit_atr > trail_activation {
            let best_price = match position.side {
                PositionSide::Long => candle.high,
                PositionSide::Short => candle.low,
                PositionSide::Flat => entry,
            };
            let trail_distance = trail_dist * atr;
            let new_sl = match position.side {
                PositionSide::Long => best_price - trail_distance,
                PositionSide::Short => best_price + trail_distance,
                PositionSide::Flat => self.current_sl.unwrap_or(entry),
            };
            // Only tighten SL, never loosen
            if let Some(current_sl) = self.current_sl {
                let should_update = match position.side {
                    PositionSide::Long => new_sl > current_sl,
                    PositionSide::Short => new_sl < current_sl,
                    PositionSide::Flat => false,
                };
                if should_update {
                    self.current_sl = Some(new_sl);
                }
            }
        } else if profit_atr > breakeven_activation {
            let be_buffer = atr * 0.2;
            let breakeven_sl = match position.side {
                PositionSide::Long => entry + be_buffer,
                PositionSide::Short => entry - be_buffer,
                PositionSide::Flat => self.current_sl.unwrap_or(entry),
            };
            if let Some(current_sl) = self.current_sl {
                let should_update = match position.side {
                    PositionSide::Long => breakeven_sl > current_sl,
                    PositionSide::Short => breakeven_sl < current_sl,
                    PositionSide::Flat => false,
                };
                if should_update {
                    self.current_sl = Some(breakeven_sl);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::learner::{LearnerConfig, LearningEngine};

    fn make_learning() -> LearningEngine {
        LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default())
    }

    #[test]
    fn test_trade_manager_initial_state() {
        let tm = TradeManager::new();
        assert_eq!(tm.candles_held, 0);
        assert_eq!(tm.cooldown_remaining, 0);
        assert!(tm.current_sl.is_none());
    }

    #[test]
    fn test_cooldown_ticks_down() {
        let mut tm = TradeManager::new();
        tm.cooldown_remaining = 3;
        assert!(tm.tick_cooldown()); // 3 -> 2
        assert!(tm.tick_cooldown()); // 2 -> 1
        assert!(tm.tick_cooldown()); // 1 -> 0
        assert!(!tm.tick_cooldown()); // 0 -> stays 0
    }

    #[test]
    fn test_reset_sets_cooldown() {
        let mut tm = TradeManager::new();
        tm.candles_held = 50;
        tm.max_adverse = 5.0;
        let learning = make_learning();
        tm.reset(&learning);
        assert_eq!(tm.candles_held, 0);
        assert_eq!(tm.max_adverse, 0.0);
        assert!(tm.cooldown_remaining > 0);
    }

    #[test]
    fn test_excursion_tracking() {
        let mut tm = TradeManager::new();
        tm.entry_atr = 1000.0;
        let learning = make_learning();
        let pos = Position {
            symbol: Symbol("BTCUSDT".into()),
            side: PositionSide::Long,
            entry_price: 50000.0,
            size: 0.1,
            unrealized_pnl: 0.0,
            entry_time: 0,
        };
        let candle = Candle {
            open_time: 0,
            open: 50000.0,
            high: 51000.0,
            low: 49000.0,
            close: 50500.0,
            volume: 1000.0,
            close_time: 900_000,
            quote_volume: 0.0,
            trades: 0,
        };
        tm.on_candle(&candle, &pos, &learning);
        assert!(tm.max_favorable > 0.0);
        assert!(tm.max_adverse > 0.0);
        assert_eq!(tm.candles_held, 1);
    }

    #[test]
    fn test_max_hold_exit() {
        let mut tm = TradeManager::new();
        tm.entry_atr = 1000.0;
        // Use candles_held just under max_hold, then push over
        tm.candles_held = 95; // default max_hold is 96
        let learning = make_learning();
        let pos = Position {
            symbol: Symbol("BTCUSDT".into()),
            side: PositionSide::Long,
            entry_price: 50000.0,
            size: 0.1,
            unrealized_pnl: 0.0,
            entry_time: 0,
        };
        let candle = Candle {
            open_time: 0,
            open: 50000.0,
            high: 50100.0,
            low: 49900.0,
            close: 50000.0,
            volume: 1000.0,
            close_time: 900_000,
            quote_volume: 0.0,
            trades: 0,
        };
        let action = tm.on_candle(&candle, &pos, &learning);
        assert_eq!(
            action,
            TradeAction::Exit {
                price: 50000.0,
                reason: ExitReason::MaxHold,
            }
        );
    }

    #[test]
    fn test_sl_tp_respects_min_hold() {
        let mut tm = TradeManager::new();
        tm.current_sl = Some(49000.0);
        tm.candles_held = 1; // below min_hold
        let learning = make_learning();
        let pos = Position {
            symbol: Symbol("BTCUSDT".into()),
            side: PositionSide::Long,
            entry_price: 50000.0,
            size: 0.1,
            unrealized_pnl: 0.0,
            entry_time: 0,
        };
        let candle = Candle {
            open_time: 0,
            open: 49500.0,
            high: 49600.0,
            low: 48000.0,
            close: 48500.0,
            volume: 1000.0,
            close_time: 900_000,
            quote_volume: 0.0,
            trades: 0,
        };
        // SL at 49000, low at 48000 — would trigger, but min_hold not met
        assert!(tm.check_sl_tp(&candle, &pos, &learning).is_none());
    }

    #[test]
    fn test_short_excursion_tracking() {
        let mut tm = TradeManager::new();
        tm.entry_atr = 1000.0;
        let learning = make_learning();
        let pos = Position {
            symbol: Symbol("BTCUSDT".into()),
            side: PositionSide::Short,
            entry_price: 50000.0,
            size: 0.1,
            unrealized_pnl: 0.0,
            entry_time: 0,
        };
        let candle = Candle {
            open_time: 0,
            open: 50000.0,
            high: 51000.0, // adverse for short
            low: 49000.0,  // favorable for short
            close: 49500.0,
            volume: 1000.0,
            close_time: 900_000,
            quote_volume: 0.0,
            trades: 0,
        };
        tm.on_candle(&candle, &pos, &learning);
        // Short: adverse = (high - entry)/entry * 100 = 2%
        assert!(tm.max_adverse > 1.5, "Short adverse should track high: {}", tm.max_adverse);
        // Short: favorable = (entry - low)/entry * 100 = 2%
        assert!(tm.max_favorable > 1.5, "Short favorable should track low: {}", tm.max_favorable);
    }

    #[test]
    fn test_on_entry_fills_fields() {
        let mut tm = TradeManager::new();
        let decision = TradingDecision {
            signal: TradeSignal::Long,
            size: 0.15,
            stop_loss: Some(49000.0),
            take_profit: Some(52000.0),
            confidence: 0.8,
            strategy_name: "test_strat".to_string(),
        };
        let pattern = MarketContext {
            volatility_tier: "medium".into(),
            trend_regime: "uptrend".into(),
        };
        tm.on_entry(&decision, 500.0, Some(pattern.clone()), 10000.0, Some(0.65));

        assert_eq!(tm.current_sl, Some(49000.0));
        assert_eq!(tm.current_tp, Some(52000.0));
        assert_eq!(tm.current_strategy, "test_strat");
        assert_eq!(tm.entry_atr, 500.0);
        assert_eq!(tm.entry_confidence, 0.8);
        assert_eq!(tm.equity_at_entry, 10000.0);
        assert_eq!(tm.position_size_frac, 0.15);
        assert_eq!(tm.last_prediction_direction, Some(0.65));
        assert_eq!(tm.current_pattern, Some(pattern));
    }

    #[test]
    fn test_add_funding() {
        let mut tm = TradeManager::new();
        assert_eq!(tm.accumulated_funding, 0.0);
        tm.add_funding(1.5);
        tm.add_funding(2.3);
        assert!((tm.accumulated_funding - 3.8).abs() < 1e-10);
    }

    #[test]
    fn test_reset_with_custom_cooldown() {
        let mut tm = TradeManager::new();
        tm.candles_held = 50;
        tm.accumulated_funding = 5.0;
        let learning = make_learning();
        tm.reset_with_cooldown(&learning, 30);
        assert_eq!(tm.candles_held, 0);
        assert_eq!(tm.accumulated_funding, 0.0);
        assert_eq!(tm.cooldown_remaining, 30);
    }

    #[test]
    fn test_tick_cooldown_zero() {
        let mut tm = TradeManager::new();
        tm.cooldown_remaining = 0;
        assert!(!tm.tick_cooldown());
        assert_eq!(tm.cooldown_remaining, 0);
    }

    #[test]
    fn test_sl_tp_short_stop_loss() {
        let mut tm = TradeManager::new();
        tm.current_sl = Some(51000.0); // SL above for short
        tm.candles_held = 100; // well past min_hold
        let learning = make_learning();
        let pos = Position {
            symbol: Symbol("BTCUSDT".into()),
            side: PositionSide::Short,
            entry_price: 50000.0,
            size: 0.1,
            unrealized_pnl: 0.0,
            entry_time: 0,
        };
        let candle = Candle {
            open_time: 0,
            open: 50500.0,
            high: 52000.0, // above SL
            low: 50000.0,
            close: 51500.0,
            volume: 1000.0,
            close_time: 900_000,
            quote_volume: 0.0,
            trades: 0,
        };
        let result = tm.check_sl_tp(&candle, &pos, &learning);
        assert!(result.is_some());
        let (price, reason) = result.unwrap();
        assert_eq!(price, 51000.0);
        assert_eq!(reason, ExitReason::StopLoss);
    }

    #[test]
    fn test_sl_tp_short_take_profit() {
        let mut tm = TradeManager::new();
        tm.current_tp = Some(48000.0); // TP below for short
        tm.candles_held = 100;
        let learning = make_learning();
        let pos = Position {
            symbol: Symbol("BTCUSDT".into()),
            side: PositionSide::Short,
            entry_price: 50000.0,
            size: 0.1,
            unrealized_pnl: 0.0,
            entry_time: 0,
        };
        let candle = Candle {
            open_time: 0,
            open: 49000.0,
            high: 49500.0,
            low: 47000.0, // below TP
            close: 47500.0,
            volume: 1000.0,
            close_time: 900_000,
            quote_volume: 0.0,
            trades: 0,
        };
        let result = tm.check_sl_tp(&candle, &pos, &learning);
        assert!(result.is_some());
        let (price, reason) = result.unwrap();
        assert_eq!(price, 48000.0);
        assert_eq!(reason, ExitReason::TakeProfit);
    }

    #[test]
    fn test_exit_reason_display() {
        assert_eq!(format!("{}", ExitReason::StopLoss), "stop_loss");
        assert_eq!(format!("{}", ExitReason::TakeProfit), "take_profit");
        assert_eq!(format!("{}", ExitReason::TrailingStop), "trailing_stop");
        assert_eq!(format!("{}", ExitReason::Breakeven), "breakeven");
        assert_eq!(format!("{}", ExitReason::MaxHold), "max_hold");
    }

    #[test]
    fn test_flat_position_excursions_zero() {
        let mut tm = TradeManager::new();
        tm.entry_atr = 1000.0;
        let learning = make_learning();
        let pos = Position {
            symbol: Symbol("BTCUSDT".into()),
            side: PositionSide::Flat,
            entry_price: 50000.0,
            size: 0.0,
            unrealized_pnl: 0.0,
            entry_time: 0,
        };
        let candle = Candle {
            open_time: 0,
            open: 50000.0,
            high: 55000.0,
            low: 45000.0,
            close: 50000.0,
            volume: 1000.0,
            close_time: 900_000,
            quote_volume: 0.0,
            trades: 0,
        };
        tm.on_candle(&candle, &pos, &learning);
        assert_eq!(tm.max_adverse, 0.0);
        assert_eq!(tm.max_favorable, 0.0);
    }
}
