use crate::domain::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Portfolio state for Binance Futures backtesting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Portfolio {
    pub equity: f64,
    pub position: Option<Position>,
    pub config: FuturesConfig,
    pub realized_pnl: f64,
    pub trade_log: Vec<TradeRecord>,
    pub equity_curve: Vec<f64>,
    peak_equity: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeRecord {
    pub symbol: String,
    pub side: PositionSide,
    pub entry_price: f64,
    pub exit_price: f64,
    pub size_usd: f64,
    pub pnl: f64,
    pub pnl_pct: f64,
    pub entry_time: i64,
    pub exit_time: i64,
    pub holding_periods: usize,
    pub strategy: String,
    pub max_adverse_excursion: f64,
    pub max_favorable_excursion: f64,
    pub fees_paid: f64,
    pub funding_fees_paid: f64,
    // Extended attributes for analysis
    pub pattern: String,
    pub confidence: f64,
    pub thompson_reward: f64,
    pub entry_atr: f64,
    pub exit_reason: String,
    pub equity_at_entry: f64,
    pub equity_at_exit: f64,
    pub position_size_frac: f64,
}

impl Portfolio {
    pub fn new(config: FuturesConfig) -> Self {
        let equity = config.initial_equity;
        Self {
            equity,
            position: None,
            config,
            realized_pnl: 0.0,
            trade_log: Vec::new(),
            equity_curve: vec![equity],
            peak_equity: equity,
        }
    }

    /// Open a new position (Binance Futures)
    pub fn open_position(
        &mut self,
        symbol: &Symbol,
        side: PositionSide,
        price: f64,
        size_fraction: f64,
        strategy: &str,
        timestamp: i64,
    ) -> bool {
        if self.position.is_some() {
            return false;
        }
        if side == PositionSide::Flat {
            return false;
        }
        if self.equity <= 0.0 {
            return false;
        }

        let size_fraction = size_fraction.clamp(0.05, 0.5);
        let notional = self.equity * size_fraction * self.config.leverage;
        let size = notional / price;

        // Apply slippage
        let slippage = price * self.config.slippage_bps / 10_000.0;
        let entry_price = match side {
            PositionSide::Long => price + slippage,
            PositionSide::Short => price - slippage,
            PositionSide::Flat => unreachable!(),
        };

        // Entry fee recorded but NOT deducted here — deducted at close time
        // so that net_pnl and pnl_pct reflect the full round-trip cost.
        let _entry_fee = notional * self.config.taker_fee;

        self.position = Some(Position {
            symbol: symbol.clone(),
            side,
            entry_price,
            size,
            unrealized_pnl: 0.0,
            entry_time: timestamp,
        });

        true
    }

    /// Close current position
    pub fn close_position(
        &mut self,
        price: f64,
        timestamp: i64,
        candles_held: usize,
        max_adverse: f64,
        strategy: &str,
        accumulated_funding: f64,
    ) -> Option<TradeRecord> {
        let pos = self.position.take()?;

        // Apply slippage
        let slippage = price * self.config.slippage_bps / 10_000.0;
        let exit_price = match pos.side {
            PositionSide::Long => price - slippage,
            PositionSide::Short => price + slippage,
            PositionSide::Flat => return None,
        };

        let notional = pos.size * pos.entry_price;

        // Calculate P&L
        let pnl = match pos.side {
            PositionSide::Long => pos.size * (exit_price - pos.entry_price),
            PositionSide::Short => pos.size * (pos.entry_price - exit_price),
            PositionSide::Flat => 0.0,
        };

        // Pay taker fee on exit
        let exit_notional = pos.size * exit_price;
        let exit_fee = exit_notional * self.config.taker_fee;
        let entry_fee = notional * self.config.taker_fee;
        let total_tx_fees = entry_fee + exit_fee;

        // Net PnL includes both entry and exit fees plus funding fees
        let net_pnl = pnl - entry_fee - exit_fee - accumulated_funding;
        let pnl_pct = net_pnl / (notional / self.config.leverage) * 100.0;

        self.equity += net_pnl;
        self.realized_pnl += net_pnl;

        let equity_at_exit = self.equity;

        let record = TradeRecord {
            symbol: pos.symbol.0,
            side: pos.side,
            entry_price: pos.entry_price,
            exit_price,
            size_usd: notional,
            pnl: net_pnl,
            pnl_pct,
            entry_time: pos.entry_time,
            exit_time: timestamp,
            holding_periods: candles_held,
            strategy: strategy.to_string(),
            max_adverse_excursion: max_adverse,
            max_favorable_excursion: 0.0, // filled by TradeManager
            fees_paid: total_tx_fees,
            funding_fees_paid: accumulated_funding,
            pattern: String::new(),     // filled by TradeManager
            confidence: 0.0,            // filled by TradeManager
            thompson_reward: 0.0,       // filled by TradeManager
            entry_atr: 0.0,             // filled by TradeManager
            exit_reason: String::new(), // filled by TradeManager
            equity_at_entry: 0.0,       // filled by TradeManager
            equity_at_exit,
            position_size_frac: 0.0, // filled by TradeManager
        };

        self.trade_log.push(record.clone());
        Some(record)
    }

    /// Update unrealized P&L and equity curve (called each candle)
    pub fn update_mark(&mut self, price: f64) -> f64 {
        let unrealized = if let Some(ref mut pos) = self.position {
            match pos.side {
                PositionSide::Long => pos.size * (price - pos.entry_price),
                PositionSide::Short => pos.size * (pos.entry_price - price),
                PositionSide::Flat => 0.0,
            }
        } else {
            0.0
        };

        if let Some(ref mut pos) = self.position {
            pos.unrealized_pnl = unrealized;
        }

        let current_equity = self.equity + unrealized;
        self.equity_curve.push(current_equity);

        if current_equity > self.peak_equity {
            self.peak_equity = current_equity;
        }

        current_equity
    }

    /// Check for liquidation (Binance Futures)
    pub fn check_liquidation(&self, price: f64) -> bool {
        let pos = match &self.position {
            Some(p) => p,
            None => return false,
        };

        let notional = pos.size * pos.entry_price;
        let margin = notional / self.config.leverage;
        let unrealized = match pos.side {
            PositionSide::Long => pos.size * (price - pos.entry_price),
            PositionSide::Short => pos.size * (pos.entry_price - price),
            PositionSide::Flat => 0.0,
        };

        // Liquidation when loss exceeds margin (simplified)
        unrealized < -margin * 0.95
    }

    pub fn has_position(&self) -> bool {
        self.position.is_some()
    }

    pub fn current_equity(&self) -> f64 {
        self.equity
            + self
                .position
                .as_ref()
                .map(|p| p.unrealized_pnl)
                .unwrap_or(0.0)
    }
}

/// Multi-symbol portfolio with shared equity pool and concurrent positions.
/// Each symbol can hold at most one position. Equity is shared across all.
#[derive(Debug, Clone)]
pub struct MultiSymbolPortfolio {
    pub positions: HashMap<String, Position>,
    pub equity: f64,
    pub config: FuturesConfig,
    pub realized_pnl: f64,
    pub trade_log: Vec<TradeRecord>,
    pub equity_curve: Vec<f64>,
    peak_equity: f64,
}

impl MultiSymbolPortfolio {
    pub fn new(config: FuturesConfig) -> Self {
        let equity = config.initial_equity;
        Self {
            positions: HashMap::new(),
            equity,
            config,
            realized_pnl: 0.0,
            trade_log: Vec::new(),
            equity_curve: vec![equity],
            peak_equity: equity,
        }
    }

    /// Open a new position for a symbol. Returns false if already positioned.
    pub fn open_position(
        &mut self,
        symbol: &Symbol,
        side: PositionSide,
        price: f64,
        size_fraction: f64,
        strategy: &str,
        timestamp: i64,
    ) -> bool {
        if self.positions.contains_key(&symbol.0) {
            return false;
        }
        if side == PositionSide::Flat {
            return false;
        }
        let current_eq = self.current_equity();
        if current_eq <= 0.0 {
            return false;
        }

        let size_fraction = size_fraction.clamp(0.05, 0.5);
        let notional = current_eq * size_fraction * self.config.leverage;
        let size = notional / price;

        let slippage = price * self.config.slippage_bps / 10_000.0;
        let entry_price = match side {
            PositionSide::Long => price + slippage,
            PositionSide::Short => price - slippage,
            PositionSide::Flat => unreachable!(),
        };

        self.positions.insert(
            symbol.0.clone(),
            Position {
                symbol: symbol.clone(),
                side,
                entry_price,
                size,
                unrealized_pnl: 0.0,
                entry_time: timestamp,
            },
        );

        true
    }

    /// Close a position for a specific symbol.
    pub fn close_position(
        &mut self,
        symbol: &str,
        price: f64,
        timestamp: i64,
        candles_held: usize,
        max_adverse: f64,
        strategy: &str,
        accumulated_funding: f64,
    ) -> Option<TradeRecord> {
        let pos = self.positions.remove(symbol)?;

        let slippage = price * self.config.slippage_bps / 10_000.0;
        let exit_price = match pos.side {
            PositionSide::Long => price - slippage,
            PositionSide::Short => price + slippage,
            PositionSide::Flat => return None,
        };

        let notional = pos.size * pos.entry_price;

        let pnl = match pos.side {
            PositionSide::Long => pos.size * (exit_price - pos.entry_price),
            PositionSide::Short => pos.size * (pos.entry_price - exit_price),
            PositionSide::Flat => 0.0,
        };

        let exit_notional = pos.size * exit_price;
        let exit_fee = exit_notional * self.config.taker_fee;
        let entry_fee = notional * self.config.taker_fee;
        let total_tx_fees = entry_fee + exit_fee;

        let net_pnl = pnl - entry_fee - exit_fee - accumulated_funding;
        let pnl_pct = net_pnl / (notional / self.config.leverage) * 100.0;

        self.equity += net_pnl;
        self.realized_pnl += net_pnl;

        let equity_at_exit = self.equity;

        let record = TradeRecord {
            symbol: pos.symbol.0,
            side: pos.side,
            entry_price: pos.entry_price,
            exit_price,
            size_usd: notional,
            pnl: net_pnl,
            pnl_pct,
            entry_time: pos.entry_time,
            exit_time: timestamp,
            holding_periods: candles_held,
            strategy: strategy.to_string(),
            max_adverse_excursion: max_adverse,
            max_favorable_excursion: 0.0,
            fees_paid: total_tx_fees,
            funding_fees_paid: accumulated_funding,
            pattern: String::new(),
            confidence: 0.0,
            thompson_reward: 0.0,
            entry_atr: 0.0,
            exit_reason: String::new(),
            equity_at_entry: 0.0,
            equity_at_exit,
            position_size_frac: 0.0,
        };

        self.trade_log.push(record.clone());
        Some(record)
    }

    /// Update mark-to-market for a specific symbol's position.
    pub fn update_mark_symbol(&mut self, symbol: &str, price: f64) {
        if let Some(pos) = self.positions.get_mut(symbol) {
            pos.unrealized_pnl = match pos.side {
                PositionSide::Long => pos.size * (price - pos.entry_price),
                PositionSide::Short => pos.size * (pos.entry_price - price),
                PositionSide::Flat => 0.0,
            };
        }
    }

    /// Push a single equity curve snapshot (call once per timestamp, not per symbol).
    pub fn push_equity_snapshot(&mut self) {
        let eq = self.current_equity();
        self.equity_curve.push(eq);
        if eq > self.peak_equity {
            self.peak_equity = eq;
        }
    }

    /// Check if a specific symbol has a position.
    pub fn has_position_for(&self, symbol: &str) -> bool {
        self.positions.contains_key(symbol)
    }

    /// Check liquidation for a specific symbol's position.
    pub fn check_liquidation_for(&self, symbol: &str, price: f64) -> bool {
        let pos = match self.positions.get(symbol) {
            Some(p) => p,
            None => return false,
        };

        let notional = pos.size * pos.entry_price;
        let margin = notional / self.config.leverage;
        let unrealized = match pos.side {
            PositionSide::Long => pos.size * (price - pos.entry_price),
            PositionSide::Short => pos.size * (pos.entry_price - price),
            PositionSide::Flat => 0.0,
        };

        unrealized < -margin * 0.95
    }

    /// Get the position for a symbol (immutable).
    pub fn position_for(&self, symbol: &str) -> Option<&Position> {
        self.positions.get(symbol)
    }

    /// Total equity = cash + sum of unrealized across all positions.
    pub fn current_equity(&self) -> f64 {
        let unrealized: f64 = self.positions.values().map(|p| p.unrealized_pnl).sum();
        self.equity + unrealized
    }

    /// All symbols currently holding a position.
    pub fn positioned_symbols(&self) -> Vec<String> {
        self.positions.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_portfolio() -> Portfolio {
        Portfolio::new(FuturesConfig::default())
    }

    #[test]
    fn test_open_close_long_profitable() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());

        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        assert!(port.has_position());

        let record = port
            .close_position(51000.0, 1000, 5, 0.5, "test", 0.0)
            .unwrap();
        assert!(
            record.pnl > 0.0,
            "Long from 50k to 51k should be profitable"
        );
        assert!(!port.has_position());
    }

    #[test]
    fn test_open_close_short_profitable() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());

        port.open_position(&sym, PositionSide::Short, 50000.0, 0.1, "test", 0);
        let record = port
            .close_position(49000.0, 1000, 5, 0.5, "test", 0.0)
            .unwrap();
        assert!(
            record.pnl > 0.0,
            "Short from 50k to 49k should be profitable"
        );
    }

    #[test]
    fn test_fees_reduce_pnl() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());

        // Open and close at same price
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        let record = port
            .close_position(50000.0, 1000, 1, 0.0, "test", 0.0)
            .unwrap();
        // Should be slightly negative due to fees + slippage
        assert!(
            record.pnl < 0.0,
            "Round-trip at same price should be negative due to fees"
        );
    }

    #[test]
    fn test_cannot_double_open() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());

        assert!(port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0));
        assert!(!port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 1));
    }

    #[test]
    fn test_equity_curve_tracking() {
        let mut port = default_portfolio();
        port.update_mark(50000.0);
        port.update_mark(50000.0);
        assert!(port.equity_curve.len() >= 3); // initial + 2 updates
    }

    #[test]
    fn test_liquidation_check() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        // Use max position size 0.5 with 3x leverage
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.5, "test", 0);

        // With 3x leverage, position notional = equity * 0.5 * 3 = 15000
        // margin = 15000/3 = 5000
        // Liquidation when unrealized loss > 95% of margin = 4750
        // size = 15000 / 50000 = 0.3 BTC (adjusted by slippage)
        // Loss at price P = 0.3 * (50000 - P)
        // Need: 0.3 * (50000 - P) > 4750  =>  P < 50000 - 15833 = 34167
        assert!(!port.check_liquidation(45000.0)); // 10% drop, no liq
        assert!(port.check_liquidation(30000.0)); // 40% drop, should liquidate
    }

    #[test]
    fn test_cannot_open_flat() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        assert!(!port.open_position(&sym, PositionSide::Flat, 50000.0, 0.1, "test", 0));
    }

    #[test]
    fn test_close_no_position() {
        let mut port = default_portfolio();
        let result = port.close_position(50000.0, 1000, 5, 0.5, "test", 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_size_fraction_clamped() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        // Size fraction 0.01 should be clamped to 0.05
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.01, "test", 0);
        assert!(port.has_position());
        let pos = port.position.as_ref().unwrap();
        // With 0.05 (clamped) * 10000 * 3x = 1500 notional, size = 1500/50001 (with slippage)
        assert!(pos.size > 0.0);
    }

    #[test]
    fn test_size_fraction_clamped_high() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        // Size fraction 0.9 should be clamped to 0.5
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.9, "test", 0);
        let pos = port.position.as_ref().unwrap();
        // Notional = 10000 * 0.5 * 3 = 15000
        let expected_notional = 10_000.0 * 0.5 * 3.0;
        let actual_notional = pos.size * pos.entry_price;
        assert!((actual_notional - expected_notional).abs() / expected_notional < 0.01);
    }

    #[test]
    fn test_zero_equity_rejects_open() {
        let config = FuturesConfig {
            initial_equity: 0.0,
            ..FuturesConfig::default()
        };
        let mut port = Portfolio::new(config);
        let sym = Symbol("BTCUSDT".into());
        assert!(!port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0));
    }

    #[test]
    fn test_funding_fees_deducted() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        let initial_equity = port.equity;
        let record = port.close_position(51000.0, 1000, 5, 0.5, "test", 50.0).unwrap();
        assert_eq!(record.funding_fees_paid, 50.0);
        // PnL should be reduced by funding
        let record_no_funding = {
            let mut port2 = default_portfolio();
            port2.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
            port2.close_position(51000.0, 1000, 5, 0.5, "test", 0.0).unwrap()
        };
        assert!(record.pnl < record_no_funding.pnl, "Funding fees should reduce PnL");
    }

    #[test]
    fn test_slippage_applied() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        let pos = port.position.as_ref().unwrap();
        // Long entry should have slippage added: 50000 + 50000 * 2/10000 = 50010
        assert!(pos.entry_price > 50000.0, "Long entry should include slippage");
    }

    #[test]
    fn test_short_slippage_applied() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Short, 50000.0, 0.1, "test", 0);
        let pos = port.position.as_ref().unwrap();
        // Short entry should have slippage subtracted
        assert!(pos.entry_price < 50000.0, "Short entry should include slippage");
    }

    #[test]
    fn test_update_mark_no_position() {
        let mut port = default_portfolio();
        let eq = port.update_mark(50000.0);
        assert_eq!(eq, port.equity); // No unrealized PnL
    }

    #[test]
    fn test_current_equity_includes_unrealized() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        port.update_mark(51000.0); // Price went up
        assert!(port.current_equity() > port.equity, "Current equity should include unrealized PnL");
    }

    #[test]
    fn test_liquidation_no_position() {
        let port = default_portfolio();
        assert!(!port.check_liquidation(0.0));
    }

    #[test]
    fn test_short_liquidation() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Short, 50000.0, 0.5, "test", 0);
        // Short: massive price increase should liquidate
        assert!(!port.check_liquidation(55000.0));
        assert!(port.check_liquidation(80000.0));
    }

    #[test]
    fn test_trade_record_fees_accurate() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        let record = port.close_position(50000.0, 1000, 1, 0.0, "test", 0.0).unwrap();
        // fees_paid should be entry_fee + exit_fee
        assert!(record.fees_paid > 0.0, "Fees should be recorded");
        // With taker_fee = 0.0004, notional ~3000: fee ~1.2 per side
        assert!(record.fees_paid > 1.0 && record.fees_paid < 5.0, "Fees should be realistic: {}", record.fees_paid);
    }

    #[test]
    fn test_long_loss_reduces_equity() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        let initial = port.equity;
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        let record = port.close_position(48000.0, 1000, 5, 0.5, "test", 0.0).unwrap();
        assert!(record.pnl < 0.0, "Long losing trade should be negative PnL");
        assert!(port.equity < initial, "Equity should decrease on loss");
    }

    #[test]
    fn test_short_loss_reduces_equity() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        let initial = port.equity;
        port.open_position(&sym, PositionSide::Short, 50000.0, 0.1, "test", 0);
        let record = port.close_position(52000.0, 1000, 5, 0.5, "test", 0.0).unwrap();
        assert!(record.pnl < 0.0, "Short losing trade should be negative PnL");
        assert!(port.equity < initial);
    }

    #[test]
    fn test_pnl_pct_sign_matches_pnl() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        let record = port.close_position(51000.0, 1000, 5, 0.5, "test", 0.0).unwrap();
        assert!(record.pnl > 0.0);
        assert!(record.pnl_pct > 0.0, "pnl_pct sign should match pnl");
    }

    #[test]
    fn test_realized_pnl_accumulates() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        port.close_position(51000.0, 1000, 5, 0.0, "test", 0.0);
        let after_first = port.realized_pnl;
        port.open_position(&sym, PositionSide::Long, 51000.0, 0.1, "test", 2000);
        port.close_position(52000.0, 3000, 5, 0.0, "test", 0.0);
        assert!(port.realized_pnl > after_first, "Realized PnL should accumulate");
    }

    #[test]
    fn test_trade_log_grows() {
        let mut port = default_portfolio();
        let sym = Symbol("BTCUSDT".into());
        assert_eq!(port.trade_log.len(), 0);
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        port.close_position(51000.0, 1000, 5, 0.0, "test", 0.0);
        assert_eq!(port.trade_log.len(), 1);
    }

    // === MultiSymbolPortfolio tests ===

    fn default_multi_portfolio() -> MultiSymbolPortfolio {
        MultiSymbolPortfolio::new(FuturesConfig::default())
    }

    #[test]
    fn test_multi_open_close_long() {
        let mut port = default_multi_portfolio();
        let sym = Symbol("BTCUSDT".into());
        assert!(port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0));
        assert!(port.has_position_for("BTCUSDT"));
        let record = port.close_position("BTCUSDT", 51000.0, 1000, 5, 0.5, "test", 0.0).unwrap();
        assert!(record.pnl > 0.0);
        assert!(!port.has_position_for("BTCUSDT"));
    }

    #[test]
    fn test_multi_concurrent_positions() {
        let mut port = default_multi_portfolio();
        let btc = Symbol("BTCUSDT".into());
        let eth = Symbol("ETHUSDT".into());
        assert!(port.open_position(&btc, PositionSide::Long, 50000.0, 0.1, "test", 0));
        assert!(port.open_position(&eth, PositionSide::Short, 3000.0, 0.1, "test", 0));
        assert!(port.has_position_for("BTCUSDT"));
        assert!(port.has_position_for("ETHUSDT"));
        assert_eq!(port.positions.len(), 2);
    }

    #[test]
    fn test_multi_cannot_double_open_same_symbol() {
        let mut port = default_multi_portfolio();
        let sym = Symbol("BTCUSDT".into());
        assert!(port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0));
        assert!(!port.open_position(&sym, PositionSide::Short, 50000.0, 0.1, "test", 1));
    }

    #[test]
    fn test_multi_shared_equity() {
        let mut port = default_multi_portfolio();
        let btc = Symbol("BTCUSDT".into());
        let eth = Symbol("ETHUSDT".into());
        port.open_position(&btc, PositionSide::Long, 50000.0, 0.1, "test", 0);
        port.update_mark_symbol("BTCUSDT", 51000.0);
        // Opening ETH should use current_equity which includes BTC unrealized
        let eq_before = port.current_equity();
        assert!(eq_before > port.equity, "Unrealized should boost equity");
        port.open_position(&eth, PositionSide::Long, 3000.0, 0.1, "test", 0);
        assert!(port.has_position_for("ETHUSDT"));
    }

    #[test]
    fn test_multi_equity_snapshot() {
        let mut port = default_multi_portfolio();
        assert_eq!(port.equity_curve.len(), 1); // initial
        port.push_equity_snapshot();
        assert_eq!(port.equity_curve.len(), 2);
    }

    #[test]
    fn test_multi_current_equity_includes_all_unrealized() {
        let mut port = default_multi_portfolio();
        let btc = Symbol("BTCUSDT".into());
        let eth = Symbol("ETHUSDT".into());
        port.open_position(&btc, PositionSide::Long, 50000.0, 0.1, "test", 0);
        port.open_position(&eth, PositionSide::Long, 3000.0, 0.1, "test", 0);
        port.update_mark_symbol("BTCUSDT", 51000.0);
        port.update_mark_symbol("ETHUSDT", 3100.0);
        let eq = port.current_equity();
        assert!(eq > port.equity, "Should include unrealized from both positions");
    }

    #[test]
    fn test_multi_close_nonexistent() {
        let mut port = default_multi_portfolio();
        let result = port.close_position("BTCUSDT", 50000.0, 1000, 5, 0.5, "test", 0.0);
        assert!(result.is_none());
    }

    #[test]
    fn test_multi_reject_flat() {
        let mut port = default_multi_portfolio();
        let sym = Symbol("BTCUSDT".into());
        assert!(!port.open_position(&sym, PositionSide::Flat, 50000.0, 0.1, "test", 0));
    }

    #[test]
    fn test_multi_liquidation_check() {
        let mut port = default_multi_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.5, "test", 0);
        assert!(!port.check_liquidation_for("BTCUSDT", 45000.0));
        assert!(port.check_liquidation_for("BTCUSDT", 30000.0));
    }

    #[test]
    fn test_multi_positioned_symbols() {
        let mut port = default_multi_portfolio();
        let btc = Symbol("BTCUSDT".into());
        let eth = Symbol("ETHUSDT".into());
        port.open_position(&btc, PositionSide::Long, 50000.0, 0.1, "test", 0);
        port.open_position(&eth, PositionSide::Short, 3000.0, 0.1, "test", 0);
        let syms = port.positioned_symbols();
        assert_eq!(syms.len(), 2);
        assert!(syms.contains(&"BTCUSDT".to_string()));
        assert!(syms.contains(&"ETHUSDT".to_string()));
    }

    #[test]
    fn test_multi_trade_log_grows() {
        let mut port = default_multi_portfolio();
        let sym = Symbol("BTCUSDT".into());
        port.open_position(&sym, PositionSide::Long, 50000.0, 0.1, "test", 0);
        port.close_position("BTCUSDT", 51000.0, 1000, 5, 0.0, "test", 0.0);
        assert_eq!(port.trade_log.len(), 1);
    }
}
