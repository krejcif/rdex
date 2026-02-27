use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// OHLCV candle from Binance Futures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Candle {
    pub open_time: i64,
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
    pub close_time: i64,
    pub quote_volume: f64,
    pub trades: u64,
}

impl Candle {
    pub fn datetime(&self) -> DateTime<Utc> {
        DateTime::from_timestamp_millis(self.open_time).unwrap_or_default()
    }

    pub fn body_ratio(&self) -> f64 {
        let range = self.high - self.low;
        if range < 1e-10 {
            return 0.0;
        }
        (self.close - self.open).abs() / range
    }

    pub fn is_bullish(&self) -> bool {
        self.close > self.open
    }
}

/// Trading symbol identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct Symbol(pub String);

impl std::fmt::Display for Symbol {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Market snapshot at a point in time — this is what the engine sees.
/// CRITICAL: Only contains data available at decision time (no future leakage).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketState {
    pub symbol: Symbol,
    pub timestamp: i64,
    /// The candle that just CLOSED (fully formed, no look-ahead)
    pub current_candle: Candle,
    /// Historical candles for indicator computation (oldest first)
    pub history: Vec<Candle>,
    /// Pre-computed indicators (computed only from history + current)
    pub indicators: IndicatorSet,
}

/// Pre-computed technical indicators
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct IndicatorSet {
    pub sma_20: f64,
    pub sma_50: f64,
    pub ema_12: f64,
    pub ema_26: f64,
    pub rsi_14: f64,
    pub atr_14: f64,
    pub macd_line: f64,
    pub macd_signal: f64,
    pub macd_histogram: f64,
    pub bb_upper: f64,
    pub bb_middle: f64,
    pub bb_lower: f64,
    pub adx_14: f64,
    pub volume_sma_20: f64,
}

/// Trading decision output
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingDecision {
    pub signal: TradeSignal,
    pub size: f64, // fraction of equity [0.0, 1.0]
    pub stop_loss: Option<f64>,
    pub take_profit: Option<f64>,
    pub confidence: f64, // [0.0, 1.0]
    pub strategy_name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TradeSignal {
    Long,
    Short,
    Close,
    Hold,
}

/// Position tracking for Binance Futures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Position {
    pub symbol: Symbol,
    pub side: PositionSide,
    pub entry_price: f64,
    pub size: f64, // in base currency
    pub unrealized_pnl: f64,
    pub entry_time: i64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PositionSide {
    Long,
    Short,
    Flat,
}

/// Context bucket for Thompson Sampling — encodes market regime
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct MarketContext {
    pub volatility_tier: String, // "low", "medium", "high"
    pub trend_regime: String,    // "uptrend", "downtrend", "ranging"
}

/// Strategy arm identifier
#[derive(Debug, Clone, Hash, PartialEq, Eq, Serialize, Deserialize)]
pub struct StrategyId(pub String);

impl std::fmt::Display for StrategyId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Result of evaluating a trading decision
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradeEvaluation {
    pub pnl: f64,
    pub pnl_pct: f64,
    pub holding_periods: usize,
    pub score: f64, // [0.0, 1.0] normalized
}

/// Historical funding rate from Binance Futures
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FundingRate {
    pub symbol: String,
    pub funding_time: i64, // milliseconds
    pub funding_rate: f64, // e.g. 0.0001 = 0.01%
}

/// Binance Futures configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FuturesConfig {
    pub leverage: f64,
    pub maker_fee: f64,    // 0.0002 (0.02%)
    pub taker_fee: f64,    // 0.0004 (0.04%)
    pub slippage_bps: f64, // basis points
    pub initial_equity: f64,
}

impl Default for FuturesConfig {
    fn default() -> Self {
        Self {
            leverage: 3.0, // conservative 3x
            maker_fee: 0.0002,
            taker_fee: 0.0004,
            slippage_bps: 2.0, // 2 bps slippage
            initial_equity: 10_000.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_candle_body_ratio() {
        let candle = Candle {
            open_time: 0,
            open: 100.0,
            high: 110.0,
            low: 90.0,
            close: 105.0,
            volume: 1000.0,
            close_time: 0,
            quote_volume: 0.0,
            trades: 0,
        };
        let ratio = candle.body_ratio();
        assert!((ratio - 0.25).abs() < 1e-10); // |105-100| / (110-90) = 5/20
        assert!(candle.is_bullish());
    }

    #[test]
    fn test_candle_bearish() {
        let candle = Candle {
            open_time: 0,
            open: 105.0,
            high: 110.0,
            low: 90.0,
            close: 95.0,
            volume: 1000.0,
            close_time: 0,
            quote_volume: 0.0,
            trades: 0,
        };
        assert!(!candle.is_bullish());
    }

    #[test]
    fn test_futures_config_defaults() {
        let cfg = FuturesConfig::default();
        assert_eq!(cfg.leverage, 3.0);
        assert_eq!(cfg.taker_fee, 0.0004);
        assert_eq!(cfg.initial_equity, 10_000.0);
    }

    #[test]
    fn test_candle_zero_range() {
        let candle = Candle {
            open_time: 0, open: 100.0, high: 100.0, low: 100.0, close: 100.0,
            volume: 1000.0, close_time: 0, quote_volume: 0.0, trades: 0,
        };
        assert_eq!(candle.body_ratio(), 0.0);
    }

    #[test]
    fn test_candle_datetime() {
        let candle = Candle {
            open_time: 1_700_000_000_000, // a valid ms timestamp
            open: 100.0, high: 101.0, low: 99.0, close: 100.0,
            volume: 1000.0, close_time: 0, quote_volume: 0.0, trades: 0,
        };
        let dt = candle.datetime();
        assert!(dt.timestamp() > 0);
    }

    #[test]
    fn test_symbol_display() {
        let sym = Symbol("BTCUSDT".to_string());
        assert_eq!(format!("{}", sym), "BTCUSDT");
    }

    #[test]
    fn test_strategy_id_display() {
        let sid = StrategyId("momentum".to_string());
        assert_eq!(format!("{}", sid), "momentum");
    }

    #[test]
    fn test_market_context_equality() {
        let a = MarketContext { volatility_tier: "high".into(), trend_regime: "uptrend".into() };
        let b = MarketContext { volatility_tier: "high".into(), trend_regime: "uptrend".into() };
        let c = MarketContext { volatility_tier: "low".into(), trend_regime: "uptrend".into() };
        assert_eq!(a, b);
        assert_ne!(a, c);
    }

    #[test]
    fn test_candle_large_body_bullish() {
        let candle = Candle {
            open_time: 0, open: 90.0, high: 110.0, low: 90.0, close: 110.0,
            volume: 1000.0, close_time: 0, quote_volume: 0.0, trades: 0,
        };
        assert!(candle.is_bullish());
        assert!((candle.body_ratio() - 1.0).abs() < 1e-10); // full body candle
    }

    #[test]
    fn test_position_side_variants() {
        assert_ne!(PositionSide::Long, PositionSide::Short);
        assert_ne!(PositionSide::Long, PositionSide::Flat);
        assert_ne!(PositionSide::Short, PositionSide::Flat);
    }

    #[test]
    fn test_trade_signal_variants() {
        assert_ne!(TradeSignal::Long, TradeSignal::Short);
        assert_ne!(TradeSignal::Hold, TradeSignal::Close);
    }

    #[test]
    fn test_candle_datetime_zero() {
        let candle = Candle {
            open_time: 0, open: 100.0, high: 101.0, low: 99.0, close: 100.0,
            volume: 1000.0, close_time: 0, quote_volume: 0.0, trades: 0,
        };
        let dt = candle.datetime();
        assert_eq!(dt.timestamp(), 0);
    }

    #[test]
    fn test_futures_config_slippage() {
        let cfg = FuturesConfig::default();
        assert_eq!(cfg.slippage_bps, 2.0);
        assert_eq!(cfg.maker_fee, 0.0002);
    }

    #[test]
    fn test_indicator_set_default() {
        let ind = IndicatorSet::default();
        assert_eq!(ind.sma_20, 0.0);
        assert_eq!(ind.rsi_14, 0.0);
        assert_eq!(ind.atr_14, 0.0);
    }

    #[test]
    fn test_symbol_hash_eq() {
        use std::collections::HashSet;
        let mut set = HashSet::new();
        set.insert(Symbol("BTC".into()));
        set.insert(Symbol("BTC".into()));
        set.insert(Symbol("ETH".into()));
        assert_eq!(set.len(), 2);
    }
}
