use crate::domain::MarketState;

/// Normalized market features for adaptive pattern recognition.
/// All values in comparable scales. No interpretation, no thresholds.
/// The system learns what these values mean through experience.
#[derive(Debug, Clone)]
pub struct MarketFeatures {
    /// Trend: price position relative to moving averages, normalized by ATR.
    /// Negative = below averages (bearish), positive = above (bullish).
    pub trend: f64,
    /// Momentum: RSI + MACD combined signal.
    pub momentum: f64,
    /// Volatility: BB width + ADX combined. 0 = low, 1 = high.
    pub volatility: f64,
    /// Volume: current relative to recent average. 0 = quiet, 1 = active.
    pub volume: f64,
    /// Candle character: body direction x strength. -1 = strong bear, +1 = strong bull.
    pub candle_character: f64,
    /// Position within Bollinger Bands. 0 = lower, 0.5 = middle, 1 = upper.
    pub bb_position: f64,
    /// Short-term price momentum (5-bar return normalized by ATR).
    pub short_momentum: f64,
    /// Raw ATR for sizing and SL/TP.
    pub atr: f64,
    /// Current close price.
    pub price: f64,
}

impl MarketFeatures {
    /// Extract normalized features from market state.
    /// Returns None if insufficient indicator data.
    pub fn extract(state: &MarketState) -> Option<Self> {
        let ind = &state.indicators;
        let c = &state.current_candle;

        if ind.atr_14 < 1e-10 || ind.sma_20 < 1e-10 || ind.bb_middle < 1e-10 {
            return None;
        }

        let price = c.close;
        let atr = ind.atr_14;

        // Trend: distance from SMAs normalized by ATR, compressed via tanh
        let dist_sma20 = (price - ind.sma_20) / atr;
        let dist_sma50 = if ind.sma_50 > 1e-10 {
            (price - ind.sma_50) / atr
        } else {
            dist_sma20
        };
        let ma_spread = if ind.sma_50 > 1e-10 {
            (ind.sma_20 - ind.sma_50) / atr
        } else {
            0.0
        };
        let trend = (dist_sma20 * 0.4 + dist_sma50 * 0.3 + ma_spread * 0.3).tanh();

        // Momentum: RSI + MACD histogram
        let rsi_norm = (ind.rsi_14 - 50.0) / 50.0;
        let macd_norm = (ind.macd_histogram / atr).clamp(-2.0, 2.0) / 2.0;
        let momentum = (rsi_norm * 0.5 + macd_norm * 0.5).tanh();

        // Volatility: BB width ratio + ADX
        let bb_width = (ind.bb_upper - ind.bb_lower) / ind.bb_middle;
        let adx_norm = (ind.adx_14 / 50.0).min(1.0);
        let volatility = (bb_width * 12.5 * 0.5 + adx_norm * 0.5).clamp(0.0, 1.0);

        // Volume: ratio to average, normalized to [0, 1]
        let volume = if ind.volume_sma_20 > 0.0 {
            ((c.volume / ind.volume_sma_20 - 1.0) / 2.0 + 0.5).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Candle character: direction x body strength
        let range = c.high - c.low;
        let candle_character = if range > 1e-10 {
            let direction = (c.close - c.open) / range;
            let body_size = c.body_ratio();
            direction * body_size
        } else {
            0.0
        };

        // BB position: where price sits within bands
        let bb_range = ind.bb_upper - ind.bb_lower;
        let bb_position = if bb_range > 1e-10 {
            ((price - ind.bb_lower) / bb_range).clamp(0.0, 1.0)
        } else {
            0.5
        };

        // Short-term momentum: 5-bar return normalized by ATR
        let short_momentum = if state.history.len() >= 5 {
            let prev = state.history[state.history.len() - 5].close;
            ((price - prev) / atr).tanh()
        } else {
            0.0
        };

        Some(MarketFeatures {
            trend,
            momentum,
            volatility,
            volume,
            candle_character,
            bb_position,
            short_momentum,
            atr,
            price,
        })
    }

    /// All feature values (excludes raw atr and price).
    pub fn as_array(&self) -> [f64; 7] {
        [
            self.trend,
            self.momentum,
            self.volatility,
            self.volume,
            self.candle_character,
            self.bb_position,
            self.short_momentum,
        ]
    }

    /// Key features for pattern discretization (4 most important).
    /// 4 features × 2 bins = 16 patterns.
    pub fn key_features(&self) -> [f64; 4] {
        [self.trend, self.momentum, self.volatility, self.volume]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::*;

    fn make_state(price: f64) -> MarketState {
        let candle = Candle {
            open_time: 0,
            open: price - 0.5,
            high: price + 1.0,
            low: price - 1.0,
            close: price,
            volume: 2000.0,
            close_time: 900_000,
            quote_volume: price * 2000.0,
            trades: 500,
        };
        let mut history = Vec::new();
        for i in 0..10 {
            history.push(Candle {
                open_time: i * 900_000,
                open: price - 5.0 + i as f64 * 0.5,
                high: price - 4.0 + i as f64 * 0.5,
                low: price - 6.0 + i as f64 * 0.5,
                close: price - 5.0 + i as f64 * 0.5 + 0.3,
                volume: 1500.0,
                close_time: (i + 1) * 900_000 - 1,
                quote_volume: 100_000.0,
                trades: 400,
            });
        }
        history.push(candle.clone());
        MarketState {
            symbol: Symbol("BTCUSDT".into()),
            timestamp: 900_000,
            current_candle: candle,
            history,
            indicators: IndicatorSet {
                sma_20: price * 0.99,
                sma_50: price * 0.98,
                ema_12: price * 0.995,
                ema_26: price * 0.99,
                rsi_14: 55.0,
                atr_14: price * 0.02,
                macd_line: 10.0,
                macd_signal: 8.0,
                macd_histogram: 2.0,
                bb_upper: price * 1.04,
                bb_middle: price,
                bb_lower: price * 0.96,
                adx_14: 25.0,
                volume_sma_20: 1500.0,
            },
        }
    }

    #[test]
    fn test_feature_extraction() {
        let state = make_state(50000.0);
        let f = MarketFeatures::extract(&state).unwrap();
        assert!(f.trend >= -1.0 && f.trend <= 1.0);
        assert!(f.momentum >= -1.0 && f.momentum <= 1.0);
        assert!(f.volatility >= 0.0 && f.volatility <= 1.0);
        assert!(f.volume >= 0.0 && f.volume <= 1.0);
        assert!(f.candle_character >= -1.0 && f.candle_character <= 1.0);
        assert!(f.bb_position >= 0.0 && f.bb_position <= 1.0);
        assert!(f.short_momentum >= -1.0 && f.short_momentum <= 1.0);
        assert!(f.atr > 0.0);
        assert!(f.price > 0.0);
    }

    #[test]
    fn test_feature_extraction_insufficient_data() {
        let state = MarketState {
            symbol: Symbol("BTCUSDT".into()),
            timestamp: 0,
            current_candle: Candle {
                open_time: 0,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: 1000.0,
                close_time: 0,
                quote_volume: 0.0,
                trades: 0,
            },
            history: vec![],
            indicators: IndicatorSet::default(),
        };
        assert!(MarketFeatures::extract(&state).is_none());
    }

    #[test]
    fn test_as_array_length() {
        let state = make_state(50000.0);
        let f = MarketFeatures::extract(&state).unwrap();
        assert_eq!(f.as_array().len(), 7);
    }

    #[test]
    fn test_key_features_length() {
        let state = make_state(50000.0);
        let f = MarketFeatures::extract(&state).unwrap();
        assert_eq!(f.key_features().len(), 4);
    }

    #[test]
    fn test_sma50_fallback() {
        let mut state = make_state(50000.0);
        state.indicators.sma_50 = 0.0; // no SMA50 data
        let f = MarketFeatures::extract(&state).unwrap();
        // Should still produce valid features
        assert!(f.trend >= -1.0 && f.trend <= 1.0);
    }

    #[test]
    fn test_zero_volume_sma() {
        let mut state = make_state(50000.0);
        state.indicators.volume_sma_20 = 0.0;
        let f = MarketFeatures::extract(&state).unwrap();
        assert_eq!(f.volume, 0.5); // default when no volume data
    }

    #[test]
    fn test_zero_bb_range() {
        let mut state = make_state(50000.0);
        state.indicators.bb_upper = 50000.0;
        state.indicators.bb_lower = 50000.0;
        let f = MarketFeatures::extract(&state).unwrap();
        assert_eq!(f.bb_position, 0.5); // default when bands collapsed
    }

    #[test]
    fn test_short_history_momentum() {
        let mut state = make_state(50000.0);
        state.history.truncate(3); // less than 5 bars
        let f = MarketFeatures::extract(&state).unwrap();
        assert_eq!(f.short_momentum, 0.0);
    }

    #[test]
    fn test_zero_range_candle() {
        let mut state = make_state(50000.0);
        state.current_candle.high = 50000.0;
        state.current_candle.low = 50000.0;
        state.current_candle.open = 50000.0;
        state.current_candle.close = 50000.0;
        let f = MarketFeatures::extract(&state).unwrap();
        assert_eq!(f.candle_character, 0.0);
    }

    #[test]
    fn test_bearish_candle_character() {
        let mut state = make_state(50000.0);
        state.current_candle.open = 50500.0;
        state.current_candle.close = 49500.0;
        state.current_candle.high = 50600.0;
        state.current_candle.low = 49400.0;
        let f = MarketFeatures::extract(&state).unwrap();
        assert!(f.candle_character < 0.0, "Bearish candle should be negative: {}", f.candle_character);
    }

    #[test]
    fn test_high_volume_clamped() {
        let mut state = make_state(50000.0);
        state.current_candle.volume = 100000.0; // much higher than SMA of 1500
        let f = MarketFeatures::extract(&state).unwrap();
        assert_eq!(f.volume, 1.0); // clamped to 1.0
    }

    #[test]
    fn test_features_at_different_prices() {
        let f_low = MarketFeatures::extract(&make_state(100.0)).unwrap();
        let f_high = MarketFeatures::extract(&make_state(100000.0)).unwrap();
        // Normalized features should be in similar ranges regardless of price
        assert!(f_low.trend >= -1.0 && f_low.trend <= 1.0);
        assert!(f_high.trend >= -1.0 && f_high.trend <= 1.0);
    }

    #[test]
    fn test_low_volume_below_average() {
        let mut state = make_state(50000.0);
        state.current_candle.volume = 500.0; // below SMA of 1500
        let f = MarketFeatures::extract(&state).unwrap();
        assert!(f.volume < 0.5, "Low volume should be below 0.5: {}", f.volume);
    }

    #[test]
    fn test_rsi_affects_momentum() {
        let mut state_high = make_state(50000.0);
        state_high.indicators.rsi_14 = 80.0; // overbought

        let mut state_low = make_state(50000.0);
        state_low.indicators.rsi_14 = 20.0; // oversold

        let f_high = MarketFeatures::extract(&state_high).unwrap();
        let f_low = MarketFeatures::extract(&state_low).unwrap();
        assert!(f_high.momentum > f_low.momentum, "High RSI should give higher momentum");
    }

    #[test]
    fn test_atr_passed_through() {
        let state = make_state(50000.0);
        let f = MarketFeatures::extract(&state).unwrap();
        assert!((f.atr - state.indicators.atr_14).abs() < 1e-10, "ATR should pass through");
    }
}
