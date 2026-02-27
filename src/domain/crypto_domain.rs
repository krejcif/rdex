use super::indicators::{classify_regime, compute_indicators};
use super::types::*;

/// Builds MarketState from a window of candles.
/// ANTI-LOOK-AHEAD: Only uses candles up to and including `index`.
/// The candle at `index` must be CLOSED before this is called.
pub fn build_market_state(
    symbol: &Symbol,
    candles: &[Candle],
    index: usize,
    lookback: usize,
) -> Option<MarketState> {
    if index >= candles.len() {
        return None;
    }

    let start = if index >= lookback {
        index - lookback
    } else {
        0
    };
    let history: Vec<Candle> = candles[start..=index].to_vec();

    let indicators = compute_indicators(&history);
    let current_candle = candles[index].clone();

    Some(MarketState {
        symbol: symbol.clone(),
        timestamp: current_candle.close_time,
        current_candle,
        history,
        indicators,
    })
}

/// Extract MarketContext (bucket) from MarketState for Thompson Sampling
pub fn extract_context(state: &MarketState) -> MarketContext {
    let (volatility_tier, trend_regime) = classify_regime(&state.indicators);
    MarketContext {
        volatility_tier,
        trend_regime,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_candles(n: usize) -> Vec<Candle> {
        (0..n)
            .map(|i| Candle {
                open_time: i as i64 * 900_000,
                open: 100.0 + (i as f64 * 0.1),
                high: 101.0 + (i as f64 * 0.1),
                low: 99.0 + (i as f64 * 0.1),
                close: 100.5 + (i as f64 * 0.1),
                volume: 1000.0,
                close_time: (i + 1) as i64 * 900_000 - 1,
                quote_volume: 100_000.0,
                trades: 500,
            })
            .collect()
    }

    #[test]
    fn test_build_market_state_no_lookahead() {
        let candles = make_test_candles(100);
        let state = build_market_state(
            &Symbol("BTCUSDT".into()),
            &candles,
            50, // current index
            60, // lookback
        )
        .unwrap();

        // CRITICAL: state should NOT contain any candle after index 50
        assert!(state.history.len() <= 51);
        let max_time = state.history.iter().map(|c| c.close_time).max().unwrap();
        assert!(max_time <= candles[50].close_time);
    }

    #[test]
    fn test_build_market_state_at_start() {
        let candles = make_test_candles(100);
        let state = build_market_state(&Symbol("BTCUSDT".into()), &candles, 5, 60).unwrap();
        assert_eq!(state.history.len(), 6); // indices 0..=5
    }

    #[test]
    fn test_build_market_state_out_of_bounds() {
        let candles = make_test_candles(10);
        assert!(build_market_state(&Symbol("BTCUSDT".into()), &candles, 15, 60,).is_none());
    }

    #[test]
    fn test_extract_context() {
        let candles = make_test_candles(100);
        let state = build_market_state(&Symbol("BTCUSDT".into()), &candles, 99, 60).unwrap();
        let ctx = extract_context(&state);
        // Should produce valid context strings
        assert!(!ctx.volatility_tier.is_empty());
        assert!(!ctx.trend_regime.is_empty());
    }

    #[test]
    fn test_build_market_state_lookback_larger_than_index() {
        let candles = make_test_candles(100);
        // lookback=60, index=10 -> should start from 0
        let state = build_market_state(&Symbol("BTCUSDT".into()), &candles, 10, 60).unwrap();
        assert_eq!(state.history.len(), 11); // 0..=10
    }

    #[test]
    fn test_build_market_state_current_candle() {
        let candles = make_test_candles(100);
        let state = build_market_state(&Symbol("BTCUSDT".into()), &candles, 50, 60).unwrap();
        assert_eq!(state.current_candle.open_time, candles[50].open_time);
        assert_eq!(state.timestamp, candles[50].close_time);
    }

    #[test]
    fn test_build_market_state_symbol_preserved() {
        let candles = make_test_candles(100);
        let state = build_market_state(&Symbol("ETHUSDT".into()), &candles, 50, 60).unwrap();
        assert_eq!(state.symbol.0, "ETHUSDT");
    }

    #[test]
    fn test_build_market_state_history_order() {
        let candles = make_test_candles(100);
        let state = build_market_state(&Symbol("BTCUSDT".into()), &candles, 80, 60).unwrap();
        // History should be in temporal order
        for i in 1..state.history.len() {
            assert!(
                state.history[i].open_time >= state.history[i - 1].open_time,
                "History should be in temporal order"
            );
        }
    }

    #[test]
    fn test_build_market_state_exact_lookback() {
        let candles = make_test_candles(100);
        let state = build_market_state(&Symbol("BTCUSDT".into()), &candles, 80, 60).unwrap();
        // index=80, lookback=60 -> start=20, history is candles[20..=80] = 61 items
        assert_eq!(state.history.len(), 61);
    }

    #[test]
    fn test_build_market_state_indicators_computed() {
        let candles = make_test_candles(100);
        let state = build_market_state(&Symbol("BTCUSDT".into()), &candles, 80, 60).unwrap();
        // With 61 candles in history (>50), indicators should be computed
        assert!(state.indicators.sma_20 > 0.0);
    }
}
