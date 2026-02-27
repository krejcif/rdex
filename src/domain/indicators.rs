use super::types::{Candle, IndicatorSet};

/// Compute all indicators from a slice of candles.
/// CRITICAL: Only uses closed candles — no look-ahead bias.
/// The last candle in `candles` is the most recent CLOSED candle.
pub fn compute_indicators(candles: &[Candle]) -> IndicatorSet {
    if candles.len() < 50 {
        return IndicatorSet::default();
    }

    let closes: Vec<f64> = candles.iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles.iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles.iter().map(|c| c.low).collect();
    let volumes: Vec<f64> = candles.iter().map(|c| c.volume).collect();

    let sma_20 = sma(&closes, 20);
    let sma_50 = sma(&closes, 50);
    let ema_12 = ema(&closes, 12);
    let ema_26 = ema(&closes, 26);
    let rsi_14 = rsi(&closes, 14);
    let atr_14 = atr(&highs, &lows, &closes, 14);
    let (macd_line, macd_signal, macd_histogram) = macd(&closes);
    let (bb_upper, bb_middle, bb_lower) = bollinger_bands(&closes, 20, 2.0);
    let adx_14 = adx(&highs, &lows, &closes, 14);
    let volume_sma_20 = sma(&volumes, 20);

    IndicatorSet {
        sma_20,
        sma_50,
        ema_12,
        ema_26,
        rsi_14,
        atr_14,
        macd_line,
        macd_signal,
        macd_histogram,
        bb_upper,
        bb_middle,
        bb_lower,
        adx_14,
        volume_sma_20,
    }
}

fn sma(data: &[f64], period: usize) -> f64 {
    if data.len() < period {
        return 0.0;
    }
    let slice = &data[data.len() - period..];
    slice.iter().sum::<f64>() / period as f64
}

fn ema(data: &[f64], period: usize) -> f64 {
    if data.len() < period {
        return 0.0;
    }
    let multiplier = 2.0 / (period as f64 + 1.0);
    let mut ema_val = sma(&data[..period], period);
    for &val in &data[period..] {
        ema_val = (val - ema_val) * multiplier + ema_val;
    }
    ema_val
}

fn rsi(closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 {
        return 50.0;
    }

    let mut avg_gain = 0.0;
    let mut avg_loss = 0.0;

    // Initial averages
    for i in 1..=period {
        let change = closes[i] - closes[i - 1];
        if change > 0.0 {
            avg_gain += change;
        } else {
            avg_loss += change.abs();
        }
    }
    avg_gain /= period as f64;
    avg_loss /= period as f64;

    // Smoothed averages
    for i in (period + 1)..closes.len() {
        let change = closes[i] - closes[i - 1];
        let (gain, loss) = if change > 0.0 {
            (change, 0.0)
        } else {
            (0.0, change.abs())
        };
        avg_gain = (avg_gain * (period as f64 - 1.0) + gain) / period as f64;
        avg_loss = (avg_loss * (period as f64 - 1.0) + loss) / period as f64;
    }

    if avg_loss < 1e-10 {
        return 100.0;
    }
    let rs = avg_gain / avg_loss;
    100.0 - (100.0 / (1.0 + rs))
}

fn atr(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if closes.len() < period + 1 {
        return 0.0;
    }

    let mut trs = Vec::with_capacity(closes.len() - 1);
    for i in 1..closes.len() {
        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
        trs.push(tr);
    }

    if trs.len() < period {
        return 0.0;
    }
    let mut atr_val: f64 = trs[..period].iter().sum::<f64>() / period as f64;
    for &tr in &trs[period..] {
        atr_val = (atr_val * (period as f64 - 1.0) + tr) / period as f64;
    }
    atr_val
}

fn macd(closes: &[f64]) -> (f64, f64, f64) {
    if closes.len() < 35 {
        return (0.0, 0.0, 0.0);
    }

    let mult12 = 2.0 / 13.0;
    let mult26 = 2.0 / 27.0;
    let mut e12 = sma(&closes[..12], 12);
    let mut e26 = sma(&closes[..26], 26);

    // Walk EMA12 from index 12 to 25 (before EMA26 is ready)
    for i in 12..26 {
        e12 = (closes[i] - e12) * mult12 + e12;
    }

    // Build consistent MACD series from index 26 onward
    let mut macd_series = Vec::new();
    for i in 26..closes.len() {
        e12 = (closes[i] - e12) * mult12 + e12;
        e26 = (closes[i] - e26) * mult26 + e26;
        macd_series.push(e12 - e26);
    }

    let macd_line = *macd_series.last().unwrap_or(&0.0);
    let macd_signal = if macd_series.len() >= 9 {
        ema(&macd_series, 9)
    } else {
        0.0
    };

    (macd_line, macd_signal, macd_line - macd_signal)
}

fn bollinger_bands(closes: &[f64], period: usize, num_std: f64) -> (f64, f64, f64) {
    if closes.len() < period {
        return (0.0, 0.0, 0.0);
    }
    let slice = &closes[closes.len() - period..];
    let mean = slice.iter().sum::<f64>() / period as f64;
    let variance = slice.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / period as f64;
    let std = variance.sqrt();
    (mean + num_std * std, mean, mean - num_std * std)
}

fn adx(highs: &[f64], lows: &[f64], closes: &[f64], period: usize) -> f64 {
    if closes.len() < period * 2 + 1 {
        return 25.0;
    } // neutral default

    let mut plus_dm = Vec::new();
    let mut minus_dm = Vec::new();
    let mut tr_vals = Vec::new();

    for i in 1..closes.len() {
        let up = highs[i] - highs[i - 1];
        let down = lows[i - 1] - lows[i];

        plus_dm.push(if up > down && up > 0.0 { up } else { 0.0 });
        minus_dm.push(if down > up && down > 0.0 { down } else { 0.0 });

        let tr = (highs[i] - lows[i])
            .max((highs[i] - closes[i - 1]).abs())
            .max((lows[i] - closes[i - 1]).abs());
        tr_vals.push(tr);
    }

    if tr_vals.len() < period {
        return 25.0;
    }

    let smooth = |vals: &[f64], p: usize| -> Vec<f64> {
        let mut result = vec![vals[..p].iter().sum::<f64>()];
        for i in p..vals.len() {
            let prev = *result.last().unwrap();
            result.push(prev - prev / p as f64 + vals[i]);
        }
        result
    };

    let str_vals = smooth(&tr_vals, period);
    let sp_dm = smooth(&plus_dm, period);
    let sm_dm = smooth(&minus_dm, period);

    let len = str_vals.len().min(sp_dm.len()).min(sm_dm.len());
    if len == 0 {
        return 25.0;
    }

    let mut dx_vals = Vec::new();
    for i in 0..len {
        if str_vals[i] < 1e-10 {
            continue;
        }
        let plus_di = 100.0 * sp_dm[i] / str_vals[i];
        let minus_di = 100.0 * sm_dm[i] / str_vals[i];
        let di_sum = plus_di + minus_di;
        if di_sum < 1e-10 {
            continue;
        }
        dx_vals.push(100.0 * (plus_di - minus_di).abs() / di_sum);
    }

    if dx_vals.len() < period {
        return 25.0;
    }
    let mut adx_val = sma(&dx_vals[..period], period);
    for &dx in &dx_vals[period..] {
        adx_val = (adx_val * (period as f64 - 1.0) + dx) / period as f64;
    }
    adx_val
}

/// Classify market regime from indicators
pub fn classify_regime(indicators: &IndicatorSet) -> (String, String) {
    // Volatility tier based on BB width relative to middle
    let bb_width = if indicators.bb_middle > 0.0 {
        (indicators.bb_upper - indicators.bb_lower) / indicators.bb_middle
    } else {
        0.0
    };

    let volatility_tier = if bb_width < 0.02 {
        "low".to_string()
    } else if bb_width < 0.05 {
        "medium".to_string()
    } else {
        "high".to_string()
    };

    // Trend regime based on ADX + moving average alignment
    let trend_regime = if indicators.adx_14 > 25.0 {
        if indicators.sma_20 > indicators.sma_50 {
            "uptrend".to_string()
        } else {
            "downtrend".to_string()
        }
    } else {
        "ranging".to_string()
    };

    (volatility_tier, trend_regime)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candles(closes: &[f64]) -> Vec<Candle> {
        closes
            .iter()
            .enumerate()
            .map(|(i, &c)| Candle {
                open_time: i as i64 * 900_000,
                open: c - 0.5,
                high: c + 1.0,
                low: c - 1.0,
                close: c,
                volume: 1000.0,
                close_time: (i + 1) as i64 * 900_000 - 1,
                quote_volume: c * 1000.0,
                trades: 100,
            })
            .collect()
    }

    #[test]
    fn test_sma() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        assert!((sma(&data, 3) - 4.0).abs() < 1e-10); // (3+4+5)/3
        assert!((sma(&data, 5) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        let result = ema(&data, 3);
        assert!(result > 8.0 && result < 10.0); // EMA lags
    }

    #[test]
    fn test_rsi_extremes() {
        // All up moves -> RSI near 100
        let up: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        assert!(rsi(&up, 14) > 90.0);

        // All down moves -> RSI near 0
        let down: Vec<f64> = (0..30).map(|i| 100.0 - i as f64).collect();
        assert!(rsi(&down, 14) < 10.0);
    }

    #[test]
    fn test_bollinger_bands() {
        let data: Vec<f64> = (0..20).map(|_| 100.0).collect();
        let (upper, middle, lower) = bollinger_bands(&data, 20, 2.0);
        assert!((middle - 100.0).abs() < 1e-10);
        assert!((upper - 100.0).abs() < 1e-10); // zero std -> bands collapse
        assert!((lower - 100.0).abs() < 1e-10);
    }

    #[test]
    fn test_compute_indicators_sufficient_data() {
        // Generate 60 candles of trending data
        let closes: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let candles = make_candles(&closes);
        let ind = compute_indicators(&candles);
        assert!(ind.sma_20 > 0.0);
        assert!(ind.rsi_14 > 50.0); // uptrend -> RSI > 50
        assert!(ind.atr_14 > 0.0);
    }

    #[test]
    fn test_compute_indicators_insufficient_data() {
        let candles = make_candles(&[100.0; 10]);
        let ind = compute_indicators(&candles);
        assert_eq!(ind.sma_20, 0.0); // not enough data
    }

    #[test]
    fn test_classify_regime() {
        let ind = IndicatorSet {
            adx_14: 30.0,
            sma_20: 105.0,
            sma_50: 100.0,
            bb_upper: 110.0,
            bb_middle: 100.0,
            bb_lower: 90.0,
            ..Default::default()
        };
        let (vol, trend) = classify_regime(&ind);
        assert_eq!(trend, "uptrend");
        // bb_width = 20/100 = 0.2 -> high
        assert_eq!(vol, "high");
    }

    #[test]
    fn test_classify_regime_downtrend() {
        let ind = IndicatorSet {
            adx_14: 30.0,
            sma_20: 95.0,
            sma_50: 100.0,
            bb_upper: 102.0,
            bb_middle: 100.0,
            bb_lower: 98.0,
            ..Default::default()
        };
        let (vol, trend) = classify_regime(&ind);
        assert_eq!(trend, "downtrend");
        // bb_width = 4/100 = 0.04 -> medium
        assert_eq!(vol, "medium");
    }

    #[test]
    fn test_classify_regime_ranging() {
        let ind = IndicatorSet {
            adx_14: 15.0, // below 25 -> ranging
            sma_20: 100.0,
            sma_50: 100.0,
            bb_upper: 100.5,
            bb_middle: 100.0,
            bb_lower: 99.5,
            ..Default::default()
        };
        let (vol, trend) = classify_regime(&ind);
        assert_eq!(trend, "ranging");
        // bb_width = 1/100 = 0.01 -> low
        assert_eq!(vol, "low");
    }

    #[test]
    fn test_classify_regime_zero_bb_middle() {
        let ind = IndicatorSet {
            bb_middle: 0.0,
            ..Default::default()
        };
        let (vol, _) = classify_regime(&ind);
        assert_eq!(vol, "low"); // bb_width = 0
    }

    #[test]
    fn test_sma_insufficient_data() {
        assert_eq!(sma(&[1.0, 2.0], 5), 0.0);
    }

    #[test]
    fn test_ema_insufficient_data() {
        assert_eq!(ema(&[1.0, 2.0], 5), 0.0);
    }

    #[test]
    fn test_rsi_insufficient_data() {
        let data = vec![100.0; 5];
        assert_eq!(rsi(&data, 14), 50.0);
    }

    #[test]
    fn test_rsi_all_gains() {
        let data: Vec<f64> = (0..30).map(|i| 100.0 + i as f64).collect();
        assert_eq!(rsi(&data, 14), 100.0);
    }

    #[test]
    fn test_atr_insufficient_data() {
        let h = vec![101.0; 5];
        let l = vec![99.0; 5];
        let c = vec![100.0; 5];
        assert_eq!(atr(&h, &l, &c, 14), 0.0);
    }

    #[test]
    fn test_atr_basic() {
        let n = 30;
        let highs: Vec<f64> = (0..n).map(|_| 102.0).collect();
        let lows: Vec<f64> = (0..n).map(|_| 98.0).collect();
        let closes: Vec<f64> = (0..n).map(|_| 100.0).collect();
        let result = atr(&highs, &lows, &closes, 14);
        assert!(result > 3.0 && result < 5.0, "ATR should be ~4.0, got {}", result);
    }

    #[test]
    fn test_macd_insufficient_data() {
        let data = vec![100.0; 20];
        let (line, signal, hist) = macd(&data);
        assert_eq!(line, 0.0);
        assert_eq!(signal, 0.0);
        assert_eq!(hist, 0.0);
    }

    #[test]
    fn test_macd_trending() {
        let data: Vec<f64> = (0..60).map(|i| 100.0 + i as f64 * 0.5).collect();
        let (line, _signal, _hist) = macd(&data);
        assert!(line > 0.0, "MACD line should be positive in uptrend");
    }

    #[test]
    fn test_bollinger_bands_insufficient_data() {
        let data = vec![100.0; 5];
        let (u, m, l) = bollinger_bands(&data, 20, 2.0);
        assert_eq!(u, 0.0);
        assert_eq!(m, 0.0);
        assert_eq!(l, 0.0);
    }

    #[test]
    fn test_bollinger_bands_with_variance() {
        let mut data = Vec::new();
        for i in 0..20 {
            data.push(100.0 + if i % 2 == 0 { 1.0 } else { -1.0 });
        }
        let (upper, middle, lower) = bollinger_bands(&data, 20, 2.0);
        assert!(upper > middle, "Upper band should be above middle");
        assert!(lower < middle, "Lower band should be below middle");
        assert!((upper - middle - (middle - lower)).abs() < 1e-10, "Bands should be symmetric");
    }

    #[test]
    fn test_adx_insufficient_data() {
        let h = vec![101.0; 10];
        let l = vec![99.0; 10];
        let c = vec![100.0; 10];
        assert_eq!(adx(&h, &l, &c, 14), 25.0); // neutral default
    }

    #[test]
    fn test_compute_indicators_all_fields_populated() {
        let closes: Vec<f64> = (0..80).map(|i| 100.0 + i as f64 * 0.3).collect();
        let candles = make_candles(&closes);
        let ind = compute_indicators(&candles);
        assert!(ind.sma_20 > 0.0);
        assert!(ind.sma_50 > 0.0);
        assert!(ind.ema_12 > 0.0);
        assert!(ind.ema_26 > 0.0);
        assert!(ind.rsi_14 > 0.0);
        assert!(ind.atr_14 > 0.0);
        assert!(ind.bb_upper > ind.bb_middle);
        assert!(ind.bb_lower < ind.bb_middle);
        assert!(ind.volume_sma_20 > 0.0);
    }

    #[test]
    fn test_sma_exact_period() {
        let data = vec![10.0, 20.0, 30.0];
        assert!((sma(&data, 3) - 20.0).abs() < 1e-10);
    }

    #[test]
    fn test_ema_tracks_trend() {
        let up: Vec<f64> = (0..20).map(|i| i as f64).collect();
        let ema_val = ema(&up, 5);
        // EMA should be below the last value (lagging indicator on uptrend)
        assert!(ema_val < 19.0 && ema_val > 10.0, "EMA should lag: {}", ema_val);
    }

    #[test]
    fn test_rsi_midpoint() {
        // Equal gains and losses -> RSI near 50
        let mut data = vec![100.0];
        for i in 0..30 {
            if i % 2 == 0 {
                data.push(data.last().unwrap() + 1.0);
            } else {
                data.push(data.last().unwrap() - 1.0);
            }
        }
        let r = rsi(&data, 14);
        assert!(r > 40.0 && r < 60.0, "Equal gains/losses should give RSI near 50: {}", r);
    }
}
