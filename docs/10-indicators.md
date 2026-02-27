# 10 — Technical Indicators

All indicators are computed from closed candle history only — no look-ahead. Requires minimum 50 candles; returns `IndicatorSet::default()` (all zeros) with insufficient data.

## Computed Indicators

| Indicator | Period | Implementation |
|-----------|--------|----------------|
| SMA 20 | 20 | Simple moving average of closes |
| SMA 50 | 50 | Simple moving average of closes |
| EMA 12 | 12 | Exponential moving average |
| EMA 26 | 26 | Exponential moving average |
| RSI 14 | 14 | Wilder's smoothed RSI (returns 50.0 if insufficient data) |
| ATR 14 | 14 | Wilder's smoothed Average True Range |
| MACD | 12/26/9 | Single consistent series: EMA12 walks from idx 12, EMA26 from idx 26. Signal = EMA9 of MACD series. |
| Bollinger Bands | 20, 2σ | Middle = SMA20, Upper/Lower = ±2 standard deviations |
| ADX 14 | 14 | Average Directional Index (returns 25.0 neutral if insufficient) |
| Volume SMA 20 | 20 | Simple moving average of volume |

## Regime Classification

`classify_regime()` produces two labels from `IndicatorSet`:

**Volatility tier** (from Bollinger Band width):
- `"low"`: BB width < 2% of middle
- `"medium"`: 2–5%
- `"high"`: > 5%

**Trend regime** (from ADX + SMA alignment):
- `"uptrend"`: ADX > 25 AND SMA20 > SMA50
- `"downtrend"`: ADX > 25 AND SMA20 < SMA50
- `"ranging"`: ADX ≤ 25

Note: This classification is used by `extract_context()` in the domain layer but the actual pattern discretizer (`PatternDiscretizer`) uses its own 16-pattern binning system based on `MarketFeatures`.
