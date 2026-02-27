# Market-Based Vector Pattern Learning — Design Document

**Date:** 2026-02-28
**Status:** Approved
**Approach:** Full KNN-Driven Hybrid (KNN informs Thompson Sampling)

## Problem

The current 16-pattern discretization (4 features x 2 bins) is too coarse — it collapses diverse market conditions into the same bucket. Learning is sparse because it's tied to trade outcomes (~50 trades/30 days), missing the dense signal from market price development at every candle (~2,880 observations/30 days).

## Solution Overview

Record a `MarketObservation` at every candle with a 21-dimensional temporal feature vector and forward returns at multiple horizons. At decision time, find the K nearest historical observations via brute-force KNN to predict forward returns, then use those predictions to warm-start Thompson Sampling priors and derive position sizing and stop-loss parameters.

## 1. Data Structures

### TemporalVector (21 dimensions)

Three consecutive snapshots of 7 `MarketFeatures`, concatenated:

```
[t-2: trend, momentum, volatility, volume, candle_char, bb_pos, short_mom]
[t-1: trend, momentum, volatility, volume, candle_char, bb_pos, short_mom]
[t-0: trend, momentum, volatility, volume, candle_char, bb_pos, short_mom]
```

All features are already normalized to roughly [-1, 1] by `MarketFeatures`. The 3-snapshot approach captures trajectory without needing a separate sequence model.

### ForwardReturns

Measured at 4 horizons from each observation candle:

| Horizon | Candles | ~Time (15m) | Purpose |
|---------|---------|-------------|---------|
| Short | 5 | 1.25h | Scalp signals |
| Medium | 10 | 2.5h | Primary signal |
| Long | 20 | 5h | Trend confirmation |
| Extended | 40 | 10h | Unlocked after library > 500 |

Each horizon records:
- `ret`: Close-to-close return (%)
- `max_up`: Maximum favorable excursion (%)
- `max_down`: Maximum adverse excursion (%)

Forward returns are computed only after all candles in the horizon have been observed — no look-ahead.

### MarketObservation

```rust
struct MarketObservation {
    vector: [f64; 21],           // temporal feature vector
    forward_returns: Option<ForwardReturns>,  // None while pending
    symbol: String,
    candle_index: usize,
    entry_price: f64,
}
```

### PatternLibrary

```rust
struct PatternLibrary {
    global: VecDeque<MarketObservation>,      // cap: 5000
    per_symbol: HashMap<String, VecDeque<MarketObservation>>,  // cap: 2000 each
    pending: VecDeque<PendingObservation>,    // awaiting forward return completion
}
```

**Scope:** Per-symbol library is used when it has >= 200 observations with completed returns; otherwise falls back to global. Results are merged when both are available (70% per-symbol, 30% global weight).

**Eviction:** FIFO via VecDeque — oldest observations drop when capacity is reached, naturally implementing recency weighting.

## 2. BacktestEngine Integration

The candle processing loop gains a new step between indicator computation and decision-making:

```
Existing steps 1-8: candle ingestion, indicator computation, market state update
NEW step 8.5: library.on_candle(market_state, features, candle_index)
Modified step 9: engine.decide(state, features, &library) — library passed as read-only ref
Existing steps 10+: position management, exit checks
```

### `library.on_candle()` does three things:

1. **Resolve pending observations:** For each pending observation whose max horizon has elapsed, compute forward returns from the candle history and move to the completed library.

2. **Record new observation:** Build a `TemporalVector` from the current and two prior feature snapshots. Create a new pending observation (requires candle_index >= 2).

3. **Anti-look-ahead guarantee:** Forward returns for observation at candle `i` are computed at candle `i + max_horizon` using only candles `i..=i+max_horizon`. The observation enters the queryable library only after all returns are resolved. Pending observations are never queryable.

## 3. KNN Decision Logic

### Query Process

```rust
fn predict(&self, query: &[f64; 21], symbol: &str, k: usize) -> Option<MarketPrediction>
```

1. Select library: per-symbol if >= 200 completed, else global
2. Compute Euclidean distances from query to all completed observations
3. Take K nearest (K=20, adaptive: min(20, sqrt(library_size)))
4. Weight by inverse distance: `w_i = 1 / (distance_i + epsilon)`

### MarketPrediction

```rust
struct MarketPrediction {
    expected_return: f64,        // weighted mean of forward returns (medium horizon)
    directional_prob: f64,       // fraction of neighbors with positive return
    avg_max_up: f64,             // weighted mean max favorable excursion
    avg_max_down: f64,           // weighted mean max adverse excursion
    neighbor_count: usize,       // how many neighbors found
    avg_distance: f64,           // mean distance to neighbors (confidence proxy)
}
```

### Minimum confidence threshold

- Need at least 10 neighbors to generate a prediction
- If `avg_distance > 2.0` (far from any known pattern), prediction is discarded
- Below threshold: fall back to existing Thompson-only behavior (backwards compatible)

## 4. Thompson Integration

### Warm-Start Priors from KNN

When KNN returns a valid prediction, Thompson priors are adjusted:

```
If directional_prob > 0.6:  // KNN suggests long
    long_arm.alpha  += directional_prob * confidence_scale
    short_arm.beta  += directional_prob * confidence_scale

If directional_prob < 0.4:  // KNN suggests short
    short_arm.alpha += (1 - directional_prob) * confidence_scale
    long_arm.beta   += (1 - directional_prob) * confidence_scale

confidence_scale = clamp(1.0 / avg_distance, 0.5, 3.0)
```

Thompson still samples and can override KNN — KNN just tilts the priors. With high distance (low confidence), the tilt is small. With close neighbors, the tilt is strong.

### Position Sizing from KNN

```
base_kelly = existing_kelly_from_adaptive_params
knn_sizing = directional_prob * (avg_max_up / avg_max_down) * base_kelly
final_size = 0.6 * knn_sizing + 0.4 * base_kelly   // blend
```

### Stop-Loss from KNN

```
sl_distance = avg_max_down * 1.2   // 20% buffer beyond typical adverse excursion
```

Falls back to ATR-based SL when KNN has insufficient data.

## 5. AdaptiveParams Split

### Market prediction params (from KNN accuracy/density):

| Parameter | Source | Formula |
|-----------|--------|---------|
| exploration_bonus | Library density | `0.3 * (1 - lib_size/5000)` — explore more when library is small |
| kelly_scale | KNN prediction accuracy | EMA of `(predicted_direction == actual_direction)` |
| position_size_cap | KNN confidence | `base_cap * (1 + kelly_scale * 0.5)` |

### Execution params (still from trade EMAs):

Cooldown, min/max hold duration, trailing stop activation/distance, breakeven threshold — these remain driven by actual trade outcome EMAs in AdaptiveParams. Trade execution dynamics differ from market prediction.

### Reward Signal Update

```
thompson_reward = 0.6 * sigmoid(trade_pnl) + 0.4 * prediction_accuracy
```

Where `prediction_accuracy = 1.0 if predicted_direction == actual_return_direction, else 0.0`.

This blends trade outcome with prediction quality, rewarding arms that lead to both profitable trades AND accurate market reads.

## 6. Adaptive Horizons

Start with short horizons (5/10/20 candles). The 40-candle extended horizon unlocks when:
- Library has > 500 completed observations
- Short-horizon prediction accuracy > 55%

This prevents noise from long-horizon predictions when the library is too small to be reliable.

## 7. Testing Strategy

### Unit Tests

1. **TemporalVector construction** — verify 21-dim vector from 3 feature snapshots
2. **Forward return computation** — known price series → exact returns and excursions
3. **Anti-look-ahead** — pending observations never appear in query results
4. **KNN with known data** — insert 3 clusters, query near each, verify correct neighbors
5. **Distance weighting** — closer neighbors weighted more heavily
6. **Library capacity** — verify FIFO eviction at 5000 global / 2000 per-symbol
7. **Minimum confidence** — predictions rejected when < 10 neighbors or avg_distance > 2.0
8. **Thompson prior warm-start** — verify alpha/beta adjustments from KNN predictions
9. **Adaptive horizon unlock** — verify 40-candle horizon activates at threshold

### Integration Test

10. **Full backtest with library** — run 100 candles through BacktestEngine with PatternLibrary enabled, verify library grows, predictions generated, no panics

## 8. File Plan

| File | Action | Content |
|------|--------|---------|
| `src/engine/pattern_library.rs` | Create | `PatternLibrary`, `MarketObservation`, `TemporalVector`, `ForwardReturns`, `MarketPrediction`, KNN logic |
| `src/engine/mod.rs` | Modify | Add `pub mod pattern_library` |
| `src/engine/learner.rs` | Modify | Accept `&PatternLibrary` in `decide()`, warm-start Thompson, blend sizing |
| `src/engine/adaptive.rs` | Modify | Add KNN-derived params (kelly_scale, exploration_bonus) |
| `src/engine/trade_manager.rs` | Modify | Pass KNN stop-loss to SL calculation |
| `src/backtest/engine.rs` | Modify | Add `library.on_candle()` call, pass library to `decide()` |
| `src/backtest/validation.rs` | Modify | Include library in walk-forward folds |

## 9. Backwards Compatibility

When the library has insufficient data (< 10 completed observations), the system falls back entirely to existing Thompson-only behavior. No existing behavior changes until the library is populated. This makes the feature purely additive — existing tests should continue passing without modification.
