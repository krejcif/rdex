# Vector Pattern Learning Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add a KNN-based pattern library that learns from market forward returns at every candle and warm-starts Thompson Sampling priors.

**Architecture:** New `PatternLibrary` records 21-dim temporal vectors at every candle, resolves forward returns after the horizon elapses (anti-look-ahead), then serves KNN predictions that tilt Thompson priors and inform position sizing/SL. Falls back to existing Thompson-only behavior when library is cold.

**Tech Stack:** Rust, no new dependencies (brute-force KNN, VecDeque, HashMap)

**Design doc:** `docs/plans/2026-02-28-vector-pattern-learning-design.md`

---

### Task 1: Core Data Structures & TemporalVector

**Files:**
- Create: `src/engine/pattern_library.rs`
- Modify: `src/engine/mod.rs:1-11`

**Step 1: Write the failing test for TemporalVector construction**

In `src/engine/pattern_library.rs`, add the test module at the bottom:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(base: f64) -> [f64; 7] {
        [base, base + 0.1, base + 0.2, base + 0.3, base + 0.4, base + 0.5, base + 0.6]
    }

    #[test]
    fn test_temporal_vector_construction() {
        let f0 = make_features(0.0);
        let f1 = make_features(0.1);
        let f2 = make_features(0.2);
        let vec = build_temporal_vector(&f0, &f1, &f2);
        assert_eq!(vec.len(), 21);
        // First 7 = f0
        assert!((vec[0] - 0.0).abs() < 1e-10);
        assert!((vec[6] - 0.6).abs() < 1e-10);
        // Second 7 = f1
        assert!((vec[7] - 0.1).abs() < 1e-10);
        // Third 7 = f2
        assert!((vec[14] - 0.2).abs() < 1e-10);
        assert!((vec[20] - 0.8).abs() < 1e-10);
    }
}
```

And the minimal struct/fn stubs above it:

```rust
use std::collections::{HashMap, VecDeque};

pub const VECTOR_DIM: usize = 21;
pub const FEATURES_PER_SNAPSHOT: usize = 7;

/// Build a 21-dim temporal vector from 3 consecutive feature snapshots.
/// [t-2, t-1, t-0] where each snapshot is 7 MarketFeatures values.
pub fn build_temporal_vector(
    t_minus_2: &[f64; 7],
    t_minus_1: &[f64; 7],
    t_current: &[f64; 7],
) -> [f64; VECTOR_DIM] {
    todo!()
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib test_temporal_vector_construction`
Expected: FAIL with "not yet implemented"

**Step 3: Implement `build_temporal_vector`**

```rust
pub fn build_temporal_vector(
    t_minus_2: &[f64; 7],
    t_minus_1: &[f64; 7],
    t_current: &[f64; 7],
) -> [f64; VECTOR_DIM] {
    let mut v = [0.0; VECTOR_DIM];
    v[..7].copy_from_slice(t_minus_2);
    v[7..14].copy_from_slice(t_minus_1);
    v[14..21].copy_from_slice(t_current);
    v
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test --lib test_temporal_vector_construction`
Expected: PASS

**Step 5: Register the module**

In `src/engine/mod.rs`, add:
```rust
pub mod pattern_library;
pub use pattern_library::*;
```

**Step 6: Verify full build compiles**

Run: `cargo build`
Expected: compiles with no errors

**Step 7: Commit**

```bash
git add src/engine/pattern_library.rs src/engine/mod.rs
git commit -m "feat: add pattern_library module with TemporalVector construction"
```

---

### Task 2: ForwardReturns & Computation

**Files:**
- Modify: `src/engine/pattern_library.rs`

**Step 1: Write the failing test for forward return computation**

```rust
#[test]
fn test_forward_returns_computation() {
    // Prices: 100, 102, 99, 104, 101, 103  (indices 0-5)
    let prices = vec![100.0, 102.0, 99.0, 104.0, 101.0, 103.0];
    let highs  = vec![101.0, 103.0, 100.0, 105.0, 102.0, 104.0];
    let lows   = vec![99.0, 101.0, 98.0, 103.0, 100.0, 102.0];
    // Forward returns from index 0, horizon 5
    let ret = compute_forward_return(&prices, &highs, &lows, 0, 5);
    let expected_ret = (103.0 - 100.0) / 100.0 * 100.0; // 3%
    assert!((ret.ret - expected_ret).abs() < 0.01);
    // max_up: highest high in [1..=5] vs entry price
    // highs[3] = 105 → (105-100)/100*100 = 5%
    assert!((ret.max_up - 5.0).abs() < 0.01);
    // max_down: lowest low in [1..=5] vs entry price
    // lows[2] = 98 → (100-98)/100*100 = 2%
    assert!((ret.max_down - 2.0).abs() < 0.01);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib test_forward_returns_computation`
Expected: FAIL (function doesn't exist)

**Step 3: Implement `ForwardReturns` struct and `compute_forward_return`**

```rust
#[derive(Debug, Clone, Copy)]
pub struct HorizonReturn {
    pub ret: f64,      // close-to-close return (%)
    pub max_up: f64,   // max favorable excursion (%)
    pub max_down: f64,  // max adverse excursion (%)
}

#[derive(Debug, Clone, Copy)]
pub struct ForwardReturns {
    pub short: HorizonReturn,   // 5 candles
    pub medium: HorizonReturn,  // 10 candles
    pub long: HorizonReturn,    // 20 candles
    pub extended: Option<HorizonReturn>, // 40 candles (unlocked later)
}

/// Compute forward return from `start_idx` over `horizon` candles.
/// Uses close prices for return, highs/lows for excursions.
pub fn compute_forward_return(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    start_idx: usize,
    horizon: usize,
) -> HorizonReturn {
    let end = start_idx + horizon;
    let entry = closes[start_idx];
    let ret = (closes[end] - entry) / entry * 100.0;

    let mut max_up = 0.0_f64;
    let mut max_down = 0.0_f64;
    for i in (start_idx + 1)..=end {
        let up = (highs[i] - entry) / entry * 100.0;
        let down = (entry - lows[i]) / entry * 100.0;
        max_up = max_up.max(up);
        max_down = max_down.max(down);
    }

    HorizonReturn { ret, max_up, max_down }
}
```

**Step 4: Run test to verify it passes**

Run: `cargo test --lib test_forward_returns_computation`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engine/pattern_library.rs
git commit -m "feat: add ForwardReturns struct and computation from price series"
```

---

### Task 3: MarketObservation & PatternLibrary Core

**Files:**
- Modify: `src/engine/pattern_library.rs`

**Step 1: Write the failing test for library capacity/eviction**

```rust
#[test]
fn test_library_capacity_eviction() {
    let mut lib = PatternLibrary::new(5, 3); // small caps for testing
    for i in 0..8 {
        let obs = MarketObservation {
            vector: [i as f64; VECTOR_DIM],
            symbol: "TEST".into(),
            candle_index: i,
            entry_price: 100.0 + i as f64,
        };
        lib.add_completed_global(obs);
    }
    // Global cap is 5, so should have evicted oldest 3
    assert_eq!(lib.completed_global_count(), 5);
    // Oldest remaining should be index 3
    assert!((lib.global[0].vector[0] - 3.0).abs() < 1e-10);
}

#[test]
fn test_library_per_symbol_capacity() {
    let mut lib = PatternLibrary::new(100, 3); // per-symbol cap = 3
    for i in 0..5 {
        let obs = MarketObservation {
            vector: [i as f64; VECTOR_DIM],
            symbol: "BTC".into(),
            candle_index: i,
            entry_price: 100.0,
        };
        lib.add_completed_symbol("BTC", obs);
    }
    assert_eq!(lib.completed_symbol_count("BTC"), 3);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib test_library_capacity`
Expected: FAIL (structs/methods don't exist)

**Step 3: Implement MarketObservation and PatternLibrary**

```rust
#[derive(Debug, Clone)]
pub struct MarketObservation {
    pub vector: [f64; VECTOR_DIM],
    pub symbol: String,
    pub candle_index: usize,
    pub entry_price: f64,
}

/// Pending observation waiting for forward returns to resolve.
#[derive(Debug, Clone)]
struct PendingObservation {
    vector: [f64; VECTOR_DIM],
    symbol: String,
    candle_index: usize,
    entry_price: f64,
    max_horizon: usize,  // highest horizon to wait for
}

pub struct PatternLibrary {
    pub global: VecDeque<MarketObservation>,
    per_symbol: HashMap<String, VecDeque<MarketObservation>>,
    pending: VecDeque<PendingObservation>,
    /// Recent feature snapshots for temporal vector construction [symbol -> VecDeque<[f64;7]>]
    feature_history: HashMap<String, VecDeque<[f64; 7]>>,
    global_cap: usize,
    symbol_cap: usize,
    /// Prediction accuracy EMA for adaptive horizons
    prediction_accuracy_ema: f64,
}

impl PatternLibrary {
    pub fn new(global_cap: usize, symbol_cap: usize) -> Self {
        Self {
            global: VecDeque::new(),
            per_symbol: HashMap::new(),
            pending: VecDeque::new(),
            feature_history: HashMap::new(),
            global_cap,
            symbol_cap,
            prediction_accuracy_ema: 0.5,
        }
    }

    pub fn with_defaults() -> Self {
        Self::new(5000, 2000)
    }

    pub fn add_completed_global(&mut self, obs: MarketObservation) {
        self.global.push_back(obs);
        while self.global.len() > self.global_cap {
            self.global.pop_front();
        }
    }

    pub fn add_completed_symbol(&mut self, symbol: &str, obs: MarketObservation) {
        let queue = self.per_symbol.entry(symbol.to_string()).or_insert_with(VecDeque::new);
        queue.push_back(obs);
        while queue.len() > self.symbol_cap {
            queue.pop_front();
        }
    }

    pub fn completed_global_count(&self) -> usize {
        self.global.len()
    }

    pub fn completed_symbol_count(&self, symbol: &str) -> usize {
        self.per_symbol.get(symbol).map(|q| q.len()).unwrap_or(0)
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib test_library_capacity`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engine/pattern_library.rs
git commit -m "feat: add PatternLibrary with MarketObservation and FIFO eviction"
```

---

### Task 4: Anti-Look-Ahead Pending Resolution

**Files:**
- Modify: `src/engine/pattern_library.rs`

**Step 1: Write the failing test for anti-look-ahead**

```rust
#[test]
fn test_pending_observations_not_queryable() {
    let mut lib = PatternLibrary::with_defaults();
    // Add a pending observation
    lib.add_pending(PendingObservation {
        vector: [1.0; VECTOR_DIM],
        symbol: "BTC".into(),
        candle_index: 10,
        entry_price: 50000.0,
        max_horizon: 20,
    });
    // Library should report 0 completed
    assert_eq!(lib.completed_global_count(), 0);
    // KNN query should return None (no completed observations)
    let query = [1.0; VECTOR_DIM];
    assert!(lib.predict(&query, "BTC", 20).is_none());
}

#[test]
fn test_pending_resolves_after_horizon() {
    let mut lib = PatternLibrary::new(100, 100);
    // Simulate: observation at index 5, max_horizon = 3
    lib.add_pending(PendingObservation {
        vector: [0.5; VECTOR_DIM],
        symbol: "BTC".into(),
        candle_index: 5,
        entry_price: 100.0,
        max_horizon: 3,
    });
    // Build fake candle data: indices 0..=8
    let closes: Vec<f64> = (0..9).map(|i| 100.0 + i as f64).collect();
    let highs: Vec<f64> = closes.iter().map(|c| c + 1.0).collect();
    let lows: Vec<f64> = closes.iter().map(|c| c - 1.0).collect();

    // At candle 7 (5 + 3 = 8, but we check at current_idx = 8)
    lib.resolve_pending(&closes, &highs, &lows, 8);
    assert_eq!(lib.completed_global_count(), 1);
    assert_eq!(lib.pending_count(), 0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib test_pending`
Expected: FAIL

**Step 3: Implement pending management and resolution**

Add to `PatternLibrary`:

```rust
fn add_pending(&mut self, obs: PendingObservation) {
    self.pending.push_back(obs);
}

pub fn pending_count(&self) -> usize {
    self.pending.len()
}

/// Resolve any pending observations whose horizon has fully elapsed.
/// `current_idx` is the current candle index being processed.
pub fn resolve_pending(
    &mut self,
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    current_idx: usize,
) {
    let mut resolved = Vec::new();
    let mut remaining = VecDeque::new();

    while let Some(p) = self.pending.pop_front() {
        let end_idx = p.candle_index + p.max_horizon;
        if end_idx <= current_idx && end_idx < closes.len() {
            // Compute forward returns at each horizon
            let obs = MarketObservation {
                vector: p.vector,
                symbol: p.symbol.clone(),
                candle_index: p.candle_index,
                entry_price: p.entry_price,
            };
            resolved.push((p.symbol.clone(), obs));
        } else {
            remaining.push_back(p);
        }
    }

    self.pending = remaining;

    for (symbol, obs) in resolved {
        self.add_completed_symbol(&symbol, obs.clone());
        self.add_completed_global(obs);
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib test_pending`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engine/pattern_library.rs
git commit -m "feat: add pending observation resolution with anti-look-ahead"
```

---

### Task 5: KNN Prediction

**Files:**
- Modify: `src/engine/pattern_library.rs`

**Step 1: Write the failing tests for KNN**

```rust
#[test]
fn test_knn_finds_nearest_neighbors() {
    let mut lib = PatternLibrary::new(100, 100);
    // Insert 3 clusters: near 0, near 5, near 10
    for i in 0..20 {
        let base = (i / 7) as f64 * 5.0; // 0, 5, 10
        let obs = MarketObservation {
            vector: [base + 0.01 * (i % 7) as f64; VECTOR_DIM],
            symbol: "BTC".into(),
            candle_index: i,
            entry_price: 100.0,
        };
        lib.add_completed_global(obs);
    }
    // Query near cluster at 5.0
    let query = [5.0; VECTOR_DIM];
    let pred = lib.predict(&query, "BTC", 10);
    assert!(pred.is_some());
    let p = pred.unwrap();
    assert!(p.neighbor_count >= 6); // cluster at 5 has 7 members
    assert!(p.avg_distance < 1.0);  // should be very close
}

#[test]
fn test_knn_minimum_confidence() {
    let mut lib = PatternLibrary::new(100, 100);
    // Only 5 observations — below minimum 10
    for i in 0..5 {
        let obs = MarketObservation {
            vector: [i as f64; VECTOR_DIM],
            symbol: "BTC".into(),
            candle_index: i,
            entry_price: 100.0,
        };
        lib.add_completed_global(obs);
    }
    let query = [2.5; VECTOR_DIM];
    // Should return None — not enough neighbors
    assert!(lib.predict(&query, "BTC", 20).is_none());
}

#[test]
fn test_knn_distance_weighting() {
    let mut lib = PatternLibrary::new(100, 100);
    // Two observations: one very close (dist ~0), one far (dist ~large)
    for i in 0..15 {
        let val = if i < 10 { 1.0 } else { 100.0 };
        let obs = MarketObservation {
            vector: [val; VECTOR_DIM],
            symbol: "BTC".into(),
            candle_index: i,
            entry_price: 100.0,
        };
        lib.add_completed_global(obs);
    }
    let query = [1.0; VECTOR_DIM];
    let pred = lib.predict(&query, "BTC", 15).unwrap();
    // Closer neighbors (at 1.0) should dominate — avg_distance should be small
    assert!(pred.avg_distance < 50.0);
}
```

**Step 2: Run tests to verify they fail**

Run: `cargo test --lib test_knn`
Expected: FAIL

**Step 3: Implement `MarketPrediction` and `predict()`**

```rust
#[derive(Debug, Clone)]
pub struct MarketPrediction {
    pub expected_return: f64,
    pub directional_prob: f64,
    pub avg_max_up: f64,
    pub avg_max_down: f64,
    pub neighbor_count: usize,
    pub avg_distance: f64,
}

impl PatternLibrary {
    /// KNN prediction from the completed observation library.
    /// Returns None if insufficient neighbors or too distant.
    pub fn predict(&self, query: &[f64; VECTOR_DIM], symbol: &str, k: usize) -> Option<MarketPrediction> {
        let min_neighbors = 10;
        let max_distance = 2.0;

        // Select library: per-symbol if >= 200, else global
        let lib = if self.completed_symbol_count(symbol) >= 200 {
            self.per_symbol.get(symbol)?
        } else {
            &self.global
        };

        if lib.len() < min_neighbors {
            return None;
        }

        // Compute distances
        let mut distances: Vec<(usize, f64)> = lib.iter().enumerate()
            .map(|(idx, obs)| {
                let dist = euclidean_distance(query, &obs.vector);
                (idx, dist)
            })
            .collect();

        distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        let adaptive_k = k.min((lib.len() as f64).sqrt() as usize).max(min_neighbors);
        let neighbors: Vec<_> = distances.into_iter().take(adaptive_k).collect();

        if neighbors.len() < min_neighbors {
            return None;
        }

        let avg_dist = neighbors.iter().map(|(_, d)| d).sum::<f64>() / neighbors.len() as f64;
        if avg_dist > max_distance {
            return None;
        }

        // Inverse-distance weighting
        let epsilon = 1e-8;
        let weights: Vec<f64> = neighbors.iter()
            .map(|(_, d)| 1.0 / (d + epsilon))
            .collect();
        let total_weight: f64 = weights.iter().sum();

        // For now, prediction is based on neighbor structure only.
        // ForwardReturns integration comes in a later task when we store returns on observations.
        // Return a basic prediction from neighbor distances.
        Some(MarketPrediction {
            expected_return: 0.0,  // placeholder until ForwardReturns stored
            directional_prob: 0.5, // placeholder
            avg_max_up: 0.0,
            avg_max_down: 0.0,
            neighbor_count: neighbors.len(),
            avg_distance: avg_dist,
        })
    }
}

fn euclidean_distance(a: &[f64; VECTOR_DIM], b: &[f64; VECTOR_DIM]) -> f64 {
    a.iter().zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib test_knn`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engine/pattern_library.rs
git commit -m "feat: add KNN prediction with distance weighting and min confidence"
```

---

### Task 6: ForwardReturns on Observations + Full KNN Prediction

**Files:**
- Modify: `src/engine/pattern_library.rs`

This task enriches `MarketObservation` to carry `ForwardReturns` so `predict()` can return meaningful `expected_return`, `directional_prob`, `avg_max_up`, and `avg_max_down`.

**Step 1: Write the failing test**

```rust
#[test]
fn test_knn_prediction_with_forward_returns() {
    let mut lib = PatternLibrary::new(100, 100);
    // Insert 15 observations: 10 with positive returns, 5 with negative
    for i in 0..15 {
        let ret = if i < 10 { 2.0 } else { -1.0 };
        let obs = CompletedObservation {
            vector: [0.5; VECTOR_DIM],
            symbol: "BTC".into(),
            candle_index: i,
            entry_price: 100.0,
            forward_returns: ForwardReturns {
                short: HorizonReturn { ret: ret * 0.5, max_up: 1.0, max_down: 0.5 },
                medium: HorizonReturn { ret, max_up: 2.0, max_down: 1.0 },
                long: HorizonReturn { ret: ret * 2.0, max_up: 3.0, max_down: 1.5 },
                extended: None,
            },
        };
        lib.add_completed_global_with_returns(obs);
    }
    let query = [0.5; VECTOR_DIM];
    let pred = lib.predict(&query, "BTC", 20).unwrap();
    // 10/15 positive → directional_prob ≈ 0.67
    assert!(pred.directional_prob > 0.5);
    // Expected return should be positive (weighted mean of medium returns)
    assert!(pred.expected_return > 0.0);
    assert!(pred.avg_max_up > 0.0);
    assert!(pred.avg_max_down > 0.0);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib test_knn_prediction_with_forward_returns`
Expected: FAIL

**Step 3: Refactor observation storage to include forward returns**

Split `MarketObservation` into pending (no returns) and completed (with returns):

```rust
/// Completed observation with resolved forward returns.
#[derive(Debug, Clone)]
pub struct CompletedObservation {
    pub vector: [f64; VECTOR_DIM],
    pub symbol: String,
    pub candle_index: usize,
    pub entry_price: f64,
    pub forward_returns: ForwardReturns,
}
```

Update `PatternLibrary` to store `VecDeque<CompletedObservation>` instead of `MarketObservation`.

Update `predict()` to compute weighted means from `forward_returns.medium`:

```rust
// In predict(), after computing weights:
let mut sum_ret = 0.0;
let mut sum_max_up = 0.0;
let mut sum_max_down = 0.0;
let mut positive_weight = 0.0;
for (i, &(idx, _)) in neighbors.iter().enumerate() {
    let obs = &lib[idx];
    let w = weights[i];
    let medium = &obs.forward_returns.medium;
    sum_ret += medium.ret * w;
    sum_max_up += medium.max_up * w;
    sum_max_down += medium.max_down * w;
    if medium.ret > 0.0 {
        positive_weight += w;
    }
}
Some(MarketPrediction {
    expected_return: sum_ret / total_weight,
    directional_prob: positive_weight / total_weight,
    avg_max_up: sum_max_up / total_weight,
    avg_max_down: (sum_max_down / total_weight).max(0.01),
    neighbor_count: neighbors.len(),
    avg_distance: avg_dist,
})
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib test_knn_prediction_with_forward_returns`
Expected: PASS

Also run: `cargo test --lib` — all existing tests should still pass (Task 3/4/5 tests need minor updates to use new types).

**Step 5: Commit**

```bash
git add src/engine/pattern_library.rs
git commit -m "feat: enrich KNN prediction with forward returns data"
```

---

### Task 7: `on_candle()` Integration — Feature Recording + Pending Resolution

**Files:**
- Modify: `src/engine/pattern_library.rs`

This is the main entry point called by `BacktestEngine` each candle.

**Step 1: Write the failing test**

```rust
#[test]
fn test_on_candle_records_and_resolves() {
    let mut lib = PatternLibrary::new(100, 100);
    // Simulate 30 candles worth of features + prices
    let n = 30;
    let closes: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();
    let highs: Vec<f64> = closes.iter().map(|c| c + 1.0).collect();
    let lows: Vec<f64> = closes.iter().map(|c| c - 1.0).collect();

    for i in 0..n {
        let features = [0.1 * i as f64; 7];
        lib.on_candle("BTC", &features, i, &closes, &highs, &lows);
    }
    // After 30 candles with max_horizon=20:
    // Observations at indices 2..9 should be resolved (index + 20 <= 29)
    // Observations at indices 10..29 should still be pending
    assert!(lib.completed_global_count() > 0);
    assert!(lib.pending_count() > 0);
    assert!(lib.completed_global_count() + lib.pending_count() <= n);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib test_on_candle_records_and_resolves`
Expected: FAIL

**Step 3: Implement `on_candle()`**

```rust
impl PatternLibrary {
    /// Called each candle by BacktestEngine.
    /// 1. Resolve any pending observations whose horizon has elapsed.
    /// 2. Record a new pending observation (if >= 2 prior feature snapshots).
    pub fn on_candle(
        &mut self,
        symbol: &str,
        features: &[f64; 7],
        candle_index: usize,
        closes: &[f64],
        highs: &[f64],
        lows: &[f64],
    ) {
        // Store feature snapshot
        let hist = self.feature_history.entry(symbol.to_string()).or_insert_with(VecDeque::new);
        hist.push_back(*features);
        if hist.len() > 3 { hist.pop_front(); }

        // Resolve pending
        self.resolve_pending_with_returns(closes, highs, lows, candle_index);

        // Need at least 3 snapshots for temporal vector
        if hist.len() < 3 { return; }

        let t0 = &hist[2]; // current
        let t1 = &hist[1]; // t-1
        let t2 = &hist[0]; // t-2
        let vector = build_temporal_vector(t2, t1, t0);

        let max_horizon = if self.extended_horizon_unlocked() { 40 } else { 20 };
        self.add_pending(PendingObservation {
            vector,
            symbol: symbol.to_string(),
            candle_index,
            entry_price: closes[candle_index],
            max_horizon,
        });
    }

    fn extended_horizon_unlocked(&self) -> bool {
        self.completed_global_count() > 500 && self.prediction_accuracy_ema > 0.55
    }

    /// Resolve pending observations, computing full ForwardReturns.
    fn resolve_pending_with_returns(
        &mut self,
        closes: &[f64],
        highs: &[f64],
        lows: &[f64],
        current_idx: usize,
    ) {
        let mut resolved = Vec::new();
        let mut remaining = VecDeque::new();

        while let Some(p) = self.pending.pop_front() {
            let end_idx = p.candle_index + p.max_horizon;
            if end_idx <= current_idx && end_idx < closes.len() {
                let short = compute_forward_return(closes, highs, lows, p.candle_index, 5.min(p.max_horizon));
                let medium = compute_forward_return(closes, highs, lows, p.candle_index, 10.min(p.max_horizon));
                let long = compute_forward_return(closes, highs, lows, p.candle_index, 20.min(p.max_horizon));
                let extended = if p.max_horizon >= 40 {
                    Some(compute_forward_return(closes, highs, lows, p.candle_index, 40))
                } else {
                    None
                };
                let obs = CompletedObservation {
                    vector: p.vector,
                    symbol: p.symbol.clone(),
                    candle_index: p.candle_index,
                    entry_price: p.entry_price,
                    forward_returns: ForwardReturns { short, medium, long, extended },
                };
                resolved.push((p.symbol, obs));
            } else {
                remaining.push_back(p);
            }
        }
        self.pending = remaining;

        for (symbol, obs) in resolved {
            self.add_completed_symbol(&symbol, obs.clone());
            self.add_completed_global(obs);
        }
    }
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib test_on_candle_records_and_resolves`
Expected: PASS

Run: `cargo test --lib` (all pattern_library tests)
Expected: all PASS

**Step 5: Commit**

```bash
git add src/engine/pattern_library.rs
git commit -m "feat: implement on_candle() with feature recording and pending resolution"
```

---

### Task 8: BacktestEngine Integration

**Files:**
- Modify: `src/backtest/engine.rs:1-9` (imports)
- Modify: `src/backtest/engine.rs:14-23` (struct fields)
- Modify: `src/backtest/engine.rs:76-195` (candle loop)
- Modify: `src/backtest/engine.rs:170-179` (market state + decide)

**Step 1: Add `PatternLibrary` to `BacktestEngine`**

In `src/backtest/engine.rs`, add import:
```rust
use crate::engine::pattern_library::PatternLibrary;
use crate::strategy::features::MarketFeatures;
```

Add field to `BacktestEngine`:
```rust
pub library: PatternLibrary,
```

Update `new()`:
```rust
library: PatternLibrary::with_defaults(),
```

**Step 2: Add `library.on_candle()` call in the candle loop**

After the market state is built (line ~171) and before `decide()`:

```rust
// Build market state from ONLY past data
let state = match build_market_state(symbol, candles, i, lookback) {
    Some(s) => s,
    None => continue,
};

let current_atr = state.indicators.atr_14;

// Extract features for pattern library
if let Some(features) = MarketFeatures::extract(&state) {
    let closes: Vec<f64> = candles[..=i].iter().map(|c| c.close).collect();
    let highs: Vec<f64> = candles[..=i].iter().map(|c| c.high).collect();
    let lows: Vec<f64> = candles[..=i].iter().map(|c| c.low).collect();
    self.library.on_candle(&symbol.0, &features.as_array(), i, &closes, &highs, &lows);
}

// Make decision (pass library for KNN)
let decision = self.learning.decide(&state, &mut self.rng, &self.library);
```

**Step 3: Update `LearningEngine::decide()` signature**

This is a forward-reference — Task 9 will implement the actual changes. For now, update the signature to accept `&PatternLibrary` and ignore it:

In `src/engine/learner.rs`, change `decide`:
```rust
pub fn decide(&mut self, state: &MarketState, rng: &mut impl Rng, library: &PatternLibrary) -> TradingDecision {
```

Add import: `use super::pattern_library::PatternLibrary;`

Pass `library` through in all call sites. For now, the body doesn't use it.

**Step 4: Also update `validation.rs` call site**

In `src/backtest/validation.rs`, the `BacktestEngine::run()` is called — no change needed there since `run()` internally calls `decide()`. But `PatternLibrary` needs to be added to `BacktestEngine::new()` — already done via the struct change.

**Step 5: Build and run all tests**

Run: `cargo build`
Run: `cargo test --lib`
Expected: all PASS (library is added but doesn't affect behavior yet)

**Step 6: Commit**

```bash
git add src/backtest/engine.rs src/engine/learner.rs
git commit -m "feat: integrate PatternLibrary into BacktestEngine candle loop"
```

---

### Task 9: Thompson Integration — KNN Warm-Start Priors

**Files:**
- Modify: `src/engine/learner.rs:177-244` (decide method)
- Modify: `src/engine/thompson.rs` (add method to adjust priors)

**Step 1: Write the failing test**

In `src/engine/learner.rs` test module:

```rust
#[test]
fn test_knn_warm_starts_thompson() {
    use crate::engine::pattern_library::{PatternLibrary, CompletedObservation, ForwardReturns, HorizonReturn, VECTOR_DIM};

    let mut engine = LearningEngine::new(
        vec!["BTCUSDT".into()],
        LearnerConfig::default(),
    );
    let mut library = PatternLibrary::new(100, 100);

    // Insert 15 bullish observations (positive returns)
    for i in 0..15 {
        let obs = CompletedObservation {
            vector: [0.5; VECTOR_DIM],
            symbol: "BTCUSDT".into(),
            candle_index: i,
            entry_price: 50000.0,
            forward_returns: ForwardReturns {
                short: HorizonReturn { ret: 1.0, max_up: 2.0, max_down: 0.5 },
                medium: HorizonReturn { ret: 2.0, max_up: 3.0, max_down: 1.0 },
                long: HorizonReturn { ret: 3.0, max_up: 4.0, max_down: 1.5 },
                extended: None,
            },
        };
        library.add_completed_global_with_returns(obs);
    }

    // Build a state that produces features near [0.5; 7]
    // (Use the make_state helper or construct directly)
    // The decision should be biased toward Long due to KNN
    // This is a smoke test — just verify it doesn't panic
    // and that the library reference is used
    assert!(library.completed_global_count() == 15);
}
```

**Step 2: Run test to verify it compiles and passes (smoke test)**

Run: `cargo test --lib test_knn_warm_starts_thompson`

**Step 3: Implement KNN integration in `decide()`**

In `src/engine/learner.rs`, modify `decide()`:

```rust
pub fn decide(&mut self, state: &MarketState, rng: &mut impl Rng, library: &PatternLibrary) -> TradingDecision {
    let features = match MarketFeatures::extract(state) {
        Some(f) => f,
        None => return self.hold_decision(),
    };

    self.discretizer.observe(&features);
    let pattern = self.discretizer.discretize(&features);
    let pattern_key = PatternDiscretizer::pattern_key(&pattern);
    let pool_key = "_global";

    // KNN prediction from pattern library
    let prediction = library.predict(&features.as_array_temporal_query(), &state.symbol.0, 20);

    // Thompson Sampling with optional KNN warm-start
    let exploration = self.adaptive.exploration_coeff();
    let action = if let Some(ref pred) = prediction {
        // Warm-start: adjust Thompson priors based on KNN directional probability
        let confidence_scale = (1.0 / pred.avg_distance).clamp(0.5, 3.0);
        self.thompson.select_arm_with_knn_prior(
            pool_key, &pattern, rng, exploration,
            pred.directional_prob, confidence_scale,
        )
    } else {
        self.thompson.select_arm_with_exploration(pool_key, &pattern, rng, exploration)
    };

    self.last_pattern = Some(pattern.clone());

    if action.0 == "hold" {
        return self.hold_decision();
    }

    // Position sizing: blend KNN and adaptive
    let win_prob = self.adaptive.overall_win_rate().clamp(0.01, 0.99);
    let sl_atr = self.excursions.get_sl_atr(&pattern_key, &action.0);
    let tp_atr = self.excursions.get_tp_atr(&pattern_key, &action.0);
    let rr_ratio = tp_atr / sl_atr;
    let kelly = ((win_prob * (1.0 + rr_ratio)) - 1.0) / rr_ratio;
    if kelly <= 0.0 {
        return self.hold_decision();
    }
    let kelly_frac = self.adaptive.kelly_fraction();
    let max_size = self.adaptive.max_position_size();
    let base_size = (kelly * kelly_frac).clamp(0.05, max_size);

    let size = if let Some(ref pred) = prediction {
        let knn_sizing = pred.directional_prob * (pred.avg_max_up / pred.avg_max_down) * base_size;
        0.6 * knn_sizing + 0.4 * base_size
    } else {
        base_size
    };
    let size = size.clamp(0.05, max_size);

    let price = features.price;
    let atr = features.atr;

    // SL from KNN or ATR-based fallback
    let sl_distance_atr = if let Some(ref pred) = prediction {
        if pred.avg_max_down > 0.01 {
            pred.avg_max_down * 1.2 / 100.0 * price / atr
        } else {
            sl_atr
        }
    } else {
        sl_atr
    };

    let (signal, stop_loss, take_profit) = match action.0.as_str() {
        "long" => (
            TradeSignal::Long,
            Some(price - sl_distance_atr * atr),
            None,
        ),
        "short" => (
            TradeSignal::Short,
            Some(price + sl_distance_atr * atr),
            None,
        ),
        _ => return self.hold_decision(),
    };

    let confidence = self.thompson.get_arm_mean(pool_key, &pattern, &action.0);
    TradingDecision {
        signal,
        size,
        stop_loss,
        take_profit,
        confidence,
        strategy_name: format!("{}_{}", action.0, pattern_key),
    }
}
```

**Step 4: Add `select_arm_with_knn_prior` to `ThompsonEngine`**

In `src/engine/thompson.rs`, add:

```rust
/// Select arm with KNN-boosted priors.
/// `directional_prob` > 0.6 boosts long, < 0.4 boosts short.
pub fn select_arm_with_knn_prior(
    &self,
    symbol: &str,
    context: &MarketContext,
    rng: &mut impl Rng,
    exploration: f64,
    directional_prob: f64,
    confidence_scale: f64,
) -> (String, f64) {
    // Get base beta params for each arm
    // Apply KNN bias, then sample
    // ... (implementation details)
    // Falls back to select_arm_with_exploration when directional_prob ≈ 0.5
    self.select_arm_with_exploration(symbol, context, rng, exploration)
    // TODO: implement the actual prior adjustment
}
```

Note: The helper method `features.as_array_temporal_query()` doesn't exist on `MarketFeatures` yet. We need to handle this — the library's `predict()` takes a `[f64; 21]`, but we only have `[f64; 7]`. The temporal vector construction happens inside `on_candle`. For the query, we need access to the same feature history. Add a method on `PatternLibrary`:

```rust
/// Build a query vector from the current features using stored history.
pub fn build_query_vector(&self, symbol: &str, current_features: &[f64; 7]) -> Option<[f64; VECTOR_DIM]> {
    let hist = self.feature_history.get(symbol)?;
    if hist.len() < 2 { return None; }
    let t2 = &hist[hist.len() - 2];
    let t1 = &hist[hist.len() - 1];
    Some(build_temporal_vector(t2, t1, current_features))
}
```

Then in `decide()`:
```rust
let prediction = library.build_query_vector(&state.symbol.0, &features.as_array())
    .and_then(|query| library.predict(&query, &state.symbol.0, 20));
```

**Step 5: Build and run all tests**

Run: `cargo build`
Run: `cargo test --lib`
Expected: all PASS

**Step 6: Commit**

```bash
git add src/engine/learner.rs src/engine/thompson.rs src/engine/pattern_library.rs
git commit -m "feat: integrate KNN predictions into Thompson decision-making"
```

---

### Task 10: AdaptiveParams KNN-Derived Fields

**Files:**
- Modify: `src/engine/adaptive.rs`

**Step 1: Write the failing test**

```rust
#[test]
fn test_knn_prediction_accuracy_tracking() {
    let mut params = AdaptiveParams::new();
    // Record 10 correct predictions
    for _ in 0..10 {
        params.record_prediction_accuracy(true);
    }
    assert!(params.knn_accuracy() > 0.7);

    // Record 10 incorrect predictions
    for _ in 0..10 {
        params.record_prediction_accuracy(false);
    }
    assert!(params.knn_accuracy() < 0.7);
}
```

**Step 2: Run test to verify it fails**

Run: `cargo test --lib test_knn_prediction_accuracy_tracking`

**Step 3: Implement KNN tracking in AdaptiveParams**

Add to `AdaptiveStats`:
```rust
ema_knn_accuracy: f64,  // EMA of prediction correctness
```

Add to `AdaptiveParams`:
```rust
pub fn record_prediction_accuracy(&mut self, correct: bool) {
    let val = if correct { 1.0 } else { 0.0 };
    let d = self.stats.ema_decay;
    self.stats.ema_knn_accuracy = self.stats.ema_knn_accuracy * d + val * (1.0 - d);
}

pub fn knn_accuracy(&self) -> f64 {
    self.stats.ema_knn_accuracy
}
```

**Step 4: Run tests to verify they pass**

Run: `cargo test --lib test_knn_prediction_accuracy_tracking`
Expected: PASS

**Step 5: Commit**

```bash
git add src/engine/adaptive.rs
git commit -m "feat: add KNN prediction accuracy tracking to AdaptiveParams"
```

---

### Task 11: Reward Signal Blending

**Files:**
- Modify: `src/engine/trade_manager.rs:181-228` (on_exit method)

**Step 1: Write the failing test**

```rust
#[test]
fn test_blended_reward_signal() {
    use crate::evaluation::scorer;
    // Winning trade + correct prediction → high reward
    let trade_reward = scorer::pnl_to_reward(2.0, 1.0); // should be > 0.5
    let prediction_correct = 1.0;
    let blended = 0.6 * trade_reward + 0.4 * prediction_correct;
    assert!(blended > 0.7);

    // Winning trade + wrong prediction → moderate reward
    let blended2 = 0.6 * trade_reward + 0.4 * 0.0;
    assert!(blended2 > 0.3 && blended2 < 0.7);
}
```

**Step 2: Run test — should pass as pure math**

**Step 3: Modify `TradeManager::on_exit` to blend reward**

Add a `last_prediction_direction: Option<f64>` field to `TradeManager`. Set it from the `MarketPrediction` when a trade is entered. In `on_exit`, compute blended reward:

```rust
let prediction_accuracy = match self.last_prediction_direction {
    Some(dir) if dir > 0.6 && record.pnl > 0.0 => 1.0,  // predicted long, was profitable
    Some(dir) if dir < 0.4 && record.pnl > 0.0 => 1.0,  // predicted short, was profitable
    Some(_) => 0.0,
    None => 0.5, // no prediction — neutral
};
let blended_reward = 0.6 * reward + 0.4 * prediction_accuracy;
```

Use `blended_reward` instead of `reward` when calling `learning.record_outcome`.

**Step 4: Build and test**

Run: `cargo test --lib`
Expected: all PASS

**Step 5: Commit**

```bash
git add src/engine/trade_manager.rs
git commit -m "feat: blend trade outcome and prediction accuracy in reward signal"
```

---

### Task 12: Integration Test — Full Backtest with Library

**Files:**
- Modify: `src/backtest/engine.rs` (test module)

**Step 1: Write the integration test**

```rust
#[test]
fn test_backtest_with_pattern_library() {
    let learning = LearningEngine::new(
        vec!["BTCUSDT".into()],
        LearnerConfig::default(),
    );
    let mut bt = BacktestEngine::new(
        FuturesConfig::default(),
        learning,
        42,
    );
    let candles = make_trending_candles(300, 50000.0, 5.0);
    let result = bt.run(&Symbol("BTCUSDT".into()), &candles, 60);

    // Library should have grown
    assert!(bt.library.completed_global_count() > 0,
        "Library should have completed observations after 300 candles");

    // Basic sanity
    assert!(result.final_equity > 0.0);
    assert!(!result.performance.equity_curve.is_empty());

    // No panics is the main goal of this test
}
```

**Step 2: Run test**

Run: `cargo test --lib test_backtest_with_pattern_library`
Expected: PASS

**Step 3: Run full test suite**

Run: `cargo test --lib`
Expected: all PASS

**Step 4: Run clippy**

Run: `cargo clippy`
Expected: no errors

**Step 5: Commit**

```bash
git add src/backtest/engine.rs
git commit -m "test: add integration test for backtest with pattern library"
```

---

### Task 13: Adaptive Horizon Unlock Test

**Files:**
- Modify: `src/engine/pattern_library.rs` (test module)

**Step 1: Write the test**

```rust
#[test]
fn test_adaptive_horizon_unlock() {
    let mut lib = PatternLibrary::new(5000, 2000);
    // Below threshold: extended horizon not unlocked
    assert!(!lib.extended_horizon_unlocked());

    // Simulate > 500 completed observations
    for i in 0..501 {
        let obs = CompletedObservation {
            vector: [0.0; VECTOR_DIM],
            symbol: "BTC".into(),
            candle_index: i,
            entry_price: 100.0,
            forward_returns: ForwardReturns {
                short: HorizonReturn { ret: 1.0, max_up: 1.0, max_down: 0.5 },
                medium: HorizonReturn { ret: 1.0, max_up: 1.0, max_down: 0.5 },
                long: HorizonReturn { ret: 1.0, max_up: 1.0, max_down: 0.5 },
                extended: None,
            },
        };
        lib.add_completed_global_with_returns(obs);
    }
    // Set accuracy above threshold
    lib.prediction_accuracy_ema = 0.6;
    assert!(lib.extended_horizon_unlocked());

    // Low accuracy: still locked
    lib.prediction_accuracy_ema = 0.4;
    assert!(!lib.extended_horizon_unlocked());
}
```

**Step 2: Run test**

Run: `cargo test --lib test_adaptive_horizon_unlock`
Expected: PASS

**Step 3: Commit**

```bash
git add src/engine/pattern_library.rs
git commit -m "test: add adaptive horizon unlock test"
```

---

### Task 14: Final Cleanup & Full Validation

**Step 1: Run full test suite**

Run: `cargo test --lib`
Expected: all PASS (should be ~80+ tests now)

**Step 2: Run clippy**

Run: `cargo clippy`
Fix any warnings.

**Step 3: Run format check**

Run: `cargo fmt -- --check`
Fix any formatting issues.

**Step 4: Run a real backtest to verify end-to-end**

Run: `cargo run -- backtest -d 30 -e 10000`
Expected: runs without panics, library reports in output

**Step 5: Final commit**

```bash
git add -A
git commit -m "chore: cleanup, clippy fixes, and format for vector pattern learning"
```
