use std::collections::{HashMap, VecDeque};

// ── Constants ──────────────────────────────────────────────────────────────────

pub const FEATURES_PER_SNAPSHOT: usize = 7;
pub const VECTOR_DIM: usize = FEATURES_PER_SNAPSHOT * 3; // 21

/// Bundled price slices for candle processing (avoids too-many-arguments).
pub struct CandleSlices<'a> {
    pub closes: &'a [f64],
    pub highs: &'a [f64],
    pub lows: &'a [f64],
}

// ── Task 1: Temporal Vector ────────────────────────────────────────────────────

/// Concatenate three consecutive feature snapshots into a 21-dimensional
/// temporal vector: [t-2 | t-1 | t-0].
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

// ── Task 2: Forward Returns ────────────────────────────────────────────────────

/// Return statistics for a single horizon window.
#[derive(Debug, Clone)]
pub struct HorizonReturn {
    /// Percentage return over the horizon.
    pub ret: f64,
    /// Maximum upside excursion (%) seen during the horizon.
    pub max_up: f64,
    /// Maximum downside excursion (%) seen during the horizon.
    pub max_down: f64,
}

/// Forward return data across multiple horizons.
#[derive(Debug, Clone)]
pub struct ForwardReturns {
    /// 5-candle horizon.
    pub short: HorizonReturn,
    /// 10-candle horizon.
    pub medium: HorizonReturn,
    /// 20-candle horizon.
    pub long: HorizonReturn,
    /// 40-candle horizon (unlocked after sufficient learning).
    pub extended: Option<HorizonReturn>,
}

/// Compute forward return statistics from `start_idx` over `horizon` candles.
///
/// - `ret`      = percentage change from close[start] to close[start + horizon]
/// - `max_up`   = max of (high[i] - entry) / entry * 100  for i in start+1..=start+horizon
/// - `max_down` = max of (entry - low[i]) / entry * 100   for i in start+1..=start+horizon
///
/// Returns `None` if the slice is too short.
pub fn compute_forward_return(
    closes: &[f64],
    highs: &[f64],
    lows: &[f64],
    start_idx: usize,
    horizon: usize,
) -> Option<HorizonReturn> {
    let end = start_idx + horizon;
    if end >= closes.len() || end >= highs.len() || end >= lows.len() {
        return None;
    }
    let entry = closes[start_idx];
    if entry.abs() < 1e-15 {
        return None;
    }

    let ret = (closes[end] - entry) / entry * 100.0;

    let mut max_up: f64 = 0.0;
    let mut max_down: f64 = 0.0;
    for i in (start_idx + 1)..=end {
        let up = (highs[i] - entry) / entry * 100.0;
        let down = (entry - lows[i]) / entry * 100.0;
        max_up = max_up.max(up);
        max_down = max_down.max(down);
    }

    Some(HorizonReturn {
        ret,
        max_up,
        max_down,
    })
}

// ── Task 3: Observations & PatternLibrary ──────────────────────────────────────

/// A completed observation: temporal vector + measured forward returns.
#[derive(Debug, Clone)]
pub struct CompletedObservation {
    pub vector: [f64; VECTOR_DIM],
    pub symbol: String,
    pub candle_index: usize,
    pub entry_price: f64,
    pub forward_returns: ForwardReturns,
}

/// An observation waiting for enough future candles to compute forward returns.
#[derive(Debug, Clone)]
struct PendingObservation {
    vector: [f64; VECTOR_DIM],
    symbol: String,
    candle_index: usize,
    entry_price: f64,
    max_horizon: usize,
}

/// Library of market pattern observations with KNN prediction.
///
/// Observations flow through a two-stage pipeline:
/// 1. **Pending** — recorded when seen, waiting for future candles.
/// 2. **Completed** — forward returns computed, available for KNN queries.
///
/// Anti-look-ahead: predictions only use completed observations whose forward
/// returns are computed from candles that have already passed.
#[derive(Debug, Clone)]
pub struct PatternLibrary {
    global: VecDeque<CompletedObservation>,
    per_symbol: HashMap<String, VecDeque<CompletedObservation>>,
    pending: VecDeque<PendingObservation>,
    feature_history: HashMap<String, VecDeque<[f64; 7]>>,
    global_cap: usize,
    symbol_cap: usize,
    /// Exponential moving average of prediction accuracy (0..1).
    pub prediction_accuracy_ema: f64,
}

impl PatternLibrary {
    /// Create with explicit capacity limits.
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

    /// Create with default caps: 5000 global, 2000 per-symbol.
    pub fn with_defaults() -> Self {
        Self::new(5000, 2000)
    }

    /// Add a completed observation to global and per-symbol stores (FIFO eviction).
    pub fn add_completed(&mut self, symbol: &str, obs: CompletedObservation) {
        // Global store
        if self.global.len() >= self.global_cap {
            self.global.pop_front();
        }
        self.global.push_back(obs.clone());

        // Per-symbol store
        let sym_deque = self.per_symbol.entry(symbol.to_string()).or_default();
        if sym_deque.len() >= self.symbol_cap {
            sym_deque.pop_front();
        }
        sym_deque.push_back(obs);
    }

    /// Number of completed observations in the global store.
    pub fn completed_global_count(&self) -> usize {
        self.global.len()
    }

    /// Number of completed observations for a specific symbol.
    pub fn completed_symbol_count(&self, symbol: &str) -> usize {
        self.per_symbol.get(symbol).map_or(0, |d| d.len())
    }

    /// Number of pending (not yet resolved) observations.
    pub fn pending_count(&self) -> usize {
        self.pending.len()
    }

    // ── Task 4: Pending resolution ─────────────────────────────────────────

    /// Enqueue a pending observation.
    fn add_pending(&mut self, obs: PendingObservation) {
        self.pending.push_back(obs);
    }

    /// Resolve any pending observations whose full horizon has elapsed.
    ///
    /// An observation at `candle_index` with `max_horizon` is resolvable when
    /// `candle_index + max_horizon <= current_idx` (all needed candles exist).
    fn resolve_pending(&mut self, closes: &[f64], highs: &[f64], lows: &[f64], current_idx: usize) {
        let mut still_pending = VecDeque::new();

        while let Some(p) = self.pending.pop_front() {
            if p.candle_index + p.max_horizon <= current_idx {
                // We have enough data — compute forward returns.
                let short = compute_forward_return(closes, highs, lows, p.candle_index, 5);
                let medium = compute_forward_return(closes, highs, lows, p.candle_index, 10);
                let long = compute_forward_return(closes, highs, lows, p.candle_index, 20);
                let extended = if p.max_horizon >= 40 {
                    compute_forward_return(closes, highs, lows, p.candle_index, 40)
                } else {
                    None
                };

                // All mandatory horizons must succeed.
                if let (Some(s), Some(m), Some(l)) = (short, medium, long) {
                    let obs = CompletedObservation {
                        vector: p.vector,
                        symbol: p.symbol.clone(),
                        candle_index: p.candle_index,
                        entry_price: p.entry_price,
                        forward_returns: ForwardReturns {
                            short: s,
                            medium: m,
                            long: l,
                            extended,
                        },
                    };
                    self.add_completed(&p.symbol, obs);
                }
                // else: data issue — silently drop (shouldn't happen with correct indices)
            } else {
                still_pending.push_back(p);
            }
        }

        self.pending = still_pending;
    }

    // ── Task 5: KNN Prediction ─────────────────────────────────────────────

    /// Query the library for a KNN-based market prediction.
    ///
    /// Returns `None` if fewer than 10 neighbors or average distance > 2.0.
    pub fn predict(
        &self,
        query: &[f64; VECTOR_DIM],
        symbol: &str,
        k: usize,
    ) -> Option<MarketPrediction> {
        // Choose per-symbol if rich enough, else global.
        let source = if self.completed_symbol_count(symbol) >= 200 {
            self.per_symbol.get(symbol).unwrap().as_slices()
        } else {
            self.global.as_slices()
        };

        let lib_size = source.0.len() + source.1.len();
        if lib_size < 10 {
            return None;
        }

        // Adaptive K.
        let adaptive_k = (k.min((lib_size as f64).sqrt() as usize)).max(10);

        // Compute all distances.
        let mut dists: Vec<(f64, &CompletedObservation)> = Vec::with_capacity(lib_size);
        for obs in source.0.iter().chain(source.1.iter()) {
            let d = euclidean_distance(query, &obs.vector);
            dists.push((d, obs));
        }
        dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));

        let neighbors = &dists[..adaptive_k.min(dists.len())];
        let neighbor_count = neighbors.len();
        if neighbor_count < 10 {
            return None;
        }

        let avg_distance: f64 =
            neighbors.iter().map(|(d, _)| d).sum::<f64>() / neighbor_count as f64;
        if avg_distance > 2.0 {
            return None;
        }

        // Inverse-distance weighting.
        let weights: Vec<f64> = neighbors.iter().map(|(d, _)| 1.0 / (d + 1e-8)).collect();
        let w_sum: f64 = weights.iter().sum();

        let expected_return = neighbors
            .iter()
            .zip(&weights)
            .map(|((_, obs), w)| obs.forward_returns.medium.ret * w)
            .sum::<f64>()
            / w_sum;

        let directional_prob = neighbors
            .iter()
            .zip(&weights)
            .map(|((_, obs), w)| {
                if obs.forward_returns.medium.ret > 0.0 {
                    *w
                } else {
                    0.0
                }
            })
            .sum::<f64>()
            / w_sum;

        let avg_max_up = neighbors
            .iter()
            .zip(&weights)
            .map(|((_, obs), w)| obs.forward_returns.medium.max_up * w)
            .sum::<f64>()
            / w_sum;

        let avg_max_down = (neighbors
            .iter()
            .zip(&weights)
            .map(|((_, obs), w)| obs.forward_returns.medium.max_down * w)
            .sum::<f64>()
            / w_sum)
            .max(0.01);

        Some(MarketPrediction {
            expected_return,
            directional_prob,
            avg_max_up,
            avg_max_down,
            neighbor_count,
            avg_distance,
        })
    }

    // ── Task 7: on_candle integration ──────────────────────────────────────

    /// Process a new candle: store features, resolve pending, create new pending.
    pub fn on_candle(
        &mut self,
        symbol: &str,
        features: &[f64; 7],
        candle_index: usize,
        candle_data: &CandleSlices<'_>,
    ) {
        // 1. Store feature snapshot (keep last 3).
        {
            let hist = self.feature_history.entry(symbol.to_string()).or_default();
            hist.push_back(*features);
            if hist.len() > 3 {
                hist.pop_front();
            }
        }

        // 2. Resolve any matured pending observations.
        self.resolve_pending(
            candle_data.closes,
            candle_data.highs,
            candle_data.lows,
            candle_index,
        );

        // 3. If we have 3 snapshots, build temporal vector and add pending.
        let hist = self.feature_history.get(symbol);
        if let Some(h) = hist {
            if h.len() == 3 {
                let vector = build_temporal_vector(&h[0], &h[1], &h[2]);

                let max_horizon = if self.extended_horizon_unlocked() {
                    40
                } else {
                    20
                };

                self.add_pending(PendingObservation {
                    vector,
                    symbol: symbol.to_string(),
                    candle_index,
                    entry_price: candle_data.closes[candle_index],
                    max_horizon,
                });
            }
        }
    }

    /// Whether the 40-candle extended horizon is unlocked.
    ///
    /// Requires >= 500 completed observations and accuracy EMA > 0.55.
    pub fn extended_horizon_unlocked(&self) -> bool {
        self.completed_global_count() > 500 && self.prediction_accuracy_ema > 0.55
    }

    /// Build a query vector for KNN from stored feature history + current features.
    ///
    /// Returns `None` if fewer than 2 historical snapshots are available for the
    /// symbol (need t-2 and t-1 from history, t-0 from `current_features`).
    pub fn build_query_vector(
        &self,
        symbol: &str,
        current_features: &[f64; 7],
    ) -> Option<[f64; VECTOR_DIM]> {
        let hist = self.feature_history.get(symbol)?;
        if hist.len() < 2 {
            return None;
        }
        let len = hist.len();
        let t_minus_2 = &hist[len - 2];
        let t_minus_1 = &hist[len - 1];
        Some(build_temporal_vector(
            t_minus_2,
            t_minus_1,
            current_features,
        ))
    }
}

// ── Prediction result ──────────────────────────────────────────────────────────

/// KNN prediction output.
#[derive(Debug, Clone)]
pub struct MarketPrediction {
    /// Weighted mean of medium-horizon returns (%).
    pub expected_return: f64,
    /// Weighted fraction of neighbors with positive medium return.
    pub directional_prob: f64,
    /// Weighted mean of medium-horizon max upside excursion (%).
    pub avg_max_up: f64,
    /// Weighted mean of medium-horizon max downside excursion (%), clamped >= 0.01.
    pub avg_max_down: f64,
    /// Number of neighbors used.
    pub neighbor_count: usize,
    /// Average distance to neighbors.
    pub avg_distance: f64,
}

// ── Private helpers ────────────────────────────────────────────────────────────

fn euclidean_distance(a: &[f64; VECTOR_DIM], b: &[f64; VECTOR_DIM]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

// ── Tests ──────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── Task 1: Temporal vector tests ──────────────────────────────────────

    #[test]
    fn test_build_temporal_vector_dimensions() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        let c = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0];
        let v = build_temporal_vector(&a, &b, &c);
        assert_eq!(v.len(), VECTOR_DIM);
        assert_eq!(v.len(), 21);
    }

    #[test]
    fn test_build_temporal_vector_values() {
        let a = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let b = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        let c = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0];
        let v = build_temporal_vector(&a, &b, &c);

        // Position 0: first element of t-2
        assert!((v[0] - 1.0).abs() < 1e-10);
        // Position 6: last element of t-2
        assert!((v[6] - 7.0).abs() < 1e-10);
        // Position 7: first element of t-1
        assert!((v[7] - 8.0).abs() < 1e-10);
        // Position 14: first element of t-0
        assert!((v[14] - 15.0).abs() < 1e-10);
        // Position 20: last element of t-0
        assert!((v[20] - 21.0).abs() < 1e-10);
    }

    // ── Task 2: Forward return tests ───────────────────────────────────────

    #[test]
    fn test_compute_forward_return_known_series() {
        // Price series: [100, 102, 99, 104, 101, 103]
        // Entry at index 0 (price 100), horizon 5 → exit at index 5 (price 103)
        // ret = (103 - 100) / 100 * 100 = 3%
        // Highs track slightly above closes for realism, lows slightly below.
        let closes = [100.0, 102.0, 99.0, 104.0, 101.0, 103.0];
        // Highs: highest high is at index 3 → 105 → max_up = (105-100)/100*100 = 5%
        let highs = [100.0, 103.0, 100.0, 105.0, 102.0, 104.0];
        // Lows: lowest low is at index 2 → 98 → max_down = (100-98)/100*100 = 2%
        let lows = [100.0, 101.0, 98.0, 103.0, 100.0, 102.0];

        let hr = compute_forward_return(&closes, &highs, &lows, 0, 5).unwrap();
        assert!((hr.ret - 3.0).abs() < 1e-10, "ret={}", hr.ret);
        assert!((hr.max_up - 5.0).abs() < 1e-10, "max_up={}", hr.max_up);
        assert!(
            (hr.max_down - 2.0).abs() < 1e-10,
            "max_down={}",
            hr.max_down
        );
    }

    #[test]
    fn test_compute_forward_return_insufficient_data() {
        let closes = [100.0, 102.0];
        let highs = [101.0, 103.0];
        let lows = [99.0, 101.0];
        assert!(compute_forward_return(&closes, &highs, &lows, 0, 5).is_none());
    }

    // ── Task 3: PatternLibrary core tests ──────────────────────────────────

    fn make_dummy_observation(idx: usize, symbol: &str) -> CompletedObservation {
        CompletedObservation {
            vector: [idx as f64; VECTOR_DIM],
            symbol: symbol.to_string(),
            candle_index: idx,
            entry_price: 100.0 + idx as f64,
            forward_returns: ForwardReturns {
                short: HorizonReturn {
                    ret: 1.0,
                    max_up: 2.0,
                    max_down: 0.5,
                },
                medium: HorizonReturn {
                    ret: 2.0,
                    max_up: 3.0,
                    max_down: 1.0,
                },
                long: HorizonReturn {
                    ret: 3.0,
                    max_up: 5.0,
                    max_down: 2.0,
                },
                extended: None,
            },
        }
    }

    #[test]
    fn test_library_basic_add_and_count() {
        let mut lib = PatternLibrary::with_defaults();
        assert_eq!(lib.completed_global_count(), 0);
        assert_eq!(lib.completed_symbol_count("BTC"), 0);

        lib.add_completed("BTC", make_dummy_observation(0, "BTC"));
        assert_eq!(lib.completed_global_count(), 1);
        assert_eq!(lib.completed_symbol_count("BTC"), 1);
        assert_eq!(lib.completed_symbol_count("ETH"), 0);
    }

    #[test]
    fn test_library_global_fifo_eviction() {
        let mut lib = PatternLibrary::new(5, 100);
        for i in 0..8 {
            lib.add_completed("BTC", make_dummy_observation(i, "BTC"));
        }
        assert_eq!(lib.completed_global_count(), 5);
        // Oldest should have been evicted; front should be index 3.
        assert!((lib.global[0].vector[0] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_library_symbol_fifo_eviction() {
        let mut lib = PatternLibrary::new(100, 3);
        for i in 0..6 {
            lib.add_completed("ETH", make_dummy_observation(i, "ETH"));
        }
        assert_eq!(lib.completed_symbol_count("ETH"), 3);
        // Oldest should be index 3.
        let sym = lib.per_symbol.get("ETH").unwrap();
        assert!((sym[0].vector[0] - 3.0).abs() < 1e-10);
    }

    // ── Task 4: Pending resolution tests ───────────────────────────────────

    #[test]
    fn test_pending_not_in_completed() {
        let mut lib = PatternLibrary::with_defaults();
        lib.add_pending(PendingObservation {
            vector: [0.0; VECTOR_DIM],
            symbol: "BTC".to_string(),
            candle_index: 5,
            entry_price: 100.0,
            max_horizon: 20,
        });
        assert_eq!(lib.pending_count(), 1);
        assert_eq!(lib.completed_global_count(), 0);
    }

    #[test]
    fn test_pending_predict_returns_none() {
        let mut lib = PatternLibrary::with_defaults();
        lib.add_pending(PendingObservation {
            vector: [0.0; VECTOR_DIM],
            symbol: "BTC".to_string(),
            candle_index: 5,
            entry_price: 100.0,
            max_horizon: 20,
        });
        let query = [0.0; VECTOR_DIM];
        assert!(lib.predict(&query, "BTC", 20).is_none());
    }

    #[test]
    fn test_pending_resolves_after_enough_candles() {
        let mut lib = PatternLibrary::with_defaults();

        // Build price data: 30 candles, linear rise from 100 to 129.
        let n = 30;
        let closes: Vec<f64> = (0..n).map(|i| 100.0 + i as f64).collect();
        let highs: Vec<f64> = (0..n).map(|i| 101.0 + i as f64).collect();
        let lows: Vec<f64> = (0..n).map(|i| 99.0 + i as f64).collect();

        // Pending at candle 5, max_horizon=20 → resolves at candle 25.
        lib.add_pending(PendingObservation {
            vector: [0.5; VECTOR_DIM],
            symbol: "BTC".to_string(),
            candle_index: 5,
            entry_price: closes[5],
            max_horizon: 20,
        });

        // Not yet resolved at candle 24.
        lib.resolve_pending(&closes, &highs, &lows, 24);
        assert_eq!(lib.pending_count(), 1);
        assert_eq!(lib.completed_global_count(), 0);

        // Resolved at candle 25.
        lib.resolve_pending(&closes, &highs, &lows, 25);
        assert_eq!(lib.pending_count(), 0);
        assert_eq!(lib.completed_global_count(), 1);
    }

    // ── Task 5: KNN Prediction tests ───────────────────────────────────────

    #[test]
    fn test_predict_too_few_observations() {
        let mut lib = PatternLibrary::with_defaults();
        for i in 0..5 {
            lib.add_completed("BTC", make_dummy_observation(i, "BTC"));
        }
        let query = [0.0; VECTOR_DIM];
        assert!(lib.predict(&query, "BTC", 20).is_none());
    }

    #[test]
    fn test_predict_three_clusters() {
        let mut lib = PatternLibrary::with_defaults();

        // Cluster A near 0.0 → positive returns
        for i in 0..10 {
            let mut obs = make_dummy_observation(i, "BTC");
            let offset = i as f64 * 0.01;
            obs.vector = [0.0 + offset; VECTOR_DIM];
            obs.forward_returns.medium.ret = 5.0;
            lib.add_completed("BTC", obs);
        }
        // Cluster B near 5.0 → negative returns
        for i in 10..20 {
            let mut obs = make_dummy_observation(i, "BTC");
            let offset = (i - 10) as f64 * 0.01;
            obs.vector = [5.0 + offset; VECTOR_DIM];
            obs.forward_returns.medium.ret = -3.0;
            lib.add_completed("BTC", obs);
        }
        // Cluster C near 10.0 → neutral
        for i in 20..30 {
            let mut obs = make_dummy_observation(i, "BTC");
            let offset = (i - 20) as f64 * 0.01;
            obs.vector = [10.0 + offset; VECTOR_DIM];
            obs.forward_returns.medium.ret = 0.0;
            lib.add_completed("BTC", obs);
        }

        // Query near cluster B (5.0).
        let query = [5.0; VECTOR_DIM];
        let pred = lib.predict(&query, "BTC", 20).unwrap();
        assert!(
            pred.expected_return < 0.0,
            "expected negative return near cluster B, got {}",
            pred.expected_return
        );
        assert!(
            pred.directional_prob < 0.5,
            "expected low directional prob near cluster B, got {}",
            pred.directional_prob
        );
    }

    #[test]
    fn test_predict_distance_weighting() {
        let mut lib = PatternLibrary::with_defaults();

        // One very close neighbor: ret=10
        let mut close_obs = make_dummy_observation(0, "BTC");
        close_obs.vector = [1.0; VECTOR_DIM];
        close_obs.forward_returns.medium.ret = 10.0;
        lib.add_completed("BTC", close_obs);

        // Many distant neighbors: ret=-5 (still within distance threshold)
        for i in 1..15 {
            let mut far_obs = make_dummy_observation(i, "BTC");
            far_obs.vector = [1.3; VECTOR_DIM]; // farther away
            far_obs.forward_returns.medium.ret = -5.0;
            lib.add_completed("BTC", far_obs);
        }

        let query = [1.0; VECTOR_DIM]; // right on top of the close neighbor
        let pred = lib.predict(&query, "BTC", 20).unwrap();
        // Close neighbor has huge weight, so expected return should be positive.
        assert!(
            pred.expected_return > 0.0,
            "close neighbor should dominate, got {}",
            pred.expected_return
        );
    }

    #[test]
    fn test_predict_high_distance_returns_none() {
        let mut lib = PatternLibrary::with_defaults();
        // Observations far from query.
        for i in 0..20 {
            let mut obs = make_dummy_observation(i, "BTC");
            obs.vector = [100.0; VECTOR_DIM]; // very far from origin
            lib.add_completed("BTC", obs);
        }
        let query = [0.0; VECTOR_DIM];
        // avg_distance will be sqrt(21 * 100^2) = ~458, way above 2.0
        assert!(lib.predict(&query, "BTC", 20).is_none());
    }

    #[test]
    fn test_predict_avg_max_down_clamped() {
        let mut lib = PatternLibrary::with_defaults();
        for i in 0..15 {
            let mut obs = make_dummy_observation(i, "BTC");
            obs.vector = [0.0; VECTOR_DIM];
            obs.forward_returns.medium.max_down = 0.0; // zero drawdown
            lib.add_completed("BTC", obs);
        }
        let query = [0.0; VECTOR_DIM];
        let pred = lib.predict(&query, "BTC", 20).unwrap();
        assert!(
            pred.avg_max_down >= 0.01,
            "avg_max_down should be clamped to >= 0.01, got {}",
            pred.avg_max_down
        );
    }

    // ── Task 7: on_candle & integration tests ──────────────────────────────

    #[test]
    fn test_on_candle_lifecycle() {
        let mut lib = PatternLibrary::with_defaults();

        let n = 30;
        let closes: Vec<f64> = (0..n).map(|i| 100.0 + i as f64 * 0.5).collect();
        let highs: Vec<f64> = (0..n).map(|i| 101.0 + i as f64 * 0.5).collect();
        let lows: Vec<f64> = (0..n).map(|i| 99.0 + i as f64 * 0.5).collect();
        let slices = CandleSlices {
            closes: &closes,
            highs: &highs,
            lows: &lows,
        };

        for i in 0..n {
            let features = [(i as f64 / n as f64), 0.5, 0.3, 0.6, 0.1, 0.5, 0.2];
            lib.on_candle("BTC", &features, i, &slices);
        }

        // First pending created at candle 2 (when 3 snapshots available).
        // With max_horizon=20, it resolves at candle 22 (2 + 20 = 22).
        // Candles 2..7 should have pending resolved (at candles 22..27).
        assert!(
            lib.completed_global_count() > 0,
            "some obs should be completed"
        );
        assert!(lib.pending_count() > 0, "some obs should still be pending");
    }

    #[test]
    fn test_extended_horizon_locked() {
        let lib = PatternLibrary::with_defaults();
        assert!(
            !lib.extended_horizon_unlocked(),
            "should be locked with 0 obs"
        );
    }

    #[test]
    fn test_extended_horizon_unlocked_conditions() {
        let mut lib = PatternLibrary::with_defaults();
        // Add 501 observations.
        for i in 0..501 {
            lib.add_completed("BTC", make_dummy_observation(i, "BTC"));
        }
        // Accuracy too low.
        lib.prediction_accuracy_ema = 0.50;
        assert!(!lib.extended_horizon_unlocked());

        // Accuracy high enough.
        lib.prediction_accuracy_ema = 0.60;
        assert!(lib.extended_horizon_unlocked());
    }

    #[test]
    fn test_build_query_vector_insufficient_history() {
        let lib = PatternLibrary::with_defaults();
        let features = [0.1; 7];
        assert!(lib.build_query_vector("BTC", &features).is_none());
    }

    #[test]
    fn test_build_query_vector_with_history() {
        let mut lib = PatternLibrary::with_defaults();

        // Manually populate feature_history with 2 snapshots.
        let snap1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0];
        let snap2 = [8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0];
        let current = [15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0];

        let hist = lib
            .feature_history
            .entry("BTC".to_string())
            .or_insert_with(VecDeque::new);
        hist.push_back(snap1);
        hist.push_back(snap2);

        let v = lib.build_query_vector("BTC", &current).unwrap();
        assert_eq!(v.len(), VECTOR_DIM);
        assert!((v[0] - 1.0).abs() < 1e-10);
        assert!((v[7] - 8.0).abs() < 1e-10);
        assert!((v[14] - 15.0).abs() < 1e-10);
        assert!((v[20] - 21.0).abs() < 1e-10);
    }

    #[test]
    fn test_euclidean_distance() {
        let a = [0.0; VECTOR_DIM];
        let mut b = [0.0; VECTOR_DIM];
        b[0] = 3.0;
        b[1] = 4.0;
        // Distance = sqrt(9 + 16) = 5.0
        let d = euclidean_distance(&a, &b);
        assert!((d - 5.0).abs() < 1e-10, "distance={}", d);
    }

    #[test]
    fn test_euclidean_distance_identical() {
        let a = [1.5; VECTOR_DIM];
        let d = euclidean_distance(&a, &a);
        assert!(d.abs() < 1e-10);
    }

    #[test]
    fn test_predict_uses_per_symbol_when_rich() {
        let mut lib = PatternLibrary::with_defaults();

        // Add 200 BTC observations (triggers per-symbol usage).
        for i in 0..200 {
            let mut obs = make_dummy_observation(i, "BTC");
            obs.vector = [0.0; VECTOR_DIM];
            obs.forward_returns.medium.ret = 2.0;
            lib.add_completed("BTC", obs);
        }

        // Add global observations with very different returns.
        for i in 200..220 {
            let mut obs = make_dummy_observation(i, "ETH");
            obs.vector = [0.0; VECTOR_DIM];
            obs.forward_returns.medium.ret = -10.0;
            lib.add_completed("ETH", obs);
        }

        let query = [0.0; VECTOR_DIM];
        let pred = lib.predict(&query, "BTC", 20).unwrap();
        // Should use per-symbol (BTC) data, so returns should be ~2.0.
        assert!(
            pred.expected_return > 1.0,
            "should use per-symbol data, got {}",
            pred.expected_return
        );
    }
}
