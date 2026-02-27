use super::features::MarketFeatures;
use crate::domain::MarketContext;
use serde::{Deserialize, Serialize};

/// Adaptive pattern discretizer using median-split binning.
/// 4 key features × 2 bins = 16 total patterns.
/// Small pattern space ensures each pattern gets enough observations to learn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PatternDiscretizer {
    binners: Vec<MedianBinner>,
}

/// Median-split binner: below median = 0, above median = 1.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MedianBinner {
    recent: Vec<f64>,
    median: f64,
    max_samples: usize,
}

impl MedianBinner {
    fn new(initial_median: f64) -> Self {
        Self {
            recent: Vec::new(),
            median: initial_median,
            max_samples: 3000,
        }
    }

    fn observe(&mut self, val: f64) {
        self.recent.push(val);
        if self.recent.len() > self.max_samples {
            let half = self.max_samples / 2;
            self.recent.drain(0..half);
        }
        if self.recent.len() >= 20 {
            self.update_median();
        }
    }

    fn update_median(&mut self) {
        let mut sorted = self.recent.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        self.median = sorted[sorted.len() / 2];
    }

    fn bin(&self, val: f64) -> u8 {
        if val <= self.median {
            0
        } else {
            1
        }
    }

    fn sample_count(&self) -> usize {
        self.recent.len()
    }
}

impl PatternDiscretizer {
    pub fn new() -> Self {
        // 4 key features: trend, momentum, volatility, volume
        let binners = vec![
            MedianBinner::new(0.0), // trend: negative/positive
            MedianBinner::new(0.0), // momentum: weak/strong
            MedianBinner::new(0.5), // volatility: low/high
            MedianBinner::new(0.5), // volume: low/high
        ];
        Self { binners }
    }

    /// Feed features to learn median boundaries.
    pub fn observe(&mut self, features: &MarketFeatures) {
        let vals = features.key_features();
        for (binner, &val) in self.binners.iter_mut().zip(vals.iter()) {
            binner.observe(val);
        }
    }

    /// Discretize features into one of 16 patterns.
    pub fn discretize(&self, features: &MarketFeatures) -> MarketContext {
        let vals = features.key_features();
        let bins: Vec<u8> = self
            .binners
            .iter()
            .zip(vals.iter())
            .map(|(binner, &val)| binner.bin(val))
            .collect();

        // Encode as compact pattern: "TMVV" where T=trend, M=momentum, V=vol, V=volume
        MarketContext {
            volatility_tier: format!("{}{}", bins[0], bins[1]),
            trend_regime: format!("{}{}", bins[2], bins[3]),
        }
    }

    /// Pattern key as single string.
    pub fn pattern_key(ctx: &MarketContext) -> String {
        format!("{}_{}", ctx.volatility_tier, ctx.trend_regime)
    }

    pub fn sample_count(&self) -> usize {
        self.binners.first().map(|b| b.sample_count()).unwrap_or(0)
    }

    pub fn is_warmed_up(&self) -> bool {
        self.sample_count() >= 20
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_features(trend: f64, momentum: f64) -> MarketFeatures {
        MarketFeatures {
            trend,
            momentum,
            volatility: 0.5,
            volume: 0.5,
            candle_character: 0.0,
            bb_position: 0.5,
            short_momentum: 0.0,
            atr: 100.0,
            price: 50000.0,
        }
    }

    #[test]
    fn test_discretizer_produces_valid_context() {
        let disc = PatternDiscretizer::new();
        let ctx = disc.discretize(&make_features(0.5, 0.3));
        assert_eq!(ctx.volatility_tier.len(), 2);
        assert_eq!(ctx.trend_regime.len(), 2);
    }

    #[test]
    fn test_discretizer_adapts_boundaries() {
        let mut disc = PatternDiscretizer::new();
        for i in 0..100 {
            disc.observe(&make_features(0.5 + i as f64 * 0.005, 0.3));
        }
        assert!(disc.is_warmed_up());

        let ctx_low = disc.discretize(&make_features(0.3, 0.0));
        let ctx_high = disc.discretize(&make_features(0.9, 0.0));
        assert_ne!(ctx_low.volatility_tier, ctx_high.volatility_tier);
    }

    #[test]
    fn test_discretizer_deterministic() {
        let disc = PatternDiscretizer::new();
        let f = make_features(0.3, -0.2);
        assert_eq!(disc.discretize(&f), disc.discretize(&f));
    }

    #[test]
    fn test_pattern_key() {
        let ctx = MarketContext {
            volatility_tier: "01".into(),
            trend_regime: "10".into(),
        };
        assert_eq!(PatternDiscretizer::pattern_key(&ctx), "01_10");
    }

    #[test]
    fn test_16_patterns_max() {
        let disc = PatternDiscretizer::new();
        let mut patterns = std::collections::HashSet::new();
        // Try extreme combinations
        for t in [-1.0, 1.0] {
            for m in [-1.0, 1.0] {
                for v in [0.0, 1.0] {
                    for vol in [0.0, 1.0] {
                        let f = MarketFeatures {
                            trend: t,
                            momentum: m,
                            volatility: v,
                            volume: vol,
                            candle_character: 0.0,
                            bb_position: 0.5,
                            short_momentum: 0.0,
                            atr: 100.0,
                            price: 50000.0,
                        };
                        let ctx = disc.discretize(&f);
                        patterns.insert(PatternDiscretizer::pattern_key(&ctx));
                    }
                }
            }
        }
        assert!(patterns.len() <= 16);
    }

    #[test]
    fn test_not_warmed_up_initially() {
        let disc = PatternDiscretizer::new();
        assert!(!disc.is_warmed_up());
        assert_eq!(disc.sample_count(), 0);
    }

    #[test]
    fn test_warmup_threshold() {
        let mut disc = PatternDiscretizer::new();
        for i in 0..19 {
            disc.observe(&make_features(0.5 + i as f64 * 0.01, 0.3));
        }
        assert!(!disc.is_warmed_up());
        disc.observe(&make_features(0.7, 0.3));
        assert!(disc.is_warmed_up());
    }

    #[test]
    fn test_bin_at_median() {
        let mut disc = PatternDiscretizer::new();
        // All features = 0.5, so median should stabilize around 0.5
        for _ in 0..50 {
            disc.observe(&make_features(0.5, 0.5));
        }
        // Value at median should go to bin 0 (<=)
        let ctx = disc.discretize(&make_features(0.5, 0.5));
        let key = PatternDiscretizer::pattern_key(&ctx);
        // All features at their median -> all bins = 0 -> "00_00"
        assert_eq!(key, "00_00");
    }

    #[test]
    fn test_binner_eviction() {
        let mut disc = PatternDiscretizer::new();
        // Feed 4000 observations (above max_samples = 3000)
        for i in 0..4000 {
            disc.observe(&make_features(i as f64 * 0.001, 0.0));
        }
        // Should not exceed max_samples
        assert!(disc.sample_count() <= 3000);
    }

    #[test]
    fn test_observe_updates_medians() {
        let mut disc = PatternDiscretizer::new();
        // Feed low values
        for _ in 0..50 {
            disc.observe(&make_features(-1.0, -1.0));
        }
        let ctx_high = disc.discretize(&make_features(0.0, 0.0));
        // 0.0 is above median of -1.0 -> bins should be 1
        let key = PatternDiscretizer::pattern_key(&ctx_high);
        assert!(key.contains('1'), "Above-median values should produce bin 1: {}", key);
    }

    #[test]
    fn test_discretize_different_features() {
        let disc = PatternDiscretizer::new();
        let ctx1 = disc.discretize(&MarketFeatures {
            trend: -1.0,
            momentum: -1.0,
            volatility: 0.0,
            volume: 0.0,
            candle_character: 0.0,
            bb_position: 0.5,
            short_momentum: 0.0,
            atr: 100.0,
            price: 50000.0,
        });
        let ctx2 = disc.discretize(&MarketFeatures {
            trend: 1.0,
            momentum: 1.0,
            volatility: 1.0,
            volume: 1.0,
            candle_character: 0.0,
            bb_position: 0.5,
            short_momentum: 0.0,
            atr: 100.0,
            price: 50000.0,
        });
        assert_ne!(
            PatternDiscretizer::pattern_key(&ctx1),
            PatternDiscretizer::pattern_key(&ctx2),
            "Extreme opposite features should produce different patterns"
        );
    }

    #[test]
    fn test_pattern_key_format() {
        let ctx = MarketContext {
            volatility_tier: "11".into(),
            trend_regime: "00".into(),
        };
        assert_eq!(PatternDiscretizer::pattern_key(&ctx), "11_00");
    }

    #[test]
    fn test_sample_count_grows() {
        let mut disc = PatternDiscretizer::new();
        assert_eq!(disc.sample_count(), 0);
        disc.observe(&make_features(0.5, 0.3));
        assert_eq!(disc.sample_count(), 1);
        disc.observe(&make_features(0.6, 0.4));
        assert_eq!(disc.sample_count(), 2);
    }

    #[test]
    fn test_context_fields_are_two_chars() {
        let mut disc = PatternDiscretizer::new();
        for i in 0..30 {
            disc.observe(&make_features(i as f64 * 0.1, i as f64 * 0.05));
        }
        let ctx = disc.discretize(&make_features(0.5, 0.3));
        assert_eq!(ctx.volatility_tier.len(), 2);
        assert_eq!(ctx.trend_regime.len(), 2);
        // Each character should be '0' or '1'
        for c in ctx.volatility_tier.chars() {
            assert!(c == '0' || c == '1');
        }
        for c in ctx.trend_regime.chars() {
            assert!(c == '0' || c == '1');
        }
    }
}
