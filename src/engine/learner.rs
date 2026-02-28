use super::adaptive::AdaptiveParams;
use super::pattern_library::PatternLibrary;
use super::thompson::ThompsonEngine;
use super::transfer::{PlateauAction, PlateauDetector, RegretTracker};
use crate::domain::*;
use crate::strategy::features::MarketFeatures;
use crate::strategy::patterns::PatternDiscretizer;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// The three possible actions the system can take.
const ACTIONS: [&str; 3] = ["long", "short", "hold"];

/// Fully adaptive learning engine.
/// No hardcoded trading rules. Learns everything from market patterns.
///
/// Flow: MarketState -> features -> adaptive discretization -> pattern
///       -> Thompson Sampling selects action -> adaptive Kelly sizing -> adaptive SL/TP
#[derive(Clone, Serialize, Deserialize)]
pub struct LearningEngine {
    pub thompson: ThompsonEngine,
    pub discretizer: PatternDiscretizer,
    pub excursions: ExcursionTracker,
    pub regret: HashMap<String, RegretTracker>,
    pub plateau: HashMap<String, PlateauDetector>,
    pub symbols: Vec<String>,
    pub config: LearnerConfig,
    /// Adaptive parameters — learns from trade outcomes
    pub adaptive: AdaptiveParams,
    /// Last pattern used for a decision (for recording outcomes)
    #[serde(skip)]
    pub last_pattern: Option<MarketContext>,
    /// Last KNN directional probability (for reward blending)
    #[serde(skip)]
    pub last_prediction: Option<f64>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LearnerConfig {
    pub decay_factor: f64,
    pub plateau_window: usize,
    pub plateau_threshold: f64,
    pub transfer_after_n: u64,
    /// Minimum edge over 'hold' to place a trade
    pub min_edge: f64,
}

impl Default for LearnerConfig {
    fn default() -> Self {
        Self {
            decay_factor: 0.995,
            plateau_window: 5,
            plateau_threshold: 0.005,
            transfer_after_n: 150,
            min_edge: 0.02,
        }
    }
}

/// Tracks trade excursions to set adaptive SL/TP.
/// Learns optimal stop loss and take profit from observed trade behavior.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExcursionTracker {
    favorable: HashMap<String, DecayingStat>,
    adverse: HashMap<String, DecayingStat>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct DecayingStat {
    mean: f64,
    count: u64,
    decay: f64,
}

impl DecayingStat {
    fn new(initial: f64, decay: f64) -> Self {
        Self {
            mean: initial,
            count: 0,
            decay,
        }
    }

    fn update(&mut self, val: f64) {
        self.count += 1;
        if self.count == 1 {
            self.mean = val;
        } else {
            self.mean = self.mean * self.decay + val * (1.0 - self.decay);
        }
    }

    fn value(&self) -> f64 {
        self.mean
    }
}

impl ExcursionTracker {
    pub fn new() -> Self {
        Self {
            favorable: HashMap::new(),
            adverse: HashMap::new(),
        }
    }

    pub fn record(
        &mut self,
        pattern_key: &str,
        side: &str,
        favorable_atr: f64,
        adverse_atr: f64,
        won: bool,
    ) {
        // Record per side+pattern for side-specific learning
        let key = format!("{}_{}", pattern_key, side);
        self.adverse
            .entry(key.clone())
            .or_insert_with(|| DecayingStat::new(2.5, 0.95))
            .update(adverse_atr.max(0.5));

        if won {
            self.favorable
                .entry(key)
                .or_insert_with(|| DecayingStat::new(4.0, 0.95))
                .update(favorable_atr.max(1.0));
        }

        // Also record pattern-only (fallback for cold-start)
        self.adverse
            .entry(pattern_key.to_string())
            .or_insert_with(|| DecayingStat::new(2.5, 0.95))
            .update(adverse_atr.max(0.5));

        if won {
            self.favorable
                .entry(pattern_key.to_string())
                .or_insert_with(|| DecayingStat::new(4.0, 0.95))
                .update(favorable_atr.max(1.0));
        }
    }

    /// Adaptive stop loss in ATR multiples. Tries side-specific first, falls back to pattern.
    pub fn get_sl_atr(&self, pattern_key: &str, side: &str) -> f64 {
        let side_key = format!("{}_{}", pattern_key, side);
        self.adverse
            .get(&side_key)
            .or_else(|| self.adverse.get(pattern_key))
            .map(|s| s.value().clamp(1.0, 5.0))
            .unwrap_or(2.5)
    }

    /// Adaptive take profit in ATR multiples. Tries side-specific first, falls back to pattern.
    pub fn get_tp_atr(&self, pattern_key: &str, side: &str) -> f64 {
        let side_key = format!("{}_{}", pattern_key, side);
        self.favorable
            .get(&side_key)
            .or_else(|| self.favorable.get(pattern_key))
            .map(|s| s.value().clamp(1.5, 8.0))
            .unwrap_or(4.0)
    }
}

impl LearningEngine {
    pub fn new(symbols: Vec<String>, config: LearnerConfig) -> Self {
        let action_names: Vec<String> = ACTIONS.iter().map(|s| s.to_string()).collect();
        let thompson = ThompsonEngine::new(action_names, config.decay_factor);

        let mut regret = HashMap::new();
        let mut plateau = HashMap::new();
        for sym in &symbols {
            regret.insert(sym.clone(), RegretTracker::default());
            plateau.insert(
                sym.clone(),
                PlateauDetector::new(config.plateau_window, config.plateau_threshold),
            );
        }

        Self {
            thompson,
            discretizer: PatternDiscretizer::new(),
            excursions: ExcursionTracker::new(),
            regret,
            plateau,
            symbols,
            config,
            adaptive: AdaptiveParams::new(),
            last_pattern: None,
            last_prediction: None,
        }
    }

    /// Core adaptive decision: features -> pattern -> Thompson selects action.
    /// Uses global pooling — all symbols contribute to same pattern learning.
    /// KNN predictions from the pattern library inform Thompson priors and sizing.
    pub fn decide(
        &mut self,
        state: &MarketState,
        rng: &mut impl Rng,
        library: &PatternLibrary,
    ) -> TradingDecision {
        let features = match MarketFeatures::extract(state) {
            Some(f) => f,
            None => return self.hold_decision(),
        };

        // Feed features to adaptive discretizer (learns boundaries)
        self.discretizer.observe(&features);
        let pattern = self.discretizer.discretize(&features);
        let pattern_key = PatternDiscretizer::pattern_key(&pattern);

        // Build KNN prediction from pattern library
        let prediction = library
            .build_query_vector(&state.symbol.0, &features.as_array())
            .and_then(|query| library.predict(&query, &state.symbol.0, 20));

        // Global pooling: all symbols share Thompson state for pattern learning.
        // Per-symbol pools tested and degraded performance (~230 trades/symbol too sparse).
        let pool_key = "_global";

        // Thompson Sampling selects action — use KNN-boosted priors when available
        // Direct KNN override tested and degraded performance; Thompson exploration is valuable.
        let exploration = self.adaptive.exploration_coeff();
        let action = if let Some(ref pred) = prediction {
            if pred.directional_prob > 0.6 || pred.directional_prob < 0.4 {
                let confidence_scale = (1.0 / pred.avg_distance).clamp(0.5, 3.0);
                self.thompson.select_arm_with_knn_prior(
                    pool_key,
                    &pattern,
                    rng,
                    exploration,
                    pred.directional_prob,
                    confidence_scale,
                )
            } else {
                self.thompson
                    .select_arm_with_exploration(pool_key, &pattern, rng, exploration)
            }
        } else {
            self.thompson
                .select_arm_with_exploration(pool_key, &pattern, rng, exploration)
        };

        // Store pattern for outcome recording
        self.last_pattern = Some(pattern.clone());

        if action.0 == "hold" {
            return self.hold_decision();
        }


        // Adaptive position sizing via Kelly criterion
        // Use observed win rate (not Thompson beta mean which is sigmoid-mapped reward)
        // Cold-start: use neutral 0.5 win rate until we have enough trades
        let has_enough_data = self.adaptive.stats.total_trades >= 15;
        let win_prob = if has_enough_data {
            self.adaptive.overall_win_rate().clamp(0.01, 0.99)
        } else {
            0.5
        };
        let sl_atr = self.excursions.get_sl_atr(&pattern_key, &action.0);
        let tp_atr = self.excursions.get_tp_atr(&pattern_key, &action.0);
        let rr_ratio = tp_atr / sl_atr;
        let kelly = ((win_prob * (1.0 + rr_ratio)) - 1.0) / rr_ratio;
        // Only reject negative Kelly after sufficient data to trust win rate
        if has_enough_data && kelly <= 0.0 {
            return self.hold_decision();
        }
        let kelly_frac = self.adaptive.kelly_fraction();
        let max_size = self.adaptive.max_position_size();
        let base_size = (kelly * kelly_frac).clamp(0.05, max_size);

        // Blend position sizing when KNN prediction exists
        let size = if let Some(ref pred) = prediction {
            if pred.avg_max_down > 0.01 {
                let knn_sizing =
                    pred.directional_prob * (pred.avg_max_up / pred.avg_max_down) * base_size;
                (0.6 * knn_sizing + 0.4 * base_size).clamp(0.05, max_size)
            } else {
                base_size
            }
        } else {
            base_size
        };

        let price = features.price;
        let atr = features.atr;

        // KNN-based stop-loss when prediction exists
        let sl_atr_final = if let Some(ref pred) = prediction {
            if pred.avg_max_down > 0.01 {
                let knn_sl = pred.avg_max_down * 1.2 / 100.0 * price / atr;
                knn_sl.clamp(1.0, 5.0)
            } else {
                sl_atr
            }
        } else {
            sl_atr
        };

        // No fixed TP — let winners run. Only SL + trailing stop.
        let (signal, stop_loss, take_profit) = match action.0.as_str() {
            "long" => (
                TradeSignal::Long,
                Some(price - sl_atr_final * atr),
                None, // trailing stop handles exit on profit side
            ),
            "short" => (TradeSignal::Short, Some(price + sl_atr_final * atr), None),
            _ => return self.hold_decision(),
        };

        let confidence = self.thompson.get_arm_mean(pool_key, &pattern, &action.0);
        // Store prediction direction for reward blending
        self.last_prediction = prediction.map(|p| p.directional_prob);
        TradingDecision {
            signal,
            size,
            stop_loss,
            take_profit,
            confidence,
            strategy_name: format!("{}_{}", action.0, pattern_key),
        }
    }

    fn hold_decision(&self) -> TradingDecision {
        TradingDecision {
            signal: TradeSignal::Hold,
            size: 0.0,
            stop_loss: None,
            take_profit: None,
            confidence: 0.0,
            strategy_name: "hold".into(),
        }
    }

    /// Record outcome after a trade closes.
    pub fn record_outcome(
        &mut self,
        symbol: &str,
        context: &MarketContext,
        strategy_id: &StrategyId,
        reward: f64,
    ) {
        // Record in global pool for shared learning
        self.thompson
            .record_outcome("_global", context, strategy_id, reward);

        if let Some(regret) = self.regret.get_mut(symbol) {
            regret.record(reward);
        }
        if let Some(plateau) = self.plateau.get_mut(symbol) {
            plateau.record(reward);
        }
    }

    /// Record excursion data for adaptive SL/TP.
    pub fn record_excursion(
        &mut self,
        pattern_key: &str,
        side: &str,
        favorable_atr: f64,
        adverse_atr: f64,
        won: bool,
    ) {
        self.excursions
            .record(pattern_key, side, favorable_atr, adverse_atr, won);
    }

    /// Record a hold observation (called periodically when not in a position).
    pub fn record_hold(&mut self, _symbol: &str, pattern: &MarketContext, _recent_return_pct: f64) {
        self.thompson
            .record_outcome("_global", pattern, &StrategyId("hold".into()), 0.5);
    }

    /// Check if transfer should be triggered.
    pub fn maybe_transfer(&mut self) -> Vec<(String, String)> {
        let mut transfers = Vec::new();

        let mut symbol_evidence: Vec<(String, f64)> = self
            .symbols
            .iter()
            .map(|s| {
                let evidence = self
                    .regret
                    .get(s)
                    .map(|r| r.total_observations as f64)
                    .unwrap_or(0.0);
                (s.clone(), evidence)
            })
            .collect();
        symbol_evidence.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        if symbol_evidence.len() < 2 {
            return transfers;
        }

        let source = &symbol_evidence[0].0;
        if symbol_evidence[0].1 < self.config.transfer_after_n as f64 {
            return transfers;
        }

        for (target, evidence) in &symbol_evidence[1..] {
            if *evidence < self.config.transfer_after_n as f64 / 2.0 {
                let should_transfer = self
                    .plateau
                    .get_mut(target)
                    .map(|pd| pd.check() != PlateauAction::Continue)
                    .unwrap_or(false);

                if should_transfer || *evidence < 50.0 {
                    self.thompson.transfer_priors(source, target);
                    transfers.push((source.clone(), target.clone()));
                }
            }
        }

        transfers
    }

    pub fn health_report(&self) -> HashMap<String, SymbolHealth> {
        let mut report: HashMap<String, SymbolHealth> = self
            .symbols
            .iter()
            .map(|s| {
                let regret = self.regret.get(s);
                let health = SymbolHealth {
                    observations: regret.map(|r| r.total_observations).unwrap_or(0),
                    average_regret: regret.map(|r| r.average_regret()).unwrap_or(0.0),
                    regret_growth_rate: regret.map(|r| r.growth_rate()).unwrap_or(1.0),
                    is_learning: regret.map(|r| r.growth_rate() < 0.8).unwrap_or(false),
                };
                (s.clone(), health)
            })
            .collect();

        report.insert(
            "_patterns".to_string(),
            SymbolHealth {
                observations: self.discretizer.sample_count() as u64,
                average_regret: 0.0,
                regret_growth_rate: 0.0,
                is_learning: self.discretizer.is_warmed_up(),
            },
        );

        report
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolHealth {
    pub observations: u64,
    pub average_regret: f64,
    pub regret_growth_rate: f64,
    pub is_learning: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_learning_engine_creation() {
        let engine = LearningEngine::new(
            vec!["BTCUSDT".into(), "ETHUSDT".into()],
            LearnerConfig::default(),
        );
        assert_eq!(engine.symbols.len(), 2);
        assert_eq!(engine.thompson.strategy_ids.len(), 3); // long, short, hold
    }

    #[test]
    fn test_learning_engine_decide_hold_initially() {
        let mut engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
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
                close_time: 900_000,
                quote_volume: 100_000.0,
                trades: 500,
            },
            history: vec![],
            indicators: IndicatorSet::default(),
        };
        let mut rng = rand::thread_rng();
        let library = PatternLibrary::with_defaults();
        let decision = engine.decide(&state, &mut rng, &library);
        assert_eq!(decision.signal, TradeSignal::Hold);
    }

    #[test]
    fn test_health_report() {
        let mut engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let ctx = MarketContext {
            volatility_tier: "1111".into(),
            trend_regime: "111".into(),
        };
        for _ in 0..20 {
            engine.record_outcome("BTCUSDT", &ctx, &StrategyId("long".into()), 0.7);
        }
        let report = engine.health_report();
        assert_eq!(report.get("BTCUSDT").unwrap().observations, 20);
    }

    #[test]
    fn test_excursion_tracker() {
        let mut tracker = ExcursionTracker::new();
        for _ in 0..20 {
            tracker.record("p", "long", 6.0, 1.5, true);
        }
        assert!(tracker.get_tp_atr("p", "long") > 4.0);
        assert!(tracker.get_sl_atr("p", "long") < 3.0);
    }

    #[test]
    fn test_excursion_defaults() {
        let tracker = ExcursionTracker::new();
        assert_eq!(tracker.get_sl_atr("unknown", "long"), 2.5);
        assert_eq!(tracker.get_tp_atr("unknown", "long"), 4.0);
    }

    #[test]
    fn test_excursion_side_specific_vs_fallback() {
        let mut tracker = ExcursionTracker::new();
        // Record only long trades for pattern "p"
        for _ in 0..20 {
            tracker.record("p", "long", 8.0, 1.0, true);
        }
        // Side-specific should exist
        let sl_long = tracker.get_sl_atr("p", "long");
        // Short should fall back to pattern-level
        let sl_short = tracker.get_sl_atr("p", "short");
        assert!(sl_long > 0.0);
        assert!(sl_short > 0.0);
    }

    #[test]
    fn test_excursion_sl_clamped() {
        let mut tracker = ExcursionTracker::new();
        // Very small adverse excursions
        for _ in 0..20 {
            tracker.record("p", "long", 5.0, 0.1, true);
        }
        let sl = tracker.get_sl_atr("p", "long");
        assert!(sl >= 1.0, "SL should be clamped to minimum 1.0 ATR: {}", sl);
    }

    #[test]
    fn test_excursion_tp_clamped() {
        let mut tracker = ExcursionTracker::new();
        // Very large favorable excursions
        for _ in 0..20 {
            tracker.record("p", "long", 20.0, 2.0, true);
        }
        let tp = tracker.get_tp_atr("p", "long");
        assert!(tp <= 8.0, "TP should be clamped to maximum 8.0 ATR: {}", tp);
    }

    #[test]
    fn test_excursion_only_winners_update_favorable() {
        let mut tracker = ExcursionTracker::new();
        // Only losing trades
        for _ in 0..20 {
            tracker.record("p", "long", 5.0, 3.0, false);
        }
        // Favorable should still be default (4.0) since we only had losses
        let tp = tracker.get_tp_atr("p", "long");
        assert!((tp - 4.0).abs() < 0.5, "TP should be near default for all-loss: {}", tp);
    }

    #[test]
    fn test_learner_config_defaults() {
        let cfg = LearnerConfig::default();
        assert_eq!(cfg.decay_factor, 0.995);
        assert_eq!(cfg.plateau_window, 5);
        assert_eq!(cfg.transfer_after_n, 150);
        assert_eq!(cfg.min_edge, 0.02);
    }

    #[test]
    fn test_record_outcome_updates_regret_and_plateau() {
        let mut engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let ctx = MarketContext {
            volatility_tier: "01".into(),
            trend_regime: "10".into(),
        };
        for _ in 0..10 {
            engine.record_outcome("BTCUSDT", &ctx, &StrategyId("long".into()), 0.8);
        }
        let health = engine.health_report();
        assert_eq!(health.get("BTCUSDT").unwrap().observations, 10);
    }

    #[test]
    fn test_record_hold() {
        let mut engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let ctx = MarketContext {
            volatility_tier: "01".into(),
            trend_regime: "10".into(),
        };
        engine.record_hold("BTCUSDT", &ctx, 0.5);
        // Should have recorded one observation in thompson global pool
        assert_eq!(engine.thompson.total_decisions, 1);
    }

    #[test]
    fn test_maybe_transfer_single_symbol() {
        let mut engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        // Single symbol should not trigger transfer
        let transfers = engine.maybe_transfer();
        assert!(transfers.is_empty());
    }

    #[test]
    fn test_maybe_transfer_insufficient_evidence() {
        let mut engine = LearningEngine::new(
            vec!["BTCUSDT".into(), "ETHUSDT".into()],
            LearnerConfig::default(),
        );
        // Not enough observations -> no transfer
        let transfers = engine.maybe_transfer();
        assert!(transfers.is_empty());
    }

    #[test]
    fn test_health_report_includes_patterns() {
        let engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let report = engine.health_report();
        assert!(report.contains_key("_patterns"));
        assert!(!report.get("_patterns").unwrap().is_learning); // not warmed up yet
    }

    #[test]
    fn test_hold_decision_fields() {
        let engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let d = engine.hold_decision();
        assert_eq!(d.signal, TradeSignal::Hold);
        assert_eq!(d.size, 0.0);
        assert!(d.stop_loss.is_none());
        assert!(d.take_profit.is_none());
        assert_eq!(d.confidence, 0.0);
        assert_eq!(d.strategy_name, "hold");
    }

    #[test]
    fn test_record_excursion() {
        let mut engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        engine.record_excursion("01_10", "long", 5.0, 2.0, true);
        engine.record_excursion("01_10", "long", 6.0, 1.5, true);
        let sl = engine.excursions.get_sl_atr("01_10", "long");
        let tp = engine.excursions.get_tp_atr("01_10", "long");
        assert!(sl > 0.0);
        assert!(tp > 0.0);
    }
}
