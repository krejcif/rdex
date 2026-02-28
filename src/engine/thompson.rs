use crate::domain::{MarketContext, StrategyId};
use rand::distributions::Distribution;
use rand::Rng;
use serde::{Deserialize, Serialize};
use statrs::distribution::Beta as BetaDist;
use std::collections::HashMap;

/// Beta distribution for Thompson Sampling
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BetaParams {
    pub alpha: f64,
    pub beta: f64,
}

impl BetaParams {
    pub fn uniform() -> Self {
        Self {
            alpha: 1.0,
            beta: 1.0,
        }
    }

    pub fn mean(&self) -> f64 {
        self.alpha / (self.alpha + self.beta)
    }

    pub fn variance(&self) -> f64 {
        let total = self.alpha + self.beta;
        (self.alpha * self.beta) / (total * total * (total + 1.0))
    }

    pub fn sample(&self, rng: &mut impl Rng) -> f64 {
        let dist = BetaDist::new(self.alpha, self.beta)
            .unwrap_or_else(|_| BetaDist::new(1.0, 1.0).unwrap());
        dist.sample(rng).clamp(0.001, 0.999)
    }

    pub fn update(&mut self, reward: f64) {
        self.alpha += reward;
        self.beta += 1.0 - reward;
    }

    pub fn evidence(&self) -> f64 {
        self.alpha + self.beta - 2.0
    }
}

/// Decaying Beta for non-stationary environments (crypto markets)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecayingBeta {
    pub params: BetaParams,
    pub decay_factor: f64,
    pub effective_n: f64,
}

impl DecayingBeta {
    pub fn new(decay_factor: f64) -> Self {
        Self {
            params: BetaParams::uniform(),
            decay_factor: decay_factor.clamp(0.9, 1.0),
            effective_n: 0.0,
        }
    }

    pub fn update(&mut self, reward: f64) {
        // Decay old evidence
        self.params.alpha = 1.0 + (self.params.alpha - 1.0) * self.decay_factor;
        self.params.beta = 1.0 + (self.params.beta - 1.0) * self.decay_factor;
        // Add new observation
        self.params.update(reward);
        self.effective_n = self.effective_n * self.decay_factor + 1.0;
    }

    pub fn effective_window(&self) -> f64 {
        if self.decay_factor >= 1.0 {
            self.effective_n
        } else {
            1.0 / (1.0 - self.decay_factor)
        }
    }
}

/// Thompson Sampling engine for strategy selection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThompsonEngine {
    /// Per-symbol, per-context, per-arm decaying beta
    pub arms: HashMap<String, HashMap<MarketContext, HashMap<StrategyId, DecayingBeta>>>,
    pub strategy_ids: Vec<StrategyId>,
    pub decay_factor: f64,
    /// Exploration coefficient for UCB bonus
    pub exploration_coeff: f64,
    /// Total decisions made
    pub total_decisions: u64,
    /// Visit counts for curiosity bonus
    visit_counts: HashMap<(String, MarketContext, StrategyId), u64>,
}

impl ThompsonEngine {
    pub fn new(strategy_names: Vec<String>, decay_factor: f64) -> Self {
        let strategy_ids: Vec<StrategyId> = strategy_names.into_iter().map(StrategyId).collect();

        Self {
            arms: HashMap::new(),
            strategy_ids,
            decay_factor,
            exploration_coeff: 0.3,
            total_decisions: 0,
            visit_counts: HashMap::new(),
        }
    }

    /// Select best arm for given symbol + context using Thompson Sampling + UCB.
    /// exploration_override allows the adaptive engine to control exploration level.
    pub fn select_arm(
        &self,
        symbol: &str,
        context: &MarketContext,
        rng: &mut impl Rng,
    ) -> StrategyId {
        self.select_arm_with_exploration(symbol, context, rng, self.exploration_coeff)
    }

    /// Select arm with a specific exploration coefficient (for adaptive control).
    pub fn select_arm_with_exploration(
        &self,
        symbol: &str,
        context: &MarketContext,
        rng: &mut impl Rng,
        exploration: f64,
    ) -> StrategyId {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_arm = self.strategy_ids[0].clone();

        for arm in &self.strategy_ids {
            let sample = self.get_beta(symbol, context, arm).params.sample(rng);
            let bonus = self.curiosity_bonus_with(symbol, context, arm, exploration);
            let score = sample + bonus;

            if score > best_score {
                best_score = score;
                best_arm = arm.clone();
            }
        }

        best_arm
    }

    /// Select arm with KNN-boosted priors.
    /// directional_prob > 0.6 boosts long, < 0.4 boosts short.
    pub fn select_arm_with_knn_prior(
        &self,
        symbol: &str,
        context: &MarketContext,
        rng: &mut impl Rng,
        exploration: f64,
        directional_prob: f64,
        confidence_scale: f64,
    ) -> StrategyId {
        let mut best_score = f64::NEG_INFINITY;
        let mut best_arm = self.strategy_ids[0].clone();

        for arm in &self.strategy_ids {
            let db = self.get_beta(symbol, context, arm);
            let mut alpha = db.params.alpha;
            let mut beta_param = db.params.beta;

            // Apply KNN bias
            if arm.0 == "long" && directional_prob > 0.6 {
                alpha += directional_prob * confidence_scale;
            } else if arm.0 == "short" && directional_prob < 0.4 {
                alpha += (1.0 - directional_prob) * confidence_scale;
            } else if arm.0 == "long" && directional_prob < 0.4 {
                beta_param += (1.0 - directional_prob) * confidence_scale;
            } else if arm.0 == "short" && directional_prob > 0.6 {
                beta_param += directional_prob * confidence_scale;
            }

            // Sample from Beta distribution with adjusted params + exploration bonus
            let adjusted_alpha = (alpha + exploration).max(0.01);
            let adjusted_beta = beta_param.max(0.01);
            let dist = BetaDist::new(adjusted_alpha, adjusted_beta)
                .unwrap_or_else(|_| BetaDist::new(1.0, 1.0).unwrap());
            let sample = dist.sample(rng);

            let bonus = self.curiosity_bonus_with(symbol, context, arm, exploration);
            let score = sample + bonus;

            if score > best_score {
                best_score = score;
                best_arm = arm.clone();
            }
        }

        best_arm
    }

    /// Record outcome of a trading decision
    pub fn record_outcome(
        &mut self,
        symbol: &str,
        context: &MarketContext,
        arm: &StrategyId,
        reward: f64,
    ) {
        let reward = reward.clamp(0.0, 1.0);

        // Update decaying beta
        let beta = self
            .arms
            .entry(symbol.to_string())
            .or_default()
            .entry(context.clone())
            .or_default()
            .entry(arm.clone())
            .or_insert_with(|| DecayingBeta::new(self.decay_factor));
        beta.update(reward);

        // Update visit counts
        let key = (symbol.to_string(), context.clone(), arm.clone());
        *self.visit_counts.entry(key).or_insert(0) += 1;
        self.total_decisions += 1;
    }

    fn get_beta(&self, symbol: &str, context: &MarketContext, arm: &StrategyId) -> DecayingBeta {
        self.arms
            .get(symbol)
            .and_then(|ctx| ctx.get(context))
            .and_then(|arms| arms.get(arm))
            .cloned()
            .unwrap_or_else(|| {
                // Hold-biased prior that decays with experience.
                // Early: strong hold bias (alpha=5, mean=0.83).
                // As total_decisions grow, hold bias weakens naturally.
                let mut db = DecayingBeta::new(self.decay_factor);
                if arm.0 == "hold" {
                    let hold_alpha = 1.0 + 4.0 / (1.0 + self.total_decisions as f64 / 100.0);
                    db.params.alpha = hold_alpha;
                    db.params.beta = 1.0;
                }
                db
            })
    }

    fn curiosity_bonus(&self, symbol: &str, context: &MarketContext, arm: &StrategyId) -> f64 {
        self.curiosity_bonus_with(symbol, context, arm, self.exploration_coeff)
    }

    fn curiosity_bonus_with(
        &self,
        symbol: &str,
        context: &MarketContext,
        arm: &StrategyId,
        exploration: f64,
    ) -> f64 {
        if self.total_decisions < 2 {
            return exploration;
        }

        let key = (symbol.to_string(), context.clone(), arm.clone());
        let visits = self.visit_counts.get(&key).copied().unwrap_or(0);

        if visits == 0 {
            return exploration * 2.0;
        }

        exploration * ((self.total_decisions as f64).ln() / visits as f64).sqrt()
    }

    /// Get the estimated mean reward for a specific arm.
    pub fn get_arm_mean(&self, symbol: &str, context: &MarketContext, arm_name: &str) -> f64 {
        let arm = StrategyId(arm_name.to_string());
        self.get_beta(symbol, context, &arm).params.mean()
    }

    /// Get summary of all arm states for reporting.
    /// Returns: Vec<(pattern_key, action, mean, evidence, variance)>
    pub fn arm_report(&self, symbol: &str) -> Vec<(String, String, f64, f64, f64)> {
        let mut report = Vec::new();
        if let Some(contexts) = self.arms.get(symbol) {
            for (ctx, arms) in contexts {
                let pattern_key = format!("{}_{}", ctx.volatility_tier, ctx.trend_regime);
                for (arm, db) in arms {
                    report.push((
                        pattern_key.clone(),
                        arm.0.clone(),
                        db.params.mean(),
                        db.params.evidence(),
                        db.params.variance(),
                    ));
                }
            }
        }
        report.sort_by(|a, b| a.0.cmp(&b.0).then(a.1.cmp(&b.1)));
        report
    }

    /// Check if engine is uncertain about best arm (triggers dual-path)
    pub fn is_uncertain(&self, symbol: &str, context: &MarketContext, threshold: f64) -> bool {
        let mut means: Vec<f64> = self
            .strategy_ids
            .iter()
            .map(|arm| self.get_beta(symbol, context, arm).params.mean())
            .collect();
        means.sort_by(|a, b| b.partial_cmp(a).unwrap());
        if means.len() < 2 {
            return true;
        }
        (means[0] - means[1]) < threshold
    }

    /// Extract compact priors for a symbol (for transfer)
    pub fn extract_priors(
        &self,
        symbol: &str,
    ) -> HashMap<MarketContext, HashMap<StrategyId, BetaParams>> {
        self.arms
            .get(symbol)
            .map(|contexts| {
                contexts
                    .iter()
                    .filter_map(|(ctx, arms)| {
                        let significant: HashMap<StrategyId, BetaParams> = arms
                            .iter()
                            .filter(|(_, db)| db.params.evidence() > 10.0)
                            .map(|(arm, db)| (arm.clone(), db.params.clone()))
                            .collect();
                        if significant.is_empty() {
                            None
                        } else {
                            Some((ctx.clone(), significant))
                        }
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Seed a symbol with dampened priors from another symbol
    pub fn transfer_priors(&mut self, source: &str, target: &str) {
        let priors = self.extract_priors(source);
        for (ctx, arms) in priors {
            for (arm, params) in arms {
                let dampened = BetaParams {
                    alpha: 1.0 + (params.alpha - 1.0).sqrt(),
                    beta: 1.0 + (params.beta - 1.0).sqrt(),
                };
                self.arms
                    .entry(target.to_string())
                    .or_default()
                    .entry(ctx.clone())
                    .or_default()
                    .entry(arm)
                    .or_insert_with(|| {
                        let mut db = DecayingBeta::new(self.decay_factor);
                        db.params = dampened;
                        db
                    });
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_context() -> MarketContext {
        MarketContext {
            volatility_tier: "medium".into(),
            trend_regime: "uptrend".into(),
        }
    }

    #[test]
    fn test_beta_params() {
        let mut b = BetaParams::uniform();
        assert!((b.mean() - 0.5).abs() < 1e-10);
        b.update(1.0);
        assert!(b.mean() > 0.5);
    }

    #[test]
    fn test_beta_sampling_in_range() {
        let b = BetaParams {
            alpha: 10.0,
            beta: 5.0,
        };
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let s = b.sample(&mut rng);
            assert!(s >= 0.0 && s <= 1.0);
        }
    }

    #[test]
    fn test_decaying_beta() {
        let mut db = DecayingBeta::new(0.995);
        for _ in 0..100 {
            db.update(0.8);
        }
        assert!(db.params.mean() > 0.7);
        assert!((db.effective_window() - 200.0).abs() < 1.0);
    }

    #[test]
    fn test_thompson_engine_select() {
        let mut engine = ThompsonEngine::new(
            vec![
                "momentum".into(),
                "mean_reversion".into(),
                "breakout".into(),
                "scalping".into(),
            ],
            0.995,
        );
        let ctx = test_context();
        let mut rng = rand::thread_rng();

        let arm = engine.select_arm("BTCUSDT", &ctx, &mut rng);
        assert!(!arm.0.is_empty());
    }

    #[test]
    fn test_thompson_engine_learns() {
        let mut engine = ThompsonEngine::new(vec!["good".into(), "bad".into()], 0.995);
        let ctx = test_context();

        // Train: "good" arm always wins
        for _ in 0..100 {
            engine.record_outcome("BTCUSDT", &ctx, &StrategyId("good".into()), 0.9);
            engine.record_outcome("BTCUSDT", &ctx, &StrategyId("bad".into()), 0.1);
        }

        // Engine should now prefer "good"
        let mut rng = rand::thread_rng();
        let mut good_count = 0;
        for _ in 0..100 {
            if engine.select_arm("BTCUSDT", &ctx, &mut rng).0 == "good" {
                good_count += 1;
            }
        }
        assert!(
            good_count > 80,
            "Engine should select good arm most of the time, got {}",
            good_count
        );
    }

    #[test]
    fn test_transfer_priors() {
        let mut engine = ThompsonEngine::new(
            vec!["a".into(), "b".into()],
            // Use higher decay so evidence doesn't decay away
            0.999,
        );
        let ctx = test_context();

        // Train on BTC with lots of observations to accumulate evidence > 10
        for _ in 0..100 {
            engine.record_outcome("BTCUSDT", &ctx, &StrategyId("a".into()), 0.85);
            engine.record_outcome("BTCUSDT", &ctx, &StrategyId("b".into()), 0.3);
        }

        // Verify source has enough evidence
        let src_priors = engine.extract_priors("BTCUSDT");
        assert!(
            !src_priors.is_empty(),
            "BTC should have priors with sufficient evidence"
        );

        // Transfer to ETH
        engine.transfer_priors("BTCUSDT", "ETHUSDT");

        // ETH should now have informative priors
        let eth_a = engine.get_beta("ETHUSDT", &ctx, &StrategyId("a".into()));
        assert!(
            eth_a.params.mean() > 0.5,
            "Transferred prior should favor arm a, got {}",
            eth_a.params.mean()
        );
    }

    #[test]
    fn test_uncertainty_detection() {
        let mut engine = ThompsonEngine::new(vec!["a".into(), "b".into()], 0.995);
        let ctx = test_context();

        // Uniform priors -> uncertain
        assert!(engine.is_uncertain("BTCUSDT", &ctx, 0.1));

        // After training, should be certain
        for _ in 0..100 {
            engine.record_outcome("BTCUSDT", &ctx, &StrategyId("a".into()), 0.95);
            engine.record_outcome("BTCUSDT", &ctx, &StrategyId("b".into()), 0.1);
        }
        assert!(!engine.is_uncertain("BTCUSDT", &ctx, 0.1));
    }

    #[test]
    fn test_hold_biased_prior() {
        let engine = ThompsonEngine::new(vec!["long".into(), "short".into(), "hold".into()], 0.995);
        let ctx = test_context();
        // Hold arm should have higher initial mean than long/short
        let hold_beta = engine.get_beta("BTCUSDT", &ctx, &StrategyId("hold".into()));
        let long_beta = engine.get_beta("BTCUSDT", &ctx, &StrategyId("long".into()));
        assert!(
            hold_beta.params.mean() > long_beta.params.mean(),
            "Hold should have higher initial mean: {} vs {}",
            hold_beta.params.mean(),
            long_beta.params.mean()
        );
    }

    #[test]
    fn test_hold_bias_decays_with_decisions() {
        let mut engine = ThompsonEngine::new(vec!["long".into(), "hold".into()], 0.995);
        let ctx = test_context();

        let early_hold = engine.get_beta("BTCUSDT", &ctx, &StrategyId("hold".into()));
        let early_mean = early_hold.params.mean();

        // Simulate many decisions
        for _ in 0..200 {
            engine.record_outcome("BTCUSDT", &ctx, &StrategyId("long".into()), 0.5);
        }

        let late_hold = engine.get_beta("BTCUSDT", &ctx, &StrategyId("hold".into()));
        // The unseen hold arm's alpha decays because total_decisions grows
        assert!(
            late_hold.params.mean() <= early_mean,
            "Hold bias should decay: {} vs {}",
            late_hold.params.mean(),
            early_mean
        );
    }

    #[test]
    fn test_get_arm_mean() {
        let mut engine = ThompsonEngine::new(vec!["a".into(), "b".into()], 0.995);
        let ctx = test_context();

        for _ in 0..50 {
            engine.record_outcome("BTC", &ctx, &StrategyId("a".into()), 0.9);
            engine.record_outcome("BTC", &ctx, &StrategyId("b".into()), 0.1);
        }

        assert!(engine.get_arm_mean("BTC", &ctx, "a") > 0.7);
        assert!(engine.get_arm_mean("BTC", &ctx, "b") < 0.3);
    }

    #[test]
    fn test_knn_prior_boosts_long() {
        let engine = ThompsonEngine::new(vec!["long".into(), "short".into()], 0.995);
        let ctx = test_context();
        let mut rng = rand::thread_rng();

        // With strong long signal (directional_prob > 0.6), long should be selected more often
        let mut long_count = 0;
        for _ in 0..100 {
            let arm = engine.select_arm_with_knn_prior(
                "BTC", &ctx, &mut rng,
                0.3, 0.8, // strong directional_prob toward long
                2.0,
            );
            if arm.0 == "long" {
                long_count += 1;
            }
        }
        assert!(long_count > 50, "KNN long boost should favor long: {}/100", long_count);
    }

    #[test]
    fn test_knn_prior_boosts_short() {
        let engine = ThompsonEngine::new(vec!["long".into(), "short".into()], 0.995);
        let ctx = test_context();
        let mut rng = rand::thread_rng();

        let mut short_count = 0;
        for _ in 0..100 {
            let arm = engine.select_arm_with_knn_prior(
                "BTC", &ctx, &mut rng,
                0.3, 0.2, // directional_prob < 0.4 -> favors short
                2.0,
            );
            if arm.0 == "short" {
                short_count += 1;
            }
        }
        assert!(short_count > 50, "KNN short boost should favor short: {}/100", short_count);
    }

    #[test]
    fn test_beta_params_variance() {
        let b = BetaParams { alpha: 10.0, beta: 10.0 };
        let var = b.variance();
        // Var = 10*10 / (20*20*21) = 100/8400 ≈ 0.0119
        assert!(var > 0.01 && var < 0.02, "Variance should be ~0.012, got {}", var);
    }

    #[test]
    fn test_beta_params_evidence() {
        let mut b = BetaParams::uniform();
        assert_eq!(b.evidence(), 0.0);
        b.update(0.8);
        assert!((b.evidence() - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_decaying_beta_decay_clamps() {
        let db = DecayingBeta::new(0.5); // below 0.9
        assert_eq!(db.decay_factor, 0.9);
        let db2 = DecayingBeta::new(1.5); // above 1.0
        assert_eq!(db2.decay_factor, 1.0);
    }

    #[test]
    fn test_extract_priors_filters_low_evidence() {
        let mut engine = ThompsonEngine::new(vec!["a".into()], 0.999);
        let ctx = test_context();
        // Only 3 observations -> evidence < 10
        for _ in 0..3 {
            engine.record_outcome("BTC", &ctx, &StrategyId("a".into()), 0.8);
        }
        let priors = engine.extract_priors("BTC");
        assert!(priors.is_empty(), "Low evidence should be filtered");
    }

    #[test]
    fn test_curiosity_bonus_unvisited_arm() {
        let mut engine = ThompsonEngine::new(vec!["a".into(), "b".into()], 0.995);
        let ctx = test_context();
        // Visit only "a"
        for _ in 0..10 {
            engine.record_outcome("BTC", &ctx, &StrategyId("a".into()), 0.5);
        }
        let bonus_b = engine.curiosity_bonus("BTC", &ctx, &StrategyId("b".into()));
        let bonus_a = engine.curiosity_bonus("BTC", &ctx, &StrategyId("a".into()));
        assert!(bonus_b > bonus_a, "Unvisited arm should have higher bonus: {} vs {}", bonus_b, bonus_a);
    }

    #[test]
    fn test_record_outcome_clamps_reward() {
        let mut engine = ThompsonEngine::new(vec!["a".into()], 0.995);
        let ctx = test_context();
        // Reward above 1.0 should be clamped
        engine.record_outcome("BTC", &ctx, &StrategyId("a".into()), 5.0);
        let beta = engine.get_beta("BTC", &ctx, &StrategyId("a".into()));
        // After one update with reward=1.0 (clamped): alpha += 1, beta += 0
        assert!(beta.params.alpha > 1.0);
    }

    #[test]
    fn test_record_outcome_clamps_negative_reward() {
        let mut engine = ThompsonEngine::new(vec!["a".into()], 0.995);
        let ctx = test_context();
        engine.record_outcome("BTC", &ctx, &StrategyId("a".into()), -1.0);
        // Reward clamped to 0 -> alpha += 0, beta += 1
        let beta = engine.get_beta("BTC", &ctx, &StrategyId("a".into()));
        assert!(beta.params.beta > 1.0);
    }

    #[test]
    fn test_total_decisions_increments() {
        let mut engine = ThompsonEngine::new(vec!["a".into()], 0.995);
        let ctx = test_context();
        assert_eq!(engine.total_decisions, 0);
        engine.record_outcome("BTC", &ctx, &StrategyId("a".into()), 0.5);
        assert_eq!(engine.total_decisions, 1);
        engine.record_outcome("BTC", &ctx, &StrategyId("a".into()), 0.5);
        assert_eq!(engine.total_decisions, 2);
    }

    #[test]
    fn test_multiple_contexts_independent() {
        let mut engine = ThompsonEngine::new(vec!["a".into()], 0.995);
        let ctx1 = MarketContext { volatility_tier: "low".into(), trend_regime: "up".into() };
        let ctx2 = MarketContext { volatility_tier: "high".into(), trend_regime: "down".into() };

        for _ in 0..50 {
            engine.record_outcome("BTC", &ctx1, &StrategyId("a".into()), 0.9);
            engine.record_outcome("BTC", &ctx2, &StrategyId("a".into()), 0.1);
        }

        let mean1 = engine.get_arm_mean("BTC", &ctx1, "a");
        let mean2 = engine.get_arm_mean("BTC", &ctx2, "a");
        assert!(mean1 > 0.7, "ctx1 should have high mean: {}", mean1);
        assert!(mean2 < 0.3, "ctx2 should have low mean: {}", mean2);
    }

    #[test]
    fn test_decaying_beta_effective_n() {
        let mut db = DecayingBeta::new(0.99);
        assert_eq!(db.effective_n, 0.0);
        db.update(0.5);
        assert!((db.effective_n - 1.0).abs() < 0.01);
        db.update(0.5);
        assert!(db.effective_n > 1.9);
    }
}
