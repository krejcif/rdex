use serde::{Deserialize, Serialize};
use std::collections::VecDeque;

/// Fully adaptive parameter engine.
/// Every trading parameter is derived from observed market data.
/// No hardcoded multipliers — relationships between statistics drive everything.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveParams {
    /// Recent trade outcomes for adaptive parameters
    recent_outcomes: VecDeque<TradeOutcome>,
    /// Window size adapts to available data
    max_window: usize,
    /// Running statistics for adaptation
    pub stats: AdaptiveStats,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct TradeOutcome {
    pnl_pct: f64,
    duration: usize, // in candles
    adverse_atr: f64,
    favorable_atr: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveStats {
    pub total_trades: u64,
    pub total_wins: u64,
    pub total_candles_seen: u64,
    /// EMA of trade PnL
    pub ema_pnl: f64,
    /// EMA of squared PnL (for variance)
    pub ema_pnl_sq: f64,
    /// EMA of winning trade duration
    pub ema_win_duration: f64,
    /// EMA of losing trade duration
    pub ema_loss_duration: f64,
    /// EMA of favorable excursion (ATR multiples)
    pub ema_favorable: f64,
    /// EMA of adverse excursion (ATR multiples)
    pub ema_adverse: f64,
    /// EMA of reward-to-risk ratio (favorable / adverse)
    pub ema_rr_ratio: f64,
    /// EMA decay factor (adapts itself!)
    pub ema_decay: f64,
    /// EMA of KNN prediction accuracy (0..1)
    pub ema_knn_accuracy: f64,
}

impl Default for AdaptiveStats {
    fn default() -> Self {
        Self {
            total_trades: 0,
            total_wins: 0,
            total_candles_seen: 0,
            ema_pnl: 0.0,
            ema_pnl_sq: 4.0, // initial variance ~4 (std ~2%)
            ema_win_duration: 15.0,
            ema_loss_duration: 8.0,
            ema_favorable: 4.0,
            ema_adverse: 2.5,
            ema_rr_ratio: 1.6, // initial: favorable/adverse = 4.0/2.5
            ema_decay: 0.92,
            ema_knn_accuracy: 0.5,
        }
    }
}

impl AdaptiveParams {
    pub fn new() -> Self {
        Self {
            recent_outcomes: VecDeque::new(),
            max_window: 100,
            stats: AdaptiveStats::default(),
        }
    }

    /// Record a completed trade — all adaptive params update from this.
    pub fn record_trade(
        &mut self,
        pnl_pct: f64,
        duration: usize,
        favorable_atr: f64,
        adverse_atr: f64,
    ) {
        let outcome = TradeOutcome {
            pnl_pct,
            duration,
            adverse_atr,
            favorable_atr,
        };

        self.recent_outcomes.push_back(outcome);
        if self.recent_outcomes.len() > self.max_window {
            self.recent_outcomes.pop_front();
        }

        let d = self.stats.ema_decay;
        self.stats.total_trades += 1;
        if pnl_pct > 0.0 {
            self.stats.total_wins += 1;
        }

        // Update EMAs
        self.stats.ema_pnl = self.stats.ema_pnl * d + pnl_pct * (1.0 - d);
        self.stats.ema_pnl_sq = self.stats.ema_pnl_sq * d + (pnl_pct * pnl_pct) * (1.0 - d);

        if pnl_pct > 0.0 {
            self.stats.ema_win_duration =
                self.stats.ema_win_duration * d + duration as f64 * (1.0 - d);
            self.stats.ema_favorable = self.stats.ema_favorable * d + favorable_atr * (1.0 - d);
        } else {
            self.stats.ema_loss_duration =
                self.stats.ema_loss_duration * d + duration as f64 * (1.0 - d);
        }
        self.stats.ema_adverse = self.stats.ema_adverse * d + adverse_atr * (1.0 - d);

        // Update reward-to-risk ratio
        if adverse_atr > 0.01 {
            let rr = favorable_atr / adverse_atr;
            self.stats.ema_rr_ratio = self.stats.ema_rr_ratio * d + rr * (1.0 - d);
        }

        // Adapt EMA decay: more trades → slower decay (more stable)
        self.stats.ema_decay = 1.0 - 1.0 / ((self.stats.total_trades as f64).sqrt() + 5.0);
    }

    pub fn tick_candle(&mut self) {
        self.stats.total_candles_seen += 1;
    }

    /// PnL standard deviation — derived from EMA of squared PnL.
    fn pnl_std(&self) -> f64 {
        let variance = (self.stats.ema_pnl_sq - self.stats.ema_pnl * self.stats.ema_pnl).max(0.01);
        variance.sqrt()
    }

    /// Adaptive sigmoid steepness for reward function.
    /// k = 1/pnl_std — scales inversely with trade PnL volatility.
    /// Small trades → steep sigmoid. Large trades → gentle sigmoid.
    pub fn reward_k(&self) -> f64 {
        (1.0 / self.pnl_std()).clamp(0.5, 5.0)
    }


    /// Win rate from all-time statistics.
    pub fn overall_win_rate(&self) -> f64 {
        if self.stats.total_trades == 0 {
            return 0.5;
        }
        self.stats.total_wins as f64 / self.stats.total_trades as f64
    }

    /// Adaptive cooldown: win_duration * (1 - wr) / wr.
    /// Win more → trade more. Lose more → trade less.
    /// wr=0.5 → win_dur, wr=0.6 → 0.67*win_dur, wr=0.4 → 1.5*win_dur.
    pub fn cooldown(&self) -> usize {
        if self.stats.total_trades < 5 {
            return 15;
        }

        let recent_wr = self.recent_win_rate().clamp(0.1, 0.9);
        let cd = self.stats.ema_win_duration * (1.0 - recent_wr) / recent_wr;
        cd.round().clamp(8.0, 40.0) as usize
    }

    /// Adaptive minimum hold: geometric mean of win and loss durations.
    /// Balances between giving winners time and cutting losers.
    pub fn min_hold(&self) -> usize {
        if self.stats.total_trades < 10 {
            return 8;
        }
        // Geometric mean = sqrt(win_dur * loss_dur)
        // Balances the two duration signals naturally
        let gm = (self.stats.ema_win_duration * self.stats.ema_loss_duration).sqrt();
        gm.round().clamp(3.0, 20.0) as usize
    }

    /// Adaptive maximum hold: win_duration / (1 - win_rate)^2.
    /// Nonlinear: high win rate → much longer hold (confidence).
    /// wr=0.4 → ~42, wr=0.5 → ~60, wr=0.6 → ~94.
    pub fn max_hold(&self) -> usize {
        if self.stats.total_trades < 10 {
            return 96;
        }
        let wr = self.overall_win_rate().clamp(0.1, 0.9);
        let loss_rate = 1.0 - wr;
        let max = self.stats.ema_win_duration / (loss_rate * loss_rate);
        max.round().clamp(20.0, 192.0) as usize
    }

    /// Adaptive trailing stop activation (in ATR multiples).
    /// Derived from favorable excursion divided by (1 + RR ratio).
    /// High RR → activate earlier (catch big moves). Low RR → activate later (be selective).
    pub fn trail_activation_atr(&self) -> f64 {
        if self.stats.total_trades < 10 {
            return 1.5;
        }
        // activation = favorable / (1 + rr_ratio)
        // With favorable=4.0 and rr=1.6: activation = 4.0/2.6 = 1.54
        let activation = self.stats.ema_favorable / (1.0 + self.stats.ema_rr_ratio);
        activation.clamp(0.5, 4.0)
    }

    /// Adaptive trailing distance (in ATR multiples).
    /// adverse * (1 - win_rate) — high WR → tight trail. Low WR → wide trail.
    pub fn trail_distance_atr(&self) -> f64 {
        if self.stats.total_trades < 10 {
            return 1.0;
        }
        let wr = self.overall_win_rate();
        let distance = self.stats.ema_adverse * (1.0 - wr);
        distance.clamp(0.3, 2.5)
    }

    /// Adaptive breakeven activation (in ATR multiples).
    /// adverse * loss_dur/(win_dur+loss_dur) — loss zone fraction of noise.
    pub fn breakeven_activation_atr(&self) -> f64 {
        if self.stats.total_trades < 10 {
            return 0.75;
        }
        let total_dur = self.stats.ema_win_duration + self.stats.ema_loss_duration;
        let loss_fraction = self.stats.ema_loss_duration / total_dur.max(1.0);
        let be = self.stats.ema_adverse * loss_fraction;
        be.clamp(0.3, 2.0)
    }

    /// Adaptive exploration coefficient.
    /// Decays as trades accumulate relative to win_dur * rr (quality-adjusted).
    /// Higher quality trades → slower decay. Lower quality → faster decay.
    pub fn exploration_coeff(&self) -> f64 {
        if self.stats.total_trades < 5 {
            return 0.4;
        }
        let trades = self.stats.total_trades as f64;
        // quality_scale = win_duration * rr_ratio (how good are our trades)
        // More quality → takes more trades to reduce exploration
        let quality_scale = self.stats.ema_win_duration * self.stats.ema_rr_ratio;
        let wr = self.overall_win_rate();
        // Start at (1-wr) — explore more when losing
        let base = 1.0 - wr;
        base / (1.0 + trades / quality_scale)
    }

    /// Adaptive Kelly multiplier.
    /// Uses actual observed win rate and reward/risk ratio (true Kelly formula).
    pub fn kelly_fraction(&self) -> f64 {
        if self.stats.total_trades < 15 {
            return 0.35;
        }

        let wr = self.overall_win_rate();
        let rr = self.stats.ema_rr_ratio;
        // Kelly formula: f = (p * b - q) / b  where p=win_rate, b=rr_ratio, q=1-p
        let kelly_full = (wr * rr - (1.0 - wr)) / rr.max(0.01);
        // Fraction = wr * rr / (wr * rr + 1-wr) — expected gain share of outcomes
        // Purely data-derived: high EV → use more Kelly, low EV → less
        let ev_plus = wr * rr;
        let ev_minus = 1.0 - wr;
        let fraction = ev_plus / (ev_plus + ev_minus).max(0.01);
        (kelly_full * fraction).clamp(0.05, 0.50)
    }

    /// Adaptive maximum position size.
    /// Derived from Kelly fraction scaled by rr_ratio / (1 + rr_ratio).
    /// High RR → allow bigger positions. Low RR → smaller positions.
    pub fn max_position_size(&self) -> f64 {
        if self.stats.total_trades < 15 {
            return 0.18;
        }

        let rr = self.stats.ema_rr_ratio;
        // rr_scale = rr / (1 + rr) — maps RR to (0, 1) range
        // rr=1.0 → 0.50, rr=1.5 → 0.60, rr=2.0 → 0.67
        let rr_scale = rr / (1.0 + rr);
        let base = self.kelly_fraction() * rr_scale;
        base.clamp(0.05, 0.30)
    }

    /// Record whether a KNN prediction was correct (tracks EMA accuracy).
    pub fn record_prediction_accuracy(&mut self, correct: bool) {
        let val = if correct { 1.0 } else { 0.0 };
        let d = self.stats.ema_decay;
        self.stats.ema_knn_accuracy = self.stats.ema_knn_accuracy * d + val * (1.0 - d);
    }

    /// Current KNN prediction accuracy EMA.
    pub fn knn_accuracy(&self) -> f64 {
        self.stats.ema_knn_accuracy
    }

    /// Confidence score [0, 1] based on recent performance and consistency.
    fn confidence_score(&self) -> f64 {
        if self.recent_outcomes.is_empty() {
            return 0.0;
        }

        let wr = self.recent_win_rate();
        let avg_pnl = self.stats.ema_pnl;
        let pnl_std = self.pnl_std();

        // Sharpe-like: mean / std gives consistency-adjusted confidence
        let sharpe_like = (avg_pnl / pnl_std.max(0.01)).clamp(-2.0, 2.0);
        let sharpe_score = (sharpe_like / 4.0 + 0.5).clamp(0.0, 1.0); // -2→0, +2→1

        // Win rate score — centered on 50%
        let wr_score = ((wr - 0.4) / 0.3).clamp(0.0, 1.0); // 0.4→0, 0.7→1

        // Blend: Sharpe-like matters more (quality), WR for consistency
        (sharpe_score * 0.6 + wr_score * 0.4).clamp(0.0, 1.0)
    }

    /// Recent win rate from outcome window.
    fn recent_win_rate(&self) -> f64 {
        if self.recent_outcomes.is_empty() {
            return 0.5;
        }
        let wins = self
            .recent_outcomes
            .iter()
            .filter(|o| o.pnl_pct > 0.0)
            .count();
        wins as f64 / self.recent_outcomes.len() as f64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adaptive_params_initial() {
        let params = AdaptiveParams::new();
        assert_eq!(params.cooldown(), 15);
        assert_eq!(params.min_hold(), 8);
        assert_eq!(params.max_hold(), 96);
        assert!(params.trail_activation_atr() > 0.0);
        assert!(params.trail_distance_atr() > 0.0);
        assert!(params.exploration_coeff() > 0.0);
        assert!(params.reward_k() > 0.0);
    }

    #[test]
    fn test_adaptive_cooldown_adjusts() {
        let mut params = AdaptiveParams::new();
        // Record winning trades
        for i in 0..20 {
            params.tick_candle();
            params.tick_candle();
            params.record_trade(2.0, 10, 3.0, 1.5);
        }
        let winning_cooldown = params.cooldown();

        let mut params2 = AdaptiveParams::new();
        // Record losing trades
        for i in 0..20 {
            params2.tick_candle();
            params2.tick_candle();
            params2.record_trade(-2.0, 5, 0.5, 3.0);
        }
        let losing_cooldown = params2.cooldown();

        assert!(
            losing_cooldown > winning_cooldown,
            "Losing streak should produce longer cooldown: {} vs {}",
            losing_cooldown,
            winning_cooldown
        );
    }

    #[test]
    fn test_adaptive_trailing_adjusts() {
        let mut params = AdaptiveParams::new();
        for _ in 0..30 {
            params.record_trade(3.0, 15, 6.0, 2.0);
        }
        let big_trail = params.trail_activation_atr();

        let mut params2 = AdaptiveParams::new();
        for _ in 0..30 {
            params2.record_trade(1.0, 8, 2.0, 1.5);
        }
        let small_trail = params2.trail_activation_atr();

        assert!(
            big_trail > small_trail,
            "Big excursions should produce higher trail activation: {} vs {}",
            big_trail,
            small_trail
        );
    }

    #[test]
    fn test_confidence_increases_with_wins() {
        let mut params = AdaptiveParams::new();
        for _ in 0..30 {
            params.record_trade(3.0, 10, 4.0, 1.5);
        }
        assert!(params.kelly_fraction() > 0.1);
        assert!(params.max_position_size() > 0.05);
    }

    #[test]
    fn test_exploration_decays() {
        let mut params = AdaptiveParams::new();
        let initial = params.exploration_coeff();
        for _ in 0..200 {
            params.record_trade(0.5, 5, 2.0, 1.5);
        }
        let later = params.exploration_coeff();
        assert!(
            later < initial,
            "Exploration should decay: {} vs {}",
            later,
            initial
        );
    }

    #[test]
    fn test_ema_decay_adapts() {
        let mut params = AdaptiveParams::new();
        let initial_decay = params.stats.ema_decay;
        for i in 0..100 {
            params.record_trade(if i % 2 == 0 { 1.0 } else { -0.5 }, 8, 3.0, 2.0);
        }
        assert!(
            params.stats.ema_decay > initial_decay,
            "EMA decay should increase with more trades"
        );
    }

    #[test]
    fn test_knn_prediction_accuracy_tracking() {
        let mut params = AdaptiveParams::new();
        for _ in 0..10 {
            params.record_prediction_accuracy(true);
        }
        assert!(params.knn_accuracy() > 0.7);
        for _ in 0..10 {
            params.record_prediction_accuracy(false);
        }
        assert!(params.knn_accuracy() < 0.7);
    }

    #[test]
    fn test_reward_k_adapts_to_volatility() {
        let mut low_vol = AdaptiveParams::new();
        for _ in 0..30 {
            low_vol.record_trade(0.3, 5, 1.0, 0.5); // small moves
        }

        let mut high_vol = AdaptiveParams::new();
        for _ in 0..30 {
            high_vol.record_trade(5.0, 5, 4.0, 3.0); // large moves
        }

        assert!(
            low_vol.reward_k() > high_vol.reward_k(),
            "Low volatility should produce higher k: {} vs {}",
            low_vol.reward_k(),
            high_vol.reward_k()
        );
    }

    #[test]
    fn test_adaptive_zero_trades_defaults() {
        let params = AdaptiveParams::new();
        assert_eq!(params.overall_win_rate(), 0.5);
        assert_eq!(params.kelly_fraction(), 0.35);
        assert_eq!(params.max_position_size(), 0.18);
        assert_eq!(params.exploration_coeff(), 0.4);
    }

    #[test]
    fn test_adaptive_few_trades_fallbacks() {
        let mut params = AdaptiveParams::new();
        for _ in 0..3 {
            params.record_trade(1.0, 5, 2.0, 1.0);
        }
        // < 5 trades: cooldown still default
        assert_eq!(params.cooldown(), 15);
        // < 10 trades: min_hold still default
        assert_eq!(params.min_hold(), 8);
        // < 10 trades: max_hold still default
        assert_eq!(params.max_hold(), 96);
        // < 10 trades: trail defaults
        assert_eq!(params.trail_activation_atr(), 1.5);
        assert_eq!(params.trail_distance_atr(), 1.0);
        assert_eq!(params.breakeven_activation_atr(), 0.75);
    }

    #[test]
    fn test_tick_candle() {
        let mut params = AdaptiveParams::new();
        assert_eq!(params.stats.total_candles_seen, 0);
        params.tick_candle();
        params.tick_candle();
        params.tick_candle();
        assert_eq!(params.stats.total_candles_seen, 3);
    }

    #[test]
    fn test_recent_outcomes_window() {
        let mut params = AdaptiveParams::new();
        // Fill beyond max_window (100)
        for i in 0..150 {
            params.record_trade(if i % 2 == 0 { 1.0 } else { -0.5 }, 5, 2.0, 1.0);
        }
        assert_eq!(params.stats.total_trades, 150);
        assert!(params.recent_outcomes.len() <= 100);
    }

    #[test]
    fn test_min_hold_adapts_with_trades() {
        let mut params = AdaptiveParams::new();
        // Record trades with known durations
        for _ in 0..20 {
            params.record_trade(2.0, 20, 3.0, 1.0); // long winning trades
        }
        for _ in 0..20 {
            params.record_trade(-1.0, 5, 0.5, 2.0); // short losing trades
        }
        let mh = params.min_hold();
        // geometric mean of ~20 and ~5 -> ~10
        assert!(mh >= 3 && mh <= 20, "min_hold should be in [3,20], got {}", mh);
    }

    #[test]
    fn test_max_hold_increases_with_win_rate() {
        let mut params_winning = AdaptiveParams::new();
        for _ in 0..50 {
            params_winning.record_trade(2.0, 12, 3.0, 1.0);
        }

        let mut params_losing = AdaptiveParams::new();
        for _ in 0..25 {
            params_losing.record_trade(2.0, 12, 3.0, 1.0);
        }
        for _ in 0..25 {
            params_losing.record_trade(-1.0, 5, 0.5, 2.0);
        }

        assert!(
            params_winning.max_hold() >= params_losing.max_hold(),
            "Higher win rate should allow longer hold: {} vs {}",
            params_winning.max_hold(),
            params_losing.max_hold()
        );
    }

    #[test]
    fn test_kelly_negative_ev() {
        let mut params = AdaptiveParams::new();
        // All losses -> negative EV
        for _ in 0..30 {
            params.record_trade(-2.0, 5, 0.5, 3.0);
        }
        let kelly = params.kelly_fraction();
        assert!(kelly >= 0.05, "Kelly should be clamped to minimum: {}", kelly);
    }

    #[test]
    fn test_max_position_size_bounded() {
        let mut params = AdaptiveParams::new();
        for _ in 0..100 {
            params.record_trade(10.0, 5, 8.0, 1.0); // extremely good trades
        }
        let max_pos = params.max_position_size();
        assert!(max_pos <= 0.30, "Max position should be clamped: {}", max_pos);
    }

    #[test]
    fn test_breakeven_activation_adapts() {
        let mut params = AdaptiveParams::new();
        for _ in 0..20 {
            params.record_trade(2.0, 15, 4.0, 2.0);
            params.record_trade(-1.0, 5, 1.0, 3.0);
        }
        let be = params.breakeven_activation_atr();
        assert!(be >= 0.3 && be <= 2.0, "Breakeven should be in range: {}", be);
    }

    #[test]
    fn test_pnl_std_minimum() {
        let params = AdaptiveParams::new();
        // With default ema_pnl_sq=4.0 and ema_pnl=0.0, variance=4.0, std=2.0
        let k = params.reward_k();
        // k = 1/std = 1/2 = 0.5
        assert!((k - 0.5).abs() < 0.01, "Initial reward_k should be 0.5: {}", k);
    }

    #[test]
    fn test_overall_win_rate_mixed() {
        let mut params = AdaptiveParams::new();
        for _ in 0..6 {
            params.record_trade(2.0, 5, 2.0, 1.0);
        }
        for _ in 0..4 {
            params.record_trade(-1.0, 3, 0.5, 2.0);
        }
        assert!((params.overall_win_rate() - 0.6).abs() < 0.01);
    }

    #[test]
    fn test_rr_ratio_updates() {
        let mut params = AdaptiveParams::new();
        let initial_rr = params.stats.ema_rr_ratio;
        for _ in 0..30 {
            // favorable=6, adverse=2 -> rr=3
            params.record_trade(2.0, 5, 6.0, 2.0);
        }
        assert!(params.stats.ema_rr_ratio > initial_rr, "RR should increase with high favorable");
    }
}
