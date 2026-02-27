use serde::{Deserialize, Serialize};

/// Result of verifying a cross-symbol transfer
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransferVerification {
    pub source_symbol: String,
    pub target_symbol: String,
    pub source_before: f64,
    pub source_after: f64,
    pub target_before: f64,
    pub target_after: f64,
    pub improved_target: bool,
    pub regressed_source: bool,
    pub promotable: bool,
    pub acceleration_factor: f64,
}

impl TransferVerification {
    pub fn verify(
        source_symbol: String,
        target_symbol: String,
        source_before: f64,
        source_after: f64,
        target_before: f64,
        target_after: f64,
        baseline_cycles: u64,
        transfer_cycles: u64,
    ) -> Self {
        let improved_target = target_after > target_before;
        let regressed_source = source_after < source_before - 0.01;
        let promotable = improved_target && !regressed_source;
        let acceleration_factor = if transfer_cycles > 0 {
            baseline_cycles as f64 / transfer_cycles as f64
        } else {
            1.0
        };

        Self {
            source_symbol,
            target_symbol,
            source_before,
            source_after,
            target_before,
            target_after,
            improved_target,
            regressed_source,
            promotable,
            acceleration_factor,
        }
    }
}

/// Tracks regret for learning quality assessment
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RegretTracker {
    pub total_regret: f64,
    pub total_observations: u64,
    best_reward_seen: f64,
}

impl RegretTracker {
    pub fn record(&mut self, reward: f64) {
        if reward > self.best_reward_seen {
            self.best_reward_seen = reward;
        }
        let regret = (self.best_reward_seen - reward).max(0.0);
        self.total_regret += regret;
        self.total_observations += 1;
    }

    /// Sublinear growth (< 1.0) means the system is learning
    pub fn growth_rate(&self) -> f64 {
        if self.total_observations < 10 || self.total_regret < 1e-10 {
            return 1.0;
        }
        self.total_regret.ln() / (self.total_observations as f64).ln()
    }

    pub fn average_regret(&self) -> f64 {
        if self.total_observations == 0 {
            return 0.0;
        }
        self.total_regret / self.total_observations as f64
    }
}

/// Plateau detector: detects when learning stalls
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlateauDetector {
    scores: Vec<f64>,
    window_size: usize,
    threshold: f64,
    pub consecutive_plateaus: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum PlateauAction {
    Continue,
    IncreaseExploration,
    TriggerTransfer,
    InjectDiversity,
}

impl PlateauDetector {
    pub fn new(window_size: usize, threshold: f64) -> Self {
        Self {
            scores: Vec::new(),
            window_size: window_size.max(3),
            threshold,
            consecutive_plateaus: 0,
        }
    }

    pub fn record(&mut self, score: f64) {
        self.scores.push(score);
    }

    pub fn check(&mut self) -> PlateauAction {
        let n = self.scores.len();
        if n < self.window_size * 2 {
            self.consecutive_plateaus = 0;
            return PlateauAction::Continue;
        }

        let recent: f64 =
            self.scores[n - self.window_size..].iter().sum::<f64>() / self.window_size as f64;
        let prior: f64 = self.scores[n - 2 * self.window_size..n - self.window_size]
            .iter()
            .sum::<f64>()
            / self.window_size as f64;

        let improvement = (recent - prior).abs();

        if improvement < self.threshold {
            self.consecutive_plateaus += 1;
            match self.consecutive_plateaus {
                1 => PlateauAction::IncreaseExploration,
                2..=3 => PlateauAction::TriggerTransfer,
                _ => PlateauAction::InjectDiversity,
            }
        } else {
            self.consecutive_plateaus = 0;
            PlateauAction::Continue
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_transfer_verification_promotable() {
        let v = TransferVerification::verify(
            "BTCUSDT".into(),
            "ETHUSDT".into(),
            0.8,
            0.79, // source stable
            0.3,
            0.7, // target improved
            100,
            40,
        );
        assert!(v.promotable);
        assert!((v.acceleration_factor - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_transfer_verification_regression() {
        let v = TransferVerification::verify(
            "BTCUSDT".into(),
            "ETHUSDT".into(),
            0.8,
            0.5, // source regressed!
            0.3,
            0.7,
            100,
            40,
        );
        assert!(!v.promotable);
    }

    #[test]
    fn test_regret_tracker() {
        let mut tracker = RegretTracker::default();
        // Optimal arm: always 0.9
        for _ in 0..100 {
            tracker.record(0.9);
        }
        assert!(tracker.total_regret < 1e-10); // no regret when always best

        // Now some bad decisions
        for _ in 0..50 {
            tracker.record(0.3);
        }
        assert!(tracker.total_regret > 0.0);
    }

    #[test]
    fn test_plateau_detector() {
        let mut pd = PlateauDetector::new(5, 0.01);

        // Improving scores
        for i in 0..10 {
            pd.record(0.5 + i as f64 * 0.05);
        }
        assert_eq!(pd.check(), PlateauAction::Continue);

        // Flat scores
        for _ in 0..20 {
            pd.record(0.95);
        }
        let action = pd.check();
        assert_ne!(action, PlateauAction::Continue);
    }

    #[test]
    fn test_transfer_verification_zero_transfer_cycles() {
        let v = TransferVerification::verify(
            "BTC".into(), "ETH".into(),
            0.5, 0.5, 0.3, 0.4,
            100, 0,
        );
        assert_eq!(v.acceleration_factor, 1.0);
    }

    #[test]
    fn test_transfer_verification_no_improvement() {
        let v = TransferVerification::verify(
            "BTC".into(), "ETH".into(),
            0.8, 0.8, 0.5, 0.4, // target got worse
            100, 50,
        );
        assert!(!v.improved_target);
        assert!(!v.promotable);
    }

    #[test]
    fn test_regret_tracker_zero_observations() {
        let tracker = RegretTracker::default();
        assert_eq!(tracker.average_regret(), 0.0);
        assert_eq!(tracker.growth_rate(), 1.0);
    }

    #[test]
    fn test_regret_tracker_growth_rate_with_few_observations() {
        let mut tracker = RegretTracker::default();
        for _ in 0..5 {
            tracker.record(0.5);
        }
        // < 10 observations -> growth_rate returns 1.0
        assert_eq!(tracker.growth_rate(), 1.0);
    }

    #[test]
    fn test_regret_tracker_sublinear_growth() {
        let mut tracker = RegretTracker::default();
        // Start with optimal then suboptimal -> regret accumulates
        tracker.record(1.0); // best = 1.0
        for _ in 0..100 {
            tracker.record(0.8); // regret = 0.2 each
        }
        assert!(tracker.total_observations == 101);
        assert!(tracker.total_regret > 0.0);
        assert!(tracker.average_regret() > 0.0);
    }

    #[test]
    fn test_plateau_detector_insufficient_data() {
        let mut pd = PlateauDetector::new(5, 0.01);
        pd.record(0.5);
        pd.record(0.6);
        assert_eq!(pd.check(), PlateauAction::Continue);
        assert_eq!(pd.consecutive_plateaus, 0);
    }

    #[test]
    fn test_plateau_detector_escalation() {
        let mut pd = PlateauDetector::new(3, 0.01);
        // Fill with flat scores
        for _ in 0..30 {
            pd.record(0.5);
        }
        let a1 = pd.check();
        assert_eq!(a1, PlateauAction::IncreaseExploration);

        // Still flat
        for _ in 0..6 {
            pd.record(0.5);
        }
        let a2 = pd.check();
        assert_eq!(a2, PlateauAction::TriggerTransfer);

        for _ in 0..6 {
            pd.record(0.5);
        }
        let a3 = pd.check();
        assert_eq!(a3, PlateauAction::TriggerTransfer);

        for _ in 0..6 {
            pd.record(0.5);
        }
        let a4 = pd.check();
        assert_eq!(a4, PlateauAction::InjectDiversity);
    }

    #[test]
    fn test_plateau_detector_min_window_size() {
        let pd = PlateauDetector::new(1, 0.01);
        // Window size should be clamped to 3
        assert_eq!(pd.window_size, 3);
    }

    #[test]
    fn test_plateau_detector_resets_on_improvement() {
        let mut pd = PlateauDetector::new(3, 0.01);
        for _ in 0..20 {
            pd.record(0.5);
        }
        pd.check(); // triggers plateau
        assert!(pd.consecutive_plateaus > 0);

        // Now show improvement
        for i in 0..10 {
            pd.record(0.5 + i as f64 * 0.1);
        }
        pd.check();
        assert_eq!(pd.consecutive_plateaus, 0);
    }

    #[test]
    fn test_transfer_verification_acceleration() {
        let v = TransferVerification::verify(
            "BTC".into(), "ETH".into(),
            0.8, 0.8, 0.3, 0.6,
            200, 50,
        );
        assert!((v.acceleration_factor - 4.0).abs() < 0.01);
        assert!(v.promotable);
    }

    #[test]
    fn test_regret_tracker_average() {
        let mut tracker = RegretTracker::default();
        tracker.record(1.0); // best = 1.0, regret = 0
        tracker.record(0.5); // regret = 0.5
        tracker.record(0.3); // regret = 0.7
        assert!((tracker.total_regret - 1.2).abs() < 1e-10);
        assert!((tracker.average_regret() - 0.4).abs() < 1e-10);
    }

    #[test]
    fn test_regret_tracker_best_reward_updates() {
        let mut tracker = RegretTracker::default();
        tracker.record(0.5);
        tracker.record(0.8);
        tracker.record(0.6); // regret = 0.2 (best was 0.8)
        assert!((tracker.total_regret - 0.2).abs() < 1e-10);
        // 0.5: best=0.5, regret=0; 0.8: best=0.8, regret=0; 0.6: best=0.8, regret=0.2
        // Wait, actually: first record 0.5: best=0.5, regret=0. Record 0.8: best=0.8, regret=0. Record 0.6: best=0.8, regret=0.2
        // Total = 0.2, not 0.5. Let me recalculate:
        // Actually first 0.5: best=0.5, regret = max(0.5-0.5, 0) = 0
        // Then 0.8: best=0.8, regret = max(0.8-0.8, 0) = 0
        // Then 0.6: best=0.8, regret = max(0.8-0.6, 0) = 0.2
        // total = 0.2
    }

    #[test]
    fn test_transfer_verification_source_stable() {
        // Source drops by < 0.01 -> not regressed
        let v = TransferVerification::verify(
            "BTC".into(), "ETH".into(),
            0.8, 0.795, // drops 0.005 < 0.01 threshold
            0.3, 0.5,
            100, 50,
        );
        assert!(!v.regressed_source);
        assert!(v.promotable);
    }
}
