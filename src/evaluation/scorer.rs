/// Convert trade P&L to a [0, 1] reward for Thompson Sampling.
/// k is adaptive — derived from observed PnL standard deviation.
/// When trades are small (low variance), k is high → sharp signal.
/// When trades are large (high variance), k is low → more gradient.
pub fn pnl_to_reward(pnl_pct: f64, adaptive_k: f64) -> f64 {
    let k = adaptive_k.clamp(0.5, 5.0);
    1.0 / (1.0 + (-k * pnl_pct).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pnl_to_reward_positive() {
        assert!(pnl_to_reward(5.0, 2.0) > 0.7);
        assert!(pnl_to_reward(10.0, 2.0) > 0.9);
    }

    #[test]
    fn test_pnl_to_reward_negative() {
        assert!(pnl_to_reward(-5.0, 2.0) < 0.3);
        assert!(pnl_to_reward(-10.0, 2.0) < 0.1);
    }

    #[test]
    fn test_pnl_to_reward_zero() {
        assert!((pnl_to_reward(0.0, 2.0) - 0.5).abs() < 0.01);
    }

    #[test]
    fn test_k_clamped_low() {
        // k below 0.5 should be clamped to 0.5
        let r1 = pnl_to_reward(1.0, 0.1);
        let r2 = pnl_to_reward(1.0, 0.5);
        assert!((r1 - r2).abs() < 1e-10, "k=0.1 should clamp to 0.5");
    }

    #[test]
    fn test_k_clamped_high() {
        // k above 5.0 should be clamped to 5.0
        let r1 = pnl_to_reward(1.0, 10.0);
        let r2 = pnl_to_reward(1.0, 5.0);
        assert!((r1 - r2).abs() < 1e-10, "k=10 should clamp to 5.0");
    }

    #[test]
    fn test_reward_always_in_zero_one() {
        for pnl in [-100.0, -10.0, -1.0, 0.0, 1.0, 10.0, 100.0] {
            for k in [0.5, 1.0, 2.0, 5.0] {
                let r = pnl_to_reward(pnl, k);
                assert!(r >= 0.0 && r <= 1.0, "Reward {} out of range for pnl={}, k={}", r, pnl, k);
            }
        }
    }

    #[test]
    fn test_reward_monotonicity() {
        // Higher PnL should give higher reward
        let r1 = pnl_to_reward(1.0, 2.0);
        let r2 = pnl_to_reward(5.0, 2.0);
        let r3 = pnl_to_reward(10.0, 2.0);
        assert!(r1 < r2 && r2 < r3, "Reward should increase with PnL");
    }

    #[test]
    fn test_reward_symmetry() {
        let r_pos = pnl_to_reward(5.0, 2.0);
        let r_neg = pnl_to_reward(-5.0, 2.0);
        // Sigmoid is symmetric: f(x) + f(-x) = 1
        assert!((r_pos + r_neg - 1.0).abs() < 0.01, "Sigmoid should be symmetric");
    }

    #[test]
    fn test_extreme_positive_near_one() {
        assert!(pnl_to_reward(100.0, 5.0) > 0.999);
    }

    #[test]
    fn test_extreme_negative_near_zero() {
        assert!(pnl_to_reward(-100.0, 5.0) < 0.001);
    }

    #[test]
    fn test_k_sensitivity() {
        // Higher k -> steeper sigmoid -> bigger spread
        let low_k = pnl_to_reward(1.0, 0.5);
        let high_k = pnl_to_reward(1.0, 5.0);
        assert!(high_k > low_k, "Higher k should push positive PnL closer to 1.0");
    }
}
