use super::engine::BacktestEngine;
use crate::domain::*;
use crate::engine::learner::{LearnerConfig, LearningEngine};

/// Walk-forward validation: train on window, test on next segment
pub fn walk_forward_validation(
    symbol: &Symbol,
    candles: &[Candle],
    config: &FuturesConfig,
    learner_config: &LearnerConfig,
    n_folds: usize,
    train_ratio: f64,
) -> ValidationResult {
    let total = candles.len();
    let fold_size = total / n_folds;
    let warmup = 60;

    if fold_size < warmup * 2 {
        return ValidationResult {
            fold_results: vec![],
            is_valid: false,
            reasons: vec!["Insufficient data for walk-forward validation".into()],
        };
    }

    let mut fold_results = Vec::new();

    for fold in 0..n_folds {
        let start = fold * fold_size;
        let end = (start + fold_size).min(total);
        let split = start + ((end - start) as f64 * train_ratio) as usize;

        if split <= start + warmup || end <= split + warmup {
            continue;
        }

        let train_data = &candles[start..split];
        let test_data = &candles[start..end]; // test includes train for indicator warmup

        // Train
        let mut learning = LearningEngine::new(vec![symbol.0.clone()], learner_config.clone());

        let mut train_bt = BacktestEngine::new(config.clone(), learning, fold as u64);
        let _train_result = train_bt.run(symbol, train_data, warmup);

        // Test on out-of-sample data (but starting from split index)
        // The learning engine carries forward what it learned in training
        learning = train_bt.learning;
        let mut test_bt = BacktestEngine::new(config.clone(), learning, fold as u64 + 1000);
        let test_result = test_bt.run(symbol, test_data, split - start);

        fold_results.push(FoldResult {
            fold,
            train_return: _train_result.performance.total_return_pct,
            test_return: test_result.performance.total_return_pct,
            test_sharpe: test_result.performance.sharpe_ratio,
            test_drawdown: test_result.performance.max_drawdown_pct,
            test_trades: test_result.performance.total_trades,
            test_win_rate: test_result.performance.win_rate,
        });
    }

    validate_results(&fold_results)
}

/// Monte Carlo permutation test: shuffle trade order to test significance
pub fn permutation_test(
    trade_pnls: &[f64],
    actual_return: f64,
    n_permutations: usize,
    seed: u64,
) -> PermutationResult {
    use rand::seq::SliceRandom;
    use rand::SeedableRng;

    if trade_pnls.is_empty() {
        return PermutationResult {
            actual_return,
            mean_random_return: 0.0,
            p_value: 1.0,
            is_significant: false,
            percentile: 50.0,
        };
    }

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut random_returns = Vec::with_capacity(n_permutations);

    for _ in 0..n_permutations {
        let mut shuffled = trade_pnls.to_vec();
        shuffled.shuffle(&mut rng);
        let cum_return: f64 = shuffled
            .iter()
            .fold(1.0, |acc, &pnl| acc * (1.0 + pnl / 100.0));
        random_returns.push((cum_return - 1.0) * 100.0);
    }

    random_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean_random = random_returns.iter().sum::<f64>() / random_returns.len() as f64;

    let better_count = random_returns
        .iter()
        .filter(|&&r| r >= actual_return)
        .count();
    let p_value = better_count as f64 / n_permutations as f64;

    let rank = random_returns
        .iter()
        .filter(|&&r| r < actual_return)
        .count();
    let percentile = rank as f64 / n_permutations as f64 * 100.0;

    PermutationResult {
        actual_return,
        mean_random_return: mean_random,
        p_value,
        is_significant: p_value < 0.05,
        percentile,
    }
}

/// Check for overfitting indicators
pub fn overfitting_check(
    train_return: f64,
    test_return: f64,
    train_sharpe: f64,
    test_sharpe: f64,
) -> OverfittingResult {
    let mut warnings = Vec::new();

    // Return degradation
    let return_degradation = if train_return > 0.0 {
        1.0 - test_return / train_return
    } else {
        0.0
    };

    if return_degradation > 0.5 {
        warnings.push("High return degradation (>50%) from train to test".into());
    }

    // Sharpe degradation
    let sharpe_degradation = if train_sharpe > 0.0 {
        1.0 - test_sharpe / train_sharpe
    } else {
        0.0
    };

    if sharpe_degradation > 0.5 {
        warnings.push("Sharpe ratio degrades >50% out-of-sample".into());
    }

    // Train too good to be true
    if train_sharpe > 5.0 {
        warnings.push(format!(
            "Suspiciously high train Sharpe: {:.2}",
            train_sharpe
        ));
    }

    if train_return > 100.0 {
        warnings.push(format!(
            "Suspiciously high train return: {:.1}%",
            train_return
        ));
    }

    OverfittingResult {
        return_degradation,
        sharpe_degradation,
        warnings,
        likely_overfit: return_degradation > 0.7 || sharpe_degradation > 0.7,
    }
}

fn validate_results(folds: &[FoldResult]) -> ValidationResult {
    if folds.is_empty() {
        return ValidationResult {
            fold_results: vec![],
            is_valid: false,
            reasons: vec!["No folds completed".into()],
        };
    }

    let mut reasons = Vec::new();

    // Check if majority of folds are profitable
    let profitable_folds = folds.iter().filter(|f| f.test_return > 0.0).count();
    if profitable_folds < folds.len() / 2 {
        reasons.push(format!(
            "Only {}/{} folds profitable out-of-sample",
            profitable_folds,
            folds.len()
        ));
    }

    // Check for consistent returns across folds
    let returns: Vec<f64> = folds.iter().map(|f| f.test_return).collect();
    let mean_return = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance = returns
        .iter()
        .map(|r| (r - mean_return).powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let cv = if mean_return.abs() > 0.01 {
        variance.sqrt() / mean_return.abs()
    } else {
        f64::INFINITY
    };

    if cv > 2.0 {
        reasons.push(format!("High variance across folds (CV={:.2})", cv));
    }

    // Check for train/test degradation
    let avg_degradation: f64 = folds
        .iter()
        .filter(|f| f.train_return > 0.0)
        .map(|f| 1.0 - f.test_return / f.train_return)
        .sum::<f64>()
        / folds.len().max(1) as f64;

    if avg_degradation > 0.5 {
        reasons.push(format!(
            "Avg return degradation: {:.0}%",
            avg_degradation * 100.0
        ));
    }

    let is_valid = reasons.is_empty();

    ValidationResult {
        fold_results: folds.to_vec(),
        is_valid,
        reasons,
    }
}

#[derive(Debug, Clone)]
pub struct FoldResult {
    pub fold: usize,
    pub train_return: f64,
    pub test_return: f64,
    pub test_sharpe: f64,
    pub test_drawdown: f64,
    pub test_trades: usize,
    pub test_win_rate: f64,
}

#[derive(Debug)]
pub struct ValidationResult {
    pub fold_results: Vec<FoldResult>,
    pub is_valid: bool,
    pub reasons: Vec<String>,
}

#[derive(Debug)]
pub struct PermutationResult {
    pub actual_return: f64,
    pub mean_random_return: f64,
    pub p_value: f64,
    pub is_significant: bool,
    pub percentile: f64,
}

#[derive(Debug)]
pub struct OverfittingResult {
    pub return_degradation: f64,
    pub sharpe_degradation: f64,
    pub warnings: Vec<String>,
    pub likely_overfit: bool,
}

impl ValidationResult {
    pub fn print_summary(&self) {
        println!("\n--- Walk-Forward Validation ---");
        println!("Valid: {}", self.is_valid);
        for fold in &self.fold_results {
            println!(
                "  Fold {}: train={:+.2}% test={:+.2}% sharpe={:.2} dd={:.1}% trades={} wr={:.0}%",
                fold.fold,
                fold.train_return,
                fold.test_return,
                fold.test_sharpe,
                fold.test_drawdown,
                fold.test_trades,
                fold.test_win_rate * 100.0,
            );
        }
        if !self.reasons.is_empty() {
            println!("  Warnings:");
            for r in &self.reasons {
                println!("    - {}", r);
            }
        }
    }
}

impl PermutationResult {
    pub fn print_summary(&self) {
        println!("\n--- Permutation Test ---");
        println!("  Actual return:       {:+.2}%", self.actual_return);
        println!("  Mean random return:  {:+.2}%", self.mean_random_return);
        println!("  P-value:             {:.4}", self.p_value);
        println!("  Significant (p<0.05): {}", self.is_significant);
        println!("  Percentile:          {:.1}%", self.percentile);
    }
}

impl OverfittingResult {
    pub fn print_summary(&self) {
        println!("\n--- Overfitting Check ---");
        println!(
            "  Return degradation:  {:.0}%",
            self.return_degradation * 100.0
        );
        println!(
            "  Sharpe degradation:  {:.0}%",
            self.sharpe_degradation * 100.0
        );
        println!("  Likely overfit:      {}", self.likely_overfit);
        for w in &self.warnings {
            println!("  WARNING: {}", w);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permutation_test() {
        let pnls = vec![2.0, -1.0, 3.0, -0.5, 1.5, -1.0, 2.0, -0.5, 1.0, 0.5];
        let actual_return: f64 = pnls.iter().sum();
        let result = permutation_test(&pnls, actual_return, 1000, 42);
        assert!(result.p_value >= 0.0 && result.p_value <= 1.0);
    }

    #[test]
    fn test_overfitting_check_good() {
        let result = overfitting_check(10.0, 8.0, 2.0, 1.8);
        assert!(!result.likely_overfit);
    }

    #[test]
    fn test_overfitting_check_bad() {
        let result = overfitting_check(50.0, 5.0, 5.5, 0.5);
        assert!(result.likely_overfit);
        assert!(!result.warnings.is_empty());
    }

    #[test]
    fn test_permutation_empty_trades() {
        let result = permutation_test(&[], 0.0, 1000, 42);
        assert_eq!(result.p_value, 1.0);
        assert!(!result.is_significant);
    }

    #[test]
    fn test_permutation_all_positive() {
        let pnls = vec![5.0, 3.0, 7.0, 2.0, 4.0];
        let actual: f64 = pnls.iter().sum();
        let result = permutation_test(&pnls, actual, 500, 42);
        // With all positive PnLs, any shuffle gives same sum -> not significant
        assert!(result.p_value > 0.01);
    }

    #[test]
    fn test_permutation_percentile_in_range() {
        let pnls = vec![2.0, -1.0, 3.0, -0.5, 1.5];
        let result = permutation_test(&pnls, 10.0, 500, 42);
        assert!(result.percentile >= 0.0 && result.percentile <= 100.0);
    }

    #[test]
    fn test_overfitting_negative_train_return() {
        let result = overfitting_check(-5.0, -3.0, 1.0, 0.5);
        // Negative train return -> return_degradation = 0
        assert_eq!(result.return_degradation, 0.0);
    }

    #[test]
    fn test_overfitting_suspicious_values() {
        let result = overfitting_check(150.0, 140.0, 6.0, 5.5);
        assert!(!result.warnings.is_empty());
        let has_sharpe_warning = result.warnings.iter().any(|w| w.contains("Sharpe"));
        let has_return_warning = result.warnings.iter().any(|w| w.contains("return"));
        assert!(has_sharpe_warning, "Should warn about high Sharpe");
        assert!(has_return_warning, "Should warn about high return");
    }

    #[test]
    fn test_validate_empty_folds() {
        let result = validate_results(&[]);
        assert!(!result.is_valid);
        assert!(result.reasons.iter().any(|r| r.contains("No folds")));
    }

    #[test]
    fn test_validate_all_profitable() {
        let folds = vec![
            FoldResult { fold: 0, train_return: 5.0, test_return: 4.0, test_sharpe: 1.5, test_drawdown: 3.0, test_trades: 20, test_win_rate: 0.6 },
            FoldResult { fold: 1, train_return: 6.0, test_return: 5.0, test_sharpe: 1.8, test_drawdown: 2.5, test_trades: 25, test_win_rate: 0.65 },
        ];
        let result = validate_results(&folds);
        assert!(result.is_valid);
        assert!(result.reasons.is_empty());
    }

    #[test]
    fn test_validate_majority_unprofitable() {
        let folds = vec![
            FoldResult { fold: 0, train_return: 10.0, test_return: -5.0, test_sharpe: -0.5, test_drawdown: 10.0, test_trades: 20, test_win_rate: 0.3 },
            FoldResult { fold: 1, train_return: 8.0, test_return: -3.0, test_sharpe: -0.2, test_drawdown: 8.0, test_trades: 15, test_win_rate: 0.35 },
            FoldResult { fold: 2, train_return: 12.0, test_return: 1.0, test_sharpe: 0.5, test_drawdown: 5.0, test_trades: 18, test_win_rate: 0.55 },
        ];
        let result = validate_results(&folds);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_validate_high_degradation() {
        let folds = vec![
            FoldResult { fold: 0, train_return: 50.0, test_return: 5.0, test_sharpe: 0.5, test_drawdown: 10.0, test_trades: 30, test_win_rate: 0.5 },
            FoldResult { fold: 1, train_return: 40.0, test_return: 3.0, test_sharpe: 0.3, test_drawdown: 12.0, test_trades: 25, test_win_rate: 0.45 },
        ];
        let result = validate_results(&folds);
        assert!(!result.is_valid);
        assert!(result.reasons.iter().any(|r| r.contains("degradation")));
    }

    #[test]
    fn test_validate_high_variance() {
        let folds = vec![
            FoldResult { fold: 0, train_return: 5.0, test_return: 50.0, test_sharpe: 2.0, test_drawdown: 5.0, test_trades: 30, test_win_rate: 0.7 },
            FoldResult { fold: 1, train_return: 5.0, test_return: -40.0, test_sharpe: -1.0, test_drawdown: 20.0, test_trades: 25, test_win_rate: 0.3 },
            FoldResult { fold: 2, train_return: 5.0, test_return: 2.0, test_sharpe: 0.2, test_drawdown: 8.0, test_trades: 35, test_win_rate: 0.55 },
        ];
        let result = validate_results(&folds);
        assert!(!result.is_valid);
    }

    #[test]
    fn test_overfitting_zero_train() {
        let result = overfitting_check(0.0, 5.0, 0.0, 1.0);
        assert_eq!(result.return_degradation, 0.0);
        assert_eq!(result.sharpe_degradation, 0.0);
    }

    #[test]
    fn test_overfitting_both_degradations() {
        let result = overfitting_check(100.0, 10.0, 4.0, 0.5);
        assert!(result.return_degradation > 0.5);
        assert!(result.sharpe_degradation > 0.5);
        assert!(result.likely_overfit);
    }
}
