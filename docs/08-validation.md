# 08 — Validation Framework

## Walk-Forward Validation

Prevents overfitting by training and testing on non-overlapping time segments.

```
walk_forward_validation(symbol, candles, config, learner_config, n_folds=3, train_ratio=0.7)
```

For each fold:
1. Split data: 70% train, 30% test
2. Train a fresh `LearningEngine` on the training segment
3. Run the trained engine on the test segment (test includes train prefix for indicator warmup)
4. Record: train return, test return, test Sharpe, test drawdown, test trades, test win rate

### Validity Criteria

A validation is deemed **valid** if:
- Majority of folds are profitable out-of-sample
- Low coefficient of variation across folds (CV < 2.0)
- Average return degradation from train to test < 50%

## Monte Carlo Permutation Test

Tests whether trade ordering matters (i.e., whether the strategy has genuine edge vs. luck).

```
permutation_test(trade_pnls, actual_return, n_permutations=1000, seed=42)
```

1. Compute actual return by compounding trade P&L percentages: `∏(1 + pnl/100) - 1`
2. Shuffle the trade P&L sequence 1000 times
3. Compute cumulative return for each shuffle (same compounding formula)
4. P-value = fraction of shuffled returns ≥ actual return
5. Significant if p < 0.05

Both actual and permuted returns use the same compounding method for apples-to-apples comparison.

**Interpretation:**
- p < 0.05: The trade *ordering* (timing) contributes significantly — genuine market edge
- p ≥ 0.05: Returns could come from any random ordering — may be luck

## Overfitting Check

```
overfitting_check(train_return, test_return, train_sharpe, test_sharpe)
```

Flags:
- Return degradation > 50%: "High return degradation from train to test"
- Sharpe degradation > 50%: "Sharpe ratio degrades >50% out-of-sample"
- Train Sharpe > 5.0: "Suspiciously high train Sharpe"
- Train return > 100%: "Suspiciously high train return"
- `likely_overfit = true` if return OR Sharpe degradation > 70%
