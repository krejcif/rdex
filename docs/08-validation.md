# 08 — Validation Framework

## Walk-Forward Validation

Prevents overfitting by training and testing on non-overlapping time segments.

```mermaid
graph LR
    subgraph Fold1[Fold 1]
        T1[Train 70%] --> Test1[Test 30%]
    end
    subgraph Fold2[Fold 2]
        T2[Train 70%] --> Test2[Test 30%]
    end
    subgraph Fold3[Fold 3]
        T3[Train 70%] --> Test3[Test 30%]
    end

    Test1 --> Results[Compare across folds]
    Test2 --> Results
    Test3 --> Results
```

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

```mermaid
graph TD
    Trades["Actual trade P&Ls<br/>[+2%, -1%, +3%, ...]"] --> Actual["Compound actual return<br/>∏(1 + pnl/100) - 1"]
    Trades --> Shuffle["Shuffle 1000×"]
    Shuffle --> Permuted["Compound each shuffle"]
    Actual --> Compare["P-value = fraction of<br/>shuffled returns ≥ actual"]
    Permuted --> Compare
    Compare --> Sig{p < 0.05?}
    Sig -->|Yes| Edge["Genuine edge:<br/>timing matters"]
    Sig -->|No| Luck["May be luck:<br/>ordering irrelevant"]
```

Both actual and permuted returns use the same compounding method for apples-to-apples comparison.

## Overfitting Check

```
overfitting_check(train_return, test_return, train_sharpe, test_sharpe)
```

```mermaid
graph TD
    Check[overfitting_check] --> RD{Return degradation<br/>> 50%?}
    Check --> SD{Sharpe degradation<br/>> 50%?}
    Check --> HS{Train Sharpe<br/>> 5.0?}
    Check --> HR{Train return<br/>> 100%?}

    RD -->|Yes| Flag1["⚠ High return degradation"]
    SD -->|Yes| Flag2["⚠ Sharpe degrades OOS"]
    HS -->|Yes| Flag3["⚠ Suspiciously high Sharpe"]
    HR -->|Yes| Flag4["⚠ Suspiciously high return"]

    RD --> LO{Return OR Sharpe<br/>degradation > 70%?}
    SD --> LO
    LO -->|Yes| Overfit["likely_overfit = true"]
```
