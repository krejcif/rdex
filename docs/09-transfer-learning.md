# 09 — Cross-Symbol Transfer Learning

## Motivation

Crypto markets share similar microstructure. A pattern like "high momentum + low volatility → go long" discovered on BTC likely applies to ETH. Transfer learning accelerates convergence on data-poor symbols.

## Components

### RegretTracker

Tracks cumulative regret per symbol:
```
regret_i = best_reward_seen - reward_i
total_regret = Σ regret_i
```

**Growth rate** = `ln(total_regret) / ln(observations)`. Sublinear growth (< 1.0) means the system is learning — regret grows slower than linearly because the engine makes better decisions over time.

### PlateauDetector

Compares recent reward window to prior window. If improvement < threshold for consecutive windows:

```mermaid
graph LR
    Detect[PlateauDetector] --> P1{1 plateau}
    P1 --> Explore[IncreaseExploration]
    Detect --> P2{2-3 plateaus}
    P2 --> Transfer[TriggerTransfer]
    Detect --> P3{4+ plateaus}
    P3 --> Diversity[InjectDiversity]
```

Window size: 5, threshold: 0.005 (from `LearnerConfig`).

## Transfer Process

```mermaid
graph TD
    Start[maybe_transfer after each symbol] --> Rank[Rank symbols by evidence<br/>total observations]
    Rank --> Source[Source = most evidence<br/>must exceed 150 obs]
    Source --> Loop["For each target with<br/>< half source evidence"]
    Loop --> Check{Plateau detected<br/>OR < 50 obs?}
    Check -->|Yes| Extract["Extract significant priors<br/>(evidence > 10)"]
    Check -->|No| Skip[Skip target]
    Extract --> Dampen["Dampen:<br/>α' = 1 + √(α - 1)<br/>β' = 1 + √(β - 1)"]
    Dampen --> Insert["Insert into target<br/>(only if slot empty)"]
```

`LearningEngine::maybe_transfer()` runs after each symbol's training:

1. Rank symbols by evidence (total observations)
2. Source = symbol with most evidence (must exceed `transfer_after_n` = 150)
3. For each target with less than half the source's evidence:
   - If plateau detected OR very few observations (<50): transfer
4. `ThompsonEngine::transfer_priors(source, target)`:
   - Extract significant priors (evidence > 10)
   - Dampen: `alpha' = 1 + sqrt(alpha - 1)`
   - Insert into target (only if slot empty — never overwrite local learning)

## Verification

`TransferVerification` evaluates after transfer:
- `improved_target`: target performance improved
- `regressed_source`: source performance degraded (threshold: -0.01)
- `promotable`: improved target AND no source regression
- `acceleration_factor`: baseline_cycles / transfer_cycles

## Symbol Processing Order

Symbols are processed in CLI order. Recommended: start with highest-liquidity symbols (BTC, ETH) so they become transfer sources for smaller alts.
