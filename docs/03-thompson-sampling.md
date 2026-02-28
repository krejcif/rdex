# 03 — Thompson Sampling Engine

## Overview

The system uses Thompson Sampling to select between three actions (`long`, `short`, `hold`) for each market pattern. This is a contextual bandit with decaying evidence for non-stationary markets.

## Key Types

### `BetaParams`
Standard Beta distribution parameterized by `alpha` (successes) and `beta` (failures). Starts at `Uniform(1, 1)`. Sampling uses the `statrs` crate for mathematically correct Beta distribution sampling.

### `DecayingBeta`
Wraps `BetaParams` with an exponential decay factor (default 0.995). Before each update, old evidence is decayed toward the prior:

```
alpha = 1 + (alpha - 1) * decay_factor
beta  = 1 + (beta  - 1) * decay_factor
```

This gives an effective observation window of `1 / (1 - decay_factor)` ≈ 200 trades.

### `ThompsonEngine`
Three-level nested HashMap: `symbol → MarketContext → StrategyId → DecayingBeta`

**Global pooling:** All symbols share a single Thompson state under the `_global` key. Crypto markets have similar microstructure, so patterns learned on BTC apply to ETH.

## Arm Selection

```mermaid
graph TD
    A[For each action:<br/>long, short, hold] --> B["sample = Beta(alpha, beta).sample()"]
    B --> C["bonus = exploration × √(ln(total_decisions) / visits)"]
    C --> D["score = sample + bonus"]
    D --> E{Highest score?}
    E -->|long| F[Go Long]
    E -->|short| G[Go Short]
    E -->|hold| H[Hold]
```

The curiosity bonus is UCB-like and decays as each arm accumulates visits.

## Hold Bias

New arms start with `Uniform(1, 1)` except `hold`, which gets an inflated alpha:

```
hold_alpha = 1 + 4 / (1 + total_decisions / 100)
```

This creates a conservative bias early on that naturally decays as the engine gains experience.

## Reward Signal

Trade P&L is converted to `[0, 1]` via adaptive sigmoid:

```mermaid
graph LR
    PnL[Trade P&L %] --> Sigmoid["reward = 1 / (1 + exp(-k × pnl_pct))"]
    Sigmoid --> Reward["[0, 1] reward"]
    K["k = 1 / pnl_std<br/>(from AdaptiveParams)"] --> Sigmoid
```

Where `k = 1 / pnl_std` (adaptive, from `AdaptiveParams`). Small trades → steep sigmoid. Large trades → gentle gradient.

## Transfer Learning

When a source symbol has accumulated enough evidence (>10 effective observations per arm):

```mermaid
graph LR
    Source[Source symbol<br/>rich evidence] -->|Extract priors| Priors["HashMap<Context,<br/>HashMap<Strategy, Beta>>"]
    Priors -->|Dampen| Dampened["α' = 1 + √(α - 1)<br/>β' = 1 + √(β - 1)"]
    Dampened -->|Seed empty slots| Target[Target symbol<br/>sparse evidence]
```

Dampening prevents source symbol's strong convictions from overwhelming the target's local learning.
