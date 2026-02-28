# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Mission

RDEX is a **fully autonomous, self-improving crypto trading system**. The ultimate goal is a system that continuously adapts to evolving market conditions and improves its own profitability without human intervention.

**Everything in this codebase serves one purpose: maximize profits and prevent losses through self-learning.**

The system must:
1. **Discover** profitable patterns from market data — not be told what they are
2. **Adapt** when markets change — parameters self-adjust from recent trade outcomes
3. **Transfer** knowledge across symbols — what works on BTC accelerates learning on alts
4. **Protect** capital — risk management tightens automatically when performance degrades
5. **Validate** itself — walk-forward tests and permutation tests reject overfitting

### Autonomous Improvement Loop

When working on this codebase, follow this loop to improve the system:

```
1. MEASURE  → Run backtest, capture current metrics (return, Sharpe, drawdown, win rate)
2. DIAGNOSE → Identify weakest metric or biggest loss source (pattern, symbol, exit reason)
3. IMPROVE  → Enhance the learning mechanism that addresses the weakness:
              - Weak pattern recognition → richer features, better discretization
              - Poor entry timing → sharper reward signal, faster Thompson convergence
              - Large losses → tighter adaptive SL from excursion data
              - Missed profits → better trailing stop adaptation, longer holds for winners
              - Slow learning → improved transfer learning, better KNN predictions
4. VALIDATE → Run backtest again. Metrics must improve. Walk-forward must stay valid.
5. OVERFIT? → Check for overfitting: walk-forward degradation < 50%? Permutation p < 0.05?
              Consistent across symbols? Train/test gap reasonable? If ANY check fails → REVERT.
6. REVERT   → If metrics don't improve OR overfitting detected, undo the change. No exceptions.
7. REPEAT   → Go to step 1.
```

**Critical: every improvement must work through the adaptive/learning mechanisms.** Never add hardcoded rules or constants. The system improves by learning better, not by being told what to do.

**Critical: every change must be checked for overfitting.** No matter what you're modifying — features, indicators, learning logic, risk management, reward signals, even test infrastructure — ask: "Could this make the backtest look better without actually generalizing?" If yes, it's overfitting. See the Overfitting Prevention section below.

## Build & Test Commands

```bash
cargo build                                    # Debug build
cargo build --release                          # Release build (LTO enabled)
cargo test --lib                               # Run all unit tests
cargo test --lib test_name                     # Run a single test
cargo test --lib -- --nocapture                # Tests with stdout output
cargo clippy                                   # Lint
cargo fmt -- --check                           # Format check
cargo run --bin rdex -- backtest -d 30 -e 10000 # Run backtest (fetches data from Binance)
cargo run --bin rdex -- fetch -d 30             # Pre-fetch and cache market data
```

## Architecture

**Self-learning crypto trading system** using Thompson Sampling on Binance Futures 15m candles. Single binary with `backtest` and `fetch` subcommands.

### Module Responsibilities

- **`domain/`** — Core types (`Candle`, `Symbol`, `MarketState`, `Position`, `FuturesConfig`), indicator computation (SMA, EMA, RSI, ATR, MACD, Bollinger, ADX). No business logic.
- **`strategy/`** — Feature extraction (`MarketFeatures`) and pattern discretization. `PatternDiscretizer` uses adaptive median-split binning to produce 16 market patterns (4 features × 2 bins).
- **`engine/`** — Decision-making core. `LearningEngine` orchestrates Thompson Sampling (`DecayingBeta` arms per symbol/context/strategy). `AdaptiveParams` derives all trading parameters (Kelly sizing, SL/TP, hold limits, cooldowns) from observed trade outcomes. `TradeManager` handles position lifecycle (entry/exit, trailing stops, excursion tracking). `PatternLibrary` provides KNN-based directional probability predictions from temporal feature vectors. `TransferLearning` copies learned arms between symbols when source plateaus.
- **`backtest/`** — `BacktestEngine` feeds candles sequentially with anti-look-ahead enforcement. Signal flips (direction reversals) are blocked during the adaptive `min_hold` period to prevent noisy Thompson samples from causing premature exits. `Portfolio` tracks equity with Binance Futures mechanics (leverage, slippage, fees, funding, liquidation). `validation` provides walk-forward cross-validation and Monte Carlo permutation tests.
- **`evaluation/`** — Performance metrics (Sharpe, Sortino, Calmar, max drawdown, profit factor) and reward scoring with adaptive sigmoid.
- **`data/`** — Async Binance Futures API fetcher with CSV caching.

### Execution Flow

Two-phase backtest: **Phase 1** runs symbols sequentially through one shared `LearningEngine` (patterns from BTC transfer to alts). **Phase 2** re-runs all symbols with the trained engine, then validates with walk-forward folds and permutation tests.

### Critical Invariant: Anti-Look-Ahead

`BacktestEngine` enforces that `MarketState` at candle `i` only contains data from candles `0..=i`. The `max_seen_index` field monotonically increases. Any code touching the backtest loop **must** preserve this — assertions will fire on violations. Indicators are computed from closed candle history only.

### ABSOLUTE RULE: Zero Constants, Zero Hardcoded Values

This is the single most important design constraint in the entire codebase.

**Every trading parameter must be learned from data.** There are NO exceptions. No hardcoded thresholds, no magic numbers, no manually picked constants anywhere in strategy or risk logic. If a value influences trading decisions, it must be derived from observed trade outcomes through `AdaptiveParams` EMAs, excursion statistics, win/loss ratios, or Thompson Sampling.

**Violations** — these are NEVER acceptable:
- `if rsi > 70` — hardcoded threshold
- `stop_loss = 2%` — hardcoded risk parameter
- `if trend == "uptrend" { go_long() }` — hardcoded trading rule
- `cooldown = 10` — hardcoded timing
- `position_size = 0.1` — hardcoded sizing
- Any `const`, literal number, or fixed value that controls when/how/how much to trade

**Correct approach** — derive everything from data:
- Stop loss → from `ema_adverse` (observed worst excursions)
- Take profit → from `ema_favorable` (observed best excursions)
- Position size → from Kelly criterion (observed win rate × reward/risk ratio)
- Cooldown → from `ema_win_duration` × loss rate (trade less when losing)
- Hold limits → from observed win/loss durations
- Entry/exit signals → from Thompson Sampling (bandit learns pattern→action from rewards)

### Strategy Development Rules

All strategy work **must** focus on two goals: **increase profits** and **prevent losses** — exclusively through the adaptive, self-learning approach.

1. **Profits come from better learning, not more rules.** Improve the reward signal, add richer features, refine pattern recognition, sharpen the Thompson Sampling feedback loop. Never add conditional trading branches.
2. **Loss prevention comes from adaptive risk management.** SL/TP, trailing stops, cooldowns, hold limits, and sizing all self-adjust from trade statistics. The system learns to cut losses faster and hold winners longer from its own experience.
3. **The system discovers what works — you don't tell it.** Thompson Sampling explores pattern→action mappings and reinforces what produces profit. Your job is to give it better inputs and faster feedback, not to prescribe rules.
4. **Validate every change with backtest metrics.** Run `cargo run -- backtest` and confirm: higher total return, better Sharpe/Sortino, lower max drawdown, or higher profit factor. Walk-forward validation must remain valid. If metrics don't improve, revert.

### CRITICAL: Overfitting Prevention — Applies to ALL Development

Overfitting is the #1 threat to this system. A backtest that looks amazing but fails on unseen data is worthless. **Every change to any part of the codebase** — features, indicators, patterns, learning, risk management, position sizing, reward signals, transfer logic, backtest mechanics — must be evaluated with overfitting in mind. This is not just a strategy concern; it applies to ALL development work.

**Overfitting red flags — STOP and investigate if you see any of these:**
- Backtest Sharpe > 5.0 or total return > 100% — suspiciously good, likely curve-fitted
- Walk-forward test return degrades > 50% from train return — the model memorized, didn't learn
- Permutation test p-value > 0.05 — trade ordering doesn't matter, edge may be illusory
- Adding complexity improves in-sample but walk-forward validation fails — textbook overfit
- A change helps one symbol dramatically but hurts others — fitting noise, not signal
- Very high win rate (>70%) with few trades — small sample, unreliable

**Mandatory overfitting checks for every change:**
1. Walk-forward validation must remain valid (majority of folds profitable, low CV, <50% degradation)
2. Permutation test p-value must stay < 0.05
3. Results must be consistent across symbols — not just one outlier carrying the portfolio
4. Compare train vs test performance — large gaps mean overfitting
5. Prefer simpler changes — more parameters = more overfitting risk

**Design principles that prevent overfitting:**
- Small pattern space (16 patterns) — each pattern gets enough data to learn reliably
- Decaying Beta (window ≈ 200 trades) — old evidence fades, preventing stale memorization
- Adaptive median-split boundaries — bins shift with the market, not fixed to historical data
- Thompson Sampling exploration — never fully commits, always tests alternatives
- Transfer learning dampening — sqrt() prevents source convictions from overwhelming targets

**When in doubt, choose the simpler approach.** A robust system that makes 10% reliably beats a fragile one that makes 50% in backtest and loses money live.

### Known Failed Approaches (Do NOT Retry)

These were tested empirically and degraded performance. Do not attempt them again:

- **Asymmetric reward function** — penalizing losses harder in the sigmoid reward made Thompson too conservative, reducing returns by 26%
- **Adaptive min_edge threshold** — requiring Thompson action mean to exceed hold mean (or neutral 0.5) by a margin killed all trading. Thompson arm means start at 0.5 during cold start, so any edge threshold blocks exploration entirely. This conflicts with Thompson's stochastic sampling mechanism.
- **Tighter cold-start excursion defaults** — reducing initial SL from 2.5 to 1.8 ATR caused premature stop-outs, reducing returns by 39%. The wider initial SL is needed to give trades room during cold start.
- **Per-symbol Thompson Sampling pools** — tested and degraded performance (~230 trades/symbol is too sparse for independent learning)
- **KNN-Thompson directional agreement filter** — looked great in one run but was fitting noise
- **KNN-based position sizing blend** — 35% worse than pure Kelly sizing
- **Pure gradient sizing (no confidence filter)** — 72% return decrease
- **Confidence-based minimum threshold at 0.505** — increased activity but didn't generalize

## Tests

All tests are inline `#[cfg(test)]` modules within their source files (290 tests across 17 modules). Tests use `approx` for float comparisons and `tempfile` for I/O tests.

## Documentation

Detailed docs in `docs/` (numbered for reading order):

| Doc | Topic | Key content |
|-----|-------|-------------|
| [01-overview](docs/01-overview.md) | Mission & overview | Autonomous self-improvement goal, feedback loops, philosophy, CLI |
| [02-architecture](docs/02-architecture.md) | Architecture | Module map, data flow diagram, responsibility boundaries |
| [03-thompson-sampling](docs/03-thompson-sampling.md) | Thompson Sampling | DecayingBeta, arm selection, hold bias, reward signal, transfer |
| [04-adaptive-parameters](docs/04-adaptive-parameters.md) | Adaptive params | All EMA statistics and derived formulas (cooldown, Kelly, trailing) |
| [05-pattern-recognition](docs/05-pattern-recognition.md) | Patterns | 7 features, median-split binning (16 patterns), excursion tracking |
| [06-backtest-engine](docs/06-backtest-engine.md) | Backtest engine | Anti-look-ahead guarantees, per-candle loop, position lifecycle |
| [07-portfolio-and-risk](docs/07-portfolio-and-risk.md) | Portfolio & risk | Fee/slippage/liquidation model, P&L calc, TradeRecord fields |
| [08-validation](docs/08-validation.md) | Validation | Walk-forward, Monte Carlo permutation test, overfitting checks |
| [09-transfer-learning](docs/09-transfer-learning.md) | Transfer learning | RegretTracker, PlateauDetector, transfer process |
| [10-indicators](docs/10-indicators.md) | Indicators | All 14 technical indicators, regime classification |
