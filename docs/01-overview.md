# 01 — System Overview

RDEX is a self-learning crypto trading system that discovers profitable patterns from Binance Futures 15-minute candle data using Thompson Sampling with cross-symbol transfer learning.

## Core Philosophy

- **ZERO constants, ZERO hardcoded values.** Every trading parameter — position sizing, stop loss, take profit, hold duration, cooldown, trailing stops — is learned from observed trade outcomes. No thresholds, no magic numbers, no manually tuned constants anywhere in strategy or risk logic. If it influences a trade, it must come from data.
- **Maximize profits through better learning.** Improve the reward signal, enrich features, sharpen pattern recognition, and refine position sizing. The system discovers profitable patterns through Thompson Sampling — never add hand-crafted conditional trading rules.
- **Prevent losses through adaptive risk management.** SL/TP derive from excursion EMAs, cooldowns from win/loss duration ratios, sizing from Kelly criterion, hold limits from observed trade durations. The system learns to cut losses and hold winners from its own experience.
- **Anti-look-ahead enforcement.** The backtest engine asserts at runtime that no future data leaks into decisions. `MarketState` at candle `i` only contains candles `[0..=i]`.
- **Cross-domain transfer.** Patterns learned on high-liquidity symbols (BTC) accelerate learning on smaller ones (DOGE, AVAX) through dampened prior transfer.
- **Validate everything.** Strategy changes must demonstrate improvement in backtest metrics (return, Sharpe, drawdown). Walk-forward validation and permutation tests guard against overfitting.

## Two-Phase Backtest

```
Phase 1: Sequential Learning
  One shared LearningEngine processes symbols in order.
  BTC → ETH → SOL → BNB → XRP → DOGE → AVAX → LINK
  Each symbol's patterns accumulate into the shared Thompson engine.

Phase 2: Exploitation
  Re-run all symbols with the fully trained engine.
  Walk-forward validation (3 folds) + Monte Carlo permutation tests.
```

## CLI

```
rdex backtest -d 30 -e 10000 -l 3 -s BTCUSDT,ETHUSDT
rdex fetch -d 30 -s BTCUSDT,ETHUSDT
```

| Flag | Default | Description |
|------|---------|-------------|
| `-d` | 30 | Days of historical data |
| `-e` | 10000 | Starting equity ($) |
| `-l` | 3 | Leverage multiplier |
| `-s` | 8 major pairs | Comma-separated symbol list |

## Data Pipeline

```
Binance FAPI (public, no key) → 15m OHLCV + funding rates (8h)
                                      ↓
                              CSV cache in data/ (1h TTL)
                                      ↓
                              BacktestEngine sequential processing
```
