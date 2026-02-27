# 04 — Adaptive Parameter System

## Design Principle

**Zero constants. Zero hardcoded values.** Every trading parameter is derived from exponential moving averages (EMAs) of observed trade outcomes. Relationships between statistics drive everything. When adding new parameters, they must be derived from observed data — never pick a number. Clamp ranges exist only as safety bounds, not as tuning knobs.

## Core Statistics (EMAs)

| Statistic | Tracks | Initial Value |
|-----------|--------|---------------|
| `ema_pnl` | Mean trade P&L (%) | 0.0 |
| `ema_pnl_sq` | Squared P&L (variance) | 4.0 |
| `ema_win_duration` | Candles held for winners | 15.0 |
| `ema_loss_duration` | Candles held for losers | 8.0 |
| `ema_favorable` | Max favorable excursion (ATR) | 4.0 |
| `ema_adverse` | Max adverse excursion (ATR) | 2.5 |
| `ema_rr_ratio` | Reward/risk = favorable/adverse | 1.6 |
| `ema_decay` | Self-adapting decay factor | 0.92 |

The EMA decay factor itself adapts: `decay = 1 - 1 / (sqrt(total_trades) + 5)`. More trades → slower decay → more stable estimates.

## Derived Parameters

### Cooldown (candles between trades)
```
cooldown = win_duration * (1 - recent_wr) / recent_wr
```
Win more → trade more. Lose more → trade less. Clamped to [8, 40].

### Min Hold (minimum candles before SL/TP active)
```
min_hold = sqrt(win_duration * loss_duration)
```
Geometric mean balances giving winners time vs cutting losers. Clamped to [3, 20].

### Max Hold (forced exit)
```
max_hold = win_duration / (1 - win_rate)^2
```
Nonlinear: 40% WR → 42 candles, 50% → 60, 60% → 94. Clamped to [20, 192].

### Trailing Stop Activation (ATR multiples)
```
activation = favorable / (1 + rr_ratio)
```
High RR → activate earlier. Default ≈ 1.5 ATR.

### Trailing Stop Distance (ATR multiples)
```
distance = adverse * (1 - win_rate)
```
High WR → tight trail. Low WR → wide trail. Clamped to [0.3, 2.5].

### Breakeven Activation (ATR multiples)
```
be_activation = adverse * loss_duration / (win_duration + loss_duration)
```
Loss zone fraction of noise. Clamped to [0.3, 2.0].

### Exploration Coefficient
```
exploration = (1 - win_rate) / (1 + trades / (win_duration * rr_ratio))
```
Explore more when losing. Decays as quality trades accumulate.

### Kelly Fraction (position sizing multiplier)
```
kelly_full = (wr * rr - (1 - wr)) / rr
fraction = (wr * rr) / (wr * rr + (1 - wr))
kelly = kelly_full * fraction
```
Clamped to [0.05, 0.50].

### Max Position Size
```
rr_scale = rr / (1 + rr)
max_size = kelly_fraction * rr_scale
```
Clamped to [0.05, 0.30] of equity.

## Reward K (Sigmoid Steepness)
```
k = 1 / pnl_std
```
Where `pnl_std = sqrt(ema_pnl_sq - ema_pnl^2)`. Scales inversely with trade P&L volatility.
