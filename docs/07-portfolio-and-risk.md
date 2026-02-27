# 07 — Portfolio & Risk Management

## Portfolio Model

Simulates Binance USDT-M Futures with realistic costs:

| Parameter | Default | Notes |
|-----------|---------|-------|
| Leverage | 3x | Conservative; configurable |
| Taker fee | 0.04% | Per side (0.08% round trip) |
| Maker fee | 0.02% | Not currently used (all market orders) |
| Slippage | 2 bps | Applied on both entry and exit |
| Funding rate | Actual historical | 8-hour intervals, default 0.0 if missing |

## Position Sizing

```
notional = equity * size_fraction * leverage
size     = notional / price
```

`size_fraction` comes from the Kelly criterion calculation in `LearningEngine.decide()`, clamped to [0.05, 0.50] of equity.

## P&L Calculation

```
raw_pnl    = size * (exit_price - entry_price)    # long
           = size * (entry_price - exit_price)    # short

entry_fee  = entry_notional * taker_fee
exit_fee   = exit_notional * taker_fee
net_pnl    = raw_pnl - entry_fee - exit_fee - accumulated_funding
pnl_pct    = net_pnl / margin * 100
           where margin = notional / leverage
```

Both entry and exit fees are deducted at close time so that `net_pnl` and `pnl_pct` reflect the full round-trip cost.

## Slippage Model

```
entry_price = price + slippage  (long)  /  price - slippage  (short)
exit_price  = price - slippage  (long)  /  price + slippage  (short)
```

Where `slippage = price * slippage_bps / 10000`.

## Liquidation

Simplified Binance model:
```
unrealized_loss > margin * 0.95
```

Where margin = notional / leverage. Liquidation check uses worst-case intra-candle price: `candle.low` for longs, `candle.high` for shorts.

## TradeRecord Fields

| Field | Source | Description |
|-------|--------|-------------|
| `pnl`, `pnl_pct` | Portfolio | Net P&L after all costs |
| `fees_paid` | Portfolio | Entry + exit taker fees |
| `funding_fees_paid` | TradeManager | Accumulated 8h funding |
| `max_adverse_excursion` | TradeManager | Worst unrealized loss (%) |
| `max_favorable_excursion` | TradeManager | Best unrealized gain (%) |
| `pattern` | TradeManager | Pattern at entry ("01_10") |
| `confidence` | TradeManager | Thompson win probability |
| `thompson_reward` | TradeManager | Sigmoid-mapped reward [0,1] |
| `entry_atr` | TradeManager | ATR at entry |
| `exit_reason` | TradeManager | stop_loss, trailing_stop, max_hold, etc. |
| `equity_at_entry/exit` | Portfolio/TradeManager | Account equity at entry and exit |
| `position_size_frac` | TradeManager | Fraction of equity allocated |
