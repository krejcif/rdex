# 06 — Backtest Engine

## Role

`BacktestEngine` is purely a simulation orchestrator. It feeds candles to the learning engine and manages the portfolio. All trading logic (SL/TP, trailing, cooldowns) lives in `TradeManager` and `LearningEngine`.

## Anti-Look-Ahead Guarantees

1. **Temporal ordering assertion:** Panics if `candles[i].open_time < candles[i-1].open_time`
2. **Monotonic index assertion:** `max_seen_index` only increases; panics if processing a candle at index < `max_seen_index`
3. **MarketState construction:** `build_market_state(symbol, candles, i, lookback)` only uses `candles[start..=i]`
4. **Indicator computation:** `compute_indicators()` operates on the passed slice only

## Per-Candle Loop

```
for i in warmup..candles.len():
    assert(i >= max_seen_index)              # anti-look-ahead
    max_seen_index = i

    1. learning.adaptive.tick_candle()        # age-based tracking
    2. portfolio.update_mark(close)           # mark-to-market
    3. apply_funding_fee(candle)              # if position open + funding event in range
    4. check_liquidation()                    # close if margin exceeded
    5. trades.check_sl_tp()                   # SL/TP (respects min_hold)
    6. trades.on_candle()                     # excursion tracking, trailing, max_hold exit
    7. trades.tick_cooldown()                 # skip if in cooldown
    8. build_market_state()                   # from ONLY past data
    9. learning.decide()                      # features → pattern → Thompson → action
   10. execute_decision()                     # open/close/flip position
```

## Funding Fee Handling

Binance Futures funding occurs every 8 hours (00:00, 08:00, 16:00 UTC). The engine checks each 8-hour boundary within the current candle's time range and looks up the actual rate from a HashMap. Default rate 0.0 if not available (no data = no fee).

- **Long + positive rate:** Longs pay
- **Short + positive rate:** Shorts receive
- Accumulated funding fees are deducted from P&L at position close.

## Position Lifecycle

```
Signal received (Long/Short)
    │
    ├─ If opposite position open → close it first (signal_flip)
    ├─ Portfolio.open_position() with slippage (fee deferred to close)
    └─ TradeManager.on_entry() captures SL/TP, ATR, pattern, equity

Each candle while open:
    ├─ TradeManager.on_candle() tracks excursions, updates trailing stop
    ├─ TradeManager.check_sl_tp() fires after min_hold
    └─ Max hold exit with 2x normal cooldown

Position closes:
    ├─ Portfolio.close_position() calculates net P&L (slippage + entry fee + exit fee + funding)
    ├─ TradeManager.on_exit() records outcome → Thompson + AdaptiveParams + ExcursionTracker
    └─ TradeManager.reset() sets cooldown from adaptive params
```

## Results Output

`BacktestResult` contains:
- `PerformanceMetrics` (30+ metrics)
- Full `Vec<TradeRecord>` with extended analysis fields
- `final_equity`
- `health` report per symbol (regret, observations, learning status)

Detailed output includes: per-symbol breakdown, exit reason distribution, pattern performance table, full trade log with MAE/MFE/confidence/equity.
