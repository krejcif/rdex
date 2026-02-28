# 02 — Architecture

## Module Map

```
src/
├── lib.rs                    # Re-exports all modules
├── main.rs                   # CLI (clap): backtest + fetch subcommands
│
├── domain/                   # Pure data types, no business logic
│   ├── types.rs             # Candle, Symbol, MarketState, IndicatorSet,
│   │                        #   TradingDecision, TradeSignal, Position,
│   │                        #   MarketContext, StrategyId, FundingRate
│   ├── crypto_domain.rs     # FuturesConfig (leverage, fees, slippage),
│   │                        #   build_market_state(), extract_context()
│   └── indicators.rs        # compute_indicators() — SMA, EMA, RSI, ATR,
│                            #   MACD, Bollinger Bands, ADX, volume SMA
│
├── strategy/                 # Feature extraction & pattern recognition
│   ├── features.rs          # MarketFeatures (7 normalized features + raw ATR/price)
│   └── patterns.rs          # PatternDiscretizer — adaptive median-split
│                            #   4 features × 2 bins = 16 patterns
│
├── engine/                   # Decision-making & learning core
│   ├── strategy.rs          # TradingStrategy trait — live/backtest separation boundary
│   ├── thompson.rs          # Thompson Sampling (DecayingBeta, ThompsonEngine)
│   ├── learner.rs           # LearningEngine — implements TradingStrategy, orchestrates everything
│   ├── adaptive.rs          # AdaptiveParams — derives all trading params from data
│   ├── pattern_library.rs   # KNN pattern library — temporal vectors, directional predictions
│   ├── transfer.rs          # RegretTracker, PlateauDetector, TransferVerification
│   └── trade_manager.rs     # TradeManager — position lifecycle via impl TradingStrategy
│
├── backtest/                 # Simulation framework
│   ├── engine.rs            # InterleavedEngine (multi-symbol) + BacktestEngine (single-symbol)
│   ├── portfolio.rs         # MultiSymbolPortfolio (shared equity) + Portfolio (single-symbol)
│   └── validation.rs        # Walk-forward validation, sign-flip permutation tests
│
├── evaluation/               # Performance measurement
│   ├── metrics.rs           # Sharpe, Sortino, Calmar, max drawdown, profit factor, etc.
│   └── scorer.rs            # pnl_to_reward() — adaptive sigmoid for Thompson feedback
│
└── data/                     # External data management
    ├── fetcher.rs           # Async Binance FAPI client (candles + funding rates)
    └── cache.rs             # CSV serialization with temporal ordering verification
```

## Data Flow

```mermaid
graph TD
    A["Candle[i] just closed"] --> B[build_market_state]
    B --> C["MarketState<br/>(history[0..=i], indicators)"]
    C --> D[MarketFeatures::extract]
    D --> E["7 features:<br/>trend, momentum, volatility,<br/>volume, candle_char, bb_pos,<br/>short_momentum"]
    E --> F[PatternDiscretizer.discretize]
    F --> G["MarketContext<br/>(1 of 16 patterns)"]
    G --> H[ThompsonEngine.select_arm]
    H --> I["Action: long / short / hold"]
    I --> J["Kelly sizing<br/>Excursion-based SL<br/>Adaptive exploration"]
    J --> K[TradingDecision]
    K --> L[Portfolio<br/>Open/close position]
    L --> M["TradeManager<br/>(trailing stop, breakeven,<br/>excursion tracking, max hold)"]
    M --> N[Trade closes]
    N --> O["AdaptiveParams.record → update EMAs<br/>Thompson.record → update Beta<br/>ExcursionTracker → update SL/TP"]
    O -->|feedback| H
```

## Key Responsibility Boundaries

| Component | Owns | Does NOT own |
|-----------|------|--------------|
| `InterleavedEngine` | Timeline merging, anti-look-ahead, two-pass processing, portfolio orchestration | Trading logic, learning, position state |
| `TradingStrategy` trait | Contract between strategy and executor (decide, adaptive, record_outcome) | Implementation details |
| `TradeManager` | SL/TP, trailing stops, excursion tracking, cooldown, exit reasons (via `impl TradingStrategy`) | Portfolio equity, order execution |
| `LearningEngine` | Feature→pattern→action pipeline, Thompson state, excursion tracker (implements `TradingStrategy`) | Position management, portfolio |
| `PatternLibrary` | KNN temporal vectors, directional probability predictions, per-symbol/global stores | Decision making, position management |
| `AdaptiveParams` | All EMA-based trading parameters | Decision making |
| `MultiSymbolPortfolio` | Shared equity, concurrent positions, aggregate exposure limits, margin tracking | When to trade |
