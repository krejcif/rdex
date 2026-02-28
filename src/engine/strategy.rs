use crate::domain::*;
use crate::engine::adaptive::AdaptiveParams;
use crate::engine::pattern_library::PatternLibrary;
use rand::Rng;

/// Trait separating trading strategy from execution infrastructure.
///
/// This defines the boundary between "what to trade" (strategy) and
/// "how to execute" (backtest simulator or live broker). Both backtest and
/// live executors use `impl TradingStrategy`, enabling seamless switching.
///
/// The `LearningEngine` is the primary implementation, using Thompson Sampling
/// and adaptive parameters learned from market data. A live executor would
/// use the same `LearningEngine` instance with a real portfolio and order API.
///
/// # For live trading
///
/// A live executor needs to:
/// 1. Feed candles from a WebSocket → build `MarketState` → call `decide()`
/// 2. Execute the resulting `TradingDecision` via broker API
/// 3. On trade close, call `record_outcome()` / `record_excursion()` for learning
/// 4. Use `adaptive()` for position management (SL/TP, trailing stops, hold limits)
///
/// The `TradeManager` already works with `impl TradingStrategy`, so the same
/// position lifecycle logic (SL/TP, trailing, cooldown) works in both contexts.
pub trait TradingStrategy {
    /// Make a trading decision based on current market state.
    fn decide(
        &mut self,
        state: &MarketState,
        rng: &mut impl Rng,
        library: &PatternLibrary,
    ) -> TradingDecision;

    /// Read-only access to adaptive parameters (for position management).
    fn adaptive(&self) -> &AdaptiveParams;

    /// Mutable access to adaptive parameters (for recording trade outcomes).
    fn adaptive_mut(&mut self) -> &mut AdaptiveParams;

    /// Record a trade outcome for strategy learning.
    fn record_outcome(
        &mut self,
        symbol: &str,
        context: &MarketContext,
        strategy_id: &StrategyId,
        reward: f64,
    );

    /// Record excursion data for adaptive SL/TP learning.
    fn record_excursion(
        &mut self,
        pattern_key: &str,
        side: &str,
        favorable_atr: f64,
        adverse_atr: f64,
        won: bool,
    );

    /// Record a hold observation (called periodically when not in a position).
    fn record_hold(&mut self, symbol: &str, pattern: &MarketContext, recent_return: f64);

    /// Get the last pattern used in the most recent decision.
    fn last_pattern(&self) -> Option<&MarketContext>;

    /// Get the last KNN directional probability from the most recent decision.
    fn last_prediction(&self) -> Option<f64>;

    /// Tick time-based decay (called once per candle/timestamp).
    fn tick_candle(&mut self);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::learner::{LearnerConfig, LearningEngine};
    use crate::engine::trade_manager::TradeManager;

    /// Verify that LearningEngine implements TradingStrategy and can be
    /// used polymorphically through the trait bound.
    fn accepts_strategy(strategy: &impl TradingStrategy) -> usize {
        strategy.adaptive().cooldown()
    }

    #[test]
    fn test_learning_engine_implements_strategy() {
        let engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let cooldown = accepts_strategy(&engine);
        assert!(cooldown > 0);
    }

    #[test]
    fn test_trade_manager_works_with_trait() {
        let engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let mut tm = TradeManager::new();
        // reset() now takes &impl TradingStrategy — verify it works with LearningEngine
        tm.reset(&engine);
        assert!(tm.cooldown_remaining > 0);
    }

    #[test]
    fn test_trait_adaptive_access() {
        let engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        let adaptive = TradingStrategy::adaptive(&engine);
        assert!(adaptive.min_hold() > 0);
        assert!(adaptive.max_hold() > adaptive.min_hold());
    }

    #[test]
    fn test_trait_tick_candle() {
        let mut engine = LearningEngine::new(vec!["BTCUSDT".into()], LearnerConfig::default());
        // tick_candle should not panic
        TradingStrategy::tick_candle(&mut engine);
    }
}
