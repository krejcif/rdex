use crate::backtest::portfolio::TradeRecord;
use crate::domain::PositionSide;
use serde::{Deserialize, Serialize};

/// Complete performance metrics for a backtest run
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    // Core risk/return
    pub total_return_pct: f64,
    pub annualized_return_pct: f64,
    pub sharpe_ratio: f64,
    pub sortino_ratio: f64,
    pub max_drawdown_pct: f64,
    pub calmar_ratio: f64,
    pub recovery_factor: f64,
    // Trade statistics
    pub win_rate: f64,
    pub profit_factor: f64,
    pub total_trades: usize,
    pub winning_trades: usize,
    pub losing_trades: usize,
    pub avg_win_pct: f64,
    pub avg_loss_pct: f64,
    pub expectancy: f64,
    pub median_trade_pct: f64,
    pub best_trade_pct: f64,
    pub worst_trade_pct: f64,
    pub avg_holding_periods: f64,
    pub max_consecutive_wins: usize,
    pub max_consecutive_losses: usize,
    pub trade_frequency: f64,
    // Long/short breakdown
    pub long_trades: usize,
    pub short_trades: usize,
    pub long_win_rate: f64,
    pub short_win_rate: f64,
    pub long_avg_pnl: f64,
    pub short_avg_pnl: f64,
    // Excursion analysis
    pub avg_mae_pct: f64,
    pub avg_mfe_pct: f64,
    // Equity curve (kept for downstream use)
    pub equity_curve: Vec<f64>,
}

/// Calculate all performance metrics from equity curve and trade records
pub fn calculate_metrics(
    equity_curve: &[f64],
    trades: &[TradeRecord],
    days: f64,
) -> PerformanceMetrics {
    let trade_pnls: Vec<f64> = trades.iter().map(|t| t.pnl_pct).collect();

    let total_return_pct = if equity_curve.len() >= 2 {
        (equity_curve.last().unwrap() / equity_curve[0] - 1.0) * 100.0
    } else {
        0.0
    };

    let annualized_return_pct = if days > 0.0 {
        ((1.0 + total_return_pct / 100.0).powf(365.0 / days) - 1.0) * 100.0
    } else {
        0.0
    };

    // Daily returns for Sharpe/Sortino
    let daily_returns = compute_daily_returns(equity_curve);
    let sharpe_ratio = sharpe(&daily_returns);
    let sortino_ratio = sortino(&daily_returns);
    let max_drawdown_pct = max_drawdown(equity_curve);
    let calmar_ratio = if max_drawdown_pct > 0.01 {
        annualized_return_pct / max_drawdown_pct
    } else {
        0.0
    };
    let recovery_factor = if max_drawdown_pct > 0.01 {
        total_return_pct / max_drawdown_pct
    } else {
        0.0
    };

    let winning: Vec<f64> = trade_pnls.iter().filter(|&&p| p > 0.0).copied().collect();
    let losing: Vec<f64> = trade_pnls.iter().filter(|&&p| p < 0.0).copied().collect();

    let win_rate = if trade_pnls.is_empty() {
        0.0
    } else {
        winning.len() as f64 / trade_pnls.len() as f64
    };

    let gross_profit: f64 = winning.iter().sum();
    let gross_loss: f64 = losing.iter().map(|l| l.abs()).sum();
    let profit_factor = if gross_loss > 0.0 {
        gross_profit / gross_loss
    } else if gross_profit > 0.0 {
        f64::INFINITY
    } else {
        0.0
    };

    let avg_win_pct = if winning.is_empty() {
        0.0
    } else {
        winning.iter().sum::<f64>() / winning.len() as f64
    };
    let avg_loss_pct = if losing.is_empty() {
        0.0
    } else {
        losing.iter().sum::<f64>() / losing.len() as f64
    };

    let expectancy = if trade_pnls.is_empty() {
        0.0
    } else {
        trade_pnls.iter().sum::<f64>() / trade_pnls.len() as f64
    };

    let median_trade_pct = median(&trade_pnls);
    let best_trade_pct = trade_pnls.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let worst_trade_pct = trade_pnls.iter().copied().fold(f64::INFINITY, f64::min);

    let holding_periods: Vec<usize> = trades.iter().map(|t| t.holding_periods).collect();
    let avg_holding = if holding_periods.is_empty() {
        0.0
    } else {
        holding_periods.iter().sum::<usize>() as f64 / holding_periods.len() as f64
    };

    let max_consecutive_wins = max_consecutive_pos(&trade_pnls);
    let max_consecutive_losses = max_consecutive_neg(&trade_pnls);

    let trade_frequency = if days > 0.0 {
        trades.len() as f64 / days
    } else {
        0.0
    };

    // Long/short breakdown
    let long_trades: Vec<&TradeRecord> = trades
        .iter()
        .filter(|t| t.side == PositionSide::Long)
        .collect();
    let short_trades: Vec<&TradeRecord> = trades
        .iter()
        .filter(|t| t.side == PositionSide::Short)
        .collect();
    let long_wins = long_trades.iter().filter(|t| t.pnl_pct > 0.0).count();
    let short_wins = short_trades.iter().filter(|t| t.pnl_pct > 0.0).count();
    let long_win_rate = if long_trades.is_empty() {
        0.0
    } else {
        long_wins as f64 / long_trades.len() as f64
    };
    let short_win_rate = if short_trades.is_empty() {
        0.0
    } else {
        short_wins as f64 / short_trades.len() as f64
    };
    let long_avg_pnl = if long_trades.is_empty() {
        0.0
    } else {
        long_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / long_trades.len() as f64
    };
    let short_avg_pnl = if short_trades.is_empty() {
        0.0
    } else {
        short_trades.iter().map(|t| t.pnl_pct).sum::<f64>() / short_trades.len() as f64
    };

    // Excursion analysis
    let avg_mae_pct = if trades.is_empty() {
        0.0
    } else {
        trades.iter().map(|t| t.max_adverse_excursion).sum::<f64>() / trades.len() as f64
    };
    let avg_mfe_pct = if trades.is_empty() {
        0.0
    } else {
        trades
            .iter()
            .map(|t| t.max_favorable_excursion)
            .sum::<f64>()
            / trades.len() as f64
    };

    PerformanceMetrics {
        total_return_pct,
        annualized_return_pct,
        sharpe_ratio,
        sortino_ratio,
        max_drawdown_pct,
        calmar_ratio,
        recovery_factor,
        win_rate,
        profit_factor,
        total_trades: trade_pnls.len(),
        winning_trades: winning.len(),
        losing_trades: losing.len(),
        avg_win_pct,
        avg_loss_pct,
        expectancy,
        median_trade_pct,
        best_trade_pct: if trade_pnls.is_empty() {
            0.0
        } else {
            best_trade_pct
        },
        worst_trade_pct: if trade_pnls.is_empty() {
            0.0
        } else {
            worst_trade_pct
        },
        avg_holding_periods: avg_holding,
        max_consecutive_wins,
        max_consecutive_losses,
        trade_frequency,
        long_trades: long_trades.len(),
        short_trades: short_trades.len(),
        long_win_rate,
        short_win_rate,
        long_avg_pnl,
        short_avg_pnl,
        avg_mae_pct,
        avg_mfe_pct,
        equity_curve: equity_curve.to_vec(),
    }
}

fn median(vals: &[f64]) -> f64 {
    if vals.is_empty() {
        return 0.0;
    }
    let mut sorted = vals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mid = sorted.len() / 2;
    if sorted.len() % 2 == 0 {
        (sorted[mid - 1] + sorted[mid]) / 2.0
    } else {
        sorted[mid]
    }
}

fn compute_daily_returns(equity: &[f64]) -> Vec<f64> {
    if equity.len() < 2 {
        return vec![];
    }
    // Group by approximate daily intervals (96 x 15min candles per day)
    let candles_per_day = 96;
    let mut daily = Vec::new();
    let mut i = 0;
    while i + candles_per_day < equity.len() {
        let ret = equity[i + candles_per_day] / equity[i] - 1.0;
        daily.push(ret);
        i += candles_per_day;
    }
    if daily.is_empty() && equity.len() >= 2 {
        // Fallback: use per-candle returns
        for i in 1..equity.len() {
            daily.push(equity[i] / equity[i - 1] - 1.0);
        }
    }
    daily
}

fn sharpe(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let variance =
        returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / (returns.len() - 1) as f64;
    let std = variance.sqrt();
    if std < 1e-10 {
        return 0.0;
    }
    // Annualize: assume daily returns
    (mean / std) * (365.0_f64).sqrt()
}

fn sortino(returns: &[f64]) -> f64 {
    if returns.len() < 2 {
        return 0.0;
    }
    let mean = returns.iter().sum::<f64>() / returns.len() as f64;
    let downside_variance = returns
        .iter()
        .filter(|&&r| r < 0.0)
        .map(|r| r.powi(2))
        .sum::<f64>()
        / returns.len() as f64;
    let downside_std = downside_variance.sqrt();
    if downside_std < 1e-10 {
        return if mean > 0.0 { 100.0 } else { 0.0 };
    }
    (mean / downside_std) * (365.0_f64).sqrt()
}

fn max_drawdown(equity: &[f64]) -> f64 {
    if equity.is_empty() {
        return 0.0;
    }
    let mut peak = equity[0];
    let mut max_dd = 0.0;

    for &val in equity {
        if val > peak {
            peak = val;
        }
        let dd = (peak - val) / peak * 100.0;
        if dd > max_dd {
            max_dd = dd;
        }
    }
    max_dd
}

fn max_consecutive_neg(pnls: &[f64]) -> usize {
    let mut max_run = 0;
    let mut current_run = 0;
    for &pnl in pnls {
        if pnl < 0.0 {
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 0;
        }
    }
    max_run
}

fn max_consecutive_pos(pnls: &[f64]) -> usize {
    let mut max_run = 0;
    let mut current_run = 0;
    for &pnl in pnls {
        if pnl > 0.0 {
            current_run += 1;
            max_run = max_run.max(current_run);
        } else {
            current_run = 0;
        }
    }
    max_run
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sharpe_positive() {
        // Vary returns slightly to have non-zero std
        let returns: Vec<f64> = (0..100)
            .map(|i| 0.001 + (i as f64 * 0.0001).sin() * 0.0002)
            .collect();
        assert!(sharpe(&returns) > 0.0);
    }

    #[test]
    fn test_sharpe_zero_std() {
        let returns = vec![0.0; 100];
        assert_eq!(sharpe(&returns), 0.0);
    }

    #[test]
    fn test_max_drawdown() {
        let equity = vec![100.0, 110.0, 95.0, 105.0, 80.0, 90.0];
        let dd = max_drawdown(&equity);
        // Peak at 110, trough at 80 -> (110-80)/110 * 100 = 27.27%
        assert!((dd - 27.27).abs() < 0.1);
    }

    #[test]
    fn test_max_drawdown_no_drawdown() {
        let equity = vec![100.0, 101.0, 102.0, 103.0];
        assert_eq!(max_drawdown(&equity), 0.0);
    }

    #[test]
    fn test_consecutive_losses() {
        let pnls = vec![1.0, -1.0, -2.0, -3.0, 1.0, -1.0, -1.0];
        assert_eq!(max_consecutive_neg(&pnls), 3);
    }

    #[test]
    fn test_consecutive_wins() {
        let pnls = vec![1.0, 2.0, 3.0, -1.0, 1.0, 1.0];
        assert_eq!(max_consecutive_pos(&pnls), 3);
    }

    #[test]
    fn test_median_odd() {
        assert!((median(&[3.0, 1.0, 2.0]) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn test_median_even() {
        assert!((median(&[4.0, 1.0, 3.0, 2.0]) - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_profit_factor() {
        use crate::domain::PositionSide;
        let trades: Vec<TradeRecord> = vec![
            make_test_trade(100.0, PositionSide::Long),
            make_test_trade(-50.0, PositionSide::Long),
            make_test_trade(80.0, PositionSide::Short),
            make_test_trade(-30.0, PositionSide::Short),
        ];
        let equity = vec![10000.0; 2];
        let metrics = calculate_metrics(&equity, &trades, 1.0);
        // Gross profit = 180, gross loss = 80, PF = 2.25
        assert!((metrics.profit_factor - 2.25).abs() < 0.01);
    }

    fn make_test_trade(pnl_pct: f64, side: PositionSide) -> TradeRecord {
        TradeRecord {
            symbol: "TEST".to_string(),
            side,
            entry_price: 100.0,
            exit_price: 100.0,
            size_usd: 1000.0,
            pnl: pnl_pct * 10.0,
            pnl_pct,
            entry_time: 0,
            exit_time: 0,
            holding_periods: 5,
            strategy: "test".to_string(),
            max_adverse_excursion: 0.5,
            max_favorable_excursion: 1.0,
            fees_paid: 1.0,
            funding_fees_paid: 0.0,
            pattern: String::new(),
            confidence: 0.5,
            thompson_reward: 0.5,
            entry_atr: 1.0,
            exit_reason: "test".to_string(),
            equity_at_entry: 10000.0,
            equity_at_exit: 10000.0,
            position_size_frac: 0.1,
        }
    }

    #[test]
    fn test_empty_trades() {
        let equity = vec![10000.0, 10000.0, 10000.0];
        let m = calculate_metrics(&equity, &[], 1.0);
        assert_eq!(m.total_trades, 0);
        assert_eq!(m.win_rate, 0.0);
        assert_eq!(m.profit_factor, 0.0);
        assert_eq!(m.best_trade_pct, 0.0);
        assert_eq!(m.worst_trade_pct, 0.0);
    }

    #[test]
    fn test_all_winning_trades() {
        let trades = vec![
            make_test_trade(5.0, PositionSide::Long),
            make_test_trade(3.0, PositionSide::Long),
            make_test_trade(7.0, PositionSide::Short),
        ];
        let equity = vec![10000.0; 2];
        let m = calculate_metrics(&equity, &trades, 1.0);
        assert_eq!(m.win_rate, 1.0);
        assert_eq!(m.losing_trades, 0);
        assert_eq!(m.profit_factor, f64::INFINITY);
    }

    #[test]
    fn test_all_losing_trades() {
        let trades = vec![
            make_test_trade(-2.0, PositionSide::Long),
            make_test_trade(-3.0, PositionSide::Short),
        ];
        let equity = vec![10000.0; 2];
        let m = calculate_metrics(&equity, &trades, 1.0);
        assert_eq!(m.win_rate, 0.0);
        assert_eq!(m.winning_trades, 0);
        assert_eq!(m.profit_factor, 0.0);
    }

    #[test]
    fn test_single_trade() {
        let trades = vec![make_test_trade(2.5, PositionSide::Long)];
        let equity = vec![10000.0; 2];
        let m = calculate_metrics(&equity, &trades, 1.0);
        assert_eq!(m.total_trades, 1);
        assert_eq!(m.winning_trades, 1);
        assert!((m.median_trade_pct - 2.5).abs() < 1e-10);
        assert!((m.best_trade_pct - 2.5).abs() < 1e-10);
        assert!((m.worst_trade_pct - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_calmar_recovery_no_drawdown() {
        let equity = vec![10000.0, 10100.0, 10200.0, 10300.0];
        let m = calculate_metrics(&equity, &[], 30.0);
        // No drawdown -> calmar and recovery = 0
        assert_eq!(m.calmar_ratio, 0.0);
        assert_eq!(m.recovery_factor, 0.0);
    }

    #[test]
    fn test_total_return_calculation() {
        let equity = vec![10000.0, 11000.0];
        let m = calculate_metrics(&equity, &[], 30.0);
        assert!((m.total_return_pct - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_max_drawdown_empty() {
        assert_eq!(max_drawdown(&[]), 0.0);
    }

    #[test]
    fn test_consecutive_empty() {
        assert_eq!(max_consecutive_pos(&[]), 0);
        assert_eq!(max_consecutive_neg(&[]), 0);
    }

    #[test]
    fn test_median_empty() {
        assert_eq!(median(&[]), 0.0);
    }

    #[test]
    fn test_median_single() {
        assert!((median(&[3.0]) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn test_long_short_breakdown() {
        let trades = vec![
            make_test_trade(5.0, PositionSide::Long),
            make_test_trade(-2.0, PositionSide::Long),
            make_test_trade(3.0, PositionSide::Short),
        ];
        let equity = vec![10000.0; 2];
        let m = calculate_metrics(&equity, &trades, 1.0);
        assert_eq!(m.long_trades, 2);
        assert_eq!(m.short_trades, 1);
        assert_eq!(m.long_win_rate, 0.5);
        assert_eq!(m.short_win_rate, 1.0);
    }

    #[test]
    fn test_sortino_all_positive() {
        let returns = vec![0.01, 0.02, 0.015, 0.03, 0.01];
        let s = sortino(&returns);
        // All positive: downside std = 0 -> returns 100.0
        assert_eq!(s, 100.0);
    }

    #[test]
    fn test_sortino_all_negative() {
        let returns = vec![-0.01, -0.02, -0.015, -0.03, -0.01];
        let s = sortino(&returns);
        assert!(s < 0.0, "All negative returns should give negative sortino: {}", s);
    }

    #[test]
    fn test_daily_returns_short_data() {
        let equity = vec![100.0, 101.0, 102.0];
        let dr = compute_daily_returns(&equity);
        // Less than 96 candles: falls back to per-candle returns
        assert_eq!(dr.len(), 2);
    }

    #[test]
    fn test_daily_returns_single_point() {
        let dr = compute_daily_returns(&[100.0]);
        assert!(dr.is_empty());
    }

    #[test]
    fn test_daily_returns_empty() {
        let dr = compute_daily_returns(&[]);
        assert!(dr.is_empty());
    }

    #[test]
    fn test_sharpe_single_return() {
        assert_eq!(sharpe(&[0.01]), 0.0);
    }

    #[test]
    fn test_sharpe_empty() {
        assert_eq!(sharpe(&[]), 0.0);
    }

    #[test]
    fn test_sortino_empty() {
        assert_eq!(sortino(&[]), 0.0);
    }

    #[test]
    fn test_sortino_single() {
        assert_eq!(sortino(&[0.01]), 0.0);
    }

    #[test]
    fn test_annualized_return_zero_days() {
        let equity = vec![10000.0, 11000.0];
        let m = calculate_metrics(&equity, &[], 0.0);
        assert_eq!(m.annualized_return_pct, 0.0);
    }

    #[test]
    fn test_trade_frequency() {
        let trades = vec![
            make_test_trade(1.0, PositionSide::Long),
            make_test_trade(2.0, PositionSide::Long),
        ];
        let equity = vec![10000.0; 2];
        let m = calculate_metrics(&equity, &trades, 10.0);
        assert!((m.trade_frequency - 0.2).abs() < 0.01, "2 trades / 10 days = 0.2");
    }

    #[test]
    fn test_avg_holding_periods() {
        let trades = vec![
            make_test_trade(1.0, PositionSide::Long),
            make_test_trade(2.0, PositionSide::Long),
        ];
        let equity = vec![10000.0; 2];
        let m = calculate_metrics(&equity, &trades, 1.0);
        assert_eq!(m.avg_holding_periods, 5.0); // both trades have 5 holding periods
    }

    #[test]
    fn test_expectancy_calculation() {
        let trades = vec![
            make_test_trade(10.0, PositionSide::Long),
            make_test_trade(-5.0, PositionSide::Long),
        ];
        let equity = vec![10000.0; 2];
        let m = calculate_metrics(&equity, &trades, 1.0);
        // expectancy = mean of [10, -5] = 2.5
        assert!((m.expectancy - 2.5).abs() < 0.01);
    }

    #[test]
    fn test_max_drawdown_single_point() {
        assert_eq!(max_drawdown(&[100.0]), 0.0);
    }
}
