use clap::Parser;

#[derive(Parser)]
#[command(name = "rdex", about = "Self-learning crypto trading system")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(clap::Subcommand)]
enum Commands {
    /// Run backtest on historical data
    Backtest {
        #[arg(short, long, default_value = "30")]
        days: i64,
        #[arg(
            short,
            long,
            default_value = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT"
        )]
        symbols: String,
        #[arg(short, long, default_value = "10000")]
        equity: f64,
        #[arg(short, long, default_value = "3")]
        leverage: f64,
    },
    /// Fetch and cache market data
    Fetch {
        #[arg(short, long, default_value = "30")]
        days: i64,
        #[arg(
            short,
            long,
            default_value = "BTCUSDT,ETHUSDT,SOLUSDT,BNBUSDT,XRPUSDT,DOGEUSDT,AVAXUSDT,LINKUSDT"
        )]
        symbols: String,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(tracing_subscriber::EnvFilter::from_default_env())
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Backtest {
            days,
            symbols,
            equity,
            leverage,
        } => {
            run_backtest(days, &symbols, equity, leverage).await?;
        }
        Commands::Fetch { days, symbols } => {
            run_fetch(days, &symbols).await?;
        }
    }

    Ok(())
}

async fn run_fetch(days: i64, symbols: &str) -> Result<(), Box<dyn std::error::Error>> {
    let symbols: Vec<&str> = symbols.split(',').collect();
    for symbol in &symbols {
        let candles = rdex::data::fetch_last_n_days(symbol, days).await?;
        rdex::data::save_to_csv(&candles, &rdex::data::cache_path(symbol, "data"))?;
        println!(
            "Fetched and cached {} candles for {}",
            candles.len(),
            symbol
        );

        let funding = rdex::data::fetch_funding_last_n_days(symbol, days).await?;
        rdex::data::save_funding_csv(&funding, &rdex::data::funding_cache_path(symbol, "data"))?;
        println!(
            "Fetched and cached {} funding rates for {}",
            funding.len(),
            symbol
        );
    }
    Ok(())
}

async fn run_backtest(
    days: i64,
    symbols_str: &str,
    equity: f64,
    leverage: f64,
) -> Result<(), Box<dyn std::error::Error>> {
    use rdex::backtest::*;
    use rdex::domain::*;
    use rdex::engine::*;

    let symbols: Vec<String> = symbols_str.split(',').map(|s| s.to_string()).collect();

    println!("=== RDEX Self-Learning Crypto Trading System ===");
    println!("Symbols: {:?}", symbols);
    println!(
        "Days: {}, Equity: ${}, Leverage: {}x",
        days, equity, leverage
    );

    // Fetch candle data + funding rates (parallel across symbols)
    let mut fetch_set = tokio::task::JoinSet::new();
    for symbol in symbols.clone() {
        fetch_set.spawn(async move {
            let candles = rdex::data::load_or_fetch(&symbol, days, "data")
                .await
                .map_err(|e| e.to_string());
            let funding = rdex::data::load_or_fetch_funding(&symbol, days, "data")
                .await
                .map_err(|e| e.to_string());
            (symbol, candles, funding)
        });
    }

    let mut all_data: Vec<(String, Vec<Candle>)> = Vec::new();
    let mut all_funding: std::collections::HashMap<String, Vec<FundingRate>> =
        std::collections::HashMap::new();
    while let Some(result) = fetch_set.join_next().await {
        let (symbol, candles_result, funding_result) = result?;
        let candles = candles_result.map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
        println!(
            "{}: {} candles ({:.1} days)",
            symbol,
            candles.len(),
            candles.len() as f64 / 96.0
        );

        let funding = funding_result.unwrap_or_else(|e: String| {
            eprintln!("  Warning: could not fetch funding for {}: {}", symbol, e);
            Vec::new()
        });
        if !funding.is_empty() {
            println!("  {} funding rates loaded", funding.len());
        }

        all_data.push((symbol.clone(), candles));
        all_funding.insert(symbol.clone(), funding);
    }
    // Maintain consistent symbol ordering
    all_data.sort_by(|a, b| {
        symbols.iter().position(|s| s == &a.0).cmp(&symbols.iter().position(|s| s == &b.0))
    });

    let config = FuturesConfig {
        leverage,
        initial_equity: equity,
        ..Default::default()
    };
    let learner_config = LearnerConfig::default();

    // ===== SINGLE-PASS INTERLEAVED BACKTEST =====
    println!("\n========== INTERLEAVED BACKTEST (all symbols, single pass) ==========");
    let mut engine = InterleavedEngine::new(
        config.clone(),
        LearningEngine::new(symbols.clone(), learner_config.clone()),
        42,
    );

    // Set funding rates for each symbol
    for (symbol, funding) in &all_funding {
        engine.set_funding_rates(symbol, funding);
    }

    let result = engine.run(&all_data, 60);

    let tx_fees: f64 = result.trades.iter().map(|t| t.fees_paid).sum();
    let fund_fees: f64 = result.trades.iter().map(|t| t.funding_fees_paid).sum();
    println!(
        "  {} trades, return: {:+.2}%, sharpe: {:.2}, tx_fees: ${:.2}, funding: ${:.2}",
        result.trades.len(),
        result.performance.total_return_pct,
        result.performance.sharpe_ratio,
        tx_fees,
        fund_fees
    );

    // ===== DETAILED RESULTS =====
    println!("\n{}", "#".repeat(80));
    println!("  INTERLEAVED BACKTEST — DETAILED RESULTS");
    println!("{}", "#".repeat(80));
    result.print_summary();

    // Walk-forward validation (interleaved, all symbols together)
    println!("\nWalk-forward validation (3 folds, interleaved)...");
    let validation = walk_forward_validation_interleaved(
        &all_data,
        &all_funding,
        &config,
        &learner_config,
        3,
        0.7,
    );
    validation.print_summary();

    // Permutation test on combined trade log
    if !result.trades.is_empty() {
        let pnls: Vec<f64> = result.trades.iter().map(|t| t.pnl_pct).collect();
        let actual_return: f64 = pnls.iter().sum();
        let perm = permutation_test(&pnls, actual_return, 1000, 42);
        perm.print_summary();
    }

    // ===== ADAPTIVE PARAMETERS STATE =====
    println!("\n{}", "=".repeat(70));
    println!("  ADAPTIVE PARAMETERS (final state)");
    println!("{}", "=".repeat(70));
    let a = &engine.learning.adaptive;
    println!("  Trades observed:      {:>10}", a.stats.total_trades);
    println!("  Reward K:             {:>10.3}", a.reward_k());
    println!("  Cooldown:             {:>10} candles", a.cooldown());
    println!("  Min Hold:             {:>10} candles", a.min_hold());
    println!("  Max Hold:             {:>10} candles", a.max_hold());
    println!(
        "  Trail Activation:     {:>10.3} ATR",
        a.trail_activation_atr()
    );
    println!(
        "  Trail Distance:       {:>10.3} ATR",
        a.trail_distance_atr()
    );
    println!(
        "  Breakeven Activation: {:>10.3} ATR",
        a.breakeven_activation_atr()
    );
    println!("  Exploration Coeff:    {:>10.4}", a.exploration_coeff());
    println!("  Kelly Fraction:       {:>10.3}", a.kelly_fraction());
    println!("  Max Position Size:    {:>10.3}", a.max_position_size());

    // ===== PORTFOLIO-WIDE SUMMARY =====
    println!("\n{}", "=".repeat(70));
    println!("  PORTFOLIO SUMMARY");
    println!("{}", "=".repeat(70));

    let total_trades = result.trades.len();
    let total_wins = result.trades.iter().filter(|t| t.pnl_pct > 0.0).count();
    let portfolio_wr = if total_trades > 0 {
        total_wins as f64 / total_trades as f64 * 100.0
    } else {
        0.0
    };

    println!("  Symbols:              {:>10}", symbols.len());
    println!(
        "  Total Return:         {:>+9.2}%",
        result.performance.total_return_pct
    );
    println!("  Total Trades:         {:>10}", total_trades);
    println!("  Portfolio Win Rate:   {:>9.1}%", portfolio_wr);
    println!(
        "  Sharpe Ratio:         {:>10.2}",
        result.performance.sharpe_ratio
    );
    println!(
        "  Sortino Ratio:        {:>10.2}",
        result.performance.sortino_ratio
    );
    println!(
        "  Max Drawdown:         {:>9.2}%",
        result.performance.max_drawdown_pct
    );
    println!(
        "  Profit Factor:        {:>10.2}",
        result.performance.profit_factor
    );
    println!(
        "  Peak Margin Use:      {:>9.1}x",
        engine.portfolio.peak_margin_utilization
    );
    if engine.portfolio.margin_rejections > 0 {
        println!(
            "  Margin Rejections:    {:>10}",
            engine.portfolio.margin_rejections
        );
    }
    println!("  Total TX Fees:        ${:>9.2}", tx_fees);
    println!("  Total Funding Fees:   ${:>9.2}", fund_fees);
    println!(
        "  Final Equity:         ${:>9.2}",
        result.final_equity
    );

    // Per-symbol breakdown (computed from combined trade log)
    println!("\n  Per-Symbol Breakdown:");
    println!(
        "  {:10} {:>8} {:>7} {:>7} {:>8} {:>8} {:>9} {:>9}",
        "Symbol", "PnL%", "Trades", "WinR%", "AvgWin%", "AvgLoss%", "TxFees", "FundFees"
    );
    for symbol in &symbols {
        let sym_trades: Vec<&rdex::backtest::portfolio::TradeRecord> = result
            .trades
            .iter()
            .filter(|t| t.symbol == *symbol)
            .collect();
        let n = sym_trades.len();
        if n == 0 {
            println!(
                "  {:10} {:>+7.2}% {:>7} {:>6.1}%       -        - {:>9} {:>9}",
                symbol, 0.0, 0, 0.0, "-", "-"
            );
            continue;
        }
        let sym_pnl: f64 = sym_trades.iter().map(|t| t.pnl_pct).sum();
        let sym_wins = sym_trades.iter().filter(|t| t.pnl_pct > 0.0).count();
        let sym_wr = sym_wins as f64 / n as f64 * 100.0;
        let winning: Vec<f64> = sym_trades
            .iter()
            .filter(|t| t.pnl_pct > 0.0)
            .map(|t| t.pnl_pct)
            .collect();
        let losing: Vec<f64> = sym_trades
            .iter()
            .filter(|t| t.pnl_pct <= 0.0)
            .map(|t| t.pnl_pct)
            .collect();
        let avg_win = if !winning.is_empty() {
            winning.iter().sum::<f64>() / winning.len() as f64
        } else {
            0.0
        };
        let avg_loss = if !losing.is_empty() {
            losing.iter().sum::<f64>() / losing.len() as f64
        } else {
            0.0
        };
        let sym_tx: f64 = sym_trades.iter().map(|t| t.fees_paid).sum();
        let sym_fund: f64 = sym_trades.iter().map(|t| t.funding_fees_paid).sum();
        println!(
            "  {:10} {:>+7.2}% {:>7} {:>6.1}% {:>+7.2}% {:>+7.2}% ${:>8.2} ${:>8.2}",
            symbol, sym_pnl, n, sym_wr, avg_win, avg_loss, sym_tx, sym_fund
        );
    }

    // ===== THOMPSON SAMPLING ARM STATE =====
    println!("\n{}", "=".repeat(90));
    println!("  THOMPSON SAMPLING — LEARNED ARM STATE (global pool)");
    println!("{}", "=".repeat(90));
    let arm_report = engine.learning.thompson.arm_report("_global");
    if arm_report.is_empty() {
        println!("  No arms learned yet.");
    } else {
        println!(
            "  {:12} {:>8} {:>8} {:>10} {:>10} {:>10}",
            "Pattern", "Action", "Mean", "Evidence", "Variance", "Status"
        );
        // Group by pattern to show side-by-side
        let mut current_pattern = String::new();
        for (pattern, action, mean, evidence, variance) in &arm_report {
            if *pattern != current_pattern {
                if !current_pattern.is_empty() {
                    println!("  {}", "-".repeat(68));
                }
                current_pattern = pattern.clone();
            }
            let status = if *evidence < 5.0 {
                "cold-start"
            } else if *evidence < 20.0 {
                "learning"
            } else if *evidence < 50.0 {
                "maturing"
            } else {
                "converged"
            };
            println!(
                "  {:12} {:>8} {:>8.4} {:>10.1} {:>10.6} {:>10}",
                pattern, action, mean, evidence, variance, status
            );
        }
        // Summary: which patterns favor long vs short vs hold
        println!("\n  --- Pattern Preferences ---");
        let mut pattern_prefs: std::collections::HashMap<String, Vec<(String, f64)>> =
            std::collections::HashMap::new();
        for (pattern, action, mean, evidence, _) in &arm_report {
            if *evidence > 5.0 {
                pattern_prefs
                    .entry(pattern.clone())
                    .or_default()
                    .push((action.clone(), *mean));
            }
        }
        for (pattern, mut actions) in pattern_prefs {
            actions.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
            if let Some((best_action, best_mean)) = actions.first() {
                let second_mean = actions.get(1).map(|a| a.1).unwrap_or(0.0);
                let edge = best_mean - second_mean;
                println!(
                    "  {:12} → {:>6} (mean={:.4}, edge={:+.4})",
                    pattern, best_action, best_mean, edge
                );
            }
        }
    }

    // ===== EXCURSION TRACKER STATE =====
    println!("\n{}", "=".repeat(70));
    println!("  EXCURSION TRACKER — LEARNED SL/TP (ATR multiples)");
    println!("{}", "=".repeat(70));
    let excursion_report = engine.learning.excursions.report();
    if excursion_report.is_empty() {
        println!("  No excursion data yet.");
    } else {
        println!(
            "  {:20} {:>8} {:>8} {:>8} {:>8}",
            "Pattern+Side", "SL(ATR)", "TP(ATR)", "AdvObs", "FavObs"
        );
        for (key, sl, tp, adv_n, fav_n) in &excursion_report {
            let display = if key.len() > 20 {
                &key[..20]
            } else {
                key.as_str()
            };
            println!(
                "  {:20} {:>8.3} {:>8.3} {:>8} {:>8}",
                display, sl, tp, adv_n, fav_n
            );
        }
    }

    // ===== ADAPTIVE PARAMETERS — RAW EMAs =====
    println!("\n{}", "=".repeat(70));
    println!("  ADAPTIVE PARAMETERS — RAW EMA STATE");
    println!("{}", "=".repeat(70));
    let s = &engine.learning.adaptive.stats;
    println!("  EMA PnL:              {:>10.4}%", s.ema_pnl);
    println!("  EMA PnL²:             {:>10.4}", s.ema_pnl_sq);
    println!(
        "  PnL Std (derived):    {:>10.4}%",
        (s.ema_pnl_sq - s.ema_pnl * s.ema_pnl).max(0.01).sqrt()
    );
    println!("  EMA Win Duration:     {:>10.1} candles", s.ema_win_duration);
    println!("  EMA Loss Duration:    {:>10.1} candles", s.ema_loss_duration);
    println!(
        "  Win/Loss Dur Ratio:   {:>10.2}",
        s.ema_win_duration / s.ema_loss_duration.max(0.01)
    );
    println!("  EMA Favorable (ATR):  {:>10.3}", s.ema_favorable);
    println!("  EMA Adverse (ATR):    {:>10.3}", s.ema_adverse);
    println!("  EMA RR Ratio:         {:>10.3}", s.ema_rr_ratio);
    println!("  EMA Decay:            {:>10.4}", s.ema_decay);
    println!("  EMA KNN Accuracy:     {:>10.4}", s.ema_knn_accuracy);
    println!("  Total Candles Seen:   {:>10}", s.total_candles_seen);

    // ===== PATTERN DISTRIBUTION =====
    println!("\n{}", "=".repeat(70));
    println!("  PATTERN DISTRIBUTION (across all trades)");
    println!("{}", "=".repeat(70));
    {
        let mut pattern_counts: std::collections::HashMap<String, (usize, usize, f64)> =
            std::collections::HashMap::new();
        for t in &result.trades {
            if t.pattern.is_empty() {
                continue;
            }
            let e = pattern_counts
                .entry(t.pattern.clone())
                .or_insert((0, 0, 0.0));
            e.0 += 1;
            if t.pnl_pct > 0.0 {
                e.1 += 1;
            }
            e.2 += t.pnl_pct;
        }
        let mut patterns: Vec<_> = pattern_counts.into_iter().collect();
        patterns.sort_by(|a, b| b.1 .2.partial_cmp(&a.1 .2).unwrap());

        println!(
            "  {:12} {:>6} {:>6} {:>8} {:>8}",
            "Pattern", "Count", "Win%", "AvgPnL%", "TotPnL%"
        );
        for (pattern, (count, wins, total_pnl)) in &patterns {
            let wr = if *count > 0 {
                *wins as f64 / *count as f64 * 100.0
            } else {
                0.0
            };
            let avg = if *count > 0 {
                total_pnl / *count as f64
            } else {
                0.0
            };
            println!(
                "  {:12} {:>6} {:>5.1}% {:>+7.3}% {:>+7.2}%",
                pattern, count, wr, avg, total_pnl
            );
        }

        // Long vs short preference by pattern
        let mut pattern_sides: std::collections::HashMap<
            String,
            (usize, usize, f64, f64),
        > = std::collections::HashMap::new();
        for t in &result.trades {
            if t.pattern.is_empty() {
                continue;
            }
            let e = pattern_sides
                .entry(t.pattern.clone())
                .or_insert((0, 0, 0.0, 0.0));
            match t.side {
                rdex::domain::PositionSide::Long => {
                    e.0 += 1;
                    e.2 += t.pnl_pct;
                }
                rdex::domain::PositionSide::Short => {
                    e.1 += 1;
                    e.3 += t.pnl_pct;
                }
                _ => {}
            }
        }
        if !pattern_sides.is_empty() {
            println!("\n  Per-Pattern Long/Short Split:");
            println!(
                "  {:12} {:>6} {:>8}  |  {:>6} {:>8}",
                "Pattern", "Long#", "LongPnL%", "Short#", "ShortPnL%"
            );
            let mut sides: Vec<_> = pattern_sides.into_iter().collect();
            sides.sort_by(|a, b| a.0.cmp(&b.0));
            for (pattern, (longs, shorts, long_pnl, short_pnl)) in &sides {
                println!(
                    "  {:12} {:>6} {:>+7.2}%  |  {:>6} {:>+7.2}%",
                    pattern, longs, long_pnl, shorts, short_pnl
                );
            }
        }
    }

    // ===== CROSS-SYMBOL COMPARISON =====
    if symbols.len() > 1 {
        println!("\n{}", "=".repeat(70));
        println!("  CROSS-SYMBOL COMPARISON");
        println!("{}", "=".repeat(70));
        println!(
            "  {:10} {:>8} {:>7} {:>7} {:>8} {:>8} {:>8} {:>9} {:>9}",
            "Symbol", "PnL%", "Trades", "WinR%", "AvgWin%", "AvgLoss%", "PF", "TxFees", "FundFees"
        );
        for symbol in &symbols {
            let sym_trades: Vec<&rdex::backtest::portfolio::TradeRecord> = result
                .trades
                .iter()
                .filter(|t| t.symbol == *symbol)
                .collect();
            let n = sym_trades.len();
            if n == 0 {
                println!("  {:10}     0.00%       0    0.0%        -        -        - $    0.00 $    0.00", symbol);
                continue;
            }
            let sym_pnl: f64 = sym_trades.iter().map(|t| t.pnl_pct).sum();
            let sym_wins = sym_trades.iter().filter(|t| t.pnl_pct > 0.0).count();
            let sym_wr = sym_wins as f64 / n as f64 * 100.0;
            let winning: Vec<f64> = sym_trades
                .iter()
                .filter(|t| t.pnl_pct > 0.0)
                .map(|t| t.pnl_pct)
                .collect();
            let losing: Vec<f64> = sym_trades
                .iter()
                .filter(|t| t.pnl_pct <= 0.0)
                .map(|t| t.pnl_pct)
                .collect();
            let avg_win = if !winning.is_empty() {
                winning.iter().sum::<f64>() / winning.len() as f64
            } else {
                0.0
            };
            let avg_loss = if !losing.is_empty() {
                losing.iter().sum::<f64>() / losing.len() as f64
            } else {
                0.0
            };
            let gross_profit: f64 = winning.iter().sum();
            let gross_loss: f64 = losing.iter().map(|l| l.abs()).sum();
            let pf = if gross_loss > 0.0 {
                gross_profit / gross_loss
            } else if gross_profit > 0.0 {
                f64::INFINITY
            } else {
                0.0
            };
            let sym_tx: f64 = sym_trades.iter().map(|t| t.fees_paid).sum();
            let sym_fund: f64 = sym_trades.iter().map(|t| t.funding_fees_paid).sum();
            let sym_name = if symbol.len() > 10 {
                &symbol[..10]
            } else {
                symbol.as_str()
            };
            println!(
                "  {:10} {:>+7.2}% {:>7} {:>6.1}% {:>+7.2}% {:>+7.2}% {:>8.2} ${:>8.2} ${:>8.2}",
                sym_name, sym_pnl, n, sym_wr, avg_win, avg_loss, pf, sym_tx, sym_fund
            );
        }

        // Best and worst by P&L contribution
        let mut sym_pnls: Vec<(&String, f64)> = symbols
            .iter()
            .map(|s| {
                let pnl: f64 = result
                    .trades
                    .iter()
                    .filter(|t| t.symbol == *s)
                    .map(|t| t.pnl_pct)
                    .sum();
                (s, pnl)
            })
            .collect();
        sym_pnls.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        if let (Some(best), Some(worst)) = (sym_pnls.first(), sym_pnls.last()) {
            println!(
                "\n  Best:  {} ({:+.2}%)  |  Worst: {} ({:+.2}%)",
                best.0, best.1, worst.0, worst.1
            );
        }

        // Consistency: how many symbols are profitable
        let profitable = sym_pnls.iter().filter(|(_, pnl)| *pnl > 0.0).count();
        println!(
            "  Profitable symbols: {}/{} ({:.0}%)",
            profitable,
            symbols.len(),
            profitable as f64 / symbols.len() as f64 * 100.0
        );
    }

    // ===== FULL TRADE LOG =====
    println!("\n{}", "#".repeat(80));
    println!("  FULL TRADE LOG (all symbols, interleaved)");
    println!("{}", "#".repeat(80));
    for symbol in &symbols {
        let sym_trades: Vec<&rdex::backtest::portfolio::TradeRecord> = result
            .trades
            .iter()
            .filter(|t| t.symbol == *symbol)
            .collect();
        if sym_trades.is_empty() {
            continue;
        }
        println!("\n  --- {} ({} trades) ---", symbol, sym_trades.len());
        // Print individual trades
        for t in &sym_trades {
            let side_str = match t.side {
                rdex::domain::PositionSide::Long => "LONG",
                rdex::domain::PositionSide::Short => "SHORT",
                _ => "FLAT",
            };
            println!(
                "    {} {:>5} entry={:.2} exit={:.2} pnl={:+.2}% size=${:.0} hold={} exit={} pat={} mae={:.3} mfe={:.3}",
                t.symbol, side_str, t.entry_price, t.exit_price,
                t.pnl_pct, t.size_usd, t.holding_periods, t.exit_reason,
                t.pattern, t.max_adverse_excursion, t.max_favorable_excursion
            );
        }
    }

    // Learning health
    println!("\n--- Learning Health ---");
    for (sym, health) in &engine.learning.health_report() {
        println!(
            "  {}: obs={} avg_regret={:.4} growth={:.2} learning={}",
            sym,
            health.observations,
            health.average_regret,
            health.regret_growth_rate,
            health.is_learning
        );
    }

    println!("\n=== Backtest Complete ===");
    Ok(())
}
