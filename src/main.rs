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

    // Fetch candle data + funding rates
    let mut all_data: Vec<(String, Vec<Candle>)> = Vec::new();
    let mut all_funding: std::collections::HashMap<String, Vec<FundingRate>> =
        std::collections::HashMap::new();
    for symbol in &symbols {
        let candles = rdex::data::load_or_fetch(symbol, days, "data").await?;
        println!(
            "{}: {} candles ({:.1} days)",
            symbol,
            candles.len(),
            candles.len() as f64 / 96.0
        );

        let funding = rdex::data::load_or_fetch_funding(symbol, days, "data")
            .await
            .unwrap_or_else(|e| {
                eprintln!("  Warning: could not fetch funding for {}: {}", symbol, e);
                Vec::new()
            });
        if !funding.is_empty() {
            println!("  {} funding rates loaded", funding.len());
        }

        all_data.push((symbol.clone(), candles));
        all_funding.insert(symbol.clone(), funding);
    }

    let config = FuturesConfig {
        leverage,
        initial_equity: equity,
        ..Default::default()
    };
    let learner_config = LearnerConfig::default();

    // ===== PHASE 1: Sequential multi-symbol learning =====
    println!("\n========== PHASE 1: SEQUENTIAL LEARNING (all symbols) ==========");
    let mut shared_learning = LearningEngine::new(symbols.clone(), learner_config.clone());
    let mut phase1_results: Vec<(String, engine::BacktestResult)> = Vec::new();

    for (symbol, candles) in &all_data {
        println!(
            "\n--- Learning on {} ({} candles) ---",
            symbol,
            candles.len()
        );
        let mut bt = engine::BacktestEngine::new(config.clone(), shared_learning, 42);
        if let Some(funding) = all_funding.get(symbol) {
            bt.set_funding_rates(funding);
        }
        let result = bt.run(&Symbol(symbol.clone()), candles, 60);
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
        shared_learning = bt.learning;
        let transfers = shared_learning.maybe_transfer();
        for (src, tgt) in &transfers {
            println!("  Transfer: {} -> {}", src, tgt);
        }
        phase1_results.push((symbol.clone(), result));
    }

    // Print Phase 1 summary
    println!("\n========== PHASE 1 RESULTS ==========");
    println!(
        "  {:10} {:>8} {:>7} {:>7} {:>7} {:>10} {:>10}",
        "Symbol", "Return%", "Trades", "Sharpe", "MaxDD%", "TX Fees", "FundFees"
    );
    for (symbol, result) in &phase1_results {
        let tx: f64 = result.trades.iter().map(|t| t.fees_paid).sum();
        let fund: f64 = result.trades.iter().map(|t| t.funding_fees_paid).sum();
        println!(
            "  {:10} {:>+7.2}% {:>7} {:>7.2} {:>6.2}% ${:>9.2} ${:>9.2}",
            symbol,
            result.performance.total_return_pct,
            result.trades.len(),
            result.performance.sharpe_ratio,
            result.performance.max_drawdown_pct,
            tx,
            fund
        );
    }

    // Learning health
    println!("\n--- Learning Health ---");
    for (sym, health) in &shared_learning.health_report() {
        println!(
            "  {}: obs={} avg_regret={:.4} growth={:.2} learning={}",
            sym,
            health.observations,
            health.average_regret,
            health.regret_growth_rate,
            health.is_learning
        );
    }

    // ===== PHASE 2: Exploit learned patterns =====
    println!("\n========== PHASE 2: EXPLOITATION (trained engine) ==========");
    let mut phase2_results: Vec<(String, engine::BacktestResult)> = Vec::new();

    for (symbol, candles) in &all_data {
        let mut bt = engine::BacktestEngine::new(config.clone(), shared_learning.clone(), 43);
        if let Some(funding) = all_funding.get(symbol) {
            bt.set_funding_rates(funding);
        }
        let result = bt.run(&Symbol(symbol.clone()), candles, 60);
        phase2_results.push((symbol.clone(), result));
        shared_learning = bt.learning;
    }

    // ===== PHASE 2 DETAILED RESULTS =====
    println!("\n{}", "#".repeat(80));
    println!("  PHASE 2 DETAILED RESULTS");
    println!("{}", "#".repeat(80));

    let mut total_pnl = 0.0;
    let mut all_phase2_trades: Vec<&rdex::backtest::portfolio::TradeRecord> = Vec::new();

    for (symbol, result) in &phase2_results {
        println!("\n  ---- {} ----", symbol);
        result.print_summary();
        total_pnl += result.performance.total_return_pct;
        all_phase2_trades.extend(result.trades.iter());

        // Walk-forward validation
        let candles = &all_data.iter().find(|(s, _)| s == symbol).unwrap().1;
        println!("Walk-forward validation (3 folds)...");
        let validation = walk_forward_validation(
            &Symbol(symbol.clone()),
            candles,
            &config,
            &learner_config,
            3,
            0.7,
        );
        validation.print_summary();

        // Permutation test — actual_return computed the same way as permuted
        // returns (compounded trade pnl_pct) for apples-to-apples comparison
        if !result.trades.is_empty() {
            let pnls: Vec<f64> = result.trades.iter().map(|t| t.pnl_pct).collect();
            let actual_compounded = pnls.iter().fold(1.0, |acc, &pnl| acc * (1.0 + pnl / 100.0));
            let actual_return = (actual_compounded - 1.0) * 100.0;
            let perm = permutation_test(&pnls, actual_return, 1000, 42);
            perm.print_summary();
        }
    }

    // ===== ADAPTIVE PARAMETERS STATE =====
    println!("\n{}", "=".repeat(70));
    println!("  ADAPTIVE PARAMETERS (final state)");
    println!("{}", "=".repeat(70));
    let a = &shared_learning.adaptive;
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
    let avg_return = total_pnl / symbols.len() as f64;
    let total_trades: usize = phase2_results.iter().map(|r| r.1.trades.len()).sum();
    let total_wins: usize = phase2_results
        .iter()
        .map(|r| r.1.performance.winning_trades)
        .sum();
    let portfolio_wr = if total_trades > 0 {
        total_wins as f64 / total_trades as f64 * 100.0
    } else {
        0.0
    };
    let avg_sharpe: f64 = phase2_results
        .iter()
        .map(|r| r.1.performance.sharpe_ratio)
        .sum::<f64>()
        / symbols.len() as f64;
    let max_dd: f64 = phase2_results
        .iter()
        .map(|r| r.1.performance.max_drawdown_pct)
        .fold(0.0, f64::max);
    let total_fees: f64 = all_phase2_trades.iter().map(|t| t.fees_paid).sum();
    let total_funding: f64 = all_phase2_trades.iter().map(|t| t.funding_fees_paid).sum();

    println!("  Symbols:              {:>10}", symbols.len());
    println!("  Avg Return:           {:>+9.2}%", avg_return);
    println!("  Total Trades:         {:>10}", total_trades);
    println!("  Portfolio Win Rate:   {:>9.1}%", portfolio_wr);
    println!("  Avg Sharpe:           {:>10.2}", avg_sharpe);
    println!("  Worst Max Drawdown:   {:>9.2}%", max_dd);
    println!("  Total TX Fees:        ${:>9.2}", total_fees);
    println!("  Total Funding Fees:   ${:>9.2}", total_funding);

    // Per-symbol one-liner
    println!("\n  Per-Symbol Breakdown:");
    println!(
        "  {:10} {:>8} {:>7} {:>7} {:>7} {:>8} {:>7} {:>7} {:>9} {:>9}",
        "Symbol",
        "Return%",
        "Trades",
        "WinR%",
        "Sharpe",
        "MaxDD%",
        "PF",
        "Expect",
        "TxFees",
        "FundFees"
    );
    for (symbol, result) in &phase2_results {
        let p = &result.performance;
        let sym = if symbol.len() > 10 {
            &symbol[..10]
        } else {
            symbol
        };
        let sym_tx: f64 = result.trades.iter().map(|t| t.fees_paid).sum();
        let sym_fund: f64 = result.trades.iter().map(|t| t.funding_fees_paid).sum();
        println!(
            "  {:10} {:>+7.2}% {:>7} {:>6.1}% {:>7.2} {:>7.2}% {:>7.2} {:>+6.3}% ${:>8.2} ${:>8.2}",
            sym,
            p.total_return_pct,
            p.total_trades,
            p.win_rate * 100.0,
            p.sharpe_ratio,
            p.max_drawdown_pct,
            p.profit_factor,
            p.expectancy,
            sym_tx,
            sym_fund
        );
    }

    // ===== FULL TRADE LOG =====
    println!("\n{}", "#".repeat(80));
    println!("  FULL TRADE LOG (all symbols, Phase 2)");
    println!("{}", "#".repeat(80));
    for (symbol, result) in &phase2_results {
        if result.trades.is_empty() {
            continue;
        }
        println!("\n  --- {} ({} trades) ---", symbol, result.trades.len());
        result.print_trade_log();
    }

    println!("\n=== Backtest Complete ===");
    Ok(())
}
