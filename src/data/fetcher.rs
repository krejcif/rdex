use crate::domain::{Candle, FundingRate};
use chrono::{Duration, Utc};

const BINANCE_FUTURES_URL: &str = "https://fapi.binance.com/fapi/v1/klines";
const BINANCE_FUNDING_URL: &str = "https://fapi.binance.com/fapi/v1/fundingRate";

/// Fetch historical klines from Binance Futures (no API key needed)
pub async fn fetch_candles(
    symbol: &str,
    interval: &str,
    start_time: i64,
    end_time: i64,
) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut all_candles = Vec::new();
    let mut current_start = start_time;
    let limit = 1500; // Binance max per request

    while current_start < end_time {
        let resp = client
            .get(BINANCE_FUTURES_URL)
            .query(&[
                ("symbol", symbol),
                ("interval", interval),
                ("startTime", &current_start.to_string()),
                ("endTime", &end_time.to_string()),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Binance API error {}: {}", status, body).into());
        }

        let data: Vec<Vec<serde_json::Value>> = resp.json().await?;

        if data.is_empty() {
            break;
        }

        for kline in &data {
            if kline.len() < 11 {
                continue;
            }

            let candle = Candle {
                open_time: kline[0].as_i64().unwrap_or(0),
                open: kline[1].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                high: kline[2].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                low: kline[3].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                close: kline[4].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                volume: kline[5].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                close_time: kline[6].as_i64().unwrap_or(0),
                quote_volume: kline[7].as_str().unwrap_or("0").parse().unwrap_or(0.0),
                trades: kline[8].as_u64().unwrap_or(0),
            };
            all_candles.push(candle);
        }

        // Move to next batch
        let last_time = data.last().and_then(|k| k[6].as_i64()).unwrap_or(end_time);
        current_start = last_time + 1;

        // Rate limit: be nice to Binance
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    // Ensure temporal ordering
    all_candles.sort_by_key(|c| c.open_time);

    Ok(all_candles)
}

/// Fetch last N days of 15m candles for a symbol
pub async fn fetch_last_n_days(
    symbol: &str,
    days: i64,
) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
    let now = Utc::now();
    let start = now - Duration::days(days);

    fetch_candles(
        symbol,
        "15m",
        start.timestamp_millis(),
        now.timestamp_millis(),
    )
    .await
}

/// Fetch candles for multiple symbols
pub async fn fetch_multi_symbol(
    symbols: &[&str],
    days: i64,
) -> Result<Vec<(String, Vec<Candle>)>, Box<dyn std::error::Error>> {
    let mut results = Vec::new();

    for symbol in symbols {
        println!("Fetching {} ({} days)...", symbol, days);
        match fetch_last_n_days(symbol, days).await {
            Ok(candles) => {
                println!("  {} candles fetched", candles.len());
                results.push((symbol.to_string(), candles));
            }
            Err(e) => {
                eprintln!("  Error fetching {}: {}", symbol, e);
            }
        }
    }

    Ok(results)
}

/// Fetch historical funding rates from Binance Futures
pub async fn fetch_funding_rates(
    symbol: &str,
    start_time: i64,
    end_time: i64,
) -> Result<Vec<FundingRate>, Box<dyn std::error::Error>> {
    let client = reqwest::Client::new();
    let mut all_rates = Vec::new();
    let mut current_start = start_time;
    let limit = 1000; // Binance max per request

    while current_start < end_time {
        let resp = client
            .get(BINANCE_FUNDING_URL)
            .query(&[
                ("symbol", symbol),
                ("startTime", &current_start.to_string()),
                ("endTime", &end_time.to_string()),
                ("limit", &limit.to_string()),
            ])
            .send()
            .await?;

        if !resp.status().is_success() {
            let status = resp.status();
            let body = resp.text().await.unwrap_or_default();
            return Err(format!("Binance funding API error {}: {}", status, body).into());
        }

        let data: Vec<serde_json::Value> = resp.json().await?;
        if data.is_empty() {
            break;
        }

        for item in &data {
            let rate = FundingRate {
                symbol: item["symbol"].as_str().unwrap_or(symbol).to_string(),
                funding_time: item["fundingTime"].as_i64().unwrap_or(0),
                funding_rate: item["fundingRate"]
                    .as_str()
                    .unwrap_or("0")
                    .parse()
                    .unwrap_or(0.0),
            };
            all_rates.push(rate);
        }

        current_start = data
            .last()
            .and_then(|r| r["fundingTime"].as_i64())
            .unwrap_or(end_time)
            + 1;

        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }

    all_rates.sort_by_key(|r| r.funding_time);
    Ok(all_rates)
}

/// Fetch last N days of funding rates for a symbol
pub async fn fetch_funding_last_n_days(
    symbol: &str,
    days: i64,
) -> Result<Vec<FundingRate>, Box<dyn std::error::Error>> {
    let now = Utc::now();
    let start = now - Duration::days(days);
    fetch_funding_rates(symbol, start.timestamp_millis(), now.timestamp_millis()).await
}
