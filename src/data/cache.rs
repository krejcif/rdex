use crate::domain::{Candle, FundingRate};
use std::path::Path;

/// Save candles to CSV cache
pub fn save_to_csv(candles: &[Candle], path: &str) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = csv::Writer::from_path(path)?;

    writer.write_record(&[
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "trades",
    ])?;

    for c in candles {
        writer.write_record(&[
            c.open_time.to_string(),
            c.open.to_string(),
            c.high.to_string(),
            c.low.to_string(),
            c.close.to_string(),
            c.volume.to_string(),
            c.close_time.to_string(),
            c.quote_volume.to_string(),
            c.trades.to_string(),
        ])?;
    }

    writer.flush()?;
    Ok(())
}

/// Load candles from CSV cache
pub fn load_from_csv(path: &str) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("Cache file not found: {}", path).into());
    }

    let mut reader = csv::Reader::from_path(path)?;
    let mut candles = Vec::new();

    for result in reader.records() {
        let record = result?;
        let candle = Candle {
            open_time: record[0].parse()?,
            open: record[1].parse()?,
            high: record[2].parse()?,
            low: record[3].parse()?,
            close: record[4].parse()?,
            volume: record[5].parse()?,
            close_time: record[6].parse()?,
            quote_volume: record[7].parse()?,
            trades: record[8].parse()?,
        };
        candles.push(candle);
    }

    // Verify temporal ordering
    for i in 1..candles.len() {
        if candles[i].open_time < candles[i - 1].open_time {
            return Err("Cache file has non-monotonic timestamps".into());
        }
    }

    Ok(candles)
}

/// Get cache path for a symbol
pub fn cache_path(symbol: &str, data_dir: &str) -> String {
    format!("{}/{}_15m.csv", data_dir, symbol.to_lowercase())
}

/// Load from cache or fetch
pub async fn load_or_fetch(
    symbol: &str,
    days: i64,
    data_dir: &str,
) -> Result<Vec<Candle>, Box<dyn std::error::Error>> {
    let path = cache_path(symbol, data_dir);

    // Try cache first
    if let Ok(candles) = load_from_csv(&path) {
        if !candles.is_empty() {
            let age_ms = chrono::Utc::now().timestamp_millis() - candles.last().unwrap().close_time;
            let age_hours = age_ms as f64 / 3_600_000.0;

            // Use cache if less than 24 hours old (historical data doesn't change)
            if age_hours < 24.0 {
                println!(
                    "Using cached data for {} ({} candles, {:.1}h old)",
                    symbol,
                    candles.len(),
                    age_hours
                );
                return Ok(candles);
            }
        }
    }

    // Fetch from Binance
    let candles = super::fetcher::fetch_last_n_days(symbol, days).await?;

    // Save to cache
    std::fs::create_dir_all(data_dir)?;
    save_to_csv(&candles, &path)?;
    println!(
        "Cached {} candles for {} at {}",
        candles.len(),
        symbol,
        path
    );

    Ok(candles)
}

/// Save funding rates to CSV cache
pub fn save_funding_csv(
    rates: &[FundingRate],
    path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut writer = csv::Writer::from_path(path)?;
    writer.write_record(&["symbol", "funding_time", "funding_rate"])?;
    for r in rates {
        writer.write_record(&[
            r.symbol.clone(),
            r.funding_time.to_string(),
            r.funding_rate.to_string(),
        ])?;
    }
    writer.flush()?;
    Ok(())
}

/// Load funding rates from CSV cache
pub fn load_funding_csv(path: &str) -> Result<Vec<FundingRate>, Box<dyn std::error::Error>> {
    if !Path::new(path).exists() {
        return Err(format!("Funding cache not found: {}", path).into());
    }
    let mut reader = csv::Reader::from_path(path)?;
    let mut rates = Vec::new();
    for result in reader.records() {
        let record = result?;
        rates.push(FundingRate {
            symbol: record[0].to_string(),
            funding_time: record[1].parse()?,
            funding_rate: record[2].parse()?,
        });
    }
    rates.sort_by_key(|r| r.funding_time);
    Ok(rates)
}

/// Get funding cache path
pub fn funding_cache_path(symbol: &str, data_dir: &str) -> String {
    format!("{}/{}_funding.csv", data_dir, symbol.to_lowercase())
}

/// Load funding rates from cache or fetch from Binance
pub async fn load_or_fetch_funding(
    symbol: &str,
    days: i64,
    data_dir: &str,
) -> Result<Vec<FundingRate>, Box<dyn std::error::Error>> {
    let path = funding_cache_path(symbol, data_dir);

    if let Ok(rates) = load_funding_csv(&path) {
        if !rates.is_empty() {
            let age_ms = chrono::Utc::now().timestamp_millis() - rates.last().unwrap().funding_time;
            let age_hours = age_ms as f64 / 3_600_000.0;
            if age_hours < 24.0 {
                println!(
                    "Using cached funding for {} ({} rates, {:.1}h old)",
                    symbol,
                    rates.len(),
                    age_hours
                );
                return Ok(rates);
            }
        }
    }

    let rates = super::fetcher::fetch_funding_last_n_days(symbol, days).await?;
    std::fs::create_dir_all(data_dir)?;
    save_funding_csv(&rates, &path)?;
    println!("Cached {} funding rates for {}", rates.len(), symbol);
    Ok(rates)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_csv_round_trip() {
        let candles = vec![
            Candle {
                open_time: 1000,
                open: 100.0,
                high: 101.0,
                low: 99.0,
                close: 100.5,
                volume: 500.0,
                close_time: 1999,
                quote_volume: 50000.0,
                trades: 100,
            },
            Candle {
                open_time: 2000,
                open: 100.5,
                high: 102.0,
                low: 100.0,
                close: 101.0,
                volume: 600.0,
                close_time: 2999,
                quote_volume: 60000.0,
                trades: 120,
            },
        ];

        let dir = tempfile::tempdir().unwrap();
        let path = format!("{}/test.csv", dir.path().display());

        save_to_csv(&candles, &path).unwrap();
        let loaded = load_from_csv(&path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].open_time, 1000);
        assert!((loaded[1].close - 101.0).abs() < 1e-10);
    }

    #[test]
    fn test_load_nonexistent_file() {
        let result = load_from_csv("/tmp/does_not_exist_rdex_test.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_cache_path_format() {
        let path = cache_path("BTCUSDT", "/data");
        assert_eq!(path, "/data/btcusdt_15m.csv");
    }

    #[test]
    fn test_funding_cache_path_format() {
        let path = funding_cache_path("ETHUSDT", "/data");
        assert_eq!(path, "/data/ethusdt_funding.csv");
    }

    #[test]
    fn test_empty_candles_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let path = format!("{}/empty.csv", dir.path().display());
        save_to_csv(&[], &path).unwrap();
        let loaded = load_from_csv(&path).unwrap();
        assert!(loaded.is_empty());
    }

    #[test]
    fn test_funding_csv_round_trip() {
        let rates = vec![
            FundingRate { symbol: "BTCUSDT".into(), funding_time: 1000, funding_rate: 0.0001 },
            FundingRate { symbol: "BTCUSDT".into(), funding_time: 2000, funding_rate: -0.0002 },
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = format!("{}/funding.csv", dir.path().display());
        save_funding_csv(&rates, &path).unwrap();
        let loaded = load_funding_csv(&path).unwrap();
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].symbol, "BTCUSDT");
        assert!((loaded[0].funding_rate - 0.0001).abs() < 1e-10);
        assert!((loaded[1].funding_rate - (-0.0002)).abs() < 1e-10);
    }

    #[test]
    fn test_load_funding_nonexistent() {
        let result = load_funding_csv("/tmp/does_not_exist_rdex_funding.csv");
        assert!(result.is_err());
    }

    #[test]
    fn test_csv_preserves_all_fields() {
        let candle = Candle {
            open_time: 1700000000000,
            open: 42567.89,
            high: 42890.12,
            low: 42100.45,
            close: 42750.33,
            volume: 12345.678,
            close_time: 1700000899999,
            quote_volume: 987654.321,
            trades: 54321,
        };
        let dir = tempfile::tempdir().unwrap();
        let path = format!("{}/fields.csv", dir.path().display());
        save_to_csv(&[candle.clone()], &path).unwrap();
        let loaded = load_from_csv(&path).unwrap();
        assert_eq!(loaded[0].open_time, candle.open_time);
        assert!((loaded[0].open - candle.open).abs() < 0.001);
        assert!((loaded[0].high - candle.high).abs() < 0.001);
        assert!((loaded[0].low - candle.low).abs() < 0.001);
        assert!((loaded[0].close - candle.close).abs() < 0.001);
        assert!((loaded[0].volume - candle.volume).abs() < 0.001);
        assert_eq!(loaded[0].close_time, candle.close_time);
        assert_eq!(loaded[0].trades, candle.trades);
    }

    #[test]
    fn test_funding_csv_sorted_on_load() {
        let rates = vec![
            FundingRate { symbol: "BTC".into(), funding_time: 3000, funding_rate: 0.0003 },
            FundingRate { symbol: "BTC".into(), funding_time: 1000, funding_rate: 0.0001 },
            FundingRate { symbol: "BTC".into(), funding_time: 2000, funding_rate: 0.0002 },
        ];
        let dir = tempfile::tempdir().unwrap();
        let path = format!("{}/unsorted.csv", dir.path().display());
        save_funding_csv(&rates, &path).unwrap();
        let loaded = load_funding_csv(&path).unwrap();
        // Should be sorted by funding_time
        assert_eq!(loaded[0].funding_time, 1000);
        assert_eq!(loaded[1].funding_time, 2000);
        assert_eq!(loaded[2].funding_time, 3000);
    }
}
