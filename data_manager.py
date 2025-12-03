"""
Intelligent Data Manager for Cross-Sectional Momentum Strategy

Implements incremental data downloading:
- Downloads ETF OHLCV data to cache directory
- Downloads macro data (VIX, yields)
- Downloads FRED data (sentiment, conditions, credit spreads)
- Only downloads missing dates
- Preserves existing data and merges intelligently

Based on alpha_signals data_manager.py philosophy.
"""

from __future__ import annotations
import datetime as dt
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Callable
import warnings
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import yfinance as yf
import logging

try:
    import schedule  # For DataRefreshDaemon
    SCHEDULE_AVAILABLE = True
except ImportError:
    SCHEDULE_AVAILABLE = False


# ============================================================================
# FRED SERIES DEFINITIONS
# ============================================================================
# These are the FRED series used for macro/sentiment features

FRED_SERIES = {
    # ---- Interest Rates & Yield Curve ----
    'DGS3MO': 'tsy_3mo',                # 3-Month Treasury (short rate)
    'DGS2': 'tsy_2y',                   # 2-Year Treasury
    'DGS10': 'tsy_10y',                 # 10-Year Treasury
    'DGS30': 'tsy_30y',                 # 30-Year Treasury
    'T10Y2Y': 'yc_slope_10_2',          # 10Y-2Y Spread (pre-calculated)
    
    # ---- Credit Spreads ----
    'BAMLC0A0CM': 'ig_spread',          # IG Corporate Bond Spread
    'BAMLH0A0HYM2': 'hy_spread',        # HY Corporate Bond Spread
    'DBAA': 'baa_yield',                # Baa Corporate Bond Yield
    'DAAA': 'aaa_yield',                # Aaa Corporate Bond Yield
    
    # ---- Inflation ----
    'T10YIE': 'breakeven_10y',          # 10Y Breakeven Inflation (daily)
    'CPIAUCSL': 'cpi_index',            # CPI All Urban Consumers (monthly)
    'PCEPI': 'pce_index',               # PCE Chain Price Index (monthly)
    
    # ---- Growth / Real Economy ----
    'INDPRO': 'indpro',                 # Industrial Production Index (monthly)
    'RSXFS': 'retail_sales',            # Advance Retail Sales (monthly)
    'UNRATE': 'unemployment_rate',      # Unemployment Rate (monthly)
    
    # ---- Financial Conditions ----
    'NFCI': 'chicago_fci',              # Chicago Fed National FCI
    'STLFSI4': 'stl_fsi',               # St. Louis Fed FSI
    
    # ---- Sentiment / Uncertainty ----
    'UMCSENT': 'umich_sentiment',       # U. Michigan Consumer Sentiment
    'USEPUINDXD': 'epu_daily',          # Economic Policy Uncertainty (daily)
    
    # ---- Labor Market ----
    'ICSA': 'initial_claims',           # Initial Jobless Claims (weekly)
}


# Default FRED API key (can be overridden by environment variable)
DEFAULT_FRED_API_KEY = "07aa386a7dd9c675f406da5e79561b8b"

# Benchmark tickers for cross-asset features
BENCHMARK_TICKERS = ['SPY', 'TLT', 'GLD']

# Special tickers for macro/risk features (must be downloaded via yfinance)
# These are used for rolling betas, correlations, and macro regime features
# NOTE: BNDW is NOT here - we create synthetic BNDW from BND+BNDX in the ETF universe
#       (BND+BNDX average has 98.45% correlation with actual BNDW, 0.055% tracking error)
MACRO_REFERENCE_TICKERS = {
    '^VIX': 'vix',       # CBOE Volatility Index (equity vol)
    '^MOVE': 'move',     # ICE BofA MOVE Index (bond vol)
    'VT': 'vt',          # Vanguard Total World Stock (global equity)
}


class CrossSecMomDataManager:
    """Manages data downloading and caching with incremental updates."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
        fred_api_key: Optional[str] = None,
    ):
        """
        Initialize data manager.

        Args:
            data_dir: Directory for data storage. If None, uses default from config.
            fred_api_key: FRED API key. If None, uses environment variable or default.
        """
        if data_dir is None:
            # Import here to avoid circular dependency
            from config import get_default_config
            data_dir = get_default_config().paths.data_dir
        
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Subdirectories for different data types
        self.ohlcv_dir = self.data_dir / "ohlcv"
        self.ohlcv_dir.mkdir(parents=True, exist_ok=True)
        
        self.macro_dir = self.data_dir / "macro"
        self.macro_dir.mkdir(parents=True, exist_ok=True)
        
        self.fred_dir = self.data_dir / "fred"
        self.fred_dir.mkdir(parents=True, exist_ok=True)
        
        self.benchmark_dir = self.data_dir / "benchmarks"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # FRED API key: env var > parameter > default
        self.fred_api_key = os.environ.get('FRED_API_KEY') or fred_api_key or DEFAULT_FRED_API_KEY
        
        # FRED API client (lazy initialization)
        self._fred = None
        
        # Suppress yfinance's verbose error messages
        logging.getLogger('yfinance').setLevel(logging.CRITICAL)

    def load_or_download_etf_prices(
        self,
        tickers: List[str],
        start_date: str,
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load ETF price data from disk or download if missing/outdated.

        Implements intelligent incremental updates:
        - If files exist, only downloads new dates
        - If new tickers requested, downloads and merges them
        - Preserves datetime index and sorting

        Args:
            tickers: List of ETF ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD), None = today

        Returns:
            Tuple of (prices DataFrame, returns DataFrame)
        """
        end_date = end_date or dt.date.today().isoformat()

        print(f"[DataManager] Requested: {len(tickers)} ETFs from {start_date} to {end_date}")

        # Check what data already exists
        existing_tickers = []
        existing_data = {}
        earliest_date = None
        latest_date = None

        for ticker in tickers:
            csv_path = self.data_dir / f"{ticker}.csv"
            if csv_path.exists():
                try:
                    df = pd.read_csv(csv_path, index_col=0, parse_dates=True)

                    # Get close price column (adjusted close)
                    if 'close' in df.columns:
                        price_series = df['close']
                    elif 'Close' in df.columns:
                        price_series = df['Close']
                    else:
                        price_series = df.iloc[:, 0]

                    existing_data[ticker] = price_series
                    existing_tickers.append(ticker)

                    # Track date range
                    first = df.index.min()
                    last = df.index.max()

                    if earliest_date is None or first < earliest_date:
                        earliest_date = first
                    if latest_date is None or last > latest_date:
                        latest_date = last

                except Exception as e:
                    warnings.warn(f"Failed to load {ticker}: {e}")

        # Determine what needs to be downloaded
        new_tickers = [t for t in tickers if t not in existing_tickers]

        needs_forward_update = False
        needs_backfill = False

        if existing_tickers:
            last_date_str = latest_date.strftime("%Y-%m-%d")
            first_date_str = earliest_date.strftime("%Y-%m-%d")

            # Check if we need to update: existing data doesn't cover requested end_date
            needs_forward_update = last_date_str < end_date
            needs_backfill = start_date < first_date_str

            print(f"[DataManager] Existing data: {len(existing_tickers)} ETFs, dates: {first_date_str} to {last_date_str}")
            print(f"[DataManager] New ETFs: {len(new_tickers)}")
            print(f"[DataManager] Needs forward update: {needs_forward_update}")
            print(f"[DataManager] Needs backfill: {needs_backfill}")
        else:
            print(f"[DataManager] No existing data found. Will download all.")

        # Case 1: Everything is up to date
        if not new_tickers and not needs_forward_update and not needs_backfill:
            print("[DataManager] Data is up to date. Using existing data.")
            prices = pd.DataFrame(existing_data).sort_index()

            # Filter to requested date range
            prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
            prices = prices.dropna()
            returns = np.log(prices / prices.shift(1)).dropna()

            print(f"Loaded ETF prices: {len(prices)} days, {len(prices.columns)} ETFs")
            print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

            return prices, returns

        # Case 2: Download needed data
        downloaded_data = {}

        # Download new tickers (full range)
        if new_tickers:
            print(f"[DataManager] Downloading {len(new_tickers)} new ETFs...")
            for ticker in new_tickers:
                try:
                    df = yf.download(
                        ticker,
                        start=start_date,
                        end=end_date,
                        progress=False,
                        auto_adjust=True
                    )

                    if not df.empty and 'Close' in df.columns:
                        # Ensure timezone-naive
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)

                        # Handle MultiIndex columns from yfinance
                        if isinstance(df.columns, pd.MultiIndex):
                            close_series = df['Close'].iloc[:, 0]
                        else:
                            close_series = df['Close']
                        
                        if isinstance(close_series, pd.Series) and len(close_series) > 0:
                            downloaded_data[ticker] = close_series

                            # Save to CSV
                            csv_path = self.data_dir / f"{ticker}.csv"
                            pd.DataFrame({'close': close_series}).to_csv(csv_path)
                            print(f"  [OK] {ticker}: {len(close_series)} days")
                        else:
                            print(f"  [WARNING] {ticker}: Invalid data format")
                    else:
                        print(f"  [WARNING] {ticker}: No data available")
                        print(f"             -> Check if {ticker} has historical data back to {start_date}")
                        print(f"             -> Consider alternative ETF if ticker is delisted or too recent")

                except Exception as e:
                    print(f"  [ERROR] {ticker}: {e}")
                    print(f"         -> Verify {ticker} exists and has data back to {start_date}")

        # Forward update: Download new dates for existing tickers
        if needs_forward_update and existing_tickers:
            update_start = (latest_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            
            # Check if the date range is too small (less than 7 calendar days)
            # yfinance has issues with very small ranges, especially over weekends
            # A week ensures at least a few trading days even with holidays
            date_diff = (pd.to_datetime(end_date) - pd.to_datetime(update_start)).days
            if date_diff < 7:
                print(f"[DataManager] Update range too small ({date_diff} days, need >=7). Skipping forward update.")
                # Use existing data
                for ticker in existing_tickers:
                    if ticker not in downloaded_data:
                        downloaded_data[ticker] = existing_data[ticker]
            else:
                print(f"[DataManager] Updating existing ETFs from {update_start} to {end_date}...")

                for ticker in existing_tickers:
                    try:
                        df = yf.download(
                            ticker,
                            start=update_start,
                            end=end_date,
                            progress=False,
                            auto_adjust=True
                        )

                        if not df.empty and 'Close' in df.columns:
                            # Ensure timezone-naive
                            if df.index.tz is not None:
                                df.index = df.index.tz_localize(None)

                            # Handle MultiIndex columns from yfinance
                            if isinstance(df.columns, pd.MultiIndex):
                                new_series = df['Close'].iloc[:, 0]
                            else:
                                new_series = df['Close']
                            
                            if isinstance(new_series, pd.Series) and len(new_series) > 0:
                                # Merge with existing data
                                existing_series = existing_data[ticker]
                                combined = pd.concat([existing_series, new_series]).sort_index()
                                combined = combined[~combined.index.duplicated(keep='last')]

                                downloaded_data[ticker] = combined

                                # Save updated CSV
                                csv_path = self.data_dir / f"{ticker}.csv"
                                pd.DataFrame({'close': combined}).to_csv(csv_path)
                                print(f"  [OK] {ticker}: +{len(new_series)} new days")
                            else:
                                # No valid new data, keep existing
                                downloaded_data[ticker] = existing_data[ticker]
                        else:
                            # No new data, keep existing
                            downloaded_data[ticker] = existing_data[ticker]

                    except Exception as e:
                        print(f"  [ERROR] {ticker}: {e}")
                        # Keep existing data on error
                        downloaded_data[ticker] = existing_data[ticker]

        # Backfill: Download earlier dates for existing tickers
        if needs_backfill and existing_tickers:
            backfill_end = (earliest_date - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            print(f"[DataManager] Backfilling from {start_date} to {backfill_end}...")

            for ticker in existing_tickers:
                try:
                    df = yf.download(
                        ticker,
                        start=start_date,
                        end=backfill_end,
                        progress=False,
                        auto_adjust=True
                    )

                    if not df.empty:
                        # Ensure timezone-naive
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)

                        # Merge with existing data
                        existing_series = downloaded_data.get(ticker, existing_data[ticker])
                        backfill_series = df['Close']
                        combined = pd.concat([backfill_series, existing_series]).sort_index()
                        combined = combined[~combined.index.duplicated(keep='first')]

                        downloaded_data[ticker] = combined

                        # Save updated CSV
                        csv_path = self.data_dir / f"{ticker}.csv"
                        pd.DataFrame({'close': combined}).to_csv(csv_path)
                        print(f"  [OK] {ticker}: +{len(backfill_series)} backfill days")
                    else:
                        # No backfill data, keep existing
                        if ticker not in downloaded_data:
                            downloaded_data[ticker] = existing_data[ticker]

                except Exception as e:
                    print(f"  [ERROR] {ticker}: {e}")
                    # Keep existing data on error
                    if ticker not in downloaded_data:
                        downloaded_data[ticker] = existing_data[ticker]

        # Combine all data (existing + downloaded)
        all_data = {**existing_data, **downloaded_data}

        if not all_data:
            raise ValueError("No data available after download attempt")

        prices = pd.DataFrame(all_data).sort_index()

        # Filter to requested date range
        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
        prices = prices.dropna()
        returns = np.log(prices / prices.shift(1)).dropna()

        print(f"\nLoaded ETF prices: {len(prices)} days, {len(prices.columns)} ETFs")
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

        return prices, returns

    def load_or_download_benchmark_prices(
        self,
        tickers: Optional[List[str]] = None,
        start_date: str = "2016-01-01",
        end_date: Optional[str] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load benchmark prices (SPY, TLT, GLD) for cross-asset features.
        
        Uses same yfinance download logic as ETF prices, stored in benchmark_dir.
        
        Parameters
        ----------
        tickers : List[str], optional
            Benchmark tickers. Default: BENCHMARK_TICKERS (SPY, TLT, GLD)
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str, optional
            End date. Default: today
            
        Returns
        -------
        Tuple[pd.DataFrame, pd.DataFrame]
            (prices, returns) DataFrames with benchmark data
        """
        tickers = tickers or BENCHMARK_TICKERS
        end_date = end_date or dt.date.today().isoformat()
        
        print(f"[Benchmarks] Loading {tickers} from {start_date} to {end_date} (parallel)")
        
        all_data = {}
        
        def download_single_benchmark(ticker: str) -> Tuple[str, Optional[pd.Series], str]:
            """Download a single benchmark ticker. Returns (ticker, series, status_msg)."""
            cache_path = self.benchmark_dir / f"{ticker}.csv"
            
            # Try to load cached data
            cached_data = None
            if cache_path.exists():
                try:
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    if 'close' in df.columns:
                        cached_data = df['close']
                    elif 'Close' in df.columns:
                        cached_data = df['Close']
                    else:
                        cached_data = df.iloc[:, 0]
                    
                    # Ensure timezone-naive
                    if cached_data.index.tz is not None:
                        cached_data.index = cached_data.index.tz_localize(None)
                    
                    last_date = cached_data.index.max().strftime("%Y-%m-%d")
                    
                    # Check if update needed
                    if last_date >= end_date:
                        return ticker, cached_data, f"[OK] {ticker}: {len(cached_data)} days (cached)"
                        
                except Exception as e:
                    cached_data = None
            
            # Download from yfinance
            try:
                df = yf.download(
                    ticker,
                    start=start_date,
                    end=end_date,
                    progress=False,
                    auto_adjust=True,
                )
                
                if not df.empty:
                    # Ensure timezone-naive
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # Handle multi-level columns from yfinance
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    price_series = df['Close']
                    
                    # Merge with cached if exists
                    if cached_data is not None:
                        combined = pd.concat([cached_data, price_series]).sort_index()
                        combined = combined[~combined.index.duplicated(keep='last')]
                        price_series = combined
                    
                    # Save to cache
                    pd.DataFrame({'close': price_series}).to_csv(cache_path)
                    return ticker, price_series, f"[OK] {ticker}: {len(price_series)} days"
                else:
                    return ticker, cached_data, f"[ERROR] {ticker}: No data returned"
                    
            except Exception as e:
                return ticker, cached_data, f"[ERROR] {ticker}: {e}"
        
        # Parallel download
        with ThreadPoolExecutor(max_workers=min(len(tickers), 5)) as executor:
            futures = {executor.submit(download_single_benchmark, t): t for t in tickers}
            for future in as_completed(futures):
                ticker, data, msg = future.result()
                print(f"  {msg}")
                if data is not None:
                    all_data[ticker] = data
        
        if not all_data:
            raise ValueError("No benchmark data available")
        
        prices = pd.DataFrame(all_data).sort_index()
        prices = prices[(prices.index >= start_date) & (prices.index <= end_date)]
        prices = prices.dropna()
        returns = np.log(prices / prices.shift(1)).dropna()
        
        print(f"[Benchmarks] Loaded: {len(prices)} days, {list(prices.columns)}")
        
        return prices, returns

    def load_or_download_macro_data(
        self,
        macro_tickers: Dict[str, str],
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.Series]:
        """
        Load macro data (VIX, yields) with incremental updates.

        Args:
            macro_tickers: Dict of {name: ticker_symbol}
            start_date: Optional start date filter (YYYY-MM-DD)
            end_date: Optional end date filter (YYYY-MM-DD)

        Returns:
            Dictionary of {name: Series} for macro data
        """
        end_date = end_date or dt.date.today().isoformat()
        start_date = start_date or "2010-01-01"

        print(f"[DataManager] Loading macro data...")

        macro_data = {}

        for name, ticker in macro_tickers.items():
            data_path = self.data_dir / f"MACRO_{name}.csv"

            # Try to load existing data
            existing_data = None
            if data_path.exists():
                try:
                    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
                    if 'close' in df.columns:
                        existing_data = df['close']
                    elif 'Close' in df.columns:
                        existing_data = df['Close']
                    else:
                        existing_data = df.iloc[:, 0]

                    # Ensure timezone-naive
                    if existing_data.index.tz is not None:
                        existing_data.index = existing_data.index.tz_localize(None)

                    last_date = existing_data.index.max().strftime("%Y-%m-%d")
                    first_date = existing_data.index.min().strftime("%Y-%m-%d")

                    # Check if update needed
                    today = dt.date.today().isoformat()
                    needs_update = last_date < end_date and last_date < today
                    needs_backfill = start_date < first_date

                    if not needs_update and not needs_backfill:
                        # Data is up to date
                        data_series = existing_data
                        if start_date:
                            start_dt = pd.to_datetime(start_date)
                            data_series = data_series[data_series.index >= start_dt]
                        if end_date:
                            end_dt = pd.to_datetime(end_date)
                            data_series = data_series[data_series.index <= end_dt]

                        # Convert yields from percentage to decimal if needed
                        if 'yield' in name or 'tbill' in name:
                            data_series = data_series / 100

                        macro_data[name] = data_series.dropna()
                        print(f"  [OK] {name}: {len(data_series)} observations (cached)")
                        continue

                    # Need to update
                    if needs_update:
                        update_start = (existing_data.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                        try:
                            if ticker.startswith('^'):
                                new_df = yf.download(ticker, start=update_start, end=end_date, progress=False, auto_adjust=True)
                            else:
                                new_df = yf.Ticker(ticker).history(start=update_start, end=end_date)

                            if not new_df.empty and 'Close' in new_df.columns:
                                if new_df.index.tz is not None:
                                    new_df.index = new_df.index.tz_localize(None)

                                # Handle MultiIndex columns
                                if isinstance(new_df.columns, pd.MultiIndex):
                                    new_data = new_df['Close'].iloc[:, 0]
                                else:
                                    new_data = new_df['Close']
                                
                                combined = pd.concat([existing_data, new_data]).sort_index()
                                combined = combined[~combined.index.duplicated(keep='last')]

                                # Save updated data
                                pd.DataFrame({'close': combined}).to_csv(data_path)
                                data_series = combined
                            else:
                                data_series = existing_data
                        except Exception as e:
                            print(f"  [WARNING] Failed to update {name}: {e}")
                            data_series = existing_data
                    else:
                        data_series = existing_data

                    # Backfill if needed
                    if needs_backfill:
                        backfill_end = (existing_data.index.min() - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

                        try:
                            if ticker.startswith('^'):
                                backfill_df = yf.download(ticker, start=start_date, end=backfill_end, progress=False, auto_adjust=True)
                            else:
                                backfill_df = yf.Ticker(ticker).history(start=start_date, end=backfill_end)

                            if not backfill_df.empty and 'Close' in backfill_df.columns:
                                if backfill_df.index.tz is not None:
                                    backfill_df.index = backfill_df.index.tz_localize(None)

                                # Handle MultiIndex columns
                                if isinstance(backfill_df.columns, pd.MultiIndex):
                                    backfill_data = backfill_df['Close'].iloc[:, 0]
                                else:
                                    backfill_data = backfill_df['Close']
                                
                                combined = pd.concat([backfill_data, data_series]).sort_index()
                                combined = combined[~combined.index.duplicated(keep='first')]

                                # Save updated data
                                pd.DataFrame({'close': combined}).to_csv(data_path)
                                data_series = combined
                        except Exception as e:
                            print(f"  [WARNING] Failed to backfill {name}: {e}")

                except Exception as e:
                    warnings.warn(f"Failed to process existing {name} data: {e}. Downloading fresh.")
                    existing_data = None

            # If no existing data or error, download fresh
            if existing_data is None:
                try:
                    print(f"[DataManager] Downloading {name} from scratch...")

                    if ticker.startswith('^'):
                        df = yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                    else:
                        df = yf.Ticker(ticker).history(start=start_date, end=end_date)

                    if not df.empty and 'Close' in df.columns:
                        if df.index.tz is not None:
                            df.index = df.index.tz_localize(None)

                        # Handle MultiIndex columns
                        if isinstance(df.columns, pd.MultiIndex):
                            data_series = df['Close'].iloc[:, 0]
                        else:
                            data_series = df['Close']

                        # Save for future use
                        pd.DataFrame({'close': data_series}).to_csv(data_path)
                        print(f"  [OK] Saved {name} data to {data_path}")
                    else:
                        print(f"  [WARNING] {name}: No data downloaded from {ticker}")
                        print(f"             -> Verify macro ticker {ticker} exists and has data back to {start_date}")
                        data_series = pd.Series(dtype=float)

                except Exception as e:
                    print(f"  [ERROR] Could not download {name} ({ticker}): {e}")
                    print(f"         -> Check if macro ticker {ticker} is correct and has historical data back to {start_date}")
                    data_series = pd.Series(dtype=float)

            # Filter to requested range and convert yields
            if start_date and len(data_series) > 0:
                start_dt = pd.to_datetime(start_date)
                data_series = data_series[data_series.index >= start_dt]
            if end_date and len(data_series) > 0:
                end_dt = pd.to_datetime(end_date)
                data_series = data_series[data_series.index <= end_dt]

            # Convert yields from percentage to decimal if needed
            if 'yield' in name or 'tbill' in name:
                data_series = data_series / 100

            macro_data[name] = data_series.dropna()
            print(f"  Loaded {name}: {len(data_series)} observations")

        return macro_data

    def load_macro_reference_returns(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> Dict[str, pd.Series]:
        """
        Load returns for macro reference tickers used in risk feature computations.
        
        Loads from PERMANENT raw OHLCV storage (ohlcv/*.parquet):
        - VT: from ohlcv/VT.parquet (already in ETF universe)
        - VIX: from ohlcv/^VIX.parquet (downloaded to permanent storage)
        - MOVE: from ohlcv/^MOVE.parquet (downloaded to permanent storage)
        
        NOTE: BNDW (global bonds) is NOT loaded here. Instead, synthetic BNDW 
        is created from BND + BNDX ETF data in add_risk_factor_features_to_panel().
        BND and BNDX are in the ETF universe and stored in ohlcv/*.parquet.
        Synthetic BNDW = (BND + BNDX returns)/2, has 98.45% correlation with actual BNDW.
        
        These are used for:
        - Rolling betas (beta_VT, beta_VIX, beta_MOVE)  
        - Idiosyncratic volatility
        - Downside beta calculations
        
        Parameters
        ----------
        start_date : str, optional
            Start date (YYYY-MM-DD). Used only for downloading missing data.
        end_date : str, optional
            End date (YYYY-MM-DD). Used only for downloading missing data.
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary of {name: returns_series} for each reference ticker
        """
        end_date = end_date or dt.date.today().isoformat()
        start_date = start_date or "2010-01-01"
        
        print(f"[MacroRef] Loading macro reference tickers from permanent ohlcv/ storage...")
        
        returns_dict = {}
        
        # Mapping of internal name -> (parquet filename, yfinance ticker for download)
        ticker_map = {
            'vt': ('VT.parquet', 'VT'),
            'vix': ('^VIX.parquet', '^VIX'),
            'move': ('^MOVE.parquet', '^MOVE'),
        }
        
        for name, (filename, yf_ticker) in ticker_map.items():
            cache_path = self.ohlcv_dir / filename
            
            # Try to load from permanent storage
            if cache_path.exists():
                try:
                    df = pd.read_parquet(cache_path)
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    close = df['Close'] if 'Close' in df.columns else df.iloc[:, 0]
                    returns = close.pct_change().dropna()
                    returns_dict[name] = returns
                    print(f"  [OK] {name}: {len(returns)} returns from {returns.index.min().date()} to {returns.index.max().date()} (from ohlcv/)")
                    continue
                except Exception as e:
                    print(f"  [WARN] {name}: Failed to load from {filename}: {e}")
            
            # Download if not in permanent storage
            print(f"  [INFO] {name}: Not in permanent storage, downloading {yf_ticker}...")
            try:
                df = yf.download(yf_ticker, start=start_date, end=end_date, progress=False, auto_adjust=True)
                
                if not df.empty:
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # Handle MultiIndex columns from yfinance
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    
                    # Save to permanent storage (ohlcv/ folder)
                    df.to_parquet(cache_path)
                    print(f"  [OK] {name}: Downloaded and saved to ohlcv/{filename}")
                    
                    close = df['Close']
                    returns = close.pct_change().dropna()
                    returns_dict[name] = returns
                    print(f"       {len(returns)} returns from {returns.index.min().date()} to {returns.index.max().date()}")
                else:
                    print(f"  [ERROR] {name}: No data returned from yfinance")
                    
            except Exception as e:
                print(f"  [ERROR] {name}: Download failed: {e}")
        
        print(f"[MacroRef] Loaded returns for: {list(returns_dict.keys())}")
        
        return returns_dict

    def _get_fred_client(self):
        """
        Lazy initialization of FRED API client.
        Uses instance fred_api_key (set from env, parameter, or default).
        """
        if self._fred is None:
            api_key = self.fred_api_key
            if api_key:
                try:
                    from fredapi import Fred
                    self._fred = Fred(api_key=api_key)
                    print("[FRED] API initialized successfully")
                except ImportError:
                    warnings.warn("fredapi not installed. Run: pip install fredapi")
                except Exception as e:
                    warnings.warn(f"Failed to initialize FRED API: {e}")
            else:
                warnings.warn(
                    "FRED_API_KEY not set. Get free key from: "
                    "https://fred.stlouisfed.org/docs/api/api_key.html"
                )
        return self._fred

    def load_or_download_fred_data(
        self,
        series_dict: Optional[Dict[str, str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        force_refresh: bool = False,
    ) -> Dict[str, pd.Series]:
        """
        Load FRED macro/sentiment data with incremental updates.
        
        Parameters
        ----------
        series_dict : Dict[str, str], optional
            Dict of {FRED_CODE: friendly_name}. If None, uses FRED_SERIES default.
        start_date : str, optional
            Start date (YYYY-MM-DD). Default "2010-01-01"
        end_date : str, optional
            End date (YYYY-MM-DD). Default today.
        force_refresh : bool
            If True, re-download even if cached.
            
        Returns
        -------
        Dict[str, pd.Series]
            Dictionary of {friendly_name: Series} for FRED data
        """
        end_date = end_date or dt.date.today().isoformat()
        start_date = start_date or "2010-01-01"
        
        if series_dict is None:
            series_dict = FRED_SERIES
        
        print(f"[FRED] Loading {len(series_dict)} series from {start_date} to {end_date} (parallel)")
        
        fred_data = {}
        to_download = []  # List of (fred_code, friendly_name, cached_data) needing download
        
        # First pass: check cache, identify what needs downloading
        for fred_code, friendly_name in series_dict.items():
            cache_path = self.fred_dir / f"FRED_{friendly_name}.csv"
            
            # Try to load cached data first
            cached_data = None
            if cache_path.exists() and not force_refresh:
                try:
                    df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                    if 'value' in df.columns:
                        cached_data = df['value']
                    else:
                        cached_data = df.iloc[:, 0]
                    
                    # Ensure timezone-naive
                    if cached_data.index.tz is not None:
                        cached_data.index = cached_data.index.tz_localize(None)
                    
                    last_date = cached_data.index.max().strftime("%Y-%m-%d")
                    
                    # Check if update needed
                    today = dt.date.today().isoformat()
                    needs_update = last_date < end_date and last_date < today
                    
                    if not needs_update:
                        # Use cached data
                        data_series = cached_data
                        if start_date:
                            start_dt = pd.to_datetime(start_date)
                            data_series = data_series[data_series.index >= start_dt]
                        if end_date:
                            end_dt = pd.to_datetime(end_date)
                            data_series = data_series[data_series.index <= end_dt]
                        
                        fred_data[friendly_name] = data_series.dropna()
                        print(f"  [OK] {friendly_name}: {len(data_series)} observations (cached)")
                        continue
                        
                except Exception as e:
                    cached_data = None
            
            # Mark for download
            to_download.append((fred_code, friendly_name, cached_data))
        
        # Initialize FRED client once if needed
        fred = None
        if to_download:
            fred = self._get_fred_client()
            if fred is None:
                warnings.warn("FRED API not available, cannot download series")
                for fred_code, friendly_name, cached_data in to_download:
                    fred_data[friendly_name] = pd.Series(dtype=float)
                return fred_data
        
        def download_single_fred(args) -> Tuple[str, pd.Series, str]:
            """Download a single FRED series. Returns (friendly_name, series, status_msg)."""
            fred_code, friendly_name, cached_data = args
            cache_path = self.fred_dir / f"FRED_{friendly_name}.csv"
            
            try:
                # Download the series
                data = fred.get_series(fred_code, observation_start=start_date, observation_end=end_date)
                
                if data is not None and len(data) > 0:
                    # Ensure timezone-naive
                    if data.index.tz is not None:
                        data.index = data.index.tz_localize(None)
                    
                    # Merge with cached if exists
                    if cached_data is not None and len(cached_data) > 0:
                        combined = pd.concat([cached_data, data]).sort_index()
                        combined = combined[~combined.index.duplicated(keep='last')]
                    else:
                        combined = data
                    
                    # Save to cache
                    pd.DataFrame({'value': combined}).to_csv(cache_path)
                    
                    # Filter to requested range
                    data_series = combined
                    if start_date:
                        start_dt = pd.to_datetime(start_date)
                        data_series = data_series[data_series.index >= start_dt]
                    if end_date:
                        end_dt = pd.to_datetime(end_date)
                        data_series = data_series[data_series.index <= end_dt]
                    
                    return friendly_name, data_series.dropna(), f"[OK] {friendly_name}: {len(data_series)} observations"
                else:
                    return friendly_name, pd.Series(dtype=float), f"[WARNING] {friendly_name}: No data from FRED"
                    
            except Exception as e:
                return friendly_name, pd.Series(dtype=float), f"[ERROR] {friendly_name}: {e}"
        
        # Parallel download (max 4 workers to avoid FRED rate limits)
        if to_download:
            with ThreadPoolExecutor(max_workers=min(len(to_download), 4)) as executor:
                futures = {executor.submit(download_single_fred, args): args[1] for args in to_download}
                for future in as_completed(futures):
                    friendly_name, data_series, msg = future.result()
                    print(f"  {msg}")
                    fred_data[friendly_name] = data_series
        
        return fred_data

    def download_ohlcv_data(
        self,
        tickers: List[str],
        start_date: str,
        end_date: str,
        batch_sleep: Optional[float] = None,
    ) -> Dict[str, pd.DataFrame]:
        """
        Download OHLCV data for crosssecmom2 strategy with intelligent caching.
        
        This is the main interface for feature_engineering.py.
        Returns dict of {ticker: DataFrame} with OHLCV columns.
        
        Implements ROBUST incremental caching:
        - Backfill: If requested_start < cached_start → download older dates, prepend
        - Forward-fill: If requested_end > cached_end → download newer dates, append
        - New tickers: Download full range
        - NEVER dedupes rows (dates are unique observations)
        - Dedupes columns only (prevents ticker duplication)
        
        Args:
            tickers: List of ETF ticker symbols
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            batch_sleep: Sleep time after every 20 downloads. If None, uses default from config.
            
        Returns:
            Dict of {ticker: DataFrame} with columns ['Open', 'High', 'Low', 'Close', 'Volume']
            DatetimeIndex is the index
        """
        if batch_sleep is None:
            from config import get_default_config
            batch_sleep = get_default_config().compute.batch_sleep
        
        print(f"[DataManager] Fetching OHLCV data for {len(tickers)} ETFs from {start_date} to {end_date}...")
        
        requested_start = pd.to_datetime(start_date)
        requested_end = pd.to_datetime(end_date)
        
        data_dict = {}
        failed = []
        stats = {'cached': 0, 'backfilled': 0, 'frontfilled': 0, 'fresh': 0}
        
        for i, ticker in enumerate(tickers):
            try:
                cache_path = self.ohlcv_dir / f"{ticker}.parquet"
                result_df = self._load_or_update_ticker_cache(
                    ticker=ticker,
                    cache_path=cache_path,
                    requested_start=requested_start,
                    requested_end=requested_end,
                    stats=stats
                )
                
                if result_df is not None and len(result_df) > 0:
                    data_dict[ticker] = result_df
                else:
                    failed.append(ticker)
                
                # Progress update
                if (i + 1) % 20 == 0:
                    print(f"  [progress] {i+1}/{len(tickers)} (cached:{stats['cached']}, backfill:{stats['backfilled']}, frontfill:{stats['frontfilled']}, fresh:{stats['fresh']})")
                    time.sleep(batch_sleep)
                    
            except Exception as e:
                print(f"  [error] {ticker}: {e}")
                failed.append(ticker)
        
        print(f"\n[DataManager] Successfully loaded: {len(data_dict)}/{len(tickers)}")
        print(f"[DataManager] Stats: cached={stats['cached']}, backfilled={stats['backfilled']}, frontfilled={stats['frontfilled']}, fresh={stats['fresh']}")
        if failed:
            print(f"[DataManager] Failed tickers ({len(failed)}): {failed[:20]}{'...' if len(failed) > 20 else ''}")
        
        return data_dict
    
    def _load_or_update_ticker_cache(
        self,
        ticker: str,
        cache_path: Path,
        requested_start: pd.Timestamp,
        requested_end: pd.Timestamp,
        stats: Dict[str, int],
    ) -> Optional[pd.DataFrame]:
        """
        Load ticker data from cache, updating cache if needed with backfill/frontfill.
        
        IMPORTANT: Never dedupes rows (dates). Only dedupes columns.
        
        Returns filtered DataFrame for requested range, or None if failed.
        """
        OHLCV_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
        
        existing_df = None
        cached_start = None
        cached_end = None
        
        # Step 1: Load existing cache if present
        if cache_path.exists():
            try:
                existing_df = pd.read_parquet(cache_path)
                
                # Ensure timezone-naive
                if existing_df.index.tz is not None:
                    existing_df.index = existing_df.index.tz_localize(None)
                
                # Ensure we have required columns
                missing_cols = [c for c in OHLCV_COLS if c not in existing_df.columns]
                if missing_cols:
                    print(f"  [warning] {ticker}: Cache missing columns {missing_cols}, will re-download")
                    existing_df = None
                else:
                    cached_start = existing_df.index.min()
                    cached_end = existing_df.index.max()
                    
            except Exception as e:
                print(f"  [warning] {ticker}: Failed to load cache ({e}), will download fresh")
                existing_df = None
        
        # Step 2: Determine what we need to do
        needs_backfill = False
        needs_frontfill = False
        
        if existing_df is not None:
            needs_backfill = requested_start < cached_start
            needs_frontfill = requested_end > cached_end
            
            # If cache fully covers requested range, just filter and return
            if not needs_backfill and not needs_frontfill:
                stats['cached'] += 1
                mask = (existing_df.index >= requested_start) & (existing_df.index <= requested_end)
                df = existing_df.loc[mask]
                if len(df) > 0:
                    return df[OHLCV_COLS].astype('float32')
                else:
                    # Cache exists but has no data in requested range - download fresh
                    existing_df = None
        
        # Step 3: Download fresh if no cache
        if existing_df is None:
            df = self._download_ticker_ohlcv(ticker, requested_start, requested_end)
            if df is None or len(df) == 0:
                return None
            
            # Save to cache
            df[OHLCV_COLS].to_parquet(cache_path)
            stats['fresh'] += 1
            
            # Return filtered to requested range
            mask = (df.index >= requested_start) & (df.index <= requested_end)
            return df.loc[mask, OHLCV_COLS].astype('float32')
        
        # Step 4: Incremental update needed - build combined dataframe
        combined_df = existing_df.copy()
        
        if needs_backfill:
            # Download older dates: [requested_start, cached_start - 1 day]
            backfill_end = cached_start - pd.Timedelta(days=1)
            print(f"  [backfill] {ticker}: downloading {requested_start.date()} to {backfill_end.date()}")
            
            backfill_df = self._download_ticker_ohlcv(ticker, requested_start, backfill_end)
            if backfill_df is not None and len(backfill_df) > 0:
                # Prepend backfill data (concat, then sort by date)
                combined_df = pd.concat([backfill_df, combined_df], axis=0)
                stats['backfilled'] += 1
            else:
                print(f"  [warning] {ticker}: Backfill returned no data for {requested_start.date()} to {backfill_end.date()}")
        
        if needs_frontfill:
            # Download newer dates: [cached_end + 1 day, requested_end]
            frontfill_start = cached_end + pd.Timedelta(days=1)
            
            # Don't try to frontfill future dates - they don't exist yet
            today = pd.Timestamp.now().normalize()
            if frontfill_start > today:
                # Cache is already up to date (or in the future)
                pass
            else:
                capped_end = min(requested_end, today)
                print(f"  [frontfill] {ticker}: downloading {frontfill_start.date()} to {capped_end.date()}")
                
                frontfill_df = self._download_ticker_ohlcv(ticker, frontfill_start, capped_end)
                if frontfill_df is not None and len(frontfill_df) > 0:
                    # Append frontfill data
                    combined_df = pd.concat([combined_df, frontfill_df], axis=0)
                    stats['frontfilled'] += 1
                else:
                    # Only warn if we expected data (not future dates)
                    if capped_end >= frontfill_start:
                        print(f"  [warning] {ticker}: Frontfill returned no data for {frontfill_start.date()} to {capped_end.date()}")
        
        # Step 5: Clean up combined dataframe
        # Sort by date index
        combined_df = combined_df.sort_index()
        
        # Remove duplicate COLUMNS only (not rows!)
        # This handles case where same ticker appears twice
        combined_df = combined_df.loc[:, ~combined_df.columns.duplicated(keep='first')]
        
        # Ensure we have OHLCV columns
        if not all(c in combined_df.columns for c in OHLCV_COLS):
            print(f"  [error] {ticker}: Missing required columns after merge")
            return None
        
        # Save updated cache (full combined data)
        combined_df[OHLCV_COLS].to_parquet(cache_path)
        
        # Return filtered to requested range
        mask = (combined_df.index >= requested_start) & (combined_df.index <= requested_end)
        result = combined_df.loc[mask, OHLCV_COLS]
        
        if len(result) == 0:
            return None
        
        return result.astype('float32')
    
    def _download_ticker_ohlcv(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Optional[pd.DataFrame]:
        """
        Download OHLCV data for a single ticker from yfinance.
        
        Returns DataFrame with OHLCV columns, or None on failure.
        """
        try:
            df = yf.download(
                ticker,
                start=start.strftime('%Y-%m-%d'),
                end=(end + pd.Timedelta(days=1)).strftime('%Y-%m-%d'),  # yfinance end is exclusive
                progress=False,
                auto_adjust=False
            )
            
            if df.empty:
                return None
            
            # Ensure timezone-naive
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            
            # Handle MultiIndex columns (happens with single ticker sometimes)
            if isinstance(df.columns, pd.MultiIndex):
                df = df.droplevel(1, axis=1)
            
            # Keep only OHLCV
            required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            if not all(c in df.columns for c in required_cols):
                return None
            
            return df[required_cols]
            
        except Exception as e:
            print(f"  [download error] {ticker}: {e}")
            return None

    # ========================================================================
    # FULL HISTORY DOWNLOAD (period='max')
    # ========================================================================
    
    def download_full_history(
        self,
        tickers: List[str],
        batch_sleep: float = 1.0,
        overwrite_existing: bool = False,
    ) -> Dict[str, Dict]:
        """
        Download full available history for all tickers using period='max'.
        
        This function downloads the maximum available history from Yahoo Finance,
        which typically goes back to 1999 or the IPO date, whichever is later.
        
        Unlike download_ohlcv_data(), this REPLACES the cached data rather than
        doing incremental updates. Use this to ensure you have the longest
        possible history for each ticker.
        
        Args:
            tickers: List of ETF ticker symbols
            batch_sleep: Sleep time after every 20 downloads (default 1.0s)
            overwrite_existing: If True, re-downloads even if cache exists.
                              If False, only downloads tickers without cache.
            
        Returns:
            Dict of {ticker: {'status': str, 'start': str, 'end': str, 'rows': int}}
        """
        print(f"[FullHistory] Downloading maximum history for {len(tickers)} tickers...")
        print(f"[FullHistory] Overwrite existing: {overwrite_existing}")
        
        results = {}
        stats = {'downloaded': 0, 'skipped': 0, 'failed': 0}
        
        for i, ticker in enumerate(tickers):
            cache_path = self.ohlcv_dir / f"{ticker}.parquet"
            
            # Skip if cache exists and not overwriting
            if cache_path.exists() and not overwrite_existing:
                try:
                    existing_df = pd.read_parquet(cache_path)
                    results[ticker] = {
                        'status': 'skipped',
                        'start': str(existing_df.index.min().date()),
                        'end': str(existing_df.index.max().date()),
                        'rows': len(existing_df),
                    }
                    stats['skipped'] += 1
                    continue
                except Exception:
                    pass  # Will re-download if can't read
            
            # Download full history using period='max'
            try:
                df = yf.download(
                    ticker,
                    period='max',  # Download all available history
                    progress=False,
                    auto_adjust=False
                )
                
                if df.empty:
                    results[ticker] = {'status': 'failed', 'error': 'No data returned'}
                    stats['failed'] += 1
                    print(f"  [FAIL] {ticker}: No data returned")
                    continue
                
                # Ensure timezone-naive
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                
                # Handle MultiIndex columns
                if isinstance(df.columns, pd.MultiIndex):
                    df = df.droplevel(1, axis=1)
                
                # Validate we have required columns
                required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                if not all(c in df.columns for c in required_cols):
                    results[ticker] = {'status': 'failed', 'error': 'Missing OHLCV columns'}
                    stats['failed'] += 1
                    print(f"  [FAIL] {ticker}: Missing required columns")
                    continue
                
                # Save to cache
                df[required_cols].to_parquet(cache_path)
                
                results[ticker] = {
                    'status': 'downloaded',
                    'start': str(df.index.min().date()),
                    'end': str(df.index.max().date()),
                    'rows': len(df),
                }
                stats['downloaded'] += 1
                print(f"  [OK] {ticker}: {df.index.min().date()} to {df.index.max().date()} ({len(df)} rows)")
                
            except Exception as e:
                results[ticker] = {'status': 'failed', 'error': str(e)}
                stats['failed'] += 1
                print(f"  [FAIL] {ticker}: {e}")
            
            # Progress and rate limiting
            if (i + 1) % 20 == 0:
                print(f"  [progress] {i+1}/{len(tickers)} (downloaded:{stats['downloaded']}, skipped:{stats['skipped']}, failed:{stats['failed']})")
                time.sleep(batch_sleep)
        
        print(f"\n[FullHistory] Complete: downloaded={stats['downloaded']}, skipped={stats['skipped']}, failed={stats['failed']}")
        return results
    
    def get_download_status(self) -> pd.DataFrame:
        """
        Get status of all cached OHLCV data files.
        
        Returns:
            DataFrame with columns: ticker, start_date, end_date, rows, file_size_kb, last_modified
        """
        records = []
        
        for cache_file in self.ohlcv_dir.glob("*.parquet"):
            ticker = cache_file.stem
            try:
                df = pd.read_parquet(cache_file)
                records.append({
                    'ticker': ticker,
                    'start_date': df.index.min().date(),
                    'end_date': df.index.max().date(),
                    'rows': len(df),
                    'file_size_kb': cache_file.stat().st_size / 1024,
                    'last_modified': dt.datetime.fromtimestamp(cache_file.stat().st_mtime),
                })
            except Exception as e:
                records.append({
                    'ticker': ticker,
                    'start_date': None,
                    'end_date': None,
                    'rows': 0,
                    'file_size_kb': cache_file.stat().st_size / 1024 if cache_file.exists() else 0,
                    'last_modified': None,
                    'error': str(e),
                })
        
        if not records:
            return pd.DataFrame(columns=['ticker', 'start_date', 'end_date', 'rows', 'file_size_kb', 'last_modified'])
        
        return pd.DataFrame(records).sort_values('ticker').reset_index(drop=True)


# ============================================================================
# BACKGROUND DATA REFRESH DAEMON
# ============================================================================

class DataRefreshDaemon:
    """
    Background daemon that automatically refreshes OHLCV data daily.
    
    Schedules data updates at a specified time (default 9 PM) to ensure
    fresh data is available for the next trading day's analysis.
    
    Requires: pip install schedule
    
    Usage:
        from data_manager import DataRefreshDaemon
        
        # Start daemon (runs in background thread)
        daemon = DataRefreshDaemon(
            tickers=['SPY', 'QQQ', ...],
            refresh_time='21:00'  # 9 PM
        )
        daemon.start()
        
        # ... your main program continues ...
        
        # Stop when done
        daemon.stop()
    """
    
    def __init__(
        self,
        tickers: List[str],
        data_dir: Optional[str] = None,
        refresh_time: str = '21:00',
        on_refresh_complete: Optional[callable] = None,
    ):
        """
        Initialize the data refresh daemon.
        
        Args:
            tickers: List of ETF ticker symbols to refresh
            data_dir: Data directory. If None, uses default from config.
            refresh_time: Time to run daily refresh (HH:MM format, 24-hour)
            on_refresh_complete: Optional callback function called after refresh
        """
        self.tickers = tickers
        self.refresh_time = refresh_time
        self.on_refresh_complete = on_refresh_complete
        self._manager = CrossSecMomDataManager(data_dir=data_dir)
        self._running = False
        self._thread = None
        self._last_refresh = None
        self._last_status = None
        
    def _refresh_data(self):
        """Execute the data refresh."""
        print(f"\n[DataRefreshDaemon] Starting scheduled refresh at {dt.datetime.now()}")
        
        try:
            # Get today's date for end_date
            today = dt.date.today().isoformat()
            
            # Download/update data for all tickers (incremental update)
            # Use a date far back to ensure we get any backfill needed
            start_date = '2000-01-01'
            
            result = self._manager.download_ohlcv_data(
                tickers=self.tickers,
                start_date=start_date,
                end_date=today,
                batch_sleep=1.0
            )
            
            self._last_refresh = dt.datetime.now()
            self._last_status = {
                'success': True,
                'tickers_updated': len(result),
                'timestamp': self._last_refresh,
            }
            
            print(f"[DataRefreshDaemon] Refresh complete: {len(result)} tickers updated")
            
            # Call callback if provided
            if self.on_refresh_complete:
                self.on_refresh_complete(self._last_status)
                
        except Exception as e:
            self._last_status = {
                'success': False,
                'error': str(e),
                'timestamp': dt.datetime.now(),
            }
            print(f"[DataRefreshDaemon] Refresh failed: {e}")
    
    def _run_scheduler(self):
        """Run the scheduler loop in background thread."""
        if not SCHEDULE_AVAILABLE:
            raise ImportError("The 'schedule' package is required for DataRefreshDaemon. Install with: pip install schedule")
        
        # Schedule the daily refresh
        schedule.every().day.at(self.refresh_time).do(self._refresh_data)
        
        print(f"[DataRefreshDaemon] Scheduled daily refresh at {self.refresh_time}")
        
        while self._running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def start(self):
        """Start the background daemon."""
        if not SCHEDULE_AVAILABLE:
            raise ImportError("The 'schedule' package is required for DataRefreshDaemon. Install with: pip install schedule")
        
        if self._running:
            print("[DataRefreshDaemon] Already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self._thread.start()
        print(f"[DataRefreshDaemon] Started (refresh at {self.refresh_time} daily)")
    
    def stop(self):
        """Stop the background daemon."""
        if not self._running:
            print("[DataRefreshDaemon] Not running")
            return
        
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if SCHEDULE_AVAILABLE:
            schedule.clear()
        print("[DataRefreshDaemon] Stopped")
    
    def refresh_now(self):
        """Trigger an immediate refresh (useful for testing)."""
        print("[DataRefreshDaemon] Triggering immediate refresh...")
        self._refresh_data()
    
    @property
    def is_running(self) -> bool:
        """Check if daemon is running."""
        return self._running
    
    @property
    def last_refresh(self) -> Optional[dt.datetime]:
        """Get timestamp of last refresh."""
        return self._last_refresh
    
    @property
    def last_status(self) -> Optional[Dict]:
        """Get status of last refresh."""
        return self._last_status


# ============================================================================
# CONVENIENCE FUNCTION FOR FEATURE ENGINEERING
# ============================================================================

def download_etf_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    batch_sleep: Optional[float] = None,
    data_dir: Optional[str] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function for backward compatibility with feature_engineering.py.
    
    Args:
        tickers: List of ETF ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        batch_sleep: Sleep time after every 20 downloads. If None, uses default from config.
        data_dir: Cache directory. If None, uses default from config.
        
    Returns:
        Dict of {ticker: DataFrame} with OHLCV data
    """
    manager = CrossSecMomDataManager(data_dir=data_dir)
    return manager.download_ohlcv_data(tickers, start_date, end_date, batch_sleep)


# ============================================================================
# DATA CLEANING AND FILTERING FUNCTIONS
# ============================================================================

def filter_tickers_for_research_window(
    data_dict: Dict[str, pd.DataFrame],
    formation_start_date: str,
    warmup_days: int = 0,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Filter tickers to keep only those with data available for the research window.
    
    ETFs that were listed after the formation_start_date (minus warmup) are dropped
    because they won't have enough history for feature calculation.
    
    IMPORTANT: Does NOT dedupe rows - dates are unique observations.
    
    Args:
        data_dict: Dict of {ticker: DataFrame} from download_etf_data
        formation_start_date: Formation window start date (YYYY-MM-DD)
        warmup_days: Additional trading days needed before formation start for feature lags
        verbose: Print filtering details
        
    Returns:
        Filtered dict with only tickers that have sufficient history
    """
    formation_start = pd.to_datetime(formation_start_date)
    
    # Convert warmup trading days to calendar days (1.5x multiplier)
    warmup_calendar_days = int(warmup_days * 1.5)
    required_start = formation_start - pd.Timedelta(days=warmup_calendar_days)
    
    # Add tolerance for weekends/holidays (5 days)
    required_start_with_tolerance = required_start + pd.Timedelta(days=5)
    
    filtered = {}
    dropped = []
    
    for ticker, df in data_dict.items():
        if df is None or len(df) == 0:
            dropped.append((ticker, "no data"))
            continue
        
        ticker_start = df.index.min()
        
        # Check if ticker has data early enough
        if ticker_start <= required_start_with_tolerance:
            filtered[ticker] = df
        else:
            gap_days = (ticker_start - required_start).days
            dropped.append((ticker, f"starts {gap_days}d late ({ticker_start.date()})"))
    
    if verbose:
        print(f"[filter] Formation start: {formation_start.date()}")
        print(f"[filter] Warmup days: {warmup_days} trading days (~{warmup_calendar_days} calendar days)")
        print(f"[filter] Required data start: ~{required_start.date()} (+5d tolerance)")
        print(f"[filter] Keeping {len(filtered)}/{len(data_dict)} tickers")
        
        if dropped and len(dropped) <= 10:
            for ticker, reason in dropped:
                print(f"  [dropped] {ticker}: {reason}")
        elif dropped:
            print(f"  [dropped] {len(dropped)} tickers (showing first 5):")
            for ticker, reason in dropped[:5]:
                print(f"    - {ticker}: {reason}")
    
    return filtered


def clean_raw_ohlcv_data(
    data_dict: Dict[str, pd.DataFrame],
    ffill_limit: int = 5,
    verbose: bool = True,
) -> Dict[str, pd.DataFrame]:
    """
    Clean raw OHLCV data by forward-filling gaps and handling missing values.
    
    This function:
    1. Forward-fills missing values (up to ffill_limit days) to handle trading halts
    2. Does NOT backward-fill (would cause look-ahead bias)
    3. Does NOT drop rows or dedupe dates (dates are unique observations)
    4. Removes tickers that have too many NaN values
    
    IMPORTANT: Never dedupes rows - each date is a unique observation.
    
    Args:
        data_dict: Dict of {ticker: DataFrame} from download_etf_data
        ffill_limit: Max consecutive days to forward-fill (default 5)
        verbose: Print cleaning details
        
    Returns:
        Cleaned dict with NaN gaps filled (up to limit)
    """
    cleaned = {}
    stats = {'cleaned': 0, 'too_sparse': 0, 'empty': 0}
    
    for ticker, df in data_dict.items():
        if df is None or len(df) == 0:
            stats['empty'] += 1
            continue
        
        # Copy to avoid modifying original
        df_clean = df.copy()
        
        # Count NaN before cleaning
        nan_before = df_clean['Close'].isna().sum() if 'Close' in df_clean.columns else 0
        
        # Forward-fill OHLCV columns (up to limit)
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        cols_to_fill = [c for c in ohlcv_cols if c in df_clean.columns]
        
        if cols_to_fill:
            df_clean[cols_to_fill] = df_clean[cols_to_fill].ffill(limit=ffill_limit)
        
        # Count NaN after cleaning
        nan_after = df_clean['Close'].isna().sum() if 'Close' in df_clean.columns else 0
        
        # Check if too sparse (more than 10% NaN in Close after cleaning)
        total_rows = len(df_clean)
        if total_rows > 0 and nan_after / total_rows > 0.10:
            stats['too_sparse'] += 1
            if verbose:
                print(f"  [too_sparse] {ticker}: {nan_after}/{total_rows} ({100*nan_after/total_rows:.1f}%) NaN after cleaning")
            continue
        
        cleaned[ticker] = df_clean
        stats['cleaned'] += 1
    
    if verbose:
        print(f"[clean] Cleaned: {stats['cleaned']}, Too sparse: {stats['too_sparse']}, Empty: {stats['empty']}")
    
    return cleaned


# ============================================================================
# DATA REQUIREMENTS CALCULATION
# ============================================================================

@dataclass
class DataRequirements:
    """Calculated data requirements based on config parameters."""
    # Config inputs
    config_start_date: pd.Timestamp
    config_end_date: pd.Timestamp
    formation_trading_days: int
    training_trading_days: int
    feature_max_lag_days: int
    holding_period_days: int
    
    # Calculated outputs
    earliest_data_needed: pd.Timestamp
    latest_data_needed: pd.Timestamp
    total_lookback_trading_days: int
    
    def __str__(self) -> str:
        return (
            f"DataRequirements(\n"
            f"  config_period: {self.config_start_date.date()} to {self.config_end_date.date()}\n"
            f"  formation: {self.formation_trading_days} trading days\n"
            f"  training: {self.training_trading_days} trading days\n"
            f"  feature_lag: {self.feature_max_lag_days} trading days\n"
            f"  holding: {self.holding_period_days} trading days\n"
            f"  total_lookback: {self.total_lookback_trading_days} trading days\n"
            f"  data_needed: {self.earliest_data_needed.date()} to {self.latest_data_needed.date()}\n"
            f")"
        )


def calculate_data_requirements(config) -> DataRequirements:
    """
    Calculate data requirements from config.
    
    The earliest date needed is:
        config.start_date - formation_window - training_window - feature_max_lag
    
    This ensures we have enough history for:
    1. Formation period (e.g., 5 years = 1260 trading days)
    2. Training period (e.g., 1 year = 252 trading days)  
    3. Feature computation lag (e.g., 252 days for 1-year momentum)
    
    Args:
        config: CrossSecMomConfig object
        
    Returns:
        DataRequirements object with calculated dates
    """
    # Extract config values
    config_start = pd.to_datetime(config.time.start_date)
    config_end = pd.to_datetime(config.time.end_date)
    
    # Convert years to trading days (252 per year)
    formation_days = int(config.features.formation_years * 252)
    training_days = int(config.features.training_years * 252)
    
    # Feature lag from config (already in trading days)
    feature_lag = config.time.FEATURE_MAX_LAG_DAYS
    
    # Holding period
    holding_days = config.time.HOLDING_PERIOD_DAYS
    
    # Total lookback needed before first rebalance
    total_lookback = formation_days + training_days + feature_lag
    
    # Convert trading days to calendar days (rough: multiply by 365/252 ≈ 1.45)
    # Add buffer for weekends/holidays
    calendar_days_buffer = int(total_lookback * 1.5)
    
    # Calculate earliest date needed
    earliest_needed = config_start - pd.Timedelta(days=calendar_days_buffer)
    
    # Latest date needed (config end + holding period buffer)
    latest_needed = config_end + pd.Timedelta(days=int(holding_days * 1.5))
    
    # Cap latest to today (can't get future data)
    today = pd.Timestamp.now().normalize()
    latest_needed = min(latest_needed, today)
    
    return DataRequirements(
        config_start_date=config_start,
        config_end_date=config_end,
        formation_trading_days=formation_days,
        training_trading_days=training_days,
        feature_max_lag_days=feature_lag,
        holding_period_days=holding_days,
        earliest_data_needed=earliest_needed,
        latest_data_needed=latest_needed,
        total_lookback_trading_days=total_lookback,
    )


def validate_data_requirements(
    ohlcv_data: Dict[str, pd.DataFrame],
    requirements: DataRequirements,
    tolerance_days: int = 5,
    verbose: bool = True,
) -> Dict[str, any]:
    """
    Validate that downloaded OHLCV data meets the requirements.
    
    Args:
        ohlcv_data: Dict of {ticker: DataFrame} from download_ohlcv_data
        requirements: DataRequirements object
        tolerance_days: Allow this many calendar days slack for weekends/holidays
        verbose: Print detailed validation report
        
    Returns:
        Dict with validation results:
            - valid: bool
            - tickers_ok: List[str]
            - tickers_insufficient: List[str]  
            - details: Dict[str, Dict] per-ticker details
    """
    results = {
        'valid': True,
        'tickers_ok': [],
        'tickers_insufficient': [],
        'details': {}
    }
    
    # Cap latest needed to today
    today = pd.Timestamp.now().normalize()
    latest_needed = min(requirements.latest_data_needed, today)
    earliest_needed = requirements.earliest_data_needed
    
    # Add tolerance
    earliest_with_tolerance = earliest_needed + pd.Timedelta(days=tolerance_days)
    
    if verbose:
        print("=" * 70)
        print("DATA AVAILABILITY VALIDATION")
        print("=" * 70)
        print(f"  Config start_date:      {requirements.config_start_date.date()}")
        print(f"  Config end_date:        {requirements.config_end_date.date()}")
        print(f"  Feature max lag:        {requirements.feature_max_lag_days} trading days")
        print(f"  Formation window:       {requirements.formation_trading_days} trading days")
        print(f"  Training window:        {requirements.training_trading_days} trading days")
        print(f"  Holding period:         {requirements.holding_period_days} trading days")
        print(f"  {'─' * 43}")
        print(f"  Earliest data needed:   {earliest_needed.date()} (+{tolerance_days}d tolerance)")
        print(f"  Latest data needed:     {latest_needed.date()} (capped at today)")
        print(f"  Min coverage required:  90%")
        print(f"  Tickers to validate:    {len(ohlcv_data)}")
        print("=" * 70)
    
    for ticker, df in ohlcv_data.items():
        if df is None or len(df) == 0:
            results['tickers_insufficient'].append(ticker)
            results['details'][ticker] = {'error': 'No data'}
            continue
        
        ticker_start = df.index.min()
        ticker_end = df.index.max()
        
        # Check if ticker data covers required range (with tolerance)
        start_ok = ticker_start <= earliest_with_tolerance
        end_ok = ticker_end >= latest_needed - pd.Timedelta(days=tolerance_days)
        
        # Calculate coverage
        if start_ok and end_ok:
            results['tickers_ok'].append(ticker)
        else:
            results['tickers_insufficient'].append(ticker)
            results['valid'] = False
            
            if verbose:
                issues = []
                if not start_ok:
                    gap = (ticker_start - earliest_needed).days
                    issues.append(f"starts {gap}d late ({ticker_start.date()})")
                if not end_ok:
                    gap = (latest_needed - ticker_end).days
                    issues.append(f"ends {gap}d early ({ticker_end.date()})")
                print(f"  ✗ {ticker}: {'; '.join(issues)}")
        
        results['details'][ticker] = {
            'start': ticker_start,
            'end': ticker_end,
            'rows': len(df),
            'start_ok': start_ok,
            'end_ok': end_ok,
        }
    
    if verbose:
        if results['valid']:
            print(f"\n✓ ALL {len(results['tickers_ok'])} tickers have sufficient data coverage")
        else:
            print(f"\n✗ {len(results['tickers_insufficient'])}/{len(ohlcv_data)} tickers have insufficient data")
    
    return results

