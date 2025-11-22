"""
Intelligent Data Manager for Cross-Sectional Momentum Strategy

Implements incremental data downloading:
- Downloads ETF OHLCV data to cache directory
- Downloads macro data (VIX, yields)
- Only downloads missing dates
- Preserves existing data and merges intelligently

Based on alpha_signals data_manager.py philosophy.
"""

from __future__ import annotations
import datetime as dt
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import warnings
import time

import numpy as np
import pandas as pd
import yfinance as yf
import logging


class CrossSecMomDataManager:
    """Manages data downloading and caching with incremental updates."""

    def __init__(
        self,
        data_dir: Optional[str] = None,
    ):
        """
        Initialize data manager.

        Args:
            data_dir: Directory for data storage. If None, uses default from config.
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
                        print(f"             → Check if {ticker} has historical data back to {start_date}")
                        print(f"             → Consider alternative ETF if ticker is delisted or too recent")

                except Exception as e:
                    print(f"  [ERROR] {ticker}: {e}")
                    print(f"         → Verify {ticker} exists and has data back to {start_date}")

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
                        print(f"             → Verify macro ticker {ticker} exists and has data back to {start_date}")
                        data_series = pd.Series(dtype=float)

                except Exception as e:
                    print(f"  [ERROR] Could not download {name} ({ticker}): {e}")
                    print(f"         → Check if macro ticker {ticker} is correct and has historical data back to {start_date}")
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
        
        Uses incremental updates:
        - Checks cache for existing data
        - Only downloads missing dates
        - Saves to cache for future use
        
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
            # Import here to avoid circular dependency
            from config import get_default_config
            batch_sleep = get_default_config().compute.batch_sleep
        
        print(f"[DataManager] Fetching OHLCV data for {len(tickers)} ETFs from {start_date} to {end_date}...")
        
        data_dict = {}
        failed = []
        
        for i, ticker in enumerate(tickers):
            try:
                # Check for cached data
                cache_path = self.ohlcv_dir / f"{ticker}.parquet"
                
                existing_df = None
                needs_download = True
                download_start = start_date
                
                if cache_path.exists():
                    try:
                        existing_df = pd.read_parquet(cache_path)
                        
                        # Ensure timezone-naive
                        if existing_df.index.tz is not None:
                            existing_df.index = existing_df.index.tz_localize(None)
                        
                        # Check date coverage
                        cached_start = existing_df.index.min()
                        cached_end = existing_df.index.max()
                        
                        requested_start = pd.to_datetime(start_date)
                        requested_end = pd.to_datetime(end_date)
                        
                        # If cache covers requested range, use it
                        if cached_start <= requested_start and cached_end >= requested_end:
                            # Filter to requested range
                            mask = (existing_df.index >= requested_start) & (existing_df.index <= requested_end)
                            filtered_df = existing_df.loc[mask]
                            
                            if len(filtered_df) > 0:
                                data_dict[ticker] = filtered_df[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float32')
                                needs_download = False
                        
                        # If we need more recent data
                        elif cached_end < requested_end:
                            # Only download from where cache ends
                            download_start = (cached_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                            # Will merge with existing later
                        
                        # If we need older data (backfill case)
                        elif cached_start > requested_start:
                            # Download full range and merge
                            pass
                            
                    except Exception as e:
                        print(f"  [warning] {ticker}: Failed to load cache ({e}), will download fresh")
                        existing_df = None
                
                # Download data if needed
                if needs_download:
                    df = yf.download(
                        ticker,
                        start=download_start,
                        end=end_date,
                        progress=False,
                        auto_adjust=False  # Keep raw OHLC
                    )
                    
                    if df.empty:
                        # If download failed but we have cached data, use it
                        if existing_df is not None:
                            requested_start = pd.to_datetime(start_date)
                            requested_end = pd.to_datetime(end_date)
                            mask = (existing_df.index >= requested_start) & (existing_df.index <= requested_end)
                            filtered_df = existing_df.loc[mask]
                            if len(filtered_df) > 0:
                                data_dict[ticker] = filtered_df[['Open', 'High', 'Low', 'Close', 'Volume']].astype('float32')
                                continue
                        failed.append(ticker)
                        continue
                    
                    # Ensure timezone-naive
                    if df.index.tz is not None:
                        df.index = df.index.tz_localize(None)
                    
                    # Handle MultiIndex columns (happens with single ticker sometimes)
                    if isinstance(df.columns, pd.MultiIndex):
                        df = df.droplevel(1, axis=1)
                    
                    # Keep only OHLCV (Adj Close not needed for features)
                    if 'Adj Close' in df.columns:
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    else:
                        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
                    
                    # Merge with existing data if we have it
                    if existing_df is not None:
                        # Combine old and new data
                        combined = pd.concat([existing_df, df]).sort_index()
                        # Remove duplicates (keep most recent)
                        combined = combined[~combined.index.duplicated(keep='last')]
                        df = combined
                    
                    # Save to cache
                    df.to_parquet(cache_path, engine='pyarrow', compression='snappy')
                    
                    # Filter to requested range
                    requested_start = pd.to_datetime(start_date)
                    requested_end = pd.to_datetime(end_date)
                    mask = (df.index >= requested_start) & (df.index <= requested_end)
                    df = df.loc[mask]
                    
                    if len(df) == 0:
                        failed.append(ticker)
                        continue
                    
                    # Convert to float32 for memory efficiency
                    data_dict[ticker] = df.astype('float32')
                
                # Progress update
                if (i + 1) % 20 == 0:
                    print(f"  [progress] {i+1}/{len(tickers)}")
                    time.sleep(batch_sleep)
                    
            except Exception as e:
                print(f"  [error] {ticker}: {e}")
                failed.append(ticker)
        
        print(f"\n[DataManager] Successfully loaded: {len(data_dict)}/{len(tickers)}")
        if failed:
            print(f"[DataManager] Failed tickers ({len(failed)}): {failed}")
        
        return data_dict


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

