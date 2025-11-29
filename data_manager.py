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
from dataclasses import dataclass
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

