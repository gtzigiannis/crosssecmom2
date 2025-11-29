"""
FRED Data Extension for CrossSecMom2
=====================================

Downloads macro and sentiment data from FRED (Federal Reserve Economic Data).

Requires: fredapi library
    pip install fredapi

Get free API key from: https://fred.stlouisfed.org/docs/api/api_key.html
Set environment variable: FRED_API_KEY=your_key_here
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, List
from datetime import datetime
import warnings


# FRED Series Definitions
# =======================

FRED_MACRO_SERIES = {
    # ---- Interest Rates & Yield Curve ----
    'DFF': 'fed_funds',                 # Effective Federal Funds Rate
    'DTB3': 'tbill_3m',                 # 3-Month Treasury Bill
    'DGS2': 'tsy_2y',                   # 2-Year Treasury Constant Maturity
    'DGS5': 'tsy_5y',                   # 5-Year Treasury Constant Maturity
    'DGS10': 'tsy_10y',                 # 10-Year Treasury Constant Maturity
    'DGS30': 'tsy_30y',                 # 30-Year Treasury Constant Maturity
    'T10Y2Y': 'yc_slope_10_2',          # 10Y-2Y Treasury Spread
    'T10Y3M': 'yc_slope_10_3m',         # 10Y-3M Treasury Spread
    
    # ---- Credit Spreads ----
    'BAMLC0A0CM': 'ig_spread',          # BofA IG Corporate Bond Spread
    'BAMLH0A0HYM2': 'hy_spread',        # BofA HY Corporate Bond Spread
    'TEDRATE': 'ted_spread',            # TED Spread (3M LIBOR - 3M T-Bill)
    'AAA10Y': 'aaa_spread',             # AAA - 10Y Treasury Spread
    'BAA10Y': 'baa_spread',             # BAA - 10Y Treasury Spread
    
    # ---- Inflation ----
    'T5YIE': 'breakeven_5y',            # 5-Year Breakeven Inflation
    'T10YIE': 'breakeven_10y',          # 10-Year Breakeven Inflation
    
    # ---- Financial Conditions ----
    'NFCI': 'chicago_fci',              # Chicago Fed National Financial Conditions Index
    'STLFSI4': 'stl_fsi',               # St. Louis Fed Financial Stress Index
}

FRED_SENTIMENT_SERIES = {
    # ---- Consumer Sentiment ----
    'UMCSENT': 'umich_sentiment',       # University of Michigan Consumer Sentiment
    
    # ---- Economic Policy Uncertainty ----
    'USEPUINDXD': 'epu_daily',          # Daily Economic Policy Uncertainty Index
}

FRED_ECONOMY_SERIES = {
    # ---- Labor Market ----
    'ICSA': 'initial_claims',           # Initial Jobless Claims (weekly)
    'UNRATE': 'unemployment',           # Unemployment Rate (monthly)
    
    # ---- Economic Activity ----
    'INDPRO': 'industrial_prod',        # Industrial Production Index (monthly)
}

# Combined for convenience
ALL_FRED_SERIES = {**FRED_MACRO_SERIES, **FRED_SENTIMENT_SERIES, **FRED_ECONOMY_SERIES}


class FREDDataManager:
    """
    Manages downloading and caching of FRED data series.
    """
    
    def __init__(
        self,
        data_dir: str,
        api_key: Optional[str] = None,
    ):
        """
        Initialize FRED data manager.
        
        Parameters
        ----------
        data_dir : str
            Directory for cached data
        api_key : str, optional
            FRED API key. If None, uses FRED_API_KEY environment variable.
        """
        self.data_dir = Path(data_dir) / "fred"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Get API key
        self.api_key = api_key or os.environ.get('FRED_API_KEY')
        
        self.fred = None
        if self.api_key:
            try:
                from fredapi import Fred
                self.fred = Fred(api_key=self.api_key)
                print(f"[FRED] API initialized successfully")
            except ImportError:
                warnings.warn("fredapi not installed. Run: pip install fredapi")
            except Exception as e:
                warnings.warn(f"Failed to initialize FRED API: {e}")
        else:
            warnings.warn(
                "FRED_API_KEY not set. Get free key from: "
                "https://fred.stlouisfed.org/docs/api/api_key.html"
            )
    
    def _get_cache_path(self, series_name: str) -> Path:
        """Get path for cached series."""
        return self.data_dir / f"FRED_{series_name}.csv"
    
    def _load_cached(self, series_name: str) -> Optional[pd.Series]:
        """Load cached data if exists."""
        cache_path = self._get_cache_path(series_name)
        
        if cache_path.exists():
            try:
                df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
                return df.iloc[:, 0]
            except Exception as e:
                warnings.warn(f"Failed to load cached {series_name}: {e}")
        
        return None
    
    def _save_cache(self, series_name: str, data: pd.Series):
        """Save data to cache."""
        cache_path = self._get_cache_path(series_name)
        pd.DataFrame({series_name: data}).to_csv(cache_path)
    
    def download_series(
        self,
        fred_code: str,
        series_name: str,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> Optional[pd.Series]:
        """
        Download a single FRED series with caching.
        
        Parameters
        ----------
        fred_code : str
            FRED series code (e.g., 'DGS10')
        series_name : str
            Local name for the series
        start_date : str
            Start date (YYYY-MM-DD)
        end_date : str
            End date (YYYY-MM-DD)
        force_refresh : bool
            If True, ignore cache and download fresh
            
        Returns
        -------
        pd.Series or None
            Series data indexed by date
        """
        # Try cache first
        if not force_refresh:
            cached = self._load_cached(series_name)
            if cached is not None:
                # Check if cache covers requested range
                start_dt = pd.to_datetime(start_date)
                end_dt = pd.to_datetime(end_date)
                
                if cached.index.min() <= start_dt and cached.index.max() >= end_dt - pd.Timedelta(days=7):
                    # Cache is sufficient (allow 7-day lag for recent data)
                    data = cached[(cached.index >= start_dt) & (cached.index <= end_dt)]
                    print(f"  [FRED] {series_name}: {len(data)} obs (cached)")
                    return data
        
        # Download from FRED
        if self.fred is None:
            warnings.warn(f"Cannot download {series_name}: FRED API not initialized")
            return self._load_cached(series_name)  # Return cached if available
        
        try:
            data = self.fred.get_series(
                fred_code,
                observation_start=start_date,
                observation_end=end_date,
            )
            
            if data is not None and len(data) > 0:
                data.name = series_name
                data.index = pd.to_datetime(data.index)
                
                # Merge with existing cache
                cached = self._load_cached(series_name)
                if cached is not None:
                    combined = pd.concat([cached, data]).sort_index()
                    combined = combined[~combined.index.duplicated(keep='last')]
                    data = combined
                
                # Save to cache
                self._save_cache(series_name, data)
                print(f"  [FRED] {series_name}: {len(data)} obs (downloaded)")
                return data
            else:
                print(f"  [FRED] {series_name}: No data available")
                return None
                
        except Exception as e:
            warnings.warn(f"Failed to download {series_name} ({fred_code}): {e}")
            return self._load_cached(series_name)  # Return cached if available
    
    def download_all_macro(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> Dict[str, pd.Series]:
        """
        Download all macro series from FRED.
        
        Returns dict of {series_name: pd.Series}
        """
        print(f"[FRED] Downloading macro data from {start_date} to {end_date}...")
        
        data = {}
        for fred_code, series_name in FRED_MACRO_SERIES.items():
            result = self.download_series(
                fred_code, series_name, start_date, end_date, force_refresh
            )
            if result is not None:
                data[series_name] = result
        
        return data
    
    def download_all_sentiment(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> Dict[str, pd.Series]:
        """
        Download all sentiment series from FRED.
        """
        print(f"[FRED] Downloading sentiment data...")
        
        data = {}
        for fred_code, series_name in FRED_SENTIMENT_SERIES.items():
            result = self.download_series(
                fred_code, series_name, start_date, end_date, force_refresh
            )
            if result is not None:
                data[series_name] = result
        
        return data
    
    def download_all_economy(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> Dict[str, pd.Series]:
        """
        Download all economic activity series from FRED.
        """
        print(f"[FRED] Downloading economic data...")
        
        data = {}
        for fred_code, series_name in FRED_ECONOMY_SERIES.items():
            result = self.download_series(
                fred_code, series_name, start_date, end_date, force_refresh
            )
            if result is not None:
                data[series_name] = result
        
        return data
    
    def download_all(
        self,
        start_date: str,
        end_date: str,
        force_refresh: bool = False,
    ) -> Dict[str, pd.Series]:
        """
        Download all FRED series (macro + sentiment + economy).
        """
        data = {}
        
        # Macro
        macro = self.download_all_macro(start_date, end_date, force_refresh)
        data.update(macro)
        
        # Sentiment
        sentiment = self.download_all_sentiment(start_date, end_date, force_refresh)
        data.update(sentiment)
        
        # Economy
        economy = self.download_all_economy(start_date, end_date, force_refresh)
        data.update(economy)
        
        print(f"[FRED] Downloaded {len(data)} total series")
        return data


def get_fred_data(
    data_dir: str,
    start_date: str,
    end_date: str,
    api_key: Optional[str] = None,
    force_refresh: bool = False,
) -> Dict[str, pd.Series]:
    """
    Convenience function to download all FRED data.
    
    Usage:
        fred_data = get_fred_data(
            data_dir="D:/REPOSITORY/Data/crosssecmom2",
            start_date="2017-01-01",
            end_date="2025-11-29",
        )
    """
    manager = FREDDataManager(data_dir, api_key)
    return manager.download_all(start_date, end_date, force_refresh)


# ============================================================================
# SF Fed News Sentiment (requires separate download)
# ============================================================================

SF_FED_NEWS_SENTIMENT_URL = "https://www.frbsf.org/wp-content/uploads/sites/4/news_sentiment_data.xlsx"

def download_sf_fed_news_sentiment(
    data_dir: str,
    force_refresh: bool = False,
) -> Optional[pd.Series]:
    """
    Download SF Fed Daily News Sentiment Index.
    
    This is a daily sentiment index derived from news articles.
    Available from: https://www.frbsf.org/research-and-insights/data-and-indicators/daily-news-sentiment-index/
    
    Returns pd.Series with daily sentiment values.
    """
    cache_path = Path(data_dir) / "fred" / "SF_NEWS_SENTIMENT.csv"
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Try cache first
    if not force_refresh and cache_path.exists():
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            print(f"  [SF Fed] News sentiment: {len(df)} obs (cached)")
            return df.iloc[:, 0]
        except Exception:
            pass
    
    # Download from SF Fed
    try:
        import requests
        from io import BytesIO
        
        print(f"  [SF Fed] Downloading Daily News Sentiment Index...")
        response = requests.get(SF_FED_NEWS_SENTIMENT_URL, timeout=30)
        response.raise_for_status()
        
        df = pd.read_excel(BytesIO(response.content), index_col=0, parse_dates=True)
        
        # Extract sentiment column
        sentiment_col = [c for c in df.columns if 'sentiment' in c.lower()]
        if sentiment_col:
            data = df[sentiment_col[0]]
            data.name = 'sf_news_sentiment'
            
            # Save to cache
            pd.DataFrame({'sf_news_sentiment': data}).to_csv(cache_path)
            print(f"  [SF Fed] News sentiment: {len(data)} obs (downloaded)")
            return data
        else:
            print(f"  [SF Fed] Could not find sentiment column in data")
            return None
            
    except Exception as e:
        warnings.warn(f"Failed to download SF Fed News Sentiment: {e}")
        return None


if __name__ == "__main__":
    # Test download
    import sys
    
    data_dir = r"D:\REPOSITORY\Data\crosssecmom2"
    
    # Check for API key
    if not os.environ.get('FRED_API_KEY'):
        print("="*60)
        print("FRED_API_KEY not set!")
        print()
        print("Get free API key from:")
        print("  https://fred.stlouisfed.org/docs/api/api_key.html")
        print()
        print("Then set environment variable:")
        print("  set FRED_API_KEY=your_key_here")
        print("="*60)
        sys.exit(1)
    
    # Download all data
    fred_data = get_fred_data(
        data_dir=data_dir,
        start_date="2017-01-01",
        end_date="2025-11-29",
    )
    
    print("\n" + "="*60)
    print("Downloaded series:")
    for name, series in fred_data.items():
        print(f"  {name}: {len(series)} observations, {series.index.min().date()} to {series.index.max().date()}")
