"""
Panel Builder for Cross-Sectional Momentum Strategy

Creates clean panels from raw OHLCV data with:
- Configurable date ranges based on TimeConfig
- Metadata tracking for reproducibility
- Support for both OHLCV panels and feature panels

Architecture:
    Raw OHLCV (per-ticker parquet) 
        → PanelBuilder.build_ohlcv_panel() 
            → Clean OHLCV Panel (multiindex: Date, Ticker)
                → feature_engineering.py 
                    → Feature Panel (multiindex: Date, Ticker)

Usage:
    from panel_builder import PanelBuilder
    
    builder = PanelBuilder(config)
    panel, metadata = builder.build_ohlcv_panel()
    
    # Or use convenience function
    from panel_builder import build_clean_panel
    panel, metadata = build_clean_panel(config)
"""

from __future__ import annotations
import datetime as dt
import json
import hashlib
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd

from data_manager import CrossSecMomDataManager, clean_raw_ohlcv_data, filter_tickers_for_research_window


# ============================================================================
# PANEL METADATA TRACKING
# ============================================================================

@dataclass
class PanelMetadata:
    """
    Metadata for a built panel, used for tracking and reproducibility.
    
    Stored alongside panel as JSON file with same name + '.metadata.json'
    """
    # Panel identification
    panel_name: str
    panel_type: str  # 'ohlcv', 'feature', 'return'
    
    # Date range
    start_date: str
    end_date: str
    trading_days: int
    
    # Universe
    tickers: List[str]
    ticker_count: int
    
    # Data quality
    coverage_pct: float  # % of cells that have data (vs NaN)
    min_ticker_coverage: float  # Worst ticker coverage
    max_ticker_coverage: float  # Best ticker coverage
    
    # Build info
    built_at: str  # ISO timestamp
    config_hash: str  # Hash of config used
    builder_version: str
    
    # Optional notes
    notes: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'PanelMetadata':
        """Create from dictionary."""
        return cls(**d)
    
    def save(self, filepath: Path) -> None:
        """Save metadata to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load(cls, filepath: Path) -> 'PanelMetadata':
        """Load metadata from JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_dict(json.load(f))


def compute_config_hash(config) -> str:
    """Compute a hash of relevant config parameters for reproducibility tracking."""
    # Extract key config values that affect panel building
    config_dict = {
        'start_date': str(config.time.start_date),
        'end_date': str(config.time.end_date),
        'feature_max_lag_days': config.time.FEATURE_MAX_LAG_DAYS,
        'formation_years': config.features.formation_years,
        'training_years': config.features.training_years,
        'etf_universe': sorted(config.universe.etf_universe) if hasattr(config.universe, 'etf_universe') else [],
    }
    
    # Create deterministic hash
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:12]


# ============================================================================
# PANEL BUILDER CLASS
# ============================================================================

class PanelBuilder:
    """
    Builds clean panels from raw OHLCV data for cross-sectional momentum strategy.
    
    Responsibilities:
    1. Load raw OHLCV data from cache (via data_manager)
    2. Filter tickers for required date range
    3. Clean data (forward-fill gaps, remove too-sparse tickers)
    4. Build multiindex panel (Date, Ticker)
    5. Track metadata for reproducibility
    
    Does NOT:
    - Download data (use data_manager for that)
    - Compute features (use feature_engineering for that)
    - Run backtests (use backtest for that)
    """
    
    VERSION = "1.0.0"
    
    def __init__(self, config, data_dir: Optional[str] = None):
        """
        Initialize panel builder.
        
        Args:
            config: CrossSecMomConfig object
            data_dir: Data directory override. If None, uses config.paths.data_dir
        """
        self.config = config
        self.data_dir = Path(data_dir) if data_dir else Path(config.paths.data_dir)
        self._manager = CrossSecMomDataManager(data_dir=str(self.data_dir))
        
    def build_ohlcv_panel(
        self,
        tickers: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        warmup_days: Optional[int] = None,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, PanelMetadata]:
        """
        Build a clean OHLCV panel from cached raw data.
        
        The panel has a MultiIndex (Date, Ticker) with columns [Open, High, Low, Close, Volume].
        
        Args:
            tickers: List of tickers. If None, uses config.universe.etf_universe
            start_date: Start date (YYYY-MM-DD). If None, uses config with lookback
            end_date: End date (YYYY-MM-DD). If None, uses config
            warmup_days: Trading days needed before start_date for feature computation.
                        If None, computed from config (formation + training + feature lag)
            verbose: Print progress
            
        Returns:
            Tuple of (panel DataFrame, PanelMetadata)
        """
        # Resolve parameters from config
        if tickers is None:
            tickers = self.config.universe.etf_universe
        
        if start_date is None:
            start_date = self.config.time.start_date
        
        if end_date is None:
            end_date = self.config.time.end_date
        
        if warmup_days is None:
            # Calculate warmup from config: formation + training + feature lag
            formation_days = int(self.config.features.formation_years * 252)
            training_days = int(self.config.features.training_years * 252)
            feature_lag = self.config.time.FEATURE_MAX_LAG_DAYS
            warmup_days = formation_days + training_days + feature_lag
        
        if verbose:
            print("=" * 70)
            print("PANEL BUILDER: Building OHLCV Panel")
            print("=" * 70)
            print(f"  Requested tickers:    {len(tickers)}")
            print(f"  Date range:           {start_date} to {end_date}")
            print(f"  Warmup days:          {warmup_days} trading days")
            print(f"  Data directory:       {self.data_dir}")
            print("=" * 70)
        
        # Step 1: Calculate earliest date needed (with warmup)
        # Convert trading days to calendar days (1.5x multiplier + buffer)
        calendar_warmup = int(warmup_days * 1.5) + 30  # Extra buffer for holidays
        earliest_date = (pd.to_datetime(start_date) - pd.Timedelta(days=calendar_warmup)).strftime('%Y-%m-%d')
        
        if verbose:
            print(f"\n[Step 1] Earliest data needed: {earliest_date}")
        
        # Step 2: Load raw OHLCV data from cache
        if verbose:
            print(f"\n[Step 2] Loading raw OHLCV data from cache...")
        
        ohlcv_dict = self._manager.download_ohlcv_data(
            tickers=tickers,
            start_date=earliest_date,
            end_date=end_date,
            batch_sleep=0.5
        )
        
        if not ohlcv_dict:
            raise ValueError("No OHLCV data could be loaded. Run data download first.")
        
        if verbose:
            print(f"  Loaded data for {len(ohlcv_dict)}/{len(tickers)} tickers")
        
        # Step 3: Filter tickers for research window
        if verbose:
            print(f"\n[Step 3] Filtering tickers for research window...")
        
        filtered_dict = filter_tickers_for_research_window(
            data_dict=ohlcv_dict,
            formation_start_date=start_date,
            warmup_days=warmup_days,
            verbose=verbose
        )
        
        # Step 4: Clean data (forward-fill gaps)
        if verbose:
            print(f"\n[Step 4] Cleaning data (forward-fill gaps)...")
        
        cleaned_dict = clean_raw_ohlcv_data(
            data_dict=filtered_dict,
            ffill_limit=5,
            verbose=verbose
        )
        
        # Step 5: Build multiindex panel
        if verbose:
            print(f"\n[Step 5] Building multiindex panel...")
        
        panel = self._build_multiindex_panel(cleaned_dict)
        
        if verbose:
            print(f"  Panel shape: {panel.shape}")
            print(f"  Date range: {panel.index.get_level_values(0).min().date()} to {panel.index.get_level_values(0).max().date()}")
            print(f"  Tickers: {panel.index.get_level_values(1).nunique()}")
        
        # Step 6: Compute metadata
        metadata = self._compute_metadata(
            panel=panel,
            panel_name='ohlcv_panel',
            panel_type='ohlcv',
        )
        
        if verbose:
            print(f"\n[Complete] OHLCV panel built successfully")
            print(f"  Coverage: {metadata.coverage_pct:.1f}%")
            print(f"  Config hash: {metadata.config_hash}")
        
        return panel, metadata
    
    def _build_multiindex_panel(self, data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Convert dict of {ticker: DataFrame} to multiindex (Date, Ticker) panel.
        
        Args:
            data_dict: Dict of {ticker: DataFrame} with OHLCV columns
            
        Returns:
            DataFrame with MultiIndex (Date, Ticker) and columns [Open, High, Low, Close, Volume]
        """
        if not data_dict:
            raise ValueError("data_dict is empty")
        
        # Stack all tickers into long format
        dfs = []
        for ticker, df in data_dict.items():
            df_copy = df.copy()
            df_copy['Ticker'] = ticker
            df_copy = df_copy.reset_index()
            df_copy = df_copy.rename(columns={'index': 'Date', 'Date': 'Date'})
            dfs.append(df_copy)
        
        long_df = pd.concat(dfs, ignore_index=True)
        
        # Ensure Date column exists and is datetime
        if 'Date' not in long_df.columns:
            # Check for common alternatives
            date_cols = [c for c in long_df.columns if 'date' in c.lower() or c == 'index']
            if date_cols:
                long_df = long_df.rename(columns={date_cols[0]: 'Date'})
            else:
                raise ValueError("Could not find date column in data")
        
        long_df['Date'] = pd.to_datetime(long_df['Date'])
        
        # Set multiindex
        long_df = long_df.set_index(['Date', 'Ticker'])
        long_df = long_df.sort_index()
        
        # Ensure we have OHLCV columns
        ohlcv_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        available_cols = [c for c in ohlcv_cols if c in long_df.columns]
        
        if not available_cols:
            raise ValueError(f"No OHLCV columns found. Available: {long_df.columns.tolist()}")
        
        return long_df[available_cols]
    
    def _compute_metadata(
        self,
        panel: pd.DataFrame,
        panel_name: str,
        panel_type: str,
    ) -> PanelMetadata:
        """Compute metadata for a built panel."""
        
        dates = panel.index.get_level_values(0)
        tickers = panel.index.get_level_values(1).unique().tolist()
        
        # Compute coverage statistics
        total_cells = panel.size
        non_nan_cells = panel.notna().sum().sum()
        coverage_pct = 100.0 * non_nan_cells / total_cells if total_cells > 0 else 0.0
        
        # Per-ticker coverage (for Close column)
        if 'Close' in panel.columns:
            ticker_coverage = panel['Close'].groupby(level='Ticker').apply(
                lambda x: 100.0 * x.notna().sum() / len(x) if len(x) > 0 else 0.0
            )
            min_coverage = ticker_coverage.min()
            max_coverage = ticker_coverage.max()
        else:
            min_coverage = max_coverage = coverage_pct
        
        return PanelMetadata(
            panel_name=panel_name,
            panel_type=panel_type,
            start_date=str(dates.min().date()),
            end_date=str(dates.max().date()),
            trading_days=dates.nunique(),
            tickers=tickers,
            ticker_count=len(tickers),
            coverage_pct=coverage_pct,
            min_ticker_coverage=min_coverage,
            max_ticker_coverage=max_coverage,
            built_at=dt.datetime.now().isoformat(),
            config_hash=compute_config_hash(self.config),
            builder_version=self.VERSION,
        )
    
    def save_panel(
        self,
        panel: pd.DataFrame,
        metadata: PanelMetadata,
        filename: str,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save panel and metadata to disk.
        
        Args:
            panel: Panel DataFrame
            metadata: PanelMetadata object
            filename: Base filename (without extension)
            output_dir: Output directory. If None, uses data_dir
            
        Returns:
            Path to saved panel file
        """
        output_dir = output_dir or self.data_dir
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save panel as parquet
        panel_path = output_dir / f"{filename}.parquet"
        panel.to_parquet(panel_path)
        
        # Save metadata as JSON
        metadata_path = output_dir / f"{filename}.metadata.json"
        metadata.save(metadata_path)
        
        print(f"[PanelBuilder] Saved panel to {panel_path}")
        print(f"[PanelBuilder] Saved metadata to {metadata_path}")
        
        return panel_path
    
    @classmethod
    def load_panel(cls, filepath: Path) -> Tuple[pd.DataFrame, Optional[PanelMetadata]]:
        """
        Load panel and metadata from disk.
        
        Args:
            filepath: Path to panel parquet file
            
        Returns:
            Tuple of (panel DataFrame, PanelMetadata or None if not found)
        """
        filepath = Path(filepath)
        
        # Load panel
        panel = pd.read_parquet(filepath)
        
        # Try to load metadata
        metadata_path = filepath.with_suffix('.metadata.json')
        metadata = None
        if metadata_path.exists():
            try:
                metadata = PanelMetadata.load(metadata_path)
            except Exception as e:
                print(f"[PanelBuilder] Warning: Could not load metadata: {e}")
        
        return panel, metadata


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def build_clean_panel(config, verbose: bool = True) -> Tuple[pd.DataFrame, PanelMetadata]:
    """
    Convenience function to build a clean OHLCV panel.
    
    Args:
        config: CrossSecMomConfig object
        verbose: Print progress
        
    Returns:
        Tuple of (panel DataFrame, PanelMetadata)
    """
    builder = PanelBuilder(config)
    return builder.build_ohlcv_panel(verbose=verbose)


def get_panel_summary(panel: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary statistics for a panel.
    
    Args:
        panel: Panel DataFrame with MultiIndex (Date, Ticker)
        
    Returns:
        Dict with summary statistics
    """
    dates = panel.index.get_level_values(0)
    tickers = panel.index.get_level_values(1).unique()
    
    return {
        'date_range': (str(dates.min().date()), str(dates.max().date())),
        'trading_days': dates.nunique(),
        'ticker_count': len(tickers),
        'total_rows': len(panel),
        'columns': panel.columns.tolist(),
        'coverage_pct': 100.0 * panel.notna().sum().sum() / panel.size if panel.size > 0 else 0.0,
        'memory_mb': panel.memory_usage(deep=True).sum() / 1024 / 1024,
    }


# ============================================================================
# PANEL METADATA TABLE (for tracking multiple panels)
# ============================================================================

class PanelMetadataTable:
    """
    Manages a table of panel metadata for tracking and versioning.
    
    Stored as a JSON file with all panel metadata records.
    
    Usage:
        table = PanelMetadataTable(data_dir)
        
        # Add metadata when building panels
        table.add(metadata)
        table.save()
        
        # Query existing panels
        all_panels = table.list_panels()
        ohlcv_panels = table.get_by_type('ohlcv')
        latest = table.get_latest('ohlcv')
    """
    
    def __init__(self, data_dir: Path):
        """Initialize metadata table."""
        self.data_dir = Path(data_dir)
        self.filepath = self.data_dir / 'panel_metadata_table.json'
        self._records: Dict[str, Dict] = {}
        self._load()
    
    def _load(self) -> None:
        """Load existing metadata table from disk."""
        if self.filepath.exists():
            try:
                with open(self.filepath, 'r') as f:
                    self._records = json.load(f)
            except Exception as e:
                print(f"[PanelMetadataTable] Warning: Could not load metadata table: {e}")
                self._records = {}
    
    def save(self) -> None:
        """Save metadata table to disk."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with open(self.filepath, 'w') as f:
            json.dump(self._records, f, indent=2)
    
    def add(self, metadata: PanelMetadata) -> str:
        """
        Add panel metadata to table.
        
        Args:
            metadata: PanelMetadata object
            
        Returns:
            Record key (panel_name + timestamp)
        """
        key = f"{metadata.panel_name}_{metadata.built_at.replace(':', '-')}"
        self._records[key] = metadata.to_dict()
        return key
    
    def list_panels(self) -> List[Dict]:
        """Get all panel metadata records."""
        return list(self._records.values())
    
    def get_by_type(self, panel_type: str) -> List[Dict]:
        """Get all panels of a specific type."""
        return [r for r in self._records.values() if r.get('panel_type') == panel_type]
    
    def get_by_name(self, panel_name: str) -> List[Dict]:
        """Get all panels with a specific name."""
        return [r for r in self._records.values() if r.get('panel_name') == panel_name]
    
    def get_latest(self, panel_type: Optional[str] = None, panel_name: Optional[str] = None) -> Optional[Dict]:
        """
        Get the most recent panel metadata matching criteria.
        
        Args:
            panel_type: Filter by panel type (optional)
            panel_name: Filter by panel name (optional)
            
        Returns:
            Most recent matching metadata dict, or None
        """
        records = self.list_panels()
        
        if panel_type:
            records = [r for r in records if r.get('panel_type') == panel_type]
        if panel_name:
            records = [r for r in records if r.get('panel_name') == panel_name]
        
        if not records:
            return None
        
        # Sort by built_at descending
        records.sort(key=lambda r: r.get('built_at', ''), reverse=True)
        return records[0]
    
    def to_dataframe(self) -> pd.DataFrame:
        """Convert metadata table to DataFrame for easy viewing."""
        if not self._records:
            return pd.DataFrame()
        
        df = pd.DataFrame(self._records.values())
        
        # Reorder columns for readability
        priority_cols = ['panel_name', 'panel_type', 'start_date', 'end_date', 
                        'ticker_count', 'trading_days', 'coverage_pct', 'built_at']
        other_cols = [c for c in df.columns if c not in priority_cols and c != 'tickers']
        
        return df[priority_cols + other_cols]


# ============================================================================
# DATA PIPELINE ORCHESTRATOR
# ============================================================================

class DataPipeline:
    """
    Orchestrates the complete data flow from raw data to feature panel.
    
    Pipeline stages:
        1. Raw OHLCV download (data_manager) 
        2. Clean panel creation (panel_builder)
        3. Feature engineering (feature_engineering)
        4. Target computation (feature_engineering)
    
    This class coordinates all stages and tracks lineage through metadata.
    
    Usage:
        from panel_builder import DataPipeline
        
        pipeline = DataPipeline(config)
        
        # Run full pipeline
        feature_panel = pipeline.run()
        
        # Or run individual stages
        pipeline.ensure_raw_data()
        ohlcv_panel = pipeline.build_ohlcv_panel()
        feature_panel = pipeline.build_feature_panel(ohlcv_panel)
    """
    
    def __init__(self, config, data_dir: Optional[str] = None):
        """
        Initialize data pipeline.
        
        Args:
            config: CrossSecMomConfig object
            data_dir: Data directory override. If None, uses config.paths.data_dir
        """
        self.config = config
        self.data_dir = Path(data_dir) if data_dir else Path(config.paths.data_dir)
        self._manager = CrossSecMomDataManager(data_dir=str(self.data_dir))
        self._builder = PanelBuilder(config, data_dir=str(self.data_dir))
        self._metadata_table = PanelMetadataTable(self.data_dir)
    
    def ensure_raw_data(
        self,
        tickers: Optional[List[str]] = None,
        use_full_history: bool = False,
        overwrite: bool = False,
    ) -> Dict[str, Dict]:
        """
        Ensure raw OHLCV data is available for all tickers.
        
        Args:
            tickers: List of tickers. If None, uses config.universe.etf_universe
            use_full_history: If True, downloads max available history (period='max')
            overwrite: If True, re-downloads even if cached
            
        Returns:
            Dict of {ticker: status_dict} for each ticker
        """
        if tickers is None:
            tickers = self.config.universe.etf_universe
        
        print("=" * 70)
        print("DATA PIPELINE: Ensuring Raw Data")
        print("=" * 70)
        
        if use_full_history:
            print("[Pipeline] Downloading full history for all tickers...")
            return self._manager.download_full_history(
                tickers=tickers,
                overwrite_existing=overwrite
            )
        else:
            # Calculate date range from config
            start_date = self._calculate_earliest_date()
            end_date = self.config.time.end_date
            
            print(f"[Pipeline] Downloading data from {start_date} to {end_date}...")
            result = self._manager.download_ohlcv_data(
                tickers=tickers,
                start_date=start_date,
                end_date=end_date
            )
            
            # Convert to status dict format
            return {ticker: {'status': 'downloaded', 'rows': len(df)} 
                    for ticker, df in result.items()}
    
    def _calculate_earliest_date(self) -> str:
        """Calculate earliest date needed based on config."""
        # formation + training + feature lag (all in trading days)
        formation_days = int(self.config.features.formation_years * 252)
        training_days = int(self.config.features.training_years * 252)
        feature_lag = self.config.time.FEATURE_MAX_LAG_DAYS
        
        total_lookback = formation_days + training_days + feature_lag
        
        # Convert to calendar days (1.5x + buffer)
        calendar_days = int(total_lookback * 1.5) + 60
        
        start = pd.to_datetime(self.config.time.start_date) - pd.Timedelta(days=calendar_days)
        return start.strftime('%Y-%m-%d')
    
    def build_ohlcv_panel(
        self,
        save: bool = True,
        verbose: bool = True,
    ) -> Tuple[pd.DataFrame, PanelMetadata]:
        """
        Build clean OHLCV panel from raw data.
        
        Args:
            save: If True, saves panel to disk
            verbose: Print progress
            
        Returns:
            Tuple of (panel DataFrame, PanelMetadata)
        """
        panel, metadata = self._builder.build_ohlcv_panel(verbose=verbose)
        
        if save:
            self._builder.save_panel(panel, metadata, 'ohlcv_panel')
            self._metadata_table.add(metadata)
            self._metadata_table.save()
        
        return panel, metadata
    
    def build_feature_panel(
        self,
        ohlcv_panel: Optional[pd.DataFrame] = None,
        save: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Build feature panel from OHLCV panel.
        
        Note: This method imports feature_engineering to avoid circular imports.
        
        Args:
            ohlcv_panel: OHLCV panel. If None, builds from raw data
            save: If True, saves panel to disk
            verbose: Print progress
            
        Returns:
            Feature panel DataFrame
        """
        # Import here to avoid circular dependency
        from feature_engineering import CrossSecMomFeatureEngineer
        
        if ohlcv_panel is None:
            print("[Pipeline] OHLCV panel not provided, building...")
            ohlcv_panel, _ = self.build_ohlcv_panel(save=save, verbose=verbose)
        
        print("\n" + "=" * 70)
        print("DATA PIPELINE: Building Feature Panel")
        print("=" * 70)
        
        # Convert multiindex panel to dict format expected by feature_engineering
        # This is a bridge until we refactor feature_engineering to work with panels directly
        ohlcv_dict = self._panel_to_dict(ohlcv_panel)
        
        # Build features
        engineer = CrossSecMomFeatureEngineer(self.config)
        feature_panel = engineer.build_feature_panel(
            ohlcv_data=ohlcv_dict,
            verbose=verbose
        )
        
        if save:
            output_path = self.data_dir / 'feature_panel.parquet'
            feature_panel.to_parquet(output_path)
            print(f"[Pipeline] Saved feature panel to {output_path}")
        
        return feature_panel
    
    def _panel_to_dict(self, panel: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Convert multiindex panel back to dict of {ticker: DataFrame}.
        
        This is a temporary bridge until feature_engineering is refactored.
        """
        result = {}
        tickers = panel.index.get_level_values('Ticker').unique()
        
        for ticker in tickers:
            ticker_data = panel.xs(ticker, level='Ticker')
            result[ticker] = ticker_data
        
        return result
    
    def run(
        self,
        ensure_data: bool = True,
        use_full_history: bool = False,
        save: bool = True,
        verbose: bool = True,
    ) -> pd.DataFrame:
        """
        Run the complete data pipeline.
        
        Args:
            ensure_data: If True, ensures raw data is downloaded first
            use_full_history: If True, downloads max available history
            save: If True, saves intermediate results to disk
            verbose: Print progress
            
        Returns:
            Feature panel DataFrame
        """
        print("\n" + "=" * 70)
        print("DATA PIPELINE: Running Full Pipeline")
        print("=" * 70)
        print(f"  Config: {self.config.time.start_date} to {self.config.time.end_date}")
        print(f"  Tickers: {len(self.config.universe.etf_universe)}")
        print(f"  Data dir: {self.data_dir}")
        print("=" * 70)
        
        # Stage 1: Ensure raw data
        if ensure_data:
            self.ensure_raw_data(use_full_history=use_full_history)
        
        # Stage 2: Build OHLCV panel
        ohlcv_panel, _ = self.build_ohlcv_panel(save=save, verbose=verbose)
        
        # Stage 3: Build feature panel
        feature_panel = self.build_feature_panel(
            ohlcv_panel=ohlcv_panel,
            save=save,
            verbose=verbose
        )
        
        print("\n" + "=" * 70)
        print("DATA PIPELINE: Complete")
        print("=" * 70)
        print(f"  Feature panel shape: {feature_panel.shape}")
        
        return feature_panel
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current status of the data pipeline.
        
        Returns:
            Dict with status information
        """
        status = {
            'data_dir': str(self.data_dir),
            'ohlcv_files': len(list((self.data_dir / 'ohlcv').glob('*.parquet'))) if (self.data_dir / 'ohlcv').exists() else 0,
        }
        
        # Check for OHLCV panel
        ohlcv_path = self.data_dir / 'ohlcv_panel.parquet'
        if ohlcv_path.exists():
            status['ohlcv_panel'] = {
                'exists': True,
                'modified': dt.datetime.fromtimestamp(ohlcv_path.stat().st_mtime).isoformat(),
                'size_mb': ohlcv_path.stat().st_size / 1024 / 1024,
            }
        else:
            status['ohlcv_panel'] = {'exists': False}
        
        # Check for feature panel
        feature_path = self.data_dir / 'feature_panel.parquet'
        if feature_path.exists():
            status['feature_panel'] = {
                'exists': True,
                'modified': dt.datetime.fromtimestamp(feature_path.stat().st_mtime).isoformat(),
                'size_mb': feature_path.stat().st_size / 1024 / 1024,
            }
        else:
            status['feature_panel'] = {'exists': False}
        
        # Get latest metadata
        latest = self._metadata_table.get_latest()
        if latest:
            status['latest_panel'] = {
                'name': latest.get('panel_name'),
                'built_at': latest.get('built_at'),
                'config_hash': latest.get('config_hash'),
            }
        
        return status


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Build clean OHLCV panel')
    parser.add_argument('--save', action='store_true', help='Save panel to disk')
    parser.add_argument('--output', type=str, default='ohlcv_panel', help='Output filename')
    args = parser.parse_args()
    
    # Load config
    from config import get_default_config
    config = get_default_config()
    
    # Build panel
    builder = PanelBuilder(config)
    panel, metadata = builder.build_ohlcv_panel()
    
    # Print summary
    print("\n" + "=" * 70)
    print("PANEL SUMMARY")
    print("=" * 70)
    summary = get_panel_summary(panel)
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Save if requested
    if args.save:
        builder.save_panel(panel, metadata, args.output)
