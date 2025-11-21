"""
Configuration Module
====================
Centralized configuration for the cross-sectional momentum research framework.
All paths, time parameters, and thresholds are defined here.
"""

from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class PathConfig:
    """File paths for data inputs and outputs."""
    
    # Input paths
    universe_csv: str = r"D:\REPOSITORY\Data\crosssecmom2\etf_universe_names.csv"
    universe_metadata_csv: str = r"D:\REPOSITORY\Data\crosssecmom2\etf_universe_full.csv"  # Full metadata with families
    returns_matrix_path: Optional[str] = None  # Optional: for correlation-based clustering
    
    # Output paths
    panel_parquet: str = r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_features.parquet"
    universe_metadata_output: str = r"D:\REPOSITORY\Data\crosssecmom2\universe_metadata.csv"
    results_csv: str = r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_results.csv"
    plots_dir: str = r"D:\REPOSITORY\Data\crosssecmom2\plots"


@dataclass
class TimeConfig:
    """Time structure parameters for walk-forward research."""
    
    # Data range
    start_date: str = "2015-01-01"
    end_date: str = "2025-11-19"
    
    # Walk-forward parameters (all in trading days)
    FEATURE_MAX_LAG_DAYS: int = 252  # Longest lookback in feature construction (1 year)
    TRAINING_WINDOW_DAYS: int = 1260  # Training window length (5 years ≈ 252*5)
    HOLDING_PERIOD_DAYS: int = 21     # Forward return horizon (1 month)
    STEP_DAYS: int = 21               # Rebalancing frequency (1 month)


@dataclass
class UniverseConfig:
    """Universe filtering and eligibility parameters."""
    
    # Liquidity filters
    min_adv_percentile: float = 0.30  # Minimum ADV percentile (30th percentile)
    adv_window: int = 63              # Window for ADV calculation
    
    # Data quality
    min_data_quality: float = 0.80    # Minimum fraction of non-NaN features
    # NOTE: min_history_days removed - history requirement is determined by
    # FEATURE_MAX_LAG_DAYS + TRAINING_WINDOW_DAYS in walk_forward_engine
    
    # Metadata filters
    exclude_leveraged: bool = True    # Exclude leveraged/inverse ETFs
    exclude_non_canonical: bool = True  # Exclude non-canonical duplicates
    
    # Duplicate detection
    dup_corr_threshold: float = 0.99  # Correlation threshold for duplicates
    
    # Clustering
    max_within_cluster_corr: float = 0.85  # Max correlation within theme clusters


@dataclass
class PortfolioConfig:
    """Portfolio construction parameters."""
    
    # Position sizing
    long_quantile: float = 0.9   # Top 10% for long portfolio
    short_quantile: float = 0.1  # Bottom 10% for short portfolio
    long_leverage: float = 1.0    # Long leverage (1.0 = 100% of capital on long side)
                                  # Shorts are separately limited by margin parameter
    
    # Portfolio mode
    long_only: bool = False      # If True, only construct long positions (cash otherwise)
    short_only: bool = False     # If True, only construct short positions (cash otherwise)
    cash_rate: float = 0.045     # Annual interest rate on cash positions (default 4.5%)
    
    # Transaction costs
    commission_bps: float = 1.0  # Commission in basis points per side
    slippage_bps: float = 2.0    # Slippage in basis points per side
    
    # Borrowing costs for shorting
    borrow_cost: float = 0.05    # Annual borrowing cost for shorts (5% on full notional)
    margin: float = 0.50         # Margin requirement: max short exposure as fraction of capital
                                 # 50% margin = can short up to 50% of capital
                                 # You pay borrow_cost on FULL short value (not margin-adjusted)
    
    # Risk limits (default caps)
    default_cluster_cap: float = 0.10  # 10% max per theme cluster
    default_per_etf_cap: float = 0.05  # 5% max per ETF
    
    @property
    def total_cost_bps_per_side(self) -> float:
        """Total transaction cost per side in basis points."""
        return self.commission_bps + self.slippage_bps
    
    # High-risk family caps (lower limits)
    high_risk_cluster_cap: float = 0.07  # 7% for risky themes
    high_risk_families: set = None
    
    def __post_init__(self):
        if self.high_risk_families is None:
            self.high_risk_families = {
                "EQ_EM_BROAD",
                "EQ_SINGLE_COUNTRY_EM",
                "COMMODITY_METALS",
                "COMMODITY_ENERGY",
                "ALT_CRYPTO"
            }


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    
    # Base features to compute (these are computed WITHOUT using targets)
    base_features: List[str] = None
    
    # Binning candidates (subset of base features for supervised binning)
    binning_candidates: List[str] = None
    
    # Feature selection
    ic_threshold: float = 0.02        # Minimum absolute IC for feature selection
    max_features: int = 20            # Maximum features to select
    
    # Binning parameters
    bin_max_depth: int = 3            # Max depth for decision tree binning
    bin_min_samples_leaf: int = 100   # Min samples per leaf
    n_bins: int = 10                  # Target number of bins
    
    # Reproducibility
    random_state: Optional[int] = 42  # Random seed for reproducible results
    
    def __post_init__(self):
        if self.base_features is None:
            # Default base feature list (no binning yet)
            # FIXED: Match actual column names from feature_engineering.py
            self.base_features = [
                # Returns at multiple horizons
                'Close%-21', 'Close%-63', 'Close%-126', 'Close%-252',
                
                # Momentum (Close_Mom*)
                'Close_Mom21', 'Close_Mom42', 'Close_Mom63',
                
                # Volatility (Close_std*, Close_ATR*)
                'Close_std21', 'Close_std63', 'Close_ATR14',
                
                # Trend indicators (Close_MA*, Close_EMA*)
                'Close_MA21', 'Close_MA63', 'Close_EMA21', 'Close_EMA63',
                
                # Oscillators (Close_RSI*, Close_MACD*)
                'Close_RSI14', 'Close_RSI21',
                'Close_MACD', 'Close_MACD_Signl', 'Close_MACD_Histo',
                
                # Liquidity
                'ADV_63', 'ADV_63_Rank',
            ]
        
        if self.binning_candidates is None:
            # Default binning candidates (subset of base features)
            # FIXED: Match actual column names from feature_engineering.py
            self.binning_candidates = [
                'Close%-21', 'Close%-63', 'Close%-126', 'Close%-252',
                'Close_Mom21', 'Close_Mom63',
                'Close_std21', 'Close_std63',
                'Close_RSI14', 'Close_RSI21',
            ]


@dataclass
class ComputeConfig:
    """Computational parameters."""
    
    n_jobs: int = 8                   # Parallel jobs for feature engineering
    verbose: bool = True              # Print progress messages
    parallelize_backtest: bool = False  # Parallelize walk-forward backtest
    
    # Persistence
    save_intermediate: bool = True    # Save intermediate objects
    ic_output_path: Optional[str] = None  # Path for IC vectors (defaults to plots_dir/ic_vectors.csv)


@dataclass
class RegimeConfig:
    """Regime classification parameters."""
    
    # Enable/disable regime-based portfolio switching
    use_regime: bool = False          # If False, use standard long/short mode
    
    # Market index for regime detection
    market_ticker: str = "SPY"        # Ticker to use for regime classification
    
    # Moving average parameters
    ma_window: int = 200              # Days for moving average
    
    # Momentum parameters
    lookback_return_days: int = 63    # Days for return calculation (3 months)
    
    # Classification thresholds
    bull_ret_threshold: float = 0.02  # 2% return threshold for bull regime
    bear_ret_threshold: float = -0.02 # -2% return threshold for bear regime
    
    # Optional hysteresis to reduce regime switching
    neutral_buffer_days: int = 21     # Days to stay in regime before switching
    use_hysteresis: bool = False      # Enable/disable hysteresis


@dataclass
class ResearchConfig:
    """Complete research configuration combining all sub-configs."""
    
    paths: PathConfig
    time: TimeConfig
    universe: UniverseConfig
    portfolio: PortfolioConfig
    features: FeatureConfig
    compute: ComputeConfig
    regime: RegimeConfig
    
    @classmethod
    def default(cls):
        """Create a default configuration."""
        return cls(
            paths=PathConfig(),
            time=TimeConfig(),
            universe=UniverseConfig(),
            portfolio=PortfolioConfig(),
            features=FeatureConfig(),
            compute=ComputeConfig(),
            regime=RegimeConfig()
        )
    
    def validate(self):
        """Validate configuration parameters."""
        # Time parameters
        assert self.time.FEATURE_MAX_LAG_DAYS > 0, "FEATURE_MAX_LAG_DAYS must be positive"
        assert self.time.TRAINING_WINDOW_DAYS > self.time.FEATURE_MAX_LAG_DAYS, \
            "TRAINING_WINDOW_DAYS must be longer than FEATURE_MAX_LAG_DAYS"
        assert self.time.HOLDING_PERIOD_DAYS > 0, "HOLDING_PERIOD_DAYS must be positive"
        assert self.time.STEP_DAYS > 0, "STEP_DAYS must be positive"
        
        # Universe filters
        assert 0 < self.universe.min_adv_percentile < 1, "min_adv_percentile must be in (0, 1)"
        assert 0 < self.universe.min_data_quality <= 1, "min_data_quality must be in (0, 1]"
        
        # Portfolio constraints
        assert 0 < self.portfolio.long_quantile < 1, "long_quantile must be in (0, 1)"
        assert 0 < self.portfolio.short_quantile < 1, "short_quantile must be in (0, 1)"
        assert self.portfolio.long_leverage >= 0, "long_leverage must be non-negative"
        assert not (self.portfolio.long_only and self.portfolio.short_only), \
            "Cannot set both long_only and short_only to True"
        assert self.portfolio.cash_rate >= 0, "cash_rate must be non-negative"
        assert self.portfolio.commission_bps >= 0, "commission_bps must be non-negative"
        assert self.portfolio.slippage_bps >= 0, "slippage_bps must be non-negative"
        assert self.portfolio.borrow_cost >= 0, "borrow_cost must be non-negative"
        assert 0 < self.portfolio.margin <= 1, "margin must be in (0, 1]"
        
        # Feature config
        assert self.features.ic_threshold >= 0, "ic_threshold must be non-negative"
        assert self.features.max_features > 0, "max_features must be positive"
        assert self.features.bin_max_depth > 0, "bin_max_depth must be positive"
        
        return True


def get_default_config() -> ResearchConfig:
    """Get the default research configuration."""
    config = ResearchConfig.default()
    config.validate()
    return config


if __name__ == "__main__":
    # Example: create and validate config
    config = get_default_config()
    
    print("Configuration loaded successfully:")
    print(f"  Training window: {config.time.TRAINING_WINDOW_DAYS} days")
    print(f"  Holding period: {config.time.HOLDING_PERIOD_DAYS} days")
    print(f"  Feature max lag: {config.time.FEATURE_MAX_LAG_DAYS} days")
    print(f"  Base features: {len(config.features.base_features)}")
    print(f"  Binning candidates: {len(config.features.binning_candidates)}")
