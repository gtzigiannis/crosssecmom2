"""
Configuration Module
====================
Centralized configuration for the cross-sectional momentum research framework.
All paths, time parameters, and thresholds are defined here.
"""

from dataclasses import dataclass
from typing import List, Optional, Dict
from pathlib import Path


@dataclass
class PathConfig:
    """File paths for data inputs and outputs."""
    
    # Base data directory
    data_dir: str = r"D:\REPOSITORY\Data\crosssecmom2"
    
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
    start_date: str = "2017-11-04"  # Start date for yfinance download
    end_date: str = "2025-11-10"  # End date
    
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
        
    # === Margin Regime ===
    # Choose margin regime: "reg_t_initial", "reg_t_maintenance", or "portfolio"
    margin_regime: str = "reg_t_maintenance"  # Most realistic for held positions
    
    # Reg T Initial Margins (50%/50% - most conservative, for opening positions)
    reg_t_initial_long: float = 0.50
    reg_t_initial_short: float = 0.50
    
    # Reg T Maintenance Margins (25%/30% - realistic for held positions)
    reg_t_maint_long: float = 0.25   # 25% maintenance for long ETF positions
    reg_t_maint_short: float = 0.30  # 30% maintenance for short ETF positions
    
    # Portfolio Margin (risk-based - requires $100k account, lower margins)
    portfolio_margin_long: float = 0.15   # Typical for diversified ETF portfolio
    portfolio_margin_short: float = 0.15  # Typical for diversified ETF portfolio
    
    # === Dollar-Neutral Constraint ===
    enforce_dollar_neutral: bool = True   # True = long exposure = short exposure
    long_short_ratio: float = 1.0         # Only used if enforce_dollar_neutral=False
    
    # === DEPRECATED Margin Requirements (kept for backward compatibility) ===
    long_margin_req: float = 0.50   # DEPRECATED: Use margin_regime instead
    short_margin_req: float = 0.50  # DEPRECATED: Use margin_regime instead
    
    # Portfolio mode
    long_only: bool = False      # If True, only construct long positions (cash otherwise)
    short_only: bool = False     # If True, only construct short positions (cash otherwise)
    cash_rate: float = 0.045     # Annual interest rate on cash positions (default 4.5%)
    
    # Transaction costs (realistic estimates for ETF trading)
    # Note: "per side" means charged on EACH trade (buy, sell, short, cover)
    commission_bps: float = 3.0  # Commission in basis points per side (0-2 bps typical)
    slippage_bps: float = 5.0    # Slippage + market impact per side (5-10 bps for liquid ETFs)
    
    # === Financing Costs (NEW - separate rates for shorts vs longs) ===
    short_borrow_rate: float = 0.055      # Annual rate to borrow shares for shorting (5.5%)
                                          # Paid on FULL notional of short positions
    margin_interest_rate: float = 0.055   # Annual interest rate on cash borrowed for leverage (5.5%)
                                          # Paid on borrowed portion of long positions
    
    # DEPRECATED: Use short_borrow_rate and margin_interest_rate instead
    borrow_cost: float = 0.055    # DEPRECATED: Now split into separate rates above
    
    # Risk limits (default caps)
    # NOTE: These are BASE caps - portfolio optimization will adaptively relax them
    #       if needed to accommodate target leverage and actual universe size.
    #       See portfolio_construction.py adaptive cap logic.
    default_cluster_cap: float = 0.15  # 15% base cap per theme cluster
    default_per_etf_cap: float = 0.10  # 10% base cap per ETF
    
    @property
    def total_cost_bps_per_side(self) -> float:
        """Total transaction cost per side in basis points."""
        return self.commission_bps + self.slippage_bps
    
    # High-risk family caps (lower limits)
    high_risk_cluster_cap: float = 0.07  # 7% for risky themes
    high_risk_families: set = None
    
    # Backward compatibility alias (deprecated)
    @property
    def margin(self) -> float:
        """Deprecated: Use short_notional instead."""
        return self.short_notional
    
    def __post_init__(self):
        if self.high_risk_families is None:
            self.high_risk_families = {
                "EQ_EM_BROAD",
                "EQ_SINGLE_COUNTRY_EM",
                "COMMODITY_METALS",
                "COMMODITY_ENERGY",
                "ALT_CRYPTO"
            }
        
        # Emit deprecation warnings for old parameters
        import warnings
        if hasattr(self, '_warned_deprecation'):
            return  # Only warn once
        self._warned_deprecation = True
        
        # Check if user is relying on deprecated parameters (only if they exist)
        if (hasattr(self, 'long_leverage') and hasattr(self, 'short_notional') and
            (self.long_leverage != 1.0 or self.short_notional != 0.50)):
            warnings.warn(
                "\n" + "="*80 + "\n"
                "DEPRECATION WARNING: long_leverage and short_notional are deprecated.\n"
                "\n"
                "Please migrate to the new margin regime system:\n"
                "  - Use 'margin_regime' to select: 'reg_t_initial', 'reg_t_maintenance', or 'portfolio'\n"
                "  - Use 'enforce_dollar_neutral=True' for cross-sectional momentum (default)\n"
                "  - Separate financing costs: 'short_borrow_rate' and 'margin_interest_rate'\n"
                "\n"
                "Your current settings will continue to work but may produce different leverage.\n"
                "Recommended: Set margin_regime='reg_t_maintenance' for realistic ETF trading.\n"
                + "="*80,
                DeprecationWarning,
                stacklevel=3
            )
    
    def get_active_margins(self) -> tuple:
        """Return active margin requirements based on margin regime.
        
        Returns
        -------
        tuple of (float, float)
            (margin_long, margin_short) - margin requirements as fractions
        
        Examples
        --------
        >>> config.margin_regime = 'reg_t_maintenance'
        >>> margin_long, margin_short = config.get_active_margins()
        >>> # margin_long = 0.25, margin_short = 0.30
        """
        if self.margin_regime == "reg_t_initial":
            return self.reg_t_initial_long, self.reg_t_initial_short
        elif self.margin_regime == "reg_t_maintenance":
            return self.reg_t_maint_long, self.reg_t_maint_short
        elif self.margin_regime == "portfolio":
            return self.portfolio_margin_long, self.portfolio_margin_short
        else:
            # Fallback to deprecated parameters
            import warnings
            warnings.warn(
                f"Unknown margin regime '{self.margin_regime}'. "
                f"Falling back to long_margin_req={self.long_margin_req}, "
                f"short_margin_req={self.short_margin_req}",
                UserWarning
            )
            return self.long_margin_req, self.short_margin_req
    
    def compute_max_exposure(self, capital: float = 1.0) -> Dict[str, float]:
        """Calculate maximum leverage given margin constraints.
        
        Returns leverage ratios (dimensionless weights relative to equity) that are
        INDEPENDENT of the absolute capital value. These are pure multipliers.
        
        For dollar-neutral strategy (long = short):
            leverage = 1 / (margin_long + margin_short)
        
        Parameters
        ----------
        capital : float, default 1.0
            Reference capital for calculation (result is normalized by this)
        
        Returns
        -------
        dict with keys:
            - long_exposure: Leverage for long positions (dimensionless)
            - short_exposure: Leverage for short positions (dimensionless)
            - gross_exposure: Total leverage (long + short)
            - gross_leverage: Same as gross_exposure
            - net_exposure: Long - short (should be ~0 for dollar-neutral)
            - margin_long: Active long margin requirement
            - margin_short: Active short margin requirement
        
        Examples
        --------
        >>> # Reg T maintenance: 25% long, 30% short
        >>> config.margin_regime = 'reg_t_maintenance'
        >>> result = config.compute_max_exposure(capital=1.0)
        >>> result['long_exposure']   # 1.0 / (0.25 + 0.30) = 1.82 (leverage ratio)
        >>> result['gross_leverage']  # 3.64x (independent of capital value)
        """
        margin_long, margin_short = self.get_active_margins()
        
        if self.enforce_dollar_neutral:
            # Dollar-neutral: leverage = 1 / (margin_long + margin_short)
            # This is a pure ratio, independent of capital
            max_leverage = 1.0 / (margin_long + margin_short)
            long_exp = max_leverage
            short_exp = max_leverage
        else:
            # User-specified ratio (e.g., 130/30 fund)
            # Total capital constraint: margin_long * L + margin_short * S <= 1
            # With ratio constraint: L / S = long_short_ratio
            ratio = self.long_short_ratio
            long_exp = ratio / (margin_long * ratio + margin_short)
            short_exp = long_exp / ratio
        
        return {
            'long_exposure': long_exp,
            'short_exposure': short_exp,
            'gross_exposure': long_exp + short_exp,
            'gross_leverage': long_exp + short_exp,
            'net_exposure': long_exp - short_exp,
            'margin_long': margin_long,
            'margin_short': margin_short
        }


@dataclass
class DebugConfig:
    """Debugging and diagnostic configuration."""
    enable_accounting_debug: bool = False
    debug_max_periods: int = 0  # 0 = no limit, >0 = limit number of logged periods


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    
    # Base features to compute (these are computed WITHOUT using targets)
    # NOTE: If None, will auto-discover ALL features from panel data
    base_features: List[str] = None
    
    # Binning candidates (subset of base features for supervised binning)
    # NOTE: These features get binned to create additional '_Bin' features
    binning_candidates: List[str] = None
    
    # Feature selection
    ic_threshold: float = 0.02        # Minimum absolute IC for feature selection
    max_features: int = 20            # Maximum features to select
    
    # Binning parameters
    bin_max_depth: int = 3            # Max depth for decision tree binning
    bin_min_samples_leaf: int = 100   # Min samples per leaf
    n_bins: int = 8                  # Target number of bins
    
    # Reproducibility
    random_state: Optional[int] = 42  # Random seed for reproducible results
    
    def __post_init__(self):
        # NOTE: base_features will remain None initially
        # It will be dynamically populated from panel data in alpha_models.py
        # This ensures ALL generated features are automatically included as candidates
        
        if self.binning_candidates is None:
            # Expanded binning candidates to include diverse feature types
            # These will be binned to create supervised features
            self.binning_candidates = [
                # Returns at multiple horizons
                'Close%-21', 'Close%-63', 'Close%-126', 'Close%-252',
                
                # Momentum indicators
                'Close_Mom21', 'Close_Mom42', 'Close_Mom63',
                
                # Volatility measures
                'Close_std21', 'Close_std42', 'Close_std63', 'Close_std126',
                'Close_ATR14',
                
                # Oscillators
                'Close_RSI14', 'Close_RSI21', 'Close_RSI42',
                
                # MACD family
                'Close_MACD', 'Close_MACD_Histo',
                
                # Higher moments
                'Close_skew63', 'Close_kurt63',
                
                # NEW from crosssecmom: Drawdown features
                'Close_DD20', 'Close_DD60',
                
                # NEW from crosssecmom: Shock features
                'Close_Ret1dZ',
                
                # NEW from crosssecmom: Relative returns
                'Rel5_vs_VT', 'Rel20_vs_VT', 'Rel60_vs_VT',
                'Rel5_vs_Basket', 'Rel20_vs_Basket', 'Rel60_vs_Basket',
                
                # NEW from crosssecmom: Correlations
                'Corr20_VT', 'Corr20_BNDW',
                
                # NEW from crosssecmom: Macro features (continuous only, exclude binary flags)
                'vix_level', 'vix_z_1y', 'yc_slope', 'short_rate', 'credit_proxy_20',
            ]


@dataclass
class ComputeConfig:
    """Computational parameters."""
    
    n_jobs: int = -1                  # Parallel jobs for feature engineering (-1 = all cores)
    verbose: bool = True              # Print progress messages
    parallelize_backtest: bool = True   # Parallelize walk-forward backtest
    
    # Data download parameters
    batch_sleep: float = 1.0          # Sleep time (seconds) after every 20 downloads to avoid rate limiting
    
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
class DebugConfig:
    """Debugging and diagnostic configuration."""
    enable_accounting_debug: bool = False
    debug_max_periods: int = 0  # 0 = no limit, >0 = limit number of logged periods


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
    debug: DebugConfig
    
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
            regime=RegimeConfig(),
            debug=DebugConfig()
        )
    
    def validate(self):
        """Validate configuration parameters."""
        # Time parameters
        assert self.time.FEATURE_MAX_LAG_DAYS > 0, "FEATURE_MAX_LAG_DAYS must be positive"
        assert self.time.TRAINING_WINDOW_DAYS > self.time.FEATURE_MAX_LAG_DAYS, \
            "TRAINING_WINDOW_DAYS must be longer than FEATURE_MAX_LAG_DAYS"
        assert self.time.HOLDING_PERIOD_DAYS > 0, "HOLDING_PERIOD_DAYS must be positive"
        assert self.time.STEP_DAYS > 0, "STEP_DAYS must be positive"
        
        # Date range validation
        import pandas as pd
        start_date = pd.to_datetime(self.time.start_date)
        end_date = pd.to_datetime(self.time.end_date)
        date_range_days = (end_date - start_date).days
        
        # Minimum data required for strategy to run
        # = FEATURE_MAX_LAG (for feature calculation) 
        # + TRAINING_WINDOW (for model training)
        # + HOLDING_PERIOD (for forward returns)
        # + buffer for walk-forward steps
        min_required_days = self.time.FEATURE_MAX_LAG_DAYS + self.time.TRAINING_WINDOW_DAYS + self.time.HOLDING_PERIOD_DAYS + 100
        
        if date_range_days < min_required_days:
            import warnings
            warnings.warn(
                f"\n" + "="*80 + "\n"
                f"WARNING: Date range may be insufficient for strategy execution!\n"
                f"\n"
                f"Current date range: {self.time.start_date} to {self.time.end_date} ({date_range_days} days)\n"
                f"Minimum required: {min_required_days} trading days\n"
                f"\n"
                f"Required for:\n"
                f"  - Feature calculation (max lag): {self.time.FEATURE_MAX_LAG_DAYS} days\n"
                f"  - Model training window: {self.time.TRAINING_WINDOW_DAYS} days\n"
                f"  - Forward return horizon: {self.time.HOLDING_PERIOD_DAYS} days\n"
                f"  - Walk-forward buffer: 100 days\n"
                f"\n"
                f"Recommendation: Set start_date to at least {(end_date - pd.Timedelta(days=min_required_days)).strftime('%Y-%m-%d')}\n"
                f"                or earlier (2007-11-04 recommended for full ETF history)\n"
                + "="*80,
                UserWarning,
                stacklevel=2
            )
        
        # Universe filters
        assert 0 < self.universe.min_adv_percentile < 1, "min_adv_percentile must be in (0, 1)"
        assert 0 < self.universe.min_data_quality <= 1, "min_data_quality must be in (0, 1]"
        
        # Portfolio constraints
        assert 0 < self.portfolio.long_quantile < 1, "long_quantile must be in (0, 1)"
        assert 0 < self.portfolio.short_quantile < 1, "short_quantile must be in (0, 1)"
        # Only validate deprecated parameters if they exist
        if hasattr(self.portfolio, 'long_leverage'):
            assert self.portfolio.long_leverage >= 0, "long_leverage must be non-negative"
        if hasattr(self.portfolio, 'short_notional'):
            assert self.portfolio.short_notional >= 0, "short_notional must be non-negative"
        assert 0 < self.portfolio.long_margin_req <= 1, "long_margin_req must be in (0, 1]"
        assert 0 < self.portfolio.short_margin_req <= 1, "short_margin_req must be in (0, 1]"
        
        # Validate margin regime
        valid_regimes = {"reg_t_initial", "reg_t_maintenance", "portfolio"}
        if self.portfolio.margin_regime not in valid_regimes:
            import warnings
            warnings.warn(
                f"Unknown margin_regime '{self.portfolio.margin_regime}'. "
                f"Valid options: {valid_regimes}. Using fallback to long_margin_req/short_margin_req.",
                UserWarning
            )
        
        # Validate margin parameters
        assert 0 < self.portfolio.reg_t_initial_long <= 1, "reg_t_initial_long must be in (0, 1]"
        assert 0 < self.portfolio.reg_t_initial_short <= 1, "reg_t_initial_short must be in (0, 1]"
        assert 0 < self.portfolio.reg_t_maint_long <= 1, "reg_t_maint_long must be in (0, 1]"
        assert 0 < self.portfolio.reg_t_maint_short <= 1, "reg_t_maint_short must be in (0, 1]"
        assert 0 < self.portfolio.portfolio_margin_long <= 1, "portfolio_margin_long must be in (0, 1]"
        assert 0 < self.portfolio.portfolio_margin_short <= 1, "portfolio_margin_short must be in (0, 1]"
        
        assert not (self.portfolio.long_only and self.portfolio.short_only), \
            "Cannot set both long_only and short_only to True"
        assert self.portfolio.cash_rate >= 0, "cash_rate must be non-negative"
        assert self.portfolio.commission_bps >= 0, "commission_bps must be non-negative"
        assert self.portfolio.slippage_bps >= 0, "slippage_bps must be non-negative"
        
        # Validate financing costs
        assert self.portfolio.short_borrow_rate >= 0, "short_borrow_rate must be non-negative"
        assert self.portfolio.margin_interest_rate >= 0, "margin_interest_rate must be non-negative"
        assert self.portfolio.borrow_cost >= 0, "borrow_cost must be non-negative (deprecated)"
        
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
