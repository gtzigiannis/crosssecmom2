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
    TRAINING_WINDOW_DAYS: int = 253  # Legacy fallback (V2). V3 uses training_years instead.
    HOLDING_PERIOD_DAYS: int = 21     # Forward return horizon (1 month)
    STEP_DAYS: int = 21               # Rebalancing frequency (1 month)


@dataclass
class UniverseConfig:
    """Universe filtering and eligibility parameters."""
    
    # Liquidity filters
    min_adv_percentile: float = 0.50  # Minimum ADV percentile (50th percentile - original strict filter)
    adv_window: int = 63              # Window for ADV calculation
    
    # Data quality
    min_data_quality: float = 0.90    # Minimum fraction of non-NaN features (original strict filter)
    # NOTE: min_history_days removed - history requirement is determined by
    # FEATURE_MAX_LAG_DAYS + TRAINING_WINDOW_DAYS in walk_forward_engine
    
    # Metadata filters
    exclude_leveraged: bool = True    # Exclude leveraged/inverse ETFs
    exclude_non_canonical: bool = True  # Exclude non-canonical duplicates
    
    # PHASE 0: Asset class filter to test equity-only vs cross-asset
    equity_only: bool = False  # DISABLED: family metadata is UNKNOWN - enable when metadata is populated
    equity_family_keywords: list = None  # Keywords to identify equity families
    
    def __post_init__(self):
        """Set default equity family keywords if not provided."""
        if self.equity_family_keywords is None:
            # Families that represent equity/stock exposure
            # Includes: US stocks, international stocks, sectors, regions, country funds
            # Excludes: pure bonds, digital assets (bonds, crypto)
            self.equity_family_keywords = [
                'Stock', 'Equity', 'Blend', 'Growth', 'Value',  # Core equity styles
                'Large', 'Mid', 'Small',  # Size categories (when combined with Blend/Growth/Value)
                'Country -', 'Region', 'Europe', 'Japan', 'India', 'Latin America',  # Geographic
                'Emerging Mkts', 'Foreign',  # International equity
                'Real Estate',  # REITs (equity-like)
                'Technology', 'Health', 'Financial', 'Industrials', 'Utilities',  # Sectors
                'Energy', 'Consumer', 'Communication',  # More sectors
                'Natural Resources',  # Includes XLB (Materials) and XME (Metals/Mining) - sector ETFs
                'Commodities Focused'  # Includes USO (oil), OIH (oil services) - commodity equity exposure
            ]
    
    # Duplicate detection
    dup_corr_threshold: float = 0.99  # Correlation threshold for duplicates
    
    # Clustering
    max_within_cluster_corr: float = 0.85  # Max correlation within theme clusters


@dataclass
class PortfolioConfig:
    """Portfolio construction parameters."""
    
    # Position sizing
    # PHASE 0 FIX: Widen from 25% to 33% to get more positions with small equity universe
    long_quantile: float = 0.67   # Top 33% for long portfolio (above 67th percentile)
    short_quantile: float = 0.33  # Bottom 33% for short portfolio (below 33rd percentile)
        
    # === Margin Regime ===
    # Choose margin regime: "reg_t_initial", "reg_t_maintenance", or "portfolio"
    # PHASE 0: Use reg_t_initial (most conservative) to test at 1.0x gross leverage
    margin_regime: str = "reg_t_initial"  # Conservative for Phase 0 diagnostic
    
    # Reg T Initial Margins (50%/50% - most conservative, for opening positions)
    reg_t_initial_long: float = 0.50
    reg_t_initial_short: float = 0.50
    
    # Reg T Maintenance Margins (25%/30% - realistic for held positions)
    reg_t_maint_long: float = 0.25   # 25% maintenance for long ETF positions
    reg_t_maint_short: float = 0.30  # 30% maintenance for short ETF positions
    
    # Portfolio Margin (risk-based - requires $100k account, lower margins)
    portfolio_margin_long: float = 0.15   # Typical for diversified ETF portfolio
    portfolio_margin_short: float = 0.15  # Typical for diversified ETF portfolio
    
    # === Margin Utilization ===
    # Fraction of available margin capacity to use
    # 1.0 = fully margin-saturated (max gross under regime)
    # With use_leverage=False: 1.0 means 50% long + 50% short = 100% gross (dollar-neutral)
    max_margin_utilization: float = 1.0  # Full utilization: 50% long + 50% short = 100% gross
    
    # === Leverage Toggle ===
    # Simple on/off switch for leverage:
    # - If False (no leverage): margin = 1.0 for both longs and shorts
    #   -> Each $1 of long exposure requires $1 of capital
    #   -> Each $1 of short exposure requires $1 of capital (from short sale proceeds)
    #   -> Dollar-neutral with 0.5x long, 0.5x short = 1.0x gross
    # - If True (use leverage): use configured margin_regime
    #   -> With reg_t_initial (50%/50%): 2.0x gross (1.0x each side)
    #   -> With reg_t_maintenance (25%/30%): 3.64x gross (1.82x each side)
    #   -> With portfolio margin (15%/15%): 6.67x gross (3.33x each side)
    use_leverage: bool = False  # Default False - no leverage
    
    # === Dollar-Neutral Constraint ===
    enforce_dollar_neutral: bool = True   # True = long exposure = short exposure
    long_short_ratio: float = 1.0         # Only used if enforce_dollar_neutral=False
    
    # === DEPRECATED Margin Requirements (kept for backward compatibility) ===
    long_margin_req: float = 0.50   # DEPRECATED: Use margin_regime instead
    short_margin_req: float = 0.50  # DEPRECATED: Use margin_regime instead
    
    # Portfolio mode
    long_only: bool = False      # If True, only construct long positions (cash otherwise)
    short_only: bool = False     # If True, only construct short positions (cash otherwise)
    cash_rate: float = 0.034     # Annual interest rate on cash positions (3.4% - IBKR Pro rate)
    
    # Transaction costs (realistic estimates for ETF trading)
    # Note: "per side" means charged on EACH trade (buy, sell, short, cover)
    commission_bps: float = 3.0  # Commission in basis points per side (0-2 bps typical)
    slippage_bps: float = 5.0    # Slippage + market impact per side (5-10 bps for liquid ETFs)
    
    # === Financing Costs (realistic rates for liquid ETF universe at IBKR) ===
    short_borrow_rate: float = 0.01       # Annual rate to borrow shares for shorting (1.0%)
                                          # Paid on FULL notional of short positions
                                          # Note: General collateral rate for liquid ETFs (not hard-to-borrow)
    margin_interest_rate: float = 0.05    # Annual interest rate on cash borrowed for leverage (5.0%)
                                          # Paid on borrowed portion of long positions
    
    # DEPRECATED: Use short_borrow_rate and margin_interest_rate instead
    borrow_cost: float = 0.055    # DEPRECATED: Now split into separate rates above
    
    # === Diagnostic Mode ===
    # If True, force zero financing costs for diagnostic runs (cash_rate still applies)
    # Useful for isolating signal quality from financing drag
    zero_financing_mode: bool = False
    
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
        """Return active margin requirements based on leverage setting and margin regime.
        
        If use_leverage=False: Returns (1.0, 1.0) - no leverage for either side.
        If use_leverage=True: Returns margins based on configured margin_regime.
        
        This ensures symmetric treatment of longs and shorts:
        - No leverage: $1 capital for each $1 of exposure (both sides)
        - With leverage: Use margin regime (same margin % for dollar-neutral)
        
        Returns
        -------
        tuple of (float, float)
            (margin_long, margin_short) - margin requirements as fractions
        
        Examples
        --------
        >>> config.use_leverage = False
        >>> margin_long, margin_short = config.get_active_margins()
        >>> # margin_long = 1.0, margin_short = 1.0 -> 1.0x gross
        
        >>> config.use_leverage = True
        >>> config.margin_regime = 'reg_t_initial'
        >>> margin_long, margin_short = config.get_active_margins()
        >>> # margin_long = 0.5, margin_short = 0.5 -> 2.0x gross
        """
        # If leverage is disabled, use 100% margin for both sides
        if not self.use_leverage:
            return 1.0, 1.0
        
        # Otherwise use configured margin regime
        if self.margin_regime == "reg_t_initial":
            margin_long = self.reg_t_initial_long
            margin_short = self.reg_t_initial_short
        elif self.margin_regime == "reg_t_maintenance":
            margin_long = self.reg_t_maint_long
            margin_short = self.reg_t_maint_short
        elif self.margin_regime == "portfolio":
            margin_long = self.portfolio_margin_long
            margin_short = self.portfolio_margin_short
        else:
            # Fallback to deprecated parameters
            import warnings
            warnings.warn(
                f"Unknown margin regime '{self.margin_regime}'. "
                f"Falling back to long_margin_req={self.long_margin_req}, "
                f"short_margin_req={self.short_margin_req}",
                UserWarning
            )
            margin_long = self.long_margin_req
            margin_short = self.short_margin_req
        
        return margin_long, margin_short
    
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
        
        # Special case: long-only or short-only mode
        if self.long_only:
            # Long-only: max_long = 1 / margin_long
            base_long = 1.0 / margin_long
            long_exp = base_long * self.max_margin_utilization
            short_exp = 0.0
        elif self.short_only:
            # Short-only: max_short = 1 / margin_short
            base_short = 1.0 / margin_short
            long_exp = 0.0
            short_exp = base_short * self.max_margin_utilization
        elif self.enforce_dollar_neutral:
            # Dollar-neutral: leverage = 1 / (margin_long + margin_short)
            # This is a pure ratio, independent of capital
            base_leverage = 1.0 / (margin_long + margin_short)
            
            # Scale by max_margin_utilization so we don't always use 100% of margin
            # This leaves (1 - max_margin_utilization) in cash earning cash_rate
            scaled_leverage = base_leverage * self.max_margin_utilization
            
            long_exp = scaled_leverage
            short_exp = scaled_leverage
        else:
            # User-specified ratio (e.g., 130/30 fund)
            # Total capital constraint: margin_long * L + margin_short * S <= 1
            # With ratio constraint: L / S = long_short_ratio
            ratio = self.long_short_ratio
            base_long = ratio / (margin_long * ratio + margin_short)
            base_short = base_long / ratio
            
            # Scale by max_margin_utilization
            long_exp = base_long * self.max_margin_utilization
            short_exp = base_short * self.max_margin_utilization
        
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
    enable_accounting_debug: bool = True  # Enable for verification
    debug_max_periods: int = 0  # 0 = no limit, >0 = limit number of logged periods


@dataclass
class FeatureConfig:
    """Feature engineering parameters."""
    
    # Base features to compute (these are computed WITHOUT using targets)
    # NOTE: If None, will auto-discover ALL features from panel data
    base_features: List[str] = None
    
    # Winsorization control
    enable_winsorization: bool = False  # If True, winsorize forward returns at ±n_sigma to reduce outlier noise
    winsorization_n_sigma: float = 2.5  # Number of standard deviations for winsorization bounds
    
    # === NaN HANDLING CONFIGURATION ===
    # NaN handling is done at the SOURCE (feature_engineering.py) to ensure clean data
    # downstream. Feature selection receives NaN-free data and will FAIL if NaN exists.
    
    # Raw data NaN handling (before feature calculation)
    raw_data_min_valid_pct: float = 0.80  # Minimum fraction of valid (non-NaN) observations for a ticker
    raw_data_ffill_limit: int = 5         # Max consecutive days to forward-fill (avoids look-ahead)
    
    # Feature NaN threshold (features with > threshold NaN are dropped)
    feature_nan_threshold: float = 0.10   # Drop features with >10% NaN (configurable)
    
    # Post-feature imputation (for remaining "innocent" NaN after rolling windows)
    # Cross-sectional median imputation (per-date) avoids forward leakage
    enable_nan_imputation: bool = True    # If True, impute remaining NaN with cross-sectional median
    
    # === V3 PIPELINE PARAMETERS ===
    
    # Formation/Training window configuration
    formation_years: float = 4.0        # Formation window length in years (4 years)
    training_years: float = 1.0         # Training window length in years (1 year)
    
    # Time-decay weighting configuration
    # When OFF (default), all time points in a window receive equal weight.
    # This is preferred for demonstrating cross-temporal feature persistence.
    use_time_decay_weights: bool = False      # OFF by default - equal weight for all time points
    formation_halflife_days: int = 252        # Formation IC weighting half-life (~12 months) - only used if enabled
    training_halflife_days: int = 63          # Training IC weighting half-life (~3 months) - only used if enabled
    
    # Formation FDR parameters
    formation_fdr_q_threshold: float = 0.10   # FDR control level (10% default, generous)
    formation_ic_floor: float = 0.02          # Mild |IC| floor for formation
    formation_use_mi: bool = False            # Toggle for MI gating (OFF by default, for future use)
    
    # Per-window ranking parameters
    per_window_ic_floor: float = 0.02         # |IC| floor for per-window ranking (filters weak features)
    per_window_top_k: int = 100               # Top K features to keep after univariate ranking
    per_window_min_features: int = 50         # Minimum features to keep (K_min)
    per_window_num_blocks: int = 3            # Number of contiguous blocks for sign consistency (B=3-5)
    
    # Redundancy/collinearity parameters
    corr_threshold: float = 0.80              # Maximum correlation for redundancy filter (relaxed from 0.70)
    
    # LassoLarsIC / LassoCV parameters (final feature selection in Training phase)
    lars_min_features: int = 12               # Minimum features to select (floor)
    lars_max_features: int = 25               # Maximum features to select (cap for sparsity)
    lars_criterion: str = 'bic'               # Information criterion ('bic' more conservative than 'aic')
    lars_use_cv: bool = True                  # Use LassoCV instead of LassoLarsIC (True = cross-validate)
    lars_cv_folds: int = 5                    # Number of CV folds for LassoCV
    ridge_refit_alpha: float = 0.01           # Ridge alpha for final refit on LARS support
    
    # === SHORT-LAG FEATURE PROTECTION (pipeline integration) ===
    # Protects short-lag (<5 day) features through the feature selection pipeline
    # to ensure the model retains exposure to short-term momentum signals.
    # 
    # Pipeline flow:
    #   Formation FDR (protect top 30) → Soft Ranking (protect top 15) 
    #       → Redundancy Filter (unchanged) → Check: if <5 survive, supplement from ranking
    #       → LassoCV (5 short-lag guaranteed in input)
    #
    enable_short_lag_protection: bool = True  # Enable short-lag feature protection through pipeline
    short_lag_max_horizon: int = 5            # Features with horizon < this are "short-lag" (days)
    short_lag_protect_fdr: int = 30           # Protect top N short-lag features through Formation FDR
    short_lag_protect_ranking: int = 15       # Protect top N short-lag features through Soft Ranking
    short_lag_min_for_lasso: int = 5          # Minimum short-lag features to deliver to LassoCV
    short_lag_min_ic: float = 0.01            # Minimum |IC| threshold (1%) for short-lag inclusion
    
    # === DEPRECATED SHORT-LAG PARAMETERS (kept for backward compatibility) ===
    # These were for the old "superimpose at end" approach - now integrated into pipeline
    enable_short_lag_inclusion: bool = False  # DEPRECATED: Use enable_short_lag_protection instead
    short_lag_bucket_1_max: int = 5           # DEPRECATED
    short_lag_bucket_1_top_k: int = 5         # DEPRECATED
    short_lag_bucket_2_min: int = 5           # DEPRECATED
    short_lag_bucket_2_max: int = 21          # DEPRECATED
    short_lag_bucket_2_top_k: int = 5         # DEPRECATED
    short_lag_min_fdr_pval: float = 0.20      # DEPRECATED
    
    # === INTERACTION FEATURE SCREENING PARAMETERS ===
    # Interactions go through a separate, stricter pre-filter before joining base features
    interaction_fdr_level: float = 0.05       # Stricter FDR for interactions (5% vs 10% for base)
    interaction_ic_floor: float = 0.03        # Higher IC floor for interactions (3% vs 2% for base)
    interaction_stability_folds: int = 5      # Number of time-series folds for stability check
    interaction_min_ic_agreement: float = 0.60  # Min fraction of folds with consistent IC sign (60%)
    interaction_max_features: int = 150       # Hard cap on approved interactions (prevents explosion)
    interaction_corr_vs_base: float = 0.75    # Max correlation with any base feature (orthogonality)
    enable_interaction_screening: bool = True # If False, interactions skip pre-filter (legacy mode)
    
    # === DIRECTION CONSISTENCY PARAMETERS (Stage A) ===
    # Features must show consistent IC sign across time blocks, not just statistical significance.
    # This is a key filter for finding PERSISTENT signals.
    direction_consistency_min: float = 0.60       # Min fraction of blocks with same sign as global mean (base features)
    direction_consistency_min_int: float = 0.80   # Stricter threshold for interactions
    ic_min_base: float = 0.02                     # Min absolute mean IC for base features
    ic_min_int: float = 0.03                      # Min absolute mean IC for interactions (stricter)
    alpha_base: float = 0.10                      # FDR threshold for base features
    alpha_int: float = 0.05                       # FDR threshold for interactions (stricter)
    alpha_parent_loose: float = 0.50              # FDR threshold for "clearly bad" parent detection
    ic_min_parent_loose: float = 0.01             # IC threshold for "clearly bad" parent detection
    
    # === BLOCK-LEVEL PERSISTENCE PARAMETERS (Stage A enhancement) ===
    # In addition to pooled FDR + direction consistency, features must also show
    # non-trivial IC in at least m_base blocks with correct sign.
    # This is a SECOND filter applied on top of the existing pooled FDR filter.
    # NOTE: m_base=2 is less aggressive than m_base=3, allowing features that may
    # be weaker in one period but strong overall to pass.
    block_persistence_enabled: bool = True        # Enable block-level persistence filter
    block_persistence_m_base: int = 2             # Min blocks passing (2 of 4 = 50%, less aggressive than 3 of 4)
    block_persistence_ic_min: float = 0.01        # Minimum |IC_block| for a block to count (1%)
    block_persistence_t_min: float = 1.5          # Minimum |t_block| for a block to count (alternative to IC)
    
    # === K-FOLD LASSO CV PARAMETERS (Stage B) ===
    # Features must be selected consistently across K time-series folds to survive.
    # This replaces the single LassoLarsIC fit with more robust cross-validation.
    # 
    # BLOCK-WISE STABILITY: Each block is an independent sample for selection.
    # With use_block_only=False (default), use leave-one-out CV where each fold trains
    # on K-1 folds (overlapping). This provides more training data per fold.
    # With use_block_only=True, fit only on each block's data (non-overlapping) which
    # is more stringent but may be underpowered with small blocks.
    use_kfold_lasso_cv: bool = True               # Use K-fold selection (True) vs single LassoCV (False)
    kfold_n_splits: int = 5                       # Number of time-series CV folds (K)
    kfold_selection_threshold: float = 0.60       # Min fraction of folds where feature selected (π_threshold)
    kfold_max_features: int = 50                  # Max features after frequency filter (before min/max bounds)
    kfold_use_block_only: bool = False            # False = leave-one-out (more data), True = block-only (stricter)
    
    # === PER-WINDOW IC RE-RANKING CONTROL ===
    # Once Stage A + redundancy + Stage B are done, treat S_final = S_stable as fixed.
    # If skip_per_window_ic_ranking=True, do NOT re-rank/filter by per-window IC.
    # If skip_per_window_ic_ranking=False (default), re-rank features by recent IC
    # to adapt to current market conditions. This provides adaptivity.
    skip_per_window_ic_ranking: bool = False      # False = re-rank by recent IC (adaptive), True = use fixed set
    
    # === BUCKET-AWARE REDUNDANCY FILTER PARAMETERS ===
    # Redundancy filter respects (family, horizon) buckets to preserve diversity
    min_features_per_bucket: int = 5          # Minimum features to keep per (family, horizon) bucket
    redundancy_corr_within_bucket: float = 0.80   # Correlation threshold within same bucket (stricter)
    redundancy_corr_cross_bucket: float = 0.90    # Correlation threshold across buckets (more lenient)
    
    # === DEPRECATED V2 PARAMETERS (kept for backward compatibility) ===
    
    # Binning control (DEPRECATED - not used in v3 pipeline)
    use_manual_binning_candidates: bool = False  # DEPRECATED: If True, use hardcoded binning_candidates
    binning_candidates: List[str] = None         # DEPRECATED: Manual list of features to bin
    bin_max_depth: int = 3                       # DEPRECATED: Max depth for decision tree binning
    bin_min_samples_leaf: int = 100              # DEPRECATED: Min samples per leaf
    n_bins: int = 8                              # DEPRECATED: Target number of bins
    
    # Feature selection (DEPRECATED - use v3 parameters instead)
    ic_threshold: float = 0.02        # DEPRECATED: Use per_window_ic_floor instead
    max_features: int = 20            # DEPRECATED: Use per_window_top_k instead
    
    # Reproducibility
    random_state: Optional[int] = 42  # Random seed for reproducible results
    
    # === DEPRECATED V3 PARAMETERS (kept for backward compatibility) ===
    # ElasticNet parameters (DEPRECATED - replaced by LassoLarsIC in v4)
    alpha_grid: List[float] = None            # DEPRECATED: Alpha values for ElasticNetCV
    l1_ratio_grid: List[float] = None         # DEPRECATED: L1 ratio values
    elasticnet_cv_folds: int = 3              # DEPRECATED: Time-series CV folds
    elasticnet_coef_threshold: float = 1e-4   # DEPRECATED: Minimum |coefficient|
    
    def __post_init__(self):
        # NOTE: base_features will remain None initially
        # It will be dynamically populated from panel data in alpha_models.py
        # This ensures ALL generated features are automatically included as candidates
        
        # Initialize deprecated ElasticNet grids (for backward compatibility)
        if self.alpha_grid is None:
            self.alpha_grid = [0.0001, 0.001, 0.01, 0.1]
        
        if self.l1_ratio_grid is None:
            self.l1_ratio_grid = [0.2, 0.5, 0.9]
        
        # Only set default binning_candidates if manual mode is enabled and list is None
        # (DEPRECATED: kept for backward compatibility only)
        if self.use_manual_binning_candidates and self.binning_candidates is None:
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
                
                # Drawdown features (standardized windows)
                'Close_DD21', 'Close_DD63',
                
                # Shock features
                'Close_Ret1dZ',
                
                # Relative returns (standardized windows)
                'Rel5_vs_VT', 'Rel21_vs_VT', 'Rel63_vs_VT',
                'Rel5_vs_Basket', 'Rel21_vs_Basket', 'Rel63_vs_Basket',
                
                # Correlations (standardized windows)
                'Corr21_VT', 'Corr21_BNDW',
                
                # Macro features (continuous only, exclude binary flags)
                'vix_level', 'vix_z_1y', 'yc_slope', 'short_rate', 'credit_proxy_21',
                
                # NEW V4: Enhanced macro features
                'yc_slope_10_3mo', 'baa_aaa_spread', 'indpro_yoy', 'cpi_yoy', 
                
                # NEW V4: Risk factor features
                'beta_VT_63', 'beta_BNDW_63', 'downside_beta_VT_63', 'idio_vol_63',
                'corr_VIX_63', 'corr_MOVE_63',
            ]


@dataclass
class TargetConfig:
    """Target label engineering parameters.
    
    Controls how the prediction target (forward returns) is processed:
    - Raw forward returns (y_raw_Hd)
    - Cross-sectionally demeaned returns (y_cs_Hd) 
    - Risk-adjusted residual z-scores (y_resid_z_Hd)
    
    The target used by models is controlled by `target_column`.
    """
    
    # === CORE TARGET PARAMETERS ===
    target_horizon_days: int = 21  # Forward return horizon (must match TimeConfig.HOLDING_PERIOD_DAYS)
    
    # Column names for different target variants
    target_column_raw: str = "y_raw_21d"       # Raw H-day forward return
    target_column_cs: str = "y_cs_21d"         # Cross-sectionally demeaned
    target_column_resid: str = "y_resid_21d"   # Risk-adjusted residual (before z-score)
    target_column_resid_z: str = "y_resid_z_21d"  # Risk-adjusted residual z-score (preferred)
    
    # The target column actually used by walk-forward engine and models
    # Options: "y_raw_21d", "y_cs_21d", "y_resid_z_21d"
    target_column: str = "y_resid_z_21d"
    
    # === CROSS-SECTIONAL DEMEANING ===
    # Controls how assets are grouped for demeaning
    target_demean_mode: str = "global"  # Options: "global", "by_asset_class", "by_sector"
    # If using grouped demeaning, which column to group by (e.g., "asset_class", "sector")
    target_demean_group_col: Optional[str] = None
    
    # === RISK ADJUSTMENT ===
    # Enable regression-based residualization against risk controls
    target_use_risk_adjustment: bool = True
    
    # Columns to use as risk controls in regression (empty = skip regression)
    # These should be pre-computed features in the panel
    # Common controls: beta, volatility, liquidity, sector dummies
    target_risk_control_columns: List[str] = None
    
    # Minimum samples per group for regression (if fewer, fallback to y_cs)
    target_min_regression_samples: int = 20
    
    # Ridge regularization for risk regression (small alpha for stability)
    target_risk_regression_alpha: float = 0.01
    
    # === WINSORIZATION AND Z-SCORING ===
    # Toggle for winsorization of residuals before z-scoring
    # If False, only z-scoring is applied (no quantile clipping)
    target_enable_winsorization: bool = False  # Default False - winsorization disabled
    
    # Quantile limits for per-date winsorization of residuals (only used if target_enable_winsorization=True)
    target_winsorization_limits: tuple = (0.01, 0.99)  # 1% and 99%
    
    # Minimum standard deviation for z-scoring (avoid divide-by-zero)
    target_min_std_for_zscore: float = 1e-6
    
    def __post_init__(self):
        """Set defaults for risk control columns."""
        if self.target_risk_control_columns is None:
            # Default risk controls using actual feature names from feature_engineering.py:
            # - beta_VT_63: Rolling 63-day beta to VT (market exposure)
            # - idio_vol_63: Idiosyncratic volatility after regressing on VT
            # These are computed by add_risk_factor_features_to_panel()
            # 
            # Note: ADV_63 is excluded because it's used for universe filtering
            # and could cause issues with the regression if it has different coverage
            self.target_risk_control_columns = ['beta_VT_63', 'idio_vol_63']


@dataclass
class ComputeConfig:
    """Computational parameters."""
    
    n_jobs: int = -1                  # Parallel jobs for feature engineering (-1 = all cores)
    verbose: bool = True              # Print progress messages
    parallelize_backtest: bool = True  # Parallel execution of backtest periods
    
    # Data download parameters
    batch_sleep: float = 1.0          # Sleep time (seconds) after every 20 downloads to avoid rate limiting
    
    # Persistence
    save_intermediate: bool = True    # Save intermediate objects
    ic_output_path: Optional[str] = None  # Path for IC vectors (defaults to plots_dir/ic_vectors.csv)
    
    # Debug/profiling mode: limit number of rebalance dates for single-window testing
    # Set to 1 for profiling a single window, None for full backtest
    max_rebalance_dates_for_debug: Optional[int] = None
    
    # Skip early rebalance dates (useful when early formation windows have NaN data)
    skip_rebalance_dates: int = 0


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
    enable_accounting_debug: bool = True  # Enable for verification
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
    target: TargetConfig
    
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
            debug=DebugConfig(),
            target=TargetConfig()
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
        assert self.features.formation_fdr_q_threshold > 0, "formation_fdr_q_threshold must be positive"
        assert self.features.formation_ic_floor >= 0, "formation_ic_floor must be non-negative"
        assert self.features.per_window_ic_floor >= 0, "per_window_ic_floor must be non-negative"
        assert self.features.per_window_top_k > 0, "per_window_top_k must be positive"
        assert self.features.per_window_min_features > 0, "per_window_min_features must be positive"
        assert 3 <= self.features.per_window_num_blocks <= 5, "per_window_num_blocks must be in [3, 5]"
        assert self.features.formation_years > 0, "formation_years must be positive"
        assert self.features.training_years > 0, "training_years must be positive"
        assert self.features.formation_halflife_days > 0, "formation_halflife_days must be positive"
        assert self.features.training_halflife_days > 0, "training_halflife_days must be positive"
        assert 0 < self.features.corr_threshold < 1, "corr_threshold must be in (0, 1)"
        
        # Deprecated parameters (warn but don't fail)
        if self.features.use_manual_binning_candidates:
            import warnings
            warnings.warn(
                "use_manual_binning_candidates is DEPRECATED in v3 pipeline. "
                "Binning is not used in production code. "
                "This parameter is kept only for backward compatibility.",
                DeprecationWarning,
                stacklevel=2
            )
        
        # Target config validation
        valid_demean_modes = {"global", "by_asset_class", "by_sector", "none"}
        assert self.target.target_demean_mode in valid_demean_modes, \
            f"target_demean_mode must be one of {valid_demean_modes}"
        assert self.target.target_horizon_days > 0, "target_horizon_days must be positive"
        assert 0 <= self.target.target_winsorization_limits[0] < self.target.target_winsorization_limits[1] <= 1, \
            "target_winsorization_limits must be (lower, upper) with 0 <= lower < upper <= 1"
        
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
    print(f"  Base features: {len(config.features.base_features) if config.features.base_features else 'auto-discovered'}")
    print(f"\nV3 Pipeline Configuration:")
    print(f"  Formation window: {config.features.formation_years} years")
    print(f"  Training window: {config.features.training_years} years")
    print(f"  Formation half-life: {config.features.formation_halflife_days} days")
    print(f"  Training half-life: {config.features.training_halflife_days} days")
    print(f"  Formation FDR q-threshold: {config.features.formation_fdr_q_threshold}")
    print(f"  Per-window top K: {config.features.per_window_top_k}")
    print(f"\nTarget Engineering Configuration:")
    print(f"  Target horizon: {config.target.target_horizon_days} days")
    print(f"  Active target column: {config.target.target_column}")
    print(f"  Demean mode: {config.target.target_demean_mode}")
    print(f"  Risk adjustment: {config.target.target_use_risk_adjustment}")
    print(f"  Winsorization: {config.target.target_winsorization_limits}")

