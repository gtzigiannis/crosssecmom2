"""
Configuration dataclasses for Alpha and Risk feature selection pipelines.

Imports target configuration from the main crosssecmom2 config for consistency.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional

# Import from main config to ensure target column consistency
from strategies.crosssecmom2.config import TargetConfig as MainTargetConfig


# =============================================================================
# Alpha Pipeline Thresholds
# =============================================================================

@dataclass
class AlphaThresholds:
    """
    Thresholds for Alpha feature hard gates and scoring.
    
    Reference: feature_alpha.md
    """
    # Hard Gate 1: Data Quality
    min_coverage: float = 0.85           # ≥85% non-null values
    max_outlier_frac: float = 0.01       # <1% beyond 10σ
    outlier_sigma: float = 10.0          # Outlier threshold in std devs
    
    # Hard Gate 2: Global IC
    min_ic_abs: float = 0.02             # Mean |IC| ≥ 2%
    
    # Hard Gate 3: Statistical Significance
    min_t_stat: float = 2.5              # t-stat ≥ 2.5 (Newey-West)
    
    # Hard Gate 4: Sign Stability
    n_periods: int = 4                   # 4 consecutive segments
    min_sign_consistency: float = 0.75   # ≥3/4 segments match global sign
    
    # Hard Gate 5: Residual IC (after orthogonalizing vs Close%-63)
    min_residual_ic: float = 0.0075      # IC ≥ 0.75%
    residual_benchmark: str = 'Close%-63'
    
    # Hard Gate 6: Long-Tail Excess
    min_long_tail_excess: float = 0.0    # Top decile excess > 0%
    
    # Scoring Layer Weights
    score_weight_ic: float = 0.30        # Global IC: 30%
    score_weight_long_tail: float = 0.25 # Long-tail excess: 25%
    score_weight_spread_sharpe: float = 0.20  # Spread Sharpe: 20%
    score_weight_turnover: float = 0.15  # Turnover (inverted): 15%
    score_weight_stress_ic: float = 0.10 # Stress IC: 10%
    
    # Selection
    top_n_before_redundancy: int = 200   # Top N features before redundancy filter
    
    # Redundancy
    within_family_max_corr: float = 0.70 # ρ ≤ 0.70 within family
    cross_family_max_corr: float = 0.75  # ρ ≤ 0.75 across families
    max_per_family: int = 10             # Max features per family
    
    # Final Pool Checks
    final_max_pairwise_corr: float = 0.75
    final_mean_pairwise_corr: float = 0.35
    final_max_condition_number: float = 30.0
    
    # Forced Inclusions (always include regardless of scoring)
    forced_features: List[str] = field(default_factory=lambda: [
        'Close%-63', 'Close%-21', 'Close%-126'
    ])


@dataclass
class AlphaConfig:
    """
    Configuration for Alpha feature selection pipeline.
    """
    # Data
    data_dir: Path = field(default_factory=lambda: Path(r"D:\REPOSITORY\Data\crosssecmom2"))
    output_dir: Path = field(default_factory=lambda: Path(r"D:\REPOSITORY\Data\crosssecmom2\feature_lab_outputs\alpha"))
    
    # Target column for IC computation - imported from main config for consistency
    target_column: str = field(default_factory=lambda: MainTargetConfig().target_column)
    
    # Return column for spread/long-tail computation
    return_column: str = "FwdRet_21"
    
    # Stress period indicator (VT return < 0)
    benchmark_return_column: str = "FwdRet_21_VT"  # Will be computed if not present
    
    # Minimum samples
    min_samples_per_date: int = 20
    min_dates: int = 100
    
    # Thresholds
    thresholds: AlphaThresholds = field(default_factory=AlphaThresholds)
    
    # Parallelization
    n_jobs: int = -1
    random_seed: int = 42
    verbose: bool = True


# =============================================================================
# Risk Pipeline Thresholds
# =============================================================================

@dataclass
class RiskThresholds:
    """
    Thresholds for Risk feature selection.
    
    Reference: feature_risk.md
    """
    # Data Quality
    min_coverage: float = 0.85           # ≥85% non-null values
    max_outlier_frac: float = 0.01       # <1% beyond 10σ
    outlier_sigma: float = 10.0
    
    # Tail Event Definition
    tail_event_k_sigma: float = 2.0      # Tail event: return < -k × σ_lookback
    tail_lookback: int = 20              # Lookback for rolling σ in tail definition
    
    # Sign Stability Hard Gate
    # Features must have consistent sign relationship with risk targets across time
    min_global_corr: float = 0.05        # Minimum |global correlation| to be relevant
    sign_flip_noise_band: float = 0.05   # Segment correlations within this band are treated as noise
    n_stability_segments: int = 4        # Number of segments to split history into
    
    # Scoring Weights (naming matches what risk.py expects)
    weight_vol: float = 0.40             # Vol/variance relevance: 40%
    weight_tail: float = 0.40            # Tail discrimination: 40%
    weight_stab: float = 0.20            # Stability: 20%
    
    # Minimum RiskScore to pass
    min_risk_score: float = 0.0          # Min RiskScore threshold (0 = no min)
    
    # Stability
    n_periods: int = 4                   # 4 segments for stability
    min_sign_consistency: float = 0.75   # ≥3/4 segments
    
    # Redundancy
    within_family_max_corr: float = 0.80 # More lenient for risk
    cross_family_max_corr: float = 0.85  # Cross-family correlation limit
    max_per_family: int = 5              # Top K per risk family
    
    # Total cap
    max_total: int = 30                  # Max total risk features after selection


@dataclass
class RiskConfig:
    """
    Configuration for Risk feature selection pipeline.
    """
    # Data
    data_dir: Path = field(default_factory=lambda: Path(r"D:\REPOSITORY\Data\crosssecmom2"))
    output_dir: Path = field(default_factory=lambda: Path(r"D:\REPOSITORY\Data\crosssecmom2\feature_lab_outputs\risk"))
    
    # Risk Targets
    # RV_1d: next-day realized vol (|r_{t+1}| or sqrt(r_{t+1}^2))
    # RV_5d: 5-day forward realized vol 
    # TailEvent_1d: binary indicator for tail event
    rv_1d_column: str = "RV_1d"          # Will be computed if not present
    rv_5d_column: str = "RV_5d"          # Will be computed if not present
    tail_event_column: str = "TailEvent_1d"  # Will be computed if not present
    
    # Liquidity target
    liq_target_column: str = "ADV_63"    # Already in panel
    
    # For computing RV and tail events
    return_column: str = "FwdRet_1"      # Daily return for RV
    vol_lookback: int = 20               # 20-day rolling vol for tail threshold
    
    # Minimum samples
    min_samples_per_date: int = 20
    min_dates: int = 100
    
    # Thresholds
    thresholds: RiskThresholds = field(default_factory=RiskThresholds)
    
    # Parallelization
    n_jobs: int = -1
    random_seed: int = 42
    verbose: bool = True


# =============================================================================
# Risk Family Definitions
# =============================================================================

RISK_FAMILIES = {
    'volatility': {
        'patterns': ['std', 'ATR', 'BBW', 'parkinson', 'garman_klass', 'rogers_satchell', 
                     'realized_vol', 'RV_', 'vol_', 'Ret1dZ'],
        'target': 'rv',  # Score vs RV_1d, RV_5d
    },
    'beta': {
        'patterns': ['beta', 'corr_mkt', 'corr_VT', 'corr_BNDW', 'r_squared', 'downside_beta'],
        'target': 'rv',  # Betas relate to systematic risk
    },
    'liquidity': {
        'patterns': ['amihud', 'spread', 'kyle', 'illiq', 'roll_spread', 'ADV_', 'volume', 
                     'rel_vol', 'turnover'],
        'target': 'liq',  # Score vs LiqTarget
    },
    'regime': {
        'patterns': ['regime', 'vol_of_vol', 'Hurst', 'crash_flag', 'meltup_flag', 
                     'high_vol', 'low_vol', 'streak', 'days_since'],
        'target': 'rv',  # Regimes predict vol changes
    },
    'tail': {
        'patterns': ['max_dd', 'DD', 'var_', 'cvar', 'semi_vol', 'skew', 'kurt', 
                     'down_corr', 'drawdown_corr'],
        'target': 'tail',  # Score vs TailEvent
    },
}


def get_risk_family(feature_name: str) -> Optional[str]:
    """
    Classify a risk feature into its risk family.
    
    Returns family name or None if not a recognized risk pattern.
    """
    feature_lower = feature_name.lower()
    
    for family, config in RISK_FAMILIES.items():
        for pattern in config['patterns']:
            if pattern.lower() in feature_lower:
                return family
    return None
