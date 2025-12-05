"""
Feature Lab: Alpha and Risk Feature Selection Pipelines
========================================================

Two separate pipelines for feature selection:
1. Alpha features → Ensemble scoring model (return prediction)
2. Risk features → Risk models (vol/covariance, regime, liquidity)

Usage:
    from feature_lab import AlphaSelector, RiskSelector, AlphaConfig, RiskConfig
    
    # Alpha feature selection
    alpha_config = AlphaConfig()
    alpha_selector = AlphaSelector(panel_df, alpha_config)
    alpha_features = alpha_selector.run_pipeline()
    
    # Risk feature selection  
    risk_config = RiskConfig()
    risk_selector = RiskSelector(panel_df, risk_config)
    risk_features = risk_selector.run_pipeline()
    
    # One-liner convenience functions
    from feature_lab import select_alpha_features, select_risk_features
    alpha_features = select_alpha_features(panel_df)
    risk_features = select_risk_features(panel_df)
"""

from .config import (
    AlphaConfig, 
    RiskConfig, 
    AlphaThresholds, 
    RiskThresholds,
    RISK_FAMILIES,
    get_risk_family,
)
from .base import (
    DataQualityGate,
    RedundancyFilter,
    classify_feature_role,
    get_alpha_features,
    get_risk_features,
    partition_features,
    RISK_PATTERNS,
    HAS_NUMBA,
)
from .alpha import AlphaSelector, AlphaFeatureResult
from .risk import RiskSelector, RiskFeatureResult, RiskTargetBuilder


# =============================================================================
# Convenience Functions
# =============================================================================

def select_alpha_features(panel_df, feature_cols=None, verbose=True, n_jobs=-1):
    """
    One-liner to select alpha features from a panel.
    
    Args:
        panel_df: Panel DataFrame with (Date, Ticker) MultiIndex
        feature_cols: Feature columns to consider (default: auto-detect)
        verbose: Print progress
        n_jobs: Parallel jobs (-1 = all cores)
        
    Returns:
        List of selected alpha feature column names
    """
    config = AlphaConfig(verbose=verbose, n_jobs=n_jobs)
    selector = AlphaSelector(panel_df, config)
    return selector.run_pipeline(feature_cols)


def select_risk_features(panel_df, feature_cols=None, verbose=True, n_jobs=-1):
    """
    One-liner to select risk features from a panel.
    
    Args:
        panel_df: Panel DataFrame with (Date, Ticker) MultiIndex
        feature_cols: Feature columns to consider (default: auto-detect)
        verbose: Print progress
        n_jobs: Parallel jobs (-1 = all cores)
        
    Returns:
        List of selected risk feature column names
    """
    config = RiskConfig(verbose=verbose, n_jobs=n_jobs)
    selector = RiskSelector(panel_df, config)
    return selector.run_pipeline(feature_cols)


def run_dual_pipeline(panel_df, feature_cols=None, verbose=True, n_jobs=-1):
    """
    Run both alpha and risk pipelines.
    
    Returns:
        Tuple (alpha_features, risk_features, alpha_selector, risk_selector)
    """
    alpha_config = AlphaConfig(verbose=verbose, n_jobs=n_jobs)
    risk_config = RiskConfig(verbose=verbose, n_jobs=n_jobs)
    
    alpha_selector = AlphaSelector(panel_df, alpha_config)
    risk_selector = RiskSelector(panel_df, risk_config)
    
    alpha_features = alpha_selector.run_pipeline(feature_cols)
    risk_features = risk_selector.run_pipeline(feature_cols)
    
    return alpha_features, risk_features, alpha_selector, risk_selector


__all__ = [
    # Config
    'AlphaConfig',
    'RiskConfig', 
    'AlphaThresholds',
    'RiskThresholds',
    'RISK_FAMILIES',
    'get_risk_family',
    # Base utilities
    'DataQualityGate',
    'RedundancyFilter',
    'classify_feature_role',
    'get_alpha_features',
    'get_risk_features',
    'partition_features',
    'RISK_PATTERNS',
    'HAS_NUMBA',
    # Selectors
    'AlphaSelector',
    'RiskSelector',
    'AlphaFeatureResult',
    'RiskFeatureResult',
    'RiskTargetBuilder',
    # Convenience
    'select_alpha_features',
    'select_risk_features',
    'run_dual_pipeline',
]
