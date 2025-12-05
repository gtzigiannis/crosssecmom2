"""
Walk-Forward Engine (Refactored)
=================================
Model-agnostic walk-forward backtesting engine with discreet-windows time structure.

Key design principles:
1. Uses explicit FEATURE_MAX_LAG_DAYS, TRAINING_WINDOW_DAYS, HOLDING_PERIOD_DAYS, STEP_DAYS
2. Model-agnostic: works with any AlphaModel via train/score interface
3. Enforces universe filters: ADV, data quality, duplicates
4. Enforces portfolio caps: per-ETF and per-cluster
5. Zero look-ahead bias: training window ends before t0
6. Regime-aware portfolio construction (optional)

Time structure at each rebalance date t0:
  t_train_start = t0 - TRAINING_WINDOW_DAYS
  t_train_end = t0 - 1 - HOLDING_PERIOD_DAYS  # Gap to avoid overlap
  Training: [t_train_start, t_train_end]
  Holding: [t0, t0 + HOLDING_PERIOD_DAYS)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path
import warnings

from config import ResearchConfig
from alpha_models import AlphaModel
# Use V4 feature selection with performance optimizations
from feature_selection_v4 import (
    train_window_model, 
    formation_fdr, 
    compute_time_decay_weights, 
    correlation_redundancy_filter, 
    formation_interaction_screening, 
    bucket_aware_redundancy_filter,
    precompute_all_daily_ics,
    get_formation_cache
)
from portfolio_construction import construct_portfolio, evaluate_portfolio_return
from regime import compute_regime_series, get_portfolio_mode_for_regime
from attribution_analysis import compute_attribution_analysis, save_attribution_results


def compute_formation_artifacts(
    panel_df: pd.DataFrame,
    universe_metadata: pd.DataFrame,
    t0: pd.Timestamp,
    config: ResearchConfig,
    verbose: bool = False
) -> Dict:
    """
    Compute Formation artifacts for v3 pipeline.
    
    Formation window is a longer lookback period (default 5 years) used to:
    1. Approve features via formation_fdr (IC + FDR filter)
    2. Tune ElasticNet hyperparameters (alpha, l1_ratio)
    
    These artifacts are then used by the Training window (default 1 year)
    to fit the final model with soft ranking and no CV.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel with (Date, Ticker) MultiIndex
    universe_metadata : pd.DataFrame
        ETF metadata
    t0 : pd.Timestamp
        Current rebalance date
    config : ResearchConfig
        Configuration with formation_years, training_years, etc.
    verbose : bool
        Print diagnostic information
        
    Returns
    -------
    dict or None
        Formation artifacts with keys:
        - 'approved_features': List of features that passed FDR + redundancy filter
        - 'formation_diagnostics': Dict with IC values, FDR stats, etc.
        Returns None if Formation window is invalid or computation fails.
        
    Note: V4 pipeline removes ElasticNet tuning from Formation.
    LassoLarsIC in Training selects lambda automatically via BIC.
    """
    # Compute Formation window
    formation_days = int(config.features.formation_years * 252)  # Business days
    t_formation_start = t0 - pd.Timedelta(days=formation_days)
    t_formation_end = t0 - pd.Timedelta(days=1 + config.time.HOLDING_PERIOD_DAYS)
    
    if verbose:
        print(f"[formation] Window: [{t_formation_start.date()}, {t_formation_end.date()}] ({formation_days} days)", flush=True)
    
    # Check Formation window is valid
    if t_formation_end <= t_formation_start:
        if verbose:
            print("[warn] Invalid Formation window")
        return None
    
    # Extract Formation panel
    formation_mask = (
        (panel_df.index.get_level_values('Date') >= t_formation_start) &
        (panel_df.index.get_level_values('Date') <= t_formation_end)
    )
    panel_formation = panel_df[formation_mask]
    
    if len(panel_formation) == 0:
        if verbose:
            print("[warn] Empty Formation panel")
        return None
    
    # Prepare data for formation_fdr
    # Use configured target column from config (SINGLE SOURCE OF TRUTH)
    target_col = config.target.target_column
    raw_target_col = f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}'
    
    # Validate target column exists - DO NOT silently fall back
    if target_col not in panel_formation.columns:
        available_targets = [c for c in panel_formation.columns if c.startswith('y_') or c.startswith('FwdRet')]
        raise ValueError(
            f"[CRITICAL] Configured target column '{target_col}' not found in panel!\\n"
            f"Available target-like columns: {available_targets}\\n"
            f"This suggests a mismatch between config and data."
        )
    
    # Log target column being used (CRITICAL for verification)
    if verbose:
        print(f"[formation] *** USING TARGET: {target_col} ***")
        print(f"[formation] Feature selection will be performed against: {target_col}")
    
    # Identify feature columns (exclude non-feature columns and all target variants)
    target_columns = [raw_target_col, 'y_raw_21d', 'y_cs_21d', 'y_resid_21d', 'y_resid_z_21d']
    exclude_cols = ['Close', 'Ticker', 'ADV_63', 'ADV_63_Rank', 'market_cap'] + \
                   [c for c in panel_formation.columns if c.startswith('FwdRet')] + \
                   [c for c in target_columns if c in panel_formation.columns]
    feature_cols = [c for c in panel_formation.columns if c not in exclude_cols and not c.startswith('_')]
    
    if len(feature_cols) == 0:
        if verbose:
            print("[warn] No feature columns found in Formation panel")
        return None
    
    # =========================================================================
    # SPLIT FEATURES: Primitive Base vs All Interactions
    # 
    # Primitive base features: No combination patterns (true raw signals) ~138
    # ALL interactions: Any combination pattern (_x_, _div_, _minus_, _sq, _cb, _in_)
    # 
    # ALL interactions (multiplicative + derived) go through screening together.
    # Top 150 from screening join primitive base, then FDR + redundancy.
    # =========================================================================
    COMBINATION_PATTERNS = ['_x_', '_div_', '_minus_', '_sq', '_cb', '_in_']
    
    # Primitive base: no combination patterns at all (~138 features)
    primitive_base_cols = [c for c in feature_cols if not any(p in c for p in COMBINATION_PATTERNS)]
    
    # Multiplicative interactions: _x_ pattern
    multiplicative_cols = [c for c in feature_cols if '_x_' in c]
    
    # Derived interactions: _div_, _minus_, _sq, _cb, _in_ (but NOT _x_)
    derived_interaction_cols = [c for c in feature_cols 
                                if '_x_' not in c 
                                and any(p in c for p in ['_div_', '_minus_', '_sq', '_cb', '_in_'])]
    
    # ALL interactions for screening = multiplicative + derived
    all_interaction_cols = multiplicative_cols + derived_interaction_cols
    
    if verbose:
        print(f"[formation] Feature split:")
        print(f"[formation]   Primitive base: {len(primitive_base_cols)}")
        print(f"[formation]   Derived interactions (div/sq/cb): {len(derived_interaction_cols)}")
        print(f"[formation]   Multiplicative interactions (_x_): {len(multiplicative_cols)}")
        print(f"[formation]   Total interactions for screening: {len(all_interaction_cols)}")
    
    # Get formation dates
    dates_formation = panel_formation.index.get_level_values('Date')
    
    # Prepare X (primitive base only for orthogonality check) and y (target)
    X_primitive_base = panel_formation[primitive_base_cols].astype(np.float32)
    y_formation = panel_formation[target_col].astype(np.float64)
    
    import time as time_module
    
    # Determine n_jobs: force sequential execution in profiling mode
    is_profiling_mode = (config.compute.max_rebalance_dates_for_debug is not None)
    effective_n_jobs = 1 if is_profiling_mode else config.compute.n_jobs
    if is_profiling_mode and verbose:
        print(f"[formation] PROFILING MODE: n_jobs forced to 1 for accurate timing")
    
    # =========================================================================
    # INTERACTION SCREENING (if enabled and interactions exist)
    # ALL interactions (multiplicative + derived) are screened together.
    # Top 150 approved interactions join primitive base for FDR.
    # =========================================================================
    approved_interactions = []
    interaction_diagnostics = None
    
    if (len(all_interaction_cols) > 0 and 
        getattr(config.features, 'enable_interaction_screening', True)):
        
        interaction_start = time_module.time()
        try:
            X_all_interactions = panel_formation[all_interaction_cols].astype(np.float32)
            
            approved_interactions, interaction_diagnostics = formation_interaction_screening(
                X_base=X_primitive_base,  # Orthogonality check vs primitive base only
                X_interaction=X_all_interactions,  # ALL interactions screened together
                y=y_formation,
                dates=dates_formation,
                target_column=target_col,  # For cache lookup
                fdr_level=getattr(config.features, 'interaction_fdr_level', 0.05),
                ic_floor=getattr(config.features, 'interaction_ic_floor', 0.03),
                stability_folds=getattr(config.features, 'interaction_stability_folds', 5),
                min_ic_agreement=getattr(config.features, 'interaction_min_ic_agreement', 0.60),
                max_features=getattr(config.features, 'interaction_max_features', 150),
                corr_vs_base_threshold=getattr(config.features, 'interaction_corr_vs_base', 0.75),
                half_life=config.features.formation_halflife_days,
                n_jobs=effective_n_jobs,
                use_time_decay=config.features.use_time_decay_weights
            )
            
            interaction_elapsed = time_module.time() - interaction_start
            if verbose:
                print(f"[formation] Interaction screening: {len(approved_interactions)} approved out of "
                      f"{len(all_interaction_cols)} in {interaction_elapsed:.2f}s", flush=True)
                
        except Exception as e:
            if verbose:
                print(f"[warn] Interaction screening failed: {e}, proceeding without interactions")
                import traceback
                traceback.print_exc()
            approved_interactions = []
    elif len(all_interaction_cols) > 0 and verbose:
        print(f"[formation] Interaction screening DISABLED - all {len(all_interaction_cols)} interactions will join primitive base")
        approved_interactions = all_interaction_cols  # Legacy mode: no pre-filtering
    
    # =========================================================================
    # MERGE: Primitive Base + Approved Interactions for Formation FDR
    # =========================================================================
    all_feature_cols = primitive_base_cols + approved_interactions
    X_formation = panel_formation[all_feature_cols].astype(np.float32)
    
    if verbose:
        print(f"[formation] Combined feature pool: {len(all_feature_cols)} features")
    
    # Run formation_fdr to approve features
    fdr_start = time_module.time()
    
    try:
        # Formation phase runs BEFORE walk-forward
        approved_features, fdr_diagnostics = formation_fdr(
            X=X_formation,
            y=y_formation,
            dates=dates_formation,
            half_life=config.features.formation_halflife_days,
            fdr_level=config.features.formation_fdr_q_threshold,
            n_jobs=effective_n_jobs,
            use_time_decay=config.features.use_time_decay_weights
        )
        fdr_elapsed = time_module.time() - fdr_start
        
        if verbose:
            print(f"[formation] FDR approved {len(approved_features)} features in {fdr_elapsed:.2f}s", flush=True)
        
        if len(approved_features) == 0:
            if verbose:
                print("[warn] No features approved by formation_fdr")
            return None
            
    except Exception as e:
        if verbose:
            print(f"[error] formation_fdr failed: {e}")
            import traceback
            traceback.print_exc()
        return None
    
    # Run bucket-aware redundancy filter on FDR-approved features
    # This preserves diversity across (family, horizon) buckets
    redundancy_start = time_module.time()
    try:
        # Filter X to only approved features for redundancy check
        X_fdr_approved = X_formation[[f for f in approved_features if f in all_feature_cols]]
        
        # Get IC scores from FDR diagnostics for tie-breaking
        ic_scores = None
        if fdr_diagnostics is not None and 'feature' in fdr_diagnostics.columns:
            ic_scores = dict(zip(fdr_diagnostics['feature'], fdr_diagnostics['ic_weighted'].abs()))
        
        features_after_redundancy, redundancy_diagnostics = bucket_aware_redundancy_filter(
            X=X_fdr_approved,
            corr_within_bucket=getattr(config.features, 'redundancy_corr_within_bucket', 0.80),
            corr_cross_bucket=getattr(config.features, 'redundancy_corr_cross_bucket', 0.90),
            min_per_bucket=getattr(config.features, 'min_features_per_bucket', 5),
            ic_scores=ic_scores,
            n_jobs=effective_n_jobs
        )
        redundancy_elapsed = time_module.time() - redundancy_start
        
        if verbose:
            print(f"[formation] Bucket redundancy filter: {len(approved_features)} -> {len(features_after_redundancy)} features in {redundancy_elapsed:.2f}s", flush=True)
        
        # Update approved_features to only keep non-redundant ones
        approved_features = features_after_redundancy
        
        if len(approved_features) == 0:
            if verbose:
                print("[warn] No features left after redundancy filter")
            return None
            
    except Exception as e:
        if verbose:
            print(f"[error] bucket_aware_redundancy_filter failed: {e}")
            import traceback
            traceback.print_exc()
        # Fall back to using FDR-approved features without redundancy filter
        redundancy_elapsed = time_module.time() - redundancy_start
        redundancy_diagnostics = {'error': str(e)}
    
    # V4 Pipeline: No ElasticNet tuning in Formation
    # LassoLarsIC in Training phase automatically selects lambda via BIC criterion
    # Formation now only does: FDR -> Redundancy filter -> output approved_features
    
    # Get FDR-approved features before redundancy (for bucket tracking)
    # fdr_diagnostics is a DataFrame with 'feature' and 'fdr_reject' columns
    if isinstance(fdr_diagnostics, pd.DataFrame) and 'feature' in fdr_diagnostics.columns:
        features_after_fdr = fdr_diagnostics[fdr_diagnostics['fdr_reject']]['feature'].tolist()
    else:
        # Fallback - shouldn't happen but be safe
        features_after_fdr = list(approved_features)
    
    # Package artifacts
    formation_artifacts = {
        'approved_features': approved_features,
        'formation_diagnostics': {
            'fdr_diagnostics': fdr_diagnostics,
            'redundancy_diagnostics': redundancy_diagnostics,
            'n_approved': len(approved_features),
            'n_total_features': len(feature_cols),
            't_formation_start': t_formation_start,
            't_formation_end': t_formation_end,
            'time_fdr': fdr_elapsed,
            'time_redundancy': redundancy_elapsed,
            # Feature breakdown
            'n_primitive_base': len(primitive_base_cols),
            'n_multiplicative': len(multiplicative_cols),
            'n_derived_interactions': len(derived_interaction_cols),
            'n_total_interactions': len(all_interaction_cols),
            'n_approved_interactions': len(approved_interactions),
            'interaction_diagnostics': interaction_diagnostics,
        },
        # Feature lists for per-bucket breakdown
        'feature_lists': {
            'primitive_base': primitive_base_cols,
            'multiplicative_interactions': multiplicative_cols,
            'derived_interactions': derived_interaction_cols,
            'all_interactions': all_interaction_cols,
            'approved_interactions': approved_interactions,
            'combined_pool': all_feature_cols,
            'after_fdr': features_after_fdr,
            'after_bucket_redundancy': approved_features,
        }
    }
    
    # Print formation summary
    if verbose:
        total_formation_time = fdr_elapsed + redundancy_elapsed
        print(f"[formation] " + "-" * 30, flush=True)
        print(f"[formation] SUMMARY:", flush=True)
        print(f"[formation]   Primitive base: {len(primitive_base_cols)}", flush=True)
        print(f"[formation]   All interactions: {len(all_interaction_cols)} (mult={len(multiplicative_cols)}, derived={len(derived_interaction_cols)})", flush=True)
        print(f"[formation]   Screened interactions: {len(all_interaction_cols)} -> {len(approved_interactions)} approved", flush=True)
        print(f"[formation]   Combined pool: {len(all_feature_cols)} features (primitive + approved)", flush=True)
        print(f"[formation]   After FDR + Redundancy: {len(approved_features)} features", flush=True)
        print(f"[formation]   Time: Total={total_formation_time:.1f}s", flush=True)
        print(f"[formation] " + "-" * 30, flush=True)
    
    return formation_artifacts


def apply_universe_filters(
    cross_section: pd.DataFrame,
    universe_metadata: pd.DataFrame,
    config: ResearchConfig
) -> pd.DataFrame:
    """
    Apply universe eligibility filters at a point in time.
    
    Filters:
    1. in_core_after_duplicates == True (removes non-canonical duplicates)
    2. ADV_63_Rank >= min_adv_percentile (liquidity filter)
    3. Data quality >= min_data_quality (fraction of non-NaN features)
    
    Parameters
    ----------
    cross_section : pd.DataFrame
        Cross-section at one date (ticker in index)
    universe_metadata : pd.DataFrame
        ETF metadata (ticker in index or column)
    config : ResearchConfig
        Configuration
        
    Returns
    -------
    pd.DataFrame
        Filtered cross-section
    """
    # Start with all tickers
    tickers = cross_section.index
    
    # 1. Core universe filter (removes leveraged, non-canonical duplicates)
    if 'ticker' in universe_metadata.columns:
        metadata_idx = universe_metadata.set_index('ticker')
    else:
        metadata_idx = universe_metadata
    
    if 'in_core_after_duplicates' in metadata_idx.columns:
        core_tickers = metadata_idx[metadata_idx['in_core_after_duplicates'] == True].index
        tickers = tickers.intersection(core_tickers)
    
    # PHASE 0: Equity-only filter (test if cross-asset momentum adds value)
    if config.universe.equity_only and 'family' in metadata_idx.columns:
        # Filter to equity families using keyword matching
        equity_keywords = config.universe.equity_family_keywords
        if equity_keywords is None:
            # Fallback if __post_init__ didn't run (shouldn't happen with dataclass)
            equity_keywords = ['Stock', 'Equity', 'Blend', 'Growth', 'Value', 'Real Estate']
        
        def is_equity_family(family_name):
            """Check if family name contains any equity keyword."""
            if pd.isna(family_name):
                return False
            family_str = str(family_name)
            return any(keyword.lower() in family_str.lower() for keyword in equity_keywords)
        
        equity_mask = metadata_idx['family'].apply(is_equity_family)
        equity_tickers = metadata_idx[equity_mask].index
        
        n_before = len(tickers)
        tickers = tickers.intersection(equity_tickers)
        n_after = len(tickers)
        
        if n_before > n_after:
            # Only print once per run (first time filter is applied)
            import sys
            if not hasattr(sys, '_equity_filter_printed'):
                print(f"[universe] PHASE 0: Equity-only filter enabled, reduced universe {n_before} -> {n_after} tickers")
                sys._equity_filter_printed = True
    
    # 2. ADV liquidity filter
    if 'ADV_63_Rank' in cross_section.columns:
        adv_filter = cross_section['ADV_63_Rank'] >= config.universe.min_adv_percentile
        tickers = tickers[adv_filter[tickers]]
    
    # 3. Data quality filter
    # Exclude all target columns (raw FwdRet and computed y_* targets)
    target_cols_exclude = ['y_raw_21d', 'y_cs_21d', 'y_resid_21d', 'y_resid_z_21d']
    feature_cols = [c for c in cross_section.columns 
                   if c not in ['Close', 'Ticker', 'ADV_63', 'ADV_63_Rank'] 
                   and not c.startswith('FwdRet')
                   and c not in target_cols_exclude]
    
    if len(feature_cols) > 0:
        data_quality = cross_section.loc[tickers, feature_cols].notna().mean(axis=1)
        quality_filter = data_quality >= config.universe.min_data_quality
        tickers = tickers[quality_filter]
    
    return cross_section.loc[tickers]


def check_history_requirement(
    panel_df: pd.DataFrame,
    ticker: str,
    t0: pd.Timestamp,
    min_history_days: int
) -> bool:
    """
    Check if ticker has sufficient history before t0.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel with (Date, Ticker) index
    ticker : str
        Ticker to check
    t0 : pd.Timestamp
        Current date
    min_history_days : int
        Minimum required history
        
    Returns
    -------
    bool
        True if sufficient history exists
    """
    try:
        ticker_data = panel_df.xs(ticker, level='Ticker')
        ticker_dates = ticker_data.index
        
        # Find earliest date with non-NaN Close
        valid_dates = ticker_dates[ticker_data['Close'].notna()]
        
        if len(valid_dates) == 0:
            return False
        
        earliest_date = valid_dates[0]
        days_available = (t0 - earliest_date).days
        
        return days_available >= min_history_days
        
    except KeyError:
        return False


def get_eligible_universe(
    panel_df: pd.DataFrame,
    universe_metadata: pd.DataFrame,
    t0: pd.Timestamp,
    config: ResearchConfig,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Get eligible universe at date t0 after all filters.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel with (Date, Ticker) index
    universe_metadata : pd.DataFrame
        ETF metadata
    t0 : pd.Timestamp
        Current date
    config : ResearchConfig
        Configuration
    verbose : bool, default False
        Whether to print diagnostic information
        
    Returns
    -------
    pd.DataFrame
        Filtered cross-section at t0
    """
    # Get cross-section
    try:
        cross_section = panel_df.loc[t0].copy()
    except KeyError:
        return pd.DataFrame()
    
    # Apply standard filters
    filtered = apply_universe_filters(cross_section, universe_metadata, config)
    
    # Check history requirement
    # WARNING: This is a STRICT requirement. By t0, each ticker must have
    # at least FEATURE_MAX_LAG_DAYS + TRAINING_WINDOW_DAYS of historical data.
    # This ensures we have enough data for both feature calculation and model training.
    # 
    # Future enhancement: Could implement a more flexible approach that allows
    # tickers to enter the universe as soon as they meet minimum requirements,
    # rather than requiring full history from the start.
    required_history = config.time.FEATURE_MAX_LAG_DAYS + config.time.TRAINING_WINDOW_DAYS
    
    if verbose:
        print(f"[universe] History requirement: {required_history} days (FEATURE_MAX_LAG + TRAINING_WINDOW)")
    
    sufficient_history = []
    for ticker in filtered.index:
        if check_history_requirement(panel_df, ticker, t0, required_history):
            sufficient_history.append(ticker)
    
    return filtered.loc[sufficient_history]


def run_walk_forward_backtest(
    panel_df: pd.DataFrame,
    universe_metadata: pd.DataFrame,
    config: ResearchConfig,
    model_type: str = 'supervised_binned',
    portfolio_method: str = 'cvxpy',  # Default to cvxpy optimization
    verbose: bool = True
) -> pd.DataFrame:
    """
    Execute walk-forward backtest with proper time structure.
    
    At each rebalance date t0:
    1. Define training window: [t_train_start, t_train_end]
       where t_train_start = t0 - TRAINING_WINDOW_DAYS
             t_train_end = t0 - 1 - HOLDING_PERIOD_DAYS
    
    2. Train model on training window with feature selection:
       model, diagnostics = train_window_model(panel_train, metadata, t0, config)
    
    3. Get eligible universe at t0
    
    4. Score eligible tickers:
       scores = model.score_at_date(panel, t0, metadata, config)
    
    5. Construct portfolio with caps:
       long_wts, short_wts = construct_portfolio(scores, metadata, config)
    
    6. Evaluate using forward returns:
       realized_ret = evaluate_portfolio_return(panel, t0, long_wts, short_wts, config)
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel with (Date, Ticker) MultiIndex
    universe_metadata : pd.DataFrame
        ETF metadata with families, clusters, caps
    config : ResearchConfig
        Configuration object
    model_type : str
        'momentum_rank' or 'supervised_binned'
    portfolio_method : str
        'cvxpy' or 'simple'
    verbose : bool
        Print progress
        
    Returns
    -------
    pd.DataFrame
        Backtest results with columns: date, ls_return, long_ret, short_ret, etc.
    """
    if verbose:
        print("="*80)
        print("WALK-FORWARD BACKTEST")
        print("="*80)
        print(f"Model type: {model_type}")
        print(f"Portfolio method: {portfolio_method}")
        print(f"Training window: {config.time.TRAINING_WINDOW_DAYS} days")
        print(f"Holding period: {config.time.HOLDING_PERIOD_DAYS} days")
        print(f"Step size: {config.time.STEP_DAYS} days")
        print(f"Feature max lag: {config.time.FEATURE_MAX_LAG_DAYS} days")
        print(f"Transaction costs: {config.portfolio.total_cost_bps_per_side:.1f} bps per side")
        print(f"Short borrow rate: {config.portfolio.short_borrow_rate:.2%} annualized")
        print(f"Margin interest rate: {config.portfolio.margin_interest_rate:.2%} annualized")
        print(f"Margin regime: {config.portfolio.margin_regime}")
        print(f"Random state: {config.features.random_state}")
    
    # Validate portfolio method
    if portfolio_method not in ['cvxpy', 'simple']:
        raise ValueError(f"portfolio_method must be 'cvxpy' or 'simple', got '{portfolio_method}'")
    
    # Get unique dates
    dates = panel_df.index.get_level_values('Date').unique().sort_values()
    
    # Minimum date to start (need sufficient history)
    min_history_required = config.time.FEATURE_MAX_LAG_DAYS + config.time.TRAINING_WINDOW_DAYS
    min_start_date = dates[0] + pd.Timedelta(days=min_history_required)
    
    # Filter dates
    rebalance_dates = dates[dates >= min_start_date]
    
    # Step through dates
    current_dates = rebalance_dates[::config.time.STEP_DAYS]
    
    # Skip early dates if configured (useful when early formation windows have NaN data)
    if hasattr(config.compute, 'skip_rebalance_dates') and config.compute.skip_rebalance_dates > 0:
        n_skip = config.compute.skip_rebalance_dates
        current_dates = current_dates[n_skip:]
        if verbose:
            print(f"[SKIP] Skipping first {n_skip} rebalance date(s)")
    
    # Apply debug limit if configured
    if config.compute.max_rebalance_dates_for_debug is not None:
        n_limit = config.compute.max_rebalance_dates_for_debug
        current_dates = current_dates[:n_limit]
        if verbose:
            print(f"\n{'='*80}")
            print(f"[DEBUG MODE] Limited to {n_limit} rebalance date(s) for profiling")
            print(f"{'='*80}")
    
    if verbose:
        print(f"\nRebalance dates: {len(current_dates)}")
        print(f"Date range: {current_dates[0].date()} to {current_dates[-1].date()}")
    
    # =====================================================================
    # Compute regime series (if enabled)
    # =====================================================================
    regime_series = None
    if config.regime.use_regime:
        if verbose:
            print(f"\n[regime] Computing regime series...")
            print(f"  Market ticker: {config.regime.market_ticker}")
            print(f"  MA window: {config.regime.ma_window} days")
            print(f"  Return lookback: {config.regime.lookback_return_days} days")
        
        try:
            regime_series = compute_regime_series(panel_df, config.regime, verbose=verbose)
            if verbose:
                print(f"[regime] Computed {len(regime_series)} regime values")
                regime_counts = regime_series.value_counts()
                print(f"[regime] Bull: {regime_counts.get('bull', 0)}, Bear: {regime_counts.get('bear', 0)}, Range: {regime_counts.get('range', 0)}")
        except Exception as e:
            if verbose:
                print(f"[warn] Regime computation failed: {e}")
                print(f"[warn] Proceeding without regime switching")
            regime_series = None
    
    # =====================================================================
    # PRE-COMPUTE all daily ICs for the entire backtest (ONCE)
    # This eliminates redundant IC computation across walk-forward windows
    # =====================================================================
    if verbose:
        print(f"\n[PRE-COMPUTE] Starting one-time daily IC computation...")
    
    # Get target column
    target_col = config.target.target_column
    raw_target_col = f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}'
    if target_col not in panel_df.columns:
        target_col = raw_target_col
    
    # Identify ALL feature columns (same logic as compute_formation_artifacts)
    target_columns = [raw_target_col, 'y_raw_21d', 'y_cs_21d', 'y_resid_21d', 'y_resid_z_21d']
    exclude_cols = ['Close', 'Ticker', 'ADV_63', 'ADV_63_Rank', 'market_cap'] + \
                   [c for c in panel_df.columns if c.startswith('FwdRet')] + \
                   [c for c in target_columns if c in panel_df.columns]
    all_feature_cols = [c for c in panel_df.columns if c not in exclude_cols and not c.startswith('_')]
    
    if len(all_feature_cols) > 0 and target_col in panel_df.columns:
        try:
            precompute_all_daily_ics(
                panel_df=panel_df,
                target_column=target_col,
                feature_columns=all_feature_cols,
                n_jobs=config.compute.n_jobs,
                verbose=verbose
            )
        except Exception as e:
            if verbose:
                print(f"[warn] Pre-computation failed: {e}, falling back to per-window computation")
                import traceback
                traceback.print_exc()
    
    # =====================================================================
    # Parallel vs Sequential Execution
    # =====================================================================
    if config.compute.parallelize_backtest:
        # SEQUENTIAL OUTER LOOP with PARALLEL INNER feature selection
        from joblib import Parallel, delayed
        
        if verbose:
            print(f"\n[backtest] Processing {len(current_dates)} rebalance dates...", flush=True)
            print(f"[backtest] Outer loop: sequential (n_jobs=1)", flush=True)
            print(f"[backtest] Inner feature selection: parallel (n_jobs={config.compute.n_jobs})", flush=True)
        
        def process_rebalance_period(i, t0, panel_df, universe_metadata, config, model_type, portfolio_method, regime_series, verbose_inner):
            """Process a single rebalance period (for parallel execution)."""
            import time
            import pandas as pd
            import numpy as np
            
            rebalance_start = time.time()
            if verbose_inner:
                print(f"\n{'='*60}", flush=True)
                print(f"[{i+1}/{len(current_dates)}] Rebalance date: {t0.date()}", flush=True)
            
            # Compute Formation artifacts (v3 pipeline)
            formation_artifacts = None
            
            if hasattr(config.features, 'formation_years') and config.features.formation_years > 0:
                # V3 PIPELINE: Use Formation/Training split
                formation_artifacts = compute_formation_artifacts(
                    panel_df=panel_df,
                    universe_metadata=universe_metadata,
                    t0=t0,
                    config=config,
                    verbose=verbose_inner  # Use verbose_inner to control output
                )
                
                if formation_artifacts is None:
                    if verbose_inner:
                        print("[skip] Formation artifacts computation failed")
                    return None
            
            # Define Training window
            if formation_artifacts is not None:
                # V3: Training window is shorter (1 year by default)
                training_days = int(config.features.training_years * 252)
                t_train_start = t0 - pd.Timedelta(days=training_days)
                t_train_end = t0 - pd.Timedelta(days=1 + config.time.HOLDING_PERIOD_DAYS)
            else:
                # V2: Use old TRAINING_WINDOW_DAYS config
                t_train_start = t0 - pd.Timedelta(days=config.time.TRAINING_WINDOW_DAYS)
                t_train_end = t0 - pd.Timedelta(days=1 + config.time.HOLDING_PERIOD_DAYS)
            
            if verbose_inner:
                print(f"Training window: [{t_train_start.date()}, {t_train_end.date()}]")
            
            if t_train_end <= t_train_start:
                if verbose_inner:
                    print("[skip] Invalid training window")
                return None
            
            # Train model using train_window_model
            if verbose_inner:
                pipeline_version = "v3" if formation_artifacts else "v2"
                print(f"[train] Training {model_type} model ({pipeline_version} pipeline)...")
            
            train_start = time.time()
            try:
                # Extract training panel
                train_mask = (
                    (panel_df.index.get_level_values('Date') >= t_train_start) &
                    (panel_df.index.get_level_values('Date') <= t_train_end)
                )
                panel_train = panel_df[train_mask]
                
                # Train model with feature selection (v3 or v2 depending on formation_artifacts)
                model, all_diagnostics = train_window_model(
                    panel=panel_train,
                    metadata=universe_metadata,
                    t0=t0,
                    config=config,
                    formation_artifacts=formation_artifacts  # Pass artifacts for v3 pipeline
                )
                
                # Extract diagnostics
                diagnostics_entry = {
                    'date': t0,
                    'n_features': len(model.selected_features) if model else 0,
                    'selected_features': model.selected_features if model else [],
                    'pipeline_version': 'v3' if formation_artifacts else 'v2',
                    **all_diagnostics  # Include stage counts, timings, etc.
                }
                
                # Add Formation artifacts for bucket breakdown analysis
                if formation_artifacts:
                    diagnostics_entry['formation_artifacts'] = formation_artifacts
                    if 'formation_diagnostics' in formation_artifacts:
                        diagnostics_entry['formation_n_approved'] = formation_artifacts['formation_diagnostics']['n_approved']
                
                train_elapsed = time.time() - train_start
            except Exception as e:
                if verbose_inner:
                    print(f"[error] Model training failed: {e}")
                return None
            
            # Get eligible universe
            eligible_universe = get_eligible_universe(
                panel_df, universe_metadata, t0, config, verbose_inner
            )
            
            if len(eligible_universe) < 10:
                if verbose_inner:
                    print(f"[skip] Insufficient universe size: {len(eligible_universe)}")
                return None
            
            if verbose_inner:
                print(f"[universe] Eligible tickers: {len(eligible_universe)}")
            
            diagnostics_entry['universe_size'] = len(eligible_universe)
            
            # Score eligible tickers
            if verbose_inner:
                print(f"[score] Generating cross-sectional scores...")
            
            score_start = time.time()
            try:
                scores = model.score_at_date(
                    panel=panel_df,
                    t0=t0,
                    universe_metadata=universe_metadata,
                    config=config
                )
                score_elapsed = time.time() - score_start
            except Exception as e:
                if verbose_inner:
                    print(f"[error] Scoring failed: {e}")
                return None
            
            scores = scores[scores.index.isin(eligible_universe.index)]
            
            if len(scores) < 10:
                if verbose_inner:
                    print(f"[skip] Insufficient scores: {len(scores)}")
                return None
            
            # Filter universe_metadata to eligible tickers
            if 'ticker' in universe_metadata.columns:
                eligible_metadata = universe_metadata[
                    universe_metadata['ticker'].isin(eligible_universe.index)
                ].copy()
            else:
                eligible_metadata = universe_metadata[
                    universe_metadata.index.isin(eligible_universe.index)
                ].copy()
            
            # Determine portfolio mode from regime
            if regime_series is not None:
                current_regime = regime_series.get(t0, 'range')
                mode = get_portfolio_mode_for_regime(current_regime)
                if verbose_inner:
                    print(f"[regime] Current regime: {current_regime} -> mode: {mode}")
            else:
                mode = 'ls'
                current_regime = 'none'
            
            # Construct portfolio
            if verbose_inner:
                print(f"[portfolio] Constructing portfolio with caps...")
            
            portfolio_start = time.time()
            try:
                long_weights, short_weights, portfolio_stats = construct_portfolio(
                    scores=scores,
                    universe_metadata=eligible_metadata,
                    config=config,
                    method=portfolio_method,
                    mode=mode
                )
                portfolio_elapsed = time.time() - portfolio_start
            except Exception as e:
                if verbose_inner:
                    print(f"[error] Portfolio construction failed: {e}")
                return None
            
            if verbose_inner:
                print(f"[portfolio] Long: {portfolio_stats['n_long']} positions, "
                      f"gross: {portfolio_stats['gross_long']:.2%}")
                print(f"[portfolio] Short: {portfolio_stats['n_short']} positions, "
                      f"gross: {portfolio_stats['gross_short']:.2%}")
                
                if 'cap_violations' in portfolio_stats:
                    print(f"[warn] Cap violations: {portfolio_stats['cap_violations']}")
            
            # Evaluate performance (no previous weights in parallel mode)
            if verbose_inner:
                print(f"[eval] Evaluating forward returns...")
            
            eval_start = time.time()
            try:
                performance = evaluate_portfolio_return(
                    panel_df=panel_df,
                    t0=t0,
                    long_weights=long_weights,
                    short_weights=short_weights,
                    config=config,
                    prev_long_weights=None,  # Not available in parallel
                    prev_short_weights=None
                )
                eval_elapsed = time.time() - eval_start
            except Exception as e:
                if verbose_inner:
                    print(f"[error] Evaluation failed: {e}")
                return None
            
            performance.update(portfolio_stats)
            
            rebalance_elapsed = time.time() - rebalance_start
            if verbose_inner:
                print(f"[result] Long: {performance['long_ret'] * 100:.2f}%, Short: {performance['short_ret'] * 100:.2f}%, "
                      f"L/S: {performance['ls_return'] * 100:.2f}%, Turnover: {performance['turnover']:.2%}, "
                      f"TxnCost: {performance['transaction_cost']:.4%}")
                print(f"[time] Rebalance completed in {rebalance_elapsed:.2f}s "
                      f"(train: {train_elapsed:.1f}s, score: {score_elapsed:.1f}s, "
                      f"portfolio: {portfolio_elapsed:.1f}s, eval: {eval_elapsed:.1f}s)")
            
            return {
                'performance': performance,
                'diagnostics': diagnostics_entry,
                'long_weights': long_weights,
                'short_weights': short_weights,
                'regime': current_regime,
                'mode': mode,
                'eligible_metadata': eligible_metadata
            }
        
        # Execute sequentially (outer loop) - inner feature selection uses max parallelism
        parallel_results = Parallel(n_jobs=1, backend='loky')(
            delayed(process_rebalance_period)(
                i, t0, panel_df, universe_metadata, config, model_type, 
                portfolio_method, regime_series, verbose  # verbose_inner=verbose for sequential outer loop
            )
            for i, t0 in enumerate(current_dates)
        )
        
        # =====================================================================
        # FIX: Recompute turnover and transaction costs with correct prev_weights
        # =====================================================================
        # Parallel execution computed portfolios independently (prev_weights=None),
        # so turnover/costs are wrong. We now ONLY recalculate turnover/costs
        # sequentially WITHOUT re-reading panel data (which is expensive).
        
        if verbose:
            print(f"\n[parallel] Recomputing turnover for {len([r for r in parallel_results if r is not None])} periods...")
        
        results = []
        diagnostics = []
        accounting_debug_rows = []
        capital_history = []
        current_capital = 1.0
        
        # Track previous weights for turnover
        prev_long_weights = None
        prev_short_weights = None
        
        for i, result in enumerate(parallel_results):
            if result is None:
                continue
            
            # Get current weights
            long_weights = result['long_weights']
            short_weights = result['short_weights']
            
            # Get original performance (has correct asset returns, cash, etc.)
            performance = result['performance'].copy()
            
            # ===== RECALCULATE TURNOVER (the only thing that changed) =====
            gross_long = long_weights.sum() if len(long_weights) > 0 else 0.0
            gross_short = abs(short_weights.sum()) if len(short_weights) > 0 else 0.0
            
            turnover_long = 0.0
            turnover_short = 0.0
            
            if prev_long_weights is not None:
                # Rebalancing: 0.5 factor for one-way turnover
                all_tickers = long_weights.index.union(prev_long_weights.index)
                curr_w = long_weights.reindex(all_tickers, fill_value=0.0)
                prev_w = prev_long_weights.reindex(all_tickers, fill_value=0.0)
                turnover_long = 0.5 * (curr_w - prev_w).abs().sum()
            else:
                # First period: full entry
                turnover_long = long_weights.abs().sum()
            
            if prev_short_weights is not None:
                # Rebalancing: 0.5 factor for one-way turnover
                all_tickers = short_weights.index.union(prev_short_weights.index)
                curr_w = short_weights.reindex(all_tickers, fill_value=0.0)
                prev_w = prev_short_weights.reindex(all_tickers, fill_value=0.0)
                turnover_short = 0.5 * (curr_w - prev_w).abs().sum()
            else:
                # First period: full entry
                turnover_short = short_weights.abs().sum()
            
            total_turnover = turnover_long + turnover_short
            
            # Recalculate transaction costs
            cost_bps = config.portfolio.total_cost_bps_per_side
            transaction_cost = cost_bps * total_turnover / 10000.0
            
            # Update performance with corrected values
            # Subtract old (wrong) transaction cost, add new (correct) one
            old_txn_cost = performance['transaction_cost']
            performance['turnover'] = total_turnover
            performance['turnover_long'] = turnover_long
            performance['turnover_short'] = turnover_short
            performance['transaction_cost'] = transaction_cost
            
            # Adjust ls_return to reflect corrected transaction cost
            performance['ls_return'] = performance['ls_return'] + old_txn_cost - transaction_cost
            
            # Update capital using corrected decimal return
            period_return_decimal = performance['ls_return']
            current_capital *= (1.0 + period_return_decimal)
            performance['capital'] = current_capital
            capital_history.append(current_capital)
            
            # Update prev_weights for next iteration
            prev_long_weights = long_weights.copy()
            prev_short_weights = short_weights.copy()
            
            # Accounting debug logging
            if config.debug.enable_accounting_debug:
                if config.debug.debug_max_periods == 0 or len(accounting_debug_rows) < config.debug.debug_max_periods:
                    accounting_debug_rows.append({
                        "date": result['diagnostics']['date'],
                        "regime": result['regime'],
                        "mode": result['mode'],
                        "universe_size": len(result['eligible_metadata']),
                        "gross_long": float(result['long_weights'].abs().sum()) if len(result['long_weights']) > 0 else 0.0,
                        "gross_short": float(result['short_weights'].abs().sum()) if len(result['short_weights']) > 0 else 0.0,
                        "cash_weight": performance.get("cash_weight", np.nan),
                        "naive_ls_ret": performance.get("naive_ls_ret", np.nan),
                        "cash_pnl": performance.get("cash_pnl", np.nan),
                        "transaction_cost": performance.get("transaction_cost", np.nan),
                        "borrow_cost": performance.get("borrow_cost", np.nan),
                        "ls_return": performance.get("ls_return", np.nan),
                        "capital": current_capital,
                    })
            
            results.append(performance)
            diagnostics.append(result['diagnostics'])
        
        if verbose:
            print(f"\n[parallel] Completed {len(results)} rebalance periods")
        
        # NOTE: In parallel mode, timing is not tracked per-step (would need aggregation)
        # Initialize variables to prevent UnboundLocalError in reporting section
        total_train_time = 0.0
        total_score_time = 0.0
        total_portfolio_time = 0.0
        total_eval_time = 0.0
    
    else:
        # SEQUENTIAL EXECUTION: Original implementation
        results = []
        diagnostics = []  # Store diagnostics for each rebalance period
        
        # FIX 4: Capital compounding and tracking
        current_capital = 1.0
        capital_history = []
        
        # Accounting debug logging
        accounting_debug_rows = []
        
        # Track previous weights for turnover calculation
        prev_long_weights = None
        prev_short_weights = None
        
        # Diagnostic tracking
        ic_history = []  # Store IC vectors for each window
        feature_selection_history = []  # Track which features were selected
        universe_size_history = []  # Track universe size after each filter
        
        # Track timing for each step
        import time
        total_train_time = 0
        total_score_time = 0
        total_portfolio_time = 0
        total_eval_time = 0
        
        for i, t0 in enumerate(current_dates):
            rebalance_start = time.time()
            if verbose:
                print(f"\n{'='*60}")
                print(f"[{i+1}/{len(current_dates)}] Rebalance date: {t0.date()}")
            
            # =====================================================================
            # 1. Compute Formation artifacts (v3 pipeline)
            # =====================================================================
            formation_artifacts = None
            
            if hasattr(config.features, 'formation_years') and config.features.formation_years > 0:
                # V3 PIPELINE: Use Formation/Training split
                if verbose:
                    print(f"[v3] Using Formation ({config.features.formation_years:.1f}yr) + Training ({config.features.training_years:.1f}yr) pipeline")
                
                formation_artifacts = compute_formation_artifacts(
                    panel_df=panel_df,
                    universe_metadata=universe_metadata,
                    t0=t0,
                    config=config,
                    verbose=verbose
                )
                
                if formation_artifacts is None:
                    if verbose:
                        print("[skip] Formation artifacts computation failed")
                    continue
            
            # =====================================================================
            # 2. Define Training window
            # =====================================================================
            if formation_artifacts is not None:
                # V3: Training window is shorter (1 year by default)
                training_days = int(config.features.training_years * 252)
                t_train_start = t0 - pd.Timedelta(days=training_days)
                t_train_end = t0 - pd.Timedelta(days=1 + config.time.HOLDING_PERIOD_DAYS)
            else:
                # V2: Use old TRAINING_WINDOW_DAYS config
                t_train_start = t0 - pd.Timedelta(days=config.time.TRAINING_WINDOW_DAYS)
                t_train_end = t0 - pd.Timedelta(days=1 + config.time.HOLDING_PERIOD_DAYS)
            
            if verbose:
                print(f"Training window: [{t_train_start.date()}, {t_train_end.date()}]")
            
            # Check training window is valid
            if t_train_end <= t_train_start:
                if verbose:
                    print("[skip] Invalid training window")
                continue
            
            # =====================================================================
            # 3. Train model with feature selection
            # =====================================================================
            if verbose:
                pipeline_version = "v3" if formation_artifacts else "v2"
                print(f"[train] Training {model_type} model ({pipeline_version} pipeline)...")
            
            train_start = time.time()
            try:
                # Extract training panel
                train_mask = (
                    (panel_df.index.get_level_values('Date') >= t_train_start) &
                    (panel_df.index.get_level_values('Date') <= t_train_end)
                )
                panel_train = panel_df[train_mask]
                
                # Train model with feature selection (v3 or v2 depending on formation_artifacts)
                model, all_diagnostics = train_window_model(
                    panel=panel_train,
                    metadata=universe_metadata,
                    t0=t0,
                    config=config,
                    formation_artifacts=formation_artifacts  # Pass artifacts for v3 pipeline
                )
                
                # Record diagnostics
                diagnostics_entry = {
                    'date': t0,
                    'n_features': len(model.selected_features) if model else 0,
                    'selected_features': model.selected_features if model else [],
                    'pipeline_version': 'v3' if formation_artifacts else 'v2',
                    **all_diagnostics  # Include stage counts, timings, etc.
                }
                
                # Add Formation diagnostics if available
                if formation_artifacts and 'formation_diagnostics' in formation_artifacts:
                    diagnostics_entry['formation_n_approved'] = formation_artifacts['formation_diagnostics']['n_approved']
                
                train_elapsed = time.time() - train_start
                total_train_time += train_elapsed
                
            except Exception as e:
                if verbose:
                    print(f"[error] Model training failed: {e}")
                    import traceback
                    traceback.print_exc()
                continue
            
            # =====================================================================
            # 3. Get eligible universe at t0
            # =====================================================================
            eligible_universe = get_eligible_universe(
                panel_df, universe_metadata, t0, config, verbose
            )
            
            if len(eligible_universe) < 10:
                if verbose:
                    print(f"[skip] Insufficient universe size: {len(eligible_universe)}")
                continue
            
            if verbose:
                print(f"[universe] Eligible tickers: {len(eligible_universe)}")
            
            # Add universe size to diagnostics
            diagnostics_entry['universe_size'] = len(eligible_universe)
            
            # =====================================================================
            # 4. Score eligible tickers
            # =====================================================================
            if verbose:
                print(f"[score] Generating cross-sectional scores...")
            
            score_start = time.time()
            try:
                scores = model.score_at_date(
                    panel=panel_df,
                    t0=t0,
                    universe_metadata=universe_metadata,
                    config=config
                )
                score_elapsed = time.time() - score_start
                total_score_time += score_elapsed
            except Exception as e:
                if verbose:
                    print(f"[error] Scoring failed: {e}")
                continue
            
            # Filter scores to eligible universe
            scores = scores[scores.index.isin(eligible_universe.index)]
            
            if len(scores) < 10:
                if verbose:
                    print(f"[skip] Insufficient scores: {len(scores)}")
                continue
            
            # =====================================================================
            # 5. Construct portfolio
            # =====================================================================
            if verbose:
                print(f"[portfolio] Constructing portfolio with caps...")
            
            # CRITICAL: Filter universe_metadata to only eligible tickers
            # This ensures portfolio construction only considers eligible universe
            if 'ticker' in universe_metadata.columns:
                eligible_metadata = universe_metadata[
                    universe_metadata['ticker'].isin(eligible_universe.index)
                ].copy()
            else:
                eligible_metadata = universe_metadata[
                    universe_metadata.index.isin(eligible_universe.index)
                ].copy()
            
            # =====================================================================
            # 5. Determine portfolio mode from regime
            # =====================================================================
            if regime_series is not None:
                # Get regime at t0 (already shifted by 1 day to avoid look-ahead)
                current_regime = regime_series.get(t0, 'range')  # Default to range if not found
                mode = get_portfolio_mode_for_regime(current_regime)
                if verbose:
                    print(f"[regime] Current regime: {current_regime} -> mode: {mode}")
            else:
                mode = 'ls'  # Default long/short mode
            
            portfolio_start = time.time()
            try:
                # Construct portfolio with dimensionless leverage weights
                # No capital parameter - weights are leverage multipliers relative to equity
                long_weights, short_weights, portfolio_stats = construct_portfolio(
                    scores=scores,
                    universe_metadata=eligible_metadata,
                    config=config,
                    method=portfolio_method,
                    mode=mode
                )
                
                portfolio_elapsed = time.time() - portfolio_start
                total_portfolio_time += portfolio_elapsed
            except Exception as e:
                if verbose:
                    print(f"[error] Portfolio construction failed: {e}")
                continue
            
            if verbose:
                print(f"[portfolio] Long: {portfolio_stats['n_long']} positions, "
                      f"gross: {portfolio_stats['gross_long']:.2%}")
                print(f"[portfolio] Short: {portfolio_stats['n_short']} positions, "
                      f"gross: {portfolio_stats['gross_short']:.2%}")
                
                if 'cap_violations' in portfolio_stats:
                    print(f"[warn] Cap violations: {portfolio_stats['cap_violations']}")
            
            # =====================================================================
            # 6. Evaluate performance
            # =====================================================================
            if verbose:
                print(f"[eval] Evaluating forward returns...")
            
            eval_start = time.time()
            try:
                performance = evaluate_portfolio_return(
                    panel_df=panel_df,
                    t0=t0,
                    long_weights=long_weights,
                    short_weights=short_weights,
                    config=config,
                    prev_long_weights=prev_long_weights,
                    prev_short_weights=prev_short_weights
                )
                eval_elapsed = time.time() - eval_start
                total_eval_time += eval_elapsed
            except Exception as e:
                if verbose:
                    print(f"[error] Evaluation failed: {e}")
                continue
            
            # Add portfolio stats to performance
            performance.update(portfolio_stats)
            
            # FIX 4: Update capital using decimal return
            period_return_decimal = performance['ls_return']  # Already decimal
            current_capital *= (1.0 + period_return_decimal)
            performance['capital'] = current_capital
            capital_history.append(current_capital)
            
            # Accounting debug logging
            if config.debug.enable_accounting_debug:
                if config.debug.debug_max_periods == 0 or len(accounting_debug_rows) < config.debug.debug_max_periods:
                    accounting_debug_rows.append({
                        "date": t0,
                        "regime": current_regime if regime_series is not None else "none",
                        "mode": mode,
                        "universe_size": len(eligible_metadata),
                        "gross_long": float(long_weights.abs().sum()) if len(long_weights) > 0 else 0.0,
                        "gross_short": float(short_weights.abs().sum()) if len(short_weights) > 0 else 0.0,
                        "naive_ls_ret": performance.get("naive_ls_ret", np.nan),
                        "cash_pnl": performance.get("cash_pnl", np.nan),
                        "transaction_cost": performance.get("transaction_cost", np.nan),
                        "borrow_cost": performance.get("borrow_cost", np.nan),
                        "ls_return": performance.get("ls_return", np.nan),
                        "capital": current_capital,
                    })
            
            # Add rebalance timing
            rebalance_elapsed = time.time() - rebalance_start
            if verbose:
                print(f"[result] Long: {performance['long_ret'] * 100:.2f}%, Short: {performance['short_ret'] * 100:.2f}%, "
                      f"L/S: {performance['ls_return'] * 100:.2f}%, Turnover: {performance['turnover']:.2%}, "
                      f"TxnCost: {performance['transaction_cost']:.4%}")
                print(f"[time] Rebalance completed in {rebalance_elapsed:.2f}s "
                      f"(train: {train_elapsed:.1f}s, score: {score_elapsed:.1f}s, "
                      f"portfolio: {portfolio_elapsed:.1f}s, eval: {eval_elapsed:.1f}s)")
            
            results.append(performance)
            
            # Append diagnostics for this period
            diagnostics.append(diagnostics_entry)
            
            # Update previous weights for next iteration
            prev_long_weights = long_weights.copy()
            prev_short_weights = short_weights.copy()
    
    # =========================================================================
    # Convert results to DataFrame
    # =========================================================================
    if len(results) == 0:
        if verbose:
            print("\n[error] No valid results")
        return pd.DataFrame()
    
    results_df = pd.DataFrame(results)
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df = results_df.set_index('date').sort_index()
    
    # FIX 4: Verify capital compounding consistency
    if len(results_df) > 0:
        cumulative_return_check = (1 + results_df['ls_return']).prod()
        capital_error = abs(cumulative_return_check - current_capital)
        
        if verbose:
            print("\n" + "="*80)
            print("CAPITAL COMPOUNDING VERIFICATION")
            print("="*80)
            print(f"Final capital (tracked):        {current_capital:.6f}")
            print(f"Final capital (from returns):   {cumulative_return_check:.6f}")
            print(f"Absolute error:                 {capital_error:.2e}")
            
            if capital_error > 1e-6:
                print(f"[WARN] Capital tracking mismatch detected!")
            else:
                print(f"[OK] Capital tracking is consistent [OK]")
    
    if verbose:
        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)
        print(f"Total periods: {len(results_df)}")
        print(f"Date range: {results_df.index[0].date()} to {results_df.index[-1].date()}")
        print(f"\n[time] Backtest timing breakdown:")
        print(f"  Total training time:    {total_train_time:.2f}s ({total_train_time/60:.2f}min)")
        print(f"  Total scoring time:     {total_score_time:.2f}s ({total_score_time/60:.2f}min)")
        print(f"  Total portfolio time:   {total_portfolio_time:.2f}s ({total_portfolio_time/60:.2f}min)")
        print(f"  Total evaluation time:  {total_eval_time:.2f}s ({total_eval_time/60:.2f}min)")
        total_backtest_time = total_train_time + total_score_time + total_portfolio_time + total_eval_time
        print(f"  Total backtest time:    {total_backtest_time:.2f}s ({total_backtest_time/60:.2f}min)")
        print(f"  Average per rebalance:  {total_backtest_time/len(results_df):.2f}s")
    
    # Save accounting debug log if enabled
    if accounting_debug_rows:
        debug_df = pd.DataFrame(accounting_debug_rows).set_index("date")
        debug_path = Path(config.paths.data_dir) / "accounting_debug_log.csv"
        debug_df.to_csv(debug_path)
        if verbose:
            print(f"\n[debug] Accounting debug log saved to: {debug_path}")
            print(f"[debug] Logged {len(accounting_debug_rows)} periods")
    
    # Save diagnostics if requested
    if config.compute.save_intermediate and len(diagnostics) > 0:
        import json
        
        # Determine output path
        if config.compute.ic_output_path:
            diag_dir = Path(config.compute.ic_output_path).parent
        else:
            diag_dir = Path(config.paths.plots_dir)
        
        diag_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary diagnostics to CSV
        diag_summary = []
        for entry in diagnostics:
            # Get top 5 features by |IC| (handle both v2 and v3 diagnostics format)
            ic_dict = entry.get('ic_values', {})
            if ic_dict:
                abs_ic = {k: abs(v) for k, v in ic_dict.items() if pd.notna(v)}
                top_features = sorted(abs_ic.items(), key=lambda x: x[1], reverse=True)[:5]
                top_features_str = '; '.join([f"{k}={v:.3f}" for k, v in top_features])
            else:
                top_features_str = ""
            
            diag_summary.append({
                'date': entry.get('date', ''),
                'universe_size': entry.get('universe_size', 0),
                'n_features': entry.get('n_features', entry.get('final_n_features', 0)),
                'top_5_features': top_features_str
            })
        
        diag_df = pd.DataFrame(diag_summary)
        diag_csv_path = diag_dir / 'diagnostics_summary.csv'
        diag_df.to_csv(diag_csv_path, index=False)
        
        # Save full IC history to JSON
        ic_json_path = diag_dir / 'ic_history.json'
        with open(ic_json_path, 'w') as f:
            # Convert dates to strings for JSON serialization
            json_data = []
            for entry in diagnostics:
                date_val = entry.get('date')
                if hasattr(date_val, 'isoformat'):
                    date_str = date_val.isoformat()
                else:
                    date_str = str(date_val) if date_val else ''
                
                json_entry = {
                    'date': date_str,
                    'universe_size': entry.get('universe_size', 0),
                    'n_features': entry.get('n_features', entry.get('final_n_features', 0)),
                    'selected_features': entry.get('selected_features', []),
                    'ic_values': entry.get('ic_values', {})
                }
                json_data.append(json_entry)
            json.dump(json_data, f, indent=2)
        
        if verbose:
            print(f"\n[save] Diagnostics saved to:")
            print(f"  Summary: {diag_csv_path}")
            print(f"  Full IC history: {ic_json_path}")
    
    # Store diagnostics in results_df as metadata
    if len(diagnostics) > 0:
        results_df.attrs['diagnostics'] = diagnostics
    
    # Compute attribution analysis
    if verbose and len(results_df) > 0 and len(diagnostics) > 0:
        try:
            attribution_results = compute_attribution_analysis(
                results_df=results_df,
                diagnostics=diagnostics,
                panel_df=panel_df,
                universe_metadata=universe_metadata,
                config=config
            )
            
            # Store attribution in results metadata
            results_df.attrs['attribution'] = attribution_results
            
            # Save attribution to CSV files
            if config.compute.save_intermediate:
                output_dir = Path(config.paths.data_dir)
                save_attribution_results(
                    attribution_results,
                    output_dir=str(output_dir),
                    prefix="attribution"
                )
        except Exception as e:
            print(f"\n[warn] Attribution analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print timing summary from cache
    if verbose:
        try:
            from feature_selection_v4 import print_cache_timing_summary
            print_cache_timing_summary()
        except Exception as e:
            print(f"\n[warn] Could not print timing summary: {e}")
    
    return results_df


def bootstrap_performance_stats(
    returns: pd.Series,
    config: ResearchConfig,
    n_bootstrap: int = 1000,
    block_size: int = 6,
    random_state: int = 42
) -> Dict:
    """
    Generate bootstrap confidence intervals for performance statistics.
    
    Uses block bootstrap to preserve time-series autocorrelation structure.
    
    Parameters
    ----------
    returns : pd.Series
        Period returns (e.g., monthly returns in %)
    n_bootstrap : int
        Number of bootstrap samples
    block_size : int
        Block size in periods (default 6 = 6 months for monthly returns)
    random_state : int
        Random seed for reproducibility
        
    Returns
    -------
    dict
        Bootstrap statistics with confidence intervals:
        - sharpe_mean: Mean Sharpe from bootstrap distribution
        - sharpe_ci_5: 5th percentile (lower bound of 90% CI)
        - sharpe_ci_95: 95th percentile (upper bound of 90% CI)
        - return_mean: Mean return from bootstrap distribution
        - return_ci_5/95: Confidence intervals for returns
    """
    np.random.seed(random_state)
    
    if len(returns) < block_size * 2:
        # Not enough data for bootstrap
        return {
            'sharpe_mean': np.nan,
            'sharpe_ci_5': np.nan,
            'sharpe_ci_95': np.nan,
            'return_mean': np.nan,
            'return_ci_5': np.nan,
            'return_ci_95': np.nan,
        }
    
    sharpe_dist = []
    return_dist = []
    
    n_blocks = len(returns) // block_size
    
    # Convert to numpy array to avoid attrs/deepcopy issues
    returns_values = returns.values
    
    for i in range(n_bootstrap):
        # Sample blocks with replacement (block bootstrap)
        sampled_blocks = np.random.choice(n_blocks, size=n_blocks, replace=True)
        
        bootstrap_returns = []
        for block_idx in sampled_blocks:
            start = block_idx * block_size
            end = min(start + block_size, len(returns_values))
            bootstrap_returns.extend(returns_values[start:end])
        
        # Compute statistics on bootstrapped sample
        boot_mean = np.mean(bootstrap_returns)
        boot_std = np.std(bootstrap_returns, ddof=1)  # Use sample std
        
        return_dist.append(boot_mean)
        
        if boot_std > 0:
            # Sharpe ratio with risk-free rate
            periods_per_year = 252 / config.time.HOLDING_PERIOD_DAYS
            rf_per_period = config.portfolio.cash_rate / periods_per_year
            excess_return = boot_mean - rf_per_period
            sharpe = excess_return / boot_std
            sharpe_dist.append(sharpe)
    
    # Compute confidence intervals
    return {
        'sharpe_mean': np.mean(sharpe_dist),
        'sharpe_ci_5': np.percentile(sharpe_dist, 5),
        'sharpe_ci_95': np.percentile(sharpe_dist, 95),
        'return_mean': np.mean(return_dist),
        'return_ci_5': np.percentile(return_dist, 5),
        'return_ci_95': np.percentile(return_dist, 95),
    }


def attribution_analysis(
    results_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    universe_metadata: pd.DataFrame,
    config: ResearchConfig
) -> Dict:
    """
    Comprehensive attribution analysis to understand return drivers.
    
    Analyzes:
    1. Feature attribution: Which features contribute most to returns?
    2. Sector/cluster attribution: Which themes drive performance?
    3. Long/short attribution: Performance breakdown by side
    4. Regime attribution: Performance by market regime
    5. IC decay: Out-of-sample IC vs in-sample IC
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Backtest results with columns: date, long_ret, short_ret, etc.
    panel_df : pd.DataFrame
        Full panel with MultiIndex (Date, Ticker)
    universe_metadata : pd.DataFrame
        ETF metadata with cluster_id
    config : ResearchConfig
        Configuration
        
    Returns
    -------
    dict
        Attribution breakdowns
    """
    attribution = {}
    
    # ===== 1. Long/Short Attribution (already available) =====
    # Calculate Sharpe with risk-free rate
    periods_per_year = 252 / config.time.HOLDING_PERIOD_DAYS
    rf_per_period = config.portfolio.cash_rate / periods_per_year
    
    long_excess = results_df['long_ret'].mean() - rf_per_period
    short_excess = results_df['short_ret'].mean() - rf_per_period
    
    attribution['long_short'] = {
        'long_mean': results_df['long_ret'].mean(),
        'short_mean': results_df['short_ret'].mean(),
        'long_std': results_df['long_ret'].std(),
        'short_std': results_df['short_ret'].std(),
        'long_sharpe': long_excess / results_df['long_ret'].std() if results_df['long_ret'].std() > 0 else 0,
        'short_sharpe': short_excess / results_df['short_ret'].std() if results_df['short_ret'].std() > 0 else 0,
    }
    
    # ===== 2. Regime Attribution =====
    if 'regime' in results_df.columns:
        regime_stats = results_df.groupby('regime').agg({
            'ls_return': ['mean', 'std', 'count'],
            'long_ret': 'mean',
            'short_ret': 'mean'
        }).to_dict()
        attribution['regime'] = regime_stats
    else:
        attribution['regime'] = None
    
    # ===== 3. Sector/Cluster Attribution =====
    # Aggregate returns by cluster
    # Use raw returns for P&L attribution (actual economic returns)
    cluster_attribution = {}
    raw_target_col = f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}'
    
    if raw_target_col in panel_df.columns and 'cluster_id' in universe_metadata.columns:
        # Set index properly for metadata lookup
        if 'ticker' in universe_metadata.columns:
            meta_idx = universe_metadata.set_index('ticker')
        else:
            meta_idx = universe_metadata
        
        for idx, row in results_df.iterrows():
            t0 = row['date']
            
            # Get long positions
            if 'long_tickers' in row and isinstance(row['long_tickers'], list):
                for ticker in row['long_tickers']:
                    if ticker in meta_idx.index:
                        cluster = meta_idx.loc[ticker, 'cluster_id']
                        
                        # Get actual return (raw, for P&L)
                        if (t0, ticker) in panel_df.index:
                            ret = panel_df.loc[(t0, ticker), raw_target_col]
                            
                            if cluster not in cluster_attribution:
                                cluster_attribution[cluster] = []
                            cluster_attribution[cluster].append(ret)
        
        # Aggregate by cluster
        cluster_summary = {
            cluster: {
                'mean_ret': np.mean(rets),
                'count': len(rets)
            }
            for cluster, rets in cluster_attribution.items()
            if len(rets) > 0
        }
        
        attribution['cluster'] = cluster_summary
    else:
        attribution['cluster'] = None
    
    # ===== 4. Feature Attribution (IC-based) =====
    # For each selected feature across all periods, compute correlation with returns
    # Use configured target column for consistency with model training
    feature_ic = {}
    target_col = config.target.target_column
    if target_col not in panel_df.columns:
        target_col = raw_target_col  # Fallback to raw
    
    # Get all binned features that were used
    binned_features = [col for col in panel_df.columns if col.endswith('_Bin')]
    
    if len(binned_features) > 0 and target_col in panel_df.columns:
        for feat in binned_features:
            # Compute overall IC (out-of-sample, since we only evaluate on scoring dates)
            valid_mask = panel_df[feat].notna() & panel_df[target_col].notna()
            
            if valid_mask.sum() > 10:
                from scipy.stats import spearmanr
                ic, pval = spearmanr(
                    panel_df.loc[valid_mask, feat],
                    panel_df.loc[valid_mask, target_col]
                )
                feature_ic[feat] = {
                    'ic': ic,
                    'pval': pval,
                    'n': valid_mask.sum()
                }
        
        attribution['feature_ic'] = feature_ic
    else:
        attribution['feature_ic'] = None
    
    # ===== 5. IC Decay Analysis =====
    # Compare in-sample IC (training) vs out-of-sample IC (scoring)
    # This requires storing IC values during training, which we'll track
    if 'ic_values' in results_df.columns:
        # If we stored IC values per period
        attribution['ic_decay'] = {
            'mean_ic': results_df['ic_values'].mean() if 'ic_values' in results_df.columns else None,
            'std_ic': results_df['ic_values'].std() if 'ic_values' in results_df.columns else None,
        }
    else:
        attribution['ic_decay'] = None
    
    return attribution


def print_attribution_report(attribution: Dict):
    """
    Print formatted attribution analysis report.
    
    Parameters
    ----------
    attribution : dict
        Attribution dictionary from attribution_analysis()
    """
    print("\n" + "="*80)
    print("ATTRIBUTION ANALYSIS")
    print("="*80)
    
    # Long/Short Attribution
    if 'long_short' in attribution and attribution['long_short']:
        print("\n1. LONG/SHORT ATTRIBUTION")
        print("-" * 40)
        ls = attribution['long_short']
        print(f"Long  : {ls['long_mean']:>6.2f}% mean, {ls['long_std']:>5.2f}% std, Sharpe={ls['long_sharpe']:>5.2f}")
        print(f"Short : {ls['short_mean']:>6.2f}% mean, {ls['short_std']:>5.2f}% std, Sharpe={ls['short_sharpe']:>5.2f}")
    
    # Regime Attribution
    if 'regime' in attribution and attribution['regime']:
        print("\n2. REGIME ATTRIBUTION")
        print("-" * 40)
        print("(Regime-based performance breakdown)")
        # TODO: Format regime stats nicely
    
    # Cluster Attribution
    if 'cluster' in attribution and attribution['cluster']:
        print("\n3. SECTOR/CLUSTER ATTRIBUTION (Top 10)")
        print("-" * 40)
        clusters = attribution['cluster']
        # Sort by mean return
        sorted_clusters = sorted(clusters.items(), key=lambda x: x[1]['mean_ret'], reverse=True)[:10]
        for cluster, stats in sorted_clusters:
            print(f"{cluster:30s}: {stats['mean_ret']:>6.2f}% (n={stats['count']})")
    
    # Feature IC
    if 'feature_ic' in attribution and attribution['feature_ic']:
        print("\n4. FEATURE ATTRIBUTION (Top 10 by |IC|)")
        print("-" * 40)
        feat_ic = attribution['feature_ic']
        # Sort by absolute IC
        sorted_feats = sorted(feat_ic.items(), key=lambda x: abs(x[1]['ic']), reverse=True)[:10]
        for feat, stats in sorted_feats:
            print(f"{feat:30s}: IC={stats['ic']:>6.3f}, p={stats['pval']:>6.4f}, n={stats['n']}")
    
    # IC Decay
    if 'ic_decay' in attribution and attribution['ic_decay']:
        print("\n5. IC DECAY ANALYSIS")
        print("-" * 40)
        decay = attribution['ic_decay']
        if decay['mean_ic'] is not None:
            print(f"Mean IC: {decay['mean_ic']:.3f}")
            print(f"Std IC:  {decay['std_ic']:.3f}")
    
    print("\n" + "="*80)


def analyze_performance(results_df: pd.DataFrame, config: ResearchConfig) -> Dict:
    """
    Compute comprehensive performance and risk statistics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Backtest results from run_walk_forward_backtest
    config : ResearchConfig
        Configuration
        
    Returns
    -------
    dict
        Performance metrics including:
        - Return metrics (total, annual, mean)
        - Risk metrics (volatility, max drawdown, VaR, CVaR)
        - Risk-adjusted metrics (Sharpe, Sortino, Calmar, Omega)
        - Win/loss statistics
        - Bootstrap confidence intervals
    """
    if len(results_df) == 0:
        return {}
    
    returns = results_df['ls_return'].dropna()
    
    if len(returns) == 0:
        return {}
    
    # Basic statistics
    total_periods = len(returns)
    winning_periods = (returns > 0).sum()
    losing_periods = (returns < 0).sum()
    win_rate = winning_periods / total_periods
    
    # Return statistics (returns are now in decimal form)
    mean_ret = returns.mean()
    std_ret = returns.std()
    skewness = returns.skew()
    kurtosis = returns.kurtosis()
    
    # Sharpe ratio: (return - risk_free_rate) / volatility
    # Risk-free rate from config (annual) needs to be converted to per-period
    periods_per_year = 252 / config.time.HOLDING_PERIOD_DAYS
    risk_free_per_period = config.portfolio.cash_rate / periods_per_year
    excess_return = mean_ret - risk_free_per_period
    sharpe = excess_return / std_ret if std_ret > 0 else 0.0
    
    # Annualization
    annual_return = mean_ret * periods_per_year
    annual_vol = std_ret * np.sqrt(periods_per_year)
    annual_sharpe = sharpe * np.sqrt(periods_per_year)
    
    # Cumulative return (returns are decimals, so no /100 needed)
    cum_ret = (1 + returns).cumprod()
    total_return = cum_ret.iloc[-1] - 1
    
    # Drawdown analysis
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()
    
    # Time underwater
    underwater = drawdown < 0
    underwater_periods = underwater.sum()
    
    # Max drawdown duration
    dd_periods = []
    current_dd_length = 0
    for is_dd in underwater:
        if is_dd:
            current_dd_length += 1
        else:
            if current_dd_length > 0:
                dd_periods.append(current_dd_length)
            current_dd_length = 0
    if current_dd_length > 0:
        dd_periods.append(current_dd_length)
    max_dd_duration = max(dd_periods) if dd_periods else 0
    
    # === RISK METRICS ===
    
    # Calmar Ratio: Annual Return / |Max Drawdown|
    calmar = annual_return / abs(max_dd) if max_dd != 0 else 0.0
    
    # Sortino Ratio: (Return - Risk-free) / Downside Deviation
    # Only uses negative returns for volatility calculation
    downside_returns = returns[returns < risk_free_per_period] - risk_free_per_period
    downside_std = np.sqrt((downside_returns ** 2).mean()) if len(downside_returns) > 0 else 0
    sortino = excess_return / downside_std if downside_std > 0 else 0.0
    annual_sortino = sortino * np.sqrt(periods_per_year)
    
    # VaR (Value at Risk) - 5th percentile
    var_5pct = np.percentile(returns, 5)
    
    # CVaR (Conditional VaR / Expected Shortfall) - average of returns below VaR
    cvar_5pct = returns[returns <= var_5pct].mean() if len(returns[returns <= var_5pct]) > 0 else var_5pct
    
    # Omega Ratio: (1 + Probability weighted gains) / (Probability weighted losses)
    threshold = risk_free_per_period
    gains = returns[returns > threshold] - threshold
    losses = threshold - returns[returns <= threshold]
    omega = (gains.sum() + len(gains) * threshold) / (losses.sum() + len(losses) * threshold) if len(losses) > 0 and losses.sum() > 0 else np.inf
    
    # Profit Factor: Sum of wins / Sum of losses
    wins = returns[returns > 0]
    losses_raw = returns[returns < 0]
    profit_factor = wins.sum() / abs(losses_raw.sum()) if len(losses_raw) > 0 and losses_raw.sum() != 0 else np.inf
    
    # Average win/loss
    avg_win = wins.mean() if len(wins) > 0 else 0
    avg_loss = losses_raw.mean() if len(losses_raw) > 0 else 0
    win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else np.inf
    
    # Long/Short breakdown
    long_mean = results_df['long_ret'].mean()
    short_mean = results_df['short_ret'].mean()
    
    # Transaction costs breakdown
    total_txn_cost = results_df['transaction_cost'].sum()
    avg_txn_cost = results_df['transaction_cost'].mean()
    avg_turnover = results_df['turnover'].mean() if 'turnover' in results_df.columns else 0
    
    # Bootstrap confidence intervals
    bootstrap_stats = bootstrap_performance_stats(
        returns,
        config,
        n_bootstrap=1000,
        block_size=6,
        random_state=config.features.random_state if hasattr(config.features, 'random_state') else 42
    )
    
    stats = {
        # === RETURN METRICS ===
        '=== RETURN METRICS ===': '',
        'Total Periods': total_periods,
        'Total Return': f"{total_return * 100:.2f}%",
        'Annual Return': f"{annual_return * 100:.2f}%",
        'Mean Period Return': f"{mean_ret * 100:.2f}%",
        
        # === RISK METRICS ===
        '=== RISK METRICS ===': '',
        'Annual Volatility': f"{annual_vol * 100:.2f}%",
        'Max Drawdown': f"{max_dd * 100:.2f}%",
        'Max DD Duration': f"{max_dd_duration} periods",
        'Time Underwater': f"{underwater_periods}/{total_periods} periods ({underwater_periods/total_periods*100:.1f}%)",
        'VaR (5%)': f"{var_5pct * 100:.2f}%",
        'CVaR (5%)': f"{cvar_5pct * 100:.2f}%",
        'Skewness': f"{skewness:.2f}",
        'Kurtosis': f"{kurtosis:.2f}",
        
        # === RISK-ADJUSTED METRICS ===
        '=== RISK-ADJUSTED METRICS ===': '',
        'Annual Sharpe': f"{annual_sharpe:.2f}",
        'Annual Sortino': f"{annual_sortino:.2f}",
        'Calmar Ratio': f"{calmar:.2f}",
        'Omega Ratio': f"{omega:.2f}" if not np.isinf(omega) else 'N/A',
        
        # === WIN/LOSS STATISTICS ===
        '=== WIN/LOSS STATISTICS ===': '',
        'Win Rate': f"{win_rate:.1%}",
        'Winning Periods': f"{winning_periods}",
        'Losing Periods': f"{losing_periods}",
        'Avg Win': f"{avg_win * 100:.2f}%",
        'Avg Loss': f"{avg_loss * 100:.2f}%",
        'Win/Loss Ratio': f"{win_loss_ratio:.2f}" if not np.isinf(win_loss_ratio) else 'N/A',
        'Profit Factor': f"{profit_factor:.2f}" if not np.isinf(profit_factor) else 'N/A',
        
        # === LONG/SHORT BREAKDOWN ===
        '=== LONG/SHORT BREAKDOWN ===': '',
        'Long Avg Return': f"{long_mean * 100:.2f}%",
        'Short Avg Return': f"{short_mean * 100:.2f}%",
        
        # === COST ANALYSIS ===
        '=== COST ANALYSIS ===': '',
        'Total Transaction Costs': f"{total_txn_cost * 100:.2f}%",
        'Avg Transaction Cost/Period': f"{avg_txn_cost * 100:.3f}%",
        'Avg Turnover': f"{avg_turnover:.1%}",
        
        # === BOOTSTRAP CONFIDENCE INTERVALS ===
        '=== BOOTSTRAP CONFIDENCE INTERVALS ===': '',
        'Sharpe (Bootstrap Mean)': f"{bootstrap_stats['sharpe_mean']:.2f}",
        'Sharpe 90% CI': f"[{bootstrap_stats['sharpe_ci_5']:.2f}, {bootstrap_stats['sharpe_ci_95']:.2f}]",
        'Return (Bootstrap Mean)': f"{bootstrap_stats['return_mean'] * 100:.2f}%",
        'Return 90% CI': f"[{bootstrap_stats['return_ci_5'] * 100:.2f}%, {bootstrap_stats['return_ci_95'] * 100:.2f}%]",
    }
    
    return stats


def analyze_benchmark_comparison(
    results_df: pd.DataFrame,
    benchmark_period_returns: pd.Series,
    benchmark_daily_returns: pd.Series,
    config: ResearchConfig,
    benchmark_name: str = 'VT'
) -> Dict:
    """
    Compare strategy performance against a benchmark (e.g., VT).
    
    FIXED: Now uses daily returns for proper buy-and-hold cumulative return,
    while still using period returns for excess return and IR calculations.
    
    Computes:
    - Excess return (strategy - benchmark)
    - Tracking error (std of excess returns)
    - Information ratio (excess return / tracking error)
    - Beta to benchmark
    - Alpha (Jensen's alpha)
    - Correlation with benchmark
    - Up/down capture ratios
    - PROPER buy-and-hold cumulative return from daily returns

    Parameters
    ----------
    results_df : pd.DataFrame
        Backtest results with 'ls_return' column
    benchmark_period_returns : pd.Series
        Benchmark 21-day period returns indexed by date (for excess return calc)
    benchmark_daily_returns : pd.Series
        Benchmark daily returns indexed by date (for proper buy-and-hold cumulative)
    config : ResearchConfig
        Configuration
    benchmark_name : str
        Name of benchmark for display (default 'VT')
        
    Returns
    -------
    dict
        Benchmark comparison metrics
    """
    if len(results_df) == 0:
        return {}
    
    strategy_returns = results_df['ls_return'].dropna()
    
    # Get first and last dates of the strategy
    first_date = strategy_returns.index.min()
    last_date = strategy_returns.index.max()
    
    # Align period benchmark to strategy dates for excess return calculations
    strategy_dates = strategy_returns.index
    benchmark_aligned = benchmark_period_returns.reindex(strategy_dates, method='ffill')
    
    # Drop any remaining NaN
    valid_mask = strategy_returns.notna() & benchmark_aligned.notna()
    strategy_ret = strategy_returns[valid_mask]
    benchmark_ret = benchmark_aligned[valid_mask]
    
    if len(strategy_ret) < 3:
        return {'error': 'Insufficient overlapping data'}
    
    # Excess returns (using period returns - apples to apples)
    excess_ret = strategy_ret - benchmark_ret
    
    # Basic metrics
    mean_excess = excess_ret.mean()
    tracking_error = excess_ret.std()
    information_ratio = mean_excess / tracking_error if tracking_error > 0 else 0.0
    
    # Beta and Alpha (via regression)
    # R_strategy = alpha + beta * R_benchmark + epsilon
    cov = strategy_ret.cov(benchmark_ret)
    var_benchmark = benchmark_ret.var()
    beta = cov / var_benchmark if var_benchmark > 0 else 0.0
    
    # Alpha = mean(strategy) - beta * mean(benchmark)
    alpha = strategy_ret.mean() - beta * benchmark_ret.mean()
    
    # Correlation
    correlation = strategy_ret.corr(benchmark_ret)
    
    # Up/Down capture ratios
    up_periods = benchmark_ret > 0
    down_periods = benchmark_ret < 0
    
    if up_periods.sum() > 0:
        up_capture = strategy_ret[up_periods].mean() / benchmark_ret[up_periods].mean() if benchmark_ret[up_periods].mean() != 0 else np.nan
    else:
        up_capture = np.nan
    
    if down_periods.sum() > 0:
        down_capture = strategy_ret[down_periods].mean() / benchmark_ret[down_periods].mean() if benchmark_ret[down_periods].mean() != 0 else np.nan
    else:
        down_capture = np.nan
    
    # Annualization factors
    periods_per_year = 252 / config.time.HOLDING_PERIOD_DAYS
    annual_excess = mean_excess * periods_per_year
    annual_tracking_error = tracking_error * np.sqrt(periods_per_year)
    annual_ir = annual_excess / annual_tracking_error if annual_tracking_error > 0 else 0.0
    annual_alpha = alpha * periods_per_year
    
    # ===== FIXED: Proper buy-and-hold cumulative return using DAILY returns =====
    # The old calculation compounded non-overlapping 21-day returns on rebalance dates,
    # which missed most of the benchmark's actual performance between windows.
    # 
    # New approach: Use daily returns over the full backtest period for accurate
    # buy-and-hold comparison.
    if benchmark_daily_returns is not None and len(benchmark_daily_returns) > 0:
        # Get daily returns between first and last strategy dates
        daily_mask = (benchmark_daily_returns.index >= first_date) & (benchmark_daily_returns.index <= last_date)
        daily_rets_in_period = benchmark_daily_returns[daily_mask]
        
        if len(daily_rets_in_period) > 0:
            # Proper buy-and-hold cumulative return
            benchmark_total_buyhold = (1 + daily_rets_in_period).prod() - 1
            n_years = len(daily_rets_in_period) / 252
            benchmark_annual_buyhold = (1 + benchmark_total_buyhold) ** (1 / n_years) - 1 if n_years > 0 else 0
            benchmark_vol_daily = daily_rets_in_period.std() * np.sqrt(252)
            benchmark_sharpe_daily = (daily_rets_in_period.mean() * 252 - config.portfolio.cash_rate) / benchmark_vol_daily if benchmark_vol_daily > 0 else 0
        else:
            benchmark_total_buyhold = np.nan
            benchmark_annual_buyhold = np.nan
            benchmark_vol_daily = np.nan
            benchmark_sharpe_daily = np.nan
    else:
        # Fallback to period-based calculation (less accurate)
        benchmark_total_buyhold = (1 + benchmark_ret).prod() - 1
        benchmark_annual_buyhold = benchmark_ret.mean() * periods_per_year
        benchmark_vol_daily = benchmark_ret.std() * np.sqrt(periods_per_year)
        benchmark_sharpe_daily = (benchmark_ret.mean() - config.portfolio.cash_rate / periods_per_year) / benchmark_ret.std() * np.sqrt(periods_per_year) if benchmark_ret.std() > 0 else 0.0
    
    # Strategy cumulative return (already computed correctly from period returns)
    strategy_total = (1 + strategy_ret).prod() - 1
    n_periods = len(strategy_ret)
    strategy_annual = strategy_ret.mean() * periods_per_year
    
    stats = {
        f'{benchmark_name} Comparison': '---',
        f'Excess Return (vs {benchmark_name})': f"{mean_excess * 100:.2f}% per period",
        'Tracking Error': f"{tracking_error * 100:.2f}%",
        'Information Ratio': f"{information_ratio:.2f}",
        f'Beta to {benchmark_name}': f"{beta:.2f}",
        f'Alpha (vs {benchmark_name})': f"{alpha * 100:.3f}% per period",
        f'Correlation with {benchmark_name}': f"{correlation:.2f}",
        'Up Capture': f"{up_capture * 100:.1f}%" if not np.isnan(up_capture) else 'N/A',
        'Down Capture': f"{down_capture * 100:.1f}%" if not np.isnan(down_capture) else 'N/A',
        '--- Annualized ---': '',
        'Annual Excess Return': f"{annual_excess * 100:.2f}%",
        'Annual Tracking Error': f"{annual_tracking_error * 100:.2f}%",
        'Annual Information Ratio': f"{annual_ir:.2f}",
        'Annual Alpha': f"{annual_alpha * 100:.2f}%",
        f'--- {benchmark_name} Buy-and-Hold Reference (FIXED) ---': '',
        f'{benchmark_name} Total Return (Buy-Hold)': f"{benchmark_total_buyhold * 100:.2f}%",
        f'{benchmark_name} Annual Return (Buy-Hold)': f"{benchmark_annual_buyhold * 100:.2f}%",
        f'{benchmark_name} Annual Vol': f"{benchmark_vol_daily * 100:.2f}%",
        f'{benchmark_name} Sharpe': f"{benchmark_sharpe_daily:.2f}",
        '--- Strategy vs Benchmark ---': '',
        'Strategy Total Return': f"{strategy_total * 100:.2f}%",
        'Strategy Annual Return': f"{strategy_annual * 100:.2f}%",
        f'Strategy Outperformance (Total)': f"{(strategy_total - benchmark_total_buyhold) * 100:.2f}%",
    }
    
    return stats


def generate_trading_ledger(results_df: pd.DataFrame, config: ResearchConfig, output_path: str = None) -> pd.DataFrame:
    """
    Generate a detailed trading ledger with position-level details.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Backtest results from run_walk_forward_backtest
    config : ResearchConfig
        Configuration
    output_path : str, optional
        Path to save CSV output
        
    Returns
    -------
    pd.DataFrame
        Trading ledger with columns: date, ticker, side, weight, return, pnl, etc.
    """
    from pathlib import Path
    import ast
    
    ledger_rows = []
    cumulative_pnl = 0.0
    
    for idx, row in results_df.iterrows():
        date = idx
        ls_return = row['ls_return']
        
        # Parse long positions
        if 'long_tickers' in row and pd.notna(row['long_tickers']):
            try:
                if isinstance(row['long_tickers'], str):
                    long_tickers = ast.literal_eval(row['long_tickers'])
                else:
                    long_tickers = row['long_tickers']
                
                if isinstance(long_tickers, dict):
                    for ticker, weight in long_tickers.items():
                        # Estimate individual position return (assuming uniform contribution)
                        position_ret = row['long_ret'] * weight / (sum(long_tickers.values()) if long_tickers else 1.0)
                        position_pnl = position_ret
                        cumulative_pnl += position_pnl
                        
                        ledger_rows.append({
                            'date': date,
                            'ticker': ticker,
                            'side': 'LONG',
                            'weight': weight,
                            'portfolio_return': row['long_ret'],
                            'position_return': position_ret,
                            'position_pnl': position_pnl,
                            'cumulative_pnl': cumulative_pnl,
                            'ls_return': ls_return,
                            'transaction_cost': row.get('transaction_cost', 0.0),
                            'borrow_cost': row.get('borrow_cost', 0.0)
                        })
            except:
                pass
        
        # Parse short positions
        if 'short_tickers' in row and pd.notna(row['short_tickers']):
            try:
                if isinstance(row['short_tickers'], str):
                    short_tickers = ast.literal_eval(row['short_tickers'])
                else:
                    short_tickers = row['short_tickers']
                
                if isinstance(short_tickers, dict):
                    for ticker, weight in short_tickers.items():
                        # Estimate individual position return
                        position_ret = row['short_ret'] * abs(weight) / (sum(abs(w) for w in short_tickers.values()) if short_tickers else 1.0)
                        position_pnl = position_ret
                        cumulative_pnl += position_pnl
                        
                        ledger_rows.append({
                            'date': date,
                            'ticker': ticker,
                            'side': 'SHORT',
                            'weight': weight,
                            'portfolio_return': row['short_ret'],
                            'position_return': position_ret,
                            'position_pnl': position_pnl,
                            'cumulative_pnl': cumulative_pnl,
                            'ls_return': ls_return,
                            'transaction_cost': row.get('transaction_cost', 0.0),
                            'borrow_cost': row.get('borrow_cost', 0.0)
                        })
            except:
                pass
    
    ledger_df = pd.DataFrame(ledger_rows)
    
    if len(ledger_df) > 0:
        # Sort by date then side
        ledger_df = ledger_df.sort_values(['date', 'side', 'ticker'])
        
        # Save to CSV if path provided
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            ledger_df.to_csv(output_path, index=False)
            print(f"[save] Trading ledger saved to: {output_path}")
    
    return ledger_df


def print_enhanced_summary(results_df: pd.DataFrame, diagnostics: list, config: ResearchConfig):
    """
    Print enhanced performance summary with detailed metrics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Backtest results
    diagnostics : list
        List of diagnostic dictionaries from walk-forward backtest
    config : ResearchConfig
        Configuration
    """
    if len(results_df) == 0:
        print("[warn] No results to summarize")
        return
    
    print("\n" + "="*80)
    print("ENHANCED PERFORMANCE SUMMARY")
    print("="*80)
    
    # Standard metrics
    stats = analyze_performance(results_df, config)
    for key, value in stats.items():
        print(f"{key:25s}: {value}")
    
    # Position sizing metrics
    print("\n" + "-"*80)
    print("POSITION SIZING METRICS")
    print("-"*80)
    
    n_long_mean = results_df['n_long'].mean()
    n_short_mean = results_df['n_short'].mean()
    n_long_std = results_df['n_long'].std()
    n_short_std = results_df['n_short'].std()
    
    print(f"{'Avg Long Positions':25s}: {n_long_mean:.1f} ± {n_long_std:.1f}")
    print(f"{'Avg Short Positions':25s}: {n_short_mean:.1f} ± {n_short_std:.1f}")
    print(f"{'Avg Total Positions':25s}: {(n_long_mean + n_short_mean):.1f}")
    
    if 'gross_long' in results_df.columns:
        print(f"{'Avg Gross Long':25s}: {results_df['gross_long'].mean():.2%}")
    if 'gross_short' in results_df.columns:
        print(f"{'Avg Gross Short':25s}: {results_df['gross_short'].mean():.2%}")
    
    # Return attribution
    print("\n" + "-"*80)
    print("RETURN ATTRIBUTION")
    print("-"*80)
    
    total_return = (1 + results_df['ls_return']).prod() - 1
    long_contribution = results_df['long_ret'].sum()
    short_contribution = results_df['short_ret'].sum()
    
    # Calculate percentage contribution
    if abs(long_contribution + short_contribution) > 0:
        long_pct = long_contribution / (abs(long_contribution) + abs(short_contribution)) * 100
        short_pct = short_contribution / (abs(long_contribution) + abs(short_contribution)) * 100
    else:
        long_pct = 0
        short_pct = 0
    
    print(f"{'Total Return':25s}: {total_return * 100:.2f}%")
    print(f"{'Long Contribution':25s}: {long_contribution * 100:.2f}% ({long_pct:.1f}% of total)")
    print(f"{'Short Contribution':25s}: {short_contribution * 100:.2f}% ({short_pct:.1f}% of total)")
    
    # Cost analysis
    if 'transaction_cost' in results_df.columns:
        total_txn_cost = results_df['transaction_cost'].sum()
        print(f"{'Total Transaction Costs':25s}: {total_txn_cost * 100:.2f}%")
    
    if 'borrow_cost' in results_df.columns:
        total_borrow_cost = results_df['borrow_cost'].sum()
        print(f"{'Total Borrow Costs':25s}: {total_borrow_cost * 100:.2f}%")
    
    # Feature selection summary
    if diagnostics and len(diagnostics) > 0:
        print("\n" + "-"*80)
        print("FEATURE SELECTION SUMMARY")
        print("-"*80)
        
        feature_counts = {}
        universe_sizes = []
        
        for diag in diagnostics:
            # Count feature occurrences
            if 'selected_features' in diag:
                for feat in diag['selected_features']:
                    feature_counts[feat] = feature_counts.get(feat, 0) + 1
            
            # Track universe sizes
            if 'universe_size' in diag:
                universe_sizes.append(diag['universe_size'])
        
        # Top features by frequency
        top_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        print(f"{'Avg Universe Size':25s}: {np.mean(universe_sizes):.1f} ETFs (min: {min(universe_sizes)}, max: {max(universe_sizes)})")
        print(f"{'Avg Features Selected':25s}: {np.mean([diag['n_features'] for diag in diagnostics if 'n_features' in diag]):.1f}")
        print(f"\n{'Top 10 Features by Frequency':}")
        for i, (feat, count) in enumerate(top_features, 1):
            freq = count / len(diagnostics) * 100
            print(f"  {i:2d}. {feat:30s} - {count:2d}/{len(diagnostics)} periods ({freq:.1f}%)")
        
        # Sample IC values from last period
        if diagnostics[-1].get('ic_values'):
            print(f"\n{'Last Period IC Values (Top 5)':}")
            ic_dict = diagnostics[-1]['ic_values']
            abs_ic = {k: abs(v) for k, v in ic_dict.items() if pd.notna(v)}
            top_ic = sorted(abs_ic.items(), key=lambda x: x[1], reverse=True)[:5]
            for i, (feat, ic) in enumerate(top_ic, 1):
                actual_ic = ic_dict[feat]
                print(f"  {i}. {feat:30s}: IC = {actual_ic:+.4f}")
    
    print("\n" + "="*80)


if __name__ == "__main__":
    print("Walk-forward engine module loaded.")
    print("\nKey function:")
    print("  results_df = run_walk_forward_backtest(panel, metadata, config, model_type)")
    print("\nModel types:")
    print("  - 'momentum_rank': Simple baseline")
    print("  - 'supervised_binned': Supervised binning + feature selection")

