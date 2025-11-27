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
from feature_selection import per_window_pipeline
from portfolio_construction import construct_portfolio, evaluate_portfolio_return
from regime import compute_regime_series, get_portfolio_mode_for_regime
from attribution_analysis import compute_attribution_analysis, save_attribution_results


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
                print(f"[universe] PHASE 0: Equity-only filter enabled, reduced universe {n_before} → {n_after} tickers")
                sys._equity_filter_printed = True
    
    # 2. ADV liquidity filter
    if 'ADV_63_Rank' in cross_section.columns:
        adv_filter = cross_section['ADV_63_Rank'] >= config.universe.min_adv_percentile
        tickers = tickers[adv_filter[tickers]]
    
    # 3. Data quality filter
    feature_cols = [c for c in cross_section.columns 
                   if c not in ['Close', 'Ticker', 'ADV_63', 'ADV_63_Rank'] 
                   and not c.startswith('FwdRet')]
    
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
       model, diagnostics = per_window_pipeline(panel_train, metadata, t0, config)
    
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
    # Parallel vs Sequential Execution
    # =====================================================================
    if config.compute.parallelize_backtest:
        # PARALLEL EXECUTION: Process rebalance dates in parallel
        from joblib import Parallel, delayed
        
        if verbose:
            print(f"\n[parallel] Processing {len(current_dates)} rebalance dates in parallel...")
            print(f"[parallel] Using {config.compute.n_jobs} jobs")
        
        def process_rebalance_period(i, t0, panel_df, universe_metadata, config, model_type, portfolio_method, regime_series, verbose_inner):
            """Process a single rebalance period (for parallel execution)."""
            import time
            import pandas as pd
            import numpy as np
            
            rebalance_start = time.time()
            if verbose_inner:
                print(f"\n{'='*60}")
                print(f"[{i+1}/{len(current_dates)}] Rebalance date: {t0.date()}")
            
            # Define training window
            t_train_start = t0 - pd.Timedelta(days=config.time.TRAINING_WINDOW_DAYS)
            t_train_end = t0 - pd.Timedelta(days=1 + config.time.HOLDING_PERIOD_DAYS)
            
            if verbose_inner:
                print(f"Training window: [{t_train_start.date()}, {t_train_end.date()}]")
            
            if t_train_end <= t_train_start:
                if verbose_inner:
                    print("[skip] Invalid training window")
                return None
            
            # Train model using per_window_pipeline
            if verbose_inner:
                print(f"[train] Training {model_type} model with feature selection...")
            
            train_start = time.time()
            try:
                # Extract training panel
                train_mask = (
                    (panel_df.index.get_level_values('Date') >= t_train_start) &
                    (panel_df.index.get_level_values('Date') <= t_train_end)
                )
                panel_train = panel_df[train_mask]
                
                # Train model with feature selection (returns model + diagnostics)
                model, all_diagnostics = per_window_pipeline(
                    panel=panel_train,
                    metadata=universe_metadata,
                    t0=t0,
                    config=config
                )
                
                # Extract diagnostics
                diagnostics_entry = {
                    'date': t0,
                    'n_features': len(model.selected_features) if model else 0,
                    'selected_features': model.selected_features if model else [],
                    **all_diagnostics  # Include stage counts, timings, etc.
                }
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
        
        # Execute in parallel
        parallel_results = Parallel(n_jobs=config.compute.n_jobs, backend='loky')(
            delayed(process_rebalance_period)(
                i, t0, panel_df, universe_metadata, config, model_type, 
                portfolio_method, regime_series, False  # verbose_inner=False for parallel
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
            # 1. Define training window
            # =====================================================================
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
            # 2. Train model with feature selection
            # =====================================================================
            if verbose:
                print(f"[train] Training {model_type} model with feature selection...")
            
            train_start = time.time()
            try:
                # Extract training panel
                train_mask = (
                    (panel_df.index.get_level_values('Date') >= t_train_start) &
                    (panel_df.index.get_level_values('Date') <= t_train_end)
                )
                panel_train = panel_df[train_mask]
                
                # Train model with feature selection (returns model + diagnostics)
                model, all_diagnostics = per_window_pipeline(
                    panel=panel_train,
                    metadata=universe_metadata,
                    t0=t0,
                    config=config
                )
                
                # Record diagnostics
                diagnostics_entry = {
                    'date': t0,
                    'n_features': len(model.selected_features) if model else 0,
                    'selected_features': model.selected_features if model else [],
                    **all_diagnostics  # Include stage counts, timings, etc.
                }
                train_elapsed = time.time() - train_start
                total_train_time += train_elapsed
                
            except Exception as e:
                if verbose:
                    print(f"[error] Model training failed: {e}")
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
            # Get top 5 features by |IC|
            if entry['ic_values']:
                ic_dict = entry['ic_values']
                abs_ic = {k: abs(v) for k, v in ic_dict.items() if pd.notna(v)}
                top_features = sorted(abs_ic.items(), key=lambda x: x[1], reverse=True)[:5]
                top_features_str = '; '.join([f"{k}={v:.3f}" for k, v in top_features])
            else:
                top_features_str = ""
            
            diag_summary.append({
                'date': entry['date'],
                'universe_size': entry['universe_size'],
                'n_features': entry['n_features'],
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
                json_entry = {
                    'date': entry['date'].isoformat(),
                    'universe_size': entry['universe_size'],
                    'n_features': entry['n_features'],
                    'selected_features': entry['selected_features'],
                    'ic_values': entry['ic_values']
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
    
    for i in range(n_bootstrap):
        # Sample blocks with replacement (block bootstrap)
        sampled_blocks = np.random.choice(n_blocks, size=n_blocks, replace=True)
        
        bootstrap_returns = []
        for block_idx in sampled_blocks:
            start = block_idx * block_size
            end = min(start + block_size, len(returns))
            bootstrap_returns.extend(returns.iloc[start:end].values)
        
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
    cluster_attribution = {}
    target_col = f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}'
    
    if target_col in panel_df.columns and 'cluster_id' in universe_metadata.columns:
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
                        
                        # Get actual return
                        if (t0, ticker) in panel_df.index:
                            ret = panel_df.loc[(t0, ticker), target_col]
                            
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
    feature_ic = {}
    
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
    Compute performance statistics.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Backtest results from run_walk_forward_backtest
    config : ResearchConfig
        Configuration
        
    Returns
    -------
    dict
        Performance metrics
    """
    if len(results_df) == 0:
        return {}
    
    returns = results_df['ls_return'].dropna()
    
    if len(returns) == 0:
        return {}
    
    # Basic statistics
    total_periods = len(returns)
    winning_periods = (returns > 0).sum()
    win_rate = winning_periods / total_periods
    
    # Return statistics (returns are now in decimal form)
    mean_ret = returns.mean()
    std_ret = returns.std()
    
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
    
    # Drawdown
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max
    max_dd = drawdown.min()
    
    # Long/Short breakdown
    long_mean = results_df['long_ret'].mean()
    short_mean = results_df['short_ret'].mean()
    
    # Bootstrap confidence intervals
    bootstrap_stats = bootstrap_performance_stats(
        returns,
        config,
        n_bootstrap=1000,
        block_size=6,
        random_state=config.features.random_state if hasattr(config.features, 'random_state') else 42
    )
    
    stats = {
        'Total Periods': total_periods,
        'Win Rate': f"{win_rate:.2%}",
        'Mean Return': f"{mean_ret * 100:.2f}%",  # Convert decimal to percent for display
        'Std Dev': f"{std_ret * 100:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Annual Return': f"{annual_return * 100:.2f}%",
        'Annual Volatility': f"{annual_vol * 100:.2f}%",
        'Annual Sharpe': f"{annual_sharpe:.2f}",
        'Total Return': f"{total_return * 100:.2f}%",
        'Max Drawdown': f"{max_dd * 100:.2f}%",
        'Long Avg': f"{long_mean * 100:.2f}%",
        'Short Avg': f"{short_mean * 100:.2f}%",
        # Bootstrap confidence intervals
        'Sharpe (Bootstrap Mean)': f"{bootstrap_stats['sharpe_mean']:.2f}",
        'Sharpe 90% CI': f"[{bootstrap_stats['sharpe_ci_5']:.2f}, {bootstrap_stats['sharpe_ci_95']:.2f}]",
        'Return (Bootstrap Mean)': f"{bootstrap_stats['return_mean'] * 100:.2f}%",
        'Return 90% CI': f"[{bootstrap_stats['return_ci_5'] * 100:.2f}%, {bootstrap_stats['return_ci_95'] * 100:.2f}%]",
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

