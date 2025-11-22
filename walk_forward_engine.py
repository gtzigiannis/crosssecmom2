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
import warnings

from config import ResearchConfig
from alpha_models import train_alpha_model, AlphaModel
from portfolio_construction import construct_portfolio, evaluate_portfolio_return
from regime import compute_regime_series, get_portfolio_mode_for_regime


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
    
    2. Train model on training window:
       model = train_alpha_model(panel, metadata, t_train_start, t_train_end, config)
    
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
            
            # Train model
            if verbose_inner:
                print(f"[train] Training {model_type} model...")
            
            train_start = time.time()
            try:
                model, selected_features, ic_series = train_alpha_model(
                    panel=panel_df,
                    universe_metadata=universe_metadata,
                    t_train_start=t_train_start,
                    t_train_end=t_train_end,
                    config=config,
                    model_type=model_type
                )
                
                diagnostics_entry = {
                    'date': t0,
                    'n_features': len(selected_features),
                    'selected_features': selected_features,
                    'ic_values': ic_series.to_dict() if ic_series is not None else {}
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
                    print(f"[regime] Current regime: {current_regime} → mode: {mode}")
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
        
        # Process results and compute capital path sequentially
        results = []
        diagnostics = []
        accounting_debug_rows = []
        capital_history = []
        current_capital = 1.0
        
        for i, result in enumerate(parallel_results):
            if result is None:
                continue
            
            performance = result['performance']
            
            # Update capital using decimal return
            period_return_decimal = performance['ls_return']
            current_capital *= (1.0 + period_return_decimal)
            performance['capital'] = current_capital
            capital_history.append(current_capital)
            
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
            # 2. Train model
            # =====================================================================
            if verbose:
                print(f"[train] Training {model_type} model...")
            
            train_start = time.time()
            try:
                model, selected_features, ic_series = train_alpha_model(
                    panel=panel_df,
                    universe_metadata=universe_metadata,
                    t_train_start=t_train_start,
                    t_train_end=t_train_end,
                    config=config,
                    model_type=model_type
                )
                
                # Record diagnostics
                diagnostics_entry = {
                    'date': t0,
                    'n_features': len(selected_features),
                    'selected_features': selected_features,
                    'ic_values': ic_series.to_dict() if ic_series is not None else {}
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
                    print(f"[regime] Current regime: {current_regime} → mode: {mode}")
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
                print(f"[OK] Capital tracking is consistent ✓")
    
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
        debug_path = config.paths.output_dir / "accounting_debug_log.csv"
        debug_df.to_csv(debug_path)
        if verbose:
            print(f"\n[debug] Accounting debug log saved to: {debug_path}")
            print(f"[debug] Logged {len(accounting_debug_rows)} periods")
    
    # Save diagnostics if requested
    if config.compute.save_intermediate and len(diagnostics) > 0:
        import json
        from pathlib import Path
        
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
    
    return results_df


def bootstrap_performance_stats(
    returns: pd.Series,
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
            sharpe = boot_mean / boot_std
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
    attribution['long_short'] = {
        'long_mean': results_df['long_ret'].mean(),
        'short_mean': results_df['short_ret'].mean(),
        'long_std': results_df['long_ret'].std(),
        'short_std': results_df['short_ret'].std(),
        'long_sharpe': results_df['long_ret'].mean() / results_df['long_ret'].std() if results_df['long_ret'].std() > 0 else 0,
        'short_sharpe': results_df['short_ret'].mean() / results_df['short_ret'].std() if results_df['short_ret'].std() > 0 else 0,
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
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
    
    # Annualization
    periods_per_year = 252 / config.time.HOLDING_PERIOD_DAYS
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


if __name__ == "__main__":
    print("Walk-forward engine module loaded.")
    print("\nKey function:")
    print("  results_df = run_walk_forward_backtest(panel, metadata, config, model_type)")
    print("\nModel types:")
    print("  - 'momentum_rank': Simple baseline")
    print("  - 'supervised_binned': Supervised binning + feature selection")
