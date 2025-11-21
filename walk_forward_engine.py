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
    portfolio_method: str = 'simple',  # Default to 'simple' for robustness
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
        print(f"Borrow cost: {config.portfolio.borrow_cost:.1%} annualized")
        print(f"Random state: {config.features.random_state}")
    
    # Validate portfolio method
    from portfolio_construction import CVXPY_AVAILABLE
    if portfolio_method == 'cvxpy' and not CVXPY_AVAILABLE:
        if verbose:
            print(f"[warn] cvxpy not available, falling back to 'simple' method")
        portfolio_method = 'simple'
    elif portfolio_method not in ['cvxpy', 'simple']:
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
    
    results = []
    diagnostics = []  # Store diagnostics for each rebalance period
    
    # Track previous weights for turnover calculation
    prev_long_weights = None
    prev_short_weights = None
    
    # Diagnostic tracking
    ic_history = []  # Store IC vectors for each window
    feature_selection_history = []  # Track which features were selected
    universe_size_history = []  # Track universe size after each filter
    
    for i, t0 in enumerate(current_dates):
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
        
        try:
            scores = model.score_at_date(
                panel=panel_df,
                t0=t0,
                universe_metadata=universe_metadata,
                config=config
            )
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
        
        try:
            long_weights, short_weights, portfolio_stats = construct_portfolio(
                scores=scores,
                universe_metadata=eligible_metadata,
                config=config,
                method=portfolio_method,
                mode=mode
            )
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
        except Exception as e:
            if verbose:
                print(f"[error] Evaluation failed: {e}")
            continue
        
        # Add portfolio stats to performance
        performance.update(portfolio_stats)
        
        if verbose:
            ret_str = f"[result] Long: {performance['long_ret']:.2%}, "
            ret_str += f"Short: {performance['short_ret']:.2%}, "
            ret_str += f"L/S: {performance['ls_return']:.2%}"
            if performance.get('turnover', 0) > 0:
                ret_str += f", Turnover: {performance['turnover']:.2%}"
            if performance.get('transaction_cost', 0) > 0:
                ret_str += f", TxnCost: {performance['transaction_cost']:.4%}"
            print(ret_str)
        
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
    
    if verbose:
        print("\n" + "="*80)
        print("BACKTEST COMPLETE")
        print("="*80)
        print(f"Total periods: {len(results_df)}")
        print(f"Date range: {results_df.index[0].date()} to {results_df.index[-1].date()}")
    
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
    
    # Return statistics
    mean_ret = returns.mean()
    std_ret = returns.std()
    sharpe = mean_ret / std_ret if std_ret > 0 else 0.0
    
    # Annualization
    periods_per_year = 252 / config.time.HOLDING_PERIOD_DAYS
    annual_return = mean_ret * periods_per_year
    annual_vol = std_ret * np.sqrt(periods_per_year)
    annual_sharpe = sharpe * np.sqrt(periods_per_year)
    
    # Cumulative return
    cum_ret = (1 + returns / 100).cumprod()
    total_return = (cum_ret.iloc[-1] - 1) * 100
    
    # Drawdown
    running_max = cum_ret.expanding().max()
    drawdown = (cum_ret - running_max) / running_max * 100
    max_dd = drawdown.min()
    
    # Long/Short breakdown
    long_mean = results_df['long_ret'].mean()
    short_mean = results_df['short_ret'].mean()
    
    stats = {
        'Total Periods': total_periods,
        'Win Rate': f"{win_rate:.2%}",
        'Mean Return': f"{mean_ret:.2f}%",
        'Std Dev': f"{std_ret:.2f}%",
        'Sharpe Ratio': f"{sharpe:.2f}",
        'Annual Return': f"{annual_return:.2f}%",
        'Annual Volatility': f"{annual_vol:.2f}%",
        'Annual Sharpe': f"{annual_sharpe:.2f}",
        'Total Return': f"{total_return:.2f}%",
        'Max Drawdown': f"{max_dd:.2f}%",
        'Long Avg': f"{long_mean:.2f}%",
        'Short Avg': f"{short_mean:.2f}%",
    }
    
    return stats


if __name__ == "__main__":
    print("Walk-forward engine module loaded.")
    print("\nKey function:")
    print("  results_df = run_walk_forward_backtest(panel, metadata, config, model_type)")
    print("\nModel types:")
    print("  - 'momentum_rank': Simple baseline")
    print("  - 'supervised_binned': Supervised binning + feature selection")
