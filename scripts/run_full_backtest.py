#!/usr/bin/env python
"""
Run Full Walk-Forward Backtest for Three Target Labels
=======================================================

This script runs the backtest separately for each target label with:
1. Full OOS windows (no limit)
2. VT as benchmark for performance comparison
3. Risk control via portfolio construction
4. Per-window feature selection tracking (coefficients, IC, significance)
5. Complete performance and risk metrics

Target labels:
1. y_raw_21d - Raw 21-day forward return
2. y_cs_21d - Cross-sectionally demeaned return  
3. y_resid_z_21d - Risk-adjusted z-score return

Usage:
    python scripts/run_full_backtest.py
"""

import sys
import os
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

# Set threading env vars before numpy import
for var in ('MKL_NUM_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 
            'NUMEXPR_NUM_THREADS', 'BLAS_NUM_THREADS', 'LAPACK_NUM_THREADS'):
    os.environ.setdefault(var, '1')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_default_config
from universe_metadata import validate_universe_metadata
from walk_forward_engine import run_walk_forward_backtest, analyze_performance, analyze_benchmark_comparison
from feature_selection_v4 import initialize_formation_cache, get_formation_cache


def load_benchmark_returns(panel_df, benchmark_ticker='VT', holding_period=21):
    """
    Load benchmark returns for comparison.
    
    Returns BOTH:
    - period_returns: Forward-looking period returns for apples-to-apples excess return comparison
    - daily_returns: Daily returns for proper buy-and-hold cumulative return calculation
    
    BUG FIX (2025-12-01): pct_change(H) is BACKWARD-looking (return from t-H to t).
    For benchmark comparison, we need FORWARD-looking returns (return from t to t+H)
    to match how strategy returns are computed.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data with (Date, Ticker) MultiIndex
    benchmark_ticker : str
        Ticker symbol for benchmark (default 'VT')
    holding_period : int
        Return holding period in days (default 21)
        
    Returns
    -------
    tuple (pd.Series, pd.Series)
        (period_returns, daily_returns) both indexed by date
    """
    # Check if benchmark is in the panel
    tickers = panel_df.index.get_level_values('Ticker').unique()
    if benchmark_ticker not in tickers:
        print(f"[warn] Benchmark {benchmark_ticker} not in panel, will compute from Close prices")
        # Try to load from OHLCV parquet
        from pathlib import Path
        ohlcv_path = Path('D:/REPOSITORY/Data/crosssecmom2/ohlcv') / f'{benchmark_ticker}.parquet'
        if ohlcv_path.exists():
            df = pd.read_parquet(ohlcv_path)
            if 'Date' in df.columns:
                df = df.set_index('Date').sort_index()
            # FORWARD-looking returns: Close[t+H] / Close[t] - 1
            period_returns = (df['Close'].shift(-holding_period) / df['Close'] - 1).dropna()
            daily_returns = df['Close'].pct_change(1).dropna()
            period_returns.name = benchmark_ticker
            daily_returns.name = benchmark_ticker
            return period_returns, daily_returns
        return None, None
    
    # Extract benchmark from panel
    benchmark_data = panel_df.xs(benchmark_ticker, level='Ticker')
    
    # Period returns (forward-looking) for excess return calculation
    fwd_ret_col = f'FwdRet_{holding_period}'
    if fwd_ret_col in benchmark_data.columns:
        # FwdRet_H should now be correctly computed as forward-looking
        period_returns = benchmark_data[fwd_ret_col].dropna()
    else:
        # Compute FORWARD-looking returns from Close
        # Close[t+H] / Close[t] - 1
        period_returns = (benchmark_data['Close'].shift(-holding_period) / benchmark_data['Close'] - 1).dropna()
    
    # Daily returns for proper buy-and-hold cumulative return
    if 'Close' in benchmark_data.columns:
        daily_returns = benchmark_data['Close'].pct_change(1).dropna()
    else:
        # Approximate from period returns (less accurate)
        daily_returns = period_returns / holding_period
    
    period_returns.name = benchmark_ticker
    daily_returns.name = benchmark_ticker
    return period_returns, daily_returns


def extract_per_window_feature_details(results_df):
    """
    Extract detailed feature selection info for each OOS window.
    
    Returns a list of dicts with per-window feature details:
    - Window date
    - Selected features with coefficients and IC
    - Model R², IC, residual std
    - Feature stability metrics
    """
    window_details = []
    
    if 'diagnostics' not in results_df.attrs:
        return window_details
    
    for i, diag in enumerate(results_df.attrs['diagnostics']):
        window_date = diag.get('date', '')
        
        # Feature details (from per_window_pipeline_v3)
        feature_details = diag.get('feature_details', [])
        
        # Model fit metrics
        model_fit = diag.get('model_fit', {})
        
        # Feature flow counts
        feature_flow = diag.get('feature_flow', {})
        
        # Formation diagnostics
        formation_artifacts = diag.get('formation_artifacts', {})
        formation_diag = formation_artifacts.get('formation_diagnostics', {})
        
        # LARS diagnostics
        lars_diag = diag.get('lars', {})
        
        window_info = {
            'window_idx': i + 1,
            'date': str(window_date)[:10] if window_date else '',
            
            # Model fit
            'r_squared': model_fit.get('r_squared', np.nan),
            'model_ic': model_fit.get('model_ic', np.nan),
            'residual_std': model_fit.get('residual_std', np.nan),
            'n_features': model_fit.get('n_features', 0),
            
            # Feature flow
            'n_formation_approved': formation_diag.get('n_approved', 0),
            'n_after_redundancy': formation_diag.get('n_after_redundancy', 0),
            'n_soft_ranking': feature_flow.get('after_soft_ranking', 0),
            'n_window_redundancy': feature_flow.get('after_redundancy', 0),
            'n_lars': feature_flow.get('after_lars', 0),
            'n_final': feature_flow.get('after_short_lag', 0),
            
            # LARS selection method
            'lars_method': lars_diag.get('selection_method', ''),
            'lars_best_alpha': lars_diag.get('best_alpha', np.nan),
            
            # Per-feature details (for stability tracking)
            'feature_details': feature_details,
        }
        
        window_details.append(window_info)
    
    return window_details


def print_feature_stability_analysis(window_details, target_name):
    """
    Analyze and print feature stability across OOS windows.
    """
    if not window_details:
        print("  No window details available")
        return
    
    # Track feature selection frequency
    feature_counts = {}
    feature_coefs = {}
    feature_ics = {}
    
    for w in window_details:
        for f in w.get('feature_details', []):
            fname = f['feature']
            coef = f['coefficient']
            ic = f['ic']
            
            if fname not in feature_counts:
                feature_counts[fname] = 0
                feature_coefs[fname] = []
                feature_ics[fname] = []
            
            feature_counts[fname] += 1
            feature_coefs[fname].append(coef)
            feature_ics[fname].append(ic)
    
    n_windows = len(window_details)
    
    # Sort by selection frequency
    sorted_features = sorted(feature_counts.items(), key=lambda x: -x[1])
    
    print(f"\n  Feature Stability Analysis ({n_windows} OOS windows):")
    print(f"  {'Feature':<50} {'Freq':>6} {'%':>6} {'Avg Coef':>10} {'Std Coef':>10} {'Avg IC':>8}")
    print(f"  {'-'*50} {'-'*6} {'-'*6} {'-'*10} {'-'*10} {'-'*8}")
    
    for fname, count in sorted_features[:20]:  # Top 20
        pct = 100 * count / n_windows
        avg_coef = np.mean(feature_coefs[fname])
        std_coef = np.std(feature_coefs[fname]) if len(feature_coefs[fname]) > 1 else 0
        avg_ic = np.mean(feature_ics[fname])
        
        # Truncate long feature names
        display_name = fname[:47] + '...' if len(fname) > 50 else fname
        print(f"  {display_name:<50} {count:>6} {pct:>5.1f}% {avg_coef:>+10.4f} {std_coef:>10.4f} {avg_ic:>+8.3f}")
    
    # Summary statistics
    if feature_counts:
        unique_features = len(feature_counts)
        avg_features_per_window = np.mean([w['n_final'] for w in window_details])
        core_features = sum(1 for f, c in feature_counts.items() if c >= n_windows * 0.5)  # Present in 50%+ windows
        
        print(f"\n  Summary:")
        print(f"    Total unique features selected: {unique_features}")
        print(f"    Average features per window: {avg_features_per_window:.1f}")
        print(f"    Core features (>=50% windows): {core_features}")


def run_single_backtest(panel_df, universe_metadata, benchmark_period_returns, benchmark_daily_returns,
                        target_col, config=None, n_windows=None, skip_windows=0, verbose=True):
    """
    Run a single backtest with a specific target column.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel data
    universe_metadata : pd.DataFrame  
        ETF metadata
    benchmark_period_returns : pd.Series
        VT 21-day returns for excess return comparison
    benchmark_daily_returns : pd.Series
        VT daily returns for proper buy-and-hold cumulative return
    target_col : str
        Target column name
    config : StrategyConfig or None
        Configuration object. If None, uses default config.
    n_windows : int or None
        Number of windows (None = all available)
    skip_windows : int
        Number of initial windows to skip
    verbose : bool
        Print progress
    """
    if config is None:
        config = get_default_config()
    cfg = copy.deepcopy(config)
    
    # Set target column
    cfg.target.target_column = target_col
    
    # Full OOS windows (no debug limit unless specified)
    if n_windows is not None:
        cfg.compute.max_rebalance_dates_for_debug = n_windows + skip_windows
    else:
        cfg.compute.max_rebalance_dates_for_debug = None  # Run all windows
    
    # Skip early windows with NaN data
    cfg.compute.skip_rebalance_dates = skip_windows
    
    # Enable accounting debug for detailed tracking
    cfg.debug.enable_accounting_debug = True
    cfg.debug.debug_max_periods = 0  # No limit on debug periods
    
    # Validate metadata
    meta = validate_universe_metadata(universe_metadata.copy(), cfg)
    
    if verbose:
        print(f"\n{'='*80}")
        print(f"BACKTEST: {target_col}")
        print(f"{'='*80}")
        print(f"Target column: {cfg.target.target_column}")
        print(f"Windows: {'ALL' if n_windows is None else n_windows} (skipping first {skip_windows})")
        print(f"Holding period: {cfg.time.HOLDING_PERIOD_DAYS} days")
        print(f"Transaction costs: {cfg.portfolio.total_cost_bps_per_side:.1f} bps per side")
    
    # Run backtest (use 'simple' portfolio method to avoid cvxpy dependency)
    results_df = run_walk_forward_backtest(
        panel_df=panel_df,
        universe_metadata=meta,
        config=cfg,
        model_type='supervised_binned',
        portfolio_method='simple',  # Use simple method - cvxpy requires C++ build tools
        verbose=verbose
    )
    
    if results_df is None or len(results_df) == 0:
        print(f"[error] No results for {target_col}")
        return None, None, None, None
    
    # Get performance metrics
    perf = analyze_performance(results_df, cfg)
    
    # Get benchmark comparison metrics
    bench_stats = {}
    if benchmark_period_returns is not None:
        bench_stats = analyze_benchmark_comparison(
            results_df, 
            benchmark_period_returns, 
            benchmark_daily_returns,
            cfg, 
            'VT'
        )
    
    # Extract per-window feature details
    window_details = extract_per_window_feature_details(results_df)
    
    return results_df, perf, bench_stats, window_details


def main():
    """Main entry point."""
    start_time = datetime.now()
    print(f"\n{'='*80}")
    print(f"FULL BACKTEST - LONG-ONLY with y_cs_21d")
    print(f"Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}")
    
    # Load config and data
    config = get_default_config()
    
    # ========================================================================
    # BACKTEST MODE CONFIGURATION
    # Toggle these settings to switch between L/S and Long-Only modes
    # ========================================================================
    config.portfolio.long_only = True   # Set True for long-only, False for L/S
    # Note: When long_only=True, short positions are replaced with cash
    print(f"\n[Mode] Portfolio mode: {'LONG-ONLY' if config.portfolio.long_only else 'LONG/SHORT'}")
    
    print(f"\nLoading panel from {config.paths.panel_parquet}", flush=True)
    panel_df = pd.read_parquet(config.paths.panel_parquet)
    print(f"Panel shape: {panel_df.shape}")
    print(f"Date range: {panel_df.index.get_level_values('Date').min()} to {panel_df.index.get_level_values('Date').max()}")
    
    print(f"\nLoading metadata from {config.paths.universe_metadata_output}", flush=True)
    universe_metadata = pd.read_csv(config.paths.universe_metadata_output)
    print(f"Universe size: {len(universe_metadata)} ETFs")
    
    # Check for target columns
    target_cols = ['y_raw_21d', 'y_cs_21d', 'y_resid_21d', 'y_resid_z_21d']
    if not all(c in panel_df.columns for c in target_cols):
        print(f"\nComputing target labels...", flush=True)
        from label_engineering import compute_targets
        panel_df = compute_targets(panel_df, config, raw_return_col='FwdRet_21')
        print(f"Target labels computed", flush=True)
    
    # Load VT benchmark returns
    print(f"\nLoading VT benchmark returns...", flush=True)
    benchmark_period_returns, benchmark_daily_returns = load_benchmark_returns(panel_df, 'VT', config.time.HOLDING_PERIOD_DAYS)
    if benchmark_period_returns is not None:
        print(f"VT period returns: {len(benchmark_period_returns)} periods from {benchmark_period_returns.index.min()} to {benchmark_period_returns.index.max()}")
        print(f"VT daily returns: {len(benchmark_daily_returns)} days from {benchmark_daily_returns.index.min()} to {benchmark_daily_returns.index.max()}")
    else:
        print("[warn] VT benchmark not available")
    
    # Store all results
    all_results = {}
    
    # ========================================================================
    # TARGET LABEL CONFIGURATION
    # Available targets:
    #   - 'y_raw_21d': Raw 21-day forward return (best for long-only total return)
    #   - 'y_cs_21d': Cross-sectional rank (best for long-only relative winners)
    #   - 'y_resid_21d': Risk-adjusted residual (market-neutral alpha)
    #   - 'y_resid_z_21d': Z-scored residual (best for L/S market-neutral)
    # ========================================================================
    targets_to_run = ['y_cs_21d']  # Using cross-sectional label for long-only
    
    # Run backtests for each target
    for target in targets_to_run:
        # Initialize formation cache for this target (computes IC once for all windows)
        print(f"\n[Cache] Initializing formation interaction cache for {target}...")
        initialize_formation_cache(panel_df, target, config)
        
        results_df, perf, bench_stats, window_details = run_single_backtest(
            panel_df=panel_df,
            universe_metadata=universe_metadata,
            benchmark_period_returns=benchmark_period_returns,
            benchmark_daily_returns=benchmark_daily_returns,
            target_col=target,
            config=config,  # Pass config with long_only setting
            n_windows=None,  # Run ALL windows
            skip_windows=0,  # Don't skip any
            verbose=True
        )
        
        # Clear cache to free memory before next target
        get_formation_cache().clear()
        
        if results_df is not None:
            all_results[target] = {
                'results_df': results_df,
                'performance': perf,
                'benchmark_stats': bench_stats,
                'window_details': window_details,
            }
    
    # ========================================================================
    # SUMMARY OUTPUT
    # ========================================================================
    print("\n" + "="*100)
    print("SUMMARY: THREE TARGET LABELS - FULL OOS WINDOWS")
    print("="*100)
    
    # Performance metrics table
    print("\n┌" + "─"*98 + "┐")
    print(f"│{'PERFORMANCE METRICS':^98}│")
    print("├" + "─"*17 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*12 + "┬" + "─"*10 + "┬" + "─"*10 + "┬" + "─"*10 + "┬" + "─"*11 + "┤")
    print(f"│ {'Target':<15} │ {'L/S Total':>10} │ {'Long Leg':>10} │ {'Short Leg':>10} │ {'Sharpe':>8} │ {'Max DD':>8} │ {'Win Rate':>8} │ {'# Windows':>9} │")
    print("├" + "─"*17 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*12 + "┼" + "─"*10 + "┼" + "─"*10 + "┼" + "─"*10 + "┼" + "─"*11 + "┤")
    
    for target, data in all_results.items():
        df = data['results_df']
        perf = data['performance']
        
        ls_total = df['ls_return'].sum() * 100
        long_total = df['long_ret'].sum() * 100
        short_total = df['short_ret'].sum() * 100
        n_windows = len(df)
        
        # Extract values from performance dict (they have formatting)
        sharpe = perf.get('Sharpe Ratio', 'N/A')
        max_dd = perf.get('Max Drawdown', 'N/A')
        win_rate = perf.get('Win Rate', 'N/A')
        
        print(f"│ {target:<15} │ {ls_total:>+10.2f}% │ {long_total:>+10.2f}% │ {short_total:>+10.2f}% │ {sharpe:>8} │ {max_dd:>8} │ {win_rate:>8} │ {n_windows:>9} │")
    
    print("└" + "─"*17 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*12 + "┴" + "─"*10 + "┴" + "─"*10 + "┴" + "─"*10 + "┴" + "─"*11 + "┘")
    
    # VT Benchmark comparison
    print("\n" + "─"*100)
    print("VT BENCHMARK COMPARISON")
    print("─"*100)
    
    for target, data in all_results.items():
        bench_stats = data.get('benchmark_stats', {})
        if bench_stats:
            print(f"\n{target}:")
            for key, val in bench_stats.items():
                if key.startswith('---'):
                    print(f"  {key}")
                elif val == '---':
                    print(f"  {key}:")
                elif val:
                    print(f"    {key}: {val}")
    
    # Cost/Fee impact
    print("\n" + "─"*100)
    print("TRANSACTION COST IMPACT")
    print("─"*100)
    
    for target, data in all_results.items():
        df = data['results_df']
        if 'transaction_cost' in df.columns:
            total_txn_cost = df['transaction_cost'].sum() * 100
            avg_turnover = df['turnover'].mean() * 100 if 'turnover' in df.columns else 0
            total_borrow_cost = df['borrow_cost'].sum() * 100 if 'borrow_cost' in df.columns else 0
            
            print(f"\n{target}:")
            print(f"  Total transaction costs: {total_txn_cost:.2f}%")
            print(f"  Average turnover: {avg_turnover:.1f}%")
            print(f"  Total borrow costs: {total_borrow_cost:.2f}%")
    
    # Per-window feature details
    print("\n" + "="*100)
    print("PER-WINDOW FEATURE SELECTION DETAILS")
    print("="*100)
    
    for target, data in all_results.items():
        window_details = data.get('window_details', [])
        print(f"\n{'-'*80}")
        print(f"{target}")
        print(f"{'-'*80}")
        
        if window_details:
            print(f"\n  Window   Date         R²       IC       n_feat   LARS Method        Alpha")
            print(f"  ------   ----------   ------   ------   ------   ----------------   --------")
            for w in window_details[:20]:  # First 20 windows
                print(f"  {w['window_idx']:^6}   {w['date']:10}   {w['r_squared']:>6.4f}   {w['model_ic']:>+6.3f}   {w['n_final']:>6}   {w['lars_method']:<16}   {w['lars_best_alpha']:.2e}")
            
            if len(window_details) > 20:
                print(f"  ... ({len(window_details) - 20} more windows)")
            
            # Feature stability analysis
            print_feature_stability_analysis(window_details, target)
        else:
            print("  No window details available")
    
    # Timing
    end_time = datetime.now()
    elapsed = end_time - start_time
    print(f"\n{'='*80}")
    print(f"Completed: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total time: {elapsed}")
    print(f"{'='*80}")
    
    # ========================================================================
    # SAVE RESULTS TO DISK
    # ========================================================================
    output_dir = Path(config.paths.data_dir)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print(f"\n[Save] Saving results to {output_dir}...")
    
    # Save each target's results
    for target, data in all_results.items():
        # Save results_df (without attrs that can cause deepcopy issues)
        results_df = data['results_df'].copy()
        if hasattr(results_df, 'attrs'):
            results_df.attrs = {}  # Clear attrs to avoid deepcopy issues
        
        # Drop columns with complex types that parquet can't handle
        cols_to_drop = ['cluster_exposures']  # Empty dicts cause Arrow errors
        for col in cols_to_drop:
            if col in results_df.columns:
                results_df = results_df.drop(columns=[col])
        
        results_path = output_dir / f'backtest_results_{target}_{timestamp}.parquet'
        results_df.to_parquet(results_path)
        print(f"[Save] {target} results: {results_path}")
        
        # Save window details as JSON
        window_details = data.get('window_details', [])
        if window_details:
            import json
            details_path = output_dir / f'window_details_{target}_{timestamp}.json'
            with open(details_path, 'w') as f:
                json.dump(window_details, f, indent=2, default=str)
            print(f"[Save] {target} window details: {details_path}")
        
        # Save performance summary as CSV
        perf = data.get('performance', {})
        if perf:
            perf_df = pd.DataFrame([{k: str(v) if not isinstance(v, (int, float)) else v 
                                      for k, v in perf.items()}])
            perf_path = output_dir / f'performance_{target}_{timestamp}.csv'
            perf_df.to_csv(perf_path, index=False)
            print(f"[Save] {target} performance: {perf_path}")
    
    # Save summary across all targets
    summary_rows = []
    for target, data in all_results.items():
        df = data['results_df']
        perf = data.get('performance', {})
        summary_rows.append({
            'target': target,
            'n_windows': len(df),
            'ls_total_pct': df['ls_return'].sum() * 100,
            'long_total_pct': df['long_ret'].sum() * 100,
            'short_total_pct': df['short_ret'].sum() * 100,
            'sharpe': perf.get('sharpe_ratio', 'N/A'),
            'max_drawdown_pct': perf.get('max_drawdown', 'N/A'),
            'win_rate_pct': perf.get('win_rate', 'N/A'),
        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_path = output_dir / f'backtest_summary_{timestamp}.csv'
    summary_df.to_csv(summary_path, index=False)
    print(f"[Save] Summary: {summary_path}")
    
    print(f"\n[Save] All results saved to {output_dir}")
    
    return all_results


if __name__ == "__main__":
    results = main()
