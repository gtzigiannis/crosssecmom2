"""
Main Entry Point for Cross-Sectional Momentum Research
=======================================================
Complete workflow demonstrating the refactored framework.

Usage:
    python main.py --step [feature_eng|build_metadata|backtest|analyze]
    
    Or run all steps:
    python main.py --all
"""

import argparse
import pandas as pd
from pathlib import Path

from config import get_default_config, ResearchConfig
from universe_metadata import build_universe_metadata
from feature_engineering_refactored import run_feature_engineering
from walk_forward_engine import run_walk_forward_backtest, analyze_performance


def step_1_feature_engineering(config: ResearchConfig):
    """
    Step 1: Generate panel data with raw features.
    
    Output: panel_parquet with (Date, Ticker) MultiIndex containing:
    - Close (raw price)
    - Raw features (returns, momentum, volatility, etc.)
    - ADV_63, ADV_63_Rank (liquidity)
    - FwdRet_H (forward returns at HOLDING_PERIOD_DAYS)
    """
    print("\n" + "="*80)
    print("STEP 1: FEATURE ENGINEERING")
    print("="*80)
    
    panel_df = run_feature_engineering(config)
    
    print(f"\n[done] Panel saved to: {config.paths.panel_parquet}")
    return panel_df


def step_2_build_metadata(config: ResearchConfig, returns_df=None):
    """
    Step 2: Build universe metadata (families, duplicates, clusters, caps).
    
    Output: universe_metadata DataFrame with:
    - family: Economic family classification
    - dup_group_id: Duplicate group ID (if applicable)
    - is_dup_canonical: True for canonical ETF in duplicate group
    - in_core_after_duplicates: True if in core universe
    - cluster_id: Theme cluster ID
    - cluster_cap: Max weight per cluster
    - per_etf_cap: Max weight per ETF
    """
    print("\n" + "="*80)
    print("STEP 2: BUILD UNIVERSE METADATA")
    print("="*80)
    
    # Check if universe metadata CSV exists
    if not Path(config.paths.universe_metadata_csv).exists():
        print(f"[error] Universe metadata file not found: {config.paths.universe_metadata_csv}")
        print("[info] Using simple universe file without families/clusters")
        
        # Load simple universe and create basic metadata
        universe_df = pd.read_csv(config.paths.universe_csv)
        universe_metadata = pd.DataFrame({
            'ticker': universe_df['ticker'],
            'name': universe_df.get('name', ''),
            'family': 'UNKNOWN',
            'in_core_universe': True,
            'dup_group_id': pd.NA,
            'is_dup_canonical': False,
            'in_core_after_duplicates': True,
            'cluster_id': pd.NA,
            'cluster_cap': config.portfolio.default_cluster_cap,
            'per_etf_cap': config.portfolio.default_per_etf_cap,
        })
        
        cluster_caps = pd.Series(dtype=float)
    
    else:
        # Full metadata build with families, duplicates, clusters
        universe_metadata, cluster_caps = build_universe_metadata(
            meta_path=config.paths.universe_metadata_csv,
            returns_df=returns_df,
            dup_corr_threshold=config.universe.dup_corr_threshold,
            max_within_cluster_corr=config.universe.max_within_cluster_corr,
            default_cluster_cap=config.portfolio.default_cluster_cap,
            default_per_etf_cap=config.portfolio.default_per_etf_cap,
            high_risk_cluster_cap=config.portfolio.high_risk_cluster_cap,
            high_risk_families=config.portfolio.high_risk_families,
        )
    
    # Save metadata
    universe_metadata.to_csv(config.paths.universe_metadata_output, index=False)
    print(f"\n[save] Universe metadata saved to: {config.paths.universe_metadata_output}")
    
    # Summary
    print("\n" + "="*80)
    print("METADATA SUMMARY")
    print("="*80)
    print(f"Total ETFs: {len(universe_metadata)}")
    print(f"Core universe: {universe_metadata['in_core_universe'].sum()}")
    print(f"After duplicate removal: {universe_metadata['in_core_after_duplicates'].sum()}")
    
    if 'family' in universe_metadata.columns:
        print("\nTop families:")
        print(universe_metadata['family'].value_counts().head(10))
    
    if 'cluster_id' in universe_metadata.columns and universe_metadata['cluster_id'].notna().any():
        print(f"\nNumber of theme clusters: {universe_metadata['cluster_id'].nunique()}")
        print(f"Cluster caps range: [{cluster_caps.min():.2%}, {cluster_caps.max():.2%}]")
    
    return universe_metadata, cluster_caps


def step_3_backtest(
    config: ResearchConfig,
    panel_df: pd.DataFrame = None,
    universe_metadata: pd.DataFrame = None,
    model_type: str = 'supervised_binned'
):
    """
    Step 3: Run walk-forward backtest.
    
    At each rebalance date:
    1. Train model on training window
    2. Score eligible universe
    3. Construct portfolio with caps
    4. Evaluate using forward returns
    """
    print("\n" + "="*80)
    print("STEP 3: WALK-FORWARD BACKTEST")
    print("="*80)
    
    # Load data if not provided
    if panel_df is None:
        print(f"[load] Loading panel from {config.paths.panel_parquet}")
        panel_df = pd.read_parquet(config.paths.panel_parquet)
    
    if universe_metadata is None:
        print(f"[load] Loading universe metadata from {config.paths.universe_metadata_output}")
        universe_metadata = pd.read_csv(config.paths.universe_metadata_output)
    
    # Run backtest
    results_df = run_walk_forward_backtest(
        panel_df=panel_df,
        universe_metadata=universe_metadata,
        config=config,
        model_type=model_type,
        portfolio_method='simple',  # Use 'cvxpy' if installed
        verbose=True
    )
    
    # Save results
    results_df.to_csv(config.paths.results_csv)
    print(f"\n[save] Results saved to: {config.paths.results_csv}")
    
    return results_df


def step_4_analyze(config: ResearchConfig, results_df: pd.DataFrame = None):
    """
    Step 4: Analyze backtest results.
    """
    print("\n" + "="*80)
    print("STEP 4: ANALYZE RESULTS")
    print("="*80)
    
    # Load results if not provided
    if results_df is None:
        print(f"[load] Loading results from {config.paths.results_csv}")
        results_df = pd.read_csv(config.paths.results_csv, index_col=0, parse_dates=True)
    
    # Compute statistics
    stats = analyze_performance(results_df, config)
    
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)
    for key, value in stats.items():
        print(f"{key:20s}: {value}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='Cross-Sectional Momentum Research')
    parser.add_argument('--step', type=str, 
                       choices=['feature_eng', 'build_metadata', 'backtest', 'analyze', 'all'],
                       default='all',
                       help='Which step to run')
    parser.add_argument('--model', type=str,
                       choices=['momentum_rank', 'supervised_binned'],
                       default='supervised_binned',
                       help='Model type for backtest')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to custom config file (optional)')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        # TODO: Implement custom config loading
        config = get_default_config()
    else:
        config = get_default_config()
    
    print("="*80)
    print("CROSS-SECTIONAL MOMENTUM RESEARCH FRAMEWORK")
    print("="*80)
    print(f"Configuration:")
    print(f"  Training window: {config.time.TRAINING_WINDOW_DAYS} days")
    print(f"  Holding period: {config.time.HOLDING_PERIOD_DAYS} days")
    print(f"  Feature max lag: {config.time.FEATURE_MAX_LAG_DAYS} days")
    print(f"  Model type: {args.model}")
    
    # Execute requested step(s)
    if args.step == 'feature_eng' or args.step == 'all':
        panel_df = step_1_feature_engineering(config)
    else:
        panel_df = None
    
    if args.step == 'build_metadata' or args.step == 'all':
        universe_metadata, cluster_caps = step_2_build_metadata(config, returns_df=None)
    else:
        universe_metadata = None
    
    if args.step == 'backtest' or args.step == 'all':
        results_df = step_3_backtest(config, panel_df, universe_metadata, args.model)
    else:
        results_df = None
    
    if args.step == 'analyze' or args.step == 'all':
        stats = step_4_analyze(config, results_df)
    
    print("\n" + "="*80)
    print("COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
