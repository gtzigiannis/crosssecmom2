"""
Integration test for walk-forward engine with feature selection pipeline.

This test verifies that:
1. per_window_pipeline integrates correctly with walk_forward_engine
2. Models are trained on selected features only (subset design)
3. Scoring works correctly at each rebalance
4. Signal features are consistently selected
5. No exceptions during multi-window walk-forward

Run with: pytest test_walk_forward_integration.py -v -s
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta
from collections import Counter

# Import functions to test
from walk_forward_engine import run_walk_forward_backtest
from config import ResearchConfig


# ============================================================================
# Test Fixtures: Synthetic Data with Known Signal
# ============================================================================

@pytest.fixture
def synthetic_panel_with_signal():
    """
    Generate synthetic panel with known signal features.
    
    Features 0-4: Strong predictive signal (correlated with forward returns)
    Features 5-9: Pure noise (no predictive power)
    
    Returns:
        panel: DataFrame with multi-index (Date, Ticker)
        metadata: DataFrame with ticker information
        config: ResearchConfig object
    """
    np.random.seed(42)
    
    # Panel dimensions - SMALLER for fast testing
    n_days = 300  # Reduced from 500
    n_tickers = 10  # Reduced from 20
    n_signal_features = 3  # Reduced from 5
    n_noise_features = 2  # Reduced from 5
    n_features = n_signal_features + n_noise_features
    
    # Create date range
    start_date = datetime(2020, 1, 1)
    dates = pd.date_range(start_date, periods=n_days, freq='D')
    
    # Create tickers
    tickers = [f'TICKER_{i:02d}' for i in range(n_tickers)]
    
    # Create multi-index
    index = pd.MultiIndex.from_product(
        [dates, tickers],
        names=['Date', 'Ticker']
    )
    
    # Initialize panel
    panel_data = {}
    
    # Generate signal features (0-4): correlated with future returns
    signal_strength = 0.3  # Moderate signal
    true_alpha = np.random.randn(n_tickers) * 0.02  # Ticker-specific alpha
    
    for feat_idx in range(n_signal_features):
        feature_vals = []
        for date_idx, date in enumerate(dates):
            # Feature value with some persistence and noise
            if date_idx == 0:
                vals = np.random.randn(n_tickers) * 0.5
            else:
                # Add persistence: 70% previous + 30% new
                vals = 0.7 * feature_vals[-1] + 0.3 * np.random.randn(n_tickers) * 0.5
            feature_vals.append(vals)
        
        feature_series = np.concatenate(feature_vals)
        panel_data[f'feature_{feat_idx}'] = feature_series
    
    # Generate noise features (5-9): no predictive power
    for feat_idx in range(n_signal_features, n_features):
        feature_vals = np.random.randn(len(index)) * 0.5
        panel_data[f'feature_{feat_idx}'] = feature_vals
    
    # Generate forward returns based on signal features
    returns = []
    for date_idx, date in enumerate(dates):
        # Combine signal features with weights
        signal = np.zeros(n_tickers)
        for feat_idx in range(n_signal_features):
            feat_slice = panel_data[f'feature_{feat_idx}'][
                date_idx * n_tickers:(date_idx + 1) * n_tickers
            ]
            signal += feat_slice * signal_strength / n_signal_features
        
        # Add ticker-specific alpha and noise
        ret = true_alpha + signal + np.random.randn(n_tickers) * 0.05
        returns.append(ret)
    
    returns_series = np.concatenate(returns)
    panel_data['ret_fwd_5d'] = returns_series
    
    # Add market cap for position sizing
    market_caps = np.tile(
        np.random.lognormal(mean=8, sigma=1.5, size=n_tickers),
        n_days
    )
    panel_data['market_cap'] = market_caps * 1e6  # In millions
    
    # Create panel DataFrame
    panel = pd.DataFrame(panel_data, index=index)
    
    # Create metadata
    metadata = pd.DataFrame({
        'ticker': tickers,
        'sector': [f'SECTOR_{i % 4}' for i in range(n_tickers)],  # 4 sectors
        'region': ['US'] * n_tickers
    })
    
    # Create config
    config = ResearchConfig()
    config.TRAINING_WINDOW_DAYS = 120  # Reduced from 200
    config.STEP_SIZE_DAYS = 60
    config.HOLDING_PERIOD_DAYS = 5
    config.MIN_OBSERVATIONS = 50  # Reduced from 100
    config.FORMATION_IC_THRESHOLD = 0.00  # Low threshold to ensure features pass
    config.MIN_STABILITY_PVALUE = 0.30  # Lower threshold for test
    config.MAX_FEATURES_SELECTED = 5  # Reduced from 10
    config.ALPHA_ELASTICNET = [0.01, 0.1]  # Reduced grid
    config.L1_RATIO = [0.5, 0.9]  # Reduced grid
    
    return panel, metadata, config


# ============================================================================
# Integration Tests
# ============================================================================

def test_walk_forward_with_feature_selection(synthetic_panel_with_signal):
    """
    Test walk-forward backtest with feature selection pipeline.
    
    Verifies:
    1. No exceptions during multi-window execution
    2. Models created at each rebalance
    3. Scores have correct shape
    4. Signal features (0-4) are frequently selected
    5. Diagnostics captured per window
    """
    panel, metadata, config = synthetic_panel_with_signal
    
    # Run walk-forward backtest
    results = run_walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        start_date=None,  # Use default (earliest date + training window)
        end_date=None,    # Use default (latest date)
        rebalance_frequency='quarterly',  # Will produce 2-3 rebalances
        parallel=False,   # Sequential for easier debugging
        n_jobs=1,
        verbose=True
    )
    
    # Basic assertions
    assert results is not None, "Walk-forward should return results"
    assert 'returns' in results, "Results should contain returns DataFrame"
    assert 'diagnostics' in results, "Results should contain diagnostics"
    
    returns_df = results['returns']
    diagnostics = results['diagnostics']
    
    # Check we got multiple rebalances
    assert len(returns_df) >= 2, f"Expected at least 2 rebalances, got {len(returns_df)}"
    
    # Check returns DataFrame structure
    expected_cols = ['date', 'realized_return', 'n_long', 'n_short', 
                     'train_time', 'score_time', 'portfolio_time']
    for col in expected_cols:
        assert col in returns_df.columns, f"Missing column: {col}"
    
    # Check all rebalances produced valid returns
    assert returns_df['realized_return'].notna().all(), "All rebalances should have returns"
    
    # Check diagnostics captured per window
    assert len(diagnostics) >= 2, f"Expected at least 2 diagnostic entries, got {len(diagnostics)}"
    
    for diag in diagnostics:
        assert 'date' in diag, "Diagnostics should have date"
        assert 'n_features' in diag, "Diagnostics should have n_features"
        assert 'selected_features' in diag, "Diagnostics should have selected_features"
        
        # Check features were selected
        n_features = diag['n_features']
        selected_features = diag['selected_features']
        assert n_features > 0, "At least some features should be selected"
        assert len(selected_features) == n_features, "selected_features length should match n_features"
    
    # Check signal features are frequently selected
    all_selected = []
    for diag in diagnostics:
        all_selected.extend(diag['selected_features'])
    
    # Count feature occurrences
    feature_counts = Counter(all_selected)
    
    # Signal features (0-2) should appear more often than noise features (3-4)
    signal_count = sum(feature_counts.get(f'feature_{i}', 0) for i in range(3))
    noise_count = sum(feature_counts.get(f'feature_{i}', 0) for i in range(3, 5))
    
    print(f"\nFeature selection statistics:")
    print(f"  Signal features (0-2) selected: {signal_count} times")
    print(f"  Noise features (3-4) selected: {noise_count} times")
    print(f"  Top 3 features: {feature_counts.most_common(3)}")
    
    # ANCHOR ASSERTION: Signal features should dominate
    # Intuition: With 120-day training windows and moderately strong signal (correlation ~0.3),
    # ElasticNet should consistently identify signal features. We expect:
    # - Signal features selected 60%+ of the time (strong bias toward predictive features)
    # - At least 2x more signal selections than noise selections
    if signal_count + noise_count > 0:
        signal_ratio = signal_count / (signal_count + noise_count)
        assert signal_ratio >= 0.6, (
            f"ANCHOR FAILURE: Signal features should be selected ≥60% of time. "
            f"Got {signal_ratio:.1%}. This suggests feature selection is not working properly. "
            f"Signal count: {signal_count}, Noise count: {noise_count}"
        )
        print(f"  ✓ Signal ratio: {signal_ratio:.1%} (target: ≥60%)")
        
        # Secondary check: signal should be at least 2x noise
        if noise_count > 0:
            signal_to_noise = signal_count / noise_count
            assert signal_to_noise >= 2.0, (
                f"ANCHOR FAILURE: Signal-to-noise ratio too low. "
                f"Got {signal_to_noise:.1f}x, expected ≥2.0x. "
                f"Feature selection should strongly favor predictive features."
            )
            print(f"  ✓ Signal-to-noise ratio: {signal_to_noise:.1f}x (target: ≥2.0x)")


def test_walk_forward_model_subset_design(synthetic_panel_with_signal):
    """
    Test that models use subset design (operate on selected features only).
    
    Verifies:
    1. Model's selected_features attribute is populated
    2. Model's scaling_params contain only selected features
    3. Scoring works with subset features
    """
    panel, metadata, config = synthetic_panel_with_signal
    
    # Run single rebalance
    config.STEP_SIZE_DAYS = 999  # Large step to get only one rebalance
    
    results = run_walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        start_date=None,
        end_date=None,
        rebalance_frequency='quarterly',
        parallel=False,
        n_jobs=1,
        verbose=True
    )
    
    # Get diagnostics for first window
    diagnostics = results['diagnostics']
    assert len(diagnostics) >= 1, "Should have at least one diagnostic entry"
    
    first_diag = diagnostics[0]
    selected_features = first_diag['selected_features']
    n_features = first_diag['n_features']
    
    print(f"\nSubset design verification:")
    print(f"  Selected features: {selected_features}")
    print(f"  Number of features: {n_features}")
    
    # Basic checks
    assert n_features > 0, "Should select at least some features"
    assert len(selected_features) == n_features, "Feature count should match"
    
    # ANCHOR ASSERTION: Subset design constraint
    # Intuition: With 5 total features and IC/stability filtering,
    # we should NOT select all features. A proper subset design should
    # select 2-4 features typically (40-80% of candidates).
    assert n_features < 5, (
        f"ANCHOR FAILURE: Selected ALL features ({n_features}/5). "
        f"Subset design should be selective, not use all candidates. "
        f"This suggests feature selection filters are not working."
    )
    assert n_features >= 2, (
        f"ANCHOR FAILURE: Too few features selected ({n_features}). "
        f"With synthetic signal data, should select at least 2 features."
    )
    print(f"  ✓ Feature count in valid range: {n_features}/5 features (40-80%)")


def test_walk_forward_parallel_execution(synthetic_panel_with_signal):
    """
    Test parallel execution path produces same results as sequential.
    
    Note: Due to potential numerical differences in parallel execution,
    we check that results are similar, not identical.
    """
    panel, metadata, config = synthetic_panel_with_signal
    
    # Run sequential
    results_seq = run_walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        start_date=None,
        end_date=None,
        rebalance_frequency='quarterly',
        parallel=False,
        n_jobs=1,
        verbose=False
    )
    
    # Run parallel
    results_par = run_walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        start_date=None,
        end_date=None,
        rebalance_frequency='quarterly',
        parallel=True,
        n_jobs=2,
        verbose=False
    )
    
    # Check both produced results
    assert results_seq is not None, "Sequential execution should succeed"
    assert results_par is not None, "Parallel execution should succeed"
    
    # Check same number of rebalances
    seq_returns = results_seq['returns']
    par_returns = results_par['returns']
    assert len(seq_returns) == len(par_returns), (
        f"Sequential and parallel should have same rebalances: "
        f"{len(seq_returns)} vs {len(par_returns)}"
    )
    
    # Check returns are similar (allow for numerical differences)
    seq_vals = seq_returns['realized_return'].values
    par_vals = par_returns['realized_return'].values
    correlation = np.corrcoef(seq_vals, par_vals)[0, 1]
    
    print(f"\nParallel execution verification:")
    print(f"  Sequential returns: {seq_vals}")
    print(f"  Parallel returns: {par_vals}")
    print(f"  Correlation: {correlation:.4f}")
    
    # ANCHOR ASSERTION: Parallel consistency check
    # Intuition: Parallel and sequential should produce nearly identical results
    # since there's no randomness in feature selection or model training.
    # We expect correlation > 0.95, ideally > 0.99.
    assert correlation > 0.95, (
        f"ANCHOR FAILURE: Parallel/sequential results diverged too much. "
        f"Correlation = {correlation:.4f}, expected > 0.95. "
        f"This suggests parallel execution has a bug or race condition."
    )
    print(f"  ✓ Parallel consistency: correlation = {correlation:.4f} (target: >0.95)")
    
    # Additional check: mean returns should be similar
    seq_mean = np.mean(seq_vals)
    par_mean = np.mean(par_vals)
    mean_diff = abs(seq_mean - par_mean)
    assert mean_diff < 0.01, (
        f"ANCHOR FAILURE: Mean returns differ too much: "
        f"sequential={seq_mean:.4f}, parallel={par_mean:.4f}, diff={mean_diff:.4f}"
    )
    print(f"  ✓ Mean consistency: |{seq_mean:.4f} - {par_mean:.4f}| < 0.01")


if __name__ == '__main__':
    # Run with pytest
    import sys
    pytest.main([__file__, '-v', '-s'] + sys.argv[1:])
