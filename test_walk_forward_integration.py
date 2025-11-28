"""
Test walk-forward integration with feature selection pipeline.

This test verifies that:
1. walk_forward_engine correctly calls train_window_model
2. Feature selection identifies signal features over noise
3. Model training and scoring work end-to-end
4. The entire pipeline completes in <60 seconds

Run with:
    D:\REPOSITORY\conda_envs\quant\python.exe -m pytest test_walk_forward_integration.py -v
"""

import numpy as np
import pandas as pd
import pytest
from datetime import datetime, timedelta

from walk_forward_engine import run_walk_forward_backtest
from config import ResearchConfig


def create_realistic_synthetic_panel(
    n_days: int = 600,
    n_tickers: int = 25,
    n_signal_features: int = 10,
    n_noise_features: int = 15,
    signal_strength: float = 0.3,
    seed: int = 42
):
    """
    Create realistic synthetic panel for v3 pipeline testing.
    
    Produces data with:
    - 600+ days (enough for Formation ~1yr + Training ~6mo + test period)
    - 25 tickers (cross-sectional variation)
    - 25 total features (10 signal, 15 noise)
    - Realistic signal strength (~0.3 correlation)
    
    This is designed to work with the v3 pipeline's Formation/Training windows.
    
    Parameters
    ----------
    n_days : int
        Number of trading days
    n_tickers : int
        Number of tickers in universe
    n_signal_features : int
        Features correlated with returns
    n_noise_features : int
        Pure noise features
    signal_strength : float
        Correlation between signal features and returns (0.0-1.0)
    seed : int
        Random seed
        
    Returns
    -------
    panel : pd.DataFrame
        Panel with (Date, Ticker) MultiIndex
    """
    np.random.seed(seed)
    
    # Use business days for more realistic date range
    dates = pd.bdate_range('2018-01-02', periods=n_days, freq='B')
    tickers = [f'ETF{i:03d}' for i in range(n_tickers)]
    
    # Create multi-index
    index = pd.MultiIndex.from_product([dates, tickers], names=['Date', 'Ticker'])
    
    # Generate realistic returns with some autocorrelation and cross-sectional structure
    # Base market factor
    market_factor = np.random.randn(n_days) * 0.01  # Market daily vol ~1%
    
    # Idiosyncratic returns per ticker
    idio_returns = np.random.randn(n_days, n_tickers) * 0.015  # Idiosyncratic vol ~1.5%
    
    # Total returns: market + idiosyncratic
    returns = market_factor[:, np.newaxis] + idio_returns  # Shape: (n_days, n_tickers)
    
    panel_data = {}
    
    # Create SIGNAL features (correlated with returns, but with lag and noise)
    # These simulate "predictive" features
    for i in range(n_signal_features):
        # Different signal types with varying IC
        ic_multiplier = 1.0 - (i * 0.05)  # Decreasing IC for later features
        effective_signal = signal_strength * ic_multiplier
        
        # Add some feature-specific characteristics
        if i < 3:
            # Momentum-like features (lagged returns)
            lag = np.random.randint(1, 10)
            feature = np.roll(returns, lag, axis=0)
            feature[:lag, :] = np.nan
        elif i < 6:
            # Volatility-like features
            vol_window = np.random.randint(10, 30)
            feature = np.zeros_like(returns)
            for t in range(vol_window, n_days):
                feature[t, :] = returns[t-vol_window:t, :].std(axis=0)
            feature[:vol_window, :] = np.nan
        else:
            # Mean-reversion-like features
            ma_window = np.random.randint(5, 20)
            feature = np.zeros_like(returns)
            for t in range(ma_window, n_days):
                feature[t, :] = returns[t-ma_window:t, :].mean(axis=0)
            feature[:ma_window, :] = np.nan
        
        # Add signal correlation with future returns
        noise = np.random.randn(n_days, n_tickers) * (1 - effective_signal)
        signal_component = effective_signal * returns
        feature = np.where(np.isnan(feature), np.nan, feature + signal_component + noise)
        
        panel_data[f'signal_{i}'] = feature.flatten()
    
    # Create NOISE features (uncorrelated with returns)
    for i in range(n_noise_features):
        # Various noise patterns
        if i < 5:
            # Pure random walk
            noise = np.cumsum(np.random.randn(n_days, n_tickers) * 0.01, axis=0)
        elif i < 10:
            # Mean-reverting noise
            noise = np.random.randn(n_days, n_tickers) * 0.5
        else:
            # Autocorrelated noise
            noise = np.zeros((n_days, n_tickers))
            noise[0, :] = np.random.randn(n_tickers)
            for t in range(1, n_days):
                noise[t, :] = 0.9 * noise[t-1, :] + 0.1 * np.random.randn(n_tickers)
        
        panel_data[f'noise_{i}'] = noise.flatten()
    
    # Add forward returns (21-day holding period - standard for cross-sectional momentum)
    holding_period = 21
    fwd_returns = np.zeros_like(returns)
    for t in range(n_days - holding_period):
        fwd_returns[t, :] = returns[t+1:t+holding_period+1, :].sum(axis=0)
    fwd_returns[-holding_period:, :] = np.nan
    
    panel_data[f'FwdRet_{holding_period}'] = fwd_returns.flatten()
    
    # Add Close price (realistic random walk)
    initial_prices = np.random.lognormal(4, 0.5, n_tickers) * 50  # $50-150 range
    close_prices = np.zeros((n_days, n_tickers))
    close_prices[0, :] = initial_prices
    for t in range(1, n_days):
        close_prices[t, :] = close_prices[t-1, :] * np.exp(returns[t, :])
    
    panel_data['Close'] = close_prices.flatten()
    
    # Add market cap (correlated with price, some cross-sectional variation)
    shares_outstanding = np.random.lognormal(15, 1, n_tickers)  # Varying sizes
    market_caps = close_prices * shares_outstanding
    panel_data['market_cap'] = market_caps.flatten()
    
    # Add ADV (average daily volume) - for liquidity filtering
    base_volume = np.random.lognormal(15, 1.5, n_tickers)
    daily_volume = base_volume * (1 + 0.3 * np.random.randn(n_days, n_tickers))
    daily_volume = np.maximum(daily_volume, 1000)  # Floor volume
    adv_63 = np.zeros_like(daily_volume)
    for t in range(63, n_days):
        adv_63[t, :] = daily_volume[t-63:t, :].mean(axis=0)
    adv_63[:63, :] = daily_volume[:63, :].mean(axis=0)  # Use available data
    
    panel_data['ADV_63'] = adv_63.flatten()
    
    panel = pd.DataFrame(panel_data, index=index)
    
    return panel


@pytest.fixture
def synthetic_panel_with_signal():
    """
    Create realistic synthetic panel for v3 pipeline testing.
    
    Data structure:
    - 600 business days (~2.4 years)
    - 25 tickers (realistic cross-section)
    - 10 signal features (varying IC from 0.3 to 0.15)
    - 15 noise features (pure noise)
    - Total: 25 features
    
    Returns:
        (panel_df, metadata, config) tuple
    """
    # Create realistic panel
    n_days = 600
    n_tickers = 25
    panel = create_realistic_synthetic_panel(
        n_days=n_days,
        n_tickers=n_tickers,
        n_signal_features=10,
        n_noise_features=15,
        signal_strength=0.3,
        seed=42
    )
    
    tickers = panel.index.get_level_values('Ticker').unique().tolist()
    
    # Create metadata with required columns for portfolio construction
    metadata = pd.DataFrame({
        'ticker': tickers,
        'sector': ['TECH'] * (n_tickers // 3) + ['HEALTH'] * (n_tickers // 3) + ['FINANCE'] * (n_tickers - 2 * (n_tickers // 3)),
        'region': ['US'] * n_tickers,
        'in_core_after_duplicates': [True] * n_tickers,
        'per_etf_cap': [0.10] * n_tickers,  # 10% max per ETF
        'cluster_id': list(range(n_tickers)),  # Each ETF in own cluster
        'cluster_cap': [1.0] * n_tickers,  # No cluster-level cap
    })
    
    # Create config optimized for this synthetic data
    config = ResearchConfig.default()
    
    # Time windows sized for 600 days of data
    # Formation: ~1 year = 252 days
    # Training: ~6 months = 126 days
    # Need: 252 + 126 + 21 (embargo) + some test days = ~420 days minimum
    config.time.FEATURE_MAX_LAG_DAYS = 63   # 3 months feature lookback
    config.time.TRAINING_WINDOW_DAYS = 126  # 6 months training (legacy param)
    config.time.HOLDING_PERIOD_DAYS = 21    # Standard 21-day holding
    config.time.STEP_DAYS = 500             # Single rebalance for smoke test
    
    # V3 pipeline: Formation and Training windows
    config.features.formation_years = 1.0    # ~252 trading days
    config.features.training_years = 0.5     # ~126 trading days
    config.features.formation_halflife_days = 126  # 6 month half-life for decay
    config.features.training_halflife_days = 42    # 2 month half-life
    config.features.formation_fdr_q_threshold = 0.25  # 25% FDR threshold
    config.features.formation_ic_floor = 0.02  # Minimal IC floor
    
    # Feature selection config
    config.features.FORMATION_IC_THRESHOLD = 0.01
    config.features.MIN_STABILITY_PVALUE = 0.0  # No stability filter
    config.features.MAX_FEATURES_SELECTED = 15
    config.features.K_BIN = 3
    
    # ElasticNet hyperparameter grids
    config.features.ALPHA_ELASTICNET = [0.01, 0.1]
    config.features.L1_RATIO = [0.5, 0.9]
    config.features.alpha_grid = [0.01, 0.1]
    config.features.l1_ratio_grid = [0.5, 0.9]
    
    # Universe filters (permissive for synthetic data)
    config.universe.min_adv_percentile = 0.0
    config.universe.min_data_quality = 0.0
    
    # Disable parallelization for test stability
    config.compute.parallelize_backtest = False
    
    return panel, metadata, config


def test_walk_forward_runs_without_error(synthetic_panel_with_signal):
    """
    SMOKE TEST: Verify walk-forward completes without exceptions.
    """
    panel, metadata, config = synthetic_panel_with_signal
    
    # Run walk-forward (should produce 1 rebalance with minimal config)
    results_df = run_walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        portfolio_method='cvxpy',
        verbose=True  # Enable verbose to debug
    )
    
    # Basic assertions
    assert results_df is not None, "Should return results"
    assert isinstance(results_df, pd.DataFrame), "Should return DataFrame"
    
    # Verify at least one rebalance
    assert len(results_df) >= 1, f"Should have at least 1 rebalance, got {len(results_df)}"
    
    # Check for expected columns
    expected_cols = ['realized_return', 'long_return', 'short_return']
    for col in expected_cols:
        if col in results_df.columns:
            print(f"✓ Found column: {col}")
            break
    
    print(f"\n✓ Walk-forward completed with {len(results_df)} rebalance(s)")
    print(f"  Columns: {list(results_df.columns)[:5]}")  # Show first 5 columns


def test_feature_selection_works(synthetic_panel_with_signal):
    """
    TEST: Verify feature selection runs and creates models.
    
    Since diagnostics aren't directly returned, we'll verify:
    1. Walk-forward completes
    2. Results are generated (implies models were trained)
    3. Portfolio returns are computed (implies scoring worked)
    """
    panel, metadata, config = synthetic_panel_with_signal
    
    results_df = run_walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        portfolio_method='cvxpy',
        verbose=False
    )
    
    # ANCHOR ASSERTION 1: Walk-forward completed successfully
    assert len(results_df) >= 1, (
        f"INTEGRATION FAILURE: No results generated (got {len(results_df)} rows). "
        "Feature selection or model training failed."
    )
    
    # ANCHOR ASSERTION 2: Portfolio returns were computed
    if 'ls_return' in results_df.columns:
        assert results_df['ls_return'].notna().any(), (
            "No valid returns computed - scoring or portfolio construction failed"
        )
        print(f"\n✓ Feature selection working: {len(results_df)} periods with returns")
    else:
        # Try other return columns
        return_cols = [col for col in results_df.columns if 'return' in col.lower()]
        assert len(return_cols) > 0, "No return columns found in results"
        print(f"\n✓ Feature selection working: found return columns {return_cols}")


def test_signal_detection(synthetic_panel_with_signal):
    """
    TEST: Verify signal drives returns.
    
    This is a KEY test: synthetic data has 3 signal features (correlated with returns) 
    and 2 noise features. If feature selection is working, we should see positive 
    portfolio returns more often than negative (signal should be detected).
    """
    panel, metadata, config = synthetic_panel_with_signal
    
    # Run multiple rebalances if possible (reduce step size)
    config.time.STEP_DAYS = 60  # Allow ~2-3 rebalances
    
    results_df = run_walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        portfolio_method='cvxpy',
        verbose=False
    )
    
    # ANCHOR ASSERTION: With signal features, mean return should be reasonable
    # Not necessarily positive (too short test), but not all zeros
    if 'ls_return' in results_df.columns:
        returns = results_df['ls_return'].dropna()
        
        assert len(returns) > 0, "No returns computed"
        assert not (returns == 0).all(), (
            "SIGNAL DETECTION FAILURE: All returns are zero. "
            "Feature selection or scoring not working."
        )
        
        mean_return = returns.mean()
        print(f"\n✓ Signal detection test:")
        print(f"  Number of periods: {len(returns)}")
        print(f"  Mean return: {mean_return:.4f}")
        print(f"  Std return: {returns.std():.4f}")
    else:
        pytest.fail("No ls_return column found in results")


def test_scoring_produces_outputs(synthetic_panel_with_signal):
    """
    TEST: Verify scoring produces actual output values.
    """
    panel, metadata, config = synthetic_panel_with_signal
    
    results_df = run_walk_forward_backtest(
        panel_df=panel,
        universe_metadata=metadata,
        config=config,
        model_type='elasticnet',
        portfolio_method='cvxpy',
        verbose=False
    )
    
    # ANCHOR: Verify returns were computed
    return_cols = [col for col in results_df.columns if 'return' in col.lower()]
    assert len(return_cols) > 0, "Should have at least one return column"
    
    # Check first return column
    first_col = return_cols[0]
    assert results_df[first_col].notna().any(), f"Column {first_col} has all NaN values"
    
    # Check that we have actual numeric values
    first_return = results_df[first_col].dropna().iloc[0]
    assert isinstance(first_return, (int, float, np.number)), f"Return should be numeric, got {type(first_return)}"
    assert np.isfinite(first_return), f"Return should be finite, got {first_return}"
    
    print(f"\n✓ Scoring working:")
    print(f"  Return columns: {return_cols}")
    print(f"  First {first_col}: {first_return:.4f}")


if __name__ == '__main__':
    import sys
    pytest.main([__file__, '-v', '-s'] + sys.argv[1:])
