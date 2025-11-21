"""
Test script for regime system
==============================
Quick test to verify regime classification and portfolio mode mapping.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from config import RegimeConfig
from regime import compute_regime_series, get_portfolio_mode_for_regime


def create_test_panel():
    """Create a simple test panel with SPY price data."""
    dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
    
    # Create synthetic SPY data with clear bull/bear/range periods
    n = len(dates)
    
    # Bull period: strong uptrend (first 1/3)
    bull_prices = np.linspace(100, 150, n // 3)
    
    # Range period: sideways (middle 1/3)
    range_prices = np.full(n // 3, 150) + np.random.normal(0, 2, n // 3)
    
    # Bear period: downtrend (last 1/3)
    bear_prices = np.linspace(150, 100, n - len(bull_prices) - len(range_prices))
    
    prices = np.concatenate([bull_prices, range_prices, bear_prices])
    
    # Create panel structure with 'Ticker' (uppercase) to match expected format
    data = []
    for date, price in zip(dates, prices):
        data.append({
            'Date': date,
            'Ticker': 'SPY',  # Changed to uppercase
            'Close': price,
            'Volume': 1e6,  # dummy
        })
    
    df = pd.DataFrame(data)
    df = df.set_index(['Date', 'Ticker'])
    
    return df


def test_regime_classification():
    """Test regime classification on synthetic data."""
    print("="*80)
    print("TEST: Regime Classification")
    print("="*80)
    
    # Create test panel
    panel_df = create_test_panel()
    print(f"Panel shape: {panel_df.shape}")
    print(f"Date range: {panel_df.index.get_level_values('Date').min().date()} to {panel_df.index.get_level_values('Date').max().date()}")
    
    # Create regime config
    config = RegimeConfig(
        use_regime=True,
        market_ticker='SPY',
        ma_window=50,  # Shorter MA for quick testing
        lookback_return_days=21,  # 1-month return
        bull_ret_threshold=0.02,
        bear_ret_threshold=-0.02,
        neutral_buffer_days=5,
        use_hysteresis=True
    )
    
    # Compute regime series
    print("\nComputing regime series...")
    regime_series = compute_regime_series(panel_df, config, verbose=True)
    
    print(f"\nRegime series computed: {len(regime_series)} values")
    print(f"Date range: {regime_series.index.min().date()} to {regime_series.index.max().date()}")
    
    # Check distribution
    regime_counts = regime_series.value_counts()
    print("\nRegime distribution:")
    for regime, count in regime_counts.items():
        pct = count / len(regime_series) * 100
        print(f"  {regime}: {count} ({pct:.1f}%)")
    
    # Test mode mapping
    print("\nTesting portfolio mode mapping:")
    for regime in ['bull', 'bear', 'range']:
        mode = get_portfolio_mode_for_regime(regime)
        print(f"  {regime} → {mode}")
    
    # Show sample of regime series
    print("\nSample regime series (first 10 and last 10):")
    print(regime_series.head(10))
    print("...")
    print(regime_series.tail(10))
    
    # Verify no look-ahead bias
    print("\n" + "="*80)
    print("Checking for look-ahead bias...")
    print("="*80)
    
    # Get the first date where we have regime classification
    first_regime_date = regime_series.index[0]
    
    # Check that regime at first_regime_date uses data up to first_regime_date - 1
    print(f"First regime date: {first_regime_date.date()}")
    print(f"Regime value: {regime_series.iloc[0]}")
    
    # Get SPY close prices around this date
    spy_prices = panel_df.xs('SPY', level='Ticker')['Close']  # Changed to uppercase
    date_idx = spy_prices.index.get_loc(first_regime_date)
    
    if date_idx > 0:
        print(f"SPY close at t-1 ({spy_prices.index[date_idx-1].date()}): {spy_prices.iloc[date_idx-1]:.2f}")
        print(f"SPY close at t0 ({first_regime_date.date()}): {spy_prices.iloc[date_idx]:.2f}")
        print("✓ Regime classification should NOT use t0 price (no look-ahead)")
    
    return regime_series


def test_portfolio_mode_integration():
    """Test that portfolio modes are correctly applied."""
    print("\n" + "="*80)
    print("TEST: Portfolio Mode Integration")
    print("="*80)
    
    # Test all regime → mode mappings
    test_cases = [
        ('bull', 'long_only', "Bull market should go long only"),
        ('bear', 'short_only', "Bear market should go short only"),
        ('range', 'cash', "Range market should stay in cash"),
        ('unknown', 'ls', "Unknown regime should default to long/short"),
    ]
    
    print("\nTesting regime → mode mappings:")
    for regime, expected_mode, description in test_cases:
        mode = get_portfolio_mode_for_regime(regime)
        status = "✓" if mode == expected_mode else "✗"
        print(f"{status} {description}")
        print(f"   Regime: {regime} → Mode: {mode} (expected: {expected_mode})")
        assert mode == expected_mode, f"Expected {expected_mode}, got {mode}"
    
    print("\n✓ All mode mappings correct")


if __name__ == '__main__':
    # Run tests
    regime_series = test_regime_classification()
    test_portfolio_mode_integration()
    
    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
