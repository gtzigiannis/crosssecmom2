"""
Regime Classification Module
============================
Classifies market regimes (bull/bear/range) based on price action and volatility.

Key principle: Uses ONLY historical data (no look-ahead bias).
Regime at t0 is computed using data up to t0-1.

Regimes:
- bull: Strong uptrend (above MA, positive momentum)
- bear: Strong downtrend (below MA, negative momentum)
- range: Sideways/neutral market

Usage:
    from config import RegimeConfig
    from regime import compute_regime_series
    
    config = RegimeConfig(
        market_ticker='SPY',
        ma_window=200,
        bull_ret_threshold=0.02
    )
    
    regime_series = compute_regime_series(panel_df, config)
    # Returns pd.Series indexed by date with values: 'bull', 'bear', 'range'
"""

import pandas as pd
import numpy as np
from typing import Optional, TYPE_CHECKING
import warnings

if TYPE_CHECKING:
    from config import RegimeConfig


def compute_regime_indicators(
    price_series: pd.Series,
    ma_window: int,
    lookback_return_days: int
) -> pd.DataFrame:
    """
    Compute regime indicators from price series.
    
    CRITICAL: All indicators use only historical data (no look-ahead).
    
    Parameters
    ----------
    price_series : pd.Series
        Price series (e.g., Close prices) indexed by date
    ma_window : int
        Window for moving average
    lookback_return_days : int
        Window for return calculation
        
    Returns
    -------
    pd.DataFrame
        Indicators with columns: close, ma, ret, above_ma
    """
    df = pd.DataFrame(index=price_series.index)
    df['close'] = price_series
    
    # Moving average (uses past data only)
    df['ma'] = price_series.rolling(window=ma_window, min_periods=ma_window).mean()
    
    # Lookback return (from t-lookback to t)
    df['ret'] = price_series.pct_change(periods=lookback_return_days)
    
    # Price relative to MA
    df['above_ma'] = df['close'] > df['ma']
    
    # Realized volatility (optional, for future use)
    df['vol'] = price_series.pct_change().rolling(window=lookback_return_days).std()
    
    return df


def classify_regime_simple(
    indicators: pd.DataFrame,
    bull_ret_threshold: float,
    bear_ret_threshold: float
) -> pd.Series:
    """
    Classify regime based on indicators (simple rules).
    
    Rules:
    - bull: Price > MA AND return > bull_threshold
    - bear: Price < MA AND return < bear_threshold  
    - range: Otherwise
    
    Parameters
    ----------
    indicators : pd.DataFrame
        Output from compute_regime_indicators
    bull_ret_threshold : float
        Return threshold for bull regime
    bear_ret_threshold : float
        Return threshold for bear regime
        
    Returns
    -------
    pd.Series
        Regime classification ('bull', 'bear', 'range') indexed by date
    """
    regime = pd.Series('range', index=indicators.index, dtype=str)
    
    # Bull: above MA and strong positive return
    bull_mask = (
        indicators['above_ma'] & 
        (indicators['ret'] > bull_ret_threshold)
    )
    regime[bull_mask] = 'bull'
    
    # Bear: below MA and strong negative return
    bear_mask = (
        ~indicators['above_ma'] & 
        (indicators['ret'] < bear_ret_threshold)
    )
    regime[bear_mask] = 'bear'
    
    # Everything else stays 'range'
    
    return regime


def apply_hysteresis(
    regime_series: pd.Series,
    buffer_days: int
) -> pd.Series:
    """
    Apply hysteresis to reduce regime switching.
    
    Once in a regime, stay for at least buffer_days before switching.
    
    Parameters
    ----------
    regime_series : pd.Series
        Raw regime classification
    buffer_days : int
        Minimum days to stay in regime
        
    Returns
    -------
    pd.Series
        Regime series with hysteresis applied
    """
    if buffer_days <= 0:
        return regime_series
    
    result = regime_series.copy()
    current_regime = 'range'
    regime_start_idx = 0
    
    for i in range(len(regime_series)):
        new_regime = regime_series.iloc[i]
        
        # If regime changed, check if enough time has passed
        if new_regime != current_regime:
            days_in_current = i - regime_start_idx
            
            if days_in_current >= buffer_days:
                # Switch regime
                current_regime = new_regime
                regime_start_idx = i
            else:
                # Stay in current regime (hysteresis)
                result.iloc[i] = current_regime
        
    return result


def compute_regime_series(
    panel_df: pd.DataFrame,
    config: "RegimeConfig",
    verbose: bool = False
) -> pd.Series:
    """
    Compute regime series from panel data.
    
    CRITICAL: Regime at date t0 is computed using data up to t0-1.
    This is achieved by:
    1. Computing indicators at each date using historical data only
    2. Shifting the regime series by 1 day to ensure no look-ahead
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel with (Date, Ticker) MultiIndex containing Close prices
    config : RegimeConfig
        Regime configuration
    verbose : bool
        Print diagnostics
        
    Returns
    -------
    pd.Series
        Regime classification indexed by date
        Values: 'bull', 'bear', 'range'
    """
    if verbose:
        print(f"[regime] Computing regime series using {config.market_ticker}")
        print(f"[regime] MA window: {config.ma_window}, Lookback: {config.lookback_return_days}")
    
    # Extract market ticker price series
    try:
        if 'Ticker' in panel_df.index.names:
            # Panel has MultiIndex (Date, Ticker)
            market_data = panel_df.xs(config.market_ticker, level='Ticker')['Close']
        else:
            # Panel has single index (assume Date)
            market_data = panel_df.loc[panel_df['Ticker'] == config.market_ticker, 'Close']
            market_data.index = panel_df.loc[panel_df['Ticker'] == config.market_ticker].index.get_level_values('Date')
    except (KeyError, ValueError) as e:
        warnings.warn(f"Market ticker {config.market_ticker} not found in panel: {e}")
        # Return all 'range' regime
        dates = panel_df.index.get_level_values('Date').unique()
        return pd.Series('range', index=dates, dtype=str, name='regime')
    
    if len(market_data) == 0:
        warnings.warn(f"No data for market ticker {config.market_ticker}")
        dates = panel_df.index.get_level_values('Date').unique()
        return pd.Series('range', index=dates, dtype=str, name='regime')
    
    # Sort and remove duplicates
    market_data = market_data.sort_index()
    if market_data.index.duplicated().any():
        market_data = market_data[~market_data.index.duplicated(keep='first')]
    
    # Compute indicators
    indicators = compute_regime_indicators(
        price_series=market_data,
        ma_window=config.ma_window,
        lookback_return_days=config.lookback_return_days
    )
    
    # Classify regime
    regime = classify_regime_simple(
        indicators=indicators,
        bull_ret_threshold=config.bull_ret_threshold,
        bear_ret_threshold=config.bear_ret_threshold
    )
    
    # Apply hysteresis if requested
    if config.use_hysteresis and config.neutral_buffer_days > 0:
        regime = apply_hysteresis(regime, config.neutral_buffer_days)
    
    # CRITICAL: Shift regime by 1 day to avoid look-ahead bias
    # Regime at t0 should be based on data up to t0-1
    regime = regime.shift(1)
    
    # Fill first value with 'range' (no prior data)
    regime = regime.fillna('range')
    
    regime.name = 'regime'
    
    if verbose:
        print(f"[regime] Computed regime for {len(regime)} dates")
        regime_counts = regime.value_counts()
        for r, count in regime_counts.items():
            print(f"[regime]   {r}: {count} days ({100*count/len(regime):.1f}%)")
    
    return regime


def get_portfolio_mode_for_regime(regime: str) -> str:
    """
    Map regime to portfolio construction mode.
    
    Parameters
    ----------
    regime : str
        Regime classification ('bull', 'bear', 'range')
        
    Returns
    -------
    str
        Portfolio mode: 'long_only', 'short_only', 'cash', 'ls' (long/short)
    """
    if regime == 'bull':
        return 'long_only'
    elif regime == 'bear':
        return 'short_only'
    elif regime == 'range':
        return 'cash'  # No positions, earn cash rate
    else:
        # Fallback
        return 'ls'


if __name__ == "__main__":
    print("Regime classification module loaded.")
    print("\nExample usage:")
    print("  from regime import compute_regime_series, RegimeConfig")
    print("  config = RegimeConfig(market_ticker='SPY')")
    print("  regime_series = compute_regime_series(panel_df, config)")
    print("\nRegime modes:")
    print("  - bull: Strong uptrend (long only)")
    print("  - bear: Strong downtrend (short only)")
    print("  - range: Sideways market (cash)")
