"""
Portfolio Construction Module
==============================
Constructs long/short portfolios from cross-sectional scores with:
1. Per-ETF caps
2. Per-cluster caps
3. Long/short quantile selection
4. Leverage constraints

Uses optimization to enforce caps when needed.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional
import warnings

try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    warnings.warn("cvxpy not available, falling back to simple capping")


def construct_portfolio_simple(
    scores: pd.Series,
    universe_metadata: pd.DataFrame,
    config,
    enforce_caps: bool = False,
    mode: str = 'ls'
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple portfolio construction: top/bottom quantiles with equal weights.
    
    Optionally enforces caps via simple scaling (not optimal, but fast).
    Supports multiple portfolio modes via mode parameter.
    
    Parameters
    ----------
    scores : pd.Series
        Cross-sectional scores indexed by ticker (higher = more attractive)
    universe_metadata : pd.DataFrame
        Metadata with cluster_id, cluster_cap, per_etf_cap
    config : ResearchConfig
        Configuration object
    enforce_caps : bool
        If True, scale weights to respect caps
    mode : str, default 'ls'
        Portfolio mode:
        - 'ls': Standard long/short (uses config.portfolio settings)
        - 'long_only': Only long positions
        - 'short_only': Only short positions
        - 'cash': No positions (returns empty Series for both sides)
        
    Returns
    -------
    tuple of (pd.Series, pd.Series)
        (long_weights, short_weights) indexed by ticker
    """
    # Handle cash mode (no positions)
    if mode == 'cash':
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Filter scores to valid tickers
    valid_tickers = scores.dropna().index
    
    if len(valid_tickers) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Sort by score
    sorted_scores = scores[valid_tickers].sort_values(ascending=False)
    
    # Select top/bottom quantiles
    n_long = max(1, int(len(sorted_scores) * (1 - config.portfolio.long_quantile)))
    n_short = max(1, int(len(sorted_scores) * config.portfolio.short_quantile))
    
    long_tickers = sorted_scores.head(n_long).index
    short_tickers = sorted_scores.tail(n_short).index
    
    # Equal weights within each side
    # IMPORTANT: 
    # - Long positions scaled by long_leverage (typically 1.0 = 100% of capital)
    # - Short positions scaled by margin requirement (e.g., 50% margin = max 50% short)
    long_leverage = config.portfolio.long_leverage
    short_leverage = config.portfolio.margin
    
    # Determine which sides to build based on mode
    # Priority: explicit mode parameter > config.portfolio flags
    if mode == 'short_only':
        build_long = False
        build_short = True
    elif mode == 'long_only':
        build_long = True
        build_short = False
    elif mode == 'ls':
        # Use config flags
        if config.portfolio.short_only:
            build_long = False
            build_short = True
        elif config.portfolio.long_only:
            build_long = True
            build_short = False
        else:
            build_long = True
            build_short = True
    else:
        raise ValueError(f"Invalid mode: {mode}. Must be 'ls', 'long_only', 'short_only', or 'cash'")
    
    # Build long side
    if build_long:
        long_weights = pd.Series(long_leverage / n_long, index=long_tickers)
    else:
        long_weights = pd.Series(dtype=float)
    
    # Build short side
    if build_short:
        short_weights = pd.Series(-short_leverage / n_short, index=short_tickers)
    else:
        short_weights = pd.Series(dtype=float)
    
    if not enforce_caps:
        return long_weights, short_weights
    
    # Apply caps via simple scaling
    metadata_subset = universe_metadata.set_index('ticker').loc[
        list(long_tickers) + list(short_tickers)
    ]
    
    # Per-ETF caps
    for side, weights in [('long', long_weights), ('short', short_weights)]:
        for ticker in weights.index:
            cap = metadata_subset.loc[ticker, 'per_etf_cap']
            if abs(weights[ticker]) > cap:
                weights[ticker] = cap * np.sign(weights[ticker])
    
    # Renormalize
    long_weights = long_weights / long_weights.sum() if long_weights.sum() > 0 else long_weights
    short_weights = short_weights / abs(short_weights.sum()) if short_weights.sum() < 0 else short_weights
    
    return long_weights, short_weights


def construct_portfolio_cvxpy(
    scores: pd.Series,
    universe_metadata: pd.DataFrame,
    config,
    mode: str = 'ls'
) -> Tuple[pd.Series, pd.Series]:
    """
    Portfolio construction with CVXPY optimization to enforce caps exactly.
    
    Formulation:
    - Maximize: sum of (score_i * weight_i) for long + short sides
    - Subject to:
      * w_i >= 0 for long, w_i <= 0 for short
      * |w_i| <= per_etf_cap_i
      * sum_{i in cluster_k} |w_i| <= cluster_cap_k
      * sum(long_weights) = target_long_gross (e.g., 1.0)
      * sum(abs(short_weights)) = target_short_gross (e.g., 1.0)
    
    Parameters
    ----------
    scores : pd.Series
        Cross-sectional scores indexed by ticker
    universe_metadata : pd.DataFrame
        Metadata with cluster_id, cluster_cap, per_etf_cap
    config : ResearchConfig
        Configuration object
    mode : str, default 'ls'
        Portfolio mode: 'ls', 'long_only', 'short_only', or 'cash'
        
    Returns
    -------
    tuple of (pd.Series, pd.Series)
        (long_weights, short_weights) indexed by ticker
    """
    if not CVXPY_AVAILABLE:
        warnings.warn("CVXPY not available, using simple construction")
        return construct_portfolio_simple(scores, universe_metadata, config, enforce_caps=True, mode=mode)
    
    # Handle cash mode
    if mode == 'cash':
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Filter to valid tickers
    valid_tickers = scores.dropna().index.tolist()
    
    if len(valid_tickers) == 0:
        return pd.Series(dtype=float), pd.Series(dtype=float)
    
    # Get metadata for valid tickers
    metadata_idx = universe_metadata.set_index('ticker')
    metadata_subset = metadata_idx.loc[valid_tickers]
    
    # Separate into long and short candidates based on quantiles
    sorted_scores = scores[valid_tickers].sort_values(ascending=False)
    n_long = max(1, int(len(sorted_scores) * (1 - config.portfolio.long_quantile)))
    n_short = max(1, int(len(sorted_scores) * config.portfolio.short_quantile))
    
    long_tickers = sorted_scores.head(n_long).index.tolist()
    short_tickers = sorted_scores.tail(n_short).index.tolist()
    
    # Determine which sides to build based on mode
    if mode == 'short_only':
        build_long = False
        build_short = True
    elif mode == 'long_only':
        build_long = True
        build_short = False
    elif mode == 'ls':
        # Use config flags
        if config.portfolio.short_only:
            build_long = False
            build_short = True
        elif config.portfolio.long_only:
            build_long = True
            build_short = False
        else:
            build_long = True
            build_short = True
    else:
        raise ValueError(f"Invalid mode: {mode}")
    
    # ===== Long portfolio =====
    # Target gross exposure scaled by long_leverage (e.g., 1.0 = 100% of capital)
    long_target_gross = config.portfolio.long_leverage
    
    if not build_long:
        # Skip long portfolio
        long_weights = pd.Series(dtype=float)
    elif len(long_tickers) > 0:
        long_weights = _optimize_one_side(
            tickers=long_tickers,
            scores=scores[long_tickers],
            metadata=metadata_subset.loc[long_tickers],
            target_gross=long_target_gross,
            side='long',
            config=config
        )
    else:
        long_weights = pd.Series(dtype=float)
    
    # ===== Short portfolio =====
    # Target gross exposure scaled by margin requirement
    # With 50% margin, can only short up to 50% of capital
    short_target_gross = config.portfolio.margin
    
    if not build_short:
        # Skip short portfolio
        short_weights = pd.Series(dtype=float)
    elif len(short_tickers) > 0:
        short_weights = _optimize_one_side(
            tickers=short_tickers,
            scores=scores[short_tickers],
            metadata=metadata_subset.loc[short_tickers],
            target_gross=short_target_gross,
            side='short',
            config=config
        )
    else:
        short_weights = pd.Series(dtype=float)
    
    return long_weights, short_weights


def _optimize_one_side(
    tickers: list,
    scores: pd.Series,
    metadata: pd.DataFrame,
    target_gross: float,
    side: str,
    config
) -> pd.Series:
    """
    Optimize weights for one side (long or short) of the portfolio.
    
    Parameters
    ----------
    tickers : list
        Tickers for this side
    scores : pd.Series
        Scores for these tickers
    metadata : pd.DataFrame
        Metadata for these tickers
    target_gross : float
        Target gross exposure (e.g., 1.0)
    side : str
        'long' or 'short'
    config : ResearchConfig
        Configuration
        
    Returns
    -------
    pd.Series
        Weights indexed by ticker (positive for long, negative for short)
    """
    n = len(tickers)
    
    # Decision variable
    w = cp.Variable(n)
    
    # Objective: maximize sum of scores * weights
    # (For short, scores are already inverted or we negate the objective)
    objective = cp.Maximize(scores.values @ w)
    
    # Constraints
    constraints = [
        w >= 0,  # Non-negative weights (will negate later for short)
        cp.sum(w) == target_gross,  # Target gross exposure
    ]
    
    # Per-ETF caps
    per_etf_caps = metadata['per_etf_cap'].values
    constraints.append(w <= per_etf_caps)
    
    # Per-cluster caps
    cluster_ids = metadata['cluster_id'].values
    unique_clusters = np.unique(cluster_ids[~np.isnan(cluster_ids)])
    
    for cid in unique_clusters:
        cluster_mask = (cluster_ids == cid)
        cluster_cap = metadata.loc[cluster_mask, 'cluster_cap'].iloc[0]
        constraints.append(cp.sum(w[cluster_mask]) <= cluster_cap)
    
    # Solve
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=cp.ECOS, verbose=False)
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            warnings.warn(f"Optimization status: {problem.status}, using fallback")
            # Fallback to equal weights
            weights_arr = np.ones(n) / n * target_gross
        else:
            weights_arr = w.value
            
    except Exception as e:
        warnings.warn(f"Optimization failed: {e}, using equal weights")
        weights_arr = np.ones(n) / n * target_gross
    
    # Convert to Series
    weights = pd.Series(weights_arr, index=tickers)
    
    # Negate for short side
    if side == 'short':
        weights = -weights
    
    return weights


def construct_portfolio(
    scores: pd.Series,
    universe_metadata: pd.DataFrame,
    config,
    method: str = 'cvxpy',
    mode: str = 'ls'
) -> Tuple[pd.Series, pd.Series, Dict]:
    """
    Main portfolio construction function.
    
    Parameters
    ----------
    scores : pd.Series
        Cross-sectional scores indexed by ticker
    universe_metadata : pd.DataFrame
        Metadata with cluster_id, cluster_cap, per_etf_cap
    config : ResearchConfig
        Configuration object
    method : str
        'cvxpy' (optimal) or 'simple' (fast approximation)
    mode : str
        Portfolio mode: 'ls' (long/short), 'long_only', 'short_only', 'cash'
        
    Returns
    -------
    tuple of (pd.Series, pd.Series, dict)
        - long_weights: Long positions
        - short_weights: Short positions
        - portfolio_stats: Dict with diagnostics
    """
    if method == 'cvxpy' and CVXPY_AVAILABLE:
        long_weights, short_weights = construct_portfolio_cvxpy(scores, universe_metadata, config, mode=mode)
    else:
        long_weights, short_weights = construct_portfolio_simple(
            scores, universe_metadata, config, enforce_caps=True, mode=mode
        )
    
    # Compute portfolio statistics
    stats = {
        'n_long': len(long_weights),
        'n_short': len(short_weights),
        'gross_long': long_weights.sum() if len(long_weights) > 0 else 0.0,
        'gross_short': abs(short_weights.sum()) if len(short_weights) > 0 else 0.0,
        'max_long_weight': long_weights.max() if len(long_weights) > 0 else 0.0,
        'max_short_weight': abs(short_weights.min()) if len(short_weights) > 0 else 0.0,
    }
    
    # Check cluster caps
    if len(long_weights) > 0 or len(short_weights) > 0:
        all_weights = pd.concat([long_weights, short_weights])
        metadata_idx = universe_metadata.set_index('ticker')
        
        cluster_exposures = {}
        for ticker, weight in all_weights.items():
            cid = metadata_idx.loc[ticker, 'cluster_id']
            if pd.notna(cid):
                cluster_exposures[cid] = cluster_exposures.get(cid, 0.0) + abs(weight)
        
        stats['cluster_exposures'] = cluster_exposures
        
        # Check violations
        violations = []
        for cid, exposure in cluster_exposures.items():
            cap = metadata_idx[metadata_idx['cluster_id'] == cid]['cluster_cap'].iloc[0]
            if exposure > cap * 1.01:  # Allow 1% tolerance
                violations.append((cid, exposure, cap))
        
        if violations:
            stats['cap_violations'] = violations
    
    return long_weights, short_weights, stats


def evaluate_portfolio_return(
    panel_df: pd.DataFrame,
    t0: pd.Timestamp,
    long_weights: pd.Series,
    short_weights: pd.Series,
    config,
    prev_long_weights: Optional[pd.Series] = None,
    prev_short_weights: Optional[pd.Series] = None
) -> Dict:
    """
    Evaluate portfolio return using forward returns.
    
    Parameters
    ----------
    panel_df : pd.DataFrame
        Panel with forward returns
    t0 : pd.Timestamp
        Portfolio formation date
    long_weights : pd.Series
        Long positions
    short_weights : pd.Series
        Short positions
    config : ResearchConfig
        Configuration
        
    Returns
    -------
    dict
        Performance metrics
    """
    target_col = f'FwdRet_{config.time.HOLDING_PERIOD_DAYS}'
    
    try:
        cross_section = panel_df.loc[t0]
    except KeyError:
        return {
            'date': t0,
            'long_ret': np.nan,
            'short_ret': np.nan,
            'ls_return': np.nan,
            'n_long': 0,
            'n_short': 0,
        }
    
    # Long return
    if len(long_weights) > 0:
        long_returns = cross_section.loc[long_weights.index, target_col]
        long_ret = (long_weights * long_returns).sum()
    else:
        long_ret = 0.0
    
    # Short return
    if len(short_weights) > 0:
        short_returns = cross_section.loc[short_weights.index, target_col]
        short_ret = (short_weights * short_returns).sum()
    else:
        short_ret = 0.0
    
    # Calculate cash return for uninvested capital
    # Gross exposures
    gross_long = long_weights.sum() if len(long_weights) > 0 else 0.0
    gross_short = abs(short_weights.sum()) if len(short_weights) > 0 else 0.0
    
    # Cash position (uninvested capital)
    # For long_only: cash = 1.0 - gross_long
    # For short_only: cash = 1.0 - gross_short
    # For long/short: cash = 0 (fully invested on both sides)
    cash_weight = max(0.0, 1.0 - gross_long - gross_short)
    
    # Convert annual cash rate to holding period return
    # Use 365 calendar days (not 252 trading days) for consistent annual rate conversion
    days_per_year = 365  # Calendar days
    holding_period_return = config.portfolio.cash_rate * (config.time.HOLDING_PERIOD_DAYS / days_per_year)
    cash_ret = cash_weight * holding_period_return
    
    # ===== Calculate turnover and transaction costs =====
    turnover_long = 0.0
    turnover_short = 0.0
    
    if prev_long_weights is not None:
        # Align previous and current weights
        all_tickers = long_weights.index.union(prev_long_weights.index)
        curr_w = long_weights.reindex(all_tickers, fill_value=0.0)
        prev_w = prev_long_weights.reindex(all_tickers, fill_value=0.0)
        turnover_long = 0.5 * (curr_w - prev_w).abs().sum()
    
    if prev_short_weights is not None:
        # Align previous and current weights (shorts are negative)
        all_tickers = short_weights.index.union(prev_short_weights.index)
        curr_w = short_weights.reindex(all_tickers, fill_value=0.0)
        prev_w = prev_short_weights.reindex(all_tickers, fill_value=0.0)
        turnover_short = 0.5 * (curr_w - prev_w).abs().sum()
    
    total_turnover = turnover_long + turnover_short
    
    # Transaction costs: cost_bps * turnover / 10000
    # Costs are per side, and turnover already captures one-way trading
    cost_bps = config.portfolio.total_cost_bps_per_side
    transaction_cost = cost_bps * total_turnover / 10000.0
    
    # ===== Calculate borrowing costs =====
    # IMPORTANT: Borrowing costs apply to BOTH long AND short positions
    # 
    # For LONG positions:
    #   - In practice, you're borrowing cash to buy securities (margin buying)
    #   - You pay interest on the borrowed amount
    #   - With long_leverage = 1.0, you use 100% of capital (no borrowing)
    #   - With long_leverage > 1.0, you borrow: borrow_amount = (long_leverage - 1.0) * capital
    # 
    # For SHORT positions:
    #   - You borrow securities to sell them
    #   - You pay interest on the FULL notional value borrowed
    #   - With margin = 50%, you post 50% collateral but borrow 100% notional
    # 
    # Cost = borrow_cost * borrowed_notional * (holding_days / 365)
    # NOTE: Use 365 calendar days for consistent annual rate conversion
    
    borrow_cost_ret = 0.0
    
    # Borrowing cost for longs (only if leveraged beyond 1.0)
    if gross_long > 1.0:
        borrowed_long = gross_long - 1.0
        borrow_cost_ret += (config.portfolio.borrow_cost * 
                           borrowed_long * 
                           (config.time.HOLDING_PERIOD_DAYS / 365.0))
    
    # Borrowing cost for shorts (on full notional)
    if gross_short > 0:
        borrow_cost_ret += (config.portfolio.borrow_cost * 
                           gross_short * 
                           (config.time.HOLDING_PERIOD_DAYS / 365.0))
    
    # Total return (long + short + cash - transaction costs - borrow costs)
    ls_return = long_ret + short_ret + cash_ret - transaction_cost - borrow_cost_ret
    
    return {
        'date': t0,
        'long_ret': long_ret,
        'short_ret': short_ret,
        'cash_ret': cash_ret,
        'cash_weight': cash_weight,
        'turnover': total_turnover,
        'turnover_long': turnover_long,
        'turnover_short': turnover_short,
        'transaction_cost': transaction_cost,
        'borrow_cost': borrow_cost_ret,
        'ls_return': ls_return,
        'n_long': len(long_weights),
        'n_short': len(short_weights),
        'long_tickers': long_weights.index.tolist() if len(long_weights) > 0 else [],
        'short_tickers': short_weights.index.tolist() if len(short_weights) > 0 else [],
    }


if __name__ == "__main__":
    print("Portfolio construction module loaded.")
    print("\nKey functions:")
    print("  - construct_portfolio(scores, metadata, config)")
    print("  - evaluate_portfolio_return(panel, t0, long_wts, short_wts, config)")
    print("\nMethods:")
    print("  - 'cvxpy': Optimal (requires cvxpy)")
    print("  - 'simple': Fast approximation")
