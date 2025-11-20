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
    enforce_caps: bool = False
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple portfolio construction: top/bottom quantiles with equal weights.
    
    Optionally enforces caps via simple scaling (not optimal, but fast).
    
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
        
    Returns
    -------
    tuple of (pd.Series, pd.Series)
        (long_weights, short_weights) indexed by ticker
    """
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
    long_weights = pd.Series(1.0 / n_long, index=long_tickers)
    short_weights = pd.Series(-1.0 / n_short, index=short_tickers)
    
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
    config
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
        
    Returns
    -------
    tuple of (pd.Series, pd.Series)
        (long_weights, short_weights) indexed by ticker
    """
    if not CVXPY_AVAILABLE:
        warnings.warn("CVXPY not available, using simple construction")
        return construct_portfolio_simple(scores, universe_metadata, config, enforce_caps=True)
    
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
    
    # ===== Long portfolio =====
    if len(long_tickers) > 0:
        long_weights = _optimize_one_side(
            tickers=long_tickers,
            scores=scores[long_tickers],
            metadata=metadata_subset.loc[long_tickers],
            target_gross=1.0,
            side='long',
            config=config
        )
    else:
        long_weights = pd.Series(dtype=float)
    
    # ===== Short portfolio =====
    if len(short_tickers) > 0:
        short_weights = _optimize_one_side(
            tickers=short_tickers,
            scores=scores[short_tickers],
            metadata=metadata_subset.loc[short_tickers],
            target_gross=1.0,
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
    method: str = 'cvxpy'
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
        
    Returns
    -------
    tuple of (pd.Series, pd.Series, dict)
        - long_weights: Long positions
        - short_weights: Short positions
        - portfolio_stats: Dict with diagnostics
    """
    if method == 'cvxpy' and CVXPY_AVAILABLE:
        long_weights, short_weights = construct_portfolio_cvxpy(scores, universe_metadata, config)
    else:
        long_weights, short_weights = construct_portfolio_simple(
            scores, universe_metadata, config, enforce_caps=True
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
    config
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
    
    # Total return (long + short, where short_weights are negative)
    ls_return = long_ret + short_ret
    
    return {
        'date': t0,
        'long_ret': long_ret,
        'short_ret': short_ret,
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
