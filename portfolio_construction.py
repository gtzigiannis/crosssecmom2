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

# cvxpy is optional - only needed for optimized portfolio construction
try:
    import cvxpy as cp
    CVXPY_AVAILABLE = True
except ImportError:
    CVXPY_AVAILABLE = False
    cp = None


def construct_portfolio_simple(
    scores: pd.Series,
    universe_metadata: pd.DataFrame,
    config,
    enforce_caps: bool = False,
    mode: str = 'ls'
) -> Tuple[pd.Series, pd.Series]:
    """
    Simple portfolio construction: top/bottom quantiles with equal weights.
    
    Returns weights as leverage multipliers (dimensionless, relative to equity).
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
        (long_weights, short_weights) as leverage multipliers (dimensionless)
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
    
    # Get target leverage from margin regime (dimensionless ratios)
    max_exposure = config.portfolio.compute_max_exposure(capital=1.0)
    long_target_gross = max_exposure['long_exposure']
    short_target_gross = max_exposure['short_exposure']
    
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
    
    # Build long side with target exposure from margin regime
    if build_long and n_long > 0:
        long_weights = pd.Series(long_target_gross / n_long, index=long_tickers)
    else:
        long_weights = pd.Series(dtype=float)
    
    # Build short side with target exposure from margin regime
    if build_short and n_short > 0:
        short_weights = pd.Series(-short_target_gross / n_short, index=short_tickers)
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
    
    # FIX 2: Rescale to target exposures (not to 1.0)
    # This preserves the leverage implied by compute_max_exposure
    gross_long = long_weights.abs().sum() if len(long_weights) > 0 else 0.0
    gross_short = short_weights.abs().sum() if len(short_weights) > 0 else 0.0
    
    if gross_long > 0:
        long_weights = long_weights * (long_target_gross / gross_long)
    if gross_short > 0:
        short_weights = short_weights * (short_target_gross / gross_short)
    
    return long_weights, short_weights


def construct_portfolio_cvxpy(
    scores: pd.Series,
    universe_metadata: pd.DataFrame,
    config,
    mode: str = 'ls'
) -> Tuple[pd.Series, pd.Series]:
    """
    Portfolio construction with CVXPY optimization to enforce caps exactly.
    
    Returns weights as leverage multipliers (dimensionless, relative to equity).
    
    Formulation:
    - Maximize: sum of (score_i * weight_i) for long + short sides
    - Subject to:
      * w_i >= 0 for long, w_i <= 0 for short
      * |w_i| <= per_etf_cap_i
      * sum_{i in cluster_k} |w_i| <= cluster_cap_k
      * sum(long_weights) = target_long_gross (leverage ratio)
      * sum(abs(short_weights)) = target_short_gross (leverage ratio)
    
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
        (long_weights, short_weights) as leverage multipliers (dimensionless)
    """
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
    
    # Get target leverage from margin regime (dimensionless ratios)
    max_exposure = config.portfolio.compute_max_exposure(capital=1.0)
    long_target_gross = max_exposure['long_exposure']
    short_target_gross = max_exposure['short_exposure']
    
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
    
    Requires cvxpy to be installed.
    
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
    if not CVXPY_AVAILABLE:
        raise ImportError("cvxpy is required for optimized portfolio construction. "
                         "Use portfolio_method='simple' instead or install cvxpy.")
    
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
    
    # ADAPTIVE PER-ETF CAPS: Ensure feasibility for any universe size
    # If sum of caps < target, relax caps proportionally
    per_etf_caps_config = metadata['per_etf_cap'].values
    sum_caps = per_etf_caps_config.sum()
    
    if sum_caps < target_gross * 0.99:  # Insufficient capacity
        # Relax caps proportionally with 10% buffer
        scale_factor = (target_gross * 1.1) / sum_caps
        per_etf_caps = per_etf_caps_config * scale_factor
        
        warnings.warn(
            f"[{side}] Adaptive caps: {n} positions, target={target_gross:.2f}, "
            f"original sum_caps={sum_caps:.2f}. Relaxing by {scale_factor:.2f}x -> "
            f"new range=[{per_etf_caps.min():.3f}, {per_etf_caps.max():.3f}]"
        )
    else:
        per_etf_caps = per_etf_caps_config
    
    constraints.append(w <= per_etf_caps)
    
    # ADAPTIVE CLUSTER CAPS: Similar logic for clusters
    cluster_ids = metadata['cluster_id'].values
    unique_clusters = [c for c in np.unique(cluster_ids) if pd.notna(c)]
    
    if len(unique_clusters) > 0:
        cluster_cap_dict = {}
        total_cluster_capacity = 0.0
        
        for cid in unique_clusters:
            cluster_mask = (cluster_ids == cid)
            cluster_cap_config = metadata.loc[cluster_mask, 'cluster_cap'].iloc[0]
            cluster_cap_dict[cid] = cluster_cap_config
            total_cluster_capacity += cluster_cap_config
        
        if total_cluster_capacity < target_gross * 0.99:
            # Relax cluster caps proportionally
            scale_factor = (target_gross * 1.1) / total_cluster_capacity
            warnings.warn(
                f"[{side}] Adaptive cluster caps: {len(unique_clusters)} clusters, "
                f"target={target_gross:.2f}, capacity={total_cluster_capacity:.2f}. "
                f"Relaxing by {scale_factor:.2f}x"
            )
            for cid in cluster_cap_dict:
                cluster_cap_dict[cid] *= scale_factor
        
        # Add cluster constraints with adaptive caps
        for cid in unique_clusters:
            cluster_mask = (cluster_ids == cid)
            constraints.append(cp.sum(w[cluster_mask]) <= cluster_cap_dict[cid])
    
    # Solve optimization
    # Try solvers in order of preference: CLARABEL (modern, robust), OSQP, SCS
    problem = cp.Problem(objective, constraints)
    
    try:
        # CLARABEL is a modern interior-point solver that handles QP/SOCP well
        # It's the recommended replacement for ECOS
        for solver in [cp.CLARABEL, cp.OSQP, cp.SCS]:
            try:
                problem.solve(solver=solver, verbose=False)
                if problem.status in ['optimal', 'optimal_inaccurate']:
                    break
            except Exception:
                continue
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            warnings.warn(f"[{side}] Optimization status: {problem.status}, using equal weights")
            weights_arr = np.ones(n) / n * target_gross
        else:
            weights_arr = w.value
            
    except Exception as e:
        warnings.warn(f"[{side}] Optimization failed: {e}, using equal weights")
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
    
    Returns weights as leverage multipliers (dimensionless, relative to equity).
    
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
        - long_weights: Long positions as leverage multipliers
        - short_weights: Short positions as leverage multipliers
        - portfolio_stats: Dict with diagnostics
    """
    if method == 'cvxpy':
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
    # CRITICAL: Cash position depends on MARGIN REQUIREMENTS, not gross exposures
    # 
    # Margin requirements determine how much COLLATERAL you need to post:
    # - Long positions: long_margin_req × gross_long (typically 50% for Reg T)
    # - Short positions: short_margin_req × gross_short (typically 50% for ETFs)
    # 
    # Example: 100% long + 50% short with 50% margin on both:
    #   long_capital = 1.0 × 0.5 = 0.50
    #   short_capital = 0.5 × 0.5 = 0.25
    #   total_margin = 0.75
    #   cash_weight = 1.0 - 0.75 = 0.25 (25% cash)
    
    gross_long = long_weights.sum() if len(long_weights) > 0 else 0.0
    gross_short = abs(short_weights.sum()) if len(short_weights) > 0 else 0.0
    
    # FIX 3: Use active margins from margin regime (not deprecated parameters)
    margin_long, margin_short = config.portfolio.get_active_margins()
    
    # Calculate capital tied up as margin
    long_margin_capital = gross_long * margin_long
    short_margin_capital = gross_short * margin_short
    total_margin_capital = long_margin_capital + short_margin_capital
    
    # Remaining cash (can be negative if insufficient capital!)
    cash_weight = 1.0 - total_margin_capital
    
    # Warn if insufficient capital
    if cash_weight < 0:
        import warnings
        warnings.warn(
            f"Insufficient capital at {t0}: need {-cash_weight:.2%} more. "
            f"Long margin: {long_margin_capital:.2%}, Short margin: {short_margin_capital:.2%}. "
            f"Consider reducing positions or increasing leverage."
        )
        cash_weight = 0.0  # Can't have negative cash in practice
    
    # Convert annual cash rate to holding period return
    # Use 365 calendar days (not 252 trading days) for consistent annual rate conversion
    days_per_year = 365  # Calendar days
    holding_period_return = config.portfolio.cash_rate * (config.time.HOLDING_PERIOD_DAYS / days_per_year)
    cash_ret = cash_weight * holding_period_return
    
    # ===== Calculate turnover and transaction costs =====
    # IMPORTANT: Turnover measures one-way trading volume
    # - For initial entry (prev_weights = None): charge on full position size
    # - For rebalancing: charge on net position changes (0.5 factor for one-way)
    
    turnover_long = 0.0
    turnover_short = 0.0
    
    if prev_long_weights is not None:
        # Rebalancing: calculate net position changes
        all_tickers = long_weights.index.union(prev_long_weights.index)
        curr_w = long_weights.reindex(all_tickers, fill_value=0.0)
        prev_w = prev_long_weights.reindex(all_tickers, fill_value=0.0)
        # 0.5 factor converts round-trip turnover to one-way turnover
        turnover_long = 0.5 * (curr_w - prev_w).abs().sum()
    else:
        # CRITICAL FIX: First period - charge for initial entry
        # No 0.5 factor because we're entering from cash (one-way only)
        turnover_long = long_weights.abs().sum()
    
    if prev_short_weights is not None:
        # Rebalancing: calculate net position changes
        all_tickers = short_weights.index.union(prev_short_weights.index)
        curr_w = short_weights.reindex(all_tickers, fill_value=0.0)
        prev_w = prev_short_weights.reindex(all_tickers, fill_value=0.0)
        # 0.5 factor converts round-trip turnover to one-way turnover
        turnover_short = 0.5 * (curr_w - prev_w).abs().sum()
    else:
        # CRITICAL FIX: First period - charge for initial entry
        # No 0.5 factor because we're entering from cash (one-way only)
        turnover_short = short_weights.abs().sum()
    
    total_turnover = turnover_long + turnover_short
    
    # Transaction costs: cost_bps * turnover / 10000
    # Costs are per side, and turnover already captures one-way trading
    cost_bps = config.portfolio.total_cost_bps_per_side
    transaction_cost = cost_bps * total_turnover / 10000.0
    
    # ===== Calculate borrowing costs =====
    # IMPORTANT: Borrowing costs apply whenever we don't fully fund positions with our capital
    # 
    # For LONG positions (margin buying):
    #   - We post margin_req × position_size as collateral
    #   - We borrow the rest: (1 - margin_req) × position_size
    #   - Example: 100% long with 50% margin:
    #     * Post 50% margin (our capital)
    #     * Borrow 50% (from broker)
    #     * Pay MARGIN INTEREST on 50% borrowed
    #   - Example: 150% long with 50% margin:
    #     * Post 75% margin (50% of 150% position)
    #     * Borrow 75% (remaining 150% - 75%)
    #     * Pay MARGIN INTEREST on 75% borrowed
    #   - Borrowed amount = gross_long × (1 - margin_req)
    # 
    # For SHORT positions (security borrowing):
    #   - Always borrowing securities (regardless of margin requirement)
    #   - Borrowed amount = FULL notional shorted
    #   - Margin requirement determines COLLATERAL needed (not interest base)
    #   - Example: 50% short with 50% margin:
    #     * Collateral posted: 25% of capital (50% × 50%)
    #     * SHORT BORROW FEE paid on: 50% of capital (full notional)
    # 
    # CRITICAL: Use SEPARATE rates for shorts vs longs
    #   - Short borrow fee: cost to borrow shares (paid to share lender)
    #   - Margin interest: cost to borrow cash (paid to broker)
    # 
    # Cost = rate × borrowed_notional × (holding_days / 365)
    # NOTE: Use 365 calendar days for consistent annual rate conversion
    
    short_borrow_cost = 0.0
    margin_interest_cost = 0.0
    
    # Apply zero_financing_mode if enabled (for diagnostic runs)
    if config.portfolio.zero_financing_mode:
        margin_interest_rate = 0.0
        short_borrow_rate = 0.0
    else:
        margin_interest_rate = config.portfolio.margin_interest_rate
        short_borrow_rate = config.portfolio.short_borrow_rate
    
    # Margin interest for longs (borrow the unfunded portion)
    if gross_long > 0:
        # We post (margin_long × gross_long) as collateral
        # We borrow the rest: gross_long × (1 - margin_long)
        margin_long, _ = config.portfolio.get_active_margins()
        borrowed_long = gross_long * (1.0 - margin_long)
        margin_interest_cost = (margin_interest_rate * 
                               borrowed_long * 
                               (config.time.HOLDING_PERIOD_DAYS / 365.0))
    
    # Short borrow fee (on full notional, not margin-adjusted)
    if gross_short > 0:
        short_borrow_cost = (short_borrow_rate * 
                            gross_short * 
                            (config.time.HOLDING_PERIOD_DAYS / 365.0))
    
    # Total financing cost
    borrow_cost_ret = short_borrow_cost + margin_interest_cost
    
    # Total return (long + short + cash - transaction costs - borrow costs)
    ls_return = long_ret + short_ret + cash_ret - transaction_cost - borrow_cost_ret
    
    # ===== CASH LEDGER - Complete Transparency =====
    # This ledger tracks all cash flows to ensure capital accounting is correct.
    # 
    # Starting point: We have 1.0 (100%) of capital available
    # 
    # STEP 1: Deploy capital to margin accounts
    #   - Long margin posted: gross_long × long_margin_req
    #   - Short margin posted: gross_short × short_margin_req
    #   - Total margin posted: long_margin_capital + short_margin_capital
    # 
    # STEP 2: Calculate remaining cash
    #   - Cash balance: 1.0 - total_margin_posted
    #   - This cash earns interest at cash_rate
    # 
    # STEP 3: Interest earned on cash
    #   - Interest: cash_balance × cash_rate × (days/365)
    #   - Added to account balance
    # 
    # STEP 4: Borrowing costs charged
    #   For longs: borrowed_long = gross_long × (1 - long_margin_req)
    #   For shorts: borrowed_short = gross_short (always full notional)
    #   Cost: borrow_cost × borrowed_amount × (days/365)
    #   - Subtracted from account balance
    # 
    # VERIFICATION: 
    #   total_capital_deployed = margin_posted + cash_balance = 1.0 [OK]
    #   net_financing = cash_interest - borrowing_costs
    #   final_capital = initial + asset_returns + net_financing - transaction_costs
    
    # Build comprehensive cash ledger
    cash_ledger = {
        # Initial state
        'initial_capital_weight': 1.0,  # Always start with 100% of capital
        
        # Capital deployment
        'long_margin_posted': long_margin_capital,
        'short_margin_posted': short_margin_capital,
        'total_margin_posted': total_margin_capital,
        'cash_balance': cash_weight,  # Remaining uninvested cash
        
        # Position exposures (for reference)
        'gross_long': gross_long,
        'gross_short': gross_short,
        'net_exposure': gross_long - gross_short,
        
        # Borrowing amounts (using active margins)
        'borrowed_long': gross_long * (1.0 - margin_long) if gross_long > 0 else 0.0,
        'borrowed_short': gross_short,  # Always full notional for shorts
        'total_borrowed': (gross_long * (1.0 - margin_long) if gross_long > 0 else 0.0) + gross_short,
        
        # Financing costs breakdown (as % return for the period)
        'cash_interest_earned': cash_ret,  # Interest on uninvested cash
        'short_borrow_cost': short_borrow_cost,  # Fee to borrow shares for shorting
        'margin_interest_cost': margin_interest_cost,  # Interest on cash borrowed for longs
        'borrowing_cost_charged': borrow_cost_ret,  # Total = short_borrow + margin_interest
        'net_financing_cost': cash_ret - borrow_cost_ret,  # Net of interest earned - costs paid
        
        # Transaction costs
        'transaction_cost': transaction_cost,
        
        # Asset returns
        'long_asset_return': long_ret,
        'short_asset_return': short_ret,
        'total_asset_return': long_ret + short_ret,
        
        # Total P&L breakdown
        'total_return': ls_return,  # = asset_returns + net_financing - transaction_costs
    }
    
    # Verification: margin posted + cash balance should equal initial capital
    capital_check = cash_ledger['total_margin_posted'] + cash_ledger['cash_balance']
    if abs(capital_check - 1.0) > 1e-6 and cash_ledger['cash_balance'] >= 0:
        import warnings
        warnings.warn(
            f"Cash ledger accounting error at {t0}: "
            f"Margin posted ({cash_ledger['total_margin_posted']:.4f}) + "
            f"Cash balance ({cash_ledger['cash_balance']:.4f}) = "
            f"{capital_check:.4f} != 1.0"
        )
    
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
        # NEW: Complete cash ledger for transparency
        'cash_ledger': cash_ledger,
    }


if __name__ == "__main__":
    print("Portfolio construction module loaded.")
    print("\nKey functions:")
    print("  - construct_portfolio(scores, metadata, config)")
    print("  - evaluate_portfolio_return(panel, t0, long_wts, short_wts, config)")
    print("\nMethods:")
    print("  - 'cvxpy': Optimal (requires cvxpy)")
    print("  - 'simple': Fast approximation")
