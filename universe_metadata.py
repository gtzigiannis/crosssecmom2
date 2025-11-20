"""
Universe Metadata Module
=========================
Handles ETF family classification, duplicate detection, theme clustering,
and portfolio constraint parameters (cluster caps, per-ETF caps).

This module implements ETF labeling and clustering to:
1. Assign each ETF to an economic family
2. Identify and remove duplicate ETFs (same family, high correlation)
3. Cluster ETFs into theme groups by correlation
4. Define per-cluster and per-ETF portfolio caps
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Set, Optional, Tuple
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
import warnings


def assign_family(row: pd.Series) -> str:
    """
    Assign each ETF to a high-level economic family for cross-sectional momentum.
    
    Uses yfinance `category` plus ETF `name` and sometimes the ticker to classify
    ETFs into families like:
    - EQ_US_SIZE_STYLE (US large/mid/small cap blend/growth/value)
    - EQ_SECTOR_TECH, EQ_SECTOR_HEALTH, etc. (sector equity)
    - EQ_EM_BROAD, EQ_DEV_EUROPE_BROAD (geographic equity)
    - FI_US_TREASURY_SHORT, FI_US_CORP_IG (fixed income)
    - COMMODITY_METALS, COMMODITY_ENERGY (commodities)
    - TRADING_LEVERAGED_EQUITY, ALT_VOLATILITY, ALT_CRYPTO (excluded from core)
    
    Parameters
    ----------
    row : pd.Series
        Row from ETF metadata DataFrame with 'category', 'name', 'ticker' columns
        
    Returns
    -------
    str
        Family label
    """
    cat = (row.get("category") or "").strip()
    name = (row.get("name") or "").strip()
    tck = (row.get("ticker") or "").strip().upper()
    c = cat.lower()
    n = name.lower()
    
    # --- 1. Cash / ultra-short ---
    if "ultrashort bond" in c:
        return "CASH_EQUIVALENT"
    
    # --- 2. Trading / leveraged / inverse / vol products ---
    if "trading--inverse" in c:
        if "commodities" in c:
            return "TRADING_INVERSE_COMMODITY"
        else:
            return "TRADING_INVERSE_EQUITY"
    if "trading--leveraged" in c:
        return "TRADING_LEVERAGED_EQUITY"
    if "trading--miscellaneous" in c or "volatility" in n:
        return "ALT_VOLATILITY"
    
    # --- 3. Digital assets ---
    if "digital assets" in c or "bitcoin" in n or "crypto" in n:
        return "ALT_CRYPTO"
    
    # --- 4. Commodities ---
    if "commodities focused" in c:
        if any(x in n for x in ["gold", "silver"]):
            return "COMMODITY_METALS"
        elif "oil" in n or "crude" in n:
            return "COMMODITY_ENERGY"
        else:
            return "COMMODITY_BROAD"
    if tck in ["GLD", "SLV", "IAU"]:
        return "COMMODITY_METALS"
    
    # --- 5. Fixed income ---
    if any(
        x in c
        for x in [
            "corporate bond",
            "emerging markets bond",
            "muni",
            "short government",
            "short-term bond",
        ]
    ) or "bond" in c:
        # Muni
        if "muni" in c:
            return "FI_US_MUNI"
        # EM sovereign / corp
        if "emerging markets bond" in c:
            return "FI_EM_SOV_CREDIT"
        # IG corporates
        if "corporate bond" in c:
            return "FI_US_CORP_IG"
        # Short and long UST
        if "short government" in c:
            return "FI_US_TREASURY_SHORT"
        if "long government" in c or "20+ year" in n:
            return "FI_US_TREASURY_LONG"
        # Global aggregate
        if "world bond" in c or "global bond" in c:
            return "FI_GLOBAL_AGG"
        # Short-term bond (non-ultra)
        if "short-term bond" in c:
            return "FI_US_SHORT_TERM"
        # Aggregate / total bond
        if "intermediate core" in c or "total bond" in n:
            return "FI_US_AGG"
        return "FI_OTHER"
    
    # --- 6. Real estate / REITs ---
    if "real estate" in c or "reit" in n:
        return "EQ_SECTOR_REAL_ESTATE"
    
    # --- 7. Global / international / EM / Europe ---
    if "diversified emerging mkts" in c:
        return "EQ_EM_BROAD"
    if "europe stock" in c:
        return "EQ_DEV_EUROPE_BROAD"
    if "foreign large" in c or "foreign small/mid" in c:
        # EAFE vs broad ex-US
        if "eafe" in n:
            return "EQ_DEV_EAFE_BROAD"
        else:
            return "EQ_DEV_EXUS_BROAD"
    
    # --- 8. Country / region single-country equity ---
    if (
        "country -" in c
        or "china region" in c
        or "india equity" in c
        or "japan stock" in c
        or "latin america stock" in c
    ):
        if any(
            x in c
            for x in [
                "china",
                "india",
                "latin america",
                "brazil",
            ]
        ) or any(
            x in n for x in ["china", "india", "brazil", "mexico", "latin america"]
        ):
            return "EQ_SINGLE_COUNTRY_EM"
        else:
            return "EQ_SINGLE_COUNTRY_DEV"
    
    # --- 9. Innovation / thematic growth (ARK, etc.) ---
    if any(
        x in n
        for x in [
            "innovation",
            "next generation",
            "genomics",
            "robotics",
            "autonomous",
        ]
    ) or name.startswith("ARK "):
        return "EQ_THEMATIC_GROWTH"
    
    # --- 10. US size/style buckets ---
    if cat in [
        "Large Blend",
        "Large Growth",
        "Large Value",
        "Mid-Cap Blend",
        "Mid-Cap Growth",
        "Mid-Cap Value",
        "Small Blend",
    ]:
        return "EQ_US_SIZE_STYLE"
    
    # --- 11. Sector equity ---
    sector_map = {
        "technology": "EQ_SECTOR_TECH",
        "health": "EQ_SECTOR_HEALTH",
        "financial": "EQ_SECTOR_FINANCIALS",
        "consumer cyclical": "EQ_SECTOR_CONSUMER",
        "consumer defensive": "EQ_SECTOR_CONSUMER",
        "industrials": "EQ_SECTOR_INDUSTRIALS",
        "utilities": "EQ_SECTOR_UTILITIES",
        "equity energy": "EQ_SECTOR_ENERGY",
        "natural resources": "EQ_SECTOR_MATERIALS_METALS",
        "equity precious metals": "EQ_SECTOR_MATERIALS_METALS",
    }
    if c in sector_map:
        return sector_map[c]
    
    # --- 12. Dividend / quality / low-vol factors ---
    if any(
        x in n
        for x in [
            "dividend",
            "high dividend",
            "income",
            "quality",
            "min vol",
            "minimum volatility",
            "low volatility",
            "low vol",
        ]
    ):
        return "EQ_FACTOR_INCOME_QUALITY"
    
    # --- 13. Fallback buckets ---
    if "world stock" in c or "global" in c:
        return "EQ_GLOBAL_BROAD"
    
    # Default: US core equity catch-all
    return "EQ_US_CORE_OTHER"


def find_duplicate_clusters(
    corr: pd.DataFrame,
    meta: pd.DataFrame,
    dup_corr_threshold: float = 0.99,
) -> pd.DataFrame:
    """
    Identify 'duplicate' ETFs: same family and correlation >= dup_corr_threshold.
    
    For each duplicate group, selects a canonical ETF based on highest 3-year dollar ADV.
    
    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix (index and columns are tickers)
    meta : pd.DataFrame
        ETF metadata with columns: ticker, family, dollar_adv_3y
    dup_corr_threshold : float
        Correlation threshold for considering ETFs as duplicates (default: 0.99)
        
    Returns
    -------
    pd.DataFrame
        Copy of meta with additional columns:
        - dup_group_id: Group ID for duplicates (NaN if not in a duplicate group)
        - is_dup_canonical: True for the canonical ETF per group
        - in_core_after_duplicates: Core universe flag excluding non-canonical duplicates
    """
    meta_idx = meta.set_index("ticker")
    families = meta_idx["family"]
    
    tickers = list(corr.index)
    n = len(tickers)
    
    # Build adjacency graph: edges when same family AND corr >= threshold
    adj: Dict[str, Set[str]] = {t: set() for t in tickers}
    for i in range(n):
        ti = tickers[i]
        for j in range(i + 1, n):
            tj = tickers[j]
            # Must be in same family
            if families.get(ti, None) != families.get(tj, None):
                continue
            # Must have high correlation
            if corr.loc[ti, tj] >= dup_corr_threshold:
                adj[ti].add(tj)
                adj[tj].add(ti)
    
    # Find connected components (duplicate groups)
    visited: Set[str] = set()
    dup_groups: List[List[str]] = []
    
    for t in tickers:
        if t in visited:
            continue
        # DFS to find connected component
        stack = [t]
        group = []
        while stack:
            cur = stack.pop()
            if cur in visited:
                continue
            visited.add(cur)
            group.append(cur)
            stack.extend(adj[cur] - visited)
        
        # Only keep groups with 2+ members
        if len(group) > 1:
            dup_groups.append(sorted(group))
    
    # Prepare output columns
    meta_out = meta.copy()
    meta_out["dup_group_id"] = np.nan
    meta_out["is_dup_canonical"] = False
    
    # Assign group IDs and select canonical by dollar ADV
    for gid, group in enumerate(dup_groups, start=1):
        group_meta = meta_idx.loc[group]
        
        # Choose canonical: highest dollar_adv_3y
        canonical_ticker = (
            group_meta["dollar_adv_3y"]
            .fillna(0.0)
            .astype(float)
            .idxmax()
        )
        
        # Mark all members of this group
        meta_out.loc[meta_out["ticker"].isin(group), "dup_group_id"] = gid
        # Mark the canonical one
        meta_out.loc[meta_out["ticker"] == canonical_ticker, "is_dup_canonical"] = True
    
    # Core universe after removing non-canonical duplicates
    meta_out["in_core_after_duplicates"] = meta_out["in_core_universe"]
    mask_non_canonical_dup = (
        meta_out["dup_group_id"].notna() & ~meta_out["is_dup_canonical"]
    )
    meta_out.loc[mask_non_canonical_dup, "in_core_after_duplicates"] = False
    
    return meta_out


def build_theme_clusters(
    corr: pd.DataFrame,
    max_within_cluster_corr: float = 0.85,
) -> pd.Series:
    """
    Hierarchical clustering on correlation matrix to produce 'theme clusters'.
    
    Clusters assets that have pairwise correlations above max_within_cluster_corr.
    Examples: gold complex (GLD, GDX, GDXJ), China complex (MCHI, ASHR, KWEB), etc.
    
    Parameters
    ----------
    corr : pd.DataFrame
        Correlation matrix (index and columns are tickers)
    max_within_cluster_corr : float
        Maximum correlation within clusters (default: 0.85)
        Distance threshold = 1 - max_within_cluster_corr
        
    Returns
    -------
    pd.Series
        cluster_id indexed by ticker
    """
    # Distance matrix: d_ij = 1 - rho_ij
    dist = 1.0 - corr.values
    # Ensure diagonal is exactly 0
    np.fill_diagonal(dist, 0.0)
    
    # Convert to condensed distance matrix
    condensed = squareform(dist, checks=False)
    
    # Average-linkage hierarchical clustering
    Z = linkage(condensed, method="average")
    
    # Distance threshold corresponding to max correlation inside clusters
    max_d = 1.0 - max_within_cluster_corr
    
    # Cut dendrogram at this threshold
    cluster_labels = fcluster(Z, t=max_d, criterion="distance")
    
    cluster_s = pd.Series(cluster_labels, index=corr.index, name="cluster_id")
    
    return cluster_s


def build_universe_metadata(
    meta_path: str,
    returns_df: Optional[pd.DataFrame] = None,
    dup_corr_threshold: float = 0.99,
    max_within_cluster_corr: float = 0.85,
    default_cluster_cap: float = 0.10,
    default_per_etf_cap: float = 0.05,
    high_risk_cluster_cap: float = 0.07,
    high_risk_families: Optional[Set[str]] = None,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Complete pipeline to build universe metadata from ETF metadata file.
    
    Steps:
    1. Load and clean ETF metadata
    2. Assign families
    3. Find and mark duplicates
    4. Build theme clusters (if returns_df provided)
    5. Assign cluster caps and per-ETF caps
    
    Parameters
    ----------
    meta_path : str
        Path to ETF metadata CSV (must have: ticker, name, category, avg_volume_3y, avg_close_3y)
    returns_df : pd.DataFrame, optional
        Return matrix for correlation-based clustering (index=dates, columns=tickers)
    dup_corr_threshold : float
        Correlation threshold for duplicates
    max_within_cluster_corr : float
        Max correlation within theme clusters
    default_cluster_cap : float
        Default max weight per cluster
    default_per_etf_cap : float
        Default max weight per ETF
    high_risk_cluster_cap : float
        Reduced cap for high-risk clusters
    high_risk_families : set, optional
        Set of family names considered high-risk
        
    Returns
    -------
    tuple of (pd.DataFrame, pd.Series)
        - universe_metadata: DataFrame with all metadata columns
        - cluster_caps: Series of cluster caps indexed by cluster_id
    """
    # --- 0. Load and clean metadata ---
    meta = pd.read_csv(meta_path)
    meta.columns = meta.columns.str.strip()
    
    # Clean avg_volume_3y
    meta["avg_volume_3y"] = (
        meta["avg_volume_3y"]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.strip()
    )
    meta["avg_volume_3y"] = pd.to_numeric(meta["avg_volume_3y"], errors="coerce")
    
    # Dollar ADV as liquidity proxy
    meta["dollar_adv_3y"] = meta["avg_volume_3y"] * meta["avg_close_3y"]
    
    # --- 1. Assign families ---
    meta["family"] = meta.apply(assign_family, axis=1)
    
    # Mark leveraged / inverse / vol products to exclude from core universe
    trading_families = {
        "TRADING_LEVERAGED_EQUITY",
        "TRADING_INVERSE_EQUITY",
        "TRADING_INVERSE_COMMODITY",
        "ALT_VOLATILITY",
        "ALT_CRYPTO",
    }
    meta["in_core_universe"] = ~meta["family"].isin(trading_families)
    
    # --- 2. Find duplicates (if returns provided) ---
    if returns_df is not None:
        # Align correlation matrix to tickers in core universe
        core_tickers = sorted(
            set(returns_df.columns) & set(meta.loc[meta["in_core_universe"], "ticker"])
        )
        
        if len(core_tickers) > 1:
            corr_core = returns_df[core_tickers].corr()
            meta = find_duplicate_clusters(corr_core, meta, dup_corr_threshold)
        else:
            warnings.warn("Insufficient tickers for duplicate detection, skipping...")
            meta["dup_group_id"] = np.nan
            meta["is_dup_canonical"] = False
            meta["in_core_after_duplicates"] = meta["in_core_universe"]
    else:
        # No returns provided, mark all as non-duplicates
        meta["dup_group_id"] = np.nan
        meta["is_dup_canonical"] = False
        meta["in_core_after_duplicates"] = meta["in_core_universe"]
    
    # --- 3. Build theme clusters (if returns provided) ---
    if returns_df is not None:
        # Correlation for post-duplicate core universe
        core_final_tickers = meta.loc[meta["in_core_after_duplicates"], "ticker"]
        core_final_tickers = sorted(
            set(core_final_tickers) & set(returns_df.columns)
        )
        
        if len(core_final_tickers) > 1:
            corr_core_final = returns_df[core_final_tickers].corr()
            cluster_ids = build_theme_clusters(corr_core_final, max_within_cluster_corr)
            
            # Attach cluster_id to meta
            meta = meta.merge(
                cluster_ids.rename("cluster_id"),
                left_on="ticker",
                right_index=True,
                how="left",
            )
        else:
            warnings.warn("Insufficient tickers for clustering, skipping...")
            meta["cluster_id"] = np.nan
    else:
        # No returns provided, no clustering
        meta["cluster_id"] = np.nan
    
    # --- 4. Assign cluster caps and per-ETF caps ---
    
    # Default cluster caps
    unique_clusters = meta.loc[
        meta["in_core_after_duplicates"] & meta["cluster_id"].notna(),
        "cluster_id"
    ].unique()
    
    cluster_caps = pd.Series(
        default_cluster_cap,
        index=sorted(unique_clusters),
        name="cluster_cap",
    )
    
    # Adjust caps for high-risk families
    if high_risk_families is None:
        high_risk_families = {
            "EQ_EM_BROAD",
            "EQ_SINGLE_COUNTRY_EM",
            "COMMODITY_METALS",
            "COMMODITY_ENERGY",
            "ALT_CRYPTO",
        }
    
    family_cluster = (
        meta.loc[
            meta["in_core_after_duplicates"] & meta["cluster_id"].notna(),
            ["cluster_id", "family"]
        ]
        .drop_duplicates()
    )
    
    for cid in cluster_caps.index:
        fams = set(family_cluster.loc[family_cluster["cluster_id"] == cid, "family"])
        if fams & high_risk_families:
            cluster_caps.loc[cid] = high_risk_cluster_cap
    
    # Per-ETF caps (add to meta)
    meta["per_etf_cap"] = default_per_etf_cap
    
    # Add cluster_cap to meta (for convenience)
    meta = meta.merge(
        cluster_caps.rename("cluster_cap"),
        left_on="cluster_id",
        right_index=True,
        how="left"
    )
    
    return meta, cluster_caps


if __name__ == "__main__":
    # Example usage
    import sys
    
    # This would normally be run with actual data
    print("Universe metadata module loaded.")
    print("\nExample functions:")
    print("  - assign_family(row): Assign ETF to economic family")
    print("  - find_duplicate_clusters(corr, meta): Find duplicate ETFs")
    print("  - build_theme_clusters(corr): Cluster by correlation")
    print("  - build_universe_metadata(path): Complete pipeline")
