"""
Simulate the Economic Prior Filter on Real Data
================================================

This script demonstrates the practical impact of applying economic priors
to your actual feature set with mock ICs.
"""

import numpy as np
import pandas as pd
from economic_priors import EconomicPriorFilter, get_prior_for_feature, ExpectedSign


def simulate_prior_filter_impact():
    """
    Simulate the impact of economic priors with realistic IC distributions.
    
    Key insight: If ICs are mostly noise (mean ~0), and we enforce sign
    consistency with theory, we reject ~50% of features just from wrong signs.
    Combined with IC magnitude requirements, we reject 70-90% of features.
    """
    
    print("="*70)
    print("SIMULATING ECONOMIC PRIOR FILTER IMPACT")
    print("="*70)
    
    # Get actual feature names from panel
    import pyarrow.parquet as pq
    panel_path = r"D:\REPOSITORY\Data\crosssecmom2\cs_momentum_features.parquet"
    pf = pq.ParquetFile(panel_path)
    cols = [f.name for f in pf.schema_arrow]
    exclude = {'Date', 'Ticker', 'Open', 'High', 'Low', 'Close', 'Volume', 
               'Dividends', 'Stock Splits', 'y_cs_21d', 'FwdRet_21'}
    features = [c for c in cols if c not in exclude]
    
    print(f"\nTotal features: {len(features)}")
    
    # Simulate ICs with realistic distribution
    # Reality: Most features have IC ~ N(0, 0.02) - mostly noise
    np.random.seed(42)
    
    # Simulate different scenarios
    scenarios = {
        "Pure noise (μ=0)": (0.0, 0.02),
        "Weak signal (μ=0.005)": (0.005, 0.02),
        "Moderate signal (μ=0.01)": (0.01, 0.02),
    }
    
    for scenario_name, (mean_ic, std_ic) in scenarios.items():
        print(f"\n{'='*60}")
        print(f"SCENARIO: {scenario_name}")
        print(f"{'='*60}")
        
        # Generate ICs for all features
        ics = np.random.normal(mean_ic, std_ic, len(features))
        feature_ics = dict(zip(features, ics))
        
        # Analyze by prior type
        prior_groups = {
            'positive': [],
            'negative': [],
            'either': [],
            'none': [],
        }
        
        for feat, ic in feature_ics.items():
            prior = get_prior_for_feature(feat)
            if prior is None:
                prior_groups['none'].append((feat, ic))
            else:
                prior_groups[prior.expected_sign.value].append((feat, ic))
        
        print(f"\nFeatures by expected sign:")
        for sign, items in prior_groups.items():
            print(f"  {sign}: {len(items)}")
        
        # Apply filter
        print(f"\nApplying filter...")
        filter = EconomicPriorFilter(
            require_sign_match=True,
            allow_unprioried_features=False,
            min_prior_confidence=0.3,
        )
        
        passed_ics = filter.filter_features(feature_ics, verbose=False)
        
        # Analyze what passed
        passed_by_sign = {'positive': 0, 'negative': 0, 'either': 0}
        for feat in passed_ics:
            prior = get_prior_for_feature(feat)
            if prior:
                passed_by_sign[prior.expected_sign.value] = \
                    passed_by_sign.get(prior.expected_sign.value, 0) + 1
        
        print(f"\nRESULTS:")
        print(f"  Total passed: {len(passed_ics)} / {len(features)} "
              f"({100*len(passed_ics)/len(features):.1f}%)")
        print(f"  Rejected: {len(features) - len(passed_ics)} "
              f"({100*(1-len(passed_ics)/len(features)):.1f}%)")
        
        print(f"\n  Breakdown of rejections:")
        print(f"    - Forbidden:   {filter.stats['rejected_forbidden']}")
        print(f"    - No prior:    {filter.stats['rejected_no_prior']}")
        print(f"    - Wrong sign:  {filter.stats['rejected_wrong_sign']}")
        print(f"    - Low IC:      {filter.stats['rejected_low_ic']}")
        
        print(f"\n  Passed by expected sign:")
        for sign, count in passed_by_sign.items():
            orig = len([x for x in prior_groups.get(sign, []) if x])
            if orig > 0:
                print(f"    - {sign}: {count} / {orig} ({100*count/orig:.1f}%)")
    
    # =========================================================================
    # KEY INSIGHT: Show what types of features survive
    # =========================================================================
    print("\n" + "="*70)
    print("KEY INSIGHT: Features Most Likely to Survive")
    print("="*70)
    
    # Use weak signal scenario
    np.random.seed(42)
    ics = np.random.normal(0.005, 0.02, len(features))
    
    # Make some features have signal in the CORRECT direction
    # This simulates what happens if momentum features truly work
    for i, feat in enumerate(features):
        prior = get_prior_for_feature(feat)
        if prior and prior.expected_sign == ExpectedSign.POSITIVE:
            # Momentum features: add positive signal
            if 'Mom' in feat or '%' in feat:
                ics[i] = np.random.normal(0.025, 0.01)  # True signal
        elif prior and prior.expected_sign == ExpectedSign.NEGATIVE:
            # Volatility features: add negative signal
            if 'std' in feat or 'vol' in feat.lower():
                ics[i] = np.random.normal(-0.02, 0.01)  # True signal
    
    feature_ics = dict(zip(features, ics))
    
    filter = EconomicPriorFilter(
        require_sign_match=True,
        allow_unprioried_features=False,
    )
    passed_ics = filter.filter_features(feature_ics, verbose=True)
    
    # Show top survivors
    print("\nTop 20 features that passed (by |IC|):")
    sorted_passed = sorted(passed_ics.items(), key=lambda x: -abs(x[1]))
    for feat, ic in sorted_passed[:20]:
        prior = get_prior_for_feature(feat)
        sign = prior.expected_sign.value if prior else 'none'
        print(f"  {feat}: IC={ic:+.4f} (expected={sign})")


if __name__ == "__main__":
    simulate_prior_filter_impact()
