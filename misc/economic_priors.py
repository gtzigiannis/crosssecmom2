"""
Economic Prior Filter for Feature Selection
============================================

This module defines theoretically-grounded priors for which features should
predict returns and in what direction. Features must pass both:
1. Economic prior (theory says this should work)
2. Statistical validation (data confirms theory)

This prevents data-mining artifacts where statistically significant but
economically nonsensical patterns get selected.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
import re


class ExpectedSign(Enum):
    """Expected relationship between feature and forward returns."""
    POSITIVE = "positive"      # Higher feature value → higher returns
    NEGATIVE = "negative"      # Higher feature value → lower returns
    EITHER = "either"          # Theory allows both (e.g., volatility)
    FORBIDDEN = "forbidden"    # No economic rationale - exclude entirely


@dataclass
class FeaturePrior:
    """Economic prior for a feature family."""
    pattern: str                    # Regex pattern to match feature names
    expected_sign: ExpectedSign     # What sign IC should have
    rationale: str                  # Why we expect this relationship
    min_abs_ic: float = 0.02        # Minimum IC magnitude to be credible
    confidence: float = 1.0         # How confident in this prior (0-1)


# =============================================================================
# ECONOMIC PRIORS LIBRARY
# =============================================================================
# These are based on decades of academic research in asset pricing

MOMENTUM_PRIORS = [
    # Close%-N = N-day raw return (your naming convention)
    FeaturePrior(
        pattern=r"^Close%-(?:63|126|252)$",  # Medium/long-term momentum
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Momentum: Past winners continue winning (Jegadeesh & Titman 1993). "
                  "3-12 month momentum is most robust.",
        min_abs_ic=0.015,
        confidence=0.9
    ),
    FeaturePrior(
        pattern=r"^Close%-(?:21|42)$",  # 1-2 month momentum
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Short-term momentum: 1-2 month returns show continuation.",
        min_abs_ic=0.015,
        confidence=0.8
    ),
    FeaturePrior(
        pattern=r"^Close%-(?:1|2|3|5|10)$",  # Very short-term
        expected_sign=ExpectedSign.EITHER,  # Can reverse
        rationale="Very short-term returns may show reversal due to microstructure.",
        min_abs_ic=0.02,
        confidence=0.5
    ),
    # Close_Mom* features (alternative momentum naming)
    FeaturePrior(
        pattern=r"^Close_Mom(?:63|126|252)$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Medium/long-term momentum: 3-12 month is most robust.",
        min_abs_ic=0.015,
        confidence=0.85
    ),
    FeaturePrior(
        pattern=r"^Close_Mom(?:21|42)$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Short-term momentum: 1-2 month returns continuation.",
        min_abs_ic=0.015,
        confidence=0.75
    ),
    FeaturePrior(
        pattern=r"^Close_Mom(?:5|10)$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Very short momentum: May reverse due to microstructure.",
        min_abs_ic=0.02,
        confidence=0.5
    ),
    # Cross-sectional ranks of returns
    FeaturePrior(
        pattern=r"^Close%-\d+_Rank$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Cross-sectional momentum rank: Top ranked assets outperform.",
        min_abs_ic=0.02,
        confidence=0.85
    ),
    # Return divided by volatility (risk-adjusted momentum)
    FeaturePrior(
        pattern=r"^Close%-\d+_div_.*(?:std|vol|ATR)",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Risk-adjusted momentum: High return per unit risk is quality.",
        min_abs_ic=0.015,
        confidence=0.8
    ),
    # Return minus return (momentum change)
    FeaturePrior(
        pattern=r"^Close%-\d+_minus_Close%-\d+",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Momentum acceleration: Recent > past momentum is bullish.",
        min_abs_ic=0.015,
        confidence=0.7
    ),
    # Relative strength vs basket/benchmark
    FeaturePrior(
        pattern=r"^Rel(?:5|21|63|126|252)_vs_",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Relative strength: Outperformers vs benchmark continue.",
        min_abs_ic=0.02,
        confidence=0.8
    ),
]

VOLATILITY_PRIORS = [
    # Standard deviation features
    FeaturePrior(
        pattern=r"^Close_std\d+$",
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="Low volatility anomaly (Ang et al. 2006): Low vol stocks earn "
                  "higher risk-adjusted returns. Higher vol → lower returns.",
        min_abs_ic=0.015,
        confidence=0.8
    ),
    # Parkinson, Garman-Klass, Rogers-Satchell volatility estimators
    FeaturePrior(
        pattern=r"^(?:parkinson|garman_klass|rogers_satchell)_vol_\d+$",
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="Range-based volatility estimators: Same low-vol anomaly.",
        min_abs_ic=0.015,
        confidence=0.8
    ),
    # ATR (Average True Range)
    FeaturePrior(
        pattern=r"^Close_ATR\d+$",
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="ATR measures volatility - same low-vol anomaly applies.",
        min_abs_ic=0.015,
        confidence=0.7
    ),
    # Kurtosis (tail risk)
    FeaturePrior(
        pattern=r"^Close_kurt\d+$",
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="High kurtosis = fat tails = crash risk. Avoid high kurtosis.",
        min_abs_ic=0.01,
        confidence=0.6
    ),
    # Skewness
    FeaturePrior(
        pattern=r"^Close_skew\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Skewness direction is context-dependent (lottery vs quality).",
        min_abs_ic=0.01,
        confidence=0.4
    ),
    # Hurst exponent
    FeaturePrior(
        pattern=r"^Close_Hurst\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Hurst > 0.5 indicates trending (persistent) behavior.",
        min_abs_ic=0.015,
        confidence=0.6
    ),
    # RSI features (including RSI divided by other things)
    FeaturePrior(
        pattern=r"^Close_RSI\d+",
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="RSI mean reversion: High RSI = overbought, expect reversal.",
        min_abs_ic=0.01,
        confidence=0.5
    ),
    # Williams %R (similar to RSI)
    FeaturePrior(
        pattern=r"^Close_WilliamsR\d+$",
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="Williams %R: Like RSI, overbought/oversold indicator.",
        min_abs_ic=0.01,
        confidence=0.5
    ),
    # MACD features
    FeaturePrior(
        pattern=r"^Close_MACD",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="MACD: Positive MACD = upward momentum.",
        min_abs_ic=0.015,
        confidence=0.6
    ),
    # Relative volume
    FeaturePrior(
        pattern=r"^rel_volume_\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Relative volume: High volume can be buying OR selling.",
        min_abs_ic=0.015,
        confidence=0.4
    ),
    # Lagged closes (for autoregression)
    FeaturePrior(
        pattern=r"^Close_lag\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Lagged price: AR structure. Direction depends on context.",
        min_abs_ic=0.02,
        confidence=0.3
    ),
    # Days since events
    FeaturePrior(
        pattern=r"^days_since_",
        expected_sign=ExpectedSign.EITHER,
        rationale="Time since event: Context-dependent, no strong prior.",
        min_abs_ic=0.02,
        confidence=0.3
    ),
]

TREND_PRIORS = [
    # EMA crossover / distance from EMA
    FeaturePrior(
        pattern=r"^Close_EMA\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Price relative to EMA: Above EMA = uptrend = continuation.",
        min_abs_ic=0.015,
        confidence=0.7
    ),
    # Moving averages
    FeaturePrior(
        pattern=r"^Close_MA\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Price relative to MA: Above MA indicates uptrend.",
        min_abs_ic=0.015,
        confidence=0.7
    ),
    # Bollinger Bands
    FeaturePrior(
        pattern=r"^Close_BollUp\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Upper Bollinger: Can indicate strength OR overbought.",
        min_abs_ic=0.015,
        confidence=0.4
    ),
    FeaturePrior(
        pattern=r"^Close_BollLo\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Lower Bollinger: Can indicate weakness OR oversold.",
        min_abs_ic=0.015,
        confidence=0.4
    ),
    # Drawdown
    FeaturePrior(
        pattern=r"^Close_DD\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Low drawdown = quality/stability. High DD = distress.",
        min_abs_ic=0.015,
        confidence=0.7
    ),
    # Max drawdown - negative = avoid deep drawdown assets
    FeaturePrior(
        pattern=r"^Close_MDD\d+$",
        expected_sign=ExpectedSign.POSITIVE,  # Less negative MDD is better
        rationale="Shallower max drawdown indicates quality/stability.",
        min_abs_ic=0.015,
        confidence=0.7
    ),
]

# Additional volume and price-volume features
VOLUME_EXTENDED_PRIORS = [
    # Volume z-score
    FeaturePrior(
        pattern=r"^volume_zscore_\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Volume z-score: Extreme volume can be buying OR selling.",
        min_abs_ic=0.015,
        confidence=0.4
    ),
    # Volume trend
    FeaturePrior(
        pattern=r"^volume_trend_\d+_\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Volume trend: Rising volume can confirm OR precede reversal.",
        min_abs_ic=0.015,
        confidence=0.4
    ),
    # Price-volume correlation
    FeaturePrior(
        pattern=r"^pv_corr_\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Price-volume correlation: Positive = healthy trend.",
        min_abs_ic=0.015,
        confidence=0.5
    ),
    # OBV slope
    FeaturePrior(
        pattern=r"^obv_slope_\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="OBV slope: Rising OBV = accumulation = bullish.",
        min_abs_ic=0.015,
        confidence=0.5
    ),
    # Up/down volume ratio
    FeaturePrior(
        pattern=r"^up_down_vol_ratio_\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Up/down volume ratio: More up volume = bullish.",
        min_abs_ic=0.015,
        confidence=0.5
    ),
    # Volume breakout
    FeaturePrior(
        pattern=r"^volume_breakout_\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Volume breakout: Can signal either direction.",
        min_abs_ic=0.02,
        confidence=0.4
    ),
]

MEAN_REVERSION_PRIORS = [
    # RSI 
    FeaturePrior(
        pattern=r"^Close_RSI\d+$",
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="RSI mean reversion: Overbought (high RSI) tends to reverse. "
                  "Note: In strong trends, high RSI can persist.",
        min_abs_ic=0.01,
        confidence=0.5  # Lower confidence - context dependent
    ),
    # Z-score features
    FeaturePrior(
        pattern=r"^Close_.*Z$",  # Ends in Z (z-score)
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="Price z-score: Extreme prices may revert to mean.",
        min_abs_ic=0.01,
        confidence=0.5
    ),
    # cb = "clip bound" or similar normalization
    FeaturePrior(
        pattern=r"^Close%-\d+_cb$",
        expected_sign=ExpectedSign.POSITIVE,  # Same as momentum
        rationale="Clipped/bounded return - follow momentum direction.",
        min_abs_ic=0.015,
        confidence=0.7
    ),
]

VOLUME_PRIORS = [
    # ADV (Average Daily Volume)
    FeaturePrior(
        pattern=r"^ADV_\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Liquidity: High ADV means liquid. No directional return prior.",
        min_abs_ic=0.01,
        confidence=0.3
    ),
    FeaturePrior(
        pattern=r"^ADV_\d+_Rank$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Liquidity rank: Higher liquidity doesn't predict returns directly.",
        min_abs_ic=0.01,
        confidence=0.3
    ),
    # Volume per ATR
    FeaturePrior(
        pattern=r"^volume_per_atr$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Volume normalized by volatility - no clear directional prior.",
        min_abs_ic=0.01,
        confidence=0.3
    ),
]

QUALITY_PRIORS = [
    # Sharpe ratio
    FeaturePrior(
        pattern=r"^Close_Sharpe\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Quality: High recent Sharpe indicates good risk-adjusted "
                  "performance that may persist.",
        min_abs_ic=0.015,
        confidence=0.7
    ),
    # Sortino ratio
    FeaturePrior(
        pattern=r"^Close_Sortino\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Sortino ratio: Good downside-adjusted returns persist.",
        min_abs_ic=0.015,
        confidence=0.7
    ),
    # Calmar ratio
    FeaturePrior(
        pattern=r"^Close_Calmar\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Calmar ratio: Good return/drawdown ratio indicates quality.",
        min_abs_ic=0.015,
        confidence=0.6
    ),
]

CROSS_SECTIONAL_PRIORS = [
    # General rank suffix
    FeaturePrior(
        pattern=r".*_Rank$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Cross-sectional rank: Higher rank in universe → outperformance. "
                  "This is the essence of cross-sectional momentum.",
        min_abs_ic=0.02,
        confidence=0.8
    ),
]

# Regime/conditional features
REGIME_PRIORS = [
    # Crash flag - in crash, returns usually mean-revert
    FeaturePrior(
        pattern=r".*_in_crash_flag$",
        expected_sign=ExpectedSign.EITHER,  # Complex - crash entries can bounce OR cascade
        rationale="Crash regime: Uncertain - can get bounce OR further decline.",
        min_abs_ic=0.02,
        confidence=0.3
    ),
    # Meltup flag
    FeaturePrior(
        pattern=r".*_in_meltup_flag$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Meltup regime: Momentum continues but reversal risk high.",
        min_abs_ic=0.02,
        confidence=0.3
    ),
    # High/low volatility regimes
    FeaturePrior(
        pattern=r".*_in_high_vol$",
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="High vol regime: Higher risk, lower expected returns.",
        min_abs_ic=0.015,
        confidence=0.5
    ),
    FeaturePrior(
        pattern=r".*_in_low_vol$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Low vol regime: Lower risk, better risk-adjusted returns.",
        min_abs_ic=0.015,
        confidence=0.5
    ),
]

# Features that should be EXCLUDED - no economic rationale
FORBIDDEN_PATTERNS = [
    FeaturePrior(
        pattern=r"^day_of_week",
        expected_sign=ExpectedSign.FORBIDDEN,
        rationale="Calendar effects in ETFs are spurious. No economic basis.",
    ),
    FeaturePrior(
        pattern=r"^month_",
        expected_sign=ExpectedSign.FORBIDDEN,
        rationale="Monthly seasonality in diversified ETFs is spurious.",
    ),
]


# =============================================================================
# INTERACTION PRIORS - Most important for your feature set!
# =============================================================================

INTERACTION_PRIORS = [
    # Momentum × Volatility (risk-adjusted momentum - VERY valid)
    FeaturePrior(
        pattern=r"^Close%-\d+_x_Close_std\d+$",
        expected_sign=ExpectedSign.EITHER,  # Depends on normalization
        rationale="Momentum × volatility: Low-vol momentum is quality factor. "
                  "Direction depends on how interaction is computed.",
        min_abs_ic=0.02,
        confidence=0.7
    ),
    # Momentum × Momentum (momentum acceleration)
    FeaturePrior(
        pattern=r"^Close%-\d+_x_Close%-\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Momentum × Momentum: Consistent winners across horizons.",
        min_abs_ic=0.02,
        confidence=0.8
    ),
    # Momentum × Trend (EMA/MA)
    FeaturePrior(
        pattern=r"^Close%-\d+_x_Close_(?:EMA|MA)\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Momentum + trend alignment: Return in direction of trend.",
        min_abs_ic=0.02,
        confidence=0.75
    ),
    # Momentum × Quality (Sharpe, Sortino)
    FeaturePrior(
        pattern=r"^Close%-\d+_x_Close_(?:Sharpe|Sortino|Calmar)\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="High-quality momentum: Momentum from quality assets persists.",
        min_abs_ic=0.02,
        confidence=0.75
    ),
    # Momentum × Drawdown (avoid damaged momentum)
    FeaturePrior(
        pattern=r"^Close%-\d+_x_Close_DD\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Momentum with low drawdown: Healthy momentum, not bounce.",
        min_abs_ic=0.02,
        confidence=0.7
    ),
    # Trend × Volatility (trend quality)
    FeaturePrior(
        pattern=r"^Close_(?:EMA|MA)\d+_x_Close_std\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Trend × Vol: Low-vol trends are more reliable.",
        min_abs_ic=0.02,
        confidence=0.6
    ),
    # Volatility × Volatility (vol clustering)
    FeaturePrior(
        pattern=r"^Close_std\d+_x_Close_std\d+$",
        expected_sign=ExpectedSign.NEGATIVE,
        rationale="Vol × Vol: High vol at multiple horizons = persistent risk.",
        min_abs_ic=0.02,
        confidence=0.6
    ),
    # ADV × anything (liquidity interaction)
    FeaturePrior(
        pattern=r"^ADV_\d+_x_",
        expected_sign=ExpectedSign.EITHER,
        rationale="Liquidity interactions: No strong directional prior.",
        min_abs_ic=0.025,  # Higher bar
        confidence=0.4
    ),
    # Volatility estimators × momentum
    FeaturePrior(
        pattern=r"^(?:parkinson|garman|rogers).*_x_Close%-\d+$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Range-vol × momentum: Context dependent.",
        min_abs_ic=0.02,
        confidence=0.5
    ),
    # Hurst × momentum (trending momentum)
    FeaturePrior(
        pattern=r"^Close_Hurst\d+_x_Close%-\d+$",
        expected_sign=ExpectedSign.POSITIVE,
        rationale="Trending (high Hurst) momentum is more reliable.",
        min_abs_ic=0.02,
        confidence=0.6
    ),
    # Kurtosis/skew interactions - generally avoid
    FeaturePrior(
        pattern=r"^Close_(?:kurt|skew)\d+_x_",
        expected_sign=ExpectedSign.EITHER,
        rationale="Higher-moment interactions: Weak theoretical basis.",
        min_abs_ic=0.03,  # Higher bar
        confidence=0.3
    ),
    # Squared features × anything
    FeaturePrior(
        pattern=r".*_sq_x_|_x_.*_sq$",
        expected_sign=ExpectedSign.EITHER,
        rationale="Squared interactions: Non-linear but weak prior.",
        min_abs_ic=0.025,
        confidence=0.4
    ),
    # Catch-all for other interactions - STRICT
    FeaturePrior(
        pattern=r".*_x_.*",
        expected_sign=ExpectedSign.EITHER,
        rationale="Generic interaction - requires very strong statistical evidence.",
        min_abs_ic=0.035,  # MUCH higher bar for untheorized
        confidence=0.2     # Low confidence = high bar
    ),
]


# =============================================================================
# PRIOR REGISTRY
# =============================================================================

def get_all_priors() -> List[FeaturePrior]:
    """Get all economic priors in order of specificity."""
    # More specific patterns should come first
    return (
        FORBIDDEN_PATTERNS +
        MOMENTUM_PRIORS +
        VOLATILITY_PRIORS +
        VOLUME_EXTENDED_PRIORS +
        TREND_PRIORS +
        MEAN_REVERSION_PRIORS +
        VOLUME_PRIORS +
        QUALITY_PRIORS +
        CROSS_SECTIONAL_PRIORS +
        REGIME_PRIORS +
        INTERACTION_PRIORS  # Interactions last (catch-all at end)
    )


def get_prior_for_feature(feature_name: str) -> Optional[FeaturePrior]:
    """
    Find the most specific economic prior for a feature.
    
    Returns None if no prior matches (feature has no economic justification).
    """
    for prior in get_all_priors():
        if re.match(prior.pattern, feature_name, re.IGNORECASE):
            return prior
    return None


# =============================================================================
# ECONOMIC PRIOR FILTER
# =============================================================================

class EconomicPriorFilter:
    """
    Filter features based on economic priors.
    
    A feature passes if:
    1. It has a defined economic prior (not ad-hoc data mining)
    2. Its empirical IC sign matches the expected sign
    3. Its IC magnitude meets the minimum threshold
    4. It's not in the forbidden list
    
    This dramatically reduces the feature space to economically sensible
    candidates before statistical selection.
    """
    
    def __init__(
        self,
        require_sign_match: bool = True,
        allow_unprioried_features: bool = False,
        min_prior_confidence: float = 0.3,
    ):
        """
        Args:
            require_sign_match: If True, IC sign must match expected sign
            allow_unprioried_features: If True, features without priors can pass
                                       (but with higher IC bar)
            min_prior_confidence: Minimum confidence in prior to use it
        """
        self.require_sign_match = require_sign_match
        self.allow_unprioried_features = allow_unprioried_features
        self.min_prior_confidence = min_prior_confidence
        self.priors = get_all_priors()
        
        # Stats tracking
        self.stats = {
            'total_evaluated': 0,
            'passed': 0,
            'rejected_forbidden': 0,
            'rejected_no_prior': 0,
            'rejected_wrong_sign': 0,
            'rejected_low_ic': 0,
        }
    
    def reset_stats(self):
        """Reset filter statistics."""
        for key in self.stats:
            self.stats[key] = 0
    
    def evaluate_feature(
        self,
        feature_name: str,
        empirical_ic: float,
    ) -> Tuple[bool, str]:
        """
        Evaluate whether a feature should be included based on economic prior.
        
        Args:
            feature_name: Name of the feature
            empirical_ic: Empirical information coefficient (IC) from data
            
        Returns:
            (passed, reason): Whether feature passed and explanation
        """
        self.stats['total_evaluated'] += 1
        
        prior = get_prior_for_feature(feature_name)
        
        # Case 1: Feature is forbidden
        if prior and prior.expected_sign == ExpectedSign.FORBIDDEN:
            self.stats['rejected_forbidden'] += 1
            return False, f"Forbidden: {prior.rationale}"
        
        # Case 2: No prior defined
        if prior is None:
            if self.allow_unprioried_features:
                # Allow but require higher bar
                if abs(empirical_ic) >= 0.04:  # Higher bar for untheorized
                    self.stats['passed'] += 1
                    return True, "No prior but strong IC"
                else:
                    self.stats['rejected_low_ic'] += 1
                    return False, "No prior and IC too weak"
            else:
                self.stats['rejected_no_prior'] += 1
                return False, "No economic prior defined"
        
        # Case 3: Prior exists - check confidence threshold
        if prior.confidence < self.min_prior_confidence:
            # Treat as unprioried
            if abs(empirical_ic) >= 0.04:
                self.stats['passed'] += 1
                return True, f"Low-confidence prior ({prior.confidence:.2f}) but strong IC"
            else:
                self.stats['rejected_low_ic'] += 1
                return False, f"Low-confidence prior and weak IC"
        
        # Case 4: Check IC sign matches expected
        if self.require_sign_match and prior.expected_sign != ExpectedSign.EITHER:
            expected_positive = prior.expected_sign == ExpectedSign.POSITIVE
            actual_positive = empirical_ic > 0
            
            if expected_positive != actual_positive:
                self.stats['rejected_wrong_sign'] += 1
                return False, f"Sign mismatch: expected {prior.expected_sign.value}, got {'positive' if actual_positive else 'negative'}"
        
        # Case 5: Check IC magnitude
        if abs(empirical_ic) < prior.min_abs_ic:
            self.stats['rejected_low_ic'] += 1
            return False, f"IC magnitude {abs(empirical_ic):.4f} < minimum {prior.min_abs_ic}"
        
        # Passed all checks
        self.stats['passed'] += 1
        return True, f"Passed: {prior.rationale[:50]}..."
    
    def filter_features(
        self,
        feature_ic_dict: Dict[str, float],
        verbose: bool = False,
    ) -> Dict[str, float]:
        """
        Filter a dictionary of features to only economically justified ones.
        
        Args:
            feature_ic_dict: {feature_name: IC_value}
            verbose: If True, print filtering details
            
        Returns:
            Filtered dict with only features that pass economic prior
        """
        self.reset_stats()
        passed_features = {}
        rejected_details = []
        
        for feature, ic in feature_ic_dict.items():
            passed, reason = self.evaluate_feature(feature, ic)
            if passed:
                passed_features[feature] = ic
            elif verbose:
                rejected_details.append((feature, ic, reason))
        
        if verbose:
            print("\n" + "="*60)
            print("ECONOMIC PRIOR FILTER RESULTS")
            print("="*60)
            print(f"Total evaluated: {self.stats['total_evaluated']}")
            print(f"Passed: {self.stats['passed']} ({100*self.stats['passed']/max(1,self.stats['total_evaluated']):.1f}%)")
            print(f"Rejected - Forbidden: {self.stats['rejected_forbidden']}")
            print(f"Rejected - No prior: {self.stats['rejected_no_prior']}")
            print(f"Rejected - Wrong sign: {self.stats['rejected_wrong_sign']}")
            print(f"Rejected - Low IC: {self.stats['rejected_low_ic']}")
            
            if rejected_details:
                print(f"\nSample rejections (first 10):")
                for feat, ic, reason in rejected_details[:10]:
                    print(f"  {feat}: IC={ic:.4f} - {reason}")
        
        return passed_features
    
    def get_approved_feature_list(self) -> Set[str]:
        """
        Get list of all feature patterns that have economic priors.
        
        Useful for understanding what features are theoretically approved.
        """
        approved = set()
        for prior in self.priors:
            if prior.expected_sign != ExpectedSign.FORBIDDEN:
                approved.add(prior.pattern)
        return approved
    
    def explain_feature(self, feature_name: str) -> str:
        """Get detailed explanation of a feature's economic prior."""
        prior = get_prior_for_feature(feature_name)
        if prior is None:
            return f"No economic prior defined for '{feature_name}'"
        
        return (
            f"Feature: {feature_name}\n"
            f"Pattern: {prior.pattern}\n"
            f"Expected Sign: {prior.expected_sign.value}\n"
            f"Min |IC|: {prior.min_abs_ic}\n"
            f"Confidence: {prior.confidence}\n"
            f"Rationale: {prior.rationale}"
        )


# =============================================================================
# INTEGRATION WITH FEATURE SELECTION
# =============================================================================

def apply_economic_prior_to_formation(
    feature_ics: Dict[str, float],
    require_sign_match: bool = True,
    verbose: bool = True,
) -> Dict[str, float]:
    """
    Apply economic prior filter during formation phase.
    
    This should be called BEFORE FDR control to reduce the feature space
    to economically justified candidates.
    
    Args:
        feature_ics: {feature_name: mean_ic} from formation window
        require_sign_match: Enforce sign consistency with theory
        verbose: Print filter statistics
        
    Returns:
        Filtered feature ICs
    """
    filter = EconomicPriorFilter(
        require_sign_match=require_sign_match,
        allow_unprioried_features=False,  # Strict mode
        min_prior_confidence=0.3,
    )
    
    return filter.filter_features(feature_ics, verbose=verbose)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Example: Test the filter with mock feature ICs
    mock_feature_ics = {
        # Should pass (momentum with positive IC)
        "mom_21d": 0.035,
        "mom_63d": 0.028,
        
        # Should pass (volatility with negative IC)
        "vol_21d": -0.025,
        "realized_vol_63d": -0.022,
        
        # Should FAIL (wrong sign - vol should be negative)
        "vol_10d": 0.018,  # Wrong sign!
        
        # Should FAIL (forbidden)
        "day_of_week_monday": 0.012,
        
        # Should FAIL (no prior)
        "random_feature_xyz": 0.025,
        
        # Should pass (interaction with theory)
        "mom_21d_x_vol_21d": 0.032,
        
        # Borderline (low IC)
        "sharpe_21d": 0.008,  # Below threshold
        
        # Should pass (trend)
        "trend_strength_21d": 0.024,
    }
    
    filter = EconomicPriorFilter(
        require_sign_match=True,
        allow_unprioried_features=False,
    )
    
    passed = filter.filter_features(mock_feature_ics, verbose=True)
    
    print("\n" + "="*60)
    print("FEATURES THAT PASSED:")
    print("="*60)
    for feat, ic in sorted(passed.items(), key=lambda x: -abs(x[1])):
        print(f"  {feat}: IC={ic:.4f}")
