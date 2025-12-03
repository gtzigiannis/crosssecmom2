#!/usr/bin/env python
"""Quick script to check leverage settings."""
import sys
sys.path.insert(0, '..')

from config import get_default_config

config = get_default_config()
port = config.portfolio

print("=== Current Leverage Configuration ===")
print()
print(f"use_leverage: {port.use_leverage}")
print(f"margin_regime: {port.margin_regime}")
ml, ms = port.get_active_margins()
print(f"margin_long (active): {ml}")
print(f"margin_short (active): {ms}")
print(f"max_margin_utilization: {port.max_margin_utilization}")
print(f"enforce_dollar_neutral: {port.enforce_dollar_neutral}")
print(f"long_only: {port.long_only}")
print()

exp = port.compute_max_exposure()
print("=== Computed Exposures ===")
print(f"long_exposure: {exp['long_exposure']:.4f}")
print(f"short_exposure: {exp['short_exposure']:.4f}")
print(f"gross_leverage: {exp['gross_leverage']:.4f}")
print(f"net_exposure: {exp['net_exposure']:.4f}")

print()
print("=== Comparison with Different Settings ===")

# Test no leverage vs leverage with different regimes
for use_lev in [False, True]:
    for regime in ['reg_t_initial', 'reg_t_maintenance', 'portfolio']:
        port.use_leverage = use_lev
        port.margin_regime = regime
        port.max_margin_utilization = 1.0  # Full utilization for comparison
        ml, ms = port.get_active_margins()
        exp = port.compute_max_exposure()
        print(f"  use_leverage={str(use_lev):5} {regime:20} => "
              f"margin=({ml:.2f}/{ms:.2f}) gross={exp['gross_leverage']:.2f}x "
              f"(L={exp['long_exposure']:.2f}, S={exp['short_exposure']:.2f})")

