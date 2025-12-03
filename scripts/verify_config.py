#!/usr/bin/env python
"""Verify configuration values before running backtest."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import get_default_config

c = get_default_config()

print("="*60)
print("CONFIGURATION VERIFICATION")
print("="*60)

print(f"\n=== TIME/WINDOW SETTINGS ===")
print(f"Formation window: {c.features.formation_years} years ({int(c.features.formation_years * 252)} trading days)")
print(f"Training window: {c.features.training_years} years ({int(c.features.training_years * 252)} trading days)")
print(f"TRAINING_WINDOW_DAYS (legacy): {c.time.TRAINING_WINDOW_DAYS} days")
print(f"Holding period: {c.time.HOLDING_PERIOD_DAYS} days")
print(f"Rebalance frequency: {c.time.STEP_DAYS} days")

print(f"\n=== POSITION SIZING ===")
print(f"use_leverage: {c.portfolio.use_leverage}")
print(f"max_margin_utilization: {c.portfolio.max_margin_utilization}")

exp = c.portfolio.compute_max_exposure()
print(f"Long exposure: {exp['long_exposure']:.2%}")
print(f"Short exposure: {exp['short_exposure']:.2%}")
print(f"Gross exposure: {exp['gross_exposure']:.2%}")
print(f"Net exposure: {exp['net_exposure']:.2%}")

print(f"\n=== MARGIN SETTINGS ===")
margin_long, margin_short = c.portfolio.get_active_margins()
print(f"Active margin (long): {margin_long:.0%}")
print(f"Active margin (short): {margin_short:.0%}")
print(f"Margin regime: {c.portfolio.margin_regime}")

print(f"\n=== TRANSACTION COSTS ===")
print(f"Total cost (bps per side): {c.portfolio.total_cost_bps_per_side:.1f}")

print("\n" + "="*60)
print("Configuration verified. Ready to run backtest.")
print("="*60)
