"""
CrossSecMom2 Scripts Package
============================

Evaluation and diagnostic utilities for the crosssecmom2 strategy.

Scripts:
--------
- run_wf_smoke_test: Quick validation of target labels via short WF run
- compare_targets: Side-by-side comparison of different target labels

Usage:
------
    python -m scripts.run_wf_smoke_test --n-windows 5
    python -m scripts.compare_targets --max-windows 12
"""

from .run_wf_smoke_test import run_smoke_test
from .compare_targets import compare_targets

__all__ = ['run_smoke_test', 'compare_targets']
