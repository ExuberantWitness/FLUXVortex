"""newton_pc — problem-agnostic window predictor-corrector coupling.

Validated against a MATLAB FSI reference to 1e-6 (see docs/newton_comparison.md).
"""
from .coupler import WindowPredictorCorrector, WindowStats
from .protocols import ForceProvider, ForceSet, StructuralEntry

__all__ = ["WindowPredictorCorrector", "WindowStats", "ForceProvider",
           "ForceSet", "StructuralEntry"]
