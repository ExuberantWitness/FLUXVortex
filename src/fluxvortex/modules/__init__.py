"""FluxVortex modular FSI framework.

Four clean modules:
  - numerical_solver: Newmark-β implicit time integrator (no domain knowledge)
  - structural_module: ANCF shell mechanics (M, Kt, Qe)
  - aerodynamic_module: UVLM unsteady vortex lattice + wake
  - coupling_solver: FSI orchestration (predictor-corrector, force averaging)
"""
from .numerical_solver import NewmarkSolver
from .structural_module import ANCFStructure
from .aerodynamic_module import UVLMAerodynamics
from .coupling_solver import FSICouplingSolver

__all__ = ['NewmarkSolver', 'ANCFStructure', 'UVLMAerodynamics', 'FSICouplingSolver']
