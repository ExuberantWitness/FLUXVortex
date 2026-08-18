"""Warp-based vortex particle method core.

Design inherits the architectural philosophy of the user's hand-written
small-particle-computing codebase (FLOWVPM.jl Python translation):
- ParticleField as a flat "super-dictionary" (SoA layout, typed lifecycle)
- Fused GPU kernels via NVIDIA Warp (replacing per-target CuPy launches)
- RK3 low-storage time stepping with reformulated VPM stretching

This module is DIAGNOSTIC-tier (DiGT-1); frozen audit files untouched.
"""
