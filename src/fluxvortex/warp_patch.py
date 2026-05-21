"""
Monkey-patch PteraSoftware's _aerodynamics_functions to use Warp GPU kernels.

Usage:
    from fluxvortex.warp_patch import patch, unpatch, benchmark

    patch()    # Replace Numba CPU functions with Warp GPU
    # ... run PteraSoftware simulations ...
    unpatch()  # Restore original Numba CPU functions

Also provides benchmark() for CPU vs GPU timing comparison.
"""
import time
import numpy as np

from pterasoftware import _aerodynamics_functions as _af
from .warp_kernels import (
    collapsed_velocities_from_ring_vortices as _gpu_collapsed_ring,
    collapsed_velocities_from_ring_vortices_chordwise_segments as _gpu_collapsed_ring_chord,
    expanded_velocities_from_ring_vortices as _gpu_expanded_ring,
    collapsed_velocities_from_horseshoe_vortices as _gpu_collapsed_horseshoe,
    expanded_velocities_from_horseshoe_vortices as _gpu_expanded_horseshoe,
)

# ── Save originals ────────────────────────────────────────────────────
_originals = {}
_patch_active = False


def _save_originals():
    names = [
        'collapsed_velocities_from_ring_vortices',
        'collapsed_velocities_from_ring_vortices_chordwise_segments',
        'expanded_velocities_from_ring_vortices',
        'collapsed_velocities_from_horseshoe_vortices',
        'expanded_velocities_from_horseshoe_vortices',
    ]
    for name in names:
        if hasattr(_af, name) and name not in _originals:
            _originals[name] = getattr(_af, name)


def patch():
    """Replace PteraSoftware Numba functions with Warp GPU kernels."""
    global _patch_active
    if _patch_active:
        return

    _save_originals()

    _af.collapsed_velocities_from_ring_vortices = _gpu_collapsed_ring
    _af.collapsed_velocities_from_ring_vortices_chordwise_segments = _gpu_collapsed_ring_chord
    _af.expanded_velocities_from_ring_vortices = _gpu_expanded_ring
    _af.collapsed_velocities_from_horseshoe_vortices = _gpu_collapsed_horseshoe
    _af.expanded_velocities_from_horseshoe_vortices = _gpu_expanded_horseshoe

    _patch_active = True
    print("[warp_patch] GPU acceleration active (NVIDIA Warp)")


def unpatch():
    """Restore original PteraSoftware Numba CPU functions."""
    global _patch_active
    if not _patch_active:
        return

    for name, fn in _originals.items():
        setattr(_af, name, fn)

    _patch_active = False
    print("[warp_patch] Restored original Numba CPU functions")


# ── Benchmark ─────────────────────────────────────────────────────────
def benchmark(N=500, M=2000, num_runs=5):
    """Compare CPU (Numba) vs GPU (Warp) performance on random Biot-Savart data."""
    rng = np.random.default_rng(42)

    points = rng.standard_normal((N, 3))
    br = rng.standard_normal((M, 3))
    fr = rng.standard_normal((M, 3))
    fl = rng.standard_normal((M, 3))
    bl = rng.standard_normal((M, 3))
    strengths = rng.standard_normal(M) * 0.5
    rc0s = np.full(M, 0.03)
    sing = np.zeros(4, dtype=np.int64)
    ages = rng.uniform(0, 1, M)
    nu = 1.5e-5

    # ── CPU warmup + benchmark ──
    print(f"Benchmark: N={N} points, M={M} ring vortices, {num_runs} runs")
    print()

    # Warmup (Numba JIT compile)
    _ = _originals['collapsed_velocities_from_ring_vortices'](
        points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu,
    )

    cpu_times = []
    for _ in range(num_runs):
        sing[:] = 0
        t0 = time.perf_counter()
        cpu_result = _originals['collapsed_velocities_from_ring_vortices'](
            points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu,
        )
        cpu_times.append(time.perf_counter() - t0)

    # ── GPU warmup + benchmark ──
    # Warmup (Warp kernel compile)
    _ = _gpu_collapsed_ring(
        points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu,
    )

    gpu_times = []
    for _ in range(num_runs):
        sing[:] = 0
        t0 = time.perf_counter()
        gpu_result = _gpu_collapsed_ring(
            points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu,
        )
        gpu_times.append(time.perf_counter() - t0)

    # ── Results ──
    cpu_mean = np.mean(cpu_times)
    gpu_mean = np.mean(gpu_times)
    speedup = cpu_mean / gpu_mean

    max_err = np.max(np.abs(cpu_result - gpu_result))
    rel_err = max_err / (np.max(np.abs(cpu_result)) + 1e-16)

    print(f"CPU (Numba): {cpu_mean*1000:.2f} ms  (avg of {num_runs})")
    print(f"GPU (Warp):  {gpu_mean*1000:.2f} ms  (avg of {num_runs})")
    print(f"Speedup:     {speedup:.1f}x")
    print(f"Max abs error: {max_err:.2e}")
    print(f"Max rel error: {rel_err:.2e}")

    return {
        'cpu_ms': cpu_mean * 1000,
        'gpu_ms': gpu_mean * 1000,
        'speedup': speedup,
        'max_abs_error': max_err,
        'max_rel_error': rel_err,
    }


if __name__ == '__main__':
    # Save originals and run benchmark
    _save_originals()
    results = benchmark()
