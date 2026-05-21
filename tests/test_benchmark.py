"""Benchmark: CPU Numba vs GPU Warp for PteraSoftware Biot-Savart at various sizes."""
import sys, os, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pterasoftware._aerodynamics_functions import (
    collapsed_velocities_from_ring_vortices as cpu_fn,
)
from fluxvortex.warp_kernels import (
    collapsed_velocities_from_ring_vortices as gpu_fn,
)

rng = np.random.default_rng(42)
runs = 3

print("=" * 60)
print("FLUXVortex GPU Benchmark: Warp (GPU) vs Numba (CPU)")
print("=" * 60)

for N, M in [(500, 2000), (1000, 5000), (500, 10000)]:
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

    print(f"\n--- N={N}, M={M} ({4*N*M/1e6:.1f}M thread-launches) ---")

    # CPU warmup + bench
    _ = cpu_fn(points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu)
    cpu_times = []
    for _ in range(runs):
        sing[:] = 0
        t0 = time.perf_counter()
        cpu_result = cpu_fn(points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu)
        cpu_times.append(time.perf_counter() - t0)

    # GPU warmup + bench
    _ = gpu_fn(points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu)
    gpu_times = []
    for _ in range(runs):
        sing[:] = 0
        t0 = time.perf_counter()
        gpu_result = gpu_fn(points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu)
        gpu_times.append(time.perf_counter() - t0)

    cpu_ms = np.mean(cpu_times) * 1000
    gpu_ms = np.mean(gpu_times) * 1000
    speedup = cpu_ms / gpu_ms
    max_err = np.max(np.abs(cpu_result - gpu_result))

    print(f"  CPU (Numba): {cpu_ms:.1f} ms")
    print(f"  GPU (Warp):  {gpu_ms:.1f} ms")
    print(f"  Speedup:     {speedup:.1f}x  (err={max_err:.1e})")
