"""Test GPU vs CPU correctness for all PteraSoftware Biot-Savart functions."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import numpy as np
from pterasoftware._aerodynamics_functions import (
    collapsed_velocities_from_ring_vortices as cpu_collapsed_ring,
    expanded_velocities_from_ring_vortices as cpu_expanded_ring,
    collapsed_velocities_from_horseshoe_vortices as cpu_collapsed_hs,
    expanded_velocities_from_horseshoe_vortices as cpu_expanded_hs,
)
from fluxvortex.warp_kernels import (
    collapsed_velocities_from_ring_vortices as gpu_collapsed_ring,
    expanded_velocities_from_ring_vortices as gpu_expanded_ring,
    collapsed_velocities_from_horseshoe_vortices as gpu_collapsed_hs,
    expanded_velocities_from_horseshoe_vortices as gpu_expanded_hs,
)

rng = np.random.default_rng(42)
N, M = 200, 100

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

all_pass = True

tests = [
    ("collapsed ring", cpu_collapsed_ring, gpu_collapsed_ring,
     (points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu)),
    ("expanded ring", cpu_expanded_ring, gpu_expanded_ring,
     (points, br, fr, fl, bl, strengths, rc0s, sing, ages, nu)),
    ("collapsed horseshoe", cpu_collapsed_hs, gpu_collapsed_hs,
     (points, br, fr, fl, bl, strengths, rc0s, sing, nu)),
    ("expanded horseshoe", cpu_expanded_hs, gpu_expanded_hs,
     (points, br, fr, fl, bl, strengths, rc0s, sing, nu)),
]

print("=" * 60)
print("FLUXVortex Precision Validation: GPU vs CPU (Numba)")
print("=" * 60)

for name, cpu_fn, gpu_fn, args in tests:
    cpu = cpu_fn(*args)
    gpu = gpu_fn(*args)
    err = np.max(np.abs(cpu - gpu))
    rel = err / (np.max(np.abs(cpu)) + 1e-16)
    ok = err < 1e-8
    if not ok:
        all_pass = False
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:30s} abs={err:.2e} rel={rel:.2e}")

print()
if all_pass:
    print("ALL TESTS PASSED")
else:
    print("SOME TESTS FAILED")
    sys.exit(1)
