"""Validation: Warp blob kernel vs frozen rvpm_reference kernel."""
import sys, time
from pathlib import Path
import numpy as np

_repo = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_repo / "src"))
sys.path.insert(0, str(_repo / "platform"))

from fluxvortex.rvpm_reference import direct_gaussian_erf_velocity_jacobian as frozen
from warp_vpm.pfield import ParticleField


def validate(n_src=500, n_tgt=300, seed=42):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n_src, 3)) * 0.1
    gam = rng.normal(size=(n_src, 3)) * 1e-4
    sig = rng.uniform(0.005, 0.05, n_src)
    tgt = rng.normal(size=(n_tgt, 3)) * 0.1

    v_ref = np.asarray(frozen(pos, gam, sig, target_positions=tgt).velocity)
    pf = ParticleField(capacity=n_src)
    pf.add_particles(pos, gam, sig, birth_step=0)
    v_warp = pf.velocity_at(tgt)

    rel = np.abs(v_warp - v_ref).max() / np.abs(v_ref).max()
    print(f"  velocity: rel_diff = {rel:.3e} ({'PASS' if rel < 1e-10 else 'FAIL'})")
    return rel < 1e-10


def validate_self(n=200, seed=7):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3)) * 0.1
    gam = rng.normal(size=(n, 3)) * 1e-4
    sig = rng.uniform(0.005, 0.05, n)

    v_ref = np.asarray(frozen(pos, gam, sig).velocity)
    pf = ParticleField(capacity=n)
    pf.add_particles(pos, gam, sig, birth_step=0)
    v_warp = pf.velocity_self()

    rel = np.abs(v_warp - v_ref).max() / np.abs(v_ref).max()
    print(f"  self-velocity: rel_diff = {rel:.3e} ({'PASS' if rel < 1e-10 else 'FAIL'})")
    return rel < 1e-10


def benchmark(n=40000, seed=99):
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 3))
    gam = rng.normal(size=(n, 3)) * 1e-4
    sig = np.full(n, 0.03)

    pf = ParticleField(capacity=n)
    pf.add_particles(pos, gam, sig, birth_step=0)

    _ = pf.velocity_self()  # warmup

    t0 = time.perf_counter()
    v = pf.velocity_self()
    t1 = time.perf_counter()
    print(f"  Warp GPU: {n} particles self-eval = {t1-t0:.2f}s")

    if n <= 5000:
        t0 = time.perf_counter()
        v_cpu = np.asarray(frozen(pos, gam, sig).velocity)
        t1 = time.perf_counter()
        print(f"  CPU frozen: {t1-t0:.2f}s, speedup = {(t1-t0)/(t1-t0):.0f}x baseline")


if __name__ == "__main__":
    print("=== Warp VPM Kernel Validation ===\n")
    print("1. Velocity at external targets:")
    ok1 = validate()
    print("\n2. Self-induced velocity:")
    ok2 = validate_self()
    print("\n3. Benchmark (40k):")
    benchmark(40000)
    print(f"\n{'ALL PASS' if ok1 and ok2 else 'SOME FAILED'}")
    sys.exit(0 if ok1 and ok2 else 1)
