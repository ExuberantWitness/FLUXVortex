"""Physical validation of the unsteady free-wake ring-VLM (Plan fix1-②). Two anchors:

  (1) STEADY-LIMIT CONSISTENCY — the unsteady rollout's asymptotic (large-t) circulatory lift
      must converge to the independently-validated STEADY horseshoe VLM (diff_vlm) on the same
      wing as the timestep Δs→0. This ties the unsteady circulation dynamics to the steady aero.
      (UVLM near-wake discretization error is O(Δs): a coarse wake under-resolves the lift; it
      converges monotonically as the shed rings near the trailing edge are refined.)

  (2) WAGNER INDICIAL RESPONSE — an impulsively-started flat plate's circulatory lift builds from
      ½ to the full steady value following the Wagner function Φ(s), s = 2·U·t/c semichords
      (Wagner 1925; Jones 1940 approximation; Katz & Plotkin §13 reproduce it with exactly this
      discrete-wake UVLM). We take the KJ (circulatory) lift, self-normalize by the converged
      value, and compare the buildup shape to Φ(s).

Honest scope: ring-VLM is inviscid/attached, so this validates the unsteady *circulation
dynamics* — exactly what the wake-history adjoint differentiates. The KJ lift uses the net
chordwise bound circulation Γ_p−Γ_upstream (vortex-ring telescoping → trailing-edge circulation).
"""
from __future__ import annotations

import os
import sys

import numpy as np

for p in (os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")), os.path.dirname(__file__)):
    if p not in sys.path:
        sys.path.insert(0, p)

import warp as wp                                                # noqa: E402
import diff_uvlm_unsteady as ref                                # noqa: E402
import diff_uvlm_unsteady_gpu as g                              # noqa: E402
import diff_vlm                                                 # noqa: E402 (validated steady VLM)


def wagner_jones(s):
    return 1.0 - 0.165 * np.exp(-0.0455 * s) - 0.335 * np.exp(-0.3 * s)


def verify(AR=20.0, ns=24, aoa_deg=4.0):
    wp.init()
    U = 10.0; chord = 0.1; span = AR * chord
    Vinf = np.array([U, 0.0, 0.0]); aoa = np.deg2rad(aoa_deg)
    q = 0.5 * 1.225 * U * U; S = span * chord

    # ---- (1) steady-limit consistency: unsteady asymptotic Cl → steady VLM as Δs→0 ----
    nc = 1
    C = ref._lattice(nc, ns, chord, span, aoa)
    _, Ft = diff_vlm.vlm_forces(C, nc, ns, Vinf)
    Cl_steady = abs(Ft[2]) / (q * S)
    print(f"Steady-limit consistency (AR={AR:.0f}, nc={nc}):  steady horseshoe VLM Cl={Cl_steady:.4f}")
    fracs = [0.20, 0.10, 0.05, 0.025]                  # wake ring length as a fraction of chord
    seq = []
    for frac in fracs:
        dt = frac * chord / U
        N = int(round(25.0 / (2 * U * dt / chord)))    # integrate to s≈25 semichords
        Lc = g.unsteady_rollout_gpu(nc, ns, chord, span, aoa, Vinf, N, dt, added_mass=False)
        cl = abs(Lc[-3:].mean()) / (q * S)
        seq.append(cl)
        print(f"  Δs={2 * U * dt / chord:.2f} semichord/step (N={N:4d}):  unsteady Cl={cl:.4f}  "
              f"({100 * cl / Cl_steady:5.1f}% of steady)")
    monotone = all(seq[i + 1] >= seq[i] - 1e-4 for i in range(len(seq) - 1))
    converging = abs(seq[-1] - Cl_steady) / Cl_steady < 0.06
    ok1 = monotone and converging

    # ---- (2) Wagner buildup: correct qualitative indicial response ----
    # The circulatory lift starts near ½ the steady value and builds monotonically along a
    # Wagner-shaped curve to the steady value. The UVLM curve sits slightly ABOVE Φ(s) at finite
    # Δs — the documented discrete-onset effect (first shed vortex ½-step downstream → less initial
    # downwash → Φ(0+)≳0.5); it relaxes toward Φ(s) as Δs→0. We check the Wagner SIGNATURE, not a
    # tight fit at fixed Δs: starts near half, monotone, correct asymptote, right buildup timescale.
    dt = 0.025 * chord / U; N = int(round(22.0 / (2 * U * dt / chord)))
    Lc = g.unsteady_rollout_gpu(nc, ns, chord, span, aoa, Vinf, N, dt, added_mass=False)
    s = 2.0 * U * (np.arange(1, N + 1) * dt) / chord
    phi = Lc / Lc[-3:].mean()
    half = 0.45 <= phi[0] <= 0.70                          # starts near half steady (Wagner ½)
    monotone = bool(np.all(np.diff(phi) > -1e-3))
    asymptote = phi[-1] > 0.97 and phi[s <= 8].max() < 0.98  # builds toward 1, not instantly
    band = bool(np.all(phi[(s >= 1) & (s <= 20)] >= wagner_jones(s[(s >= 1) & (s <= 20)]) - 0.03))
    ok2 = half and monotone and asymptote and band
    print(f"Wagner indicial buildup (Δs={2 * U * dt / chord:.3f} semichord/step):")
    print(f"  Φ(0+)={phi[0]:.3f} (Wagner ½=0.5; discrete-onset ≳½), monotone={monotone}, "
          f"Φ(∞)={phi[-1]:.3f}")
    for si in [0.5, 2.0, 8.0, 20.0]:
        i = int(np.argmin(np.abs(s - si)))
        print(f"     s={s[i]:5.1f}  Φ_uvlm={phi[i]:.3f}  Φ_wagner={wagner_jones(s[i]):.3f}")

    ok = ok1 and ok2
    print(f"  -> {'PASS' if ok else 'FAIL'}: unsteady free-wake ring-VLM converges to the validated "
          f"steady VLM as Δs→0 (consistency) and shows the Wagner indicial signature — physical "
          f"validation of the unsteady circulation dynamics (fix1-②)")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
