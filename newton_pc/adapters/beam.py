"""Bending-torsion beam adapter (Goland-wing flutter) for newton_pc.

Couples an Euler-Bernoulli bending + torsion beam (fluxvortex.beam_fe.BeamFE,
3 DOF/node: w, dw/dy, theta) to the validated ring-UVLM provider through the
two-pass window predictor-corrector. This is the aeroelastic FLUTTER
validation of the newton_pc coupler against a classic benchmark.

  - GolandBeamEntry: StructuralEntry. Holds the beam; state() deforms a flat
    chord x span vertex ribbon by heave w(y) + twist theta(y) about the
    elastic axis (x_ea); velocities from the beam rate state.
  - BeamUVLMProvider: ForceProvider. Reuses FlapUVLMProvider's UVLM + free
    wake + particle machinery; converts panel forces to beam generalized
    forces (spanwise lift on the w-DOF, moment about EA on the theta-DOF).

Flutter is detected by the envelope growth rate of the tip heave/twist after
an initial perturbation, swept over freestream speed (same protocol as
tests/benchmark_goland.py, but driven by the newton_pc two-pass coupler
instead of the legacy lagged staggered scheme).
"""
from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if os.path.join(_ROOT, "src") not in sys.path:
    sys.path.insert(0, os.path.join(_ROOT, "src"))

from fluxvortex.beam_fe import BeamFE  # noqa: E402

from .flap import FlapUVLMProvider  # noqa: E402


# ── force container (beam generalized forces) ─────────────────────────────
class BeamForceSet:
    """Interpolable beam generalized-force vector (ndof,) + solve payload."""

    def __init__(self, gen: np.ndarray, payload: dict | None = None):
        self.gen = gen
        self.payload = payload

    def affine(self, other: "BeamForceSet", beta: float) -> "BeamForceSet":
        return BeamForceSet(self.gen + (other.gen - self.gen) * beta)

    def lincomb(self, pairs) -> "BeamForceSet":
        acc = None
        for fs, w in pairs:
            term = fs.gen * w
            acc = term if acc is None else acc + term
        return BeamForceSet(acc)


# ── structural entry ──────────────────────────────────────────────────────
class GolandBeamEntry:
    """StructuralEntry: bending-torsion beam driving a deformable ribbon."""

    def __init__(self, chord, span, nc, ns, beam_params, alpha_deg=2.0,
                 x_ea_chord=0.33):
        self.chord, self.span = chord, span
        self.nc, self.ns = nc, ns
        self.x_ea = x_ea_chord * chord
        self.alpha = np.deg2rad(alpha_deg)
        # beam with one element per spanwise panel -> nodes align with grid
        bp = dict(beam_params)
        bp["n_elements"] = ns
        self.beam = BeamFE(**bp)
        self.y_nodes = self.beam.y_nodes
        # flat reference ribbon (at angle of attack): (nc+1, ns+1, 3)
        xc = np.linspace(0.0, chord, nc + 1)
        ref = np.zeros((nc + 1, ns + 1, 3))
        ref[:, :, 0] = xc[:, None]
        ref[:, :, 1] = self.y_nodes[None, :]
        # rigid AoA pitch about the elastic axis (steady incidence)
        x_rel = xc - self.x_ea
        ca, sa = np.cos(self.alpha), np.sin(self.alpha)
        ref[:, :, 0] = (self.x_ea + x_rel * ca)[:, None]
        ref[:, :, 2] = (-x_rel * sa)[:, None]
        self.ref = ref
        self.t = 0.0

    def perturb(self, w_tip=0.05, theta_tip_deg=2.0):
        """Initial tip perturbation to seed flutter (heave + twist)."""
        n = self.beam.nnodes - 1
        self.beam.d[3 * n] = w_tip
        self.beam.d[3 * n + 2] = np.radians(theta_tip_deg)
        K_r, M_r, _, free = self.beam.apply_bc(self.beam.K, self.beam.M)
        self.beam.a[free] = np.linalg.solve(M_r, -K_r @ self.beam.d[free])

    # protocol -----------------------------------------------------------
    def snapshot(self) -> Any:
        return (self.t, self.beam.d.copy(), self.beam.v.copy(),
                self.beam.a.copy())

    def restore(self, snap: Any) -> None:
        self.t = snap[0]
        self.beam.d[:] = snap[1]
        self.beam.v[:] = snap[2]
        self.beam.a[:] = snap[3]

    def substep(self, t: float, dt: float, forces: BeamForceSet) -> None:
        self.beam.step(forces.gen, dt)
        self.t = t

    def _deform(self, d):
        """Deform the reference ribbon by (w, theta) from a beam DOF vector."""
        w = d[0::3]
        th = d[2::3]
        ref = self.ref
        verts = ref.copy()
        x_rel = ref[:, :, 0] - self.x_ea
        z_rel = ref[:, :, 2]
        ct, st = np.cos(th)[None, :], np.sin(th)[None, :]
        # incremental twist about EA applied to the (already pitched) ribbon
        verts[:, :, 0] = self.x_ea + x_rel * ct + z_rel * st
        verts[:, :, 2] = -x_rel * st + z_rel * ct + w[None, :]
        return verts

    def state(self) -> dict:
        verts = self._deform(self.beam.d)
        # velocity field: d(verts)/dt from beam rate (finite via analytic w,th)
        w, th = self.beam.d[0::3], self.beam.d[2::3]
        wd, thd = self.beam.v[0::3], self.beam.v[2::3]
        x_rel = self.ref[:, :, 0] - self.x_ea
        z_rel = self.ref[:, :, 2]
        ct, st = np.cos(th)[None, :], np.sin(th)[None, :]
        vels = np.zeros_like(verts)
        # d/dt of twist-rotated position + heave
        vx = (-x_rel * st + z_rel * ct) * thd[None, :]
        vz = (-x_rel * ct - z_rel * st) * thd[None, :] + wd[None, :]
        vels[:, :, 0] = vx
        vels[:, :, 2] = vz
        return dict(verts=verts.copy(), vels=vels)

    # force transfer: panel forces -> beam generalized forces -----------
    def panel_to_beam(self, f_panel: np.ndarray) -> np.ndarray:
        """Lump panel z-forces to beam stations: lift on w-DOF, moment about
        EA on theta-DOF. f_panel: (nc, ns, 3)."""
        verts = self._deform(self.beam.d)
        # panel centroid x (for moment arm about EA) and z-force
        cx = 0.25 * (verts[:-1, :-1, 0] + verts[1:, :-1, 0]
                     + verts[:-1, 1:, 0] + verts[1:, 1:, 0])   # (nc, ns)
        fz = f_panel[:, :, 2]                                   # (nc, ns)
        arm = cx - self.x_ea
        gen = np.zeros(self.beam.ndof)
        # each panel j-strip lies between beam nodes j and j+1: split 50/50
        lift_strip = fz.sum(axis=0)              # (ns,) total lift per strip
        mom_strip = (fz * arm).sum(axis=0)       # (ns,) moment about EA
        for j in range(self.ns):
            for nd, wgt in ((j, 0.5), (j + 1, 0.5)):
                gen[3 * nd] += wgt * lift_strip[j]       # w-DOF
                gen[3 * nd + 2] += wgt * mom_strip[j]    # theta-DOF
        return gen


# ── UVLM provider returning beam generalized forces ───────────────────────
class BeamUVLMProvider(FlapUVLMProvider):
    """FlapUVLMProvider whose solve() converts panel forces to beam forces."""

    def bind(self, entry: GolandBeamEntry):
        self.entry = entry
        return self

    def solve(self, state: dict) -> BeamForceSet:
        out = self._trial(state)
        self.n_solves += 1
        gen = self.entry.panel_to_beam(out["f_panel"])
        return BeamForceSet(gen, payload=out)

    def commit(self, forces: BeamForceSet) -> None:
        # reuse the parent wake/particle commit via a shaped stand-in
        from .flap import NodalForceSet
        FlapUVLMProvider.commit(self, NodalForceSet(np.zeros(1),
                                                    payload=forces.payload))
