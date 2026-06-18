"""Quasi-steady strip-theory aerodynamics for the free-flying flapping aircraft.

Turns the multibody aircraft (aircraft_assembly) into an aerodynamically flying
6-DOF vehicle: for each lifting body (2 main wings + 2 V-tail panels) it discretizes
the surface into spanwise strips, computes the local relative wind (body translation +
rotation + flapping + gust), the strip lift/drag from a finite-AR thin-airfoil model
with control-surface camber, and accumulates the net wrench fed back to the
Featherstone bodies as `state.body_f` (spatial [angular; linear], world frame).

Flapping the wings (the flap-hinge rotation) sweeps the strips through the air and
generates thrust — the aircraft is propelled by the flap, steered by the 14 control
surfaces (12 wing camber + 2 V-tail), and disturbed by the gust. This is the standard
quasi-steady flapping-flight model used for control; the validated UVLM coupled FSI is
the high-fidelity path for per-design efficiency (plan's two-fidelity structure).

PPO-first control is model-free, so numpy here is fine (no tape through the aero); the
Featherstone dynamics stay Newton/Warp. A Warp port enables SHAC later (plan §6).
"""
from __future__ import annotations

import numpy as np

RHO = 1.225
CD0 = 0.02
OSWALD = 0.85
STALL = np.deg2rad(16.0)


def _quat_to_R(q):
    """xyzw quaternion -> 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


class StripAero:
    """Strip-theory aero wrench provider for the assembled aircraft."""

    def __init__(self, idx, n_strips=5, rho=RHO):
        self.idx = idx
        self.ns = n_strips
        self.rho = rho
        # precompute, per lifting body: chord, half-span, is_wing, side-sign, AR, CLa
        self.bodies = {}
        for b, (chord, span, is_wing, sgn) in idx.lift_geom.items():
            AR = (2 * span) ** 2 / ((2 * span) * chord) if is_wing else span / chord
            AR = max(AR, 1.0)
            CLa = 2 * np.pi * AR / (AR + 2.0)            # finite-AR lift slope (1/rad)
            # strip spanwise stations (fraction of the half-span, outboard) + width
            fr = (np.arange(self.ns) + 0.5) / self.ns
            self.bodies[b] = dict(chord=chord, span=span, is_wing=is_wing, sgn=sgn,
                                  AR=AR, CLa=CLa, frac=fr, ds=span / self.ns)
        # map each wing body -> list of (surf metadata) for camber
        self.wing_surfs = {}
        for k, m in idx.surf_meta.items():
            self.wing_surfs.setdefault(m["wing_body"], []).append((k, m))

    def _camber_at(self, body, frac, defl):
        """Effective camber (rad) at a strip station from the wing's control surfaces."""
        cam = 0.0
        for k, m in self.wing_surfs.get(body, []):
            if m["edge"] == "VT":
                continue                                 # V-tail deflects as a whole body
            # surface influence ~ Gaussian in spanwise station; TE-down raises lift
            w = np.exp(-((frac - m["frac"]) / 0.22) ** 2)
            sign = +1.0 if m["edge"] == "TE" else -1.0
            cam += sign * m["eff"] * w * defl[k]
        return cam

    def wrenches(self, body_q, body_qd, defl, wind=None):
        """Return body_f array (nbody, 6) = world-frame spatial [angular; linear].

        body_q  : (nbody, 7) pos(3)+quat(4, xyzw) per body (state.body_q.numpy()).
        body_qd : (nbody, 6) spatial [linear(0:3); angular(3:6)] world, at body origin.
        defl    : (14,) control-surface joint angles (rad).
        wind    : (3,) world gust velocity added to the freestream (default 0).
        Output body_f layout = [force(0:3); moment(3:6)], world (verified convention).
        """
        nb = body_q.shape[0]
        bf = np.zeros((nb, 6), dtype=np.float32)
        Vw = np.zeros(3) if wind is None else np.asarray(wind, float)
        for b, B in self.bodies.items():
            p = body_q[b, 0:3]
            R = _quat_to_R(body_q[b, 3:7])
            vlin = body_qd[b, 0:3]
            omega = body_qd[b, 3:6]
            chord_hat = R @ np.array([1.0, 0.0, 0.0])    # body x = chordwise (aft +x is TE)
            span_hat = R @ np.array([0.0, 1.0, 0.0])
            norm_hat = R @ np.array([0.0, 0.0, 1.0])
            Fsum = np.zeros(3); Msum = np.zeros(3)
            for i, fr in enumerate(B["frac"]):
                y_s = B["sgn"] * fr * B["span"]          # spanwise station (body frame)
                r_rel = R @ np.array([0.0, y_s, 0.0])    # strip pos rel body origin (world)
                v_pt = vlin + np.cross(omega, r_rel)     # strip point velocity (world)
                u = Vw - v_pt                            # air velocity relative to strip
                u_c = float(u @ chord_hat)               # chordwise component
                u_n = float(u @ norm_hat)                # normal component
                V2 = u_c * u_c + u_n * u_n
                if V2 < 1e-8:
                    continue
                Vmag = np.sqrt(V2)
                aoa = np.arctan2(u_n, -u_c)              # +AoA when wind hits from below-front
                cam = self._camber_at(b, fr, defl) if B["is_wing"] else 0.0
                a_eff = np.clip(aoa + cam, -3 * STALL, 3 * STALL)
                # post-stall soft saturation of CL
                if abs(a_eff) <= STALL:
                    CL = B["CLa"] * a_eff
                else:
                    CL = B["CLa"] * np.sign(a_eff) * STALL * (1.0 - 0.4 *
                         (abs(a_eff) - STALL) / STALL)
                CD = CD0 + CL * CL / (np.pi * B["AR"] * OSWALD)
                q = 0.5 * self.rho * V2
                dA = B["chord"] * B["ds"]
                u_hat = u / Vmag
                # lift ⊥ u in the (chord,norm) plane, toward +norm side; drag along u
                lift_dir = norm_hat - float(norm_hat @ u_hat) * u_hat
                ln = np.linalg.norm(lift_dir)
                lift_dir = lift_dir / ln if ln > 1e-6 else np.zeros(3)
                lift_dir *= np.sign(u_n) if u_n != 0 else 1.0
                Fstrip = q * dA * (CL * lift_dir + CD * u_hat)
                Fsum += Fstrip
                Msum += np.cross(r_rel, Fstrip)
            bf[b, 0:3] = Fsum
            bf[b, 3:6] = Msum
        return bf
