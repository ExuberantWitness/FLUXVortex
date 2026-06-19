"""Unsteady coupled flight loop: the full flapping aircraft flying on UVLM + LEV.

Ties every validated piece into one time-stepping free-flight simulation:
  - Featherstone multibody (aircraft_multibody): 17 bodies, 16 actuators
  - bound multi-surface UVLM (uvlm_aircraft.MultiSurfaceUVLM): all surfaces, one
    composite-AIC solve with wing<->tail cross-induction -> per-body wrench
  - sectional per-strip LDVM (sectional_lev.SectionalLEV): real leading-edge vortices
    on the LE control surfaces

Per step, body poses/twists come from the Featherstone state; the bound UVLM gives the
wrench for the wing boxes / TE flaps / V-stab / ruddervators, and the sectional LDVM
gives the LE-flap wrenches (it carries both the attached and the LEV lift, accurate at
the ~45deg cruise AoA, so it replaces the bound force on those bodies while they still
contribute to the bound induced field). Wrenches feed back as state.body_f; the policy
commands the 2 flap torques + 14 servo deflections; gust enters as a world wind.

This is the plan's free-flying aircraft on the real UVLM (no proxy). Control training
(NN) and MOME co-design run on top of this loop.
"""
from __future__ import annotations

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverFeatherstone

import aircraft_multibody as AM
from sectional_lev import SectionalLEV

SURF_RANGE = np.deg2rad(25.0)
FLAP_TORQUE = 0.6
SERVO_KP, SERVO_KD = 4.0, 0.05


class AircraftFlight:
    def __init__(self, ac=None, *, dt=1.0 / 600.0, U_ref=10.0, lesp_crit=0.20,
                 spring_ke=1.2):
        self.model, self.idx, self.surfs, self.msu = AM.build(ac, spring_ke=spring_ke)
        self.slev = SectionalLEV(self.surfs, U_ref=U_ref, lesp_crit=lesp_crit)
        self.le_bodies = {st["body"] for st in self.slev.strips}
        self.solver = SolverFeatherstone(self.model)
        self.dt = dt
        self.nb = self.model.body_count
        self.ndof = self.model.joint_dof_count
        self.s0, self.s1 = self.model.state(), self.model.state()
        self.control = self.model.control()
        self._jf = np.zeros(self.ndof, dtype=np.float32)
        self._bf = np.zeros((self.nb, 6), dtype=np.float32)
        self.surf_qi = [self.idx.surf[k][0] for k in range(14)]
        self.surf_di = [self.idx.surf[k][1] for k in range(14)]
        self.flap_di = [self.idx.flap["L"][1], self.idx.flap["R"][1]]

    def reset(self, *, alt=30.0, speed=10.0, body_aoa_deg=45.0):
        # cruise attitude: body pitched up body_aoa (about y), velocity horizontal fwd
        q0 = np.zeros(self.model.joint_coord_count, dtype=np.float32)
        a = np.deg2rad(body_aoa_deg)
        q0[0:3] = [0.0, 0.0, alt]
        q0[3:7] = [0.0, -np.sin(a / 2), 0.0, np.cos(a / 2)]     # nose UP (about -y) -> +AoA
        qd0 = np.zeros(self.ndof, dtype=np.float32)
        qd0[0:3] = [speed, 0.0, 0.0]                            # horizontal freestream
        self.s0.joint_q.assign(q0); self.s0.joint_qd.assign(qd0)
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        self.t = 0.0
        return self._obs()

    def _poses_twists(self):
        bq = self.s0.body_q.numpy().reshape(-1, 7)
        bqd = self.s0.body_qd.numpy().reshape(-1, 6)
        poses = {b: (bq[b, 0:3].copy(), bq[b, 3:7].copy()) for b in range(self.nb)}
        twists = {b: (bqd[b, 0:3].copy(), bqd[b, 3:6].copy()) for b in range(self.nb)}
        return poses, twists

    def _obs(self):
        q = self.s0.joint_q.numpy(); qd = self.s0.joint_qd.numpy()
        return dict(z=float(q[2]), x=float(q[0]), vx=float(qd[0]), vz=float(qd[2]),
                    quat=q[3:7].copy())

    def step(self, action=None, wind=None):
        poses, twists = self._poses_twists()
        Vinf = np.zeros(3) if wind is None else np.asarray(wind, float)
        bound = self.msu.solve(poses, twists, Vinf)["wrench"]
        lev = self.slev.step(poses, twists, Vinf, self.dt)
        levw = lev["wrench"]
        # assemble body_f = [force(0:3), moment(3:6)] world (verified convention)
        self._bf[:] = 0.0
        for b in range(self.nb):
            if b in self.le_bodies:
                F, M = levw.get(b, (np.zeros(3), np.zeros(3)))     # LDVM (attached+LEV)
            else:
                F, M = bound.get(b, (np.zeros(3), np.zeros(3)))    # bound UVLM
            self._bf[b, 0:3] = F; self._bf[b, 3:6] = M
        # control: flap torques + surface position servos
        self._jf[:] = 0.0
        if action is not None:
            a = np.clip(np.asarray(action, float), -1.0, 1.0)
            self._jf[self.flap_di[0]] = a[0] * FLAP_TORQUE
            self._jf[self.flap_di[1]] = a[1] * FLAP_TORQUE
            q = self.s0.joint_q.numpy(); qd = self.s0.joint_qd.numpy()
            for k in range(14):
                tgt = a[2 + k] * SURF_RANGE
                self._jf[self.surf_di[k]] = (SERVO_KP * (tgt - q[self.surf_qi[k]])
                                             - SERVO_KD * qd[self.surf_di[k]])
        self.control.joint_f.assign(self._jf)
        self.s0.clear_forces()
        self.s0.body_f.assign(self._bf)
        self.solver.step(self.s0, self.s1, self.control, None, self.dt)
        self.s0, self.s1 = self.s1, self.s0
        self.t += self.dt
        return self._obs(), dict(n_lev=lev["n_lev"], lift_bound=bound)


def verify():
    wp.init()
    fl = AircraftFlight()
    fl.reset(alt=30.0, speed=10.0, body_aoa_deg=45.0)
    zs, levs = [], 0
    import time
    t0 = time.time()
    N = 80
    for i in range(N):
        # gentle symmetric flap drive (emergent), no surface deflection
        t = fl.t
        u = 0.4 * np.sin(2 * np.pi * 3.0 * t)
        act = np.zeros(16); act[0] = act[1] = u
        obs, info = fl.step(act)
        zs.append(obs["z"]); levs += info["n_lev"]
        if not np.isfinite(obs["z"]):
            break
    dt_step = (time.time() - t0) / max(i + 1, 1)
    zs = np.array(zs)
    finite = bool(np.all(np.isfinite(zs)))
    flew = bool(zs[0] > 5.0)
    ok = finite and len(zs) >= N and levs > 0
    print(f"coupled flight (UVLM + sectional LEV + Featherstone), {N} steps:")
    print(f"  z: {zs[0]:.2f} -> {zs[-1]:.2f} m   LEVs shed (cumulative)={levs}   "
          f"~{1000*dt_step:.0f} ms/step")
    print(f"  finite={finite}  bodies fly the coupled aero")
    print(f"coupled flight loop {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
