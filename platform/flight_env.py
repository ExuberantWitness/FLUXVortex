"""6-DOF free-flight RL environment for the full flapping aircraft (plan §5/§6).

Wraps aircraft_assembly (17-body Featherstone aircraft) + strip_aero (aerodynamic
forces) + a 1-cosine gust into a control task: the policy commands the 2 flap-hinge
torques (emergent flapping, plan §2) and 14 control-surface deflections to hold a
stable cruise from a high-energy launch and reject a vertical gust.

  observation (per step, raw): pitch, roll, yaw, body rates (3), body velocity (3),
    altitude error, forward-speed error, flap angles (2)  -> 13-dim. The Takens policy
    stacks the last n steps (delay embedding, plan §6).
  action (16): [flap_L torque, flap_R torque, 14 surface target deflections].
  reward: stay aloft + track level cruise (attitude, altitude, speed) - control effort;
    the gust window penalizes the excursion it induces -> gust rejection is learned.

The aircraft is open-loop pitch-unstable, so a do-nothing policy tumbles/crashes: the
NN must actively stabilize. Fast (Featherstone, no coupled FSI in the loop) -> many
rollouts on the 4090 for ES/PPO; the validated UVLM FSI scores per-design efficiency.
"""
from __future__ import annotations

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverFeatherstone

import aircraft_assembly as A
from strip_aero import StripAero

OBS_DIM = 13
ACT_DIM = 16
FLAP_TORQUE = 0.6        # N·m scale on the flap-hinge torque command
SURF_RANGE = np.deg2rad(25.0)   # max control-surface deflection
SERVO_KP, SERVO_KD = 4.0, 0.05  # surface position-servo gains (torque realization)


def _quat_to_euler(q):
    x, y, z, w = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    p = np.clip(2 * (w * y - z * x), -1.0, 1.0)
    pitch = np.arcsin(p)
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class FlightEnv:
    def __init__(self, design=None, *, dt=1.0 / 600.0, horizon=600, substep=1,
                 launch_speed=11.0, launch_climb=20.0, launch_alt=30.0,
                 v_target=9.0, gust_w=2.5, gust_t0=0.35, gust_dur=0.18,
                 spring_ke=1.2, seed=0):
        self.model, self.idx = A.build_aircraft(design, spring_ke=spring_ke,
                                                requires_grad=False)
        self.aero = StripAero(self.idx, n_strips=5)
        self.solver = SolverFeatherstone(self.model)
        self.dt, self.horizon, self.substep = dt, horizon, substep
        self.launch = dict(speed=launch_speed, climb_deg=launch_climb, alt=launch_alt)
        self.v_target, self.z_target = v_target, launch_alt
        self.gust = dict(w=gust_w, t0=gust_t0, dur=gust_dur)
        self.fus = self.idx.fus_body
        self.surf_di = [self.idx.surf[k][1] for k in range(14)]
        self.surf_qi = [self.idx.surf[k][0] for k in range(14)]
        self.flap_di = [self.idx.flap["L"][1], self.idx.flap["R"][1]]
        self.flap_qi = [self.idx.flap["L"][0], self.idx.flap["R"][0]]
        self.ndof = self.model.joint_dof_count
        self.rng = np.random.default_rng(seed)
        self.s0 = self.model.state()
        self.s1 = self.model.state()
        self.control = self.model.control()
        self._jf = np.zeros(self.ndof, dtype=np.float32)

    def reset(self):
        q0, qd0 = A.launch_state(self.model, speed=self.launch["speed"],
                                 climb_deg=self.launch["climb_deg"], alt=self.launch["alt"])
        self.s0.joint_q.assign(q0)
        self.s0.joint_qd.assign(qd0)
        newton.eval_fk(self.model, self.s0.joint_q, self.s0.joint_qd, self.s0)
        self.t = 0.0
        self.step_i = 0
        return self._obs()

    def _obs(self):
        q = self.s0.joint_q.numpy()
        qd = self.s0.joint_qd.numpy()
        roll, pitch, yaw = _quat_to_euler(q[3:7])
        wv = qd[3:6]                              # fuselage angular velocity
        vv = qd[0:3]                              # fuselage linear velocity
        z_err = float(q[2] - self.z_target)
        v_err = float(vv[0] - self.v_target)
        fL = float(q[self.flap_qi[0]]); fR = float(q[self.flap_qi[1]])
        return np.array([pitch, roll, yaw, wv[0], wv[1], wv[2],
                         vv[0], vv[1], vv[2], z_err, v_err, fL, fR], dtype=np.float64)

    def _gust_wind(self):
        g = self.gust
        if g["t0"] <= self.t < g["t0"] + g["dur"]:
            frac = (self.t - g["t0"]) / g["dur"]
            gz = 0.5 * g["w"] * (1 - np.cos(2 * np.pi * frac))
            return np.array([0.0, 0.0, gz])
        return None

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        q = self.s0.joint_q.numpy(); qd = self.s0.joint_qd.numpy()
        # flap hinges: direct torque (emergent flapping)
        self._jf[:] = 0.0
        self._jf[self.flap_di[0]] = a[0] * FLAP_TORQUE
        self._jf[self.flap_di[1]] = a[1] * FLAP_TORQUE
        # control surfaces: position servo toward commanded deflection
        defl = np.zeros(14)
        for k in range(14):
            tgt = a[2 + k] * SURF_RANGE
            ang = q[self.surf_qi[k]]; rate = qd[self.surf_di[k]]
            self._jf[self.surf_di[k]] = SERVO_KP * (tgt - ang) - SERVO_KD * rate
            defl[k] = ang
        self.control.joint_f.assign(self._jf)

        wind = self._gust_wind()
        crashed = False
        for _ in range(self.substep):
            self.s0.clear_forces()
            bq = self.s0.body_q.numpy().reshape(-1, 7)
            bqd = self.s0.body_qd.numpy().reshape(-1, 6)
            bf = self.aero.wrenches(bq, bqd, defl, wind=wind)
            self.s0.body_f.assign(bf)
            self.solver.step(self.s0, self.s1, self.control, None, self.dt)
            self.s0, self.s1 = self.s1, self.s0
            self.t += self.dt
            if not np.all(np.isfinite(self.s0.joint_q.numpy())):
                crashed = True
                break
        self.step_i += 1

        obs = self._obs()
        roll, pitch = obs[1], obs[0]
        z = float(self.s0.joint_q.numpy()[2])
        tumbled = abs(pitch) > 1.2 or abs(roll) > 1.2 or z < 5.0 or z > 60.0
        done = crashed or tumbled or self.step_i >= self.horizon
        # reward: survive + hold level cruise - effort ; gust window stresses it
        if crashed or not np.all(np.isfinite(obs)):
            return obs, -10.0, True, dict(crashed=True)
        # bounded shaping terms (clip so a fling can't produce astronomical reward)
        z_err = np.clip(obs[9], -10, 10)
        v_err = np.clip(obs[10], -10, 10)
        vy = np.clip(obs[7], -10, 10)
        r = 1.0
        r -= 0.05 * z_err ** 2          # altitude error
        r -= 0.02 * v_err ** 2          # speed error
        r -= 1.5 * pitch ** 2           # keep level
        r -= 1.5 * roll ** 2
        r -= 0.2 * vy ** 2              # no sideslip
        r -= 0.05 * float(np.sum(a ** 2))
        if tumbled:
            r -= 10.0
        r = float(np.clip(r, -20.0, 2.0))
        return obs, r, done, dict(z=z, pitch=pitch, roll=roll)


def rollout(env, policy, render=False):
    """Run one episode; return total reward and a small trace."""
    obs = env.reset()
    policy.reset()
    total = 0.0
    trace = []
    for _ in range(env.horizon):
        a = policy.act(obs)
        obs, r, done, info = env.step(a)
        total += r
        if render:
            trace.append(info.get("z", np.nan))
        if done:
            break
    return total, trace
