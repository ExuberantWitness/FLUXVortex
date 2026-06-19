"""Fast 6-DOF flight environment for PPO control training (plan §6, PPO-first).

Rigid-body 6-DOF flight dynamics with the UVLM-DERIVED aero (uvlm_db, the validated
multi-surface UVLM tabulated -> microsecond interpolation, NOT strip theory). This is
the plan's two-fidelity structure: PPO control trains on this fast model; the expensive
coupled FSI scores per-design efficiency.

The flapping MAV cruises at ~45deg body AoA with the wing FEATHERED to the attached
domain (literature). The policy commands:
  [wing_aoa_cmd, roll_moment, yaw_moment, thrust]
i.e. it controls the feathered wing AoA (lift), the roll/yaw control authority, and the
flapping thrust (speed). The task: hold a stable cruise from the high-energy launch and
reject a 1-cosine vertical gust. Reward: stay aloft + track cruise - effort; the gust
window stresses it -> gust rejection is learned.

Fast (no UVLM in the loop) -> thousands of rollouts for PPO. Observations are the Takens
delay-embedding stack (the policy net handles the stacking).
"""
from __future__ import annotations

import numpy as np

from uvlm_db import AeroDB

G = 9.81
OBS_DIM = 10
ACT_DIM = 4


def _quat_to_R(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def _quat_integrate(q, om_world, dt):
    wx, wy, wz = om_world
    x, y, z, w = q
    dq = 0.5 * np.array([wx * w + wy * z - wz * y, wy * w + wz * x - wx * z,
                         wz * w + wx * y - wy * x, -(wx * x + wy * y + wz * z)])
    q = q + dq * dt
    return q / (np.linalg.norm(q) + 1e-12)


def _euler(q):
    x, y, z, w = q
    roll = np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2 * (w * y - z * x), -1, 1))
    yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return roll, pitch, yaw


class FlightPPOEnv:
    def __init__(self, *, dt=0.01, horizon=400, m=0.45, V_cruise=8.0, alt=30.0,
                 body_aoa_deg=45.0, gust_w=2.5, gust_t0=1.0, gust_dur=0.5, seed=0):
        self.aero = AeroDB()
        self.dt, self.horizon, self.m = dt, horizon, m
        self.S = self.aero.S
        self.I = np.diag([3e-3, 5e-3, 5e-3])
        self.Iinv = np.linalg.inv(self.I)
        self.V_cruise, self.alt0 = V_cruise, alt
        self.body_aoa = np.deg2rad(body_aoa_deg)
        self.gust = dict(w=gust_w, t0=gust_t0, dur=gust_dur)
        self.T_max = 6.0          # flapping thrust authority (N)
        self.M_max = 0.4          # roll/yaw control moment authority (N*m)
        self.rng = np.random.default_rng(seed)

    def reset(self):
        a = self.body_aoa
        self.x = np.array([0.0, 0.0, self.alt0])
        self.q = np.array([0.0, -np.sin(a / 2), 0.0, np.cos(a / 2)])
        self.v = np.array([self.V_cruise, 0.0, 0.0])
        self.om = np.zeros(3)
        self.t = 0.0; self.step_i = 0
        return self._obs()

    def _gust(self):
        g = self.gust
        if g["t0"] <= self.t < g["t0"] + g["dur"]:
            fr = (self.t - g["t0"]) / g["dur"]
            return 0.5 * g["w"] * (1 - np.cos(2 * np.pi * fr))
        return 0.0

    def _obs(self):
        roll, pitch, yaw = _euler(self.q)
        return np.array([pitch, roll, self.om[0], self.om[1], self.om[2],
                         self.v[0] - self.V_cruise, self.v[1], self.v[2],
                         self.x[2] - self.alt0, self._gust()], dtype=np.float64)

    def step(self, action):
        a = np.clip(np.asarray(action, float), -1.0, 1.0)
        wing_aoa = a[0] * self.aero.att          # commanded feathered wing AoA (<= attached)
        roll_m, yaw_m, thr = a[1] * self.M_max, a[2] * self.M_max, 0.5 * (a[3] + 1.0)
        R = _quat_to_R(self.q)
        Vg = np.array([0.0, 0.0, self._gust()])
        Vrel = R.T @ (Vg - self.v)               # body-frame relative wind
        sp = np.linalg.norm(Vrel) + 1e-9
        q_dyn = 0.5 * 1.225 * sp * sp
        cl, cd = self.aero.coeffs(wing_aoa)
        # lift perpendicular to relative wind (body x-z), drag along it
        uhat = Vrel / sp
        lift_dir = np.array([0, 0, 1.0]) - np.dot(np.array([0, 0, 1.0]), uhat) * uhat
        ln = np.linalg.norm(lift_dir); lift_dir = lift_dir / ln if ln > 1e-6 else np.array([0, 0, 1.0])
        F_body = q_dyn * self.S * (cl * lift_dir + cd * uhat)
        F_body += np.array([thr * self.T_max, 0.0, 0.0])          # flapping thrust (body +x)
        F_world = R @ F_body + np.array([0, 0, -self.m * G])
        M_world = R @ np.array([roll_m, 0.0, yaw_m])
        # pitch authority via the wing-AoA-induced moment is implicit; add light pitch damping
        M_world += -0.05 * self.om
        self.v = self.v + (F_world / self.m) * self.dt
        self.x = self.x + self.v * self.dt
        self.om = self.om + (self.Iinv @ (M_world - np.cross(self.om, self.I @ self.om))) * self.dt
        self.q = _quat_integrate(self.q, self.om, self.dt)
        self.t += self.dt; self.step_i += 1

        obs = self._obs()
        roll, pitch, yaw = _euler(self.q)
        z = self.x[2]
        crashed = (not np.all(np.isfinite(obs))) or z < 5 or z > 60 or abs(roll) > 1.4
        r = 1.0
        r -= 0.04 * np.clip(z - self.alt0, -15, 15) ** 2
        r -= 0.05 * np.clip(self.v[0] - self.V_cruise, -10, 10) ** 2
        r -= 0.3 * np.clip(self.v[2], -10, 10) ** 2          # minimize vertical excursion
        r -= 0.2 * roll ** 2 + 0.1 * self.v[1] ** 2
        r -= 0.02 * float(np.sum(a ** 2))
        if crashed:
            return obs, -20.0, True, {}
        done = self.step_i >= self.horizon
        return obs, float(np.clip(r, -20, 2)), done, {"z": float(z)}


def rollout(env, policy):
    obs = env.reset(); policy.reset(); tot = 0.0; traj = []
    for _ in range(env.horizon):
        a = policy.act(obs); obs, r, d, info = env.step(a); tot += r
        traj.append(obs)
        if d:
            break
    return tot, np.array(traj)


if __name__ == "__main__":
    import warp as wp; wp.init()
    import time
    env = FlightPPOEnv()
    o = env.reset()
    t0 = time.time(); n = 2000
    for _ in range(n):
        env.step(np.zeros(4))
        if env.step_i >= env.horizon:
            env.reset()
    print(f"fast flight env: {1e6 * (time.time() - t0) / n:.1f} us/step "
          f"({n} steps); obs dim={OBS_DIM} act dim={ACT_DIM}")
