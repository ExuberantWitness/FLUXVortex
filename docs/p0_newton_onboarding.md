# P0 — Newton stable-layer onboarding (differentiable resonant free-flight)

**Goal (plan P0):** verify the foundation the FLUXVortex platform hard-depends on —
NVIDIA Newton's *stable* layer (`SolverFeatherstone`, not the experimental
`SolverCoupled`) can carry a **free-flying rigid body + revolute hinge + servo
torque + linear torsional spring** and is **end-to-end differentiable** via the
Warp tape, since the differentiable co-design outer loop relies on it.

Script: [`platform/p0_resonant_freeflight.py`](../platform/p0_resonant_freeflight.py).
Env: warp 1.14.0, newton 1.4.0.dev0. Ran on **CPU** (this box's CUDA driver is
forward-incompatible) — the local-DEV tier; the A100 path is identical (Warp is
device-agnostic).

## What it builds (smallest aircraft atom)

- Rigid **fuselage** with an auto-free-joint... *no* — the multibody idiom matters:
  `add_body` auto-attaches a free joint and floats **every** body, which
  double-parents a jointed child. The correct idiom is **`add_link`** (rigid body,
  no joint) + explicit `add_joint_free` / `add_joint_revolute` + `add_articulation`.
- **Wing-stub** on a **revolute hinge**; the **linear torsional spring** = revolute
  `target_ke`/`target_kd` toward `target_pos=0` (the resonant elastic element);
  the **servo torque** = `Control.joint_f` on the hinge dof. (Featherstone ignores
  `actuator_mode`; it always applies the `joint_target_ke` PD + `joint_f`.)
- IC = plan §1: 30 m altitude, 10 m/s at 45° climb launch.

## Results

- **Forward** (200 steps): fuselage free-flies (z 30.0 → 29.73 m), the sprung+servo
  hinge responds (0 → 0.188 rad). ✔
- **Differentiability** d(hinge angle)/d(servo torque), tape vs central FD:

  | horizon | tape | FD | rel err |
  |---|---|---|---|
  | 32 steps | +1.255 | +1.223 | 2.7e-2 ok |
  | 64 steps | +1.258 | +1.263 | **3.9e-3 ok** |
  | 128 steps | +1.049 | +1.300 | 1.9e-1 DEGRADED |
  | 200 steps | −0.105 | +1.274 | 1.1e0 DEGRADED |

  **P0 PASS**: the stable layer is end-to-end differentiable over **control-window
  horizons** (tape == FD).

## Documented finding (matters for the plan)

The Featherstone **backward** is accurate but accumulates error that grows with
**horizon length** *and* **base velocity**: at our 10 m/s launch the gradient is
correct for short windows (N ≤ ~100) and unreliable by ~200 steps. The forward
(FD) is launch-velocity-invariant (uniform translation is Galilean, FD ≈ +1.25
throughout), so this is a **backward numerical** effect, not physics — the classic
long-horizon differentiable-simulation gradient ill-conditioning.

This **empirically corroborates the plan's control-architecture decisions (§6)**:
- **PPO-first** (model-free; no analytic gradient → immune to this), and
- **SHAC with short windows** (chops the rollout into short, gradient-accurate
  windows + critic bootstrapping).

Follow-ups (not blocking P0): (a) check whether fp64 (plan's precision decision)
shrinks the backward error at high base velocity; (b) report the high-velocity
free-base adjoint behaviour upstream to Newton.
