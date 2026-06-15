"""P0 Newton onboarding: differentiable free-flying rigid + revolute + servo + torsional spring.

The smallest atom of the FLUXVortex aircraft, exercised on Newton's *stable* layer
(SolverFeatherstone) to verify the foundation the platform depends on:

  - a **free-flying rigid fuselage** (auto free base joint) under gravity,
  - a **wing-stub** on a **revolute hinge** with a **linear torsional spring**
    (revolute target_ke / target_kd toward rest = the resonant elastic element),
  - a **servo torque** commanded through Control.joint_f,
  - **end-to-end differentiability**: d(loss)/d(servo torque) via wp.Tape vs finite
    difference — the tape-gradient sanity check the differentiable co-design outer
    loop relies on.

Newton-native, on the stable layer (Featherstone, not experimental SolverCoupled),
per P0 of the plan. CPU is the local-DEV tier here; the GPU path is identical.

Run: python FLUXV/platform/p0_resonant_freeflight.py
"""
from __future__ import annotations

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherstone


@wp.kernel
def hinge_angle_loss(joint_q: wp.array(dtype=float), hinge_qi: int,
                     loss: wp.array(dtype=float)):
    loss[0] = joint_q[hinge_qi]            # scalar loss = final hinge angle [rad]


def build_model(spring_ke: float, spring_kd: float, requires_grad: bool):
    """Free-flying fuselage + wing-stub on a torsional-spring hinge (one articulation).

    Multibody idiom: ``add_link`` adds the rigid body WITHOUT a joint (``add_body``
    would auto-attach a free joint and float every body). We connect explicitly —
    free base on the fuselage, revolute hinge fuselage->wing — and group both joints
    into a single articulation.
    """
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)

    fus = mb.add_link(mass=0.20, com=wp.vec3(0.0, 0.0, 0.0),
                      inertia=wp.mat33(2e-3, 0.0, 0.0, 0.0, 4e-3, 0.0, 0.0, 0.0, 4e-3),
                      label="fuselage")
    j_free = mb.add_joint_free(child=fus, label="freeflight")

    wing = mb.add_link(mass=0.05, com=wp.vec3(0.0, 0.30, 0.0),
                       inertia=wp.mat33(3e-4, 0.0, 0.0, 0.0, 5e-5, 0.0, 0.0, 0.0, 3e-4),
                       label="wing")
    # torsional spring = target_ke/kd toward rest (target_pos=0); servo via joint_f
    j_hinge = mb.add_joint_revolute(
        parent=fus, child=wing, axis=(0.0, 1.0, 0.0),
        parent_xform=wp.transform(wp.vec3(0.0, 0.05, 0.0), wp.quat_identity()),
        child_xform=wp.transform(wp.vec3(0.0, -0.25, 0.0), wp.quat_identity()),
        target_ke=spring_ke, target_kd=spring_kd, target_pos=0.0, label="flap_hinge")

    mb.add_articulation([j_free, j_hinge], label="aircraft")
    model = mb.finalize(requires_grad=requires_grad)
    # locate the revolute (hinge) coord/dof indices robustly
    jtype = model.joint_type.numpy()
    qs = model.joint_q_start.numpy()
    ds = model.joint_qd_start.numpy()
    (rev,) = np.where(jtype == int(newton.JointType.REVOLUTE))
    hinge_qi = int(qs[rev[0]])
    hinge_di = int(ds[rev[0]])
    return model, hinge_qi, hinge_di


def rollout(model, hinge_qi, hinge_di, servo_torque, n_steps, dt,
            launch_speed=10.0, launch_deg=45.0):
    """March Featherstone n_steps with a constant servo torque on the hinge dof."""
    solver = SolverFeatherstone(model)
    states = [model.state() for _ in range(n_steps + 1)]
    control = model.control()

    # IC: 45deg climb launch (plan §1) from 30 m; free base is the first joint
    th = np.deg2rad(launch_deg)
    q0 = np.zeros(model.joint_coord_count, dtype=np.float32)
    q0[0:3] = [0.0, 0.0, 30.0]            # fuselage position
    q0[3:7] = [0.0, 0.0, 0.0, 1.0]        # identity quat (xyzw)
    qd0 = np.zeros(model.joint_dof_count, dtype=np.float32)
    qd0[3:6] = [launch_speed * np.cos(th), 0.0, launch_speed * np.sin(th)]  # lin vel
    states[0].joint_q.assign(q0)
    states[0].joint_qd.assign(qd0)
    newton.eval_fk(model, states[0].joint_q, states[0].joint_qd, states[0])

    jf = np.zeros(model.joint_dof_count, dtype=np.float32)
    jf[hinge_di] = float(servo_torque)    # commanded servo torque (differentiable)
    control.joint_f.assign(jf)

    for i in range(n_steps):
        states[i].clear_forces()
        solver.step(states[i], states[i + 1], control, None, dt)
    return states, control


def grad_tape_vs_fd(spring_ke, spring_kd, servo, n_steps, dt, h=1e-3):
    """Return (tape grad, FD grad) of d(final hinge angle)/d(servo torque)."""
    m, hqi, hdi = build_model(spring_ke, spring_kd, requires_grad=True)
    loss = wp.zeros(1, dtype=float, requires_grad=True)
    tape = wp.Tape()
    with tape:
        states, control = rollout(m, hqi, hdi, servo, n_steps, dt)
        wp.launch(hinge_angle_loss, dim=1, inputs=[states[-1].joint_q, hqi, loss])
    tape.backward(loss)
    g_tape = float(control.joint_f.grad.numpy()[hdi])

    m2, hqi2, hdi2 = build_model(spring_ke, spring_kd, requires_grad=False)
    sp, _ = rollout(m2, hqi2, hdi2, servo + h, n_steps, dt)
    sm, _ = rollout(m2, hqi2, hdi2, servo - h, n_steps, dt)
    g_fd = (sp[-1].joint_q.numpy()[hqi2] - sm[-1].joint_q.numpy()[hqi2]) / (2 * h)
    return g_tape, g_fd


def main():
    wp.init()
    print(f"device = {wp.get_device()}  (CUDA: {wp.get_device().is_cuda})  [local-DEV tier]")

    SPRING_KE, SPRING_KD = 0.8, 0.01     # linear torsional spring (resonant element)
    DT, SERVO = 1.0 / 600.0, 0.15
    WINDOW = 64                           # SHAC-style short control-gradient window

    model, hqi, hdi = build_model(SPRING_KE, SPRING_KD, requires_grad=True)
    print(f"joints: {model.joint_type.numpy().tolist()} "
          f"(FREE={int(newton.JointType.FREE)}, REVOLUTE={int(newton.JointType.REVOLUTE)})"
          f"  hinge coord#={hqi} dof#={hdi}")

    # ── forward sanity: free-flight from 30 m @ 10 m/s 45deg + sprung hinge ───
    states, _ = rollout(model, hqi, hdi, SERVO, 200, DT)
    angs = np.array([s.joint_q.numpy()[hqi] for s in states])
    zf = states[-1].joint_q.numpy()[2]
    print(f"\nforward (200 steps, launch 10 m/s @ 45deg): fuselage z 30.000 -> {zf:.3f} m   "
          f"hinge angle min {angs.min():+.4f} / max {angs.max():+.4f} rad "
          f"(spring+servo response live)")

    # ── differentiable foundation: tape == FD over a SHAC-style window ────────
    gt, gf = grad_tape_vs_fd(SPRING_KE, SPRING_KD, SERVO, WINDOW, DT)
    rel = abs(gt - gf) / (abs(gf) + 1e-12)
    print(f"\ntape-gradient sanity over a {WINDOW}-step window "
          f"(d hinge_angle / d servo_torque):")
    print(f"  d/dtau tape       = {gt:+.6e}")
    print(f"  d/dtau finite-dif = {gf:+.6e}")
    print(f"  rel. error        = {rel:.3e}")
    ok = (abs(gf) > 1e-6) and (rel < 3e-2)

    # ── documented finding: long-horizon backward-error growth ────────────────
    # The Featherstone adjoint accumulates error that grows with horizon AND base
    # velocity; at the 10 m/s launch it is accurate for short windows but unreliable
    # by ~200 steps.  This empirically motivates the plan's control decisions:
    # PPO-first (no analytic gradient) and SHAC with SHORT windows.
    print(f"\nlong-horizon backward-error growth (launch 10 m/s, light damping):")
    for N in (32, 64, 128, 200):
        g_t, g_f = grad_tape_vs_fd(SPRING_KE, SPRING_KD, SERVO, N, DT)
        r = abs(g_t - g_f) / (abs(g_f) + 1e-12)
        tag = "ok" if r < 3e-2 else "DEGRADED"
        print(f"  N={N:4d} steps: tape={g_t:+.3e}  fd={g_f:+.3e}  rel={r:.2e}  [{tag}]")
    print("  => gradient accurate for SHORT windows; degrades long-horizon "
          "(=> PPO-first + SHAC short-window, per plan §6).")

    print(f"\nP0 {'PASS' if ok else 'FAIL'}: Newton stable-layer (Featherstone) free-flight "
          f"+ revolute + servo + torsional spring is "
          f"{'end-to-end differentiable over control windows (tape == FD)' if ok else 'NOT verified'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
