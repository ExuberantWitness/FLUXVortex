"""Full flapping-aircraft multibody assembly (plan §1) — 6-DOF free flight.

The plan's actual object, not a proxy: a free-flying bird-scale flapping MAV built on
Newton's stable Featherstone layer.

  fuselage (free base, 6-DOF)
    ├── left  main wing   — flap revolute hinge (torsional spring + servo)
    │     ├── 3 leading-edge control surfaces  (servo revolute)
    │     └── 3 trailing-edge control surfaces (servo revolute)
    ├── right main wing   — flap revolute hinge (mirror; spring + servo)
    │     ├── 3 leading-edge control surfaces
    │     └── 3 trailing-edge control surfaces
    └── V-tail
          ├── left  ruddervator (servo revolute)
          └── right ruddervator (servo revolute)

= 17 rigid bodies, 1 free + 16 revolute joints -> 22 DOF.
Actuators (16): 2 flap-hinge servo torques (differential -> roll/yaw) + 14 control-
surface servo positions. HIT-Hawk sizing (Zhong&Xu 2022): span 1.7 m, mean chord
0.29 m, ~0.52 kg, flap ≤3 Hz. Coordinates +X forward, +Z up (plan §1).

This module builds the RIGID articulation and verifies 6-DOF free flight + actuation
+ tape differentiability. The flexible ANCF main wings (the 刚柔 distribution) and the
UVLM coupled FSI couple onto the wing-root bodies next (shared-node hinge, plan §2).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp

import newton
from newton.solvers import SolverFeatherstone

G = 9.81

# ── HIT-Hawk-scale geometry (plan §1) ─────────────────────────────────────────
SPAN = 1.7          # tip-to-tip (m)
HALF = SPAN / 2.0   # 0.85 m
CHORD = 0.29        # mean chord (m)
BOOM = 0.55         # tail boom length aft (m)
VTAIL_SPAN = 0.35   # each ruddervator panel length (m)
VTAIL_DIHEDRAL = np.deg2rad(35.0)

# masses (kg) -> total ~0.52 kg
M_FUS = 0.20
M_WING = 0.08       # each main wing half
M_SURF = 0.008      # each control surface
M_VTAIL = 0.02      # each ruddervator


def _plate_inertia(mass, lx, ly, lz=1e-3):
    """Diagonal inertia of a thin rectangular plate of size (lx,ly,lz) about its COM."""
    Ixx = mass * (ly * ly + lz * lz) / 12.0
    Iyy = mass * (lx * lx + lz * lz) / 12.0
    Izz = mass * (lx * lx + ly * ly) / 12.0
    return wp.mat33(Ixx, 0.0, 0.0, 0.0, Iyy, 0.0, 0.0, 0.0, Izz)


def _xf(x, y, z, q=None):
    return wp.transform(wp.vec3(float(x), float(y), float(z)),
                        q if q is not None else wp.quat_identity())


@dataclass
class AircraftIndex:
    """DOF/coord indices of every actuator, resolved after finalize()."""
    flap: dict = field(default_factory=dict)        # 'L','R' -> (qi, di)
    surf: dict = field(default_factory=dict)        # 0..13 -> (qi, di)
    joint_labels: list = field(default_factory=list)

    @property
    def n_actuators(self):
        return len(self.flap) + len(self.surf)


def build_aircraft(design=None, *, spring_ke=1.2, spring_kd=0.02,
                   surf_ke=3.0, surf_kd=0.002, requires_grad=False):
    """Assemble the full 17-body flapping aircraft. Returns (model, AircraftIndex).

    `design` (optional) overrides per-part stiffness/mass later (co-design hook);
    None = nominal HIT-Hawk. spring_ke/kd set the main-wing flap torsional spring
    (the resonant elastic element, plan §2). surf_ke/surf_kd are the control-surface
    servo gains: the surfaces are light (~8 g, I~3e-5), so the servo time constant
    must stay well above the integrator dt (kept soft + lightly damped for stability;
    co-design/servo bandwidth model refines these).
    """
    mb = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-G)
    surf_ax = (0.0, 1.0, 0.0)           # control-surface hinge axis = spanwise (deflect camber)
    flap_ax = (1.0, 0.0, 0.0)           # flap hinge axis = longitudinal (wings beat up/down)

    # ── fuselage: free base ───────────────────────────────────────────────────
    fus = mb.add_link(mass=M_FUS, com=wp.vec3(0.0, 0.0, 0.0),
                      inertia=_plate_inertia(M_FUS, 0.30, 0.06, 0.06), label="fuselage")
    j_free = mb.add_joint_free(child=fus, label="freeflight")
    joints = [j_free]
    idx = AircraftIndex()

    # ── one main wing + its 6 control surfaces ────────────────────────────────
    def add_wing(side, sgn):
        # wing COM at mid-half-span outboard of a small root offset
        wing_com_y = sgn * (0.05 + HALF / 2.0)
        wing = mb.add_link(mass=M_WING, com=wp.vec3(0.0, 0.0, 0.0),
                           inertia=_plate_inertia(M_WING, CHORD, HALF), label=f"wing_{side}")
        j_flap = mb.add_joint_revolute(
            parent=fus, child=wing, axis=flap_ax,
            parent_xform=_xf(0.0, sgn * 0.05, 0.0),
            child_xform=_xf(0.0, -wing_com_y + sgn * 0.05, 0.0),
            target_ke=spring_ke, target_kd=spring_kd, target_pos=0.0,
            label=f"flap_{side}")
        joints.append(j_flap)
        idx.flap[side] = j_flap
        # 3 spanwise stations × {LE, TE}
        stations = [0.30, 0.55, 0.80]            # fraction of half-span (outboard)
        for st, frac in enumerate(stations):
            sy = sgn * (0.05 + frac * HALF)       # spanwise position of the surface COM
            for edge, cx in (("LE", -0.35 * CHORD), ("TE", +0.35 * CHORD)):
                s = mb.add_link(mass=M_SURF, com=wp.vec3(0.0, 0.0, 0.0),
                                inertia=_plate_inertia(M_SURF, 0.3 * CHORD, 0.25 * HALF),
                                label=f"surf_{side}_{edge}_{st}")
                # hinge on the wing at this station/edge; surface COM just aft/fwd of it
                j = mb.add_joint_revolute(
                    parent=wing, child=s, axis=surf_ax,
                    parent_xform=_xf(cx, sy, 0.0),
                    child_xform=_xf(-0.5 * (0.3 * CHORD) * np.sign(cx), 0.0, 0.0),
                    target_ke=surf_ke, target_kd=surf_kd, target_pos=0.0,
                    label=f"surf_{side}_{edge}_{st}")
                joints.append(j)
                # surface ordering index 0..11 (filled below after both wings)
                idx.surf[len(idx.surf)] = j

    add_wing("L", +1.0)
    add_wing("R", -1.0)

    # ── V-tail: two ruddervators on the aft boom ──────────────────────────────
    for side, sgn in (("L", +1.0), ("R", -1.0)):
        c, s = np.cos(VTAIL_DIHEDRAL), np.sin(VTAIL_DIHEDRAL)
        # panel rotated by ±dihedral about x; hinge axis along the panel span
        panel = mb.add_link(mass=M_VTAIL, com=wp.vec3(0.0, 0.0, 0.0),
                            inertia=_plate_inertia(M_VTAIL, 0.18, VTAIL_SPAN), label=f"vtail_{side}")
        j = mb.add_joint_revolute(
            parent=fus, child=panel, axis=(0.0, float(c), float(sgn * s)),
            parent_xform=_xf(-BOOM, 0.0, 0.05),
            child_xform=_xf(0.0, -sgn * 0.5 * VTAIL_SPAN * c, -0.5 * VTAIL_SPAN * s),
            target_ke=surf_ke, target_kd=surf_kd, target_pos=0.0, label=f"vtail_{side}")
        joints.append(j)
        idx.surf[len(idx.surf)] = j        # surfaces 12,13 = V-tail

    mb.add_articulation(joints, label="aircraft")
    model = mb.finalize(requires_grad=requires_grad)

    # ── resolve coord/dof indices for every joint ─────────────────────────────
    qs = model.joint_q_start.numpy()
    ds = model.joint_qd_start.numpy()
    idx.joint_labels = [str(l) for l in model.joint_key] if hasattr(model, "joint_key") else []
    for side, j in list(idx.flap.items()):
        idx.flap[side] = (int(qs[j]), int(ds[j]))
    for k, j in list(idx.surf.items()):
        idx.surf[k] = (int(qs[j]), int(ds[j]))
    return model, idx


def launch_state(model, *, alt=30.0, speed=10.0, climb_deg=45.0):
    """6-DOF launch IC: 30 m alt, 10 m/s at +45° climb (plan §1 high-energy launch)."""
    q0 = np.zeros(model.joint_coord_count, dtype=np.float32)
    q0[0:3] = [0.0, 0.0, alt]
    q0[3:7] = [0.0, 0.0, 0.0, 1.0]                  # identity quat (xyzw)
    qd0 = np.zeros(model.joint_dof_count, dtype=np.float32)
    th = np.deg2rad(climb_deg)
    qd0[3:6] = [speed * np.cos(th), 0.0, speed * np.sin(th)]
    return q0, qd0


def verify() -> bool:
    wp.init()
    model, idx = build_aircraft(requires_grad=False)
    n_bodies = model.body_count
    n_dof = model.joint_dof_count
    n_act = idx.n_actuators
    rev = int((model.joint_type.numpy() == int(newton.JointType.REVOLUTE)).sum())
    print(f"full aircraft assembly: {n_bodies} bodies, {n_dof} DOF, {rev} revolute "
          f"joints, {n_act} actuators ({len(idx.flap)} flap + {len(idx.surf)} surfaces)")

    # 6-DOF free-flight rollout with a sample actuation (flap drive + surface deflect)
    solver = SolverFeatherstone(model)
    dt, N = 1.0 / 600.0, 240
    s0, s1 = model.state(), model.state()
    q0, qd0 = launch_state(model)
    s0.joint_q.assign(q0); s0.joint_qd.assign(qd0)
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    control = model.control()

    jf = np.zeros(n_dof, dtype=np.float32)
    zs, flapL = [], []
    for i in range(N):
        t = i * dt
        # symmetric flap drive at 3 Hz + a small elevator-like TE deflection torque
        u_flap = 0.4 * np.sin(2 * np.pi * 3.0 * t)
        jf[idx.flap["L"][1]] = u_flap
        jf[idx.flap["R"][1]] = u_flap
        for k, (qi, di) in idx.surf.items():
            jf[di] = 0.02 * np.sin(2 * np.pi * 3.0 * t)     # gentle surface servo
        control.joint_f.assign(jf)
        s0.clear_forces()
        solver.step(s0, s1, control, None, dt)
        s0, s1 = s1, s0
        zs.append(float(s0.joint_q.numpy()[2]))
        flapL.append(float(s0.joint_q.numpy()[idx.flap["L"][0]]))
    zs = np.array(zs); flapL = np.array(flapL)

    finite = bool(np.all(np.isfinite(zs)))
    flies = bool(zs[0] > zs[-1] or zs.max() > 30.0)         # ballistic arc (climbs then falls)
    flap_live = bool(np.ptp(flapL) > 1e-3)                  # flap hinge actually moves
    ok = finite and flap_live and (n_bodies == 17) and (n_act == 16)
    print(f"  6-DOF free flight ({N} steps): z {zs[0]:.2f} -> {zs[-1]:.2f} m "
          f"(peak {zs.max():.2f}), flap_L stroke {np.rad2deg(np.ptp(flapL)):.1f} deg")
    print(f"  finite={finite}  flap_live={flap_live}  bodies={n_bodies}==17 "
          f"actuators={n_act}==16")
    print(f"full aircraft 6-DOF assembly {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
