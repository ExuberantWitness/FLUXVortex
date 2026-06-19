"""Full flapping aircraft = Featherstone multibody + multi-surface UVLM, on the
corrected parametric geometry (aircraft_geom).

Per the user's decision, the 14 control surfaces are independent Featherstone bodies
with servo hinges (plan §2). Body tree:

  fuselage (free, 6-DOF)  [carries the fixed V-stab UVLM lattices]
    ├ wing box L/R  — flap revolute hinge (axis = longitudinal x) + torsional spring;
    │                 dihedral = the hinge rest angle (flapping oscillates about it)
    │   ├ 3 LE flaps — revolute hinge at the LE-flap/box edge (servo)
    │   └ 3 TE flaps — revolute hinge at the box/TE-flap edge (servo)
    └ 2 ruddervators — revolute hinge at the V-stab TE edge (servo)

Each body's rest pose = identity in the aircraft frame, so its UVLM lattice (built in
the aircraft frame by build_aircraft_surfaces) rides directly on the body pose; flap
deflection rotates the real panel lattice -> true UVLM camber. Hinge lines are taken
from the shared lattice edge between a flap and the wing box, so they sit exactly at
the real leading/trailing edges.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import warp as wp
import newton
from newton.solvers import SolverFeatherstone

from aircraft_geom import Aircraft
import uvlm_aircraft as U

# masses (kg), total ~0.45
M_FUS = 0.20      # fuselage + fixed V-stab
M_BOX = 0.06      # each wing box
M_FLAP = 0.005    # each wing control surface
M_RUD = 0.010     # each ruddervator


def _com_inertia(V, mass):
    pts = V.reshape(-1, 3)
    com = pts.mean(0)
    ext = np.maximum(pts.max(0) - pts.min(0), 2e-3)
    lx, ly, lz = ext
    I = wp.mat33(mass * (ly * ly + lz * lz) / 12.0, 0.0, 0.0,
                 0.0, mass * (lx * lx + lz * lz) / 12.0, 0.0,
                 0.0, 0.0, mass * (lx * lx + ly * ly) / 12.0)
    return wp.vec3(*com), I


def _xf(pt):
    return wp.transform(wp.vec3(float(pt[0]), float(pt[1]), float(pt[2])),
                        wp.quat_identity())


@dataclass
class ACIndex:
    fus: int = 0
    flap: dict = field(default_factory=dict)     # 'L','R' -> (qi,di)
    surf: dict = field(default_factory=dict)     # 0..13 -> (qi,di)
    surf_name: dict = field(default_factory=dict)
    body_of_surface: dict = field(default_factory=dict)   # uvlm surface name -> body idx

    @property
    def n_actuators(self):
        return len(self.flap) + len(self.surf)


def build(ac=None, *, spring_ke=1.2, spring_kd=0.02, surf_ke=2.0, surf_kd=0.002,
          requires_grad=False):
    """Assemble the aircraft. Returns (model, ACIndex, uvlm_surfaces, msu)."""
    ac = ac or Aircraft()
    surfs = U.build_aircraft_surfaces(ac)
    byname = {s["name"]: s for s in surfs}
    dih_w = np.deg2rad(ac.wing.dihedral_deg)

    mb = newton.ModelBuilder(up_axis=newton.Axis.Z, gravity=-9.81)
    idx = ACIndex()
    joints = []

    fus = mb.add_link(mass=M_FUS, com=wp.vec3(0.0, 0.0, 0.0),
                      inertia=wp.mat33(3e-3, 0, 0, 0, 5e-3, 0, 0, 0, 5e-3),
                      label="fuselage")
    j_free = mb.add_joint_free(child=fus, label="freeflight")
    joints.append(j_free)
    idx.fus = fus
    # the fixed V-stab lattices ride on the fuselage
    for nm in ("vstab_L", "vstab_R"):
        idx.body_of_surface[nm] = fus

    surf_k = 0   # running control-surface index 0..13

    def add_servo_body(name, mass, hinge_pt, hinge_axis, parent_body, ke, kd):
        nonlocal surf_k
        s = byname[name]
        com, I = _com_inertia(s["V"], mass)
        b = mb.add_link(mass=mass, com=com, inertia=I, label=name)
        j = mb.add_joint_revolute(
            parent=parent_body, child=b,
            axis=(float(hinge_axis[0]), float(hinge_axis[1]), float(hinge_axis[2])),
            parent_xform=_xf(hinge_pt), child_xform=_xf(hinge_pt),
            target_ke=ke, target_kd=kd, target_pos=0.0, label=name)
        joints.append(j)
        idx.body_of_surface[name] = b
        idx.surf[surf_k] = j
        idx.surf_name[surf_k] = name
        surf_k += 1
        return b, j

    for sgn, side in ((+1, "L"), (-1, "R")):
        # wing box on the flap spring hinge (axis = longitudinal x; dihedral = rest)
        box = byname[f"box_{side}"]
        com, I = _com_inertia(box["V"], M_BOX)
        b_box = mb.add_link(mass=M_BOX, com=com, inertia=I, label=f"box_{side}")
        root = np.array([0.0, sgn * ac.wing.root_offset, 0.0])     # wing root hinge pt
        j_flap = mb.add_joint_revolute(
            parent=fus, child=b_box, axis=(1.0, 0.0, 0.0),
            parent_xform=_xf(root), child_xform=_xf(root),
            target_ke=spring_ke, target_kd=spring_kd,
            target_pos=float(sgn * 0.0), label=f"flap_{side}")
        joints.append(j_flap)
        idx.flap[side] = j_flap
        idx.body_of_surface[f"box_{side}"] = b_box
        # LE flaps: hinge = aft edge of the flap lattice (shared with box LE)
        for k in range(ac.wing.n_le):
            Vk = byname[f"le_{side}{k}"]["V"]
            e0, e1 = Vk[-1, 0], Vk[-1, -1]            # aft row endpoints
            add_servo_body(f"le_{side}{k}", M_FLAP, 0.5 * (e0 + e1),
                           _safe_axis(e1 - e0), b_box, surf_ke, surf_kd)
        # TE flaps: hinge = forward edge of the flap lattice (shared with box TE)
        for k in range(ac.wing.n_te):
            Vk = byname[f"te_{side}{k}"]["V"]
            e0, e1 = Vk[0, 0], Vk[0, -1]              # forward row endpoints
            add_servo_body(f"te_{side}{k}", M_FLAP, 0.5 * (e0 + e1),
                           _safe_axis(e1 - e0), b_box, surf_ke, surf_kd)
    # ruddervators: hinge = forward edge (shared with V-stab TE), parent = fuselage
    for sgn, side in ((+1, "L"), (-1, "R")):
        Vr = byname[f"rud_{side}"]["V"]
        e0, e1 = Vr[0, 0], Vr[0, -1]
        add_servo_body(f"rud_{side}", M_RUD, 0.5 * (e0 + e1),
                       _safe_axis(e1 - e0), fus, surf_ke, surf_kd)

    mb.add_articulation(joints, label="aircraft")
    model = mb.finalize(requires_grad=requires_grad)
    qs = model.joint_q_start.numpy(); ds = model.joint_qd_start.numpy()
    for side, j in list(idx.flap.items()):
        idx.flap[side] = (int(qs[j]), int(ds[j]))
    for k, j in list(idx.surf.items()):
        idx.surf[k] = (int(qs[j]), int(ds[j]))

    # bind the UVLM surfaces to their bodies
    for s in surfs:
        s["body"] = idx.body_of_surface[s["name"]]
    msu = U.MultiSurfaceUVLM(surfs)
    return model, idx, surfs, msu


def _safe_axis(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else np.array([0.0, 1.0, 0.0])


def verify():
    wp.init()
    model, idx, surfs, msu = build()
    nb = model.body_count
    nact = idx.n_actuators
    rev = int((model.joint_type.numpy() == int(newton.JointType.REVOLUTE)).sum())
    print(f"aircraft multibody on aircraft_geom: {nb} bodies, {model.joint_dof_count} DOF, "
          f"{rev} revolute, {nact} actuators ({len(idx.flap)} flap + {len(idx.surf)} surfaces)")
    print(f"  UVLM: {len(surfs)} surfaces / {msu.P} panels bound to bodies")
    # rest-configuration consistency: eval_fk should reproduce the aircraft-frame lattice
    s0 = model.state()
    q0 = np.zeros(model.joint_coord_count, dtype=np.float32); q0[3:7] = [0, 0, 0, 1]
    s0.joint_q.assign(q0); s0.joint_qd.assign(np.zeros(model.joint_dof_count, np.float32))
    newton.eval_fk(model, s0.joint_q, s0.joint_qd, s0)
    bq = s0.body_q.numpy().reshape(-1, 7)
    # binding check: UVLM at the rest body poses must reproduce the standalone
    # aircraft-frame geometry solve (the lattices ride the bodies correctly)
    poses = {b: (bq[b, 0:3].copy(), bq[b, 3:7].copy()) for b in range(nb)}
    twists = {b: (np.zeros(3), np.zeros(3)) for b in range(nb)}
    V0 = 10.0; a = np.deg2rad(5.0); Vinf = np.array([V0 * np.cos(a), 0, -V0 * np.sin(a)])
    L_bound = abs(msu.solve(poses, twists, Vinf)["lift"])
    standalone = U.MultiSurfaceUVLM(U.build_aircraft_surfaces(ac if False else Aircraft()))
    rest = {s["body"]: (np.zeros(3), np.array([0, 0, 0, 1.0])) for s in standalone.surf}
    rest_t = {s["body"]: (np.zeros(3), np.zeros(3)) for s in standalone.surf}
    L_ref = abs(standalone.solve(rest, rest_t, Vinf)["lift"])
    bind_ok = abs(L_bound - L_ref) / (L_ref + 1e-9) < 1e-6
    # free flight smoke test (zero control, gravity) — finite for 60 steps
    solver = SolverFeatherstone(model); dt = 1 / 800.
    s1 = model.state(); ctrl = model.control()
    fin = True
    for i in range(60):
        s0.clear_forces(); solver.step(s0, s1, ctrl, None, dt); s0, s1 = s1, s0
        if not np.all(np.isfinite(s0.joint_q.numpy())):
            fin = False; break
    ok = (nb == 17) and (nact == 16) and bind_ok and fin
    print(f"  UVLM-on-bodies lift={L_bound:.3f}N == geometry-frame {L_ref:.3f}N (bind_ok={bind_ok}); "
          f"free-flight finite={fin}; bodies={nb}==17 act={nact}==16")
    print(f"aircraft multibody {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)
