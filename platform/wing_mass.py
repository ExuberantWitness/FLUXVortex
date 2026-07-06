"""Parametric structural mass model (P2, user-requested co-design channel).

Zero-fit by construction: mass = material density x true section x member
layout. One place computes every mass number the platform reports (WingModel
mass_report, assembly figures, future co-design mass/inertia objectives), so
a design-variable change (rod diameter/wall, rib depth, skin thickness, rod
layout) propagates everywhere consistently.

All functions are plain closed-form numpy — differentiable w.r.t. the section
parameters for the P4 adjoint (no branching on design variables).
"""
from __future__ import annotations

import numpy as np

# material densities, kg/m^3 (literature-anchored constants; see
# research_notes_wing_params.md: carbon rod/tube 1592, aviation plywood 700,
# Mylar film 1390)
MATERIALS = dict(carbon=1592.0, ply=700.0, mylar=1390.0)


def tube_mlin(D_out, wall, rho=MATERIALS["carbon"]):
    """Line mass (kg/m) of a circular tube; wall = D/2 gives a solid rod."""
    d_in = D_out - 2.0 * wall
    return rho * np.pi / 4.0 * (D_out ** 2 - d_in ** 2)


def rod_mlin(D, rho=MATERIALS["carbon"]):
    return tube_mlin(D, D / 2.0, rho)


def rect_mlin(w, d, rho=MATERIALS["ply"]):
    """Line mass (kg/m) of a rectangular strip (rib)."""
    return rho * w * d


def chain_mass(nodes, chain, mlin):
    """Lump a beam chain's mass onto its nodes (half of each element to each
    end). Returns (m_node contribution array over ALL nodes, total kg)."""
    m = np.zeros(len(nodes))
    total = 0.0
    for a, b in zip(chain[:-1], chain[1:]):
        mL = mlin * np.linalg.norm(nodes[b] - nodes[a])
        m[a] += mL / 2.0
        m[b] += mL / 2.0
        total += mL
    return m, total


def membrane_mass(nodes, tris, h, rho=MATERIALS["mylar"]):
    """Skin mass lumped by tributary area (1/3 of each triangle per vertex)."""
    m = np.zeros(len(nodes))
    v1 = nodes[tris[:, 1]] - nodes[tris[:, 0]]
    v2 = nodes[tris[:, 2]] - nodes[tris[:, 0]]
    A = 0.5 * np.linalg.norm(np.cross(v1, v2), axis=1)
    for t, a in zip(tris, A):
        m[t] += rho * h * a / 3.0
    return m, float(rho * h * A.sum())


def budget(nodes, tris, members, mem_h, mem_rho=MATERIALS["mylar"]):
    """Full wing mass budget.

    members: dict name -> (chain node-id list, mlin kg/m)
    Returns dict with per-member totals (kg), the membrane total, per-node
    lumped mass m_node (kg, len(nodes)), grand total, and the spanwise
    distribution rows (kg per spanwise station, from node y coordinates).
    """
    m_node = np.zeros(len(nodes))
    totals = {}
    for name, (chain, mlin) in members.items():
        mc, tot = chain_mass(nodes, chain, mlin)
        m_node += mc
        totals[name] = tot
    m_skin, tot_skin = membrane_mass(nodes, tris, mem_h, mem_rho)
    m_node += m_skin
    totals["membrane"] = tot_skin
    return dict(totals=totals, m_node=m_node, total=float(m_node.sum()))


if __name__ == "__main__":
    import wing_mesh
    mesh = wing_mesh.flat_wing_mesh()
    ch = mesh["chains"]
    members = {"le_spar": (ch["le"], rod_mlin(8e-3)),
               "main_spar": (ch["main"], tube_mlin(10e-3, 1e-3)),
               "aux_spar": (ch["aux"], tube_mlin(6e-3, 1e-3))}
    for k, r in enumerate(ch["ribs"]):
        members[f"rib{k}"] = (r, rect_mlin(3e-3, 4.43e-3))
    b = budget(mesh["nodes"], mesh["tris"], members, mem_h=5e-5)
    ribs = sum(v for n, v in b["totals"].items() if n.startswith("rib"))
    print("mass budget (g):",
          {n: round(v * 1e3, 1) for n, v in b["totals"].items() if not n.startswith("rib")},
          f"ribs(sum) {ribs*1e3:.1f}  TOTAL {b['total']*1e3:.1f}")
