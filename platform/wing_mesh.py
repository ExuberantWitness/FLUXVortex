"""Flat structural mesh for the RoboEagle half wing (P2 assembly v3, user-approved).

Decisions encoded here (2026-07-06, three review rounds):
  - ALL nodes on z=0: the membrane is BUILT FLAT (real wing is a flat-stretched
    skin; camber belongs to the separate AERO surface, see WingModel.aero_off).
  - Three STRAIGHT carbon rods, each extended to the planform edge arc:
      LE   : x=0 line, root -> tip corner (0, span)      [grid line i=0]
      main : x=X_MAIN const, root -> TE-arc crossing      [grid line i=3 + arc node]
      aux  : x=X_AUX  const, root -> TE-arc crossing      [grid line i=5 + arc node]
  - Chordwise node lines i=3 / i=5 are PINNED to the rod x wherever the rod is
    inside the local chord; outboard of a rod's TE crossing the line falls back
    to renormalized cosine fractions (membrane-only, may bend).
  - The two rod ends are EXTRA boundary nodes on the true TE arc; the two tip
    strips crossed by the rod tails are re-triangulated so that (a) the tail
    segments are membrane edges (conforming beam-membrane coupling) and (b) the
    local boundary follows the arc through the rod end (the straight discrete
    TE edge would cut the convex arc inside the rod tip).

Zero-fit: every coordinate follows from the measured planform (chord_at) and
the rod layout constants — nothing tuned.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq

import _v2_robogeom as rg


def _cosine_fractions(nc):
    return 0.5 * (1.0 - np.cos(np.linspace(0.0, np.pi, nc + 1)))


def _xrow(c, xf, x_main, x_aux, margin):
    """Chordwise node positions at local chord c: pin i=3/i=5 to the straight
    rods where inside, renormalized cosine fractions between the anchors."""
    a1 = x_main if x_main < margin * c else xf[3] * c
    a2 = x_aux if x_aux < margin * c else max(xf[5] * c, a1 + 0.05 * c)
    x = np.empty(len(xf))
    for i, f in enumerate(xf):
        if f <= xf[3]:
            x[i] = f / xf[3] * a1
        elif f <= xf[5]:
            x[i] = a1 + (f - xf[3]) / (xf[5] - xf[3]) * (a2 - a1)
        else:
            x[i] = a2 + (f - xf[5]) / (1.0 - xf[5]) * (c - a2)
    return x


def flat_wing_mesh(nc=8, ns=16, x_main=None, x_aux=None, half_span=None,
                   margin=0.97):
    """Build the flat structural mesh. Returns a dict:
      nodes (nn,3) z=0 | tris (nt,3) CCW(+z) | nid_grid (ns+1,nc+1) node ids of
      the structured grid | xf_grid (ns+1,nc+1) rest chord fractions x/c |
      chains {'le':[...], 'main':[...], 'aux':[...], 'ribs':[[...]x7]} |
      rib_js | y_main/y_aux rod-end spanwise positions (m).
    """
    assert nc == 8 and ns == 16, "tip closeout is laid out for the 9x17 grid"
    half_span = rg.HALF_SPAN if half_span is None else half_span
    xf = _cosine_fractions(nc)
    c_root = float(rg.chord_at(0.0))
    x_main = xf[3] * c_root if x_main is None else float(x_main)
    x_aux = xf[5] * c_root if x_aux is None else float(x_aux)

    # rod-end crossings with the TE arc: chord(y) = x_rod
    f_cross = lambda y, x0: float(rg.chord_at(y)) - x0
    y_main = brentq(f_cross, 0.5 * half_span, 0.9995 * half_span, args=(x_main,))
    y_aux = brentq(f_cross, 0.5 * half_span, 0.9995 * half_span, args=(x_aux,))

    yj = np.linspace(0.0, half_span, ns + 1)
    cj = rg.chord_at(yj)
    dy = yj[1] - yj[0]
    j_aux = int(np.floor(y_aux / dy))            # strip [j_aux, j_aux+1] holds aux tail
    j_main = int(np.floor(y_main / dy))          # strip [j_main, j_main+1] holds main tail
    assert j_aux < j_main <= ns - 1, (j_aux, j_main)
    assert x_main < margin * cj[j_main] and not x_main < margin * cj[j_main + 1]
    assert x_aux < margin * cj[j_aux] and not x_aux < margin * cj[j_aux + 1]

    nid = lambda i, j: j * (nc + 1) + i
    nn_grid = (nc + 1) * (ns + 1)
    nodes = np.zeros((nn_grid + 2, 3))
    for j in range(ns + 1):
        xr = _xrow(cj[j], xf, x_main, x_aux, margin)
        for i in range(nc + 1):
            nodes[nid(i, j)] = [xr[i], yj[j], 0.0]
    ID_MAIN, ID_AUX = nn_grid, nn_grid + 1       # rod-end nodes on the TE arc
    nodes[ID_MAIN] = [x_main, y_main, 0.0]
    nodes[ID_AUX] = [x_aux, y_aux, 0.0]

    # ── membrane triangulation ───────────────────────────────────────────────
    # base structured cells, EXCEPT the two closeout strips crossed by rod tails
    skip = {(i, j_aux) for i in range(5, nc)} | {(i, j_main) for i in range(3, nc)}
    tris = []
    for j in range(ns):
        for i in range(nc):
            if (i, j) in skip:
                continue
            p = nid(i, j)
            tris.append([p, p + 1, p + nc + 2])
            tris.append([p, p + nc + 2, p + nc + 1])

    def _fan_right(i0, j, idA):
        """Right of the tail: fan from the arc node over the inboard row edges."""
        for i in range(i0, nc):
            tris.append([nid(i, j), nid(i + 1, j), idA])

    def _fan_left(i0, j, idA):
        """Left of the tail: fan from the rod's last inboard node over the arc
        node and the outboard row (TE -> rod line, keeps CCW orientation)."""
        p0 = nid(i0, j)
        tris.append([p0, idA, nid(nc, j + 1)])
        for i in range(nc, i0, -1):
            tris.append([p0, nid(i, j + 1), nid(i - 1, j + 1)])

    _fan_right(5, j_aux, ID_AUX)                 # aux tail strip
    _fan_left(5, j_aux, ID_AUX)
    _fan_right(3, j_main, ID_MAIN)               # main tail strip
    _fan_left(3, j_main, ID_MAIN)
    tris = np.asarray(tris, dtype=np.int64)

    # ── beam chains ──────────────────────────────────────────────────────────
    rib_js = sorted({int(round((k + 1) / 8 * ns)) for k in range(7)} - {0})
    chains = dict(
        le=[nid(0, j) for j in range(ns + 1)],
        main=[nid(3, j) for j in range(j_main + 1)] + [ID_MAIN],
        aux=[nid(5, j) for j in range(j_aux + 1)] + [ID_AUX],
        ribs=[[nid(i, j) for i in range(nc + 1)] for j in rib_js],
        # TE hem: kite-fabric construction ALWAYS hems the raw trailing edge
        # (folded + stitched, tension-stiff / bending-soft). Runs along the
        # full membrane boundary polyline incl. the rod-end arc nodes.
        te=[nid(nc, j) for j in range(j_aux + 1)] + [ID_AUX]
           + [nid(nc, j) for j in range(j_aux + 1, j_main + 1)] + [ID_MAIN]
           + [nid(nc, j) for j in range(j_main + 1, ns + 1)],
    )

    # ── assertions (mesh gate; every one is a decision made above) ──────────
    # 1. rods are straight lines
    for name in ("le", "main", "aux"):
        p = nodes[chains[name]]
        d = p[-1] - p[0]
        d /= np.linalg.norm(d)
        dev = np.linalg.norm(np.cross(p - p[0], d), axis=1).max()
        assert dev < 1e-9, f"{name} rod not straight (dev {dev:.2e})"
    # 2. rod ends on the planform edge arc
    assert abs(float(rg.chord_at(y_main)) - x_main) < 1e-9
    assert abs(float(rg.chord_at(y_aux)) - x_aux) < 1e-9
    assert np.allclose(nodes[chains["le"][-1]], [0.0, half_span, 0.0])
    # 3. all triangles positively oriented (+z), none degenerate
    v1 = nodes[tris[:, 1]] - nodes[tris[:, 0]]
    v2 = nodes[tris[:, 2]] - nodes[tris[:, 0]]
    az = 0.5 * (v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    assert az.min() > 1e-9, f"degenerate/flipped tri (min area {az.min():.2e})"
    # 4. conforming: every beam edge is a membrane triangle edge
    eset = set()
    for t in tris:
        for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
            eset.add((min(a, b), max(a, b)))
    for name, ch in (("le", chains["le"]), ("main", chains["main"]),
                     ("aux", chains["aux"]), ("te", chains["te"]),
                     *(("rib", r) for r in chains["ribs"])):
        for a, b in zip(ch[:-1], ch[1:]):
            assert (min(a, b), max(a, b)) in eset, f"{name} edge {(a, b)} not conforming"
    # 5. rib-spar crossings share node ids by construction (grid indexing)
    for j in rib_js:
        assert nid(0, j) in chains["le"] and nid(3, j) in chains["main"] \
            and nid(5, j) in chains["aux"]

    nid_grid = np.array([[nid(i, j) for i in range(nc + 1)] for j in range(ns + 1)])
    xf_grid = nodes[nid_grid.ravel(), 0].reshape(ns + 1, nc + 1) / np.maximum(
        cj[:, None], 1e-6)
    return dict(nodes=nodes, tris=tris, nid_grid=nid_grid, xf_grid=xf_grid,
                chains=chains, rib_js=rib_js, y_main=y_main, y_aux=y_aux,
                x_main=x_main, x_aux=x_aux, id_main=ID_MAIN, id_aux=ID_AUX)


if __name__ == "__main__":
    m = flat_wing_mesh()
    print(f"nodes {len(m['nodes'])}  tris {len(m['tris'])}  "
          f"rods: main x={m['x_main']*1e3:.2f}mm -> y={m['y_main']*1e3:.1f}mm, "
          f"aux x={m['x_aux']*1e3:.2f}mm -> y={m['y_aux']*1e3:.1f}mm")
    nb = sum(len(c) - 1 for c in (m['chains']['le'], m['chains']['main'],
                                  m['chains']['aux'])) + sum(
        len(r) - 1 for r in m['chains']['ribs'])
    bn = set(m['chains']['le']) | set(m['chains']['main']) | set(m['chains']['aux'])
    for r in m['chains']['ribs']:
        bn |= set(r)
    print(f"beam elems {nb}  beam nodes {len(bn)}  "
          f"ndof {3*len(m['nodes'])+3*len(bn)}")
    print("mesh assertions all passed")
