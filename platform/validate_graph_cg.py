"""THOROUGH validation of the graph-CG PC adjoint (use_graph_cg=True) across ALL working conditions,
before any upload. Matrix: {design ∂E/∂ρ, control gC, closed-loop dL/dk} × {no-wake, wake}, each
graph-CG vs the adaptive path (which is itself FD-validated by verify_pc_grad / _control / _policy_grad),
plus direct vs-FD spot checks. graph == adaptive AND adaptive == FD  ⟹  graph is FD-correct.
"""
import numpy as np
import warp as wp
from fluxvortex.warp_fsi import config as cfg
from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants
import diff_coupled_unsteady_gpu as g
import diff_coupled_unsteady as dcu
from diff_struct_design import _build_shell

NITER = 170


def _setup(nx, ny, seed, big=False):
    sh = _build_shell(nx=nx, ny=ny); sh.set_bc([i for i in range(nx + 1)])
    C = ANCFConstants(sh, device=cfg.DEVICE); ne = sh.ne; ndof = sh.ndof
    rng = np.random.default_rng(seed)
    s = 0.1 if not big else 0.1
    Es = np.exp(s * rng.standard_normal(ne)); Rs = np.exp(s * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    a = 1e-3 if big else 1e-4; b = 1e-2 if big else 1e-3
    q0 = sh.q.copy(); q0[free] += a * rng.standard_normal(len(free))
    dq0 = np.zeros(ndof); dq0[free] = b * rng.standard_normal(len(free))
    w = np.zeros(ndof); w[free] = rng.standard_normal(len(free))
    P, dist = dcu._index_maps(sh, nx, ny)
    return sh, C, ne, ndof, free, q0, dq0, w, P, dist, rng, Es, Rs


def main(nx=6, ny=4, N=8, dt=2e-4):
    wp.init()
    rows = []

    def gradrun(graph, **kw):
        sh, C, ne, ndof, free, q0, dq0, w, P, dist, rng, Es, Rs = st
        return g.coupled_unsteady_pc_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, w, Es, Rs, nx, ny,
                                              pc_it=30, pc_tol=1e-9,
                                              **({"use_graph_cg": True, "cg_niter": NITER} if graph else {"cg_tol": 1e-11}),
                                              **kw)

    # ---- DESIGN ∂E/∂ρ  (no-wake, wake) ----
    for uw in [False, True]:
        st = _setup(nx, ny, seed=0)
        _, gEa, gRa, _, _ = gradrun(False, use_wake=uw)
        _, gEg, gRg, _, _ = gradrun(True, use_wake=uw)
        rE = np.max(np.abs(gEg - gEa)) / (np.max(np.abs(gEa)) + 1e-30)
        rR = np.max(np.abs(gRg - gRa)) / (np.max(np.abs(gRa)) + 1e-30)
        rows.append((f"design {'wake ' if uw else 'no-wk'} ∂E", rE)); rows.append((f"design {'wake ' if uw else 'no-wk'} ∂ρ", rR))

    # ---- CONTROL gC  (no-wake, wake) ----
    for uw in [False, True]:
        st = _setup(nx, ny, seed=1)
        sh, C, ne, ndof, free, q0, dq0, w, P, dist, rng, Es, Rs = st
        u = np.zeros((N, ndof)); u[:, free] = 1e-2 * rng.standard_normal((N, len(free)))
        _, _, _, gCa, _ = gradrun(False, use_wake=uw, control=u)
        _, _, _, gCg, _ = gradrun(True, use_wake=uw, control=u)
        rC = np.max(np.abs(gCg - gCa)) / (np.max(np.abs(gCa)) + 1e-30)
        rows.append((f"control {'wake ' if uw else 'no-wk'} gC", rC))

    # ---- CLOSED-LOOP dL/dk  (no-wake, wake) ----
    for uw in [False, True]:
        st = _setup(nx, ny, seed=3, big=True)
        _, _, _, _, dka = gradrun(False, use_wake=uw, fb_gain=6.0)
        _, _, _, _, dkg = gradrun(True, use_wake=uw, fb_gain=6.0)
        rk = abs(dkg - dka) / (abs(dka) + 1e-30)
        rows.append((f"closed-loop {'wake ' if uw else 'no-wk'} dL/dk", rk))

    # ---- direct vs-FD spot checks (graph path), no-wake (fast) ----
    st = _setup(nx, ny, seed=0)
    sh, C, ne, ndof, free, q0, dq0, w, P, dist, rng, Es, Rs = st
    _, gEg, gRg, _, _ = gradrun(True, use_wake=False)
    els = [0, ne // 2, ne - 1]
    gE_fd, gR_fd = dcu.design_grad_fd_pc(sh, Es, Rs, q0, dq0, N, dt, free, w, nx, ny, elems=els, use_wake=False)
    rEfd = max(abs(gEg[e] - gE_fd[e]) for e in els) / (max(abs(gE_fd[e]) for e in els) + 1e-30)
    rRfd = max(abs(gRg[e] - gR_fd[e]) for e in els) / (max(abs(gR_fd[e]) for e in els) + 1e-30)
    rows.append(("design no-wk ∂E  vs FD (graph)", rEfd)); rows.append(("design no-wk ∂ρ  vs FD (graph)", rRfd))

    print("=" * 64)
    print("THOROUGH graph-CG validation (graph vs adaptive[=FD-validated], + vs FD):")
    allok = True
    for name, rel in rows:
        tol = 5e-2 if "vs FD" in name and "∂E" in name else (1e-2 if "vs FD" in name else 1e-6)
        ok = rel < tol; allok = allok and ok
        print(f"  {name:36s} rel={rel:.2e}  {'OK' if ok else 'FAIL (tol %.0e)'%tol}")
    print("=" * 64)
    print("RESULT:", "ALL WORKING CONDITIONS PASS" if allok else "SOME FAIL")
    return allok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
