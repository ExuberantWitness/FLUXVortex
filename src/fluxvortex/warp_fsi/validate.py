"""Per-kernel validation of the Warp GPU port against the MATLAB ground truth.

Each GPU kernel is fed MATLAB's actual (deformed) intermediate state from a
fixture .mat and compared to the corresponding MATLAB output array — the same
layered checks as tests/compare_layered.py, but exercising the GPU kernels.
Also reports GPU-vs-CPU (porting correctness, expect machine precision).

NO toy/smoke data: geometry, circulation, wake all come from the MATLAB run.

Usage:
  python -m fluxvortex.warp_fsi.validate --layer L1 [--fixture <path>] [--dtype float64] [--envs 4]
  python -m fluxvortex.warp_fsi.validate --layer all
"""
from __future__ import annotations
import argparse
import os
import numpy as np
from scipy.io import loadmat

_FIX_DIR = os.path.join(os.path.dirname(__file__), '..', '..', '..',
                        'FSI_by_FEM_and_UVLM', 'single_sheet', 'fixtures')
DEFAULT_FIXTURE = os.path.abspath(
    os.path.join(_FIX_DIR, 'fixture_step3_t0.1995.mat'))


def _load(path):
    return loadmat(path, squeeze_me=True, struct_as_record=False)


def _ml_vec3_grid(v, Nx, Ny):
    return np.asarray(v).reshape(Nx, Ny, 3)


def _report(name, gpu, ref, atol, label, rtol=None):
    """Pass if max|Δ| < atol OR max_rel < rtol. rtol defaults to a dtype-aware
    value so the same check holds in fp64 (tight) and fp32 (rounding-limited)."""
    from . import config as cfg
    if rtol is None:
        rtol = 1e-12 if cfg.dtype_name() == 'float64' else 1e-5
    gpu = np.asarray(gpu); ref = np.asarray(ref)
    if gpu.shape != ref.shape:
        print(f"  [FAIL] {name:22s} SHAPE gpu={gpu.shape} ref={ref.shape}")
        return False
    d = float(np.max(np.abs(gpu - ref)))
    rel = d / (float(np.max(np.abs(ref))) + 1e-30)
    ok = (d < atol) or (rel < rtol)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:22s} max|Δ|={d:.3e} rel={rel:.3e} "
          f"|ref|={float(np.max(np.abs(ref))):.3e}  ({label})")
    return ok


# ─── Layers ──────────────────────────────────────────────────────────────

def layer_L1_aic(m, B):
    """GPU AIC vs MATLAB A_mat (sign: GPU AIC = -A_mat) and vs CPU build_aic."""
    import warp as wp
    from . import config as cfg
    from .kernels_uvlm import build_aic_batched
    from ..standalone_uvlm import StandaloneUVLM

    Nx, Ny = int(m['Nx']), int(m['Ny'])
    P = Nx * Ny
    core = float(m['eps_v']) if 'eps_v' in m else 1e-9
    # core_radius used by MATLAB ring is r_eps.fine=1e-6 in the AIC; the fixture
    # A_mat was built with that. Use 1e-6 (Yamano fine core) to match.
    core = 1e-6

    colloc = np.asarray(m['rc_vec']).reshape(P, 3)
    normals = np.asarray(m['n_vec_i']).reshape(P, 3)
    r = [np.asarray(m[f'r_panel_vec_{k}']).reshape(P, 3) for k in (1, 2, 3, 4)]
    corners = np.stack(r, axis=1)  # (P,4,3), MATLAB [r1,r2,r3,r4] order

    A_ml = np.asarray(m['A_mat'])  # (P,P) MATLAB convention

    # CPU reference (StandaloneUVLM with injected MATLAB geometry)
    verts = np.zeros((Nx + 1, Ny + 1, 3))  # stub; corners injected directly
    uvlm = StandaloneUVLM(verts, np.array([1.0, 0, 0]), rho=1.0, core_radius=core)
    uvlm._corners = corners.reshape(Nx, Ny, 4, 3)
    uvlm._colloc = colloc.reshape(Nx, Ny, 3)
    uvlm._normals = normals.reshape(Nx, Ny, 3)
    AIC_cpu = uvlm.build_aic()  # = -A_ml (sign), ~2e-3 vs A_ml on diagonal

    # GPU
    NP = cfg.NP_DTYPE
    col = np.broadcast_to(colloc, (B, P, 3)).astype(NP).copy()
    nrm = np.broadcast_to(normals, (B, P, 3)).astype(NP).copy()
    crn = np.broadcast_to(corners, (B, P, 4, 3)).astype(NP).copy()
    col_wp = wp.array(col, dtype=cfg.VEC3, device=cfg.DEVICE)
    nrm_wp = wp.array(nrm, dtype=cfg.VEC3, device=cfg.DEVICE)
    crn_wp = wp.array(crn, dtype=cfg.VEC3, device=cfg.DEVICE)
    aic_wp = build_aic_batched(col_wp, nrm_wp, crn_wp, core)
    wp.synchronize()
    aic_gpu = aic_wp.numpy()  # (B,P,P), GPU convention = -A_ml

    ok = True
    # vs MATLAB (physics): -AIC_gpu ≈ A_ml. Diagonal self-induction differs ~1e-3.
    ok &= _report('AIC vs MATLAB', -aic_gpu[0], A_ml, atol=2e-3, label='-AIC_gpu vs A_mat')
    # vs CPU (porting): machine precision at fp64, fp32-rounding at fp32
    ok &= _report('AIC vs CPU', aic_gpu[0], AIC_cpu, atol=cfg.PORT_ATOL,
                  label='gpu vs cpu build_aic')
    # batch consistency
    if B > 1:
        dmax = float(np.max(np.abs(aic_gpu - aic_gpu[0:1])))
        print(f"  [{'PASS' if dmax<1e-30 else 'FAIL'}] AIC batch-consistent  "
              f"max|env-env0|={dmax:.3e}  (B={B})")
        ok &= dmax < 1e-30
    return ok


def layer_L4_induction(m, B):
    """GPU bound/wake induction at colloc vs MATLAB V_gamma / V_wake_plate.

    Inject MATLAB Gamma (bound) and wake (r_wake_*, Gamma_wake); GPU sums ring
    velocities (Python sign = -V_matlab) → compare -V_gpu to MATLAB arrays.
    """
    import warp as wp
    from . import config as cfg
    from .kernels_uvlm import induce_velocity_batched

    Nx, Ny = int(m['Nx']), int(m['Ny'])
    P = Nx * Ny
    core = 1e-6
    NP = cfg.NP_DTYPE

    colloc = np.asarray(m['rc_vec']).reshape(P, 3)
    # bound rings + bound circulation (MATLAB sign)
    rb = [np.asarray(m[f'r_panel_vec_{k}']).reshape(P, 3) for k in (1, 2, 3, 4)]
    bound_corners = np.stack(rb, axis=1)              # (P,4,3)
    Gamma_ml = np.asarray(m['Gamma']).ravel()         # (P,)

    col = np.broadcast_to(colloc, (B, P, 3)).astype(NP).copy()
    col_wp = wp.array(col, dtype=cfg.VEC3, device=cfg.DEVICE)

    def induce(src_corners, src_gamma, S):
        c = np.broadcast_to(src_corners, (B, S, 4, 3)).astype(NP).copy()
        g = np.broadcast_to(src_gamma, (B, S)).astype(NP).copy()
        c_wp = wp.array(c, dtype=cfg.VEC3, device=cfg.DEVICE)
        g_wp = wp.array(g, dtype=cfg.DTYPE, device=cfg.DEVICE)
        V = induce_velocity_batched(col_wp, c_wp, g_wp, core)
        wp.synchronize()
        return V.numpy()  # (B,P,3) Python sign = -V_matlab

    ok = True
    # bound induction -> MATLAB V_gamma
    V_b = induce(bound_corners, Gamma_ml, P)
    V_gamma_ml = np.asarray(m['V_gamma']).reshape(P, 3)
    ok &= _report('V_gamma vs MATLAB', -V_b[0], V_gamma_ml, atol=1e-5,
                  label='-V_bound_gpu vs V_gamma')

    # wake induction -> MATLAB V_wake_plate
    Gw = np.asarray(m['Gamma_wake']).ravel()
    if Gw.size > 0:
        S = Gw.size
        rw = [np.asarray(m[f'r_wake_{k}']).reshape(S, 3) for k in (1, 2, 3, 4)]
        wake_corners = np.stack(rw, axis=1)           # (S,4,3)
        V_w = induce(wake_corners, Gw, S)
        V_wake_ml = np.asarray(m['V_wake_plate']).reshape(P, 3)
        ok &= _report('V_wake vs MATLAB', -V_w[0], V_wake_ml, atol=1e-5,
                      label='-V_wake_gpu vs V_wake_plate')
    return ok


def layer_L5_dp_lift1(m, B):
    """GPU dp_lift1 vs MATLAB dp_lift1.

    Inject MATLAB Gamma (bound, MATLAB sign), corners, and V_surf1-V_in =
    V_wake_plate + V_gamma (MATLAB). rho=1 (nondim). Per-panel Bernoulli.
    """
    import warp as wp
    from . import config as cfg
    from .kernels_uvlm import compute_dp_lift1_batched

    Nx, Ny = int(m['Nx']), int(m['Ny'])
    P = Nx * Ny
    NP = cfg.NP_DTYPE

    # grids: MATLAB flat[i*Ny+j] -> (Nx,Ny[,3]) via C-order reshape
    Gamma = np.asarray(m['Gamma']).reshape(Nx, Ny)
    rb = [np.asarray(m[f'r_panel_vec_{k}']).reshape(Nx, Ny, 3) for k in (1, 2, 3, 4)]
    corners = np.stack(rb, axis=2)                          # (Nx,Ny,4,3)
    V_ext = (np.asarray(m['V_wake_plate']).reshape(Nx, Ny, 3)
             + np.asarray(m['V_gamma']).reshape(Nx, Ny, 3))  # V_surf1 - V_in
    dp_ml = np.asarray(m['dp_lift1']).reshape(Nx, Ny)

    g = np.broadcast_to(Gamma, (B, Nx, Ny)).astype(NP).copy()
    c = np.broadcast_to(corners, (B, Nx, Ny, 4, 3)).astype(NP).copy()
    ve = np.broadcast_to(V_ext, (B, Nx, Ny, 3)).astype(NP).copy()
    g_wp = wp.array(g, dtype=cfg.DTYPE, device=cfg.DEVICE)
    c_wp = wp.array(c, dtype=cfg.VEC3, device=cfg.DEVICE)
    ve_wp = wp.array(ve, dtype=cfg.VEC3, device=cfg.DEVICE)

    dp = compute_dp_lift1_batched(g_wp, c_wp, ve_wp, [1.0, 0.0, 0.0], 1.0)
    wp.synchronize()
    dp_gpu = dp.numpy()  # (B,Nx,Ny)

    ok = _report('dp_lift1 vs MATLAB', dp_gpu[0], dp_ml, atol=1e-6,
                 label='gpu vs dp_lift1')
    return ok


def layer_L6_mf2_vec1(m, B):
    """GPU Mf2_vec1 vs MATLAB. Two checks:
      (a) scalar Gamma_wake_dt_q1234_n vs MATLAB (machine precision)
      (b) full Mf2_vec1 = AIC⁻¹·(-scalar) vs MATLAB Mf2_vec1
    Uses MATLAB wake state (r_wake_*, dt_r_wake_*, Gamma_wake) and dt_rc_vec.
    """
    import warp as wp
    from . import config as cfg
    from .kernels_uvlm import compute_mf2_scalar_batched

    Nx, Ny = int(m['Nx']), int(m['Ny'])
    P = Nx * Ny
    NP = cfg.NP_DTYPE
    core_dt = 1e-9  # MATLAB eps_v for the time-derivative kernel

    colloc = np.asarray(m['rc_vec']).reshape(P, 3)
    normals = np.asarray(m['n_vec_i']).reshape(P, 3)
    dt_rc = np.asarray(m['dt_rc_vec']).reshape(P, 3)
    Gw = np.asarray(m['Gamma_wake']).ravel()
    Sw = Gw.size
    rw = [np.asarray(m[f'r_wake_{k}']).reshape(Sw, 3) for k in (1, 2, 3, 4)]
    dtw = [np.asarray(m[f'dt_r_wake_{k}']).reshape(Sw, 3) for k in (1, 2, 3, 4)]
    wcorn = np.stack(rw, axis=1)   # (Sw,4,3)
    wdt = np.stack(dtw, axis=1)    # (Sw,4,3)

    def bc(a, shp):
        return np.broadcast_to(a, (B,) + shp).astype(NP).copy()
    col_wp = wp.array(bc(colloc, (P, 3)), dtype=cfg.VEC3, device=cfg.DEVICE)
    nrm_wp = wp.array(bc(normals, (P, 3)), dtype=cfg.VEC3, device=cfg.DEVICE)
    dtc_wp = wp.array(bc(dt_rc, (P, 3)), dtype=cfg.VEC3, device=cfg.DEVICE)
    wc_wp = wp.array(bc(wcorn, (Sw, 4, 3)), dtype=cfg.VEC3, device=cfg.DEVICE)
    wd_wp = wp.array(bc(wdt, (Sw, 4, 3)), dtype=cfg.VEC3, device=cfg.DEVICE)
    wg_wp = wp.array(bc(Gw, (Sw,)), dtype=cfg.DTYPE, device=cfg.DEVICE)

    scal = compute_mf2_scalar_batched(col_wp, nrm_wp, dtc_wp, wc_wp, wd_wp,
                                      wg_wp, core_dt)
    wp.synchronize()
    scal_gpu = scal.numpy()[0]  # (P,) = Gamma_wake_dt_q1234_n

    ok = True
    # GPU scalar uses the Python -V sign convention = -(MATLAB Gamma_wake_dt_q1234_n).
    if 'Gamma_wake_dt_q1234_n' in m:
        scal_ml = np.asarray(m['Gamma_wake_dt_q1234_n']).ravel()
        ok &= _report('Gw_dt_q1234_n vs MATLAB', -scal_gpu, scal_ml, atol=1e-9,
                      label='-scal_gpu (Python sign)')

    # CPU path (matches MATLAB): Mf2 = solve(AIC_py, -scal) with AIC_py = -A_ml,
    #   => solve(-A_ml, -scal_gpu) = solve(A_ml, scal_gpu).
    A_ml = np.asarray(m['A_mat'])
    Mf2_gpu = np.linalg.solve(A_ml, scal_gpu)   # AIC solve (host; GPU in Phase 1c)
    Mf2_ml = np.asarray(m['Mf2_vec1']).ravel()
    ok &= _report('Mf2_vec1 vs MATLAB', Mf2_gpu, Mf2_ml, atol=1e-5, label='solve(A_ml, scal_gpu)')
    return ok


def layer_L2_gamma_solve(m, B):
    """GPU batched dense solve vs MATLAB Gamma.

    Isolates the linear solver from RHS construction (which has the known
    Kutta-timing subtlety): recover the exact RHS MATLAB solved with,
    V_normal = A_mat·Gamma, then GPU-solve A_mat·x = V_normal and compare to
    Gamma (must be machine precision — it's the same linear system).
    """
    import warp as wp
    from . import config as cfg
    from .batched_solver import batched_dense_solve

    A_ml = np.asarray(m['A_mat'])               # (P,P)
    Gamma = np.asarray(m['Gamma']).ravel()      # (P,)
    P = A_ml.shape[0]
    V_normal = A_ml @ Gamma                      # exact MATLAB RHS
    NP = cfg.NP_DTYPE

    A = np.broadcast_to(A_ml, (B, P, P)).astype(NP).copy()
    b = np.broadcast_to(V_normal, (B, P)).astype(NP).copy()
    A_wp = wp.array(A, dtype=cfg.DTYPE, device=cfg.DEVICE)
    b_wp = wp.array(b, dtype=cfg.DTYPE, device=cfg.DEVICE)
    x = batched_dense_solve(A_wp, b_wp)
    wp.synchronize()
    x_gpu = x.numpy()  # (B,P)

    ok = _report('gamma solve vs MATLAB', x_gpu[0], Gamma, atol=1e-9,
                 label='batched LU, A_ml·x=V_normal')
    if B > 1:
        dmax = float(np.max(np.abs(x_gpu - x_gpu[0:1])))
        print(f"  [{'PASS' if dmax<1e-30 else 'FAIL'}] gamma batch-consistent  "
              f"max|env-env0|={dmax:.3e}  (B={B})")
        ok &= dmax < 1e-30
    return ok


def layer_ANCF_K(m_unused, B):
    """GPU ANCF membrane tangent K_mem vs MATLAB dq_Qe_mem_global (K_at_qref.mat).

    Build Yamano shell, set q=q_ref (flat plate + unit slopes), GPU-assemble the
    per-element K_mem blocks, scatter to global, and compare to MATLAB (unit
    scale K*=ρf·V²·L=122.5 + node perm). Also vs CPU _tangent_K_mem (porting).
    """
    import os
    import warp as wp
    from . import config as cfg
    from .kernels_ancf import ANCFConstants, assemble_kmem_blocks, scatter_kmem_global

    here = os.path.dirname(__file__)
    tests_dir = os.path.abspath(os.path.join(here, '..', '..', '..', 'tests'))
    import sys
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from run_standalone_yamano import yamano_params, build_yamano_shell

    kpath = os.path.join(_FIX_DIR, 'K_at_qref.mat')
    K = _load(os.path.abspath(kpath))
    Nx, Ny = int(K['Nx']), int(K['Ny'])

    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=Nx, ny=Ny)
    ndof = shell.ndof

    # q_ref: flat plate with unit slopes (matches dump_K_at_qref.m / compare_K_at_qref)
    q_ref = np.zeros(ndof)
    for k in range(shell.nn):
        q_ref[9 * k + 0] = shell.nodes[k, 0]
        q_ref[9 * k + 1] = shell.nodes[k, 1]
        q_ref[9 * k + 3] = 1.0   # dx_r = [1,0,0]
        q_ref[9 * k + 7] = 1.0   # dy_r = [0,1,0]

    # CPU reference
    K_cpu = shell._tangent_K_mem(q_ref).toarray()

    # GPU
    NP = cfg.NP_DTYPE
    C = ANCFConstants(shell)
    q = np.broadcast_to(q_ref, (B, ndof)).astype(NP).copy()
    q_wp = wp.array(q, dtype=cfg.DTYPE, device=cfg.DEVICE)
    Kblk = assemble_kmem_blocks(q_wp, C)
    wp.synchronize()
    Kblk_np = Kblk.numpy()  # (B, ne, 36, 36)
    K_gpu = scatter_kmem_global(Kblk_np[0], C.edofs_np, ndof)

    # MATLAB membrane tangent (nondim) -> dimensional + node perm
    Kmem_ml = K['dq_Qe_mem_global']
    Kmem_ml = Kmem_ml.toarray() if hasattr(Kmem_ml, 'toarray') else np.asarray(Kmem_ml)
    scale_K = 1.225 * 10.0**2 * 1.0   # ρf·V²·L
    perm = _ml_to_py_dof_perm(Nx, Ny)
    Kmem_ml_py = (Kmem_ml * scale_K)[np.ix_(perm, perm)]

    ok = True
    ok &= _report('K_mem vs MATLAB', K_gpu, Kmem_ml_py, atol=1e-3,
                  label='gpu assembled vs dq_Qe_mem_global', rtol=1e-10)
    ok &= _report('K_mem vs CPU', K_gpu, K_cpu, atol=cfg.PORT_ATOL,
                  label='gpu vs cpu _tangent_K_mem', rtol=1e-9)
    if B > 1:
        dmax = float(np.max(np.abs(Kblk_np - Kblk_np[0:1])))
        print(f"  [{'PASS' if dmax<1e-30 else 'FAIL'}] K_mem batch-consistent  "
              f"max|env-env0|={dmax:.3e}  (B={B})")
        ok &= dmax < 1e-30
    return ok


def _ml_to_py_dof_perm(Nx, Ny):
    """perm[k_py] = k_ml DOF index. MATLAB i-outer/j-inner, Python j-outer/i-inner."""
    nn = (Nx + 1) * (Ny + 1)
    perm = np.empty(9 * nn, dtype=np.int64)
    for j in range(Ny + 1):
        for i in range(Nx + 1):
            k_p = j * (Nx + 1) + i
            k_m = i * (Ny + 1) + j
            for d in range(9):
                perm[9 * k_p + d] = 9 * k_m + d
    return perm


def layer_ANCF_F(m_unused, B):
    """GPU ANCF internal force (Q_mem+Q_bend) vs CPU _internal_forces_separated
    (the MATLAB-faithful element algorithm, bit-exact to MATLAB on K) at a
    DEFORMED state; plus the near-zero check at q_ref vs MATLAB Qe_global+Qk."""
    import os, sys
    import warp as wp
    from . import config as cfg
    from .kernels_ancf import ANCFConstants, assemble_internal_force

    here = os.path.dirname(__file__)
    tests_dir = os.path.abspath(os.path.join(here, '..', '..', '..', 'tests'))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from run_standalone_yamano import yamano_params, build_yamano_shell

    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    ndof = shell.ndof
    NP = cfg.NP_DTYPE
    C = ANCFConstants(shell)

    q_ref = np.zeros(ndof)
    for k in range(shell.nn):
        q_ref[9*k+0] = shell.nodes[k, 0]; q_ref[9*k+1] = shell.nodes[k, 1]
        q_ref[9*k+3] = 1.0; q_ref[9*k+7] = 1.0

    # deformed state: add a smooth z-bump + slope perturbation (free DOFs)
    rng = np.random.default_rng(3)
    q_def = q_ref.copy()
    for k in range(shell.nn):
        x = shell.nodes[k, 0]
        if 9*k not in shell._bc_dofs:  # not clamped
            q_def[9*k+2] += 0.05 * x * x            # rz
            q_def[9*k+5] += 0.1 * x                  # dx_rz
            q_def[9*k+8] += 0.02 * x                 # dy_rz

    ok = True
    # q_ref is the flat-plate equilibrium (force ~0): use an absolute near-zero
    # floor (both GPU and CPU are ~1e-10). deformed is the real porting check.
    for tag, q, atol in [('q_ref', q_ref, 1e-8), ('deformed', q_def, cfg.PORT_ATOL)]:
        Qm_cpu, Qb_cpu = shell._internal_forces_separated(q)
        Q_cpu = Qm_cpu + Qb_cpu
        qb = np.broadcast_to(q, (B, ndof)).astype(NP).copy()
        q_wp = wp.array(qb, dtype=cfg.DTYPE, device=cfg.DEVICE)
        F = assemble_internal_force(q_wp, C)
        wp.synchronize()
        Q_gpu = F.numpy()[0]
        ok &= _report(f'Q_int[{tag}] vs CPU', Q_gpu, Q_cpu, atol=atol,
                      label='gpu vs cpu _internal_forces_separated', rtol=1e-9)

    # q_ref absolute level vs MATLAB Qe_global+Qk_global (both ~0 at flat ref)
    kpath = os.path.join(_FIX_DIR, 'K_at_qref.mat')
    K = _load(os.path.abspath(kpath))
    Qe_ml = np.asarray(K['Qe_global'], dtype=float).ravel() + np.asarray(K['Qk_global'], dtype=float).ravel()
    q_wp = wp.array(np.broadcast_to(q_ref, (1, ndof)).astype(NP).copy(), dtype=cfg.DTYPE, device=cfg.DEVICE)
    Q_gpu_ref = assemble_internal_force(q_wp, C).numpy()[0]
    print(f"        |Q_int(q_ref)|_gpu={np.max(np.abs(Q_gpu_ref)):.3e}  "
          f"|Qe+Qk|_MATLAB={np.max(np.abs(Qe_ml)):.3e}  (both ~0 at flat ref)")
    return ok


def layer_STRUCT_CG(m_unused, B):
    """Matrix-free batched CG for the Newmark operator S = M + coef·K_mem on free
    DOFs, vs CPU scipy splu of the assembled S_ff. coef = α·c_damp·dt²/2.
    M and K_mem are the MATLAB-validated builders. b = M·(masked random)."""
    import os, sys
    import warp as wp
    from scipy.sparse.linalg import splu
    from . import config as cfg
    from .kernels_ancf import ANCFConstants, assemble_kmem_blocks
    from .batched_solver import structural_cg

    here = os.path.dirname(__file__)
    tests_dir = os.path.abspath(os.path.join(here, '..', '..', '..', 'tests'))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from run_standalone_yamano import yamano_params, build_yamano_shell

    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    ndof = shell.ndof
    NP = cfg.NP_DTYPE

    q_ref = np.zeros(ndof)
    for k in range(shell.nn):
        q_ref[9*k+0] = shell.nodes[k, 0]; q_ref[9*k+1] = shell.nodes[k, 1]
        q_ref[9*k+3] = 1.0; q_ref[9*k+7] = 1.0

    alpha_v, c_damp, dt = 0.5, 2.0, 2e-4
    coef = alpha_v * c_damp * dt * dt / 2.0

    # CPU reference: S_ff = (M + coef·K_mem)[free,free], splu solve
    free_idx = np.array(sorted(set(range(ndof)) - set(shell._bc_dofs)))
    M = shell.M.tocsc()
    K = shell._tangent_K_mem(q_ref).tocsc()
    S = (M + coef * K).tocsc()
    S_ff = S[np.ix_(free_idx, free_idx)].tocsc()
    lu = splu(S_ff)

    rng = np.random.default_rng(0)
    b_full = np.zeros(ndof); b_full[free_idx] = rng.standard_normal(len(free_idx))
    x_cpu_ff = lu.solve(b_full[free_idx])
    x_cpu = np.zeros(ndof); x_cpu[free_idx] = x_cpu_ff

    # GPU: build K_mem blocks, run matrix-free CG
    C = ANCFConstants(shell)
    qb = np.broadcast_to(q_ref, (B, ndof)).astype(NP).copy()
    q_wp = wp.array(qb, dtype=cfg.DTYPE, device=cfg.DEVICE)
    Kblk = assemble_kmem_blocks(q_wp, C)
    bb = np.broadcast_to(b_full, (B, ndof)).astype(NP).copy()
    b_wp = wp.array(bb, dtype=cfg.DTYPE, device=cfg.DEVICE)
    x_wp, iters = structural_cg(b_wp, C.Me, Kblk, C.edofs, C.free, coef, ndof,
                                max_iter=2000, tol=1e-12)
    wp.synchronize()
    x_gpu = x_wp.numpy()[0]

    ok = _report(f'CG solve vs CPU splu', x_gpu, x_cpu, atol=1e-8,
                 label=f'matrix-free CG ({iters} its)', rtol=1e-7)
    # residual check on GPU solution
    res = np.max(np.abs(S_ff @ x_gpu[free_idx] - b_full[free_idx]))
    print(f"        GPU residual |S·x-b|_max = {res:.3e}")
    if B > 1:
        xn = x_wp.numpy()
        dmax = float(np.max(np.abs(xn - xn[0:1])))
        rel = dmax / (float(np.max(np.abs(xn))) + 1e-30)
        # atomic-add float non-associativity gives rel ~machine-eps differences
        bc_ok = rel < 1e-12
        print(f"  [{'PASS' if bc_ok else 'FAIL'}] CG batch-consistent  "
              f"max|env-env0|={dmax:.3e} rel={rel:.3e}  (B={B}; atomic non-assoc)")
        ok &= bc_ok
    return ok


def _build_yamano_solver(nx=15, ny=10):
    import os, sys
    here = os.path.dirname(__file__)
    tests_dir = os.path.abspath(os.path.join(here, '..', '..', '..', 'tests'))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from run_standalone_yamano import yamano_params, build_yamano_shell
    from ..standalone_hybrid_solver import StandaloneHybridSolver
    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=nx, ny=ny)
    solver = StandaloneHybridSolver(
        shell, np.array([params['V_inf'], 0.0, 0.0]), rho_fluid=params['rho_fluid'],
        structural_dt=2e-4, uvlm_dt_ratio=34, integrator='implicit', relaxation=1.0,
        newton_tol=1e-4, max_newton=20, max_particles=5000, wake_truncation=5.5,
        core_radius=1e-6, coupling='strong')
    return solver, shell, params


def layer_TRAJ(m_unused, B):
    """GPU structural pulse trajectory (loop of block-reduced Newmark steps with
    state carry-over) vs an identical CPU loop (numerical_solver.step). Validates
    the time-loop wiring (pulse, multi-step carry) before adding aero coupling."""
    import os, sys
    import warp as wp
    from . import config as cfg
    from .kernels_ancf import ANCFConstants
    from .runner import gpu_structural_trajectory

    here = os.path.dirname(__file__)
    tests_dir = os.path.abspath(os.path.join(here, '..', '..', '..', 'tests'))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    src_dir = os.path.abspath(os.path.join(here, '..', '..'))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from run_standalone_yamano import yamano_params, build_yamano_shell
    from fluxvortex.modules.numerical_solver import NewmarkSolver

    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    ndof = shell.ndof
    NP = cfg.NP_DTYPE
    C = ANCFConstants(shell)
    free_idx = np.array(sorted(set(range(ndof)) - set(shell._bc_dofs)))
    V_inf, L = params['V_inf'], params['Length']
    dt = 2e-4
    n_steps = 20

    q0 = np.zeros(ndof)
    for k in range(shell.nn):
        q0[9*k] = shell.nodes[k, 0]; q0[9*k+1] = shell.nodes[k, 1]
        q0[9*k+3] = 1.0; q0[9*k+7] = 1.0
    dq0 = np.zeros(ndof)
    f_density = params['rho_fluid'] * V_inf**2 / params['thickness']
    pulse_shape = shell.distributed_load(np.array([0.0, 0.0, 0.5 * f_density]))

    def profile(t):  # Yamano half-sine in nondim t* = t·V/L
        ts = t * V_inf / L
        return 0.5 * np.sin(np.pi * ts / 0.2) if ts < 0.2 else 0.0

    tip_node = int(np.argmax(shell.nodes[:, 0] + shell.nodes[:, 1]))
    tip_dof = tip_node * 9 + 2

    # CPU reference loop
    solver = NewmarkSolver(alpha_v=0.5, c_damp=2.0)
    M_ff = shell.M[np.ix_(free_idx, free_idx)].tocsc()
    qc = q0.copy(); dqc = dq0.copy()
    tip_cpu = np.zeros(n_steps)
    for step in range(n_steps):
        t = (step + 1) * dt
        Fc = pulse_shape * profile(t)
        Kt_ff = shell._tangent_K_mem(qc)[np.ix_(free_idx, free_idx)].tocsc()
        qc, dqc = solver.step(M_ff=M_ff, Kt_ff=Kt_ff, q_n=qc, dq_n=dqc,
                              free_dofs=free_idx, dt=dt, F_constant=Fc,
                              F_velocity_callback=None,
                              Q_internal_callback=shell._internal_forces_separated)
        tip_cpu[step] = qc[tip_dof]

    # GPU loop
    q0_wp = wp.array(np.broadcast_to(q0, (B, ndof)).astype(NP).copy(), dtype=cfg.DTYPE, device=cfg.DEVICE)
    dq0_wp = wp.array(np.broadcast_to(dq0, (B, ndof)).astype(NP).copy(), dtype=cfg.DTYPE, device=cfg.DEVICE)
    qg, dqg, tip_gpu = gpu_structural_trajectory(C, q0_wp, dq0_wp, pulse_shape, profile,
                                                 dt, n_steps, cg_tol=1e-12, tip_dof=tip_dof)
    wp.synchronize()

    ok = _report('tip trajectory vs CPU', tip_gpu[:, 0], tip_cpu, atol=1e-10,
                 label=f'{n_steps}-step pulse loop', rtol=1e-9)
    ok &= _report('final q vs CPU', qg.numpy()[0], qc, atol=1e-9,
                  label='end-of-trajectory state', rtol=1e-9)
    print(f"        tip_gpu[-1]={tip_gpu[-1,0]:.6e}  tip_cpu[-1]={tip_cpu[-1]:.6e}")
    return ok


def layer_NEWMARK_AM(m_unused, B):
    """GPU Newmark step WITH added mass (M_eff = M − M_added) vs CPU
    numerical_solver.step. Validates the −M_added CSR term in the operator."""
    import warp as wp
    from . import config as cfg
    from .kernels_ancf import ANCFConstants, assemble_kmem_blocks, assemble_internal_force_sep
    from .kernels_coupling import CSR
    from .batched_solver import gpu_newmark_step
    import sys, os
    here = os.path.dirname(__file__)
    src_dir = os.path.abspath(os.path.join(here, '..', '..'))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from fluxvortex.modules.numerical_solver import NewmarkSolver

    solver, shell, params = _build_yamano_solver()
    ndof = shell.ndof
    NP = cfg.NP_DTYPE
    C = ANCFConstants(shell)
    free_idx = np.array(sorted(set(range(ndof)) - set(shell._bc_dofs)))
    M_added = solver._M_added_full.tocsc()       # constant added-mass (M_eff = M − M_added)

    q_n = np.zeros(ndof)
    for k in range(shell.nn):
        q_n[9*k] = shell.nodes[k, 0]; q_n[9*k+1] = shell.nodes[k, 1]
        q_n[9*k+3] = 1.0; q_n[9*k+7] = 1.0
        x = shell.nodes[k, 0]
        if 9*k not in shell._bc_dofs:
            q_n[9*k+2] += 0.02 * x * x
    rng = np.random.default_rng(7)
    dq_n = np.zeros(ndof); dq_n[free_idx] = 0.01 * rng.standard_normal(len(free_idx))
    F_const = np.zeros(ndof); F_const[free_idx] = rng.standard_normal(len(free_idx))
    alpha_v, c_damp, dt = 0.5, 2.0, 2e-4

    # CPU: M_ff = (M − M_added)[free,free]
    solverN = NewmarkSolver(alpha_v=alpha_v, c_damp=c_damp)
    M_eff = (shell.M - M_added).tocsc()
    M_ff = M_eff[np.ix_(free_idx, free_idx)].tocsc()
    Kt_ff = shell._tangent_K_mem(q_n)[np.ix_(free_idx, free_idx)].tocsc()
    q_new_cpu, dq_new_cpu = solverN.step(
        M_ff=M_ff, Kt_ff=Kt_ff, q_n=q_n.copy(), dq_n=dq_n.copy(),
        free_dofs=free_idx, dt=dt, F_constant=F_const,
        F_velocity_callback=None, Q_internal_callback=shell._internal_forces_separated)

    # GPU with madd
    def bcast(a):
        return wp.array(np.broadcast_to(a, (B, ndof)).astype(NP).copy(), dtype=cfg.DTYPE, device=cfg.DEVICE)
    q_wp = bcast(q_n); dq_wp = bcast(dq_n); Fc_wp = bcast(F_const)
    Kblk = assemble_kmem_blocks(q_wp, C)
    Qmem_n, Qbend_n = assemble_internal_force_sep(q_wp, C)
    Fvel0 = wp.zeros((B, ndof), dtype=cfg.DTYPE, device=cfg.DEVICE)
    madd = CSR(M_added)
    madd_diag = wp.array(np.asarray(M_added.diagonal()).astype(NP), dtype=cfg.DTYPE, device=cfg.DEVICE)

    def recompute_bend(q_p1):
        return assemble_internal_force_sep(q_p1, C)[1]

    q_new_wp, dq_new_wp = gpu_newmark_step(
        q_wp, dq_wp, Kblk, C.Me, C.edofs, C.free, ndof,
        Fc_wp, Qmem_n, Qbend_n, Fvel0, recompute_bend, None,
        alpha_v=alpha_v, c_damp=c_damp, dt=dt, cg_tol=1e-12,
        madd=madd, madd_diag=madd_diag)
    wp.synchronize()
    ok = _report('q_new(+AM) vs CPU', q_new_wp.numpy()[0], q_new_cpu, atol=1e-9,
                 label='Newmark with M_eff=M−M_added', rtol=1e-9)
    ok &= _report('dq_new(+AM) vs CPU', dq_new_wp.numpy()[0], dq_new_cpu, atol=1e-7,
                  label='Newmark with M_eff=M−M_added', rtol=1e-9)
    return ok


def layer_COUPLING(m_unused, B):
    """GPU aero<->structure coupling vs CPU:
      - load transfer (_P_load·dp_n) vs solver._load_transfer (CPU=MATLAB Qf_p_global)
      - struct velocity @ colloc vs solver._compute_structural_velocity_at_colloc
    """
    import warp as wp
    from . import config as cfg
    from .kernels_coupling import CSR, CouplingConstants

    solver, shell, params = _build_yamano_solver()
    nc, ns = solver._nx, solver._ny
    P = nc * ns
    ndof = shell.ndof
    NP = cfg.NP_DTYPE
    rng = np.random.default_rng(11)

    # ── load transfer ──
    panel_forces = rng.standard_normal((nc, ns, 3))
    F_cpu = solver._load_transfer(panel_forces)              # (ndof,)
    areas = solver.uvlm._areas
    dp_n = np.zeros((nc, ns, 3))
    for i in range(nc):
        for j in range(ns):
            if areas[i, j] > 0:
                dp_n[i, j] = panel_forces[i, j] / areas[i, j]
    dp_flat = dp_n.reshape(-1)                               # (3P,)
    Pload = CSR(solver._P_load)
    x = wp.array(np.broadcast_to(dp_flat, (B, 3 * P)).astype(NP).copy(),
                 dtype=cfg.DTYPE, device=cfg.DEVICE)
    F_gpu = Pload.matvec(x); wp.synchronize()
    ok = _report('load_transfer vs CPU', F_gpu.numpy()[0], F_cpu, atol=cfg.PORT_ATOL,
                 label='_P_load·dp_n (CPU=MATLAB Qf_p_global)', rtol=1e-10)

    # ── struct velocity @ colloc ──
    dq = rng.standard_normal(ndof)
    V_cpu = solver._compute_structural_velocity_at_colloc(dq)   # (nc,ns,3)
    CC = CouplingConstants(solver)
    dq_wp = wp.array(np.broadcast_to(dq, (B, ndof)).astype(NP).copy(),
                     dtype=cfg.DTYPE, device=cfg.DEVICE)
    V_gpu = CC.struct_velocity(dq_wp); wp.synchronize()
    V_gpu_np = V_gpu.numpy()[0].reshape(nc, ns, 3)
    ok &= _report('struct_vel vs CPU', V_gpu_np, V_cpu, atol=cfg.PORT_ATOL,
                  label='S·dq @ colloc', rtol=1e-10)
    return ok


def layer_NEWMARK(m_unused, B):
    """Full GPU Newmark step (block-reduced, 2-stage bending averaging) vs CPU
    modules/numerical_solver.NewmarkSolver.step, structural-only (F_vel=0).
    Wires K_mem assembly + force assembly + Jacobi-PCG into one step."""
    import os, sys
    import warp as wp
    from . import config as cfg
    from .kernels_ancf import (ANCFConstants, assemble_kmem_blocks,
                               assemble_internal_force_sep)
    from .batched_solver import gpu_newmark_step

    here = os.path.dirname(__file__)
    tests_dir = os.path.abspath(os.path.join(here, '..', '..', '..', 'tests'))
    if tests_dir not in sys.path:
        sys.path.insert(0, tests_dir)
    from run_standalone_yamano import yamano_params, build_yamano_shell
    src_dir = os.path.abspath(os.path.join(here, '..', '..'))
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)
    from fluxvortex.modules.numerical_solver import NewmarkSolver

    params = yamano_params()
    shell, _, _, _ = build_yamano_shell(params, nx=15, ny=10)
    ndof = shell.ndof
    NP = cfg.NP_DTYPE
    C = ANCFConstants(shell)
    free_idx = np.array(sorted(set(range(ndof)) - set(shell._bc_dofs)))

    # state: q_ref + small deformation; nonzero dq + F_const on free DOFs
    q_n = np.zeros(ndof)
    for k in range(shell.nn):
        q_n[9*k] = shell.nodes[k, 0]; q_n[9*k+1] = shell.nodes[k, 1]
        q_n[9*k+3] = 1.0; q_n[9*k+7] = 1.0
        x = shell.nodes[k, 0]
        if 9*k not in shell._bc_dofs:
            q_n[9*k+2] += 0.02 * x * x
    rng = np.random.default_rng(7)
    dq_n = np.zeros(ndof); dq_n[free_idx] = 0.01 * rng.standard_normal(len(free_idx))
    F_const = np.zeros(ndof); F_const[free_idx] = rng.standard_normal(len(free_idx))
    alpha_v, c_damp, dt = 0.5, 2.0, 2e-4

    # CPU reference
    solver = NewmarkSolver(alpha_v=alpha_v, c_damp=c_damp)
    M_ff = shell.M[np.ix_(free_idx, free_idx)].tocsc()
    Kt_ff = shell._tangent_K_mem(q_n)[np.ix_(free_idx, free_idx)].tocsc()
    q_new_cpu, dq_new_cpu = solver.step(
        M_ff=M_ff, Kt_ff=Kt_ff, q_n=q_n.copy(), dq_n=dq_n.copy(),
        free_dofs=free_idx, dt=dt, F_constant=F_const,
        F_velocity_callback=None, Q_internal_callback=shell._internal_forces_separated)

    # GPU
    def bcast(a):
        return wp.array(np.broadcast_to(a, (B, ndof)).astype(NP).copy(),
                        dtype=cfg.DTYPE, device=cfg.DEVICE)
    q_wp = bcast(q_n); dq_wp = bcast(dq_n); Fc_wp = bcast(F_const)
    Kblk = assemble_kmem_blocks(q_wp, C)
    Qmem_n, Qbend_n = assemble_internal_force_sep(q_wp, C)
    Fvel0 = wp.zeros((B, ndof), dtype=cfg.DTYPE, device=cfg.DEVICE)

    def recompute_bend(q_p1):
        return assemble_internal_force_sep(q_p1, C)[1]

    q_new_wp, dq_new_wp = gpu_newmark_step(
        q_wp, dq_wp, Kblk, C.Me, C.edofs, C.free, ndof,
        Fc_wp, Qmem_n, Qbend_n, Fvel0, recompute_bend, None,
        alpha_v=alpha_v, c_damp=c_damp, dt=dt, cg_tol=1e-12)
    wp.synchronize()
    q_new_gpu = q_new_wp.numpy()[0]
    dq_new_gpu = dq_new_wp.numpy()[0]

    ok = True
    ok &= _report('q_new vs CPU Newmark', q_new_gpu, q_new_cpu, atol=1e-9,
                  label='GPU step vs numerical_solver.step', rtol=1e-9)
    ok &= _report('dq_new vs CPU Newmark', dq_new_gpu, dq_new_cpu, atol=1e-7,
                  label='GPU step vs numerical_solver.step', rtol=1e-9)
    return ok


LAYERS = {'L1': layer_L1_aic, 'L2': layer_L2_gamma_solve, 'L4': layer_L4_induction,
          'L5': layer_L5_dp_lift1, 'L6': layer_L6_mf2_vec1, 'ANCF_K': layer_ANCF_K,
          'ANCF_F': layer_ANCF_F, 'STRUCT_CG': layer_STRUCT_CG, 'NEWMARK': layer_NEWMARK,
          'COUPLING': layer_COUPLING, 'TRAJ': layer_TRAJ, 'NEWMARK_AM': layer_NEWMARK_AM}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--layer', default='all')
    ap.add_argument('--fixture', default=DEFAULT_FIXTURE)
    ap.add_argument('--dtype', default='float64', choices=['float64', 'float32'])
    ap.add_argument('--envs', type=int, default=4)
    args = ap.parse_args()

    from . import config as cfg
    cfg.set_dtype(args.dtype)
    print(cfg.summary())
    print(f"fixture: {args.fixture}")
    m = _load(args.fixture)

    layers = LAYERS if args.layer == 'all' else {args.layer: LAYERS[args.layer]}
    results = {}
    for name, fn in layers.items():
        print(f"\n── {name} ──")
        results[name] = fn(m, args.envs)
    print("\n══ summary ══")
    for name, ok in results.items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    raise SystemExit(0 if all(results.values()) else 1)


if __name__ == '__main__':
    main()
