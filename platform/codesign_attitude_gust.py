"""Gradient-driven DYNAMIC gust-load-alleviation co-design (route A, piece 4) — the scientific payoff
of the differentiable strong-coupling (predictor-corrector) unsteady FSI adjoint.

Setting: a light/flexible (Pazy-class) clamped wing in the regime where PARTITIONED (lagged-wake)
coupling diverges by the fluid added-mass instability — so a STRONG (predictor-corrector) coupling is
mandatory, and a differentiable one is what makes gradient co-design in this regime possible. A
1-cosine vertical gust hits the wing; the wing's gust-induced deformation drives a mean-axis attitude
excursion that, by conservation of angular momentum, tilts the (gimbal-mounted, IMU-sensed) fuselage.

Objective (gust-load alleviation for payload/gimbal attitude stability):
    J = Σ_t ½[ (φ_pitch·u_t)² + (φ_roll·u_t)² ]          (u_t = q_t − q_ref, mean-axis angular excursion)
        + λ_ctrl · ½ Σ_t ‖ctrl_t‖²·dt                    (control effort / power, optional 2nd objective)
where φ_pitch/φ_roll are nominal-inertia-weighted lever functionals of the vertical deformation.

Co-design variable: spanwise stiffness (E) and mass (ρ) taper via a low-D spline (control points →
per-element log-scale). The gradient ∂J/∂θ flows through the validated strong-coupled FSI adjoint
(coupled_unsteady_pc_grad_gpu with the loss_fn callback + per-step gust), and Adam descends it.

This module: (1) build_wing — Pazy-class clamped cantilever; (2) stability_demo — loose (explicit)
diverges vs PC stable, the why-strong-coupling anchor; (3) optimize — gradient co-design; (4) figure.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_HERE = os.path.dirname(__file__)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import warp as wp                                            # noqa: E402
from fluxvortex.warp_fsi import config as cfg               # noqa: E402
from fluxvortex.warp_fsi.kernels_ancf import ANCFConstants  # noqa: E402
from diff_struct_design import _build_shell                 # noqa: E402
import diff_coupled_unsteady as dcu                          # noqa: E402
import diff_coupled_unsteady_gpu as dcg                      # noqa: E402

VINF = dcu.VINF


def build_wing(nx=6, ny=4, h=4.0e-4, rho=1500.0, E=2.0e6, nu=0.35):
    """Pazy-class light/flexible clamped cantilever: thin (h=0.4 mm), so structural mass per area
    ρ·h ≈ 0.6 kg/m² is comparable to the fluid added mass — the regime where lagged/partitioned
    coupling hits the added-mass instability and STRONG (PC) coupling is mandatory."""
    sh = _build_shell(nx=nx, ny=ny, L=0.4, W=0.3, h=h, rho=rho, E=E, nu=nu)
    sh.set_bc([i for i in range(nx + 1)])                    # clamp the span-root edge
    return sh


def _elem_span_y(sh, nx, ny):
    """Per-element spanwise (y) centroid in [0,1], from the reference node geometry."""
    q0 = sh.q
    ys = np.zeros(sh.ne)
    for e in range(sh.ne):
        dofs = sh._elem_dofs(e)
        yy = [q0[d] for d in dofs if d % 9 == 1]             # y-position DOFs of the element's nodes
        ys[e] = np.mean(yy)
    ys = (ys - ys.min()) / (ys.max() - ys.min() + 1e-30)
    return ys


def spanwise_basis(sh, nx, ny, nctrl=3):
    """RBF spanwise basis B (ne, nctrl): smooth taper from `nctrl` control points along the span."""
    ys = _elem_span_y(sh, nx, ny)
    centers = np.linspace(0.0, 1.0, nctrl)
    width = 1.0 / max(nctrl - 1, 1)
    B = np.exp(-0.5 * ((ys[:, None] - centers[None, :]) / width) ** 2)
    B = B / (B.sum(axis=1, keepdims=True) + 1e-30)           # partition-of-unity rows
    return B, ys


def theta_to_scales(thetaE, thetaR, B):
    """log-scale spline: E_scale_e = exp(B·θ_E), ρ_scale_e = exp(B·θ_R)."""
    return np.exp(B @ thetaE), np.exp(B @ thetaR)


def gust_1cos(N, dt, w0=3.0, t_lead=0.15, frac=0.6):
    """1-cosine vertical gust schedule (N,3) added to the freestream: w(t)=½w0(1−cos(2π(t−t0)/Tg))."""
    g = np.zeros((N, 3))
    t = np.arange(N) * dt
    T = N * dt
    t0 = t_lead * T
    Tg = frac * T
    for i in range(N):
        if t0 <= t[i] <= t0 + Tg:
            g[i, 2] = 0.5 * w0 * (1.0 - np.cos(2.0 * np.pi * (t[i] - t0) / Tg))
    return g


def attitude_weights(sh):
    """Nominal-inertia-weighted lever functionals of the vertical (z) deformation:
    φ_pitch[node z-DOF] = m_node·x0   (pitch ∝ chordwise-arm × vertical disp)
    φ_roll [node z-DOF] = −m_node·y0  (roll  ∝ spanwise-arm  × vertical disp).
    These proxy the mean-axis body attitude excursion a gimbal/IMU on the fuselage would sense."""
    q0 = sh.q; ndof = sh.ndof; M0 = sh.M
    phi_p = np.zeros(ndof); phi_r = np.zeros(ndof)
    for n in range(ndof // 9):
        x0 = q0[9 * n + 0]; y0 = q0[9 * n + 1]; m_n = M0[9 * n + 2, 9 * n + 2]
        phi_p[9 * n + 2] = m_n * x0
        phi_r[9 * n + 2] = -m_n * y0
    s = np.sqrt(phi_p @ phi_p + phi_r @ phi_r) + 1e-30        # normalise so J is O(1)-scaled
    return phi_p / s, phi_r / s


def make_loss_fn(phi_p, phi_r, q_ref, pos=None, k=0.0, lam=0.0, dt=1.0):
    """Gust-load-alleviation loss
        J = Σ_t ½[(φ_p·u_t)²+(φ_r·u_t)²]              (attitude excursion, u_t=q_t−q_ref)
            + ½·λ·k²·Σ_t ‖dq_t⊙pos‖²·dt              (control effort / power of u_t=−k·dq_t⊙pos)
    Returns (L, dLdq, dLddq, dLda, dLdk_extra): the attitude term seeds q (dLdq), the effort term
    seeds the velocity (dLddq) and contributes an EXPLICIT ∂L/∂k (the closed-loop chain ∂L/∂k via the
    trajectory is handled by the adjoint's dL_dk)."""
    def loss_fn(q_outs, dq_outs, a_stars, q0_, dq0_):
        Nn, ndof = q_outs.shape
        dLdq = np.zeros((Nn + 1, ndof)); dLddq = np.zeros((Nn + 1, ndof)); L = 0.0; dLdk = 0.0
        for t in range(Nn):
            u = q_outs[t] - q_ref
            tp = float(phi_p @ u); tr = float(phi_r @ u)
            L += 0.5 * (tp * tp + tr * tr)
            dLdq[t + 1] = tp * phi_p + tr * phi_r
            if pos is not None and lam > 0.0:
                up = dq_outs[t] * pos
                L += 0.5 * lam * k * k * float(up @ up) * dt
                dLddq[t + 1] = lam * k * k * up * dt          # pos²=pos (binary mask)
                dLdk += lam * k * float(up @ up) * dt          # explicit ∂(½λk²‖·‖²dt)/∂k
        return L, dLdq, dLddq if (pos is not None and lam > 0.0) else None, None, dLdk
    return loss_fn


def stability_demo(nx=6, ny=4, N=30, dt=2e-4, seed=0):
    """Why strong coupling is mandatory here: the partitioned/explicit (loose) coupling diverges by
    the fluid added-mass instability, while the predictor-corrector (strong) coupling stays bounded —
    on the SAME Pazy-class wing, gust impulse, and rollout. Reports peak tip deflection (×span). Uses
    the GPU forwards (the production path); the instability is intrinsic to the partitioned scheme."""
    wp.init()
    sh = build_wing(nx, ny); C = ANCFConstants(sh, device=cfg.DEVICE)
    ndof = sh.ndof; ne = sh.ne
    Es = np.ones(ne); Rs = np.ones(ne); sh.set_distribution(E_scale=Es, rho_scale=Rs)
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy()
    dq0 = np.zeros(ndof)
    for n in range(ndof // 9):                               # seed with a uniform vertical gust impulse
        if (9 * n + 2) in free: dq0[9 * n + 2] = 1.0
    P, dist = dcu._index_maps(sh, nx, ny)
    span = 0.3
    # loose / explicit (GPU symplectic, M⁻¹) — the partitioned coupling
    q_loose, _ = dcg.coupled_unsteady_forward_gpu(sh, C, P, dist, q0, dq0, N, dt, Es, Rs, nx, ny,
                                                  use_wake=True, cg_tol=1e-8)
    # strong / PC (GPU, the validated strong solver)
    q_pc = dcg.coupled_unsteady_forward_pc_gpu(sh, C, P, dist, q0, dq0, N, dt, Es, Rs, nx, ny,
                                               use_wake=True, pc_it=30, pc_tol=1e-9, cg_tol=1e-8)
    zdofs = [9 * n + 2 for n in range(ndof // 9)]
    defl_loose = np.max(np.abs((q_loose - q0)[zdofs])) / span * 100.0
    defl_pc = np.max(np.abs((q_pc - q0)[zdofs])) / span * 100.0
    fin_loose = np.all(np.isfinite(q_loose)); fin_pc = np.all(np.isfinite(q_pc))
    print(f"Pazy-class wing ({ne} elems, {N} steps, dt={dt:g}, with wake):")
    print(f"  LOOSE (partitioned/explicit) peak deflection: "
          f"{'DIVERGED (' + format(defl_loose, '.0f') + '% span)' if (not fin_loose or defl_loose > 100) else format(defl_loose, '.1f') + '% span'}")
    print(f"  STRONG (predictor-corrector) peak deflection: "
          f"{'DIVERGED' if not fin_pc else format(defl_pc, '.1f') + '% span'}")
    ok = (not fin_loose or defl_loose > 100) and fin_pc and defl_pc < 100
    print(f"  -> {'PASS' if ok else 'NOTE'}: strong coupling is {'MANDATORY' if ok else 'compared'} here "
          f"(added-mass instability); the differentiable strong-coupled adjoint is what enables gradient co-design")
    return ok


def optimize(nx=6, ny=4, N=24, dt=2e-4, nctrl=3, iters=40, lr=0.08, w0=3.0,
             use_wake=True, pc_it=25, pc_tol=1e-9, cg_tol=1e-7, seed=0, verbose=True):
    """Gradient-driven dynamic gust-load-alleviation co-design: Adam on the spanwise (E,ρ) taper to
    minimise the mean-axis attitude excursion J under a 1-cosine gust, gradients through the
    differentiable strong-coupled (PC) FSI adjoint. Returns the history + optimal design."""
    wp.init()
    sh = build_wing(nx, ny); C = ANCFConstants(sh, device=cfg.DEVICE)
    ndof = sh.ndof; ne = sh.ne
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); dq0 = np.zeros(ndof)
    P, dist = dcu._index_maps(sh, nx, ny)
    B, ys = spanwise_basis(sh, nx, ny, nctrl)
    phi_p, phi_r = attitude_weights(sh)
    loss_fn = make_loss_fn(phi_p, phi_r, q0)
    gust = gust_1cos(N, dt, w0=w0)
    w_unused = np.zeros(ndof)

    volw = np.ones(ne) / ne                                  # regular grid → equal element budget weight

    def _renorm(theta):                                       # hold the (geom-mean) budget at 1 → redistribution
        return theta - float(np.log(np.mean(np.exp(B @ theta))))

    def eval_grad(thetaE, thetaR):
        Es, Rs = theta_to_scales(thetaE, thetaR, B)
        sh.set_distribution(E_scale=Es, rho_scale=Rs)
        L, gE, gR, _, _ = dcg.coupled_unsteady_pc_grad_gpu(
            sh, C, P, dist, q0, dq0, N, dt, w_unused, Es, Rs, nx, ny, use_wake=use_wake,
            cg_tol=cg_tol, pc_it=pc_it, pc_tol=pc_tol, gust=gust, loss_fn=loss_fn)
        gthE = B.T @ (gE * Es)                                # ∂J/∂θ_E (chain through E_e=exp(Bθ))
        gthR = B.T @ (gR * Rs)
        dE = B.T @ (volw * Es); dR = B.T @ (volw * Rs)        # budget-change directions (total stiffness / mass)
        gthE = gthE - (gthE @ dE) / (dE @ dE + 1e-30) * dE    # project out → FIXED-budget REDISTRIBUTION
        gthR = gthR - (gthR @ dR) / (dR @ dR + 1e-30) * dR
        return L, gthE, gthR

    thetaE = np.zeros(nctrl); thetaR = np.zeros(nctrl)        # start uniform (E_scale=ρ_scale=1)
    mE = np.zeros(nctrl); vE = np.zeros(nctrl); mR = np.zeros(nctrl); vR = np.zeros(nctrl)
    b1, b2, eps = 0.9, 0.999, 1e-8
    LO_E, HI_E, LO_R, HI_R = -1.5, 1.5, -0.8, 0.8            # box on the log-scales (manufacturable)
    hist = []
    L0 = None
    for it in range(1, iters + 1):
        L, gE, gR = eval_grad(thetaE, thetaR)
        if L0 is None: L0 = L
        hist.append((it, L, thetaE.copy(), thetaR.copy()))
        if verbose:
            print(f"  it {it:3d}  J={L:.4e}  (J/J0={L / L0:.3f})  θE={np.round(thetaE, 3)}  θR={np.round(thetaR, 3)}",
                  flush=True)
        for (th, g, m, v, lo, hi) in [(thetaE, gE, mE, vE, LO_E, HI_E), (thetaR, gR, mR, vR, LO_R, HI_R)]:
            m[:] = b1 * m + (1 - b1) * g
            v[:] = b2 * v + (1 - b2) * g * g
            mh = m / (1 - b1 ** it); vh = v / (1 - b2 ** it)
            th -= lr * mh / (np.sqrt(vh) + eps)
            np.clip(th, lo, hi, out=th)
        thetaE[:] = _renorm(thetaE); thetaR[:] = _renorm(thetaR)  # exact fixed-budget redistribution
    L_final, _, _ = eval_grad(thetaE, thetaR)
    if verbose:
        print(f"  final J={L_final:.4e}  reduction {100 * (1 - L_final / L0):.1f}%  vs uniform baseline")
    return dict(hist=hist, thetaE=thetaE, thetaR=thetaR, B=B, ys=ys, L0=L0, Lf=L_final, ne=ne, nctrl=nctrl)


def optimize_joint(nx=6, ny=4, N=20, dt=2e-4, nctrl=4, iters=16, lr=0.12, lr_k=0.4, w0=3.0,
                   k_init=2.0, lam=3e-3, k_hi=12.0, use_wake=True, pc_it=22, pc_tol=1e-8,
                   cg_tol=1e-6, seed=0, verbose=True):
    """JOINT structure + control co-design: Adam on the spanwise (E,ρ) taper (fixed material+mass
    budget, redistribution) AND a closed-loop feedback gain k, minimising J = attitude excursion +
    λ·control effort under a 1-cosine gust — all gradients through the differentiable strong-coupled
    FSI adjoint (dL/dk = closed-loop chain + explicit effort ∂/∂k). Returns history + optimal design."""
    wp.init()
    sh = build_wing(nx, ny); C = ANCFConstants(sh, device=cfg.DEVICE); ndof = sh.ndof; ne = sh.ne
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); dq0 = np.zeros(ndof)
    P, dist = dcu._index_maps(sh, nx, ny)
    B, ys = spanwise_basis(sh, nx, ny, nctrl)
    phi_p, phi_r = attitude_weights(sh); pos = dcg._pos_mask(C)
    gust = gust_1cos(N, dt, w0=w0); volw = np.ones(ne) / ne

    def _renorm(th): return th - float(np.log(np.mean(np.exp(B @ th))))

    def eval_grad(thE, thR, k):
        Es, Rs = theta_to_scales(thE, thR, B); sh.set_distribution(E_scale=Es, rho_scale=Rs)
        lf = make_loss_fn(phi_p, phi_r, q0, pos=pos, k=k, lam=lam, dt=dt)
        L, gE, gR, _, gk = dcg.coupled_unsteady_pc_grad_gpu(
            sh, C, P, dist, q0, dq0, N, dt, np.zeros(ndof), Es, Rs, nx, ny, use_wake=use_wake,
            fb_gain=k, cg_tol=cg_tol, pc_it=pc_it, pc_tol=pc_tol, gust=gust, loss_fn=lf)
        gthE = B.T @ (gE * Es); gthR = B.T @ (gR * Rs)
        dE = B.T @ (volw * Es); dR = B.T @ (volw * Rs)
        gthE -= (gthE @ dE) / (dE @ dE + 1e-30) * dE; gthR -= (gthR @ dR) / (dR @ dR + 1e-30) * dR
        return L, gthE, gthR, gk

    thE = np.zeros(nctrl); thR = np.zeros(nctrl); k = float(k_init)
    mE = np.zeros(nctrl); vE = np.zeros(nctrl); mR = np.zeros(nctrl); vR = np.zeros(nctrl); mk = 0.0; vk = 0.0
    b1, b2, eps = 0.9, 0.999, 1e-8
    LO_E, HI_E, LO_R, HI_R = -1.5, 1.5, -0.8, 0.8
    hist = []; L0 = None
    for it in range(1, iters + 1):
        L, gE, gR, gk = eval_grad(thE, thR, k)
        if L0 is None: L0 = L
        hist.append((it, L, thE.copy(), thR.copy(), k))
        if verbose:
            print(f"  it {it:3d}  J={L:.4e} (J/J0={L / L0:.3f})  k={k:.3f}  θE={np.round(thE, 2)}", flush=True)
        for (th, g, m, v, lo, hi) in [(thE, gE, mE, vE, LO_E, HI_E), (thR, gR, mR, vR, LO_R, HI_R)]:
            m[:] = b1 * m + (1 - b1) * g; v[:] = b2 * v + (1 - b2) * g * g
            th -= lr * (m / (1 - b1 ** it)) / (np.sqrt(v / (1 - b2 ** it)) + eps); np.clip(th, lo, hi, out=th)
        thE[:] = _renorm(thE); thR[:] = _renorm(thR)
        mk = b1 * mk + (1 - b1) * gk; vk = b2 * vk + (1 - b2) * gk * gk
        k -= lr_k * (mk / (1 - b1 ** it)) / (np.sqrt(vk / (1 - b2 ** it)) + eps)
        k = float(np.clip(k, 0.0, k_hi))
    Lf, _, _, _ = eval_grad(thE, thR, k)
    if verbose:
        print(f"  final J={Lf:.4e}  reduction {100 * (1 - Lf / L0):.1f}%  k*={k:.3f}", flush=True)
    return dict(hist=hist, thetaE=thE, thetaR=thR, k=k, B=B, ys=ys, L0=L0, Lf=Lf, ne=ne, nctrl=nctrl, lam=lam)


def verify_joint_grad(nx=4, ny=3, N=6, dt=2e-4, k0=4.0, lam=5e-3, seed=0):
    """Self-FD validation of the JOINT structure+control gradient (attitude + control-effort loss,
    closed-loop gain k): ∂J/∂E and ∂J/∂k vs central-FD of the adjoint's own deterministic loss."""
    wp.init()
    sh = build_wing(nx, ny); C = ANCFConstants(sh, device=cfg.DEVICE); ndof = sh.ndof; ne = sh.ne
    rng = np.random.default_rng(seed)
    Es = np.exp(0.1 * rng.standard_normal(ne)); Rs = np.exp(0.1 * rng.standard_normal(ne))
    sh.set_distribution(E_scale=Es, rho_scale=Rs)
    free = np.array(sorted(set(range(ndof)) - set(sh._bc_dofs)))
    q0 = sh.q.copy(); dq0 = np.zeros(ndof)
    for n in range(ndof // 9):
        if (9 * n + 2) in free: dq0[9 * n + 2] = 2.0           # gust impulse → non-trivial design sensitivity
    P, dist = dcu._index_maps(sh, nx, ny)
    phi_p, phi_r = attitude_weights(sh); pos = dcg._pos_mask(C)
    gust = gust_1cos(N, dt, w0=4.0)

    def run(E_, R_, k_):
        sh.set_distribution(E_scale=E_, rho_scale=R_)
        lf = make_loss_fn(phi_p, phi_r, q0, pos=pos, k=k_, lam=lam, dt=dt)
        return dcg.coupled_unsteady_pc_grad_gpu(sh, C, P, dist, q0, dq0, N, dt, np.zeros(ndof), E_, R_,
                  nx, ny, use_wake=False, fb_gain=k_, pc_it=30, pc_tol=1e-11, cg_tol=1e-11, gust=gust, loss_fn=lf)
    L, gE, gR, _, gk = run(Es, Rs, k0)
    ek = 1e-5; Lp = run(Es, Rs, k0 + ek)[0]; Lm = run(Es, Rs, k0 - ek)[0]; fdk = (Lp - Lm) / (2 * ek)
    e = int(np.argmax(np.abs(gE)))                            # the max-sensitivity element (avoid 0/0 FD noise)
    ee = 1e-6; ep = Es.copy(); ep[e] += ee; em = Es.copy(); em[e] -= ee
    fdE = (run(ep, Rs, k0)[0] - run(em, Rs, k0)[0]) / (2 * ee)
    relk = abs(gk - fdk) / (abs(fdk) + 1e-30); relE = abs(gE[e] - fdE) / (abs(fdE) + 1e-30)
    ok = relk < 2e-3 and relE < 5e-2
    print(f"JOINT structure+control gradient (attitude + control-effort, k={k0}, λ={lam}, {ne} elems):")
    print(f"  ∂J/∂k (closed-loop chain + explicit) adj={gk:+.4e} fd={fdk:+.4e}  rel={relk:.2e}")
    print(f"  ∂J/∂E (design)                       adj={gE[e]:+.4e} fd={fdE:+.4e}  rel={relE:.2e}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: joint design+control gradient flows through the "
          f"differentiable STRONG-coupled FSI — structure & closed-loop controller co-optimise together")
    return ok


def figure(result, path=None):
    """Hero figure: (a) J/J0 convergence of the gradient co-design; (b) the discovered spanwise
    stiffness (E) and mass (ρ) taper that alleviates the gust-induced attitude excursion."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    hist = result["hist"]; B = result["B"]; ys = result["ys"]
    its = [h[0] for h in hist]; Js = [h[1] for h in hist]; J0 = result["L0"]
    yy = np.argsort(ys); ysS = ys[yy]
    Es, Rs = theta_to_scales(result["thetaE"], result["thetaR"], B)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    ax[0].plot(its, np.array(Js) / J0, "-o", color="#c0392b", ms=4)
    ax[0].set_xlabel("Adam iteration"); ax[0].set_ylabel("J / J0  (attitude excursion)")
    ax[0].set_title(f"Gradient co-design convergence ({100*(1-result['Lf']/J0):.0f}% reduction)")
    ax[0].grid(alpha=0.3)
    ax[1].plot(ysS, Es[yy], "-s", color="#2c3e50", label="E_scale (stiffness)")
    ax[1].plot(ysS, Rs[yy], "-^", color="#27ae60", label="ρ_scale (mass)")
    ax[1].set_xlabel("spanwise position  (root→tip)"); ax[1].set_ylabel("design scale")
    ax[1].set_title("Discovered stiffness / mass taper"); ax[1].legend(); ax[1].grid(alpha=0.3)
    fig.suptitle("Dynamic gust-load-alleviation co-design via the differentiable strong-coupled FSI adjoint",
                 fontsize=11)
    fig.tight_layout()
    if path is None:
        path = os.path.join(_HERE, "..", "docs", "codesign_attitude_gust.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"saved figure -> {os.path.abspath(path)}")
    return path


if __name__ == "__main__":
    import sys as _s
    if "--stab" in _s.argv:
        raise SystemExit(0 if stability_demo() else 1)
    if "--opt" in _s.argv:
        r = optimize()
        figure(r)
        raise SystemExit(0)
    print("usage: python codesign_attitude_gust.py [--stab | --opt]")
