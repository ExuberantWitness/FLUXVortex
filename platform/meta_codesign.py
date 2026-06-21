"""FLAGSHIP co-design (AST): meta-RL + 刚柔(stiffness) + 质量(mass) distributions, MAP-Elites archive.

Unifies the two existing tracks into the plan's two-layer optimiser:
  · DESIGN layer  — MAP-Elites (+ DQD) over a wing's spanwise STIFFNESS field AND MASS field
    (low-D splines; design_field.StiffnessField + MassField), the morphology design space.
  · CONTROL layer — a single RL² META-POLICY (Takens context) that ADAPTS per design few-shot, so the
    archive's every morphology is flown by the SAME amortised controller (no per-design retraining).
Each design is flown through a gust on the flight env (structure → dynamics via the validated reduced
aggregates) → quality (gust rejection × cruise efficiency); the archive illuminates the
(morphology × dynamics) behaviour space. Mechanisms grounded in the validated differentiable coupled FSI.

MetaFlightEnv2 adds the MASS field to MetaFlightEnv: combined_gust_factor (stiffness washout + tip-mass
inertia), control_authority (stiffness − roll-inertia sluggishness), total mass m (weight). A UNIFORM
mass field reduces EXACTLY to MetaFlightEnv (the stiffness-only env is the diagonal slice).
"""
from __future__ import annotations

import os

import numpy as np

import design_field as dfield
from meta_rl_train import MetaFlightEnv, K_CTRL
from flight_ppo_env import FlightPPOEnv

NOMINAL_M = 0.45                                              # base total mass (FlightPPOEnv default)


class MetaFlightEnv2(MetaFlightEnv):
    """Flight env whose per-episode design is (stiffness field, mass field). The mass field adds passive
    gust resistance (tip inertia), a weight cost, and control sluggishness — on top of the stiffness
    washout/efficiency trade-off. Uniform mass ⇒ identical to MetaFlightEnv (validated)."""

    def _apply_design2(self, sf, mf):
        self.sf = sf; self.mf = mf
        self.field = sf                                       # for any base logging
        self.s = float(sf.aggregates()["s_mean"])
        self.gust_factor = dfield.combined_gust_factor(sf, mf)   # stiffness washout + tip-mass inertia
        self.ctrl_factor = dfield.control_authority(sf, mf)      # stiffness ctrl − roll-inertia sluggishness
        self.m = NOMINAL_M * mf.m_total()                       # heavier wing = more weight (efficiency cost)

    def sample_design(self):
        sf = dfield.StiffnessField.sample(self.rng, K=K_CTRL)
        mf = dfield.MassField.sample(self.rng, K=K_CTRL)
        self._apply_design2(sf, mf)
        return (sf, mf)

    def reset(self, design=None):
        if design is None:
            self.sample_design()
        else:
            sf, mf = design
            sf = dfield.as_field(sf, K=K_CTRL)
            mf = mf if isinstance(mf, dfield.MassField) else dfield.MassField.uniform(
                1.0 if mf is None else float(mf), K=K_CTRL)
            self._apply_design2(sf, mf)
        return FlightPPOEnv.reset(self)


def efficiency(sf, mf):
    """Cruise L/D of a (stiffness, mass) design (root-stiffness shape retention − over-flex − weight)."""
    return dfield.combined_efficiency(sf, mf)


def train2(iters=120, steps=2048, epochs=8, mb=256, lr=3e-4, seed=0, log=print):
    """Meta-train the RL² policy over the (stiffness × mass) design distribution on MetaFlightEnv2 — the
    single amortised controller that flies any morphology in the co-design archive."""
    import torch
    import torch.nn as nn
    from meta_rl_train import RL2Policy, CtxEmbedder, collect, gae
    torch.manual_seed(seed)
    env = MetaFlightEnv2(seed=seed); net = RL2Policy(); emb = CtxEmbedder()
    opt = torch.optim.Adam(net.parameters(), lr=lr); hist = []
    for it in range(iters):
        E, A, LP, R, V, D, lv = collect(env, net, emb, steps)
        adv, ret = gae(R, V, D, lv)
        ep, cur = [], 0.0
        for t in range(len(R)):
            cur += float(R[t])
            if D[t]:
                ep.append(cur); cur = 0.0
        mean_ep = float(np.mean(ep)) if ep else float(R.sum())
        idx = np.arange(len(E))
        for _ in range(epochs):
            np.random.shuffle(idx)
            for s in range(0, len(E), mb):
                b = idx[s:s + mb]
                dist, val = net.dist(E[b]); lp = dist.log_prob(A[b]).sum(-1)
                ratio = (lp - LP[b]).exp()
                loss = (-torch.min(ratio * adv[b], torch.clamp(ratio, 0.8, 1.2) * adv[b]).mean()
                        + 0.5 * ((val - ret[b]) ** 2).mean() - 0.01 * dist.entropy().sum(-1).mean())
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5); opt.step()
        hist.append(mean_ep)
        if it % 10 == 0 or it == iters - 1:
            log(f"  iter {it:3d}: mean episode return={mean_ep:8.2f} (n_ep={len(ep)})")
    return net, hist


def eval_design(net, sf, mf, emb=None, seed=1):
    """Fly a (stiffness, mass) design through the gust with the adapting meta-policy → controlled gust
    EXCURSION (lower = better gust rejection). The policy adapts per design via the RL² context."""
    import torch
    from meta_rl_train import CtxEmbedder
    emb = emb or CtxEmbedder()
    env = MetaFlightEnv2(seed=seed)
    obs = env.reset(design=(sf, mf)); emb.reset(); gz = []
    for _ in range(env.horizon):
        with torch.no_grad():
            mu, _ = net(emb.push(obs).unsqueeze(0))
        obs, r, d, _ = env.step(mu[0].numpy()); emb.record(mu[0].numpy(), r)
        if env.gust["t0"] <= env.t < env.gust["t0"] + env.gust["dur"] + 0.5:
            gz.append(env.x[2])
        if d:
            break
    return (max(gz) - min(gz)) if gz else np.nan


# ───────────────────────── MAP-Elites archive (Step 3) ──────────────────────
# Genotype θ = [log stiffness ctrl (K), log mass ctrl (K)] (2K-D smooth splines). Each design is FLOWN
# by the adapted meta-policy through the gust → quality = gust rejection (−excursion); efficiency tracked.
# Behaviour space = (stiffness washout s_gust [翼面 morphology axis] × mass inertia m_gust [动力系统 /
# 质量分布 axis]) — the archive illuminates which (stiffness × mass) morphologies the SINGLE meta-policy
# flies best. Emitters: random init + Gaussian mutation (DQD's value is on the expensive coupled FSI, not
# this fast surrogate — known result; the novelty here is the amortised META-control across the archive).
GE = 12; GM = 8                                              # archive grid (s_gust × m_gust)
S_LO, S_HI = 0.45, 2.6                                       # s_gust range (stiffness washout)
M_LO, M_HI = 0.7, 1.55                                       # m_gust range (mass inertia)
KC = K_CTRL


def theta_to_fields(theta):
    sf = dfield.StiffnessField(np.exp(theta[:KC]), )
    mf = dfield.MassField(np.exp(theta[KC:2 * KC]))
    return sf, mf


def rand_theta(rng):
    thE = rng.uniform(np.log(0.35), np.log(2.4), size=KC)
    thM = rng.uniform(np.log(0.6), np.log(1.7), size=KC)
    return np.concatenate([thE, thM])


def descriptors(sf, mf):
    return float(np.clip(sf.s_gust(), S_LO, S_HI)), float(np.clip(mf.m_gust(), M_LO, M_HI))


class Archive:
    def __init__(self):
        self.q = {}; self.d = {}; self.meta = {}

    def cell(self, b1, b2):
        i = min(GE - 1, int((b1 - S_LO) / (S_HI - S_LO) * GE))
        j = min(GM - 1, int((b2 - M_LO) / (M_HI - M_LO) * GM))
        return (max(0, i), max(0, j))

    def add(self, theta, qual, b1, b2, gz, eff):
        c = self.cell(b1, b2)
        if c not in self.q or qual > self.q[c]:
            self.q[c] = qual; self.d[c] = theta.copy(); self.meta[c] = (gz, eff, b1, b2)

    def coverage(self):
        return len(self.q) / (GE * GM)

    def elites(self):
        return [(c, self.q[c], self.d[c], self.meta[c]) for c in self.q]


def run(net, n_init=60, n_iter=400, sigma=0.18, seed=0, log=print):
    """MAP-Elites: random init + Gaussian mutation over (stiffness × mass) splines; each design FLOWN by
    the adapted meta-policy → quality=−gust excursion (gust rejection), efficiency tracked."""
    from meta_rl_train import CtxEmbedder
    rng = np.random.default_rng(seed); arch = Archive(); emb = CtxEmbedder()

    def evaluate(theta):
        sf, mf = theta_to_fields(theta)
        gz = eval_design(net, sf, mf, emb)
        if not np.isfinite(gz):
            return None
        eff = efficiency(sf, mf); b1, b2 = descriptors(sf, mf)
        # OVER-FLEX penalty (FSI-grounded): the flight surrogate's load-washout over-credits structurally
        # over-flexible wings, which the high-fidelity coupled FSI penalises via large deflection. Penalise
        # tip compliance beyond the threshold (same C0 as cruise_efficiency) so archive elites stay FSI-feasible.
        overflex = max(0.0, sf.feather_compliance() - dfield._C0)
        return dict(q=-(gz + 2.0 * overflex), b1=b1, b2=b2, gz=gz, eff=eff)

    for _ in range(n_init):
        th = rand_theta(rng); r = evaluate(th)
        if r: arch.add(th, r["q"], r["b1"], r["b2"], r["gz"], r["eff"])
    for it in range(n_iter):
        elites = arch.elites()
        if not elites:
            th = rand_theta(rng)
        else:
            _, _, base, _ = elites[rng.integers(len(elites))]
            th = base + sigma * rng.standard_normal(base.shape)
            th[:KC] = np.clip(th[:KC], np.log(0.3), np.log(2.6)); th[KC:] = np.clip(th[KC:], np.log(0.5), np.log(1.8))
        r = evaluate(th)
        if r: arch.add(th, r["q"], r["b1"], r["b2"], r["gz"], r["eff"])
        if it % 80 == 0:
            log(f"  iter {it:4d}: coverage {arch.coverage()*100:.0f}% ({len(arch.q)}/{GE*GM})")
    log(f"  FINAL coverage {arch.coverage()*100:.0f}% ({len(arch.q)}/{GE*GM} niches)")
    return arch


def ground_fsi(thetas, nx=6, ny=4, log=print):
    """Ground archive designs on the VALIDATED differentiable coupled FSI: map each (stiffness × mass)
    spline to per-element (E,ρ) and run the coupled unsteady free-wake FSI under the SAME gust IC,
    measuring the passive structural gust-deflection energy. If the high-fidelity FSI deflection RANKS
    consistently with the fast-env gust excursion, the reduced flight surrogate is validated (the AST
    credibility check). Returns the per-design FSI deflection J."""
    import codesign_qd_unsteady as cq
    env = cq.Env(nx=nx, ny=ny, seed=0)
    sfrac = np.array([j / max(ny - 1, 1) for j in range(ny) for i in range(nx)])  # element span fraction (e=j*nx+i)
    out = []
    for th in thetas:
        sf, mf = theta_to_fields(th)
        E = sf.value(sfrac); R = mf.value(sfrac)            # per-element E/ρ from the splines
        try:
            qN, _ = cq.cg.coupled_unsteady_forward_gpu(env.sh, env.C, env.P, env.dist, env.q0, env.dq0,
                       cq.NSTEP, cq.DT, E, R, nx, ny, use_wake=True, fb_gain=0.0, cg_tol=cq.CG_TOL)
            if not np.all(np.isfinite(qN)) or np.max(np.abs(qN)) > 1e3:
                out.append(np.inf); continue
            d = (qN - env.qref) * env.fmask
            out.append(float(np.sum(d * d)))
        except Exception:
            out.append(np.inf)
    return out


def figure(arch, path=None):
    """Hero figure: (A) the meta-RL co-design archive over (stiffness washout × mass inertia), each cell
    coloured by the meta-policy's gust rejection; (B) the gust-rejection × cruise-efficiency frontier of
    all co-designed elites (each a distinct stiffness×mass morphology, all flown by ONE meta-policy)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    try:
        plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP"]; plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass
    grid = np.full((GM, GE), np.nan)
    gz_all, eff_all, mg_all = [], [], []
    for (c, q, th, meta) in arch.elites():
        i, j = c; grid[j, i] = -q                            # -q = gust excursion (lower=better)
        gz, eff, b1, b2 = meta
        gz_all.append(gz); eff_all.append(eff); mg_all.append(b2)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    im = ax[0].imshow(grid, origin="lower", aspect="auto", cmap="viridis_r",
                      extent=[S_LO, S_HI, M_LO, M_HI])
    ax[0].set_xlabel("stiffness washout  s_gust  (tip-compliance →)")
    ax[0].set_ylabel("mass inertia  m_gust  (tip-mass →)")
    ax[0].set_title(f"Meta-RL co-design archive ({len(arch.q)}/{GE*GM} niches, {arch.coverage()*100:.0f}%)")
    cb = fig.colorbar(im, ax=ax[0]); cb.set_label("gust excursion (lower = better rejection)")
    sc = ax[1].scatter(eff_all, gz_all, c=mg_all, cmap="plasma", s=28, edgecolor="k", linewidth=0.3)
    ax[1].set_xlabel("cruise efficiency  L/D"); ax[1].set_ylabel("gust excursion (lower = better)")
    ax[1].set_title("Gust-rejection × efficiency frontier (all elites, one meta-policy)")
    cb2 = fig.colorbar(sc, ax=ax[1]); cb2.set_label("mass inertia m_gust")
    ax[1].grid(alpha=0.3)
    fig.suptitle("Meta-RL-amortised structure (stiffness × mass) co-design via MAP-Elites", fontsize=12)
    fig.tight_layout()
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "docs", "meta_codesign_archive.png")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    print(f"saved figure -> {os.path.abspath(path)}")
    return path


def verify_env(seed=3):
    """MetaFlightEnv2 with UNIFORM mass reproduces MetaFlightEnv EXACTLY (same rollout) — backward-
    compatible; and a tip-mass design measurably changes the dynamics (gust factor / weight)."""
    import warp as wp; wp.init()
    rng = np.random.default_rng(seed)
    sf = dfield.StiffnessField.from_root_tip(1.8, 0.5, K=K_CTRL)
    # uniform-mass env vs the original MetaFlightEnv on the SAME stiffness design + same actions
    e1 = MetaFlightEnv(seed=seed); o1 = e1.reset(design=sf)
    e2 = MetaFlightEnv2(seed=seed); o2 = e2.reset(design=(sf, dfield.MassField.uniform(1.0, K=K_CTRL)))
    rng2 = np.random.default_rng(99); maxd = abs(np.asarray(o1) - np.asarray(o2)).max()
    for _ in range(120):
        a = rng2.uniform(-1, 1, size=4)
        s1, r1, d1, _ = e1.step(a); s2, r2, d2, _ = e2.step(a)
        maxd = max(maxd, abs(np.asarray(s1) - np.asarray(s2)).max(), abs(r1 - r2))
        if d1 or d2:
            break
    # tip-mass design changes dynamics
    e3 = MetaFlightEnv2(seed=seed); e3.reset(design=(sf, dfield.MassField.from_root_tip(0.5, 1.7, K=K_CTRL)))
    gf_u = e2.gust_factor; gf_t = e3.gust_factor; m_u = e2.m; m_t = e3.m
    ok = maxd < 1e-9 and gf_t < gf_u                          # uniform identical; tip-mass lowers gust factor
    print(f"MetaFlightEnv2 validation:")
    print(f"  uniform-mass vs MetaFlightEnv (same design+actions): max|Δstate,Δreward| = {maxd:.1e}")
    print(f"  tip-mass dynamics: gust_factor {gf_u:.3f} (uniform) -> {gf_t:.3f} (tip-mass);  m {m_u:.3f} -> {m_t:.3f}")
    print(f"  -> {'PASS' if ok else 'FAIL'}: mass env reduces exactly to the stiffness env (uniform) and "
          f"tip-mass adds passive gust resistance — the mass design axis is wired into the flight dynamics")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if verify_env() else 1)
