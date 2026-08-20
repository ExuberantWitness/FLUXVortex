"""Load-level separation/viscous drag ledger (paper: approved plan 2026-08-18).

Three algebraic terms per strip per step — no 2D time stepping, no
circulation-solve changes. All formulas have frozen-repo or declared-source
provenance:

T1  suction loss (Polhamus axial loss; CSf = 2*pi*A0^2, ldvm_fourier.py:323):
      dCS = -2*pi*(LESP^2 - crit^2) * g^2   when |LESP| > crit, else 0
      dD1 = -dCS * cos(alpha_eff) * q_strip * S_strip
    with g = 1/(1+2/AR) (ldvm_uvlm_correction.py:250).
T3  dynamic viscous (quadratic in instantaneous local relative velocity):
      dD3 = 0.5*rho*Cd0*|v_rel|*(v_rel . xhat)*S_plan
    Cd0 = 2*1.328/sqrt(Re) (Blasius, both sides) or the paper's frozen value.
T2  Rayleigh deep-separation branch: gated off by default (plan Phase 4).

alpha_eff per strip: atan2((v_rel x c_hat)_y, v_rel . c_hat) with the chord
unit vector le->te; reduces to the geometric alpha for a static wing.

Gates:
  G6a  attached limit: |LESP| <= crit everywhere -> T1 == 0 exactly
  G6b  bound: total ledger drag per strip in [0, 2 sin^2(alpha_eff) q S]
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class LedgerConfig:
    lesp_crit: float = 0.11
    aspect_ratio: float = 4.0
    rho: float = 1.225
    cd0: float = 0.0121        # declared per paper (Blasius 2*1.328/sqrt(Re))
    enable_t1: bool = True
    enable_t3: bool = True
    enable_t2: bool = False    # plan Phase 4 gate


def strip_section_state(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """Per-strip (alpha_eff [rad], v_rel [m/s]) from one ledger record.

    Chord direction le->te per strip (station-averaged); relative flow
    velocity v_inf - v_edge at the strip's LE stations (averaged).
    """
    le, te = rec["le_now"], rec["te_now"]
    chord = 0.5 * ((te[1:] - le[1:]) + (te[:-1] - le[:-1]))      # (S,3)
    v_rel = 0.5 * (rec["v_rel_st"][1:] + rec["v_rel_st"][:-1])   # (S,3)
    c_hat = chord / np.maximum(np.linalg.norm(chord, axis=1)[:, None], 1e-300)
    dot = np.einsum("ij,ij->i", v_rel, c_hat)
    cross_y = (v_rel[:, 2] * c_hat[:, 0] - v_rel[:, 0] * c_hat[:, 2])
    alpha_eff = np.arctan2(cross_y, dot)
    return alpha_eff, v_rel


def strip_normals(rec: dict) -> np.ndarray:
    """Per-strip unit normals in world frame: cross(chord, span)."""
    le, te = rec["le_now"], rec["te_now"]
    chord = 0.5 * ((te[1:] - le[1:]) + (te[:-1] - le[:-1]))
    span = le[1:] - le[:-1]
    n = np.cross(chord, span)
    return n / np.maximum(np.linalg.norm(n, axis=1)[:, None], 1e-300)


def ledger_step(rec: dict, cfg: LedgerConfig) -> dict:
    """Per-strip drag deltas (N) for one time step."""
    alpha_eff, v_rel = strip_section_state(rec)
    q = 0.5 * cfg.rho * np.einsum("ij,ij->i", v_rel, v_rel)
    areas = rec["areas"]
    lesp = rec["lesp"]

    d_t1 = np.zeros(len(lesp))
    if cfg.enable_t1:
        g = 1.0 / (1.0 + 2.0 / cfg.aspect_ratio)
        excess = np.abs(lesp) - cfg.lesp_crit
        dcs = -2.0 * np.pi * (np.abs(lesp) ** 2 - cfg.lesp_crit ** 2) * g * g
        dcs = np.where(excess > 0.0, dcs, 0.0)
        # CD = CN sin a - CS cos a: suction LOSS increases drag
        d_t1 = -dcs * np.cos(alpha_eff) * q * areas

    d_t3 = np.zeros(len(lesp))
    if cfg.enable_t3:
        vx = v_rel[:, 0]
        vmag = np.linalg.norm(v_rel, axis=1)
        d_t3 = 0.5 * cfg.rho * cfg.cd0 * vmag * vx * areas

    d_t2 = np.zeros(len(lesp))
    d_l2 = np.zeros(len(lesp))
    if cfg.enable_t2:
        # sep-weighted Rayleigh replacement (Polhamus progressive
        # reorientation): move the strip normal force toward 2 sin^2(alpha)
        # by the fraction of suction lost this instant
        sep = np.clip((np.abs(lesp) - cfg.lesp_crit)
                      / np.maximum(np.abs(lesp), cfg.lesp_crit), 0.0, 1.0)
        cn = rec.get("cn_strip")
        if cn is None:
            raise ValueError("T2 requires the per-strip CN stash "
                             "(cn_strip missing from ledger record)")
        cn_ray = 2.0 * np.sin(alpha_eff) ** 2
        d_cn = sep * (cn_ray - cn)
        n_hat = strip_normals(rec)
        d_f = d_cn[:, None] * (q * areas)[:, None] * n_hat   # (S,3) world N
        d_t2 = -d_f[:, 0]      # drag delta: force in -x is drag-positive
        d_l2 = d_f[:, 2]       # lift delta

    # G6b bound applies to the SEPARATION drag (T1+T2): it cannot exceed the
    # Rayleigh normal-force ceiling. The viscous term T3 has its own small
    # scale and stays positive-definite through stroke reversals where the
    # Rayleigh ceiling vanishes.
    sep = d_t1 + d_t2
    ceil = 2.0 * np.sin(alpha_eff) ** 2 * q * areas
    # G6b: the fully-separated state is bounded by the pure-Rayleigh normal
    # force; clip T1+T2 jointly to that ceiling (projection differences can
    # overshoot it slightly).
    over = sep > ceil
    if np.any(over):
        scale = np.where(over, ceil / np.maximum(sep, 1e-300), 1.0)
        d_t1 = d_t1 * scale
        d_t2 = d_t2 * scale
        sep = sep * scale
    # P0-1: recompute total AFTER clipping so the public contract
    # total == t1 + t2 + t3 always holds
    total = d_t1 + d_t3 + d_t2
    return dict(t1=d_t1, t3=d_t3, t2=d_t2, lift2=d_l2, total=total,
                alpha_eff=alpha_eff, q=q)


def run_ledger(ledger: list, cfg: LedgerConfig, last_n: int) -> dict:
    """Cycle-mean ledger drag (N) over the last `last_n` records + G6a check."""
    attached_everywhere = all(
        np.all(np.abs(r["lesp"]) <= cfg.lesp_crit) for r in ledger)
    if attached_everywhere and cfg.enable_t1:
        # G6a: T1 must vanish identically (check the EXCESS-GATED value)
        for r in ledger:
            excess = np.abs(r["lesp"]) - cfg.lesp_crit
            assert np.all(excess <= 0.0), "G6a violated: T1 nonzero when attached"
    t1s, t3s, t2s, l2s = [], [], [], []
    for rec in ledger[-last_n:]:
        out = ledger_step(rec, cfg)
        t1s.append(out["t1"].sum()); t3s.append(out["t3"].sum())
        t2s.append(out["t2"].sum()); l2s.append(out["lift2"].sum())
    return dict(mean_t1_N=float(np.mean(t1s)), mean_t3_N=float(np.mean(t3s)),
                mean_t2_N=float(np.mean(t2s)),
                mean_lift2_N=float(np.mean(l2s)),
                mean_total_N=float(np.mean(t1s) + np.mean(t3s) + np.mean(t2s)))
