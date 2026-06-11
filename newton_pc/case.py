"""Config-driven coupled-case runner (one solver configuration, many cases).

The platform configuration is FIXED and validated:
  - aero: ring-vortex UVLM, connected-lattice near wake (K rows) +
    far-field VORTEX PARTICLES (FLUXVortex hybrid), Lamb-Oseen+Squire cores
  - coupling: two-pass window PREDICTOR-CORRECTOR (newton_pc)
  - structure: rigid-kinematic or driven-root ANCF shell

Cases vary only through ``CaseConfig``. Built-in guards encode the lessons
from the validation campaigns (resonance proximity, explicit-elastic dt
stability, wake cost caps) so a new case fails loudly instead of silently
diverging.

Example:
    from newton_pc.case import CaseConfig, run_case
    cfg = CaseConfig(chord=1.5, span=6.0, kin_type="flap",
                     kin_amp_deg=5.0, kin_period=1.0, v_inf=10.0)
    res = run_case(cfg)
"""
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field

import numpy as np

from . import WindowPredictorCorrector
from .adapters.flap import (FlapEntry, FlapKinematics, FlapUVLMProvider,
                            NodalForceSet)


@dataclass
class CaseConfig:
    # geometry (rectangular planform)
    chord: float = 1.5
    span: float = 6.0
    nc: int = 6
    ns: int = 8
    # kinematics: none | flap (x-axis rotation at root); amp in degrees
    kin_type: str = "flap"
    kin_amp_deg: float = 5.0
    kin_period: float = 1.0
    # structure: rigid | elastic
    structure: str = "rigid"
    kscale: float = 1.0
    hscale: float = 1.0
    thickness: float = 2e-3
    rho_s: float = 1200.0
    E0: float = 5e9
    # fluid
    v_inf: float = 10.0
    alpha_deg: float = 1.0
    rho: float = 1.225
    nu: float = 15.06e-6
    # wake (platform default: K-ring near field + particle far field)
    K: int = 8
    particles: bool = True
    max_particles: int = 60000
    # coupling (platform default: two-pass predictor-corrector)
    mode: str = "two-pass"
    substeps: int = 8
    iterations: int = 1
    window_dt: float | None = None   # default: chord/(4*nc) convective scale
    cycles: float = 3.0
    # guard behaviour: "error" | "warn" | "off"
    guards: str = "error"


@dataclass
class CaseResult:
    lift: np.ndarray
    times: np.ndarray
    tip_z: np.ndarray
    config: CaseConfig
    wall_s: float
    n_solves: int
    warnings: list = field(default_factory=list)


def _estimate_f1(cfg: CaseConfig) -> float:
    """First bending frequency of the span-cantilevered plate (analytic)."""
    h = cfg.thickness * cfg.hscale
    E = cfg.E0 * cfg.kscale
    return (3.516 / (2 * np.pi)) * np.sqrt(E * h * h / (12 * cfg.rho_s)) \
        / cfg.span ** 2


def _run_guards(cfg: CaseConfig, dtw: float) -> list:
    msgs = []
    if cfg.structure == "elastic":
        # resonance proximity (the 1Hz-flapping trap)
        f1 = _estimate_f1(cfg)
        f_drive = 1.0 / cfg.kin_period if cfg.kin_type != "none" else 0.0
        if f_drive > 0 and 0.5 < f_drive / max(f1, 1e-9) < 2.0:
            msgs.append(f"RESONANCE: drive {f_drive:.2f}Hz within 2x of plate "
                        f"f1={f1:.2f}Hz (undamped response will grow)")
        # explicit-elastic dt stability (membrane wave bound)
        c_mem = np.sqrt(cfg.E0 * cfg.kscale / cfg.rho_s)
        dx = cfg.chord / cfg.nc
        dt_struct = dtw / cfg.substeps
        if dt_struct > 2.0 * dx / c_mem:
            need = int(np.ceil(dtw / (2.0 * dx / c_mem)))
            msgs.append(f"DT-STABILITY: dt_struct={dt_struct:.2e}s exceeds "
                        f"membrane bound {2 * dx / c_mem:.2e}s "
                        f"(suggest substeps>={need})")
    if cfg.K * cfg.ns > 4000:
        msgs.append(f"WAKE-COST: K={cfg.K} keeps {cfg.K * cfg.ns} near rings; "
                    "consider smaller K (particles carry the far field)")
    return msgs


def run_case(cfg: CaseConfig, on_window=None) -> CaseResult:
    # convective window: shed wake panel matches the chordwise panel size
    dtw = cfg.window_dt or (cfg.chord / cfg.nc) / cfg.v_inf
    guard_msgs = _run_guards(cfg, dtw)
    for m in guard_msgs:
        if cfg.guards == "error":
            raise ValueError(f"case guard: {m} (set guards='warn' to override)")
        if cfg.guards == "warn":
            warnings.warn(m)

    amp = np.deg2rad(cfg.kin_amp_deg) if cfg.kin_type != "none" else 0.0
    kin = FlapKinematics(amp, cfg.kin_period)
    entry = FlapEntry(cfg.chord, cfg.span, cfg.nc, cfg.ns, kin,
                      mode=("kinematic" if cfg.structure == "rigid"
                            else "elastic"),
                      kscale=cfg.kscale, hscale=cfg.hscale,
                      thickness=cfg.thickness, rho_s=cfg.rho_s, E0=cfg.E0)
    al = np.deg2rad(cfg.alpha_deg)
    V_vec = cfg.v_inf * np.array([np.cos(al), 0.0, np.sin(al)])
    provider = FlapUVLMProvider(V_vec, cfg.rho, dtw, K=cfg.K, nu=cfg.nu,
                                chord=cfg.chord, particles=cfg.particles,
                                max_particles=cfg.max_particles)
    pc = WindowPredictorCorrector(
        entry=entry, provider=provider, substeps=cfg.substeps,
        dt=dtw / cfg.substeps, mode=cfg.mode, iterations=cfg.iterations)
    pc.initialize(NodalForceSet(np.zeros(entry.shell.ndof)))

    horizon = cfg.cycles * cfg.kin_period if cfg.kin_type != "none" \
        else cfg.cycles
    n_windows = int(round(horizon / dtw))
    tip_dof = 9 * ((cfg.ns + 1) * (cfg.nc + 1) - 1) + 2
    lift, times, tip = [], [], []
    t0 = time.time()
    pc.advance(n_substeps=1)
    for w in range(n_windows):
        pc.advance()
        F = pc._F_cur.payload["f_panel"].sum(axis=(0, 1))
        L = -F[0] * np.sin(al) + F[2] * np.cos(al)
        if not np.isfinite(L):
            raise FloatingPointError(f"case diverged at window {w} "
                                     f"(t={pc._t:.3f}s)")
        lift.append(L)
        times.append(pc._t)
        tip.append(entry.shell.q[tip_dof])
        if on_window is not None:
            on_window(w, pc._t, L)
    return CaseResult(np.array(lift), np.array(times), np.array(tip), cfg,
                      time.time() - t0, provider.n_solves, guard_msgs)
