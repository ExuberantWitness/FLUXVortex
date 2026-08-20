"""CUDA-only V5M correction ledgers used by the four-paper validation.

Geometry objects and final dictionaries remain host-owned.  Every numerical
operation that changes a predicted force or coefficient is performed with
float64 Torch CUDA tensors.  No CPU implementation is called as a fallback.
"""
from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any

import numpy as np
import torch

from forward_flight_benchmarks.ldvm_uvlm_correction import (
    LDVMSectionSettings,
    LESPThreshold,
)
from forward_flight_benchmarks.uvlm_polar_correction import (
    DEFAULT_POLAR_PARAMETERS,
    FullAnglePolarParameters,
)
from ldvm_torch_gpu import LDVM2DCuda


def _device(device: str) -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA corrections require an available CUDA device")
    out = torch.device(device)
    if out.type != "cuda":
        raise ValueError("CUDA correction device must be CUDA")
    torch.empty(1, dtype=torch.float64, device=out)
    return out


def _cuda(value: Any, device: torch.device) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, dtype=torch.float64)
    host = np.array(value, dtype=np.float64, order="C", copy=True)
    return torch.from_numpy(host).to(device=device)


def _require_cuda_float64_tensor(
    name: str, value: Any, device: torch.device
) -> torch.Tensor:
    """Validate a production input without silently transferring or casting it."""

    if type(value) is not torch.Tensor:
        raise TypeError(f"{name} must be an exact torch.Tensor")
    if value.device.type != "cuda":
        raise ValueError(f"{name} must be CUDA; implicit host upload is forbidden")
    if value.device != device:
        raise ValueError(f"{name} must be on {device}, got {value.device}")
    if value.dtype is not torch.float64:
        raise TypeError(f"{name} must use torch.float64")
    return value


def _unit(value: torch.Tensor, floor: float = 1.0e-12) -> torch.Tensor:
    norm = torch.linalg.vector_norm(value, dim=-1, keepdim=True)
    return value / torch.clamp(norm, min=floor)


def run_ldvm_separation_pair_cuda(
    *,
    alpha_rad: torch.Tensor,
    alpha_rate_per_convective_time: torch.Tensor,
    heave_rate_over_u: torch.Tensor,
    delta_time_convective: float,
    pivot_fraction_chord: float,
    threshold: LESPThreshold,
    settings: LDVMSectionSettings = LDVMSectionSettings(),
    device: str = "cuda:0",
) -> dict[str, Any]:
    """CUDA equivalent of ``run_ldvm_separation_pair``."""
    d = _device(device)
    alpha = _require_cuda_float64_tensor("alpha_rad", alpha_rad, d).reshape(-1)
    alpha_rate = _require_cuda_float64_tensor(
        "alpha_rate_per_convective_time", alpha_rate_per_convective_time, d
    ).reshape(-1)
    heave_rate = _require_cuda_float64_tensor(
        "heave_rate_over_u", heave_rate_over_u, d
    ).reshape(-1)
    if (
        alpha.numel() < 4
        or alpha.shape != alpha_rate.shape
        or alpha.shape != heave_rate.shape
    ):
        raise ValueError(
            "CUDA LDVM histories must align and contain at least four rows"
        )
    if not bool(torch.isfinite(alpha + alpha_rate + heave_rate).all().item()):
        raise FloatingPointError("CUDA LDVM history contains non-finite values")
    if not math.isfinite(delta_time_convective) or delta_time_convective <= 0.0:
        raise ValueError("delta_time_convective must be finite and positive")
    if not 0.0 <= pivot_fraction_chord <= 1.0:
        raise ValueError("pivot must lie on the chord")
    core = (
        settings.core_radius_chord
        if settings.core_radius_chord is not None
        else settings.core_radius_time_step_ratio * float(delta_time_convective)
    )
    common = dict(
        U=1.0,
        c=1.0,
        ndiv=settings.ndiv,
        naterm=settings.naterm,
        dt=float(delta_time_convective),
        rho=1.0,
        pivot_xc=float(pivot_fraction_chord),
        core_rc=float(core),
        max_wake=settings.max_wake_steps,
        device=str(d),
    )
    separated = LDVM2DCuda(lesp_crit=threshold.value, **common)
    attached = LDVM2DCuda(lesp_crit=settings.attached_lesp_critical, **common)
    fields = ("CLf", "CDf", "CNf", "CNc", "CNnc", "CNnonl", "CSf", "A0", "lesp")
    separated_rows: dict[str, list[torch.Tensor]] = {field: [] for field in fields}
    attached_rows: dict[str, list[torch.Tensor]] = {field: [] for field in fields}
    lev_count: list[torch.Tensor] = []
    for index in range(alpha.numel()):
        sep = separated.step(alpha[index], alpha_rate[index], heave_rate[index])
        att = attached.step(alpha[index], alpha_rate[index], heave_rate[index])
        for field in fields:
            separated_rows[field].append(sep[field].clone())
            attached_rows[field].append(att[field].clone())
        lev_count.append(sep["n_lev"].to(dtype=torch.float64).clone())
    separated_history = {
        key: torch.stack(value) for key, value in separated_rows.items()
    }
    attached_history = {key: torch.stack(value) for key, value in attached_rows.items()}
    delta = {
        field: separated_history[field] - attached_history[field]
        for field in ("CLf", "CDf", "CNf", "CNc", "CNnc", "CNnonl", "CSf")
    }
    if not all(bool(torch.isfinite(value).all().item()) for value in delta.values()):
        raise FloatingPointError("CUDA LDVM pair produced a non-finite correction")
    counts = torch.stack(lev_count)
    return {
        "separated": separated_history,
        "attached": attached_history,
        "delta": delta,
        "lev_count": counts,
        "shed_lev": torch.diff(torch.cat((torch.zeros(1, device=d), counts))) > 0,
        "threshold": threshold.manifest(),
        "settings": asdict(settings),
        "delta_time_convective": float(delta_time_convective),
        "pivot_fraction_chord": float(pivot_fraction_chord),
        "numerical_contract": "torch-cuda-float64-no-cpu-fallback-v1",
    }


def project_ldvm_delta_to_finite_wing_cuda(
    delta_cnc: torch.Tensor,
    delta_cnnc: torch.Tensor,
    delta_cn_nonl: torch.Tensor,
    delta_cs: torch.Tensor,
    alpha_rad: torch.Tensor,
    *,
    aspect_ratio: float,
) -> dict[str, torch.Tensor | float]:
    named = (
        ("delta_cnc", delta_cnc),
        ("delta_cnnc", delta_cnnc),
        ("delta_cn_nonl", delta_cn_nonl),
        ("delta_cs", delta_cs),
        ("alpha_rad", alpha_rad),
    )
    if any(type(value) is not torch.Tensor for _, value in named):
        raise TypeError(
            "CUDA finite-wing projection requires exact torch.Tensor inputs"
        )
    d = _device(str(alpha_rad.device))
    for name, value in named:
        if value.device.type != "cuda":
            raise ValueError(f"{name} must be CUDA; implicit host upload is forbidden")
        if value.device != d:
            raise ValueError(f"{name} must be on {d}, got {value.device}")
        if value.dtype is not torch.float64:
            raise TypeError(f"{name} must use torch.float64")
    values = tuple(value.reshape(-1) for _, value in named)
    cnc, cnnc, cn_nonl, cs, alpha = values
    if not (cnc.shape == cnnc.shape == cn_nonl.shape == cs.shape == alpha.shape):
        raise ValueError("CUDA LDVM projection histories must align")
    if not math.isfinite(aspect_ratio) or aspect_ratio <= 0.0:
        raise ValueError("aspect ratio must be finite and positive")
    if not bool(torch.isfinite(cnc + cnnc + cn_nonl + cs + alpha).all().item()):
        raise FloatingPointError("CUDA finite-wing projection input is non-finite")
    aspect_ratio_t = torch.as_tensor(aspect_ratio, device=d, dtype=torch.float64)
    gain = 1.0 / (1.0 + 2.0 / aspect_ratio_t)
    added_mass_gain = torch.clamp(0.75 + aspect_ratio_t / 30.0, 0.0, 1.0)
    p_cnc = gain * cnc
    p_cnnc = added_mass_gain * cnnc
    p_nonl = gain * gain * cn_nonl
    p_cs = gain * gain * cs
    p_cn = p_cnc + p_cnnc + p_nonl
    return {
        "delta_CL": p_cn * torch.cos(alpha) + p_cs * torch.sin(alpha),
        "delta_CD": p_cn * torch.sin(alpha) - p_cs * torch.cos(alpha),
        "projected_delta_CNc": p_cnc,
        "projected_delta_CNnc": p_cnnc,
        "projected_delta_CNnonl": p_nonl,
        "projected_delta_CS": p_cs,
        "normal_gain": float(gain.item()),
        "added_mass_gain": float(added_mass_gain.item()),
        "nonlinear_normal_gain": float((gain * gain).item()),
        "axial_suction_gain": float((gain * gain).item()),
    }


def _extract_strip_geometry(airplane: Any) -> tuple[np.ndarray, ...]:
    """Copy Ptera geometry fields without performing aerodynamic arithmetic."""
    left_le: list[np.ndarray] = []
    right_le: list[np.ndarray] = []
    left_te: list[np.ndarray] = []
    right_te: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    areas: list[np.ndarray] = []
    for wing in airplane.wings:
        panels = wing.panels
        for span_index in range(panels.shape[1]):
            leading = panels[0, span_index]
            trailing = panels[-1, span_index]
            left_le.append(np.asarray(leading.Flpp_G_Cg, dtype=float))
            right_le.append(np.asarray(leading.Frpp_G_Cg, dtype=float))
            left_te.append(np.asarray(trailing.Blpp_G_Cg, dtype=float))
            right_te.append(np.asarray(trailing.Brpp_G_Cg, dtype=float))
            strip = panels[:, span_index]
            normals.append(
                np.asarray([panel.unitNormal_G for panel in strip], dtype=float)
            )
            areas.append(np.asarray([panel.area for panel in strip], dtype=float))
    if not left_le:
        raise ValueError("movement airplane contains no aerodynamic strips")
    return tuple(
        np.asarray(value)
        for value in (left_le, right_le, left_te, right_te, normals, areas)
    )


def _periodic_resample_cuda(
    phase: torch.Tensor, values: torch.Tensor, output_samples: int
) -> tuple[torch.Tensor, torch.Tensor]:
    if output_samples < 2:
        raise ValueError("output_samples must be at least two")
    phase = torch.remainder(phase, 1.0)
    order = torch.argsort(phase)
    xp = phase[order]
    yp = values[order]
    xp_ext = torch.cat((xp[-1:] - 1.0, xp, xp[:1] + 1.0))
    yp_ext = torch.cat((yp[-1:], yp, yp[:1]), dim=0)
    target = (
        torch.arange(output_samples, device=phase.device, dtype=torch.float64)
        / output_samples
    )
    right = torch.searchsorted(xp_ext, target, right=True)
    right = torch.clamp(right, 1, xp_ext.numel() - 1)
    left = right - 1
    weight = (target - xp_ext[left]) / (xp_ext[right] - xp_ext[left])
    shape = (output_samples,) + (1,) * (values.ndim - 1)
    out = yp_ext[left] + weight.reshape(shape) * (yp_ext[right] - yp_ext[left])
    return target, out


def movement_polar_residual_cuda(
    movement: Any,
    *,
    source_cycle_step_range: tuple[int, int] | list[int],
    period_s: float,
    freestream_m_s: float,
    rho_kg_m3: float,
    aspect_ratio: float,
    output_samples: int = 128,
    parameters: FullAnglePolarParameters = DEFAULT_POLAR_PARAMETERS,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """CUDA finite-wing full-angle polar residual."""
    d = _device(device)
    start, stop = (int(value) for value in source_cycle_step_range)
    if stop < start or min(period_s, freestream_m_s, rho_kg_m3, aspect_ratio) <= 0.0:
        raise ValueError("invalid CUDA polar inputs")
    step_ids = np.arange(start, stop + 1, dtype=int)
    expected = int(round(period_s / movement.delta_time))
    if step_ids.size != expected:
        raise ValueError("CUDA polar source cycle does not match period/delta_time")
    geometry = [
        _extract_strip_geometry(movement.airplanes[0][step]) for step in step_ids
    ]
    raw = [
        torch.stack([_cuda(item[index], d) for item in geometry]) for index in range(6)
    ]
    left_le, right_le, left_te, right_te, panel_normals, panel_areas = raw
    fraction = parameters.section_velocity_reference_fraction_chord
    left_ref = left_le + fraction * (left_te - left_le)
    right_ref = right_le + fraction * (right_te - right_le)
    centres = 0.5 * (left_ref + right_ref)
    chord = 0.5 * (left_te + right_te) - 0.5 * (left_le + right_le)
    span = right_ref - left_ref
    chord_hat = _unit(chord)
    span_hat = _unit(span)
    area = torch.sum(panel_areas, dim=2)
    normal_hat = _unit(
        torch.sum(panel_areas.unsqueeze(3) * panel_normals, dim=2) / area.unsqueeze(2)
    )
    panel_velocity = (torch.roll(centres, -1, 0) - torch.roll(centres, 1, 0)) / (
        2.0 * movement.delta_time
    )
    relative = (
        torch.tensor([freestream_m_s, 0.0, 0.0], device=d, dtype=torch.float64).reshape(
            1, 1, 3
        )
        - panel_velocity
    )
    relative = relative - torch.sum(relative * span_hat, dim=2, keepdim=True) * span_hat
    speed = torch.linalg.vector_norm(relative, dim=2)
    drag_hat = _unit(relative)
    alpha = torch.atan2(
        torch.sum(relative * normal_hat, dim=2), torch.sum(relative * chord_hat, dim=2)
    )
    lift_hat = _unit(
        normal_hat - torch.sum(normal_hat * drag_hat, dim=2, keepdim=True) * drag_hat
    )
    a0 = torch.as_tensor(
        parameters.section_lift_slope_per_rad, device=d, dtype=torch.float64
    )
    span_efficiency = torch.as_tensor(
        parameters.span_efficiency, device=d, dtype=torch.float64
    )
    aspect_ratio_t = torch.as_tensor(aspect_ratio, device=d, dtype=torch.float64)
    slope = a0 / (1.0 + a0 / (torch.pi * span_efficiency * aspect_ratio_t))
    alpha_abs_deg = torch.abs(torch.rad2deg(alpha))
    blend = torch.clamp(
        (alpha_abs_deg - parameters.attached_limit_deg)
        / (parameters.fully_separated_deg - parameters.attached_limit_deg),
        0.0,
        1.0,
    )
    blend = blend * blend * (3.0 - 2.0 * blend)
    delta_cl = blend * slope * (torch.sin(alpha) * torch.cos(alpha) - alpha)
    delta_cd = blend * parameters.drag_coefficient_at_90_deg * torch.sin(alpha) ** 2
    dynamic_pressure = 0.5 * rho_kg_m3 * speed * speed
    strip_force = (
        dynamic_pressure.unsqueeze(2)
        * area.unsqueeze(2)
        * (delta_cl.unsqueeze(2) * lift_hat + delta_cd.unsqueeze(2) * drag_hat)
    )
    unit_profile = torch.sum(
        dynamic_pressure.unsqueeze(2) * area.unsqueeze(2) * drag_hat, dim=1
    )
    raw_phase = _cuda(step_ids, d) * float(movement.delta_time) / period_s
    phase, strip_force_out = _periodic_resample_cuda(
        raw_phase, strip_force, output_samples
    )
    force_out = torch.sum(strip_force_out, dim=1)
    _, area_out = _periodic_resample_cuda(raw_phase, area, output_samples)
    _, unit_profile_out = _periodic_resample_cuda(
        raw_phase, unit_profile, output_samples
    )
    _, alpha_out = _periodic_resample_cuda(raw_phase, alpha, output_samples)
    _, speed_out = _periodic_resample_cuda(raw_phase, speed, output_samples)
    mean_strip = torch.mean(strip_force_out, dim=0)
    mean_force = torch.sum(mean_strip, dim=0)
    mean_unit_profile = torch.mean(unit_profile_out, dim=0)
    return {
        "phase": phase,
        "delta_force_g_n": force_out,
        "strip_delta_force_g_n": strip_force_out,
        "mean_strip_delta_force_g_n": mean_strip,
        "strip_area_m2": area_out,
        "mean_strip_area_m2": torch.mean(area, dim=0),
        "delta_lift_n": force_out[:, 2],
        "delta_drag_n": force_out[:, 0],
        "mean_delta_force_g_n": mean_force,
        "mean_delta_lift_n": mean_force[2],
        "mean_delta_drag_n": mean_force[0],
        "unit_profile_drag_force_g_n": unit_profile_out,
        "unit_profile_drag_lift_n": unit_profile_out[:, 2],
        "unit_profile_drag_drag_n": unit_profile_out[:, 0],
        "mean_unit_profile_drag_force_g_n": mean_unit_profile,
        "mean_unit_profile_drag_lift_n": mean_unit_profile[2],
        "mean_unit_profile_drag_drag_n": mean_unit_profile[0],
        "alpha_rad": alpha_out,
        "relative_speed_m_s": speed_out,
        "max_abs_alpha_deg": torch.max(torch.abs(torch.rad2deg(alpha))),
        "finite_wing_lift_slope_per_rad": float(slope.item()),
        "strip_count": int(centres.shape[1]),
        "source_cycle_step_range": [start, stop],
        "parameters": parameters.manifest(),
        "numerical_contract": "torch-cuda-float64-no-cpu-fallback-v1",
    }


def ledger_step_cuda(
    rec: dict[str, Any], cfg: Any, *, device: str = "cuda:0"
) -> dict[str, torch.Tensor]:
    """CUDA form of the frozen T1/T2/T3 per-strip ledger."""
    d = _device(device)
    le, te = _cuda(rec["le_now"], d), _cuda(rec["te_now"], d)
    chord = 0.5 * ((te[1:] - le[1:]) + (te[:-1] - le[:-1]))
    v_st = _cuda(rec["v_rel_st"], d)
    v_rel = 0.5 * (v_st[1:] + v_st[:-1])
    c_hat = _unit(chord, floor=1.0e-300)
    dot = torch.sum(v_rel * c_hat, dim=1)
    cross_y = v_rel[:, 2] * c_hat[:, 0] - v_rel[:, 0] * c_hat[:, 2]
    alpha = torch.atan2(cross_y, dot)
    q = 0.5 * cfg.rho * torch.sum(v_rel * v_rel, dim=1)
    areas = _cuda(rec["areas"], d)
    lesp = _cuda(rec["lesp"], d)
    t1 = torch.zeros_like(lesp)
    if cfg.enable_t1:
        gain = 1.0 / (1.0 + 2.0 / cfg.aspect_ratio)
        dcs = -2.0 * math.pi * (torch.abs(lesp) ** 2 - cfg.lesp_crit**2) * gain * gain
        dcs = torch.where(torch.abs(lesp) > cfg.lesp_crit, dcs, 0.0)
        t1 = -dcs * torch.cos(alpha) * q * areas
    t3 = torch.zeros_like(lesp)
    if cfg.enable_t3:
        t3 = (
            0.5
            * cfg.rho
            * cfg.cd0
            * torch.linalg.vector_norm(v_rel, dim=1)
            * v_rel[:, 0]
            * areas
        )
    t2 = torch.zeros_like(lesp)
    lift2 = torch.zeros_like(lesp)
    if cfg.enable_t2:
        sep = torch.clamp(
            (torch.abs(lesp) - cfg.lesp_crit)
            / torch.clamp(torch.abs(lesp), min=cfg.lesp_crit),
            0.0,
            1.0,
        )
        if "cn_strip" not in rec:
            raise ValueError("T2 requires cn_strip")
        d_cn = sep * (2.0 * torch.sin(alpha) ** 2 - _cuda(rec["cn_strip"], d))
        span = le[1:] - le[:-1]
        n_hat = _unit(torch.linalg.cross(chord, span, dim=1), floor=1.0e-300)
        d_force = d_cn.unsqueeze(1) * (q * areas).unsqueeze(1) * n_hat
        t2, lift2 = -d_force[:, 0], d_force[:, 2]
    separated = t1 + t2
    ceiling = 2.0 * torch.sin(alpha) ** 2 * q * areas
    scale = torch.where(
        separated > ceiling, ceiling / torch.clamp(separated, min=1.0e-300), 1.0
    )
    t1, t2 = t1 * scale, t2 * scale
    total = t1 + t2 + t3
    return {
        "t1": t1,
        "t2": t2,
        "t3": t3,
        "lift2": lift2,
        "total": total,
        "alpha_eff": alpha,
        "q": q,
    }


def run_ledger_cuda(
    ledger: list[dict[str, Any]], cfg: Any, last_n: int, *, device: str = "cuda:0"
) -> dict[str, torch.Tensor]:
    if last_n <= 0 or len(ledger) < last_n:
        raise ValueError("CUDA ledger does not contain the requested rows")
    rows = [ledger_step_cuda(record, cfg, device=device) for record in ledger[-last_n:]]
    means = {
        name: torch.mean(torch.stack([torch.sum(row[name]) for row in rows]))
        for name in ("t1", "t2", "t3", "lift2")
    }
    return {
        "mean_t1_N": means["t1"],
        "mean_t2_N": means["t2"],
        "mean_t3_N": means["t3"],
        "mean_lift2_N": means["lift2"],
        "mean_total_N": means["t1"] + means["t2"] + means["t3"],
        "numerical_contract": "torch-cuda-float64-no-cpu-fallback-v1",
    }


__all__ = [
    "ledger_step_cuda",
    "movement_polar_residual_cuda",
    "project_ldvm_delta_to_finite_wing_cuda",
    "run_ldvm_separation_pair_cuda",
    "run_ledger_cuda",
]
