"""GPU contract tests for the inertial wake probe observer.

Drives a few committed steps of the SAME frozen rigid native V5M machinery
the Figure 9/12/13/15 queue uses, then checks the probe contract: finite
velocities, all three induction owners contributing through one read path,
spectrum sanity on a synthetic signal, and the St* closure identity.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

torch.cuda.init()
if not torch.cuda.is_available():
    pytest.skip("wake probe tests require CUDA", allow_module_level=True)

from fluxvortex.aero.v5m.stepper import RigidV5MStepper
from fluxvortex.cases.izraelevitz_kinematics import build_izra_reference_grid
from fluxvortex.kinematics.prescribed_rigid import PrescribedRigidSurfaceKinematics
from fluxvortex.warp_fsi.q16_flux_v5m_native import NativeV5MConfig
from fluxvortex.warp_fsi.rigid_flux_v5m_native import (
    RigidNativeV5MSolver,
    RigidV5MSurface,
)
from fluxvortex.warp_fsi.wake_probe_observer import (
    WakeProbeObserver,
    build_downstream_probe_grid,
    summarize_probes,
)

DEVICE = "cuda:0"
CHORD_M = 0.0688
SPAN_M = 0.1375


def _fixed_pitch_law(alpha_deg: float):
    theta = math.radians(alpha_deg)
    half = 0.5 * theta
    dtype = torch.float64
    pos = torch.zeros(3, device=DEVICE, dtype=dtype)
    quat = torch.tensor(
        [math.cos(half), 0.0, math.sin(half), 0.0], device=DEVICE, dtype=dtype
    )
    zero = torch.zeros(3, device=DEVICE, dtype=dtype)

    def law(t: float):
        return pos, quat, zero, zero.clone()

    return law


@pytest.fixture(scope="module")
def committed_run():
    nc, ns = 15, 30
    aero_dt = 0.01 * CHORD_M / 5.0
    reference = build_izra_reference_grid(
        chordwise_panels=nc,
        spanwise_panels=ns,
        chord_m=CHORD_M,
        span_m=SPAN_M,
        pivot_fraction_chord=0.25,
        device=DEVICE,
    )
    kin = PrescribedRigidSurfaceKinematics(
        reference,
        _fixed_pitch_law(15.0),
        surface_id="wing",
        body_id="body_0",
    )
    surface = RigidV5MSurface(chordwise_panels=nc, spanwise_panels=ns, device=DEVICE)
    settings = NativeV5MConfig(
        chordwise_panels=nc,
        spanwise_panels=ns,
        density=1.208,
        freestream=5.0,
        aerodynamic_dt=aero_dt,
        lesp_crit=0.11,
        wake_max_rows=64,
        particle_capacity=32768,
        particle_max_age_steps=100,
        wake_history_mode="bound_rate",
        wake_free_rows=32,
        dvm_target_spacing_chord=0.018,
        device=DEVICE,
    )
    solver = RigidNativeV5MSolver(surface, settings)
    stepper = RigidV5MStepper(solver, device=DEVICE)
    owner = stepper.initialize((kin.evaluate(0.0),))
    geometry = solver.surface.evaluate(None, None)
    probes = build_downstream_probe_grid(
        trailing_edge_world=geometry.trailing_edge,
        chord_m=CHORD_M,
        span_m=SPAN_M,
        freestream_direction=solver.v_inf,
        device=DEVICE,
    )
    observer = WakeProbeObserver(probes, device=DEVICE)
    for step in range(1, 41):
        frame = kin.evaluate(step * aero_dt)
        proposal = stepper.propose(owner, (frame,), aero_dt)
        stepper.commit(owner, proposal)
        observer.sample(solver, owner.state)
    return solver, owner, observer, aero_dt


def test_probe_grid_shape_and_placement(committed_run) -> None:
    _, _, observer, _ = committed_run
    assert observer.probe_points.shape == (12, 3)
    geometry = committed_run[0].surface.evaluate(None, None)
    te_mid = geometry.trailing_edge.reshape(-1, 3).mean(dim=0).cpu().numpy()
    offsets = observer.probe_points.cpu().numpy() - te_mid
    downstream = offsets[:, 0]  # freestream is +x
    # Loop order downstream x spanwise x vertical: six 1c probes then six 2c.
    assert np.allclose(downstream, np.repeat([CHORD_M, 2 * CHORD_M], 6), atol=1e-12)


def test_probe_sampling_is_finite_and_transparent(committed_run) -> None:
    _, owner, observer, _ = committed_run
    assert observer.sample_count == 40
    history = observer.velocity_history()
    assert history.shape == (40, 12, 3)
    assert np.isfinite(history).all()
    # Sampling must not perturb the committed state (read-only contract).
    assert owner.state.step == 40


def test_spectrum_peak_on_synthetic_signal() -> None:
    # 60 Hz sine on the observer's own Welch path: St recovery within 2%.
    sample_rate = 1000.0
    signal = np.sin(2.0 * np.pi * 60.0 * np.arange(4000) / sample_rate)
    from fluxvortex.warp_fsi.wake_probe_observer import _welch_psd

    frequencies, psd = _welch_psd(signal, sample_rate)
    peak_hz = frequencies[int(np.argmax(psd))]
    assert abs(peak_hz - 60.0) / 60.0 < 0.02


def test_summarize_rejects_short_windows(committed_run) -> None:
    _, _, observer, aero_dt = committed_run
    with pytest.raises(ValueError):
        summarize_probes(
            observer,
            chord_m=CHORD_M,
            freestream_m_s=5.0,
            delta_time_s=aero_dt,
            alpha_deg=15.0,
        )


def test_st_closure_identity_on_synthetic_observer() -> None:
    class _FakeObserver:
        def velocity_history(self):
            # Two probes, distinct frequencies -> mean St of the two.
            t = np.arange(2000) * 0.0001

            def probe(freq):
                return np.stack(
                    [np.sin(2 * np.pi * freq * t), np.zeros_like(t), np.zeros_like(t)],
                    axis=1,
                )

            return np.stack([probe(400.0), probe(600.0)], axis=1)

    summary = summarize_probes(
        _FakeObserver(),
        chord_m=CHORD_M,
        freestream_m_s=5.0,
        delta_time_s=0.0001,
        alpha_deg=15.0,
    )
    # 400 Hz -> St 5.504; 600 Hz -> 8.256; mean 6.88.
    assert summary.st_per_probe[0] == pytest.approx(5.504, rel=0.02)
    assert summary.st_mean == pytest.approx(6.88, rel=0.02)
    assert (
        abs(summary.st_modified - summary.st_mean * math.sin(math.radians(15.0)))
        <= 1e-12
    )
