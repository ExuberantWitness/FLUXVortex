"""Kinematics of the published cases of note A24.  CPU only, no solver.

Usage
-----
    CUDA_VISIBLE_DEVICES="" FLUXV_GPU_ONLY=0 PFIELD_DEVICE=cpu PYTHONPATH=src \\
        python3 -m pytest tests/test_research_kinematics_cpu.py -q
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from fluxvortex.cases import research_kinematics as rk
from fluxvortex.cases.izraelevitz_kinematics import (
    build_izra_reference_grid,
    izra_motion_law,
)

DEVICE = "cpu"
TOL = 1.0e-12


def tensors_of(law, time: float) -> tuple[torch.Tensor, ...]:
    return tuple(value.detach().clone() for value in law(time))


def test_heave_pitch_law_is_the_izraelevitz_law_at_zero_phase():
    config = SimpleNamespace(
        heave_amplitude_m=0.06096,
        omega_rad_s=31.41592653589793,
        theta_max_deg=15.0,
        phase_offset_deg=0.0,
    )
    reference = izra_motion_law(config, device=DEVICE)
    law = rk.heave_pitch_law(
        omega_rad_s=config.omega_rad_s,
        heave_amplitude_m=config.heave_amplitude_m,
        pitch_amplitude_rad=math.radians(config.theta_max_deg),
        device=DEVICE,
    )
    for time in (0.0, 0.0037, 0.011, 0.02):
        for ours, theirs in zip(tensors_of(law, time), tensors_of(reference, time), strict=True):
            torch.testing.assert_close(ours, theirs, rtol=0.0, atol=TOL)


def test_heave_pitch_law_phases_give_the_li_waveform():
    h, omega, alpha_max = 0.025, 1.0, math.radians(16.84)
    law = rk.heave_pitch_law(
        omega_rad_s=omega,
        heave_amplitude_m=h,
        pitch_amplitude_rad=alpha_max,
        phase_heave_rad=math.pi,
        phase_pitch_rad=-0.5 * math.pi,
        device=DEVICE,
    )
    for time in (0.0, 0.4, 1.1, 2.6):
        pos, quat, lin_vel, ang_vel = tensors_of(law, time)
        # the plate at -h cos(omega t), the geometric angle alpha_max sin(omega t)
        assert abs(float(pos[2]) + h * math.cos(omega * time)) < TOL
        theta = 2.0 * math.atan2(float(quat[2]), float(quat[0]))
        assert abs(theta - alpha_max * math.sin(omega * time)) < TOL
        assert abs(float(lin_vel[2]) - h * omega * math.sin(omega * time)) < TOL
        assert abs(float(ang_vel[1]) - alpha_max * omega * math.cos(omega * time)) < TOL


def test_pitch_ramp_law_rate_and_holds():
    speed, chord = 0.3, 1.0
    # Fig. 9 of Hirato et al.: -5 deg held, then a ramp at K = 0.3 (0.6 rad per unit of
    # convective time) to 45 deg; the start is the one that puts 24.15 deg at t* = 1.710
    start = 0.862
    knots = [(0.0, -5.0), (start, -5.0), (start + math.radians(50.0) / 0.6, 45.0), (10.0, 45.0)]
    law = rk.pitch_ramp_law(knots=knots, speed=speed, chord=chord, device=DEVICE)
    # (convective time, angle in degrees, rate d(alpha)/d(t*) in rad)
    for star, angle_deg, rate_star in ((0.4, -5.0, 0.0), (1.710, 24.15, 0.6), (2.040, 35.49, 0.6), (2.8, 45.0, 0.0)):
        pos, quat, lin_vel, ang_vel = tensors_of(law, star * chord / speed)
        theta = 2.0 * math.atan2(float(quat[2]), float(quat[0]))
        # the labels of the paper's figures are given to 0.01 deg and 0.0005 in t*
        assert abs(math.degrees(theta) - angle_deg) < 0.02
        assert abs(float(ang_vel[1]) - rate_star * speed / chord) < 1.0e-12
        assert float(pos.abs().max()) == 0.0 and float(lin_vel.abs().max()) == 0.0


def test_smoothed_ramp_profile_goes_up_holds_and_returns():
    # case II of Martinez-Carmena et al.: K = 0.2, 45 deg, the maximum near t* = 4, hold to 6
    profile = rk.smoothed_ramp_profile(rate=0.2, smoothing=11.0, start=2.0, maximum_deg=45.0, hold=2.037)
    assert abs(math.degrees(profile(0.5)[0])) < 1.0e-6
    assert abs(math.degrees(profile(5.0)[0]) - 45.0) < 0.05
    assert abs(math.degrees(profile(9.5)[0])) < 1.0e-6
    # far from the corners the rate is d(alpha)/d(t*) = 2 K
    assert abs(profile(3.0)[1] - 0.4) < 0.02
    assert abs(profile(7.0)[1] + 0.4) < 0.02
    # and the stated derivative is the derivative of the stated angle
    step = 1.0e-6
    for star in (2.3, 3.0, 3.96, 6.0, 7.0, 7.9):
        angle, rate = profile(star)
        numeric = (profile(star + step)[0] - profile(star - step)[0]) / (2.0 * step)
        assert abs(rate - numeric) < 1.0e-6 * max(1.0, abs(rate)) + 1.0e-9
        assert abs(angle - profile(star)[0]) == 0.0


def test_piecewise_profile_is_held_outside_the_knots():
    profile = rk.piecewise_linear_profile([(0.0, -5.0), (1.0, -5.0), (2.0, 40.0)])
    assert profile(-1.0) == (math.radians(-5.0), 0.0)
    assert profile(1.5) == (math.radians(17.5), math.radians(45.0))
    assert profile(9.0) == (math.radians(40.0), 0.0)
    with pytest.raises(ValueError):
        rk.piecewise_linear_profile([(1.0, 0.0), (1.0, 3.0)])


def test_rigid_case_kinematics_puts_the_leading_edge_up_for_positive_pitch():
    kin = rk.rigid_case_kinematics(
        law=rk.heave_pitch_law(omega_rad_s=2.0, pitch_amplitude_rad=0.0, pitch_offset_rad=math.radians(10.0), device=DEVICE),
        chord_m=0.1, span_m=0.6, pivot_fraction_chord=0.25,
        chordwise_panels=4, spanwise_panels=6, device=DEVICE,
    )
    frame = kin.evaluate(0.0)
    leading = frame.leading_edge_I[:, 2].mean()
    trailing = frame.trailing_edge_I[:, 2].mean()
    assert float(leading) > 0.0 > float(trailing)
    # the leading edge is a quarter chord ahead of the pitch axis and the trailing edge
    # three quarters behind it; the axis itself (the origin) does not move
    assert abs(float(leading) - 0.25 * 0.1 * math.sin(math.radians(10.0))) < TOL
    assert abs(float(trailing) + 0.75 * 0.1 * math.sin(math.radians(10.0))) < TOL


def hinged(alpha_deg: float, delta0_deg: float = 0.0) -> rk.HingedFlapKinematics:
    return rk.HingedFlapKinematics(
        chord_m=0.1, half_span_m=0.3, alpha_deg=alpha_deg, delta0_deg=delta0_deg,
        delta1_deg=30.0, omega_rad_s=22.0, chordwise_panels=8, spanwise_panels=34, device=DEVICE,
    )


def test_hinged_frame_matches_the_two_rotations():
    kin = hinged(20.0, -30.0)
    chord, dx, dy = 0.1, 0.1 / 8, 0.6 / 34
    for time in (0.0, 0.0137, 0.0421):
        delta, _ = kin.delta(time)
        frame = kin.evaluate(time)
        cos_a, sin_a = math.cos(math.radians(20.0)), math.sin(math.radians(20.0))
        # corner k of ring (i, j) sits on the quarter lines x = (i + 0.25) dx and x = (i + 1.25) dx
        for i in range(8):
            for j in (0, 7, 17, 25, 33):
                for corner, (line, node) in enumerate(((i, j), (i, j + 1), (i + 1, j + 1), (i + 1, j))):
                    x_ref = (line + 0.25) * dx if line < 8 else chord + 0.25 * dx
                    y_ref = -0.3 + node * dy
                    flapped_z = abs(y_ref) * math.sin(delta)
                    expected = (
                        x_ref * cos_a + flapped_z * sin_a,
                        y_ref * math.cos(delta),
                        -x_ref * sin_a + flapped_z * cos_a,
                    )
                    actual = frame.panel_rings_I[i * 34 + j, corner]
                    torch.testing.assert_close(
                        actual, torch.tensor(expected, dtype=torch.float64), rtol=0.0, atol=TOL
                    )


def test_hinged_frame_velocities_are_the_derivative_of_the_positions():
    kin = hinged(2.0, -30.0)
    step = 1.0e-6
    for time in (0.0113, 0.0302, 0.0771):
        before, frame, after = kin.evaluate(time - step), kin.evaluate(time), kin.evaluate(time + step)
        for name in ("panel_rings_I", "collocation_I", "leading_edge_I", "trailing_edge_I"):
            positions = getattr(frame, name)
            derivative = (getattr(after, name) - getattr(before, name)) / (2.0 * step)
            velocity = getattr(
                frame,
                {"panel_rings_I": "panel_ring_velocity_I", "collocation_I": "collocation_velocity_I",
                 "leading_edge_I": "leading_velocity_I", "trailing_edge_I": "trailing_velocity_I"}[name],
            )
            torch.testing.assert_close(velocity, derivative, rtol=0.0, atol=1.0e-6)


def test_hinged_frame_is_mirror_symmetric_and_keeps_the_centreline():
    kin = hinged(20.0, 0.0)
    for time in (0.005, 0.0171, 0.0299, 0.0611):
        frame = kin.evaluate(time)
        for nodes in (frame.leading_edge_I, frame.trailing_edge_I):
            mirrored = nodes.flip(0).clone()
            mirrored[:, 1] = -mirrored[:, 1]
            torch.testing.assert_close(nodes, mirrored, rtol=0.0, atol=TOL)
        # the two centreline nodes stay on the pitched root chord line
        centre = torch.stack((frame.leading_edge_I[17], frame.trailing_edge_I[17]))
        expected = torch.tensor(
            [[0.0, 0.0, 0.0], [0.1 * math.cos(math.radians(20.0)), 0.0, -0.1 * math.sin(math.radians(20.0))]],
            dtype=torch.float64,
        )
        torch.testing.assert_close(centre, expected, rtol=0.0, atol=TOL)


def test_hinged_frame_normals_are_unit_and_each_half_is_planar():
    kin = hinged(20.0, 0.0)
    for time in (0.005, 0.0171, 0.0299):
        delta, _ = kin.delta(time)
        frame = kin.evaluate(time)
        torch.testing.assert_close(
            torch.linalg.vector_norm(frame.normals_I, dim=1), torch.ones(272, dtype=torch.float64), rtol=0.0, atol=1.0e-14
        )
        rings = frame.panel_rings_I.reshape(8, 34, 4, 3)
        normals = frame.normals_I.reshape(8, 34, 3)
        alpha = math.radians(20.0)
        # the normal of a half is R_y(alpha) R_x(+-delta) z
        for strips, sign in ((slice(0, 17), -1.0), (slice(17, 34), 1.0)):
            expected = torch.tensor(
                [math.cos(delta) * math.sin(alpha), -sign * math.sin(delta), math.cos(delta) * math.cos(alpha)],
                dtype=torch.float64,
            )
            torch.testing.assert_close(normals[:, strips], expected.expand(8, 17, 3), rtol=0.0, atol=1.0e-14)
            # and every corner of that half lies in the plane through the origin with that normal
            assert float((rings[:, strips] * expected).sum(dim=-1).abs().max()) < 1.0e-14


def test_hinged_start_ramp_begins_from_rest_and_reaches_the_paper_law():
    kin = rk.HingedFlapKinematics(
        chord_m=0.1, half_span_m=0.3, alpha_deg=0.0, delta0_deg=-30.0, delta1_deg=30.0,
        omega_rad_s=22.0, start_ramp_s=2.0 * math.pi / 22.0, chordwise_panels=4, spanwise_panels=6,
        device=DEVICE,
    )
    assert kin.delta(0.0) == (0.0, 0.0)
    plain = hinged(0.0, -30.0)
    plain._omega = kin._omega
    period = 2.0 * math.pi / kin._omega
    for time in (period, period + 0.01, 3.7 * period):
        assert kin.delta(time) == plain.delta(time)
    before = kin.delta(0.4)[0]
    assert -math.radians(30.0) < before < 0.0      # folded part of the way towards the mean angle
    step = 1.0e-7
    numeric = (kin.delta(0.4 + step)[0] - kin.delta(0.4 - step)[0]) / (2.0 * step)
    assert abs(kin.delta(0.4)[1] - numeric) < 1.0e-6


def test_hinged_frame_at_rest_is_the_native_rectangular_layout():
    kin = hinged(0.0, 0.0)
    kin._delta1 = 0.0
    frame = kin.evaluate(0.031)
    reference = build_izra_reference_grid(
        chordwise_panels=8, spanwise_panels=34, chord_m=0.1, span_m=0.6,
        pivot_fraction_chord=0.0, device=DEVICE,
    )
    for name in ("panel_rings_I", "collocation_I", "normals_I", "areas", "leading_edge_I", "trailing_edge_I"):
        torch.testing.assert_close(getattr(frame, name), getattr(reference, name), rtol=0.0, atol=TOL)
    assert float(frame.panel_ring_velocity_I.abs().max()) == 0.0
    assert float(frame.collocation_velocity_I.abs().max()) == 0.0
