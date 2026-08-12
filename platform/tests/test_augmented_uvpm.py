from __future__ import annotations

import numpy as np

from forward_flight_benchmarks.augmented_uvpm import (
    added_mass_aspect_ratio_factor,
    blend_periodic_ullt_state_shape,
    hoerner_lifting_surface_gain,
    run_augmented_fluxv,
)
from forward_flight_benchmarks.cases import IZRAELEVITZ_2017_FIG11
from forward_flight_benchmarks.ptera_adapter import (
    build_izraelevitz_fig11_movement,
)
from forward_flight_benchmarks.uvlm_polar_correction import (
    movement_polar_residual,
)


def test_source_derived_aspect_ratio_gains() -> None:
    assert added_mass_aspect_ratio_factor(1.9) == 0.85
    assert added_mass_aspect_ratio_factor(3.0) == 0.85
    assert added_mass_aspect_ratio_factor(6.0) == 0.95
    assert added_mass_aspect_ratio_factor(20.0) == 0.95
    assert np.isclose(hoerner_lifting_surface_gain(3.0), 0.9101698376462753)
    assert 0.0 < hoerner_lifting_surface_gain(1.9) < 1.0


def test_gains_reject_nonphysical_aspect_ratio() -> None:
    for function in (added_mass_aspect_ratio_factor, hoerner_lifting_surface_gain):
        try:
            function(0.0)
        except ValueError:
            pass
        else:
            raise AssertionError("non-positive aspect ratio should fail")


def test_periodic_state_blend_preserves_uvlm_owned_mean() -> None:
    phase = np.arange(8, dtype=float) / 8.0
    ullt = {
        "phase": phase,
        "lift_n": 2.0 + np.sin(2.0 * np.pi * phase),
        "drag_n": -1.0 + np.cos(2.0 * np.pi * phase),
    }
    separated = {
        "phase": phase,
        "lift_n": 8.0 + 3.0 * np.sin(2.0 * np.pi * phase),
        "drag_n": 4.0 + 2.0 * np.cos(2.0 * np.pi * phase),
        "mean_lift_n": 8.0,
        "mean_drag_n": 4.0,
    }
    result = blend_periodic_ullt_state_shape(
        ullt,
        separated,
        np.linspace(0.0, 1.0, 8),
        rho_kg_m3=1.0,
        freestream_m_s=2.0,
        area_m2=0.5,
    )
    assert np.isclose(np.mean(result["lift_n"]), 8.0)
    assert np.isclose(np.mean(result["drag_n"]), 4.0)
    assert result["mean_lift_n"] == 8.0
    assert result["mean_drag_n"] == 4.0


def test_particle_fast_path_is_exactly_load_equivalent_on_smoke_case() -> None:
    case = IZRAELEVITZ_2017_FIG11
    histories = []
    for record_particles in (False, True):
        movement, _ = build_izraelevitz_fig11_movement("smoke")
        residual = movement_polar_residual(
            movement,
            source_cycle_step_range=(24, 47),
            period_s=case.period_s,
            freestream_m_s=case.freestream_m_s,
            rho_kg_m3=case.rho_kg_m3,
            aspect_ratio=case.aspect_ratio,
            output_samples=32,
        )
        histories.append(
            run_augmented_fluxv(
                movement,
                period_s=case.period_s,
                rho_kg_m3=case.rho_kg_m3,
                freestream_m_s=case.freestream_m_s,
                area_m2=case.area_m2,
                aspect_ratio=case.aspect_ratio,
                polar_residual=residual,
                output_samples=32,
                record_vpm_particles=record_particles,
            )
        )
    np.testing.assert_allclose(histories[0]["lift_n"], histories[1]["lift_n"])
    np.testing.assert_allclose(histories[0]["drag_n"], histories[1]["drag_n"])
    assert histories[0]["particle_count"] == 0
    assert histories[1]["particle_count"] > 0
