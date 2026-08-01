import sys
import unittest
from pathlib import Path

import numpy as np

PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from claim_runtime.svi_dw_types import SVIDWValidationError  # noqa: E402
from claim_runtime.svi_dw_weak_uk_2d import (  # noqa: E402
    BoundCirculationStep2D,
    MaterialWakeState2D,
    TEBernoulliPressureJump2D,
    TEControlVolume2D,
    WeakUKReferenceScales2D,
    kelvin_material_wake_ledger,
    material_wake_transport_ledger,
    te_control_volume_vorticity_flux,
    te_transverse_momentum_ledger,
    weak_unsteady_kutta_ledger,
)


def square_control_volume(stage_id="stage-1"):
    return TEControlVolume2D(
        control_volume_id="te-cv",
        stage_id=stage_id,
        vertices=np.array(
            (
                (0.0, 0.0),
                (1.0, 0.0),
                (1.0, 1.0),
                (0.0, 1.0),
                (0.0, 0.0),
            )
        ),
    )


class SVIDWWeakUK2DTests(unittest.TestCase):
    def setUp(self):
        self.scales = WeakUKReferenceScales2D(
            chord=2.0,
            velocity=3.0,
            density=1.2,
        )

    def test_material_transport_and_kelvin_manufactured_identity(self):
        previous = MaterialWakeState2D(
            history_id="wake",
            stage_id="stage-0",
            time=0.0,
            material_ids=("old-0", "old-1", "old-2"),
            positions=((0.0, 0.0), (1.0, 0.5), (2.0, 0.75)),
            potential_jump=(10.0, 10.1, 10.3),
            oriented_edges=((0, 1), (1, 2)),
        )
        velocity = np.array(
            ((2.0, -1.0), (-0.5, 0.25), (0.2, -0.1))
        )
        current = MaterialWakeState2D(
            history_id="wake",
            stage_id="stage-1",
            time=0.2,
            material_ids=("old-0", "old-1", "old-2"),
            positions=previous.positions + 0.2 * velocity,
            potential_jump=(10.0, 10.1, 10.3),
            oriented_edges=((0, 1), (1, 2)),
        )
        transport = material_wake_transport_ledger(
            previous,
            current,
            stage_velocity=velocity,
            scales=self.scales,
        )
        self.assertTrue(transport.passed)
        self.assertEqual(transport.maximum_jump_mutation, 0.0)
        np.testing.assert_allclose(
            transport.position_residual, 0.0, rtol=0.0, atol=1.0e-15
        )

        kelvin = kelvin_material_wake_ledger(
            bound_circulation=0.7,
            wake=current,
            reference_circulation=1.0,
            scales=self.scales,
        )
        self.assertTrue(kelvin.passed)
        self.assertAlmostEqual(kelvin.wake_circulation, 0.3)
        self.assertAlmostEqual(kelvin.residual, 0.0)

        mutated = MaterialWakeState2D(
            history_id="wake",
            stage_id="stage-1-mutated",
            time=0.2,
            material_ids=("old-0", "old-1", "old-2"),
            positions=previous.positions + 0.2 * velocity,
            potential_jump=(10.0, 10.1, 10.3001),
            oriented_edges=((0, 1), (1, 2)),
        )
        mutation_ledger = material_wake_transport_ledger(
            previous,
            mutated,
            stage_velocity=velocity,
            scales=self.scales,
        )
        self.assertFalse(mutation_ledger.passed)
        self.assertAlmostEqual(
            mutation_ledger.maximum_jump_mutation, 0.0001
        )

    def test_kelvin_is_gauge_invariant_and_orientation_sensitive(self):
        common = dict(
            history_id="wake",
            stage_id="stage-1",
            time=0.2,
            material_ids=("te", "middle", "far"),
            positions=((0.0, 0.0), (1.0, 0.0), (2.0, 0.0)),
        )
        wake = MaterialWakeState2D(
            **common,
            potential_jump=(10.0, 10.1, 10.3),
            oriented_edges=((0, 1), (1, 2)),
        )
        shifted = MaterialWakeState2D(
            **common,
            potential_jump=(1010.0, 1010.1, 1010.3),
            oriented_edges=((0, 1), (1, 2)),
        )
        reversed_wake = MaterialWakeState2D(
            **common,
            potential_jump=(10.0, 10.1, 10.3),
            oriented_edges=((1, 0), (2, 1)),
        )
        with_far_vortex = MaterialWakeState2D(
            **common,
            potential_jump=(10.0, 10.1, 10.3),
            oriented_edges=((0, 1), (1, 2)),
            far_point_vortex_circulation=0.2,
        )
        self.assertAlmostEqual(wake.total_circulation, 0.3)
        self.assertAlmostEqual(
            shifted.total_circulation, wake.total_circulation
        )
        self.assertNotAlmostEqual(
            wake.total_circulation,
            float(np.sum(wake.potential_jump)),
        )
        self.assertAlmostEqual(
            reversed_wake.total_circulation, -wake.total_circulation
        )
        self.assertAlmostEqual(with_far_vortex.total_circulation, 0.5)
        forward = kelvin_material_wake_ledger(
            bound_circulation=0.7,
            wake=wake,
            reference_circulation=1.0,
            scales=self.scales,
        )
        reverse = kelvin_material_wake_ledger(
            bound_circulation=0.7,
            wake=reversed_wake,
            reference_circulation=1.0,
            scales=self.scales,
        )
        self.assertTrue(forward.passed)
        self.assertFalse(reverse.passed)

    def test_old_transport_excludes_newborn_boundary_nodes(self):
        previous = MaterialWakeState2D(
            history_id="wake",
            stage_id="stage-0",
            time=0.0,
            material_ids=("old-0", "old-1"),
            positions=((0.0, 0.0), (1.0, 0.0)),
            potential_jump=(2.0, 2.25),
            oriented_edges=((0, 1),),
        )
        current = MaterialWakeState2D(
            history_id="wake",
            stage_id="stage-1",
            time=0.1,
            material_ids=("old-0", "old-1", "newborn"),
            positions=((0.1, 0.0), (1.1, 0.0), (-0.2, 0.4)),
            potential_jump=(2.0, 2.25, -30.0),
            oriented_edges=((0, 1), (1, 2)),
        )
        ledger = material_wake_transport_ledger(
            previous,
            current,
            stage_velocity=((1.0, 0.0), (1.0, 0.0)),
            scales=self.scales,
        )
        self.assertTrue(ledger.passed)
        self.assertEqual(
            ledger.retained_material_ids, ("old-0", "old-1")
        )
        self.assertEqual(ledger.newborn_material_ids, ("newborn",))

    def test_explicit_empty_history_is_valid_but_missing_history_fails(self):
        previous = MaterialWakeState2D.empty(
            history_id="wake",
            stage_id="stage-0",
            time=0.0,
        )
        current = MaterialWakeState2D.empty(
            history_id="wake",
            stage_id="stage-1",
            time=0.1,
        )
        transport = material_wake_transport_ledger(
            previous,
            current,
            stage_velocity=np.empty((0, 2)),
            scales=self.scales,
        )
        self.assertTrue(transport.passed)
        kelvin = kelvin_material_wake_ledger(
            bound_circulation=0.0,
            wake=current,
            reference_circulation=0.0,
            scales=self.scales,
        )
        self.assertTrue(kelvin.passed)
        with self.assertRaisesRegex(
            SVIDWValidationError, "missing history"
        ):
            material_wake_transport_ledger(
                None,
                current,
                stage_velocity=np.empty((0, 2)),
                scales=self.scales,
            )

    def test_closed_control_volume_and_weak_uk_manufactured_identity(self):
        control_volume = square_control_volume()
        fluid_velocity = np.zeros((4, 2))
        fluid_velocity[1] = (2.0, 0.0)
        vorticity = np.array((0.0, 3.0, 0.0, 0.0))
        flux = te_control_volume_vorticity_flux(
            control_volume,
            vorticity=vorticity,
            fluid_velocity=fluid_velocity,
            control_volume_velocity=np.zeros((4, 2)),
            provenance="manufactured",
            scales=self.scales,
        )
        self.assertAlmostEqual(flux.total_flux, 6.0)

        circulation = BoundCirculationStep2D(
            stage_id="stage-1",
            previous=1.0,
            current=1.6,
            timestep=0.2,
        )
        pressure = TEBernoulliPressureJump2D(
            stage_id="stage-1",
            pressure_lower=10.8,
            pressure_upper=0.0,
            density=1.2,
            provenance="manufactured",
        )
        ledger = weak_unsteady_kutta_ledger(
            circulation,
            flux,
            pressure,
            scales=self.scales,
        )
        self.assertTrue(ledger.passed)
        self.assertAlmostEqual(ledger.bound_circulation_rate, 3.0)
        self.assertAlmostEqual(ledger.specific_pressure_jump, 9.0)
        self.assertAlmostEqual(ledger.residual, 0.0)

    def test_transverse_momentum_is_an_independent_signed_guard(self):
        control_volume = square_control_volume()
        pressure = np.zeros(4)
        pressure[2] = 2.0
        traction = np.zeros((4, 2))
        traction[2] = (0.0, 0.5)
        ledger = te_transverse_momentum_ledger(
            control_volume,
            density=1.2,
            previous_area_momentum=(0.0, 0.0),
            current_area_momentum=(0.0, 0.0),
            timestep=0.1,
            fluid_velocity=np.zeros((4, 2)),
            control_volume_velocity=np.zeros((4, 2)),
            pressure=pressure,
            viscous_traction=traction,
            forming_angle_rad=0.0,
            scales=self.scales,
            tolerance=1.0,
        )
        np.testing.assert_allclose(
            ledger.pressure_flux, (0.0, 2.0)
        )
        np.testing.assert_allclose(
            ledger.viscous_flux, (0.0, 0.5)
        )
        self.assertAlmostEqual(ledger.transverse_residual, 1.5)
        self.assertAlmostEqual(
            ledger.scaled_transverse_residual,
            1.5 / self.scales.force_per_span,
        )

    def test_fail_closed_for_nonclosed_nonfinite_or_mismatched_inputs(self):
        with self.assertRaisesRegex(SVIDWValidationError, "closed"):
            TEControlVolume2D(
                control_volume_id="open",
                stage_id="stage-1",
                vertices=((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)),
            )
        with self.assertRaises(SVIDWValidationError):
            WeakUKReferenceScales2D(
                chord=1.0, velocity=np.nan, density=1.2
            )
        with self.assertRaisesRegex(
            SVIDWValidationError, "unsteady_bernoulli"
        ):
            TEBernoulliPressureJump2D(
                stage_id="stage-1",
                pressure_lower=1.0,
                pressure_upper=0.0,
                density=1.2,
                provenance="weak_uk_backsolve",
            )

        control_volume = square_control_volume()
        flux = te_control_volume_vorticity_flux(
            control_volume,
            vorticity=np.zeros(4),
            fluid_velocity=np.zeros((4, 2)),
            control_volume_velocity=np.zeros((4, 2)),
            provenance="manufactured",
            scales=self.scales,
        )
        circulation = BoundCirculationStep2D(
            stage_id="stage-1",
            previous=0.0,
            current=0.0,
            timestep=0.1,
        )
        wrong_stage_pressure = TEBernoulliPressureJump2D(
            stage_id="stage-2",
            pressure_lower=0.0,
            pressure_upper=0.0,
            density=1.2,
        )
        with self.assertRaisesRegex(SVIDWValidationError, "same"):
            weak_unsteady_kutta_ledger(
                circulation,
                flux,
                wrong_stage_pressure,
                scales=self.scales,
            )
        with self.assertRaises(TypeError):
            weak_unsteady_kutta_ledger(
                circulation,
                flux,
                TEBernoulliPressureJump2D(
                    stage_id="stage-1",
                    pressure_lower=0.0,
                    pressure_upper=0.0,
                    density=1.2,
                    provenance="manufactured",
                ),
                scales=self.scales,
                target_force=0.0,
            )
        with self.assertRaises(TypeError):
            weak_unsteady_kutta_ledger(
                circulation,
                flux,
                TEBernoulliPressureJump2D(
                    stage_id="stage-1",
                    pressure_lower=0.0,
                    pressure_upper=0.0,
                    density=1.2,
                    provenance="manufactured",
                ),
                scales=self.scales,
                endpoint_gamma=0.0,
            )
        with self.assertRaisesRegex(
            SVIDWValidationError, "Gamma_birth/dt"
        ):
            te_control_volume_vorticity_flux(
                control_volume,
                vorticity=np.zeros(4),
                fluid_velocity=np.zeros((4, 2)),
                control_volume_velocity=np.zeros((4, 2)),
                provenance="gamma_birth_over_dt",
                scales=self.scales,
            )


if __name__ == "__main__":
    unittest.main()
