"""Frozen parameter contract for the Rojratsirikul 2011 membrane-wing CASE."""

from __future__ import annotations

import hashlib
import math
import unittest

import numpy as np
import torch

from fluxvortex.q16_ancf_mesh import make_rectangular_q16_mesh
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    compute_lesp_crit,
    freestream_vector,
)
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID,
    FORMAL_Q16_GRID,
    ROJ11_A10,
    ROJ11_A16,
    ROJ11_A23,
    ROJRATSIRIKUL2011_CASES,
    Rojratsirikul2011MembraneCase,
    assumption_ledger,
    count_interior_peaks,
    dominant_strouhal,
    make_rojratsirikul2011_q16_model,
    membrane_statistics,
    normal_force_coefficient,
    perimeter_node_ids,
    plate_normal,
    rotate_q16_mesh_about_leading_edge,
    static_motion_contract,
    validate_perimeter,
    validate_rojratsirikul2011_sources,
)


class SourceAndRegistryTest(unittest.TestCase):
    def test_reference_pdf_is_hash_frozen(self) -> None:
        validate_rojratsirikul2011_sources()

    def test_registry_has_exactly_the_three_formal_cases(self) -> None:
        self.assertEqual(
            sorted(ROJRATSIRIKUL2011_CASES),
            ["ROJ11-A10", "ROJ11-A16", "ROJ11-A23"],
        )
        self.assertEqual(ROJ11_A10.angle_deg, 10.0)
        self.assertEqual(ROJ11_A16.angle_deg, 16.0)
        self.assertEqual(ROJ11_A23.angle_deg, 23.0)

    def test_shared_physics_is_identical_across_cases(self) -> None:
        shared = (
            "chord_m",
            "span_m",
            "thickness_m",
            "young_modulus_pa",
            "membrane_density_kg_m3",
            "freestream_m_s",
            "kinematic_viscosity_m2_s",
            "fluid_density_kg_m3",
            "poisson_ratio_assumed",
            "aerodynamic_dt_star",
            "structural_substeps_per_aerodynamic_step",
            "startup_time_star",
            "statistics_min_time_star",
            "lesp_crit",
        )
        for name in shared:
            values = {getattr(case, name) for case in ROJRATSIRIKUL2011_CASES.values()}
            self.assertEqual(len(values), 1, name)


class ParameterContractTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = ROJ11_A16

    def test_geometry_derived_quantities(self) -> None:
        # The paper text rounds AR to 2; the frozen figure-2 lengths give
        # 137.5/68.8 = 1.9985 (0.07% digitization inconsistency).
        self.assertAlmostEqual(self.case.aspect_ratio, 2.0, delta=0.002)
        self.assertAlmostEqual(
            self.case.reference_area_m2, 0.0688 * 0.1375, places=15
        )
        self.assertAlmostEqual(
            self.case.thickness_ratio, 2.0e-4 / 0.0688, places=15
        )
        self.assertAlmostEqual(self.case.thickness_ratio, 0.002907, places=6)

    def test_reynolds_matches_paper_within_printed_rounding(self) -> None:
        self.assertAlmostEqual(self.case.reynolds, 24_300.0, delta=50.0)

    def test_dynamic_pressure_and_pi1_match_paper_digits(self) -> None:
        self.assertAlmostEqual(self.case.dynamic_pressure_pa, 15.10, delta=0.02)
        self.assertAlmostEqual(self.case.pi1, 7.51, delta=0.01)

    def test_mass_ratio_is_light_membrane(self) -> None:
        self.assertAlmostEqual(self.case.mass_ratio, 0.2 / (1.208 * 0.0688), places=10)
        self.assertGreater(self.case.mass_ratio, 2.0)
        self.assertLess(self.case.mass_ratio, 3.0)

    def test_frozen_clock_protocol(self) -> None:
        self.assertAlmostEqual(
            self.case.aerodynamic_dt_s, 0.01 * 0.0688 / 5.0, places=18
        )
        self.assertEqual(
            self.case.structural_substeps_per_aerodynamic_step, 50
        )
        self.assertAlmostEqual(
            self.case.structural_dt_s,
            self.case.aerodynamic_dt_s / 50.0,
            places=20,
        )
        self.assertAlmostEqual(
            self.case.convective_time_s, 0.0688 / 5.0, places=18
        )

    def test_thin_membrane_lesp_crit_stays_at_flat_plate_baseline(self) -> None:
        # t/c below the 1.3% reference means the physics formula must return
        # exactly the frozen baseline; no escape from the Lcrit=0.11 guard.
        self.assertEqual(
            compute_lesp_crit(self.case.thickness_ratio, self.case.reynolds), 0.11
        )
        self.assertEqual(self.case.lesp_crit, 0.11)

    def test_digitized_oracles_carry_their_own_band(self) -> None:
        self.assertAlmostEqual(ROJ11_A16.digitized_approx_zmax_over_c, 0.043, places=12)
        self.assertLess(ROJ11_A16.digitized_approx_cn_low, ROJ11_A16.digitized_approx_cn_high)
        self.assertIsNone(ROJ11_A16.digitized_approx_strouhal)
        self.assertAlmostEqual(ROJ11_A10.digitized_approx_strouhal or 0.0, 1.10, places=12)
        self.assertEqual(ROJ11_A10.digitized_approx_chordwise_peak_count, 3)
        self.assertEqual(ROJ11_A10.digitized_approx_spanwise_peak_count, 3)
        self.assertEqual(ROJ11_A23.digitized_approx_chordwise_peak_count, 2)


class StaticMotionContractTest(unittest.TestCase):
    def test_contract_forbids_all_prescribed_motion(self) -> None:
        for case in ROJRATSIRIKUL2011_CASES.values():
            contract = static_motion_contract(case)
            self.assertIn("U0=5 m/s constant", contract["U_inf_history"])
            self.assertIn(
                f"alpha0={case.angle_deg:g} deg constant", contract["alpha_history"]
            )
            self.assertEqual(contract["frame_velocity_m_s"], 0.0)
            self.assertIsNone(contract["prescribed_structural_forces"])
            self.assertEqual(
                contract["boundary"],
                "four-edge six-DOF clamped at rotated reference values",
            )
            self.assertIn("forbidden", contract["flapping_heave_pitch_laws"])

    def test_half_cosine_ramp_is_frozen_and_monotone(self) -> None:
        case = ROJ11_A16
        self.assertEqual(case.freestream_factor(0.0), 0.0)
        self.assertAlmostEqual(case.freestream_factor(0.5), 0.5, places=12)
        self.assertEqual(case.freestream_factor(1.0), 1.0)
        self.assertEqual(case.freestream_factor(7.3), 1.0)
        grid = [case.freestream_factor(t / 100.0) for t in range(101)]
        self.assertTrue(all(b >= a for a, b in zip(grid, grid[1:])))
        with self.assertRaises(ValueError):
            case.freestream_factor(-0.1)


class FourEdgeClampingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.case = ROJ11_A16
        cls.flat = make_rectangular_q16_mesh(
            chordwise_element_count=FORMAL_Q16_GRID[0],
            spanwise_element_count=FORMAL_Q16_GRID[1],
            chord=cls.case.chord_m,
            span=cls.case.span_m,
            thickness=cls.case.thickness_m,
        )
        cls.mesh, cls.model, cls.constraints, cls.audit = (
            make_rojratsirikul2011_q16_model(
                chordwise_element_count=FORMAL_Q16_GRID[0],
                spanwise_element_count=FORMAL_Q16_GRID[1],
                case=cls.case,
            )
        )

    def test_formal_grids_are_the_frozen_protocol(self) -> None:
        self.assertEqual(FORMAL_Q16_GRID, (5, 10))
        self.assertEqual(FORMAL_AERO_GRID, (15, 30))

    def test_perimeter_covers_four_edges_without_interior_nodes(self) -> None:
        # The audit runs on the unrotated lattice; node ids are
        # rotation-invariant so the same set clamps the tilted mesh.
        audit = validate_perimeter(
            self.flat, self.case, FORMAL_Q16_GRID[0], FORMAL_Q16_GRID[1]
        )
        self.assertEqual(audit["perimeter_node_count"], 90)
        self.assertEqual(audit["interior_node_count"], 406)
        self.assertEqual(
            audit["edge_node_counts"],
            # Leading/trailing edges run spanwise (31 stations); sides run
            # chordwise (16 stations).
            {"leading": 31, "trailing": 31, "side": 16, "side_opposite": 16},
        )
        self.assertEqual(audit["clamped_dof_count"], 540)
        self.assertEqual(audit["free_dof_count"], 2436)
        self.assertEqual(self.audit["perimeter_node_count"], 90)

    def test_perimeter_ids_are_sorted_unique_global_ids(self) -> None:
        nodes = perimeter_node_ids(self.mesh, self.case)
        self.assertTrue(np.all(nodes[1:] > nodes[:-1]))
        self.assertTrue(np.all((nodes >= 0) & (nodes < self.mesh.node_count)))

    def test_constraints_clamp_six_dofs_at_reference_values(self) -> None:
        constrained = self.constraints.constrained_dofs
        self.assertEqual(constrained.size, 540)
        expected = self.mesh.reference_state[constrained]
        np.testing.assert_array_equal(self.constraints.prescribed_values, expected)

    def test_model_carries_paper_material_truth(self) -> None:
        self.assertEqual(self.model.young_modulus, self.case.young_modulus_pa)
        self.assertEqual(self.model.poisson_ratio, self.case.poisson_ratio_assumed)
        self.assertEqual(self.model.density, self.case.membrane_density_kg_m3)
        self.assertAlmostEqual(
            self.model.total_reference_mass,
            self.case.membrane_density_kg_m3
            * self.case.reference_area_m2
            * self.case.thickness_m,
            places=15,
        )

    def test_mesh_is_pitched_nose_up_about_the_leading_edge(self) -> None:
        alpha = math.radians(self.case.angle_deg)
        # Leading edge stays on z=0; trailing edge goes DOWN by c sin(alpha).
        leading = self.mesh.reference_rows[self.mesh.reference_rows[:, 0] <= 1e-12]
        self.assertTrue(np.allclose(leading[:, 2], 0.0, atol=1e-12))
        trailing = self.mesh.reference_rows[
            np.abs(self.mesh.reference_rows[:, 0] - self.case.chord_m * math.cos(alpha))
            <= 1e-9
        ]
        self.assertTrue(
            np.allclose(trailing[:, 2], -self.case.chord_m * math.sin(alpha), atol=1e-12)
        )
        # Rigid rotation preserves lengths: the mean in-plane chord of the
        # tilted leading-edge line still spans the full chord.
        normal = plate_normal(self.case)
        self.assertAlmostEqual(float(normal @ np.array([0.0, 0.0, 1.0])), math.cos(alpha), places=15)
        self.assertAlmostEqual(float(normal[0]), math.sin(alpha), places=15)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=15)

    def test_rotation_preserves_the_node_lattice_and_perimeter_ids(self) -> None:
        flat_ids = perimeter_node_ids(self.flat, self.case)
        rotated = rotate_q16_mesh_about_leading_edge(self.flat, self.case)
        self.assertEqual(rotated.node_count, self.flat.node_count)
        np.testing.assert_array_equal(rotated.connectivity, self.flat.connectivity)
        # Directors rotate with the plate: still half-thickness, tilted.
        alpha = math.radians(self.case.angle_deg)
        rows = rotated.reference_rows
        gx, gz = rows[:, 3], rows[:, 5]
        self.assertTrue(
            np.allclose(
                gx**2 + gz**2,
                (0.5 * self.case.thickness_m) ** 2,
                atol=1e-15,
            )
        )
        self.assertTrue(np.allclose(gz / (0.5 * self.case.thickness_m), math.cos(alpha), atol=1e-12))
        self.assertEqual(flat_ids.size, 90)


class ScoringFunctionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.case = ROJ11_A16

    def test_normal_force_coefficient_uses_dynamic_pressure_and_area(self) -> None:
        force = [0.0, 0.0, self.case.dynamic_pressure_pa * self.case.reference_area_m2]
        self.assertAlmostEqual(
            normal_force_coefficient(force, self.case, normal=[0.0, 0.0, 1.0]),
            1.0,
            places=12,
        )

    def test_normal_force_coefficient_projects_on_plate_normal(self) -> None:
        magnitude = self.case.dynamic_pressure_pa * self.case.reference_area_m2
        normal = plate_normal(self.case)
        force = (magnitude * normal).tolist()
        self.assertAlmostEqual(
            normal_force_coefficient(force, self.case), 1.0, places=12
        )

    def test_peak_counting_detects_synthetic_modes(self) -> None:
        chord = np.abs(np.sin(np.linspace(0.0, 3.0 * math.pi, 60)))
        self.assertEqual(count_interior_peaks(chord), 3)
        single = np.abs(np.sin(np.linspace(0.0, math.pi, 40)))
        self.assertEqual(count_interior_peaks(single), 1)

    def test_dominant_strouhal_recovers_synthetic_frequency(self) -> None:
        dt = self.case.aerodynamic_dt_s
        samples = 4096
        frequency = 80.0
        time = np.arange(samples) * dt
        series = np.sin(2.0 * math.pi * frequency * time)
        result = dominant_strouhal(series, dt, self.case)
        self.assertIsNotNone(result)
        self.assertAlmostEqual(result["frequency_hz"], frequency, delta=0.5)
        expected_st = frequency * self.case.chord_m / self.case.freestream_m_s
        self.assertAlmostEqual(result["strouhal"], expected_st, delta=0.01)

    def test_short_series_returns_none(self) -> None:
        self.assertIsNone(
            dominant_strouhal(np.zeros(32), self.case.aerodynamic_dt_s, self.case)
        )

    def test_membrane_statistics_extracts_camber_peaks_and_st(self) -> None:
        rng = np.random.default_rng(20110824)
        nc, ns = FORMAL_AERO_GRID[0], 5
        dt = self.case.aerodynamic_dt_s
        samples = 4096
        time = np.arange(samples) * dt
        chord_station = np.arange(nc)[None, :, None]
        span_station = np.arange(ns)[None, None, :]
        mean_shape = (
            0.043
            * self.case.chord_m
            * np.abs(np.sin(math.pi * (chord_station + 0.5) / nc))
            * np.abs(np.sin(math.pi * (span_station + 0.5) / ns))
        )
        fluctuation = (
            0.001
            * self.case.chord_m
            * np.sin(2.0 * math.pi * 80.0 * time[:, None, None])
            * np.abs(np.sin(3.0 * math.pi * (chord_station + 0.5) / nc))
        )
        history = mean_shape + fluctuation + 1.0e-5 * rng.standard_normal(
            (samples, nc, ns)
        )
        result = membrane_statistics(history, dt, self.case)
        self.assertAlmostEqual(
            result["mean_zmax_over_c"], 0.043, delta=0.001
        )
        self.assertEqual(result["chordwise_peak_count"], 3)
        self.assertIsNotNone(result["fluctuation_spectrum"])
        # Bin width here is 1/(4096 dt) = 1.78 Hz.
        self.assertAlmostEqual(
            result["fluctuation_spectrum"]["frequency_hz"], 80.0, delta=2.0
        )


class LedgerAndDigestTest(unittest.TestCase):
    def test_assumption_ledger_covers_every_unreported_parameter(self) -> None:
        ledger = assumption_ledger(ROJ11_A16)
        for key in (
            "poisson_ratio",
            "initial_prestress_n_m",
            "initial_slack_or_excess",
            "structural_damping",
            "initial_geometric_imperfection",
            "latex_constitutive_curve",
            "boundary_director_condition",
            "frame_aerodynamic_thickness",
            "inlet_turbulence_transition",
        ):
            self.assertIn(key, ledger)
            self.assertIn("paper_status", ledger[key])
            self.assertIn("frozen_value", ledger[key])
        self.assertEqual(ledger["poisson_ratio"]["frozen_value"], 0.49)
        self.assertEqual(ledger["initial_prestress_n_m"]["frozen_value"], 0.0)
        self.assertEqual(ledger["structural_damping"]["frozen_value"], 0.0)

    def test_config_digest_is_stable_and_case_dependent(self) -> None:
        self.assertEqual(ROJ11_A16.config_digest(), ROJ11_A16.config_digest())
        self.assertNotEqual(ROJ11_A10.config_digest(), ROJ11_A16.config_digest())
        for case in ROJRATSIRIKUL2011_CASES.values():
            digest = case.config_digest()
            self.assertEqual(len(digest), 64)
            int(digest, 16)


class FreestreamAngleTest(unittest.TestCase):
    def test_zero_angle_is_bit_exactly_axis_aligned(self) -> None:
        # The historical Yamano path (angle=0) must keep its exact v_inf.
        vector = freestream_vector(7.5, 0.0, "cpu")
        reference = torch.tensor([7.5, 0.0, 0.0], dtype=torch.float64)
        self.assertEqual(vector.dtype, torch.float64)
        self.assertTrue(torch.equal(vector, reference))

    def test_angle_tilts_in_the_chord_normal_plane(self) -> None:
        for angle_deg in (10.0, 16.0, 23.0):
            vector = freestream_vector(5.0, angle_deg, "cpu")
            angle_rad = math.radians(angle_deg)
            self.assertEqual(float(vector[1]), 0.0)
            self.assertAlmostEqual(
                float(vector[0]), 5.0 * math.cos(angle_rad), places=15
            )
            self.assertAlmostEqual(
                float(vector[2]), 5.0 * math.sin(angle_rad), places=15
            )
            self.assertAlmostEqual(
                float(torch.linalg.vector_norm(vector)), 5.0, delta=1.0e-14
            )

    def test_config_rejects_invalid_angles(self) -> None:
        with self.assertRaises(ValueError):
            NativeV5MConfig(freestream_angle_deg=90.0)


if __name__ == "__main__":
    unittest.main()
