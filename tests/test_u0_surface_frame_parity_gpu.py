"""U0-P parity: SurfaceFrame adapter vs existing Q16NativeV5MSurface.evaluate().

Gate: rings, velocities, normals, areas bit-identical (torch.equal) on the
formal A16 grid with a deformed state; GPU residency unchanged.
"""
import unittest
import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import Q16NativeV5MSurface
from fluxvortex.kinematics.frames import SurfaceFrame
from fluxvortex.kinematics.q16_surface import Q16SurfaceFrameAdapter
from forward_flight_benchmarks.rojratsirikul2011_q16 import (
    FORMAL_AERO_GRID, FORMAL_Q16_GRID, ROJ11_A16,
    make_rojratsirikul2011_q16_model,
)


@unittest.skipUnless(torch.cuda.is_available() and wp.is_cuda_available(), "CUDA required")
class U0SurfaceFrameParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mesh, _, _, _ = make_rojratsirikul2011_q16_model(
            chordwise_element_count=FORMAL_Q16_GRID[0],
            spanwise_element_count=FORMAL_Q16_GRID[1],
            case=ROJ11_A16,
        )
        cls.native = Q16NativeV5MSurface(
            cls.mesh,
            q16_chordwise_elements=FORMAL_Q16_GRID[0],
            q16_spanwise_elements=FORMAL_Q16_GRID[1],
            aerodynamic_chordwise_panels=FORMAL_AERO_GRID[0],
            aerodynamic_spanwise_panels=FORMAL_AERO_GRID[1],
            device=config.DEVICE,
        )
        cls.adapter = Q16SurfaceFrameAdapter(cls.native)
        rng = np.random.default_rng(20260826)
        deformed = np.array(cls.mesh.reference_state, dtype=np.float64, copy=True)
        deformed[1::6] += rng.standard_normal(deformed[1::6].shape[0]) * 1e-3
        deformed[3::6] += rng.standard_normal(deformed[3::6].shape[0]) * 1e-5
        cls.state = wp.array(
            np.ascontiguousarray(deformed[None, :]),
            dtype=config.DTYPE, device=config.DEVICE,
        )
        cls.velocity = wp.array(
            np.ascontiguousarray(
                (rng.standard_normal((1, deformed.size)) * 1e-2).astype(np.float64)
            ),
            dtype=config.DTYPE, device=config.DEVICE,
        )

    def test_bit_identical_geometry(self):
        ref = self.native.evaluate(self.state, self.velocity)
        frame = self.adapter.evaluate(self.state, self.velocity)
        self.assertTrue(torch.equal(frame.panel_rings_I, ref.rings))
        self.assertTrue(torch.equal(frame.panel_ring_velocity_I, ref.ring_velocity))
        self.assertTrue(torch.equal(frame.collocation_I, ref.collocation))
        self.assertTrue(torch.equal(frame.collocation_velocity_I, ref.collocation_velocity))
        self.assertTrue(torch.equal(frame.normals_I, ref.normals))
        self.assertTrue(torch.equal(frame.areas, ref.areas))
        self.assertTrue(torch.equal(frame.leading_edge_I, ref.leading_edge))
        self.assertTrue(torch.equal(frame.trailing_edge_I, ref.trailing_edge))
        self.assertTrue(torch.equal(frame.leading_velocity_I, ref.leading_velocity))
        self.assertTrue(torch.equal(frame.trailing_velocity_I, ref.trailing_velocity))

    def test_frame_metadata(self):
        frame = self.adapter.evaluate(self.state, self.velocity)
        self.assertEqual(frame.chordwise_panels, FORMAL_AERO_GRID[0])
        self.assertEqual(frame.spanwise_panels, FORMAL_AERO_GRID[1])
        self.assertEqual(frame.surface_id, "wing_0")

    def test_gpu_residency(self):
        frame = self.adapter.evaluate(self.state, self.velocity)
        for tensor in (frame.panel_rings_I, frame.normals_I, frame.areas,
                       frame.collocation_I):
            self.assertTrue(tensor.is_cuda)
            self.assertEqual(tensor.dtype, torch.float64)

    def test_result_status_exit_code(self):
        from fluxvortex.runtime.result_schema import ResultStatus
        ok = ResultStatus(execution_status="completed", numerical_status="converged",
                          physics_gate_status="passed", accuracy_gate_status="passed",
                          reproduction_status="passed")
        self.assertEqual(ok.exit_code, 0)
        self.assertTrue(ok.is_formal_reproduction)
        acc_fail = ResultStatus(execution_status="completed", numerical_status="converged",
                                physics_gate_status="passed", accuracy_gate_status="failed",
                                reproduction_status="failed")
        self.assertEqual(acc_fail.exit_code, 2)
        self.assertFalse(acc_fail.is_formal_reproduction)


if __name__ == "__main__":
    unittest.main()
