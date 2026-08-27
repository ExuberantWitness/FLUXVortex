"""Regression: bound_rate wake history uses the ONE-step-old circulation.

External review (2026-08-27) confirmed an implementation-contract bug in the
``wake_history_mode="bound_rate"`` branch of
``warp_fsi/q16_flux_v5m_native.py`` ``propose()``: it read
``trial.gamma_previous``, which the state-update ordering
(``gamma_previous = gamma_bound`` BEFORE ``gamma_bound = gamma``) leaves at
Gamma_(n-2) — TWO steps old.  The author's weak-scheme dp_add
(calc_fluid_force.m) is the one-step difference (Gamma_n - Gamma_(n-1))/dt;
reading the two-step-old value doubled the derivative's amplitude and shifted
its phase.

These tests step the native solver three times and require the recorded
mf2_history of steps 2 and 3 to equal the ONE-step difference exactly and to
differ grossly from the two-step difference the bug produced.
"""
from __future__ import annotations

import os
import unittest

import numpy as np
import torch
import warp as wp

from fluxvortex.warp_fsi import config
from fluxvortex.warp_fsi.q16_flux_v5m_native import (
    NativeV5MConfig,
    Q16NativeV5MSolver,
    Q16NativeV5MSurface,
)
from forward_flight_benchmarks.yamano2020_q16 import (
    YAMANO_2020_SINGLE_SHEET,
    make_yamano2020_q16_model,
)

DEVICE = os.environ.get("FLUXV_DEVICE", "cuda:0")
CUDA_OK = torch.cuda.is_available() and wp.is_cuda_available()


@unittest.skipUnless(CUDA_OK, "CUDA required")
class BoundRateOneStepGammaHistoryTest(unittest.TestCase):
    """mf2_history must be (Gamma_n - Gamma_(n-1))/dt, never (n - n-2)/dt."""

    @classmethod
    def setUpClass(cls):
        cls.mesh, _, _ = make_yamano2020_q16_model(
            chordwise_element_count=5, spanwise_element_count=3
        )
        cls.surface = Q16NativeV5MSurface(
            cls.mesh,
            q16_chordwise_elements=5,
            q16_spanwise_elements=3,
            aerodynamic_chordwise_panels=15,
            aerodynamic_spanwise_panels=10,
            device=config.DEVICE,
        )
        cls.solver = Q16NativeV5MSolver(
            cls.surface,
            NativeV5MConfig(
                density=YAMANO_2020_SINGLE_SHEET.fluid_density_kg_m3,
                freestream=YAMANO_2020_SINGLE_SHEET.freestream_m_s,
                aerodynamic_dt=YAMANO_2020_SINGLE_SHEET.aerodynamic_dt_s,
                # Nonzero angle so the bound circulation is nonzero and keeps
                # evolving as the wake builds (no LEV release at this angle).
                freestream_angle_deg=5.0,
                wake_history_mode="bound_rate",
                device=config.DEVICE,
            ),
        )
        cls.state_wp = wp.array(
            np.ascontiguousarray(cls.mesh.reference_state[None, :]),
            dtype=config.DTYPE,
            device=config.DEVICE,
        )
        cls.zero_wp = wp.zeros_like(cls.state_wp)
        cls.dt = cls.solver.settings.aerodynamic_dt

    def test_one_step_difference_at_every_step(self):
        committed = self.solver.initialize(self.state_wp, self.zero_wp)
        gamma_zero = committed.gamma_bound.clone()
        self.assertEqual(float(gamma_zero.abs().max().item()), 0.0)

        history = [gamma_zero]
        for step in (1, 2, 3):
            proposal = self.solver.propose(committed, self.state_wp, self.zero_wp)
            gamma_n = proposal.trial_state.gamma_bound
            gamma_previous = history[-1]
            gamma_two_back = history[-2] if len(history) >= 2 else gamma_zero

            mf2 = proposal.author_load.mf2_history
            one_step = (gamma_n - gamma_previous) / self.dt
            # The recorded history IS the one-step difference, exactly (same
            # tensors inside the same propose() call).
            self.assertTrue(
                torch.equal(mf2, one_step),
                f"step {step}: mf2_history is not the one-step gamma difference",
            )
            # The gate must discriminate: the mf2 values are nonzero and, from
            # step 2 on, the one/two-step candidates differ grossly (the
            # pre-fix code read gamma_previous = Gamma_(n-2) and produced the
            # two-step value).
            self.assertGreater(
                float(mf2.abs().max().item()),
                1.0e-6,
                f"step {step}: mf2_history is zero: gate is vacuous",
            )
            if step >= 2:
                two_step = (gamma_n - gamma_two_back) / self.dt
                self.assertGreater(
                    float((two_step - one_step).abs().max().item()),
                    1.0,
                    f"step {step}: one-step and two-step candidates coincide; "
                    "the gate cannot distinguish them",
                )

            # State bookkeeping pinned: after the step commits, gamma_bound is
            # Gamma_n and gamma_previous is the pre-update Gamma_(n-1).
            self.assertTrue(
                torch.equal(proposal.trial_state.gamma_previous, gamma_previous)
            )
            self.assertTrue(
                torch.equal(proposal.trial_state.gamma_bound, gamma_n)
            )
            committed = proposal.trial_state
            history.append(gamma_n)

        # The gamma sequence itself actually evolved every step (Gamma_1 != 0,
        # Gamma_2 != Gamma_1, Gamma_3 != Gamma_2): otherwise the differences
        # above could not discriminate one-step from two-step history.
        for older, newer in zip(history[:-1], history[1:], strict=True):
            self.assertGreater(
                float((newer - older).abs().max().item()),
                1.0e-6,
                "bound circulation did not evolve between steps",
            )

    def test_state_fields_keep_their_lag_one_semantics(self):
        """gamma_previous lags gamma_bound by exactly one propose()."""
        committed = self.solver.initialize(self.state_wp, self.zero_wp)
        first = self.solver.propose(committed, self.state_wp, self.zero_wp)
        committed = first.trial_state
        self.assertTrue(
            torch.equal(committed.gamma_previous, gamma_zero_like(committed))
        )
        second = self.solver.propose(committed, self.state_wp, self.zero_wp)
        committed = second.trial_state
        # After step 2: gamma_previous == Gamma_1 (one step behind), not 0.
        self.assertTrue(
            torch.equal(committed.gamma_previous, first.trial_state.gamma_bound)
        )
        self.assertFalse(
            torch.equal(committed.gamma_previous, gamma_zero_like(committed))
        )


def gamma_zero_like(state) -> torch.Tensor:
    return torch.zeros_like(state.gamma_bound)


if __name__ == "__main__":
    unittest.main()
