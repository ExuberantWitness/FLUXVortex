"""Reference checks for the existing clean-room Python LDVM primitive.

The numerical landmarks come from the author-supplied LDVM v2.5
``motion_pr_amp45_k0.2.dat`` and ``force_pr_amp45_k0.2_le.dat`` files.  The
Fortran source and its GPL-covered output files are deliberately not vendored.
The test reconstructs the analytic Eldredge motion and the already-vendored
SD7003 mean camber line instead.

This is an event-history parity guard for the explicitly isolated
``source_parity`` mode.  It does not claim bitwise strength parity: the Python
oracle uses a clean linear Kelvin solve while the source uses a finite-tolerance
Newton iteration, and their public wake-sign conventions differ.  See
``SOURCE_AUDIT_RAMESH_LDVM.md`` for the scope of the source audit.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import numpy as np
from scipy.interpolate import CubicSpline


PLATFORM = Path(__file__).resolve().parents[1]
ROOT = PLATFORM.parent
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from ldvm_fourier import LDVM2D  # noqa: E402


AUTHOR_OUTPUT_ROWS = 499
AUTHOR_FIRST_LEV_OUTPUT_ROW = 116
AUTHOR_FIRST_LEV_TSTAR = 1.6383540
AUTHOR_FIRST_LEV_ALPHA_DEG = 14.630000
AUTHOR_LESP_CRIT = 0.18
AUTHOR_FIRST_LEV_CL = 3.0860116
AUTHOR_FIRST_LEV_CD = 0.59517464
AUTHOR_FINAL_LEV_COUNT = 174
AUTHOR_LAST_LEV_OUTPUT_ROW = 289


def _eldredge_reference_motion() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a source-arithmetic clean-room approximation to the motion."""

    # The author's source promotes ``acos(-1.)`` from default real precision.
    # Retaining that arithmetic reproduces every printed time sample without
    # copying the distributed motion file into this clean-room regression.
    source_pi = float(np.float32(np.pi))
    amplitude = 45.0 * source_pi / 180.0
    reduced_pitch_rate = 0.2
    smoothing = 11.0
    t1 = 1.0
    t2 = t1 + amplitude / (2.0 * reduced_pitch_rate)
    t3 = (
        t2
        + source_pi * amplitude / (4.0 * reduced_pitch_rate)
        - amplitude / (2.0 * reduced_pitch_rate)
    )
    t4 = t3 + amplitude / (2.0 * reduced_pitch_rate)

    def eldredge_shape(time: np.ndarray | float) -> np.ndarray:
        time = np.asarray(time)
        return np.log(
            (np.cosh(smoothing * (time - t1)) * np.cosh(smoothing * (time - t4)))
            / (np.cosh(smoothing * (time - t2)) * np.cosh(smoothing * (time - t3)))
        )

    # The supplied motion contains 500 samples from t*=0 through t*=t4+1.
    time_unrounded = np.linspace(0.0, t4 + 1.0, 500)
    shape = eldredge_shape(time_unrounded)
    maximum = float(np.max(shape))
    alpha_deg_unrounded = 45.0 * shape / maximum

    # The .7e quantization mirrors the distributed field precision.  This is
    # not claimed to be a row-identical copy of the unvendored author file: an
    # external audit found 500/500 time values and 493/500 alpha values equal,
    # with the remaining alpha differences <= 1.1e-15 degree.
    time = np.asarray([float(f"{value:.7e}") for value in time_unrounded])
    alpha_deg = np.asarray([float(f"{value:.7e}") for value in alpha_deg_unrounded])
    alpha = alpha_deg * np.pi / 180.0

    # ldvm.f95:181-187 uses first-order backward differences and copies the
    # second sample's derivative into the unused first sample.
    alpha_rate = np.empty_like(alpha)
    alpha_rate[1:] = np.diff(alpha) / np.diff(time)
    alpha_rate[0] = alpha_rate[1]
    return time, alpha, alpha_rate


def _author_sd7003_camber_line(model: LDVM2D) -> tuple[np.ndarray, np.ndarray]:
    """Mirror the source camber ordinate/slope construction for SD7003."""

    coordinates = np.loadtxt(
        ROOT / "researchpaper" / "uiuc_airfoils" / "sd7003.dat",
        skiprows=1,
    )
    surface_coordinate = np.concatenate(
        ([0.0], np.cumsum(np.abs(np.diff(coordinates[:, 0]))))
    )
    surface_y = CubicSpline(
        surface_coordinate,
        coordinates[:, 1],
        bc_type=((1, 0.0), (1, 0.0)),
    )

    x_over_c = model.xs / model.c
    upper_path = surface_y(x_over_c)
    lower_path = surface_y(1.0 + x_over_c)
    camber = 0.5 * (upper_path[::-1] + lower_path) * model.c

    slope = np.empty_like(camber)
    slope[0] = (camber[1] - camber[0]) / (model.xs[1] - model.xs[0])
    slope[1:] = np.diff(camber) / np.diff(model.xs)
    return camber, slope


def test_clean_room_motion_uses_source_precision_and_format_contract() -> None:
    """Guard the reconstruction without claiming archive-row identity."""

    time, alpha, alpha_rate = _eldredge_reference_motion()
    assert time.shape == alpha.shape == alpha_rate.shape == (500,)
    assert time[0] == 0.0
    assert time[-1] == 7.0477470
    assert len(np.unique(np.diff(time))) == 24
    assert np.all(np.isfinite(alpha_rate))
    assert abs(float(np.rad2deg(np.max(alpha))) - 45.0) <= 5.0e-7


@lru_cache(maxsize=1)
def _reference_replay() -> dict[str, np.ndarray | int]:
    time, alpha, alpha_rate = _eldredge_reference_motion()
    model = LDVM2D(
        U=1.0,
        c=1.0,
        ndiv=70,
        naterm=45,
        dt=float(time[1] - time[0]),
        lesp_crit=AUTHOR_LESP_CRIT,
        pivot_xc=0.0,
        core_rc=0.02,
        max_wake=100_000,
        source_parity=True,
    )
    model.set_camber_line(*_author_sd7003_camber_line(model))

    rows: list[list[float]] = []
    kelvin_residuals: list[float] = []
    cl_identity_residuals: list[float] = []
    cd_identity_residuals: list[float] = []
    cs_identity_residuals: list[float] = []

    # The Fortran program advances motion rows 2..500 and therefore emits 499
    # force rows.  ``step`` follows the same no-output-at-initial-state contract.
    for motion_index in range(1, len(time)):
        result = model.step(
            float(alpha[motion_index]),
            float(alpha_rate[motion_index]),
            0.0,
            dt_i=float(time[motion_index] - time[motion_index - 1]),
        )
        output_row = motion_index
        rows.append(
            [
                float(output_row),
                float(time[motion_index]),
                float(np.rad2deg(alpha[motion_index])),
                float(result["A0"]),
                float(result["lesp"]),
                float(result["CNf"]),
                float(result["CSf"]),
                float(result["CLf"]),
                float(result["CDf"]),
                float(result["gamb"]),
                float(result["n_lev"]),
                float(result["n_tev"]),
            ]
        )

        kelvin_residuals.append(
            float(np.sum(model.tg) + np.sum(model.lg) + model.gam_lost - result["gamb"])
        )
        cl_identity_residuals.append(
            float(
                result["CLf"]
                - result["CNf"] * np.cos(alpha[motion_index])
                - result["CSf"] * np.sin(alpha[motion_index])
            )
        )
        cd_identity_residuals.append(
            float(
                result["CDf"]
                - result["CNf"] * np.sin(alpha[motion_index])
                + result["CSf"] * np.cos(alpha[motion_index])
            )
        )
        cs_identity_residuals.append(
            float(result["CSf"] - 2.0 * np.pi * result["A0"] ** 2)
        )

    return {
        "rows": np.asarray(rows),
        "kelvin": np.asarray(kelvin_residuals),
        "cl_identity": np.asarray(cl_identity_residuals),
        "cd_identity": np.asarray(cd_identity_residuals),
        "cs_identity": np.asarray(cs_identity_residuals),
        "final_lev_count": len(model.lg),
    }


def test_author_reference_replay_is_finite() -> None:
    replay = _reference_replay()
    rows = np.asarray(replay["rows"])

    assert rows.shape == (AUTHOR_OUTPUT_ROWS, 12)
    assert np.all(np.isfinite(rows))
    assert np.all(np.diff(rows[:, 10]) >= 0.0)
    assert np.all(np.diff(rows[:, 11]) == 1.0)
    assert rows[-1, 11] == AUTHOR_OUTPUT_ROWS


def test_author_reference_first_lesp_cap_is_reproduced() -> None:
    replay = _reference_replay()
    rows = np.asarray(replay["rows"])
    first_index = int(np.flatnonzero(rows[:, 10] > 0.0)[0])
    first = rows[first_index]

    assert int(first[0]) == AUTHOR_FIRST_LEV_OUTPUT_ROW
    assert np.isclose(first[1], AUTHOR_FIRST_LEV_TSTAR, atol=5.0e-7)
    assert np.isclose(first[2], AUTHOR_FIRST_LEV_ALPHA_DEG, atol=2.0e-6)
    assert first[4] > AUTHOR_LESP_CRIT  # pre-constraint LESP triggers shedding
    assert np.isclose(first[3], AUTHOR_LESP_CRIT, atol=5.0e-14)
    assert np.isclose(first[7], AUTHOR_FIRST_LEV_CL, atol=2.0e-3)
    assert np.isclose(first[8], AUTHOR_FIRST_LEV_CD, atol=6.0e-4)

    assert int(replay["final_lev_count"]) == AUTHOR_FINAL_LEV_COUNT


def test_author_reference_shedding_event_mask_is_exact() -> None:
    replay = _reference_replay()
    rows = np.asarray(replay["rows"])
    lev_counts = rows[:, 10].astype(int)
    birth_mask = np.diff(np.concatenate(([0], lev_counts))) == 1
    birth_rows = rows[birth_mask, 0].astype(int)

    assert np.array_equal(
        birth_rows,
        np.arange(AUTHOR_FIRST_LEV_OUTPUT_ROW, AUTHOR_LAST_LEV_OUTPUT_ROW + 1),
    )
    assert birth_rows.size == AUTHOR_FINAL_LEV_COUNT
    assert rows[-1, 11] == AUTHOR_OUTPUT_ROWS


def test_source_parity_first_tev_is_zeroed_after_solve() -> None:
    model = LDVM2D(source_parity=True, lesp_crit=99.0)
    result = model.step(np.deg2rad(2.0), 0.0)

    assert result["first_tev_zeroed"] is True
    assert result["tev_strength_solved"] != 0.0
    assert result["tev_strength_stored"] == 0.0
    assert model.tg == [0.0]


def test_te_only_provisional_is_frozen_without_changing_legacy_placement() -> None:
    def run_and_capture(
        source_parity: bool,
    ) -> tuple[LDVM2D, dict, list[tuple[float, float]]]:
        model = LDVM2D(source_parity=source_parity, lesp_crit=0.18)
        calls: list[tuple[float, float]] = []
        original = model._wcol

        def capture(px: float, py: float) -> np.ndarray:
            calls.append((float(px), float(py)))
            return original(px, py)

        model._wcol = capture
        result = model.step(np.deg2rad(35.0), 0.0)
        return model, result, calls

    legacy, legacy_result, legacy_calls = run_and_capture(False)
    parity, parity_result, parity_calls = run_and_capture(True)
    assert legacy_result["shed_lev"] is True
    assert parity_result["shed_lev"] is True
    assert len(legacy_calls) == len(parity_calls) == 2
    legacy_edge = legacy._world(0.0)
    np.testing.assert_array_equal(
        legacy_calls[1],
        (legacy_edge[0] + 0.5 * legacy.U * legacy_result["dt_i"], legacy_edge[1]),
    )
    assert parity_calls[1] != legacy_calls[1]
    assert (
        parity_result["tev_strength_te_only_provisional"]
        != parity_result["tev_strength_solved"]
    )
    assert (
        legacy_result["tev_strength_te_only_provisional"]
        == parity_result["tev_strength_te_only_provisional"]
    )


def test_kelvin_and_force_definition_identities_close() -> None:
    replay = _reference_replay()

    # The source deliberately zeros its first solved TEV after the first
    # Kelvin solve.  Kelvin closes again from the following output row onward.
    assert np.max(np.abs(replay["kelvin"][1:])) < 2.0e-12
    assert np.max(np.abs(replay["cl_identity"])) < 2.0e-14
    assert np.max(np.abs(replay["cd_identity"])) < 2.0e-14
    assert np.max(np.abs(replay["cs_identity"])) < 2.0e-14
