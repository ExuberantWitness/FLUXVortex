"""Frozen identity tests for the Rojratsirikul 2011 digitized oracle package.

G0 of HANDOFF_USE_FIG06_09_12_15_DIGITIZED_DATA_20260829: source integrity,
row counts, key uniqueness, finiteness, exact-match lookup semantics, the
Figure 13/15 St* closure, the frozen anchors, and the documented A10 Figure 9
correction.  CPU-only (data contract, no solver).
"""

from __future__ import annotations

import math

import pytest

from fluxvortex.cases import rojratsirikul2011_observations as obs


def test_g0_validate_passes_and_hashes_frozen() -> None:
    observed = obs.validate()
    assert observed == obs.FROZEN_SHA256


def test_frozen_row_counts() -> None:
    rows6 = obs.figure6_rows()
    rows9 = obs.figure9_rows()
    rows1315 = obs.figure1315_rows()
    assert len(rows6) == obs.FROZEN_ROW_COUNTS["figure06"] == 78
    assert len(rows9) == obs.FROZEN_ROW_COUNTS["figure09"] == 186
    assert len(rows1315) == obs.FROZEN_ROW_COUNTS["figure13_15"] == 35
    # 34 crosscheck curve rows + the single Re=48,700 alpha=15 anchor.
    fc, psd, _ = obs.figure12_spectrum()
    assert len(fc) == len(psd) == obs.FROZEN_ROW_COUNTS["figure12"] == 301
    assert len(obs.figure14_relation()) == obs.FROZEN_ROW_COUNTS["figure14"]


def test_a16_frozen_anchors() -> None:
    zmax = obs.figure6_value(5.0, 16.0)
    cn = obs.figure9_value("flexible_membrane", 5.0, 16.0)
    assert zmax.zmax_over_c == pytest.approx(obs.A16_U5_ZMAX_OVER_C)
    assert cn.cn == pytest.approx(obs.A16_U5_CN)


def test_a10_figure9_correction_is_documented() -> None:
    # Legacy sparse CSV said 0.50-0.52; the full-curve oracle reads 0.5569.
    cn = obs.figure9_value("flexible_membrane", 5.0, 10.0)
    assert cn.cn == pytest.approx(obs.A10_U5_CN, abs=1e-4)
    assert not (0.50 <= cn.cn <= 0.52)


def test_exact_match_lookup_forbids_neighbour_substitution() -> None:
    with pytest.raises(KeyError):
        obs.figure6_value(5.0, 16.5)  # not a sampled angle
    with pytest.raises(KeyError):
        obs.figure9_value("flexible_membrane", 6.0, 16.0)  # not a sampled speed
    with pytest.raises(KeyError):
        obs.figure9_value("nonexistent_wing", 5.0, 16.0)


def test_figure1315_st_modified_closure() -> None:
    for row in obs.figure1315_rows():
        closure = abs(row.st_modified - row.st * math.sin(math.radians(row.alpha_deg)))
        assert closure <= obs.ST_MODIFIED_CLOSURE_TOL


def test_figure12_anchor_condition() -> None:
    _, _, condition = obs.figure12_spectrum()
    assert condition == {"AR": 2, "alpha_deg": 15.0, "Re": 48700}
    assert obs.FIGURE12_PEAK_ST == pytest.approx(0.58, abs=1e-9)
    assert obs.FIGURE12_PEAK_UNCERTAINTY == pytest.approx(0.02)


def test_figure14_is_diagnostic_only() -> None:
    relation = obs.figure14_relation()
    for row in relation:
        assert row["st_lower"] < row["st_fit"] < row["st_upper"]


def test_figure6_fig9_key_uniqueness_already_enforced_by_validate() -> None:
    # validate() raises on duplicates; reaching here means keys are unique.
    obs.validate()
