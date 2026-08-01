import csv
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from score_riziotis_fig12 import (  # noqa: E402
    NRMSE_LIMIT,
    RowKey,
    ScoringInputError,
    _percentile,
    load_candidate,
    load_frozen_reference,
    main,
    score_candidate,
)


def _candidate_rows(
    *,
    use_row_key: bool = False,
) -> list[dict[str, str]]:
    reference = load_frozen_reference()
    rows = []
    for key in sorted(reference):
        point = reference[key]
        row = {
            "x_over_c": f"{point.x_over_c:.12g}",
            "cp_candidate": f"{point.cp:.12g}",
        }
        if use_row_key:
            row["row_key"] = key.stable
        else:
            row.update(
                {
                    "panel": key.panel,
                    "surface": key.surface,
                    "point_index": str(key.point_index),
                }
            )
        rows.append(row)
    return rows


def _write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    fields = tuple(rows[0])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


class RiziotisFigure12ScorerTests(unittest.TestCase):
    def test_perfect_structured_candidate_passes_all_16_groups(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.csv"
            _write_rows(candidate, _candidate_rows())
            report = score_candidate(candidate)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["passed"])
        self.assertEqual(report["summary"]["passed_groups"], 16)
        self.assertEqual(report["summary"]["maximum_nrmse"], 0.0)
        self.assertEqual(len(report["groups"]), 16)

    def test_stable_row_key_candidate_is_equivalent(self):
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.csv"
            _write_rows(candidate, _candidate_rows(use_row_key=True))
            report = score_candidate(candidate)
        self.assertEqual(report["status"], "PASS")
        self.assertTrue(report["passed"])

    def test_one_failed_surface_cannot_be_hidden_by_aggregate(self):
        reference = load_frozen_reference()
        values = [
            point.cp
            for key, point in reference.items()
            if key.panel == "a" and key.surface == "upper"
        ]
        robust_range = _percentile(values, 0.95) - _percentile(values, 0.05)
        imposed_nrmse = NRMSE_LIMIT + 0.01
        rows = _candidate_rows()
        for row in rows:
            if row["panel"] == "a" and row["surface"] == "upper":
                row["cp_candidate"] = str(
                    float(row["cp_candidate"]) + imposed_nrmse * robust_range
                )
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.csv"
            _write_rows(candidate, rows)
            report = score_candidate(candidate)
        self.assertEqual(report["status"], "FAIL")
        self.assertFalse(report["passed"])
        self.assertEqual(report["summary"]["passed_groups"], 15)
        failed = [group for group in report["groups"] if not group["passed"]]
        self.assertEqual(
            [(group["panel"], group["surface"]) for group in failed],
            [("a", "upper")],
        )
        self.assertAlmostEqual(failed[0]["nrmse"], imposed_nrmse)

    def test_missing_duplicate_nonfinite_and_wrong_x_fail_closed(self):
        cases = {}

        missing = _candidate_rows()
        missing.pop()
        cases["missing"] = (missing, "missing 1 frozen rows")

        duplicate = _candidate_rows()
        duplicate.append(dict(duplicate[0]))
        cases["duplicate"] = (duplicate, "duplicate key")

        nonfinite = _candidate_rows()
        nonfinite[0]["cp_candidate"] = "nan"
        cases["nonfinite"] = (nonfinite, "must be finite")

        wrong_x = _candidate_rows()
        wrong_x[0]["x_over_c"] = str(float(wrong_x[0]["x_over_c"]) + 1e-5)
        cases["wrong_x"] = (wrong_x, "interpolation/extrapolation is forbidden")

        for name, (rows, message) in cases.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                candidate = Path(directory) / "candidate.csv"
                _write_rows(candidate, rows)
                with self.assertRaisesRegex(ScoringInputError, message):
                    load_candidate(candidate, load_frozen_reference())

    def test_structured_and_stable_keys_must_agree(self):
        rows = _candidate_rows()
        for row in rows:
            row["row_key"] = RowKey(
                row["panel"],
                row["surface"],
                int(row["point_index"]),
            ).stable
        rows[0]["row_key"] = RowKey("h", "upper", 0).stable
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.csv"
            _write_rows(candidate, rows)
            with self.assertRaisesRegex(ScoringInputError, "disagrees"):
                load_candidate(candidate, load_frozen_reference())

    def test_cli_emits_machine_json_and_input_error_exit_code(self):
        rows = _candidate_rows()
        rows[0]["x_over_c"] = "inf"
        with tempfile.TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate.csv"
            _write_rows(candidate, rows)
            stdout = StringIO()
            stderr = StringIO()
            with redirect_stdout(stdout), redirect_stderr(stderr):
                exit_code = main([str(candidate)])
        self.assertEqual(exit_code, 2)
        report = json.loads(stdout.getvalue())
        self.assertEqual(report["status"], "INPUT_ERROR")
        self.assertFalse(report["passed"])
        self.assertIn("INPUT_ERROR", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
