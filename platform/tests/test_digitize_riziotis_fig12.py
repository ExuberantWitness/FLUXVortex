import csv
import hashlib
import sys
import tempfile
import unittest
from pathlib import Path


PLATFORM = Path(__file__).resolve().parents[1]
if str(PLATFORM) not in sys.path:
    sys.path.insert(0, str(PLATFORM))

from digitize_riziotis_fig12 import (  # noqa: E402
    CSV_FIELDS,
    DEFAULT_OUTPUT_DIRECTORY,
    DigitizationError,
    OFFICIAL_TAP_X,
    SOURCE_PDF_SHA256,
    SourcePage,
    _csv_rows,
    _phase_radians,
    parse_svg_page,
)


def _diamond_path(x_value: float, y_value: float) -> str:
    return (
        f"M {x_value:.8f} {y_value + 0.9:.8f} "
        f"L {x_value - 0.9:.8f} {y_value:.8f} "
        f"L {x_value:.8f} {y_value - 0.9:.8f} "
        f"L {x_value + 0.9:.8f} {y_value:.8f} Z"
    )


def _synthetic_svg(*, extra_matching_tap: bool = False) -> str:
    x_min, x_max = 100.0, 262.0
    y_min, y_max = 100.0, 226.0
    transform = "matrix(1,0,0,-1,0,300)"
    frame = (
        f"M {x_min} {y_min} L {x_max} {y_min} " f"L {x_max} {y_max} L {x_min} {y_max} Z"
    )
    grids = " ".join(
        f"M {x_min} {102.0 + 15.0 * index} " f"L {x_max} {102.0 + 15.0 * index}"
        for index in range(9)
    )

    lower = [
        (
            x_max - 1.0 - (x_max - 1.0 - x_min) * index / 37.0,
            112.0 + 2.0 * index / 37.0,
        )
        for index in range(38)
    ]
    upper = [
        (
            x_min + (x_max - 1.0 - x_min) * index / 36.0,
            176.0 - 18.0 * index / 36.0,
        )
        for index in range(37)
    ]
    model = " ".join(
        (
            f"M {lower[0][0]:.8f} {lower[0][1]:.8f}",
            *(
                f"L {x_value:.8f} {y_value:.8f}"
                for x_value, y_value in lower[1:] + upper
            ),
        )
    )

    diamonds = [_diamond_path(x_min + 0.92575 * 162.0, 214.0)]
    for tap_x in OFFICIAL_TAP_X:
        svg_x = x_min + tap_x * 162.0
        diamonds.append(_diamond_path(svg_x, 174.0 - 8.0 * tap_x))
        diamonds.append(_diamond_path(svg_x, 111.0 + 4.0 * tap_x))
    if extra_matching_tap:
        diamonds.append(_diamond_path(x_min + 0.5 * 162.0, 142.0))

    return f"""<?xml version="1.0"?>
<svg xmlns="http://www.w3.org/2000/svg">
  <path style="fill:none;stroke-width:0.5;" d="{frame}"
        transform="{transform}"/>
  <path style="fill:none;stroke-width:0.142;" d="{grids}"
        transform="{transform}"/>
  <path style="fill:none;stroke-width:0.284;"
        d="M 245 218 L 255 218 {model} {' '.join(diamonds)}"
        transform="{transform}"/>
</svg>
"""


class RiziotisFigure12DigitizationTests(unittest.TestCase):
    def _parse_synthetic(self, *, extra_matching_tap: bool = False):
        with tempfile.TemporaryDirectory() as directory:
            svg_path = Path(directory) / "figure12.svg"
            svg_path.write_text(
                _synthetic_svg(extra_matching_tap=extra_matching_tap),
                encoding="utf-8",
            )
            source = SourcePage(
                path=svg_path,
                page_number=17,
                sha256=hashlib.sha256(svg_path.read_bytes()).hexdigest(),
                source_name="synthetic.svg",
            )
            return parse_svg_page(source, (("a", "upstroke", 12.0, 17),))

    def test_vector_topology_and_15_taps_per_surface(self):
        panels = self._parse_synthetic()
        self.assertEqual(len(panels), 1)
        panel = panels[0]
        self.assertEqual(panel.model_path_points, 75)
        self.assertEqual(panel.excluded_legend_diamonds, 1)
        self.assertEqual(len(panel.data), 105)
        for surface in ("upper", "lower"):
            model = [
                datum
                for datum in panel.data
                if datum.category == "model" and datum.surface == surface
            ]
            experiment = [
                datum
                for datum in panel.data
                if datum.category == "experiment" and datum.surface == surface
            ]
            self.assertGreater(len(model), 35)
            self.assertEqual(len(experiment), 15)
            self.assertEqual(
                tuple(datum.tap_x_nominal for datum in experiment),
                OFFICIAL_TAP_X,
            )
        rows = _csv_rows(panels)
        self.assertTrue(
            all(float(row["cp"]) == -float(row["minus_cp"]) for row in rows)
        )
        required = {
            "source_id",
            "source_sha256",
            "doi",
            "pdf_page",
            "article_page",
            "figure",
            "panel",
            "series",
            "surface",
            "phase_branch",
            "alpha_label_deg",
            "phase_rad",
            "phase_status",
            "tap_id",
            "x_over_c",
            "minus_cp",
            "cp",
            "vector_path_id",
            "axis_bbox_pt",
            "digitization_sigma_xc",
            "digitization_sigma_cp",
            "extraction_version",
            "exclusion_flag",
            "notes",
        }
        self.assertTrue(required.issubset(CSV_FIELDS))
        experiment_rows = [
            row for row in rows if row["series"] == "experimental_measurements"
        ]
        upper = [row for row in experiment_rows if row["surface"] == "upper"]
        lower = [row for row in experiment_rows if row["surface"] == "lower"]
        self.assertEqual(
            [row["tap_id"] for row in upper],
            [f"X{value}" for value in range(15, 0, -1)],
        )
        self.assertEqual(
            [row["tap_id"] for row in lower], [f"X{value}" for value in range(16, 31)]
        )
        self.assertEqual(float(upper[0]["x_over_c"]), 0.00025)
        self.assertEqual(float(lower[0]["x_over_c"]), 0.00025)
        self.assertEqual(upper[0]["digitization_sigma_xc"], "0.002")
        self.assertEqual(upper[0]["digitization_sigma_cp"], "0.1")
        self.assertEqual(upper[0]["source_sha256"], SOURCE_PDF_SHA256)
        self.assertEqual(rows[0]["axis_bbox_pt"], "[100,74,262,200]")
        self.assertTrue(all(row["vector_path_id"] for row in experiment_rows))
        self.assertEqual(DEFAULT_OUTPUT_DIRECTORY.name, "n26e1_fig12")
        self.assertEqual(DEFAULT_OUTPUT_DIRECTORY.parent.name, "diag")

    def test_duplicate_pressure_tap_fails_closed(self):
        with self.assertRaisesRegex(DigitizationError, "ambiguous measurement vectors"):
            self._parse_synthetic(extra_matching_tap=True)

    def test_reversal_phase_is_intentionally_empty(self):
        self.assertIsNone(_phase_radians("upstroke", 19.0))
        self.assertIsNone(_phase_radians("downstroke", 19.0))
        self.assertAlmostEqual(_phase_radians("upstroke", 15.0), 0.5235987755982988)
        self.assertAlmostEqual(_phase_radians("downstroke", 15.0), 2.6179938779914944)

    def test_persisted_asset_obeys_frozen_source_contract(self):
        csv_path = DEFAULT_OUTPUT_DIRECTORY / "fig12_digitized.csv"
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        self.assertEqual(len(rows), 859)
        self.assertTrue(all(row["source_sha256"] == SOURCE_PDF_SHA256 for row in rows))
        self.assertTrue(all(row["figure"] == "12" for row in rows))
        experiment = [
            row for row in rows if row["series"] == "experimental_measurements"
        ]
        self.assertEqual(len(experiment), 240)
        for panel in "abcdefgh":
            for surface in ("upper", "lower"):
                subset = [
                    row
                    for row in experiment
                    if row["panel"] == panel and row["surface"] == surface
                ]
                self.assertEqual(len(subset), 15)
                self.assertEqual(
                    [float(row["x_over_c"]) for row in subset],
                    list(OFFICIAL_TAP_X),
                )
        leading_edge = [row for row in experiment if row["x_over_c"] == "0.00025"]
        self.assertEqual(
            {(row["surface"], row["tap_id"]) for row in leading_edge},
            {("upper", "X15"), ("lower", "X16")},
        )
        reversal = [row for row in rows if row["alpha_label_deg"] == "19"]
        self.assertTrue(all(row["phase_rad"] == "" for row in reversal))
        self.assertTrue(
            all(
                row["phase_status"]
                == "reported categorical branch; exact phase undisclosed"
                for row in reversal
            )
        )
        self.assertLessEqual(
            max(abs(float(row["tap_x_residual"])) for row in experiment),
            1.1e-4,
        )


if __name__ == "__main__":
    unittest.main()
