"""Digitize the two pitching-wing lift traces in Mancini (2017) Figure 4.13b.

The dissertation embeds the lower panel as a lossless RGB image.  Yellow and
orange each contain one solid pitching trace and one dashed surging trace.  The
solid traces form long connected components, whereas every dash is a short
component.  This utility therefore selects only long colour-connected paths;
it never infers a curve from the model output or from the dashed surge data.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage


MANCINI_PDF_SHA256 = "afd15346f7177b90828b826d026560ad6887673563484ecfe52f502926ccee24"
PDF_PAGE = 127
IMAGE_SHAPE = (608, 1232, 3)

# Exact dominant RGB values in the lossless embedded image.
FAST_RGB = np.asarray((237.0, 177.0, 32.0))
SLOW_RGB = np.asarray((217.0, 83.0, 25.0))
COLOUR_DISTANCE_MAX = 45.0

# Tick centres read from the original embedded plot, not a page screenshot.
X_TICK_VALUES = np.asarray((0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0))
X_TICK_PIXELS = np.asarray((195.0, 336.0, 478.0, 619.0, 761.0, 902.0, 1044.0))
Y_TICK_VALUES = np.asarray((0.0, 1.0, 2.0, 3.0, 4.0, 5.0))
Y_TICK_PIXELS = np.asarray((500.0, 418.0, 335.0, 252.0, 170.0, 87.0))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _extract_lower_panel(pdf_path: Path, directory: Path) -> np.ndarray:
    prefix = directory / "mancini-page127"
    subprocess.run(
        [
            "pdfimages",
            "-f",
            str(PDF_PAGE),
            "-l",
            str(PDF_PAGE),
            "-png",
            str(pdf_path),
            str(prefix),
        ],
        check=True,
    )
    candidates = sorted(directory.glob("mancini-page127-*.png"))
    if len(candidates) != 2:
        raise ValueError(
            f"expected two embedded Figure 4.13 panels, found {len(candidates)}"
        )
    image = np.asarray(Image.open(candidates[1]).convert("RGB"))
    if image.shape != IMAGE_SHAPE:
        raise ValueError(f"unexpected Figure 4.13b image shape: {image.shape}")
    return image


def _solid_curve_pixels(image: np.ndarray, target_rgb: np.ndarray) -> np.ndarray:
    distance = np.linalg.norm(image.astype(float) - target_rgb, axis=2)
    row, column = np.indices(distance.shape)
    colour = distance <= COLOUR_DISTANCE_MAX
    plot_crop = (row >= 45) & (row <= 505) & (column >= 155) & (column <= 1118)
    labels, count = ndimage.label(colour & plot_crop, np.ones((3, 3), dtype=int))
    selected = np.zeros(labels.shape, dtype=bool)
    retained: list[tuple[int, int, int]] = []
    for component in range(1, count + 1):
        rows, columns = np.nonzero(labels == component)
        if rows.size == 0:
            continue
        width = int(columns.max() - columns.min() + 1)
        # The solid paths are the only components with >=100 pixels and a
        # horizontal extent of at least 20 pixels.  All same-colour dashes are
        # short components (<=65 pixels in the frozen source image).
        if rows.size >= 100 and width >= 20:
            selected |= labels == component
            retained.append((int(rows.size), int(columns.min()), int(columns.max())))
    if len(retained) < 4:
        raise ValueError(f"too few solid-curve components: {retained!r}")

    points: list[tuple[float, float]] = []
    for x_pixel in range(selected.shape[1]):
        y_pixels = np.flatnonzero(selected[:, x_pixel])
        if y_pixels.size:
            points.append((float(x_pixel), float(np.median(y_pixels))))
    output = np.asarray(points, dtype=float)
    if output.shape[0] < 500:
        raise ValueError("solid-curve extraction is too sparse")
    return output


def _calibrate(points: np.ndarray, target: np.ndarray) -> np.ndarray:
    x_slope, x_intercept = np.polyfit(X_TICK_VALUES, X_TICK_PIXELS, 1)
    y_slope, y_intercept = np.polyfit(Y_TICK_VALUES, Y_TICK_PIXELS, 1)
    t_star = (points[:, 0] - x_intercept) / x_slope
    lift = (points[:, 1] - y_intercept) / y_slope
    valid = (t_star >= -0.05) & (t_star <= 13.05) & (lift >= -0.1) & (lift <= 5.5)
    t_star = t_star[valid]
    lift = lift[valid]
    order = np.argsort(t_star, kind="stable")
    t_star = t_star[order]
    lift = lift[order]
    if t_star[0] > 0.03 or t_star[-1] < 12.95:
        raise ValueError(
            f"curve does not span the published domain: {t_star[0]}, {t_star[-1]}"
        )
    return np.interp(target, t_star, lift)


def digitize(pdf_path: Path, output_csv: Path) -> None:
    pdf_path = pdf_path.resolve()
    if _sha256(pdf_path) != MANCINI_PDF_SHA256:
        raise ValueError("Mancini dissertation PDF hash mismatch")
    target = np.linspace(0.0, 13.0, 1301)
    with tempfile.TemporaryDirectory(prefix="mancini2017-digitize-") as temporary:
        image = _extract_lower_panel(pdf_path, Path(temporary))
        fast = _calibrate(_solid_curve_pixels(image, FAST_RGB), target)
        slow = _calibrate(_solid_curve_pixels(image, SLOW_RGB), target)

    fast_peak_index = int(np.argmax(fast))
    slow_peak_index = int(np.argmax(slow))
    if not (4.7 <= fast[fast_peak_index] <= 5.2 and target[fast_peak_index] <= 1.1):
        raise ValueError("fast pitching trace landmark changed")
    if not (
        1.9 <= slow[slow_peak_index] <= 2.2 and 4.0 <= target[slow_peak_index] <= 6.0
    ):
        raise ValueError("slow pitching trace landmark changed")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "t_star",
                "experiment_CL_fast_pitch",
                "experiment_CL_slow_pitch",
                "source_figure",
                "data_role",
            ]
        )
        for distance, fast_value, slow_value in zip(target, fast, slow, strict=True):
            writer.writerow(
                [
                    f"{distance:.8f}",
                    f"{fast_value:.9f}",
                    f"{slow_value:.9f}",
                    "4.13b",
                    "digitized_experimental_curve",
                ]
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("output_csv", type=Path)
    arguments = parser.parse_args()
    digitize(arguments.pdf, arguments.output_csv)


if __name__ == "__main__":
    main()
