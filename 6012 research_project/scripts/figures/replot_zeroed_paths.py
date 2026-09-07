"""Replot the two trapezoidal validation paths in a local 0, 0 frame.

The tracking CSV values use the checkerboard coordinate frame, which is useful
for registration but produces negative labels in report figures.  This script
retains the original path geometry and millimetre scale while translating each
complete measured/commanded dataset so that its lower-left extent is (0, 0).
"""

from __future__ import annotations

from html import escape
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "20260810_zeroed_path_plots"

# Pixel-to-data mappings for the archived high-resolution plots.  Retaining the
# archived orange path is important because gait_table.h was regenerated after
# these validation figures were produced.
DATASETS = {
    "trap pull": {
        "label": "Before PWM calibration",
        "x_zero_px": 319.5,
        "y_zero_px": 334.0,
    },
    "trap new": {
        "label": "After PWM calibration",
        "x_zero_px": 499.5,
        "y_zero_px": 515.5,
    },
}

PIXELS_PER_10_MM = 171.0
AXES_X_MIN_PX = 205
AXES_X_MAX_PX = 1713
AXES_Y_MIN_PX = 205
AXES_Y_MAX_PX = 1531


def extract_commanded_path(
    image_path: Path, x_zero_px: float, y_zero_px: float
) -> np.ndarray:
    """Recover the orange commanded-path centreline from its archived plot."""
    image = np.asarray(Image.open(image_path).convert("RGB"))
    red = image[:, :, 0]
    green = image[:, :, 1]
    blue = image[:, :, 2]
    orange = (
        (red > 180)
        & (red > 1.25 * green)
        & (green > 60)
        & (green < 180)
        & (blue < 100)
    )

    # Restrict extraction to the original data axes and below the legend.
    keep = np.zeros_like(orange)
    keep[
        700 : AXES_Y_MAX_PX + 1,
        AXES_X_MIN_PX : AXES_X_MAX_PX + 1,
    ] = True
    orange &= keep

    centres: list[tuple[float, float]] = []
    for x_px in range(AXES_X_MIN_PX, AXES_X_MAX_PX + 1):
        y_values = np.flatnonzero(orange[:, x_px])
        if len(y_values) == 0:
            continue

        # A vertical slice normally intersects the upper and lower strokes.
        # Split separated pixel runs and retain the median of each thick stroke.
        split_indices = np.flatnonzero(np.diff(y_values) > 2) + 1
        for run in np.split(y_values, split_indices):
            if len(run) < 2:
                continue
            centres.append((float(x_px), float(np.median(run))))

    pixels = np.asarray(centres)
    scale = PIXELS_PER_10_MM / 10.0
    x_mm = (pixels[:, 0] - x_zero_px) / scale
    y_mm = (y_zero_px - pixels[:, 1]) / scale
    return np.column_stack((x_mm, y_mm))


def viridis(value: float) -> str:
    """Return a compact interpolation of the Matplotlib viridis palette."""
    anchors = np.asarray(
        [
            [68, 1, 84],
            [59, 82, 139],
            [33, 145, 140],
            [94, 201, 98],
            [253, 231, 37],
        ],
        dtype=float,
    )
    scaled = np.clip(value, 0.0, 1.0) * (len(anchors) - 1)
    index = min(int(scaled), len(anchors) - 2)
    fraction = scaled - index
    rgb = anchors[index] * (1.0 - fraction) + anchors[index + 1] * fraction
    return "#" + "".join(f"{int(round(channel)):02x}" for channel in rgb)


def plot_dataset(folder: Path, settings: dict[str, float | str]) -> Path:
    measured = pd.read_csv(folder / "measured_vs_expected.csv")
    commanded = extract_commanded_path(
        folder / "measured_vs_commanded_path.png",
        float(settings["x_zero_px"]),
        float(settings["y_zero_px"]),
    )

    measured_xy = measured[["x_measured_mm", "y_measured_mm"]].to_numpy()
    all_xy = np.vstack((commanded, measured_xy))
    offset = np.min(all_xy, axis=0)
    commanded -= offset
    measured_xy -= offset

    x_max = 10.0 * np.ceil(max(commanded[:, 0].max(), measured_xy[:, 0].max()) / 10.0)
    y_max = 10.0 * np.ceil(max(commanded[:, 1].max(), measured_xy[:, 1].max()) / 10.0)
    x_max = max(10.0, x_max)
    y_max = max(10.0, y_max)

    width = 1600
    margin_left = 110
    margin_top = 115
    plot_width = 1260
    scale = plot_width / x_max
    plot_height = y_max * scale
    height = int(margin_top + plot_height + 105)
    plot_bottom = margin_top + plot_height

    def sx(x_value: float) -> float:
        return margin_left + x_value * scale

    def sy(y_value: float) -> float:
        return plot_bottom - y_value * scale

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}'
        '.tick{font-size:20px}.label{font-size:23px}.title{font-size:28px}'
        '.legend{font-size:19px}</style>',
        f'<text x="{margin_left + plot_width / 2:.1f}" y="36" text-anchor="middle" '
        f'class="title">{escape(str(settings["label"]))}</text>',
    ]

    # Legend above the axes so it does not obscure either path.
    legend_y = 72
    legend_x = margin_left + 300
    svg.extend(
        [
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 48}" '
            f'y2="{legend_y}" stroke="#e67e22" stroke-width="5"/>',
            f'<text x="{legend_x + 60}" y="{legend_y + 7}" class="legend">'
            'Path commanded by gait table</text>',
            f'<circle cx="{legend_x + 475}" cy="{legend_y}" r="5" fill="#6a1b72"/>',
            f'<text x="{legend_x + 490}" y="{legend_y + 7}" class="legend">'
            'Measured green-dot positions</text>',
        ]
    )

    # Neutral 10 mm grid and axes.
    for tick in np.arange(0.0, x_max + 0.1, 10.0):
        x_position = sx(float(tick))
        svg.append(
            f'<line x1="{x_position:.2f}" y1="{margin_top}" x2="{x_position:.2f}" '
            f'y2="{plot_bottom:.2f}" stroke="#d1d5db" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{x_position:.2f}" y="{plot_bottom + 31:.2f}" '
            f'text-anchor="middle" class="tick">{int(tick)}</text>'
        )
    for tick in np.arange(0.0, y_max + 0.1, 10.0):
        y_position = sy(float(tick))
        svg.append(
            f'<line x1="{margin_left}" y1="{y_position:.2f}" '
            f'x2="{margin_left + plot_width}" y2="{y_position:.2f}" '
            f'stroke="#d1d5db" stroke-width="1"/>'
        )
        svg.append(
            f'<text x="{margin_left - 16}" y="{y_position + 7:.2f}" '
            f'text-anchor="end" class="tick">{int(tick)}</text>'
        )
    svg.append(
        f'<rect x="{margin_left}" y="{margin_top}" width="{plot_width}" '
        f'height="{plot_height:.2f}" fill="none" stroke="#111827" stroke-width="2"/>'
    )

    # The archived commanded centreline is dense enough to render as a solid line.
    for x_value, y_value in commanded:
        svg.append(
            f'<circle cx="{sx(float(x_value)):.2f}" cy="{sy(float(y_value)):.2f}" '
            'r="1.8" fill="#e67e22"/>'
        )

    times = measured["tracking_time_s"].to_numpy(dtype=float)
    time_min = float(times.min())
    time_max = float(times.max())
    time_span = max(1e-12, time_max - time_min)
    for (x_value, y_value), time_value in zip(measured_xy, times):
        colour = viridis((float(time_value) - time_min) / time_span)
        svg.append(
            f'<circle cx="{sx(float(x_value)):.2f}" cy="{sy(float(y_value)):.2f}" '
            f'r="3.6" fill="{colour}" fill-opacity="0.78"/>'
        )

    # Compact viridis colour bar and labels.
    colour_x = margin_left + plot_width + 55
    gradient_id = folder.name.replace(" ", "_")
    svg.extend(
        [
            '<defs>',
            f'<linearGradient id="{gradient_id}" x1="0" y1="1" x2="0" y2="0">',
            '<stop offset="0%" stop-color="#440154"/>',
            '<stop offset="25%" stop-color="#3b528b"/>',
            '<stop offset="50%" stop-color="#21918c"/>',
            '<stop offset="75%" stop-color="#5ec962"/>',
            '<stop offset="100%" stop-color="#fde725"/>',
            '</linearGradient></defs>',
            f'<rect x="{colour_x}" y="{margin_top}" width="24" height="{plot_height:.2f}" '
            f'fill="url(#{gradient_id})" stroke="#111827" stroke-width="1.5"/>',
        ]
    )
    for fraction in np.linspace(0.0, 1.0, 6):
        y_position = plot_bottom - fraction * plot_height
        time_label = time_min + fraction * time_span
        if abs(time_label) < 0.5:
            time_label = 0.0
        svg.append(
            f'<text x="{colour_x + 34}" y="{y_position + 6:.2f}" '
            f'class="tick">{time_label:.0f}</text>'
        )

    svg.extend(
        [
            f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 18}" '
            'text-anchor="middle" class="label">Relative X position (mm)</text>',
            f'<text x="30" y="{margin_top + plot_height / 2:.1f}" '
            'text-anchor="middle" class="label" '
            f'transform="rotate(-90 30 {margin_top + plot_height / 2:.1f})">'
            'Relative Y position (mm)</text>',
            f'<text x="{colour_x + 104}" y="{margin_top + plot_height / 2:.1f}" '
            'text-anchor="middle" class="label" '
            f'transform="rotate(-90 {colour_x + 104} {margin_top + plot_height / 2:.1f})">'
            'Tracking time (s)</text>',
            '</svg>',
        ]
    )

    output_path = OUTPUT_DIR / f"{folder.name.replace(' ', '_')}_zeroed.svg"
    output_path.write_text("\n".join(svg), encoding="utf-8")
    return output_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tracking_root = SCRIPT_DIR / "green_dot_tracking_outputs"
    for folder_name, settings in DATASETS.items():
        output_path = plot_dataset(tracking_root / folder_name, settings)
        print(output_path)


if __name__ == "__main__":
    main()
