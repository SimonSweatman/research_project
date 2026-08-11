"""Analyse the 2 x 3 friction-versus-cycle-duration experiment.

Design: high/low friction x 1.5/3/6 s gait cycle x 3 repeats = 18 videos.
The 180 degree phase, camera calibration, finish line and LED ROI are reused
from the phase experiment.

Recommended filename:
    run01_high_cycle1p5s_rep2.mp4

The script creates a manifest, tracks all videos, performs balanced two-way
ANOVA (friction, cycle duration and interaction), and exports one workbook,
comparison charts, per-run CSVs, diagnostics and optional overlay videos.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import analyse_phase_experiments as core


SCRIPT_DIR = Path(__file__).resolve().parent
VIDEO_DIR = SCRIPT_DIR / "doe_friction_speed_videos"
MANIFEST_CSV = VIDEO_DIR / "experiment_manifest.csv"
OUTPUT_DIR = SCRIPT_DIR / "doe_friction_speed_analysis_outputs"
WORKBOOK_PATH = OUTPUT_DIR / "friction_speed_experiment_results.xlsx"

EXPECTED_FRICTION_LEVELS = ["Low", "High"]
EXPECTED_CYCLE_DURATIONS_S = [1.5, 3.0, 6.0]
EXPECTED_REPEATS = 3
FIXED_PHASE_DEG = 180
WAVEFORM = "Rounded triangle"
DRIVE_LENGTH_MM = 70.0
CILIA_SPACING_MM = 87.0
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".avi"}

WRITE_ANNOTATED_VIDEOS = True
COMPLETE_CYCLE_DURATION_TOLERANCE = 0.20

# Stricter than the original phase script: a candidate pulse must remain on
# for the required frames AND peak at least this far above its threshold.
LED_EXTRA_PEAK_MARGIN = 15.0


RUN_COLUMNS = [
    "Randomised Run Order", "Test ID", "Video Filename", "Friction Level",
    "Measured Friction Coefficient", "Cycle Duration (s)",
    "Cycle-Time Multiplier", "Beat Frequency (Hz)",
    "Frequency Multiplier", "Repeat Number", "Phase Shift (deg)",
    "Start Blink Time (s)", "Finish Crossing Time (s)", "Transport Time (s)",
    "Net Travel Distance (mm)", "Mean Transport Speed (mm/s)",
    "Median Transport Speed (mm/s)", "Number of Complete Cycles",
    "Mean Net Advance per Cycle (mm)", "SD Net Advance per Cycle (mm)",
    "Total Backward Travel (mm)", "Backward Travel Percentage (%)",
    "Maximum Backward Slip in One Cycle (mm)",
    "Vertical Centroid Peak-to-Peak (mm)", "Vertical Centroid RMS (mm)",
    "Mean Pitch Angle (deg)", "Pitch Peak-to-Peak (deg)",
    "Maximum Absolute Pitch Angle (deg)", "Green Marker Detection Rate (%)",
    "Frames Requiring Interpolation", "Finish Crossing Detected",
    "Quality-Control Status", "Notes",
]

CYCLE_COLUMNS = [
    "Test ID", "Friction Level", "Cycle Duration Setting (s)", "Cycle Number",
    "Cycle Start Time (s)", "Cycle End Time (s)",
    "Measured Cycle Duration (s)", "Centroid Start X (mm)",
    "Centroid End X (mm)", "Net Advance (mm)", "Total Forward Movement (mm)",
    "Total Backward Movement (mm)", "Maximum Backward Excursion (mm)",
    "Vertical Peak-to-Peak (mm)", "Mean Pitch Angle (deg)",
    "Pitch Peak-to-Peak (deg)",
]

REJECTED_COLUMNS = [
    "Test ID", "Friction Level", "Cycle Duration Setting (s)",
    "Candidate Interval Number", "Interval Start Time (s)",
    "Interval End Time (s)", "Measured Interval Duration (s)",
    "Accepted Duration Minimum (s)", "Accepted Duration Maximum (s)",
    "Rejection Reason",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-overlay", action="store_true",
        help="Skip annotated MP4 creation to reduce processing time/storage.",
    )
    parser.add_argument(
        "--reset-led-roi", action="store_true",
        help="Select the LED area again instead of reusing the phase-test ROI.",
    )
    return parser.parse_args()


def normalise_friction(value: object) -> str:
    text = str(value).strip().lower()
    if text in {"high", "h", "high friction", "rough"}:
        return "High"
    if text in {"low", "l", "low friction", "smooth"}:
        return "Low"
    return ""


def filename_fields(path: Path) -> tuple[float, str, float, float]:
    text = path.stem.lower()
    run_match = re.search(r"run[ _-]?(\d+)", text)
    repeat_match = re.search(r"rep(?:eat)?[ _-]?(\d+)", text)
    friction = "High" if re.search(r"(?:^|[_ -])(?:high|hf|rough)(?:[_ -]|$)", text) else ""
    if not friction and re.search(r"(?:^|[_ -])(?:low|lf|smooth)(?:[_ -]|$)", text):
        friction = "Low"
    duration_match = re.search(
        r"(?:cycle|time|ct)[ _-]?(1(?:[p.]5)?|3(?:[p.]0)?|6(?:[p.]0)?)s?", text
    )
    if duration_match is None:
        duration_match = re.search(r"(?:^|[_ -])(1[p.]5|3|6)s(?:[_ .-]|$)", text)
    duration = math.nan
    if duration_match:
        duration = float(duration_match.group(1).replace("p", "."))
    return (
        float(run_match.group(1)) if run_match else math.nan,
        friction,
        duration,
        float(repeat_match.group(1)) if repeat_match else math.nan,
    )


def discover_videos() -> list[Path]:
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    return sorted(
        path for path in VIDEO_DIR.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        and "overlay" not in path.stem.lower()
    )


def create_or_update_manifest(videos: list[Path]) -> pd.DataFrame:
    columns = [
        "Include", "Randomised Run Order", "Test ID", "Video Filename",
        "Friction Level", "Measured Friction Coefficient", "Cycle Duration (s)",
        "Repeat Number", "Phase Shift (deg)", "Waveform", "Drive Length (mm)",
        "Cilia Spacing (mm)", "Recording Date", "Operator Notes",
    ]
    existing = pd.read_csv(MANIFEST_CSV) if MANIFEST_CSV.exists() else pd.DataFrame()
    previous = (
        existing.set_index("Video Filename").to_dict("index")
        if not existing.empty and "Video Filename" in existing else {}
    )
    rows: list[dict[str, object]] = []
    for video in videos:
        run, friction, duration, repeat = filename_fields(video)
        old = previous.get(video.name, {})
        friction_value = normalise_friction(old.get("Friction Level", friction))
        duration_value = old.get("Cycle Duration (s)", duration)
        repeat_value = old.get("Repeat Number", repeat)
        test_id = str(old.get("Test ID", "")).strip()
        if (
            not test_id and friction_value and not pd.isna(duration_value)
            and not pd.isna(repeat_value)
        ):
            duration_text = str(float(duration_value)).replace(".", "p")
            test_id = f"{friction_value.lower()}_{duration_text}s_rep{int(repeat_value)}"
        rows.append({
            "Include": old.get("Include", 1),
            "Randomised Run Order": old.get("Randomised Run Order", run),
            "Test ID": test_id,
            "Video Filename": video.name,
            "Friction Level": friction_value,
            "Measured Friction Coefficient": old.get(
                "Measured Friction Coefficient", math.nan
            ),
            "Cycle Duration (s)": duration_value,
            "Repeat Number": repeat_value,
            "Phase Shift (deg)": old.get("Phase Shift (deg)", FIXED_PHASE_DEG),
            "Waveform": old.get("Waveform", WAVEFORM),
            "Drive Length (mm)": old.get("Drive Length (mm)", DRIVE_LENGTH_MM),
            "Cilia Spacing (mm)": old.get("Cilia Spacing (mm)", CILIA_SPACING_MM),
            "Recording Date": old.get("Recording Date", ""),
            "Operator Notes": old.get("Operator Notes", ""),
        })
    manifest = pd.DataFrame(rows, columns=columns)
    if not manifest.empty:
        manifest = manifest.sort_values(
            "Randomised Run Order", na_position="last"
        ).reset_index(drop=True)
    manifest.to_csv(MANIFEST_CSV, index=False)
    return manifest


def validate_manifest(manifest: pd.DataFrame) -> pd.DataFrame:
    if manifest.empty:
        raise SystemExit(f"No videos found. Add the 18 recordings to:\n{VIDEO_DIR}")
    included = manifest[
        pd.to_numeric(manifest["Include"], errors="coerce").fillna(0) != 0
    ].copy()
    required = [
        "Randomised Run Order", "Test ID", "Friction Level",
        "Cycle Duration (s)", "Repeat Number",
    ]
    if any(included[column].isna().any() for column in required) or included[
        "Test ID"
    ].astype(str).str.strip().eq("").any():
        raise SystemExit(
            f"Complete missing fields in:\n{MANIFEST_CSV}\n"
            "Recommended name: run01_high_cycle1p5s_rep2.mp4"
        )
    included["Friction Level"] = included["Friction Level"].map(normalise_friction)
    included["Cycle Duration (s)"] = pd.to_numeric(
        included["Cycle Duration (s)"], errors="coerce"
    )
    if included["Friction Level"].eq("").any() or included[
        "Cycle Duration (s)"
    ].isna().any():
        raise SystemExit(f"Correct friction/duration entries in:\n{MANIFEST_CSV}")
    if len(included) != 18:
        print(f"WARNING: expected 18 included videos, found {len(included)}.")
    for friction in EXPECTED_FRICTION_LEVELS:
        for duration in EXPECTED_CYCLE_DURATIONS_S:
            count = len(included[
                included["Friction Level"].eq(friction)
                & np.isclose(included["Cycle Duration (s)"], duration)
            ])
            if count != EXPECTED_REPEATS:
                print(
                    f"WARNING: {friction}, {duration:g} s has {count} repeats; "
                    f"expected {EXPECTED_REPEATS}."
                )
    return included.sort_values("Randomised Run Order")


def load_saved_led_roi(
    first_video: Path, calibration: core.Calibration, reset: bool
) -> tuple[int, int, int, int]:
    # The camera is unchanged, so reuse the phase study selection by default.
    if core.TRACKING_SETTINGS_JSON.exists() and not reset:
        settings = json.loads(core.TRACKING_SETTINGS_JSON.read_text(encoding="utf-8"))
        roi = settings.get("led_roi_px")
        if roi and len(roi) == 4:
            return tuple(int(value) for value in roi)
    return core.select_or_load_led_roi(first_video, calibration, reset=True)


def detect_led_blinks_strict(
    data: pd.DataFrame, fps: float
) -> tuple[np.ndarray, list[int], float]:
    score = data["LED Score"].to_numpy(float)
    smooth = pd.Series(score).rolling(3, center=True, min_periods=1).median().to_numpy()
    baseline_count = max(
        3, min(len(smooth), int(round(core.LED_BASELINE_SECONDS * fps)))
    )
    baseline = smooth[:baseline_count]
    median = float(np.nanmedian(baseline))
    mad = float(np.nanmedian(np.abs(baseline - median)))
    threshold = median + max(core.LED_THRESHOLD_MIN_RISE, 7 * 1.4826 * mad)
    raw_on = smooth >= threshold
    on = np.zeros_like(raw_on, dtype=bool)
    changes = np.flatnonzero(np.diff(np.r_[False, raw_on, False]))
    for start, end in changes.reshape(-1, 2):
        peak = float(np.nanmax(smooth[start:end]))
        if (
            end - start >= core.LED_MIN_ON_FRAMES
            and peak >= threshold + LED_EXTRA_PEAK_MARGIN
        ):
            on[start:end] = True
    edges = core.rising_edges(
        on, max(1, int(round(core.LED_MIN_GAP_SECONDS * fps)))
    )
    data["LED Score Smoothed"] = smooth
    data["LED Detected"] = on
    return on, edges, threshold


def variable_cycle_table(
    test_id: str,
    friction: str,
    expected_duration: float,
    frames: pd.DataFrame,
    blink_frames: list[int],
    start_frame: int,
    finish_frame: int | None,
    fps: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    end_limit = finish_frame if finish_frame is not None else len(frames) - 1
    boundaries = [value for value in blink_frames if start_frame <= value <= end_limit]
    minimum = expected_duration * (1 - COMPLETE_CYCLE_DURATION_TOLERANCE)
    maximum = expected_duration * (1 + COMPLETE_CYCLE_DURATION_TOLERANCE)
    accepted: list[dict[str, object]] = []
    rejected: list[dict[str, object]] = []
    for candidate_number, (first, last) in enumerate(
        zip(boundaries, boundaries[1:]), 1
    ):
        duration = (last - first) / fps
        if not minimum <= duration <= maximum:
            rejected.append({
                "Test ID": test_id,
                "Friction Level": friction,
                "Cycle Duration Setting (s)": expected_duration,
                "Candidate Interval Number": candidate_number,
                "Interval Start Time (s)": first / fps,
                "Interval End Time (s)": last / fps,
                "Measured Interval Duration (s)": duration,
                "Accepted Duration Minimum (s)": minimum,
                "Accepted Duration Maximum (s)": maximum,
                "Rejection Reason": "LED interval outside complete-cycle tolerance",
            })
            continue
        segment = frames.iloc[first:last + 1]
        x = segment["Object Centroid X (mm)"].rolling(
            core.SMOOTHING_WINDOW_FRAMES, center=True, min_periods=1
        ).mean().to_numpy(float)
        if len(x) < 2 or not np.isfinite(x[[0, -1]]).all():
            rejected.append({
                "Test ID": test_id,
                "Friction Level": friction,
                "Cycle Duration Setting (s)": expected_duration,
                "Candidate Interval Number": candidate_number,
                "Interval Start Time (s)": first / fps,
                "Interval End Time (s)": last / fps,
                "Measured Interval Duration (s)": duration,
                "Accepted Duration Minimum (s)": minimum,
                "Accepted Duration Maximum (s)": maximum,
                "Rejection Reason": "Insufficient valid three-marker centroid data",
            })
            continue
        forward_position = x[0] - x
        total_forward, total_backward, maximum_backward = core.path_distances(
            forward_position
        )
        vertical = segment["Object Centroid Y (mm)"].to_numpy(float)
        pitch = segment["Pitch Angle (deg)"].to_numpy(float)
        accepted.append({
            "Test ID": test_id,
            "Friction Level": friction,
            "Cycle Duration Setting (s)": expected_duration,
            "Cycle Number": len(accepted) + 1,
            "Cycle Start Time (s)": first / fps,
            "Cycle End Time (s)": last / fps,
            "Measured Cycle Duration (s)": duration,
            "Centroid Start X (mm)": x[0],
            "Centroid End X (mm)": x[-1],
            "Net Advance (mm)": forward_position[-1],
            "Total Forward Movement (mm)": total_forward,
            "Total Backward Movement (mm)": total_backward,
            "Maximum Backward Excursion (mm)": maximum_backward,
            "Vertical Peak-to-Peak (mm)": float(np.nanmax(vertical) - np.nanmin(vertical)),
            "Mean Pitch Angle (deg)": float(np.nanmean(pitch)),
            "Pitch Peak-to-Peak (deg)": float(np.nanmax(pitch) - np.nanmin(pitch)),
        })
    return (
        pd.DataFrame(accepted, columns=CYCLE_COLUMNS),
        pd.DataFrame(rejected, columns=REJECTED_COLUMNS),
    )


def summarise_run(
    row: pd.Series,
    frames: pd.DataFrame,
    cycles: pd.DataFrame,
    fps: float,
    start_frame: int,
    finish_frame: int | None,
    crossing_float: float,
    notes: list[str],
) -> dict[str, object]:
    end = finish_frame if finish_frame is not None else len(frames) - 1
    segment = frames.iloc[start_frame:end + 1]
    centroid_x = segment["Object Centroid X (mm)"].rolling(
        core.SMOOTHING_WINDOW_FRAMES, center=True, min_periods=1
    ).mean().to_numpy(float)
    valid_x = centroid_x[np.isfinite(centroid_x)]
    if len(valid_x):
        initial_count = max(1, min(len(valid_x), round(0.2 * fps)))
        start_x = float(np.nanmedian(valid_x[:initial_count]))
        end_x = float(valid_x[-1])
        net_distance = start_x - end_x
    else:
        start_x = end_x = net_distance = math.nan
    duration = (
        (crossing_float - start_frame) / fps
        if finish_frame is not None else math.nan
    )
    forward_position = start_x - centroid_x
    _, backward, _ = core.path_distances(forward_position)
    velocity = segment["Forward Velocity (mm/s)"].to_numpy(float)
    vertical = segment["Object Centroid Y (mm)"].to_numpy(float)
    pitch = segment["Pitch Angle (deg)"].to_numpy(float)
    valid_vertical = vertical[np.isfinite(vertical)]
    valid_pitch = pitch[np.isfinite(pitch)]
    detection_rate = 100 * float(
        np.mean(segment["Number of Markers Detected"].to_numpy() == 3)
    )
    qc = "PASS"
    if finish_frame is None:
        qc = "REVIEW - finish not crossed"
    elif detection_rate < 90:
        qc = "REVIEW - marker detection below 90%"
    if not len(valid_x):
        qc = "FAIL - no valid three-marker positions"
    expected_cycle = float(row["Cycle Duration (s)"])
    return {
        "Randomised Run Order": int(row["Randomised Run Order"]),
        "Test ID": str(row["Test ID"]),
        "Video Filename": str(row["Video Filename"]),
        "Friction Level": str(row["Friction Level"]),
        "Measured Friction Coefficient": row["Measured Friction Coefficient"],
        "Cycle Duration (s)": expected_cycle,
        "Cycle-Time Multiplier": expected_cycle / 3.0,
        "Beat Frequency (Hz)": 1.0 / expected_cycle,
        "Frequency Multiplier": 3.0 / expected_cycle,
        "Repeat Number": int(row["Repeat Number"]),
        "Phase Shift (deg)": int(row["Phase Shift (deg)"]),
        "Start Blink Time (s)": start_frame / fps,
        "Finish Crossing Time (s)": crossing_float / fps if finish_frame is not None else math.nan,
        "Transport Time (s)": duration,
        "Net Travel Distance (mm)": net_distance,
        "Mean Transport Speed (mm/s)": net_distance / duration if duration > 0 else math.nan,
        "Median Transport Speed (mm/s)": float(np.nanmedian(velocity)),
        "Number of Complete Cycles": len(cycles),
        "Mean Net Advance per Cycle (mm)": cycles["Net Advance (mm)"].mean(),
        "SD Net Advance per Cycle (mm)": cycles["Net Advance (mm)"].std(ddof=1),
        "Total Backward Travel (mm)": backward,
        "Backward Travel Percentage (%)": 100 * backward / abs(net_distance) if net_distance else math.nan,
        "Maximum Backward Slip in One Cycle (mm)": cycles[
            "Maximum Backward Excursion (mm)"
        ].max(),
        "Vertical Centroid Peak-to-Peak (mm)": float(np.ptp(valid_vertical)) if len(valid_vertical) else math.nan,
        "Vertical Centroid RMS (mm)": float(
            np.sqrt(np.mean((valid_vertical - np.mean(valid_vertical)) ** 2))
        ) if len(valid_vertical) else math.nan,
        "Mean Pitch Angle (deg)": float(np.mean(valid_pitch)) if len(valid_pitch) else math.nan,
        "Pitch Peak-to-Peak (deg)": float(np.ptp(valid_pitch)) if len(valid_pitch) else math.nan,
        "Maximum Absolute Pitch Angle (deg)": float(np.max(np.abs(valid_pitch))) if len(valid_pitch) else math.nan,
        "Green Marker Detection Rate (%)": detection_rate,
        "Frames Requiring Interpolation": int(segment["Interpolated Frame"].sum()),
        "Finish Crossing Detected": "Yes" if finish_frame is not None else "No",
        "Quality-Control Status": qc,
        "Notes": "; ".join(notes),
    }


def condition_summary(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Transport Time (s)", "Mean Transport Speed (mm/s)",
        "Mean Net Advance per Cycle (mm)", "Total Backward Travel (mm)",
        "Backward Travel Percentage (%)", "Vertical Centroid Peak-to-Peak (mm)",
        "Pitch Peak-to-Peak (deg)",
    ]
    rows: list[dict[str, object]] = []
    valid = summary[summary["Finish Crossing Detected"].eq("Yes")]
    for friction in EXPECTED_FRICTION_LEVELS:
        for duration in EXPECTED_CYCLE_DURATIONS_S:
            group = valid[
                valid["Friction Level"].eq(friction)
                & np.isclose(valid["Cycle Duration (s)"], duration)
            ]
            result: dict[str, object] = {
                "Friction Level": friction,
                "Cycle Duration (s)": duration,
                "Cycle-Time Multiplier": duration / 3.0,
                "Beat Frequency (Hz)": 1.0 / duration,
                "Frequency Multiplier": 3.0 / duration,
                "Valid Repeats": len(group),
            }
            for metric in metrics:
                label = metric.replace(" (", "_").replace(")", "")
                result[f"Mean {label}"] = group[metric].mean()
                result[f"SD {label}"] = group[metric].std(ddof=1)
            rows.append(result)
    return pd.DataFrame(rows)


def main_effects(summary: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "Transport Time (s)", "Mean Transport Speed (mm/s)",
        "Mean Net Advance per Cycle (mm)", "Backward Travel Percentage (%)",
        "Vertical Centroid Peak-to-Peak (mm)", "Pitch Peak-to-Peak (deg)",
    ]
    valid = summary[summary["Finish Crossing Detected"].eq("Yes")]
    rows: list[dict[str, object]] = []
    for factor, levels in [
        ("Friction Level", EXPECTED_FRICTION_LEVELS),
        ("Cycle Duration (s)", EXPECTED_CYCLE_DURATIONS_S),
    ]:
        for level in levels:
            group = valid[
                valid[factor].eq(level)
                if factor == "Friction Level"
                else np.isclose(valid[factor], float(level))
            ]
            result: dict[str, object] = {
                "Factor": factor, "Level": level, "Valid Runs": len(group)
            }
            for metric in metrics:
                result[f"Mean {metric}"] = group[metric].mean()
                result[f"SD {metric}"] = group[metric].std(ddof=1)
            rows.append(result)
    return pd.DataFrame(rows)


def betacf(a: float, b: float, x: float) -> float:
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    d = 1e30 if abs(d) < 1e-30 else 1.0 / d
    h = d
    for m in range(1, 201):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = max(abs(1 + aa * d), 1e-30) * (1 if 1 + aa * d >= 0 else -1)
        c = max(abs(1 + aa / c), 1e-30) * (1 if 1 + aa / c >= 0 else -1)
        d = 1 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = max(abs(1 + aa * d), 1e-30) * (1 if 1 + aa * d >= 0 else -1)
        c = max(abs(1 + aa / c), 1e-30) * (1 if 1 + aa / c >= 0 else -1)
        d = 1 / d
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-14:
            break
    return h


def regularised_beta(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    value = math.exp(
        math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
        + a * math.log(x) + b * math.log1p(-x)
    )
    if x < (a + 1) / (a + b + 2):
        return value * betacf(a, b, x) / a
    return 1 - value * betacf(b, a, 1 - x) / b


def f_survival(f_value: float, df1: int, df2: int) -> float:
    if not math.isfinite(f_value) or f_value < 0:
        return math.nan
    x = df2 / (df2 + df1 * f_value)
    return regularised_beta(df2 / 2, df1 / 2, x)


def format_anova_p(p_value: float) -> str:
    """Format an ANOVA p-value for a concise plain-language conclusion."""
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.3f}"


def anova_conclusion(
    response: str,
    source: str,
    p_value: float,
    friction_means: dict[str, float],
    duration_means: dict[float, float],
    cell_means: dict[tuple[str, float], float],
) -> str:
    """Translate each ANOVA test into a data-specific interpretation."""
    units = {
        "Transport Time (s)": "s",
        "Mean Transport Speed (mm/s)": "mm/s",
        "Mean Net Advance per Cycle (mm)": "mm/cycle",
        "Backward Travel Percentage (%)": "%",
        "Vertical Centroid Peak-to-Peak (mm)": "mm",
        "Pitch Peak-to-Peak (deg)": "deg",
    }
    decimals = {
        "Transport Time (s)": 2,
        "Mean Transport Speed (mm/s)": 2,
        "Mean Net Advance per Cycle (mm)": 2,
        "Backward Travel Percentage (%)": 3,
        "Vertical Centroid Peak-to-Peak (mm)": 2,
        "Pitch Peak-to-Peak (deg)": 3,
    }
    unit = units[response]
    digits = decimals[response]
    significant = p_value < 0.05
    finding = "Significant" if significant else "Not statistically significant"
    p_text = format_anova_p(p_value)

    if source == "Friction Level":
        low = friction_means["Low"]
        high = friction_means["High"]
        direction = "higher" if high > low else "lower"
        return (
            f"{finding} friction effect ({p_text}). Averaged across cycle durations: "
            f"Low = {low:.{digits}f} {unit}; High = {high:.{digits}f} {unit} "
            f"(High was {direction} by {abs(high - low):.{digits}f} {unit})."
        )

    if source == "Cycle Duration":
        ordered = sorted(duration_means)
        means_text = "; ".join(
            f"{duration:g} s = {duration_means[duration]:.{digits}f} {unit}"
            for duration in ordered
        )
        return f"{finding} cycle-duration effect ({p_text}). {means_text}."

    if source == "Friction x Cycle Duration":
        differences = "; ".join(
            f"{duration:g} s: {cell_means[('High', duration)] - cell_means[('Low', duration)]:+.{digits}f} {unit}"
            for duration in sorted(duration_means)
        )
        if significant:
            interpretation = (
                "The friction effect changes with cycle duration, so compare the six "
                "condition means rather than relying on either main effect alone."
            )
        else:
            interpretation = (
                "There is no reliable evidence that the friction effect changes with "
                "cycle duration; the main effects can be interpreted independently."
            )
        return (
            f"{finding} interaction ({p_text}). {interpretation} "
            f"High minus Low at each duration: {differences}."
        )

    raise ValueError(f"Unsupported ANOVA source: {source}")


def two_way_anova(summary: pd.DataFrame) -> pd.DataFrame:
    responses = [
        "Transport Time (s)", "Mean Transport Speed (mm/s)",
        "Mean Net Advance per Cycle (mm)", "Backward Travel Percentage (%)",
        "Vertical Centroid Peak-to-Peak (mm)", "Pitch Peak-to-Peak (deg)",
    ]
    valid = summary[summary["Finish Crossing Detected"].eq("Yes")].copy()
    rows: list[dict[str, object]] = []
    for response in responses:
        subset = valid.dropna(subset=[response])
        cell_groups: dict[tuple[str, float], np.ndarray] = {}
        balanced = True
        for friction in EXPECTED_FRICTION_LEVELS:
            for duration in EXPECTED_CYCLE_DURATIONS_S:
                values = subset.loc[
                    subset["Friction Level"].eq(friction)
                    & np.isclose(subset["Cycle Duration (s)"], duration), response
                ].to_numpy(float)
                cell_groups[(friction, duration)] = values
                balanced &= len(values) == EXPECTED_REPEATS
        if not balanced:
            rows.append({
                "Response": response, "Source": "Not calculated",
                "Conclusion": "Requires three valid repeats in every condition",
            })
            continue
        n = EXPECTED_REPEATS
        all_values = np.concatenate(list(cell_groups.values()))
        grand = float(np.mean(all_values))
        friction_means = {
            level: float(np.mean(np.concatenate([
                cell_groups[(level, duration)]
                for duration in EXPECTED_CYCLE_DURATIONS_S
            ]))) for level in EXPECTED_FRICTION_LEVELS
        }
        duration_means = {
            duration: float(np.mean(np.concatenate([
                cell_groups[(level, duration)] for level in EXPECTED_FRICTION_LEVELS
            ]))) for duration in EXPECTED_CYCLE_DURATIONS_S
        }
        cell_means = {key: float(np.mean(values)) for key, values in cell_groups.items()}
        ss_friction = len(EXPECTED_CYCLE_DURATIONS_S) * n * sum(
            (value - grand) ** 2 for value in friction_means.values()
        )
        ss_duration = len(EXPECTED_FRICTION_LEVELS) * n * sum(
            (value - grand) ** 2 for value in duration_means.values()
        )
        ss_interaction = n * sum(
            (
                cell_means[(friction, duration)] - friction_means[friction]
                - duration_means[duration] + grand
            ) ** 2
            for friction in EXPECTED_FRICTION_LEVELS
            for duration in EXPECTED_CYCLE_DURATIONS_S
        )
        ss_error = sum(
            float(np.sum((values - cell_means[key]) ** 2))
            for key, values in cell_groups.items()
        )
        components = [
            ("Friction Level", ss_friction, 1),
            ("Cycle Duration", ss_duration, 2),
            ("Friction x Cycle Duration", ss_interaction, 2),
        ]
        error_df = 12
        error_ms = ss_error / error_df
        for source, ss_value, degrees in components:
            mean_square = ss_value / degrees
            f_value = mean_square / error_ms if error_ms > 0 else math.inf
            p_value = f_survival(f_value, degrees, error_df)
            rows.append({
                "Response": response, "Source": source,
                "Sum of Squares": ss_value, "df": degrees,
                "Mean Square": mean_square, "F": f_value, "p-value": p_value,
                "Significant at 0.05": "Yes" if p_value < 0.05 else "No",
                "Conclusion": anova_conclusion(
                    response, source, p_value,
                    friction_means, duration_means, cell_means,
                ),
            })
        rows.append({
            "Response": response, "Source": "Error",
            "Sum of Squares": ss_error, "df": error_df,
            "Mean Square": error_ms, "F": math.nan, "p-value": math.nan,
            "Significant at 0.05": "",
            "Conclusion": (
                "Residual repeat-to-repeat variation within the six experimental "
                "conditions; this mean square is the denominator used in the F-tests."
            ),
        })
        rows.append({
            "Response": response, "Source": "Total",
            "Sum of Squares": float(np.sum((all_values - grand) ** 2)),
            "df": len(all_values) - 1, "Mean Square": math.nan,
            "F": math.nan, "p-value": math.nan,
            "Significant at 0.05": "",
            "Conclusion": "Total observed variation across all 18 experimental runs.",
        })
    return pd.DataFrame(rows)


def plot_interactions(conditions: pd.DataFrame, destination: Path) -> None:
    plots = [
        ("Mean Mean Transport Speed_mm/s", "SD Mean Transport Speed_mm/s", "Transport speed (mm/s)"),
        ("Mean Transport Time_s", "SD Transport Time_s", "Transport time (s)"),
        ("Mean Mean Net Advance per Cycle_mm", "SD Mean Net Advance per Cycle_mm", "Advance per cycle (mm)"),
        ("Mean Backward Travel Percentage_%", "SD Backward Travel Percentage_%", "Backward travel (%)"),
    ]
    colours = {"Low": "#2878B5", "High": "#D95F02"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for axis, (mean_column, sd_column, ylabel) in zip(axes.flat, plots):
        for friction in EXPECTED_FRICTION_LEVELS:
            group = conditions[conditions["Friction Level"].eq(friction)].sort_values(
                "Cycle Duration (s)"
            )
            axis.errorbar(
                group["Cycle Duration (s)"], group[mean_column],
                yerr=group[sd_column], marker="o", linewidth=2, capsize=4,
                label=f"{friction} friction", color=colours[friction],
            )
        axis.set_xlabel("Commanded cycle duration (s)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(EXPECTED_CYCLE_DURATIONS_S)
        axis.grid(True, alpha=0.25)
    axes[0, 0].legend()
    fig.suptitle("Friction x cycle-duration interaction (mean +/- SD, n=3)")
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_stability(conditions: pd.DataFrame, destination: Path) -> None:
    plots = [
        ("Mean Vertical Centroid Peak-to-Peak_mm", "SD Vertical Centroid Peak-to-Peak_mm", "Vertical peak-to-peak (mm)"),
        ("Mean Pitch Peak-to-Peak_deg", "SD Pitch Peak-to-Peak_deg", "Pitch peak-to-peak (deg)"),
    ]
    colours = {"Low": "#2878B5", "High": "#D95F02"}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for axis, (mean_column, sd_column, ylabel) in zip(axes, plots):
        for friction in EXPECTED_FRICTION_LEVELS:
            group = conditions[conditions["Friction Level"].eq(friction)].sort_values(
                "Cycle Duration (s)"
            )
            axis.errorbar(
                group["Cycle Duration (s)"], group[mean_column],
                yerr=group[sd_column], marker="o", linewidth=2, capsize=4,
                label=f"{friction} friction", color=colours[friction],
            )
        axis.set_xlabel("Commanded cycle duration (s)")
        axis.set_ylabel(ylabel)
        axis.set_xticks(EXPECTED_CYCLE_DURATIONS_S)
        axis.grid(True, alpha=0.25)
    axes[0].legend()
    fig.suptitle("Object stability by friction and cycle duration")
    fig.tight_layout()
    fig.savefig(destination, dpi=200, bbox_inches="tight")
    plt.close(fig)


def definitions() -> pd.DataFrame:
    return pd.DataFrame([
        ("Experimental unit", "One complete video/run. Cycles and frames are repeated measurements, not independent DOE replicates."),
        ("Cycle-time multiplier", "Commanded duration divided by the 3 s phase-test duration: 0.5, 1 or 2."),
        ("Frequency multiplier", "3 s divided by commanded duration: 2, 1 or 0.5."),
        ("Beat frequency", "One divided by cycle duration, in Hz."),
        ("Complete cycle", "LED-to-LED interval within +/-20% of that run's commanded duration."),
        ("Transport time", "First accepted LED rising edge to interpolated leading-marker finish crossing."),
        ("Mean transport speed", "Net centroid travel divided by transport time."),
        ("Net advance per cycle", "Signed horizontal centroid displacement across a complete cycle."),
        ("Backward travel", "Sum of reverse increments exceeding the 0.20 mm movement-noise floor."),
        ("Two-way ANOVA", "Tests friction, cycle duration and their interaction using the 18 run-level observations."),
        ("Interaction", "A significant interaction means the effect of cycle duration depends on friction level."),
    ], columns=["Metric or term", "Definition"])


def write_workbook(
    summary: pd.DataFrame,
    conditions: pd.DataFrame,
    effects: pd.DataFrame,
    anova: pd.DataFrame,
    cycles: pd.DataFrame,
    rejected: pd.DataFrame,
    frames: pd.DataFrame,
    manifest: pd.DataFrame,
) -> None:
    with pd.ExcelWriter(WORKBOOK_PATH, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Run Summary", index=False)
        conditions.to_excel(writer, sheet_name="Condition Summary", index=False)
        effects.to_excel(writer, sheet_name="Main Effects", index=False)
        anova.to_excel(writer, sheet_name="Two-Way ANOVA", index=False)
        cycles.to_excel(writer, sheet_name="Cycle Data", index=False)
        rejected.to_excel(writer, sheet_name="Rejected Cycles", index=False)
        frames.to_excel(writer, sheet_name="Frame Data", index=False)
        manifest.to_excel(writer, sheet_name="Experiment Log", index=False)
        definitions().to_excel(writer, sheet_name="Metric Definitions", index=False)
    core.style_workbook(WORKBOOK_PATH)


def main() -> None:
    args = parse_args()
    calibration = core.load_calibration(core.CALIBRATION_NPZ)
    videos = discover_videos()
    manifest = create_or_update_manifest(videos)
    included = validate_manifest(manifest)
    first_video = VIDEO_DIR / str(included.iloc[0]["Video Filename"])
    led_roi = load_saved_led_roi(first_video, calibration, args.reset_led_roi)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, object]] = []
    all_cycles: list[pd.DataFrame] = []
    all_rejected: list[pd.DataFrame] = []
    all_frames: list[pd.DataFrame] = []
    for position, (_, row) in enumerate(included.iterrows(), 1):
        video_path = VIDEO_DIR / str(row["Video Filename"])
        test_id = str(row["Test ID"])
        friction = str(row["Friction Level"])
        expected_cycle = float(row["Cycle Duration (s)"])
        run_folder = OUTPUT_DIR / test_id
        run_folder.mkdir(parents=True, exist_ok=True)
        print(
            f"\n[{position}/{len(included)}] {video_path.name} | "
            f"{friction} friction | {expected_cycle:g} s"
        )
        notes: list[str] = []
        try:
            raw, metadata, marker_px = core.track_raw_video(
                video_path, calibration, led_roi
            )
            fps = float(metadata["fps"])
            _, blink_frames, led_threshold = detect_led_blinks_strict(raw, fps)
            if not blink_frames:
                raise RuntimeError("No high-confidence red LED blink detected.")
            start_frame = blink_frames[0]
            finish_frame, crossing_float = core.first_finish_crossing(
                marker_px[:, 0, :], start_frame, calibration.finish_line
            )
            if finish_frame is None:
                crossing_float = math.nan
                notes.append("Leading marker did not cross the calibrated finish line")
            frames, filled_marker_px = core.prepare_frame_data(
                test_id, raw, marker_px, calibration, fps, start_frame,
                finish_frame, crossing_float
            )
            cycles, rejected = variable_cycle_table(
                test_id, friction, expected_cycle, frames, blink_frames,
                start_frame, finish_frame, fps
            )
            if len(blink_frames) < 2:
                notes.append("Only one accepted LED edge; cycle metrics unavailable")
            if not rejected.empty:
                notes.append(
                    f"{len(rejected)} incomplete/malformed LED interval(s) "
                    "excluded from cycle metrics"
                )
            result = summarise_run(
                row, frames, cycles, fps, start_frame, finish_frame,
                crossing_float, notes
            )
            end = finish_frame if finish_frame is not None else len(frames) - 1
            analysis_frames = frames.iloc[start_frame:end + 1]
            analysis_frames.to_csv(run_folder / "frame_tracking.csv", index=False)
            cycles.to_csv(run_folder / "cycle_metrics.csv", index=False)
            rejected.to_csv(run_folder / "rejected_cycle_intervals.csv", index=False)
            pd.DataFrame([result]).to_csv(run_folder / "run_summary.csv", index=False)
            led_audit = raw[[
                "Frame Number", "Video Time (s)", "LED Score",
                "LED Score Smoothed", "LED Detected",
            ]].copy()
            led_audit["Detection Threshold"] = led_threshold
            led_audit["Required Peak Threshold"] = (
                led_threshold + LED_EXTRA_PEAK_MARGIN
            )
            led_audit.to_csv(run_folder / "led_detection.csv", index=False)
            core.save_run_plot(
                test_id, frames, start_frame, end,
                run_folder / "motion_diagnostics.png"
            )
            if WRITE_ANNOTATED_VIDEOS and not args.no_overlay:
                core.write_overlay(
                    video_path, run_folder / "tracking_overlay.mp4", calibration,
                    led_roi, frames, filled_marker_px, led_threshold,
                    start_frame, finish_frame
                )
            summaries.append(result)
            all_cycles.append(cycles)
            all_rejected.append(rejected)
            all_frames.append(analysis_frames)
            print(
                f"  {result['Quality-Control Status']} | "
                f"speed={result['Mean Transport Speed (mm/s)']:.2f} mm/s | "
                f"3-dot detection={result['Green Marker Detection Rate (%)']:.1f}%"
            )
        except Exception as error:
            print(f"  FAILED: {error}")
            failed = {column: math.nan for column in RUN_COLUMNS}
            failed.update({
                "Randomised Run Order": int(row["Randomised Run Order"]),
                "Test ID": test_id, "Video Filename": video_path.name,
                "Friction Level": friction,
                "Measured Friction Coefficient": row["Measured Friction Coefficient"],
                "Cycle Duration (s)": expected_cycle,
                "Cycle-Time Multiplier": expected_cycle / 3,
                "Beat Frequency (Hz)": 1 / expected_cycle,
                "Frequency Multiplier": 3 / expected_cycle,
                "Repeat Number": int(row["Repeat Number"]),
                "Phase Shift (deg)": int(row["Phase Shift (deg)"]),
                "Finish Crossing Detected": "No",
                "Quality-Control Status": "FAIL - processing error",
                "Notes": str(error),
            })
            summaries.append(failed)

    summary_df = pd.DataFrame(summaries, columns=RUN_COLUMNS).sort_values(
        "Randomised Run Order"
    )
    cycles_df = (
        pd.concat(all_cycles, ignore_index=True)
        if all_cycles else pd.DataFrame(columns=CYCLE_COLUMNS)
    )
    rejected_df = (
        pd.concat(all_rejected, ignore_index=True)
        if all_rejected else pd.DataFrame(columns=REJECTED_COLUMNS)
    )
    frames_df = (
        pd.concat(all_frames, ignore_index=True)
        if all_frames else pd.DataFrame(columns=core.FRAME_COLUMNS)
    )
    conditions_df = condition_summary(summary_df)
    effects_df = main_effects(summary_df)
    anova_df = two_way_anova(summary_df)
    write_workbook(
        summary_df, conditions_df, effects_df, anova_df, cycles_df,
        rejected_df, frames_df, manifest
    )
    plot_interactions(
        conditions_df, OUTPUT_DIR / "friction_speed_interaction.png"
    )
    plot_stability(
        conditions_df, OUTPUT_DIR / "friction_speed_stability.png"
    )
    summary_df.to_csv(OUTPUT_DIR / "run_summary.csv", index=False)
    conditions_df.to_csv(OUTPUT_DIR / "condition_summary.csv", index=False)
    effects_df.to_csv(OUTPUT_DIR / "main_effects.csv", index=False)
    anova_df.to_csv(OUTPUT_DIR / "two_way_anova.csv", index=False)

    review_count = int(summary_df["Quality-Control Status"].astype(str).str.contains(
        "REVIEW|FAIL", regex=True
    ).sum())
    print("\nFriction-speed analysis complete.")
    print(f"Workbook: {WORKBOOK_PATH}")
    print(f"Interaction chart: {OUTPUT_DIR / 'friction_speed_interaction.png'}")
    print(f"Stability chart: {OUTPUT_DIR / 'friction_speed_stability.png'}")
    print(f"Runs requiring review: {review_count} of {len(summary_df)}")
    print(f"Rejected LED intervals: {len(rejected_df)}")


if __name__ == "__main__":
    main()
