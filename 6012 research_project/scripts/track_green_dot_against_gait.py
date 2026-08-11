"""Track an 8 mm green cilium-tip marker and compare it with the gait table.

Recording sequence expected by this script:

* 0-2 s: keep the complete checkerboard and a 100 mm ruler span visible in the
  cilium motion plane;
* 2-4 s: remove the checkerboard and your hand;
* 4 s onward: track the green marker during gait entry and looping motion.

The checkerboard supplies the planar perspective mapping and an initial
pixel/mm scale. Two clicks on ruler marks 100 mm apart then correct that scale
in the cilium motion plane. The theoretical path is reconstructed from the current raw-PWM
``gait_table.h`` using the same two-link forward kinematics as the path
designer. The measured and commanded paths are compared geometrically, without
assuming a gait start time, cycle speed or phase. The checkerboard fixes the
axes, so alignment translates the commanded path only: it cannot rotate, flip,
mirror or resize the path.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================================
# USER SETTINGS
# ============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Replace this filename after copying the recording into camera_calibration.
VIDEO_PATH = Path(
    r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260808_181841.mp4"
)

GAIT_HEADER_PATH = SCRIPT_DIR.parent / "include" / "gait_table.h"

CAMERA_CALIBRATION_NPZ = SCRIPT_DIR / "checkerboard_calibration_outputs" / "camera_calibration.npz"

OUTPUT_ROOT = SCRIPT_DIR / "green_dot_tracking_outputs"

# Checkerboard: 10 x 8 squares gives 9 x 7 internal corners.
CHECKERBOARD_CORNERS = (9, 7)
CHECKERBOARD_SQUARE_MM = 10.0
CALIBRATION_START_S = 0.0
CALIBRATION_END_S = 2.0
CHECKERBOARD_FRAME_STEP = 2

# After checkerboard detection, click two ruler marks exactly this far apart.
# The ruler must be in the same plane as the green marker's motion.
RULER_REFERENCE_LENGTH_MM = 100.0

# Two seconds are deliberately left between calibration and tracking.
TRACKING_START_S = 6.0

# Artificial-cilium geometry and servo convention.
LOWER_LINK_LENGTH_MM = 50.0
UPPER_LINK_LENGTH_MM = 50.0
UPPER_COMMAND_TO_MECHANICAL_OFFSET_DEG = -90.0

# Lime-green marker settings. Adjust the HSV range if lighting changes.
EXPECTED_DOT_DIAMETER_MM = 8.0
MIN_DOT_DIAMETER_MM = 3.0
MAX_DOT_DIAMETER_MM = 15.0
MIN_CONTOUR_AREA_PX2 = 20.0
MIN_CIRCULARITY = 0.35
LOWER_GREEN_HSV = np.array([35, 70, 60], dtype=np.uint8)
UPPER_GREEN_HSV = np.array([90, 255, 255], dtype=np.uint8)

# After corner canonicalisation, checkerboard X always points image-right and
# checkerboard Y image-down. These fixed signs match the path orientation used
# by this test rig and no longer need changing between recordings.
MEASURED_X_SIGN = 1.0
MEASURED_Y_SIGN = -1.0

# Geometry-only registration settings. ICP refines translation only because the
# canonical checkerboard already establishes orientation. The closest 90% of
# detections control alignment so occasional false green blobs have less
# influence; every detection is still included in the reported error.
ICP_MAX_ITERATIONS = 50
ICP_KEEP_FRACTION = 0.90
ICP_CONVERGENCE_MM = 0.0001
MAX_ALIGNMENT_ROTATION_DEG = 0.0


@dataclass
class GaitTable:
    lower_pwm: np.ndarray
    upper_pwm: np.ndarray
    zero_degree_pwm: float
    counts_per_degree: float


@dataclass
class PlaneCalibration:
    image_to_checkerboard_mm: np.ndarray
    pixels_per_mm: float
    mm_per_pixel: float
    frame_index: int
    reprojection_error_mm: float


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        default=VIDEO_PATH,
        help="Video to analyse (defaults to the VIDEO_PATH placeholder).",
    )
    parser.add_argument(
        "--gait-header",
        type=Path,
        default=GAIT_HEADER_PATH,
        help="Raw-PWM gait_table.h exported by the path designer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output folder; defaults to a folder named after the video.",
    )
    parser.add_argument(
        "--skip-ruler",
        action="store_true",
        help=(
            "Use the checkerboard scale without the interactive 100 mm ruler "
            "correction (mainly for rerunning older recordings)."
        ),
    )
    return parser.parse_args()


def require_file(path: Path, description: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"{description} not found:\n{resolved}")
    return resolved


def parse_cpp_number(text: str, name: str) -> float:
    pattern = rf"\b{name}\s*=\s*([-+]?\d+(?:\.\d+)?)\s*[fFuUlL]*\s*;"
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Could not find {name} in the gait header.")
    return float(match.group(1))


def parse_cpp_array(text: str, name: str) -> np.ndarray:
    pattern = rf"\b{name}\s*\[[^\]]+\][^=]*=\s*\{{(.*?)\}}\s*;"
    match = re.search(pattern, text, flags=re.DOTALL)
    if match is None:
        raise ValueError(f"Could not find {name} in the gait header.")
    values = [int(value) for value in re.findall(r"\b\d+\b", match.group(1))]
    if not values:
        raise ValueError(f"{name} contains no numeric values.")
    return np.asarray(values, dtype=np.float64)


def load_gait_table(path: Path) -> GaitTable:
    text = path.read_text(encoding="utf-8")
    table_size = int(parse_cpp_number(text, "GAIT_TABLE_SIZE"))
    lower_pwm = parse_cpp_array(text, "LOWER_TABLE")
    upper_pwm = parse_cpp_array(text, "UPPER_TABLE")
    if len(lower_pwm) != table_size or len(upper_pwm) != table_size:
        raise ValueError(
            "The gait header table lengths do not match GAIT_TABLE_SIZE: "
            f"expected {table_size}, got {len(lower_pwm)} and {len(upper_pwm)}."
        )
    return GaitTable(
        lower_pwm=lower_pwm,
        upper_pwm=upper_pwm,
        zero_degree_pwm=parse_cpp_number(text, "GAIT_ZERO_DEG_COUNT"),
        counts_per_degree=parse_cpp_number(text, "GAIT_COUNTS_PER_DEG"),
    )


def gait_pwm_to_tip_mm(
    lower_pwm: np.ndarray,
    upper_pwm: np.ndarray,
    gait: GaitTable,
) -> tuple[np.ndarray, np.ndarray]:
    lower_deg = (lower_pwm - gait.zero_degree_pwm) / gait.counts_per_degree
    upper_deg = (upper_pwm - gait.zero_degree_pwm) / gait.counts_per_degree
    q1 = np.deg2rad(lower_deg)
    q2 = np.deg2rad(upper_deg + UPPER_COMMAND_TO_MECHANICAL_OFFSET_DEG)
    x_mm = (
        LOWER_LINK_LENGTH_MM * np.cos(q1)
        + UPPER_LINK_LENGTH_MM * np.cos(q1 + q2)
    )
    y_mm = (
        LOWER_LINK_LENGTH_MM * np.sin(q1)
        + UPPER_LINK_LENGTH_MM * np.sin(q1 + q2)
    )
    return x_mm, y_mm


def load_lens_calibration(
    calibration_path: Path,
    image_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    calibration = np.load(calibration_path)
    camera_matrix = calibration["camera_matrix"]
    distortion = calibration["dist_coeffs"]
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        image_size,
        alpha=1,
        newImgSize=image_size,
    )
    return camera_matrix, distortion, new_camera_matrix


def undistort_frame(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    new_camera_matrix: np.ndarray,
) -> np.ndarray:
    return cv2.undistort(
        frame,
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
    )


def detect_checkerboard(frame: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD_CORNERS,
        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        ),
    )
    if found:
        return corners.reshape(-1, 2).astype(np.float32)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_CORNERS,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.001,
    )
    refined = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return refined.reshape(-1, 2).astype(np.float32)


def canonicalize_checkerboard_corners(corners: np.ndarray) -> np.ndarray:
    """Make checkerboard columns image-right and rows image-down.

    OpenCV may choose the opposite end of a symmetrical checkerboard as its
    first corner. Canonicalising the detected grid removes that per-video
    180-degree ambiguity before constructing the millimetre homography.
    """

    columns, rows = CHECKERBOARD_CORNERS
    grid = corners.reshape(rows, columns, 2).copy()
    horizontal_direction = np.mean(grid[:, -1, :] - grid[:, 0, :], axis=0)
    if horizontal_direction[0] < 0.0:
        grid = grid[:, ::-1, :]
    vertical_direction = np.mean(grid[-1, :, :] - grid[0, :, :], axis=0)
    if vertical_direction[1] < 0.0:
        grid = grid[::-1, :, :]
    return np.ascontiguousarray(grid.reshape(-1, 2), dtype=np.float32)


def checkerboard_object_points() -> np.ndarray:
    columns, rows = CHECKERBOARD_CORNERS
    points = np.mgrid[0:columns, 0:rows].T.reshape(-1, 2).astype(np.float32)
    return points * CHECKERBOARD_SQUARE_MM


def estimate_pixels_per_mm(corners: np.ndarray) -> float:
    columns, rows = CHECKERBOARD_CORNERS
    grid = corners.reshape(rows, columns, 2)
    estimates: list[float] = []
    for row in range(rows):
        distances = np.linalg.norm(np.diff(grid[row, :, :], axis=0), axis=1)
        estimates.extend((distances / CHECKERBOARD_SQUARE_MM).tolist())
    for column in range(columns):
        distances = np.linalg.norm(np.diff(grid[:, column, :], axis=0), axis=1)
        estimates.extend((distances / CHECKERBOARD_SQUARE_MM).tolist())
    return float(np.median(estimates))


def calibrate_video_plane(
    capture: cv2.VideoCapture,
    fps: float,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    new_camera_matrix: np.ndarray,
    output_folder: Path,
) -> PlaneCalibration:
    start_frame = max(0, int(round(CALIBRATION_START_S * fps)))
    end_frame = int(round(CALIBRATION_END_S * fps))
    object_points = checkerboard_object_points()
    candidates: list[tuple[PlaneCalibration, np.ndarray, np.ndarray]] = []
    records: list[dict[str, float | int]] = []

    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for frame_index in range(start_frame, end_frame):
        success, frame = capture.read()
        if not success:
            break
        if (frame_index - start_frame) % CHECKERBOARD_FRAME_STEP != 0:
            continue
        processed = undistort_frame(
            frame, camera_matrix, distortion, new_camera_matrix
        )
        corners = detect_checkerboard(processed)
        if corners is None:
            continue
        corners = canonicalize_checkerboard_corners(corners)
        homography, _ = cv2.findHomography(corners, object_points, method=0)
        if homography is None:
            continue
        projected = cv2.perspectiveTransform(
            corners.reshape(-1, 1, 2), homography
        ).reshape(-1, 2)
        reprojection_error = float(
            np.sqrt(np.mean(np.sum((projected - object_points) ** 2, axis=1)))
        )
        pixels_per_mm = estimate_pixels_per_mm(corners)
        calibration = PlaneCalibration(
            image_to_checkerboard_mm=homography,
            pixels_per_mm=pixels_per_mm,
            mm_per_pixel=1.0 / pixels_per_mm,
            frame_index=frame_index,
            reprojection_error_mm=reprojection_error,
        )
        candidates.append((calibration, processed, corners))
        records.append(
            {
                "frame": frame_index,
                "time_s": frame_index / fps,
                "pixels_per_mm": pixels_per_mm,
                "mm_per_pixel": 1.0 / pixels_per_mm,
                "homography_reprojection_error_mm": reprojection_error,
            }
        )

    if not candidates:
        raise RuntimeError(
            "The 9 x 7 checkerboard was not detected in the first two seconds. "
            "Keep the complete board visible and approximately in the cilium "
            "motion plane during that interval."
        )

    scale_values = np.asarray([item[0].pixels_per_mm for item in candidates])
    median_scale = float(np.median(scale_values))
    # Prefer a geometrically accurate frame whose scale is representative of
    # all successful first-two-second detections.
    best_index = min(
        range(len(candidates)),
        key=lambda index: (
            candidates[index][0].reprojection_error_mm
            + 0.25
            * abs(candidates[index][0].pixels_per_mm - median_scale)
            / median_scale
        ),
    )
    best, preview, corners = candidates[best_index]
    best.pixels_per_mm = median_scale
    best.mm_per_pixel = 1.0 / median_scale

    preview = preview.copy()
    cv2.drawChessboardCorners(
        preview,
        CHECKERBOARD_CORNERS,
        corners.reshape(-1, 1, 2),
        True,
    )
    columns, rows = CHECKERBOARD_CORNERS
    corner_grid = corners.reshape(rows, columns, 2)
    origin = tuple(np.round(corner_grid[0, 0]).astype(int))
    x_tip = tuple(np.round(corner_grid[0, min(3, columns - 1)]).astype(int))
    y_tip = tuple(np.round(corner_grid[min(3, rows - 1), 0]).astype(int))
    cv2.arrowedLine(preview, origin, x_tip, (255, 255, 0), 3, tipLength=0.18)
    cv2.arrowedLine(preview, origin, y_tip, (0, 0, 255), 3, tipLength=0.18)
    cv2.putText(
        preview,
        "+X",
        x_tip,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 0),
        2,
    )
    cv2.putText(
        preview,
        "+Y",
        y_tip,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )
    cv2.putText(
        preview,
        f"Scale: {best.pixels_per_mm:.4f} px/mm",
        (25, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        (0, 255, 0),
        2,
    )
    cv2.imwrite(str(output_folder / "checkerboard_calibration.png"), preview)
    pd.DataFrame(records).to_csv(
        output_folder / "checkerboard_calibration.csv", index=False
    )
    return best


def select_two_ruler_marks(image: np.ndarray) -> np.ndarray:
    """Interactively select two ruler marks exactly 100 mm apart."""

    title = "Ruler scale correction - select two marks 100 mm apart"
    points: list[tuple[int, int]] = []

    def mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(1500, image.shape[1]), min(850, image.shape[0]))
    cv2.setMouseCallback(title, mouse)
    while True:
        display = image.copy()
        cv2.rectangle(display, (0, 0), (display.shape[1], 96), (0, 0, 0), -1)
        instructions = [
            "Click two ruler marks exactly 100 mm apart in the green-dot motion plane.",
            "Enter = accept | Backspace = undo | Esc = cancel",
        ]
        for row, line in enumerate(instructions):
            cv2.putText(
                display,
                line,
                (15, 33 + row * 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for number, point in enumerate(points, 1):
            cv2.circle(display, point, 9, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(
                display,
                str(number),
                (point[0] + 12, point[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
        if len(points) == 2:
            cv2.line(display, points[0], points[1], (0, 255, 255), 3, cv2.LINE_AA)
        cv2.imshow(title, display)
        key = cv2.waitKey(25) & 0xFF
        if key in (8, 127) and points:
            points.pop()
        elif key in (10, 13) and len(points) == 2:
            break
        elif key == 27:
            cv2.destroyWindow(title)
            raise KeyboardInterrupt("Ruler selection cancelled.")
    cv2.destroyWindow(title)
    return np.asarray(points, dtype=np.float32)


def apply_ruler_scale_correction(
    capture: cv2.VideoCapture,
    plane: PlaneCalibration,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    new_camera_matrix: np.ndarray,
    output_folder: Path,
) -> dict[str, float]:
    """Correct the checkerboard homography using a clicked 100 mm reference."""

    capture.set(cv2.CAP_PROP_POS_FRAMES, plane.frame_index)
    success, raw = capture.read()
    if not success:
        raise RuntimeError(
            f"Could not read ruler-selection frame {plane.frame_index}."
        )
    frame = undistort_frame(raw, camera_matrix, distortion, new_camera_matrix)
    points_px = select_two_ruler_marks(frame)
    points_mm_before = cv2.perspectiveTransform(
        points_px.reshape(-1, 1, 2), plane.image_to_checkerboard_mm
    ).reshape(-1, 2)
    measured_before_mm = float(
        np.linalg.norm(points_mm_before[1] - points_mm_before[0])
    )
    if measured_before_mm <= 1e-6:
        raise ValueError("The two ruler clicks must be at different positions.")

    correction = RULER_REFERENCE_LENGTH_MM / measured_before_mm
    plane.image_to_checkerboard_mm[:2, :] *= correction
    plane.mm_per_pixel *= correction
    plane.pixels_per_mm /= correction

    points_mm_after = cv2.perspectiveTransform(
        points_px.reshape(-1, 1, 2), plane.image_to_checkerboard_mm
    ).reshape(-1, 2)
    measured_after_mm = float(
        np.linalg.norm(points_mm_after[1] - points_mm_after[0])
    )

    annotated = frame.copy()
    first = tuple(np.round(points_px[0]).astype(int))
    second = tuple(np.round(points_px[1]).astype(int))
    cv2.line(annotated, first, second, (0, 255, 255), 4, cv2.LINE_AA)
    for number, point in enumerate((first, second), 1):
        cv2.circle(annotated, point, 10, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.putText(
            annotated,
            str(number),
            (point[0] + 12, point[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )
    label = (
        f"Checkerboard measured {measured_before_mm:.2f} mm; "
        f"scale x {correction:.6f}; corrected to {measured_after_mm:.2f} mm"
    )
    cv2.rectangle(annotated, (0, 0), (annotated.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(
        annotated,
        label,
        (15, 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(output_folder / "ruler_scale_calibration.png"), annotated):
        raise RuntimeError("Could not save the ruler calibration preview.")

    result = {
        "ruler_reference_length_mm": RULER_REFERENCE_LENGTH_MM,
        "checkerboard_measured_ruler_length_before_correction_mm": measured_before_mm,
        "ruler_scale_correction_factor": correction,
        "corrected_ruler_length_mm": measured_after_mm,
        "corrected_pixels_per_mm": plane.pixels_per_mm,
        "corrected_mm_per_pixel": plane.mm_per_pixel,
        "ruler_point_1_x_px": float(points_px[0, 0]),
        "ruler_point_1_y_px": float(points_px[0, 1]),
        "ruler_point_2_x_px": float(points_px[1, 0]),
        "ruler_point_2_y_px": float(points_px[1, 1]),
    }
    pd.DataFrame([result]).to_csv(
        output_folder / "ruler_scale_calibration.csv", index=False
    )

    checkerboard_csv = output_folder / "checkerboard_calibration.csv"
    checkerboard_records = pd.read_csv(checkerboard_csv)
    checkerboard_records["ruler_scale_correction_factor"] = correction
    checkerboard_records["corrected_pixels_per_mm"] = (
        checkerboard_records["pixels_per_mm"] / correction
    )
    checkerboard_records["corrected_mm_per_pixel"] = (
        checkerboard_records["mm_per_pixel"] * correction
    )
    checkerboard_records.to_csv(checkerboard_csv, index=False)
    return result


def find_green_marker(
    frame: np.ndarray,
    mm_per_pixel: float,
    previous_position: tuple[float, float] | None,
) -> dict[str, float] | None:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN_HSV, UPPER_GREEN_HSV)
    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    candidates: list[tuple[float, dict[str, float]]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < MIN_CONTOUR_AREA_PX2 or perimeter <= 0.0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        equivalent_diameter_px = 2.0 * np.sqrt(area / np.pi)
        diameter_mm = equivalent_diameter_px * mm_per_pixel
        if not MIN_DOT_DIAMETER_MM <= diameter_mm <= MAX_DOT_DIAMETER_MM:
            continue
        if circularity < MIN_CIRCULARITY:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0.0:
            continue
        x_px = float(moments["m10"] / moments["m00"])
        y_px = float(moments["m01"] / moments["m00"])
        size_error = abs(diameter_mm - EXPECTED_DOT_DIAMETER_MM)
        continuity_penalty = 0.0
        if previous_position is not None:
            continuity_penalty = 0.02 * np.hypot(
                x_px - previous_position[0], y_px - previous_position[1]
            )
        score = size_error + continuity_penalty - 0.25 * circularity
        candidates.append(
            (
                score,
                {
                    "x_px": x_px,
                    "y_px": y_px,
                    "area_px2": area,
                    "diameter_px": float(equivalent_diameter_px),
                    "diameter_mm": float(diameter_mm),
                    "circularity": float(circularity),
                },
            )
        )
    return min(candidates, key=lambda item: item[0])[1] if candidates else None


def image_point_to_plane_mm(
    x_px: float,
    y_px: float,
    homography: np.ndarray,
) -> tuple[float, float]:
    image_point = np.asarray([[[x_px, y_px]]], dtype=np.float32)
    plane_point = cv2.perspectiveTransform(image_point, homography)[0, 0]
    return (
        MEASURED_X_SIGN * float(plane_point[0]),
        MEASURED_Y_SIGN * float(plane_point[1]),
    )


def track_video(
    capture: cv2.VideoCapture,
    fps: float,
    frame_count: int,
    frame_size: tuple[int, int],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    new_camera_matrix: np.ndarray,
    plane: PlaneCalibration,
    output_folder: Path,
) -> pd.DataFrame:
    start_frame = int(round(TRACKING_START_S * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    overlay_path = output_folder / "green_dot_tracking_overlay.mp4"
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        frame_size,
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create overlay video:\n{overlay_path}")

    results: list[dict[str, float | int]] = []
    previous_position: tuple[float, float] | None = None
    trail: list[tuple[int, int]] = []
    frame_index = start_frame
    while frame_index < frame_count:
        success, frame = capture.read()
        if not success:
            break
        processed = undistort_frame(
            frame, camera_matrix, distortion, new_camera_matrix
        )
        marker = find_green_marker(
            processed, plane.mm_per_pixel, previous_position
        )
        overlay = processed.copy()
        record: dict[str, float | int] = {
            "frame": frame_index,
            "video_time_s": frame_index / fps,
            "tracking_time_s": frame_index / fps - TRACKING_START_S,
            "x_px": np.nan,
            "y_px": np.nan,
            "x_measured_mm": np.nan,
            "y_measured_mm": np.nan,
            "dot_diameter_mm": np.nan,
            "dot_circularity": np.nan,
        }
        if marker is not None:
            x_px = marker["x_px"]
            y_px = marker["y_px"]
            x_mm, y_mm = image_point_to_plane_mm(
                x_px, y_px, plane.image_to_checkerboard_mm
            )
            record.update(
                {
                    "x_px": x_px,
                    "y_px": y_px,
                    "x_measured_mm": x_mm,
                    "y_measured_mm": y_mm,
                    "dot_diameter_mm": marker["diameter_mm"],
                    "dot_circularity": marker["circularity"],
                }
            )
            centre = (int(round(x_px)), int(round(y_px)))
            previous_position = (x_px, y_px)
            trail.append(centre)
            trail = trail[-500:]
            cv2.circle(
                overlay,
                centre,
                max(3, int(round(marker["diameter_px"] / 2.0))),
                (0, 255, 0),
                2,
            )
            cv2.circle(overlay, centre, 3, (0, 0, 255), -1)
            status = (
                f"dot {marker['diameter_mm']:.1f} mm | "
                f"x={x_mm:.1f}, y={y_mm:.1f} mm"
            )
            colour = (0, 255, 0)
        else:
            status = "green dot not detected"
            colour = (0, 0, 255)
        for first, second in zip(trail, trail[1:]):
            cv2.line(overlay, first, second, (0, 200, 0), 1)
        cv2.putText(
            overlay,
            f"t={record['tracking_time_s']:.2f} s | {status}",
            (20, 38),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            colour,
            2,
        )
        writer.write(overlay)
        results.append(record)
        frame_index += 1

    writer.release()
    return pd.DataFrame(results)


def nearest_points_on_closed_path(
    measured_points: np.ndarray,
    path_points: np.ndarray,
    chunk_size: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the closest point on any commanded-path segment."""

    segment_start = path_points
    segment_vector = np.roll(path_points, -1, axis=0) - segment_start
    segment_length_squared = np.sum(segment_vector**2, axis=1)
    segment_length_squared = np.maximum(segment_length_squared, 1e-12)
    nearest = np.empty_like(measured_points, dtype=np.float64)
    distances = np.empty(len(measured_points), dtype=np.float64)

    for start in range(0, len(measured_points), chunk_size):
        stop = min(start + chunk_size, len(measured_points))
        points = measured_points[start:stop]
        point_from_start = points[:, None, :] - segment_start[None, :, :]
        fraction = np.sum(
            point_from_start * segment_vector[None, :, :], axis=2
        ) / segment_length_squared[None, :]
        fraction = np.clip(fraction, 0.0, 1.0)
        projected = (
            segment_start[None, :, :]
            + fraction[:, :, None] * segment_vector[None, :, :]
        )
        distance_squared = np.sum(
            (points[:, None, :] - projected) ** 2, axis=2
        )
        best_segment = np.argmin(distance_squared, axis=1)
        row_index = np.arange(len(points))
        nearest[start:stop] = projected[row_index, best_segment]
        distances[start:stop] = np.sqrt(
            distance_squared[row_index, best_segment]
        )
    return nearest, distances


def rigid_transform(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Find the no-scale, no-reflection rigid transform source -> target."""

    source_centre = np.mean(source, axis=0)
    target_centre = np.mean(target, axis=0)
    covariance = (source - source_centre).T @ (target - target_centre)
    left, _singular_values, right_transpose = np.linalg.svd(covariance)
    rotation = right_transpose.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_transpose[-1, :] *= -1.0
        rotation = right_transpose.T @ left.T
    translation = target_centre - rotation @ source_centre
    return rotation, translation


def align_commanded_path(
    commanded_path: np.ndarray,
    measured_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Align the complete commanded curve without changing its orientation."""

    measured_centre = np.median(measured_points, axis=0)
    commanded_centre = np.median(commanded_path, axis=0)
    # The checkerboard already fixes the physical axes. Starting at zero and
    # enforcing the configured zero-degree limit prevents all path rotation.
    initial_angles = (0.0,)
    best_result: tuple[np.ndarray, np.ndarray, np.ndarray, float] | None = None

    for initial_angle in initial_angles:
        cosine = np.cos(initial_angle)
        sine = np.sin(initial_angle)
        total_rotation = np.array(
            [[cosine, -sine], [sine, cosine]], dtype=np.float64
        )
        total_translation = measured_centre - total_rotation @ commanded_centre
        transformed = (
            (total_rotation @ commanded_path.T).T + total_translation
        )
        previous_rmse = np.inf

        for _iteration in range(ICP_MAX_ITERATIONS):
            nearest, distances = nearest_points_on_closed_path(
                measured_points, transformed
            )
            keep_count = max(
                3, int(np.ceil(ICP_KEEP_FRACTION * len(measured_points)))
            )
            kept = np.argpartition(distances, keep_count - 1)[:keep_count]
            incremental_rotation, incremental_translation = rigid_transform(
                nearest[kept], measured_points[kept]
            )
            candidate_total_rotation = incremental_rotation @ total_rotation
            candidate_angle_deg = float(
                np.rad2deg(
                    np.arctan2(
                        candidate_total_rotation[1, 0],
                        candidate_total_rotation[0, 0],
                    )
                )
            )
            if (
                MAX_ALIGNMENT_ROTATION_DEG <= 0.0
                or abs(candidate_angle_deg) > MAX_ALIGNMENT_ROTATION_DEG
            ):
                # Retain translation refinement at the rotation boundary, but
                # reject an update that would rotate the path toward a flipped
                # local minimum.
                incremental_rotation = np.eye(2, dtype=np.float64)
                incremental_translation = np.mean(
                    measured_points[kept] - nearest[kept], axis=0
                )
            transformed = (
                (incremental_rotation @ transformed.T).T
                + incremental_translation
            )
            total_rotation = incremental_rotation @ total_rotation
            total_translation = (
                incremental_rotation @ total_translation
                + incremental_translation
            )
            current_rmse = float(
                np.sqrt(np.mean(np.sort(distances)[:keep_count] ** 2))
            )
            if abs(previous_rmse - current_rmse) < ICP_CONVERGENCE_MM:
                break
            previous_rmse = current_rmse

        _nearest, final_distances = nearest_points_on_closed_path(
            measured_points, transformed
        )
        final_rmse = float(np.sqrt(np.mean(final_distances**2)))
        result = (
            transformed,
            total_rotation,
            total_translation,
            final_rmse,
        )
        if best_result is None or final_rmse < best_result[3]:
            best_result = result

    if best_result is None:
        raise RuntimeError("Commanded-path alignment failed.")
    return best_result


def compare_with_theory(
    raw_tracking: pd.DataFrame,
    gait: GaitTable,
    output_folder: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid = raw_tracking.dropna(
        subset=["x_measured_mm", "y_measured_mm"]
    ).copy()
    if len(valid) < 10:
        raise RuntimeError(
            "Fewer than ten valid green-dot positions were detected. Check the "
            "HSV limits, lighting, marker size and camera view."
        )

    full_x, full_y = gait_pwm_to_tip_mm(
        gait.lower_pwm, gait.upper_pwm, gait
    )
    commanded_path = np.column_stack((full_x, full_y))
    measured_points = valid[["x_measured_mm", "y_measured_mm"]].to_numpy()
    (
        aligned_path,
        alignment_rotation,
        alignment_translation,
        _alignment_rmse,
    ) = align_commanded_path(commanded_path, measured_points)
    nearest_commanded, path_deviation = nearest_points_on_closed_path(
        measured_points, aligned_path
    )
    valid["nearest_commanded_x_mm"] = nearest_commanded[:, 0]
    valid["nearest_commanded_y_mm"] = nearest_commanded[:, 1]
    valid["x_path_deviation_mm"] = (
        valid["x_measured_mm"] - valid["nearest_commanded_x_mm"]
    )
    valid["y_path_deviation_mm"] = (
        valid["y_measured_mm"] - valid["nearest_commanded_y_mm"]
    )
    valid["path_deviation_mm"] = path_deviation
    valid.to_csv(output_folder / "measured_vs_expected.csv", index=False)

    detection_percent = 100.0 * len(valid) / max(1, len(raw_tracking))
    alignment_angle_deg = float(
        np.rad2deg(
            np.arctan2(alignment_rotation[1, 0], alignment_rotation[0, 0])
        )
    )
    summary = pd.DataFrame(
        [
            {
                "total_tracking_frames": len(raw_tracking),
                "detected_tracking_frames": len(valid),
                "valid_detection_percent": detection_percent,
                "comparison_is_time_synchronised": False,
                "checkerboard_axes_canonicalised": True,
                "measured_x_sign": MEASURED_X_SIGN,
                "measured_y_sign": MEASURED_Y_SIGN,
                "maximum_allowed_alignment_rotation_deg":
                    MAX_ALIGNMENT_ROTATION_DEG,
                "alignment_rotation_deg": alignment_angle_deg,
                "alignment_x_translation_mm": alignment_translation[0],
                "alignment_y_translation_mm": alignment_translation[1],
                "alignment_scale_factor": 1.0,
                "mean_path_deviation_mm": valid["path_deviation_mm"].mean(),
                "rmse_path_deviation_mm": np.sqrt(
                    np.mean(valid["path_deviation_mm"] ** 2)
                ),
                "95th_percentile_path_deviation_mm": valid[
                    "path_deviation_mm"
                ].quantile(0.95),
                "maximum_path_deviation_mm": valid[
                    "path_deviation_mm"
                ].max(),
                "mean_detected_dot_diameter_mm": valid[
                    "dot_diameter_mm"
                ].mean(),
            }
        ]
    )
    summary.to_csv(output_folder / "tracking_summary.csv", index=False)

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.7))
    axes[0].plot(
        np.append(aligned_path[:, 0], aligned_path[0, 0]),
        np.append(aligned_path[:, 1], aligned_path[0, 1]),
        color="#e67e22",
        linewidth=2.2,
        label="Path commanded by gait table",
    )
    measured_scatter = axes[0].scatter(
        valid["x_measured_mm"],
        valid["y_measured_mm"],
        c=valid["tracking_time_s"],
        cmap="viridis",
        s=9,
        alpha=0.75,
        label="Measured green-dot positions",
    )
    axes[0].set_xlabel("X position (mm)")
    axes[0].set_ylabel("Y position (mm)")
    axes[0].set_title("Measured and commanded cilium-tip paths")
    axes[0].axis("equal")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()
    figure.colorbar(
        measured_scatter,
        ax=axes[0],
        label="Tracking time (s)",
        fraction=0.046,
        pad=0.04,
    )

    axes[1].plot(
        valid["tracking_time_s"],
        valid["path_deviation_mm"],
        color="#2c7fb8",
        linewidth=1.2,
    )
    axes[1].axhline(
        summary.loc[0, "mean_path_deviation_mm"],
        color="#d95f0e",
        linestyle="--",
        label="Mean error",
    )
    axes[1].set_xlabel("Tracking time (s)")
    axes[1].set_ylabel("Shortest path deviation (mm)")
    axes[1].set_title("Geometric deviation from commanded path")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    figure.suptitle("Geometry-only comparison (independent of gait timing and phase)")
    figure.tight_layout()
    figure.savefig(
        output_folder / "measured_vs_commanded_path.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.close(figure)
    return valid, summary


def main() -> None:
    arguments = parse_arguments()
    video_path = require_file(arguments.video, "Video")
    gait_header = require_file(arguments.gait_header, "Gait header")
    calibration_path = require_file(
        CAMERA_CALIBRATION_NPZ, "Camera lens-calibration file"
    )
    output_folder = (
        arguments.output.expanduser().resolve()
        if arguments.output is not None
        else OUTPUT_ROOT / video_path.stem
    )
    output_folder.mkdir(parents=True, exist_ok=True)

    gait = load_gait_table(gait_header)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video:\n{video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if fps <= 0.0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("The video reports invalid frame-rate or dimensions.")
    if frame_count / fps <= TRACKING_START_S:
        capture.release()
        raise RuntimeError(
            f"The video ends before tracking starts at {TRACKING_START_S:.1f} s."
        )

    camera_matrix, distortion, new_camera_matrix = load_lens_calibration(
        calibration_path, (width, height)
    )
    print(f"Video: {video_path.name}")
    print(f"Resolution: {width} x {height} at {fps:.3f} fps")
    print(f"Gait samples: {len(gait.lower_pwm)}")
    print("Calibrating checkerboard scale from the first two seconds...")
    plane = calibrate_video_plane(
        capture,
        fps,
        camera_matrix,
        distortion,
        new_camera_matrix,
        output_folder,
    )
    print(
        f"Checkerboard-only scale: {plane.pixels_per_mm:.5f} px/mm "
        f"({plane.mm_per_pixel:.5f} mm/px)"
    )
    ruler_result: dict[str, float] | None = None
    if not arguments.skip_ruler:
        print(
            "Click two ruler marks exactly "
            f"{RULER_REFERENCE_LENGTH_MM:.0f} mm apart, then press Enter..."
        )
        ruler_result = apply_ruler_scale_correction(
            capture,
            plane,
            camera_matrix,
            distortion,
            new_camera_matrix,
            output_folder,
        )
        print(
            "Checkerboard measured the clicked ruler span as "
            f"{ruler_result['checkerboard_measured_ruler_length_before_correction_mm']:.3f} mm"
        )
        print(
            f"Applied ruler scale correction x "
            f"{ruler_result['ruler_scale_correction_factor']:.6f}"
        )
        print(
            f"Corrected scale: {plane.pixels_per_mm:.5f} px/mm "
            f"({plane.mm_per_pixel:.5f} mm/px)"
        )
    else:
        print("Ruler correction skipped; using the checkerboard-only scale.")
    print(f"Tracking the green dot from {TRACKING_START_S:.1f} s...")
    raw = track_video(
        capture,
        fps,
        frame_count,
        (width, height),
        camera_matrix,
        distortion,
        new_camera_matrix,
        plane,
        output_folder,
    )
    capture.release()
    raw.to_csv(output_folder / "green_dot_raw_tracking.csv", index=False)
    valid, summary = compare_with_theory(raw, gait, output_folder)
    summary["ruler_scale_correction_applied"] = ruler_result is not None
    summary["ruler_reference_length_mm"] = RULER_REFERENCE_LENGTH_MM
    summary["ruler_scale_correction_factor"] = (
        ruler_result["ruler_scale_correction_factor"]
        if ruler_result is not None else 1.0
    )
    summary["final_pixels_per_mm"] = plane.pixels_per_mm
    summary["final_mm_per_pixel"] = plane.mm_per_pixel
    summary.to_csv(output_folder / "tracking_summary.csv", index=False)

    print(
        f"Valid detections: {int(summary.loc[0, 'detected_tracking_frames'])}"
        f"/{len(raw)} ({summary.loc[0, 'valid_detection_percent']:.1f}%)"
    )
    print(f"Geometry-comparison points: {len(valid)}")
    print(
        f"Mean shortest-path deviation: "
        f"{summary.loc[0, 'mean_path_deviation_mm']:.3f} mm"
    )
    print(f"Outputs saved to:\n{output_folder.resolve()}")


if __name__ == "__main__":
    main()
