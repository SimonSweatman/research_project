"""Create one reusable calibration for the full DOE recording session.

Workflow
--------
1. Set VIDEO_PATH to the wide-view calibration recording.
2. Run once. A timestamp contact sheet is always written.
3. Use the contact sheet to fill CHECKERBOARD_WINDOWS and
   REFERENCE_FRAME_TIME_S below, then run again.
4. Click the requested cilia pivot centres and the two ends of the LEFT EDGE
   of the finish marker. Press Enter to accept or Backspace to undo a click.

Only frames inside CHECKERBOARD_WINDOWS are used. Movement while the board is
being repositioned is therefore ignored, and a robust stability filter removes
remaining blurred/moving outliers inside each chosen window.

The saved NPZ is intended to be loaded by the later three-dot DOE tracker.
The JSON and PNG files are human-readable checks of the same calibration.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


# =============================================================================
# EDIT THESE SETTINGS AFTER THE CALIBRATION VIDEO HAS BEEN ADDED
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Example:
# VIDEO_PATH = Path(r"C:\full\path\to\your_calibration_video.mp4")
VIDEO_PATH: Path | None = None

# Add only periods during which the complete checkerboard is held still.
# Moving/repositioning periods between these windows are deliberately ignored.
# Use as many windows and locations as are useful.
CHECKERBOARD_WINDOWS: list[tuple[str, float, float]] = [
    # ("left", 2.0, 4.0),
    # ("centre", 7.0, 9.0),
    # ("right", 12.0, 14.0),
]

# Pick a clear frame after the checkerboard/your hand has left the view. This
# frame is used for clicking the pivot reference and finish line.
REFERENCE_FRAME_TIME_S: float | None = None

# Optional stationary windows in which all three green dots are unobscured.
# Leave empty if the reference frame already shows the dots clearly.
GREEN_MARKER_WINDOWS: list[tuple[str, float, float]] = [
    # ("markers_at_start", 17.0, 18.0),
]


# =============================================================================
# PHYSICAL AND DETECTION SETTINGS
# =============================================================================

CAMERA_CALIBRATION_NPZ = (
    SCRIPT_DIR / "checkerboard_calibration_outputs" / "camera_calibration.npz"
)
OUTPUT_DIR = SCRIPT_DIR / "doe_session_calibration_outputs"

# Number of INTERNAL checkerboard corners and physical square side length.
CHECKERBOARD_CORNERS = (9, 7)
CHECKERBOARD_SQUARE_MM = 10.0

# Click two lower-pivot centres with these cilia numbers, in this order.
# With cilia 1 and 12 this validates 11 gaps x 87 mm = 957 mm.
PIVOT_A_CILIUM = 1
PIVOT_B_CILIUM = 12
CILIA_SPACING_MM = 87.0
APPLY_PIVOT_SCALE_CORRECTION = True

# Change this to the diameter actually used on the 50 cm test piece.
GREEN_DOT_DIAMETER_MM = 25.0
GREEN_HSV_LOWER = (35, 65, 45)
GREEN_HSV_UPPER = (95, 255, 255)

TRAVEL_DIRECTION = "left"
CONTACT_SHEET_INTERVAL_S = 1.0
CHECKERBOARD_SAMPLE_INTERVAL_S = 0.10
GREEN_SAMPLE_INTERVAL_S = 0.10
CONTACT_THUMBNAIL_WIDTH = 420


@dataclass
class CheckerObservation:
    label: str
    frame_index: int
    time_s: float
    corners: np.ndarray
    sharpness: float
    centre_px: np.ndarray
    pixels_per_mm: float
    normal_camera: np.ndarray
    plane_distance_mm: float
    reprojection_error_px: float
    stability_score_px: float = math.nan
    accepted: bool = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "video",
        nargs="?",
        type=Path,
        help="Calibration video; overrides VIDEO_PATH at the top of the file.",
    )
    parser.add_argument(
        "--contact-sheet-only",
        action="store_true",
        help="Write the timestamp sheet and stop before calibration.",
    )
    return parser.parse_args()


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video metadata could not be read reliably.")
    return capture, fps, frame_count, width, height


def read_frame_at(
    capture: cv2.VideoCapture, time_s: float, fps: float
) -> tuple[int, np.ndarray]:
    frame_index = max(0, int(round(time_s * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read the frame at {time_s:.3f} s.")
    return frame_index, frame


def load_lens_calibration(
    path: Path, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Lens calibration file not found: {path}")
    with np.load(path) as data:
        camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(data["dist_coeffs"], dtype=np.float64)
    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix, distortion, image_size, 1.0, image_size
    )
    return camera_matrix, distortion, new_camera_matrix


def undistort(
    frame: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    new_camera_matrix: np.ndarray,
) -> np.ndarray:
    return cv2.undistort(frame, camera_matrix, distortion, None, new_camera_matrix)


def checker_object_points_3d() -> np.ndarray:
    columns, rows = CHECKERBOARD_CORNERS
    points = np.zeros((columns * rows, 3), np.float32)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows].T.reshape(-1, 2) * CHECKERBOARD_SQUARE_MM
    )
    return points


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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.001)
    return cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria).reshape(
        -1, 2
    ).astype(np.float32)


def canonicalize_corners(corners: np.ndarray) -> np.ndarray:
    columns, rows = CHECKERBOARD_CORNERS
    grid = corners.reshape(rows, columns, 2).copy()
    if np.mean(grid[:, -1, 0] - grid[:, 0, 0]) < 0:
        grid = grid[:, ::-1, :]
    if np.mean(grid[-1, :, 1] - grid[0, :, 1]) < 0:
        grid = grid[::-1, :, :]
    return np.ascontiguousarray(grid.reshape(-1, 2), dtype=np.float32)


def estimate_pixels_per_mm(corners: np.ndarray) -> float:
    columns, rows = CHECKERBOARD_CORNERS
    grid = corners.reshape(rows, columns, 2)
    lengths: list[float] = []
    for row in grid:
        lengths.extend(np.linalg.norm(np.diff(row, axis=0), axis=1).tolist())
    for column in range(columns):
        lengths.extend(
            np.linalg.norm(np.diff(grid[:, column, :], axis=0), axis=1).tolist()
        )
    return float(np.median(lengths) / CHECKERBOARD_SQUARE_MM)


def robust_limit(values: np.ndarray, multiplier: float = 3.5) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + multiplier * max(1.4826 * mad, 1e-9)


def collect_checker_observations(
    capture: cv2.VideoCapture,
    fps: float,
    windows: Iterable[tuple[str, float, float]],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    new_camera_matrix: np.ndarray,
) -> list[CheckerObservation]:
    object_points = checker_object_points_3d()
    observations: list[CheckerObservation] = []
    step = max(1, int(round(CHECKERBOARD_SAMPLE_INTERVAL_S * fps)))

    for label, start_s, end_s in windows:
        if end_s <= start_s:
            raise ValueError(f"Invalid checkerboard window {label!r}: end <= start")
        start_frame = max(0, int(math.floor(start_s * fps)))
        end_frame = int(math.ceil(end_s * fps))
        for frame_index in range(start_frame, end_frame + 1, step):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, raw = capture.read()
            if not ok:
                continue
            frame = undistort(raw, camera_matrix, distortion, new_camera_matrix)
            corners = detect_checkerboard(frame)
            if corners is None:
                continue
            corners = canonicalize_corners(corners)
            ok_pose, rvec, tvec = cv2.solvePnP(
                object_points,
                corners.reshape(-1, 1, 2),
                new_camera_matrix,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok_pose:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            normal = rotation[:, 2].astype(np.float64)
            distance = float(np.dot(normal, tvec.reshape(3)))
            if distance < 0:
                normal = -normal
                distance = -distance
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, new_camera_matrix, None
            )
            error = float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            (projected.reshape(-1, 2) - corners) ** 2, axis=1
                        )
                    )
                )
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            observations.append(
                CheckerObservation(
                    label=label,
                    frame_index=frame_index,
                    time_s=frame_index / fps,
                    corners=corners,
                    sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    centre_px=np.mean(corners, axis=0),
                    pixels_per_mm=estimate_pixels_per_mm(corners),
                    normal_camera=normal,
                    plane_distance_mm=distance,
                    reprojection_error_px=error,
                )
            )

    if not observations:
        raise RuntimeError(
            "No checkerboard detections were found in CHECKERBOARD_WINDOWS. "
            "Check the timestamps and ensure all 9 x 7 internal corners are visible."
        )

    # Score movement separately inside each named stationary window. Frames far
    # from that window's median corner positions are likely hand/motion frames.
    for label in {item.label for item in observations}:
        group = [item for item in observations if item.label == label]
        median_corners = np.median(
            np.stack([item.corners for item in group]), axis=0
        )
        scores = np.asarray(
            [
                np.median(np.linalg.norm(item.corners - median_corners, axis=1))
                for item in group
            ],
            dtype=float,
        )
        stability_limit = robust_limit(scores)
        sharpness_values = np.asarray([item.sharpness for item in group])
        sharpness_floor = max(15.0, 0.45 * float(np.median(sharpness_values)))
        error_values = np.asarray([item.reprojection_error_px for item in group])
        error_limit = max(0.75, robust_limit(error_values))
        for item, score in zip(group, scores):
            item.stability_score_px = float(score)
            item.accepted = bool(
                score <= stability_limit
                and item.sharpness >= sharpness_floor
                and item.reprojection_error_px <= error_limit
            )

    if sum(item.accepted for item in observations) < 3:
        raise RuntimeError(
            "Fewer than three stable checkerboard frames survived filtering. "
            "Use longer stationary timestamp windows."
        )
    return observations


def average_plane_pose(
    observations: list[CheckerObservation], new_camera_matrix: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    accepted = [item for item in observations if item.accepted]
    normals = np.stack([item.normal_camera for item in accepted])
    distances = np.asarray([item.plane_distance_mm for item in accepted])

    provisional_normal = np.mean(normals, axis=0)
    provisional_normal /= np.linalg.norm(provisional_normal)
    angle_errors = np.degrees(
        np.arccos(np.clip(normals @ provisional_normal, -1.0, 1.0))
    )
    distance_errors = np.abs(distances - np.median(distances))
    angle_limit = max(0.35, robust_limit(angle_errors))
    distance_limit = max(2.0, robust_limit(distance_errors))
    keep = (angle_errors <= angle_limit) & (distance_errors <= distance_limit)
    if int(np.sum(keep)) < 3:
        keep[:] = True

    normal = np.mean(normals[keep], axis=0)
    normal /= np.linalg.norm(normal)
    distance = float(np.median(distances[keep]))

    # Define global plane X as image-right and Y as image-down. This avoids the
    # checkerboard's arbitrary location/orientation becoming the DOE origin.
    camera_x = np.array([1.0, 0.0, 0.0])
    axis_x = camera_x - normal * float(np.dot(camera_x, normal))
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    if float(np.dot(axis_y, np.array([0.0, 1.0, 0.0]))) < 0:
        axis_y = -axis_y

    inverse_k = np.linalg.inv(new_camera_matrix)
    homography = np.vstack(
        [
            distance * (axis_x @ inverse_k),
            distance * (axis_y @ inverse_k),
            normal @ inverse_k,
        ]
    )
    homography /= homography[2, 2]
    return homography, distance, normal


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(points, homography).reshape(-1, 2)


def local_scale_at(
    point_px: tuple[float, float], homography: np.ndarray
) -> tuple[float, float]:
    x, y = point_px
    mapped = transform_points(
        np.asarray([[x, y], [x + 1.0, y], [x, y + 1.0]]), homography
    )
    mm_per_px_x = float(np.linalg.norm(mapped[1] - mapped[0]))
    mm_per_px_y = float(np.linalg.norm(mapped[2] - mapped[0]))
    return mm_per_px_x, mm_per_px_y


def create_contact_sheet(
    capture: cv2.VideoCapture,
    fps: float,
    frame_count: int,
    destination: Path,
) -> None:
    duration = frame_count / fps
    times = np.arange(0.0, duration, CONTACT_SHEET_INTERVAL_S)
    thumbnails: list[np.ndarray] = []
    for time_s in times:
        try:
            _, frame = read_frame_at(capture, float(time_s), fps)
        except RuntimeError:
            continue
        height, width = frame.shape[:2]
        thumb_height = int(round(height * CONTACT_THUMBNAIL_WIDTH / width))
        thumb = cv2.resize(frame, (CONTACT_THUMBNAIL_WIDTH, thumb_height))
        cv2.rectangle(thumb, (0, 0), (155, 33), (0, 0, 0), -1)
        cv2.putText(
            thumb,
            f"{time_s:6.1f} s",
            (10, 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumb)
    if not thumbnails:
        raise RuntimeError("No frames were available for the timestamp sheet.")
    columns = 3
    rows = math.ceil(len(thumbnails) / columns)
    blank = np.zeros_like(thumbnails[0])
    cells = thumbnails + [blank] * (rows * columns - len(thumbnails))
    sheet = np.vstack(
        [np.hstack(cells[row * columns : (row + 1) * columns]) for row in range(rows)]
    )
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"Could not save {destination}")


def select_points(
    image: np.ndarray,
    title: str,
    instructions: list[str],
    required_count: int,
    colour: tuple[int, int, int],
) -> np.ndarray:
    points: list[tuple[int, int]] = []
    window_name = title

    def on_mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < required_count:
            points.append((x, y))

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, min(1500, image.shape[1]), min(850, image.shape[0]))
    cv2.setMouseCallback(window_name, on_mouse)
    while True:
        display = image.copy()
        overlay_height = 38 + 27 * len(instructions)
        cv2.rectangle(display, (0, 0), (display.shape[1], overlay_height), (0, 0, 0), -1)
        for row, line in enumerate(instructions):
            cv2.putText(
                display,
                line,
                (14, 30 + row * 27),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.68,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        for index, point in enumerate(points, start=1):
            cv2.circle(display, point, 8, colour, -1, cv2.LINE_AA)
            cv2.putText(
                display,
                str(index),
                (point[0] + 10, point[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                colour,
                2,
                cv2.LINE_AA,
            )
        if len(points) == 2:
            cv2.line(display, points[0], points[1], colour, 3, cv2.LINE_AA)
        cv2.imshow(window_name, display)
        key = cv2.waitKey(25) & 0xFF
        if key in (8, 127) and points:
            points.pop()
        elif key in (13, 10) and len(points) == required_count:
            break
        elif key == 27:
            cv2.destroyWindow(window_name)
            raise KeyboardInterrupt("Point selection cancelled.")
    cv2.destroyWindow(window_name)
    return np.asarray(points, dtype=np.float32)


def line_from_points(points: np.ndarray) -> np.ndarray:
    (x1, y1), (x2, y2) = np.asarray(points, dtype=float)
    line = np.asarray([y1 - y2, x2 - x1, x1 * y2 - x2 * y1])
    magnitude = float(np.linalg.norm(line[:2]))
    if magnitude < 1e-9:
        raise ValueError("The finish-line points must be different.")
    return line / magnitude


def green_candidates(
    frame: np.ndarray, homography: np.ndarray
) -> tuple[list[dict[str, float]], np.ndarray]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv, np.asarray(GREEN_HSV_LOWER, np.uint8), np.asarray(GREEN_HSV_UPPER, np.uint8)
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    detections: list[dict[str, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        if area < 15.0 or perimeter <= 0:
            continue
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        centre = np.asarray(
            [moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]],
            dtype=np.float32,
        )
        equivalent_diameter_px = math.sqrt(4.0 * area / math.pi)
        mm_x, mm_y = local_scale_at(tuple(centre), homography)
        diameter_mm = equivalent_diameter_px * math.sqrt(mm_x * mm_y)
        circularity = 4.0 * math.pi * area / (perimeter * perimeter)
        diameter_error = abs(diameter_mm - GREEN_DOT_DIAMETER_MM)
        detections.append(
            {
                "x_px": float(centre[0]),
                "y_px": float(centre[1]),
                "diameter_px": equivalent_diameter_px,
                "diameter_mm": diameter_mm,
                "circularity": circularity,
                "score": diameter_error + 10.0 * max(0.0, 0.65 - circularity),
            }
        )
    detections.sort(key=lambda item: item["score"])
    return detections[:3], mask


def write_checker_csv(path: Path, observations: list[CheckerObservation]) -> None:
    fields = [
        "window",
        "frame",
        "time_s",
        "accepted",
        "stability_score_px",
        "sharpness",
        "pixels_per_mm_at_board",
        "plane_distance_mm",
        "pose_reprojection_error_px",
        "centre_x_px",
        "centre_y_px",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for item in observations:
            writer.writerow(
                {
                    "window": item.label,
                    "frame": item.frame_index,
                    "time_s": item.time_s,
                    "accepted": int(item.accepted),
                    "stability_score_px": item.stability_score_px,
                    "sharpness": item.sharpness,
                    "pixels_per_mm_at_board": item.pixels_per_mm,
                    "plane_distance_mm": item.plane_distance_mm,
                    "pose_reprojection_error_px": item.reprojection_error_px,
                    "centre_x_px": float(item.centre_px[0]),
                    "centre_y_px": float(item.centre_px[1]),
                }
            )


def save_overview(
    path: Path,
    frame: np.ndarray,
    pivot_points: np.ndarray,
    finish_points: np.ndarray,
    validation_text: str,
) -> None:
    image = frame.copy()
    for index, point in enumerate(pivot_points.astype(int), start=1):
        cv2.circle(image, tuple(point), 9, (255, 80, 0), -1, cv2.LINE_AA)
        cv2.putText(
            image,
            f"P{index}",
            tuple(point + np.asarray([12, -10])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 80, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.line(
        image,
        tuple(finish_points[0].astype(int)),
        tuple(finish_points[1].astype(int)),
        (0, 0, 255),
        4,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "FINISH: left edge",
        tuple(finish_points[0].astype(int) + np.asarray([12, -12])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.rectangle(image, (0, 0), (image.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(
        image,
        validation_text,
        (15, 31),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save {path}")


def main() -> None:
    args = parse_args()
    selected_video = args.video if args.video is not None else VIDEO_PATH
    if selected_video is None:
        raise SystemExit(
            "Set VIDEO_PATH near the top of this script, or drag/pass the video "
            "path after the script name."
        )
    video_path = Path(selected_video).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Calibration video does not exist: {video_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    capture, fps, frame_count, width, height = open_video(video_path)
    duration_s = frame_count / fps
    contact_path = OUTPUT_DIR / "timestamp_contact_sheet.png"
    create_contact_sheet(capture, fps, frame_count, contact_path)
    print(f"Timestamp contact sheet: {contact_path}")
    if args.contact_sheet_only:
        capture.release()
        return
    if not CHECKERBOARD_WINDOWS or REFERENCE_FRAME_TIME_S is None:
        capture.release()
        raise SystemExit(
            "Contact sheet created. Now fill CHECKERBOARD_WINDOWS and "
            "REFERENCE_FRAME_TIME_S near the top of this script, then run again."
        )
    if REFERENCE_FRAME_TIME_S < 0 or REFERENCE_FRAME_TIME_S > duration_s:
        raise ValueError("REFERENCE_FRAME_TIME_S lies outside the video.")

    camera_matrix, distortion, new_camera_matrix = load_lens_calibration(
        CAMERA_CALIBRATION_NPZ, (width, height)
    )
    observations = collect_checker_observations(
        capture,
        fps,
        CHECKERBOARD_WINDOWS,
        camera_matrix,
        distortion,
        new_camera_matrix,
    )
    homography, plane_distance_mm, plane_normal = average_plane_pose(
        observations, new_camera_matrix
    )

    reference_index, reference_raw = read_frame_at(
        capture, REFERENCE_FRAME_TIME_S, fps
    )
    reference = undistort(
        reference_raw, camera_matrix, distortion, new_camera_matrix
    )
    pivot_points = select_points(
        reference,
        "Select two cilia lower-pivot centres",
        [
            f"Click cilium {PIVOT_A_CILIUM}, then cilium {PIVOT_B_CILIUM} lower pivot.",
            "Enter = accept | Backspace = undo | Esc = cancel",
        ],
        2,
        (255, 80, 0),
    )
    pivot_mm_before = transform_points(pivot_points, homography)
    measured_pivot_span_before = float(
        np.linalg.norm(pivot_mm_before[1] - pivot_mm_before[0])
    )
    expected_pivot_span = (
        abs(PIVOT_B_CILIUM - PIVOT_A_CILIUM) * CILIA_SPACING_MM
    )
    scale_correction = expected_pivot_span / measured_pivot_span_before
    if APPLY_PIVOT_SCALE_CORRECTION:
        homography[0, :] *= scale_correction
        homography[1, :] *= scale_correction
    pivot_mm = transform_points(pivot_points, homography)
    measured_pivot_span = float(np.linalg.norm(pivot_mm[1] - pivot_mm[0]))

    finish_points = select_points(
        reference,
        "Select the finish marker's left edge",
        [
            "Click two well-separated points on the LEFT EDGE of the vertical wood.",
            "Enter = accept | Backspace = undo | Esc = cancel",
        ],
        2,
        (0, 0, 255),
    )
    finish_line = line_from_points(finish_points)
    finish_mm = transform_points(finish_points, homography)

    centre_scale_x, centre_scale_y = local_scale_at(
        (width / 2.0, height / 2.0), homography
    )
    checker_scales = np.asarray(
        [item.pixels_per_mm for item in observations if item.accepted]
    )

    marker_rows: list[dict[str, float | int | str]] = []
    marker_windows = GREEN_MARKER_WINDOWS or [
        ("reference_frame", REFERENCE_FRAME_TIME_S, REFERENCE_FRAME_TIME_S)
    ]
    for label, start_s, end_s in marker_windows:
        times = (
            [start_s]
            if end_s <= start_s
            else np.arange(start_s, end_s + 1e-9, GREEN_SAMPLE_INTERVAL_S)
        )
        best_frame: np.ndarray | None = None
        best_detections: list[dict[str, float]] = []
        best_time = float(start_s)
        for time_s in times:
            try:
                frame_index, raw = read_frame_at(capture, float(time_s), fps)
            except RuntimeError:
                continue
            frame = undistort(raw, camera_matrix, distortion, new_camera_matrix)
            detections, _ = green_candidates(frame, homography)
            if len(detections) > len(best_detections) or (
                len(detections) == len(best_detections)
                and sum(item["score"] for item in detections)
                < sum(item["score"] for item in best_detections)
            ):
                best_frame = frame
                best_detections = detections
                best_time = float(time_s)
                best_index = frame_index
        if best_frame is None:
            continue
        annotated = best_frame.copy()
        for number, detection in enumerate(
            sorted(best_detections, key=lambda item: item["x_px"]), start=1
        ):
            centre = (int(round(detection["x_px"])), int(round(detection["y_px"])))
            cv2.circle(annotated, centre, 13, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(
                annotated,
                f"{number}: {detection['diameter_mm']:.1f} mm",
                (centre[0] + 15, centre[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            marker_rows.append(
                {
                    "window": label,
                    "frame": best_index,
                    "time_s": best_time,
                    "marker": number,
                    **detection,
                }
            )
        cv2.imwrite(str(OUTPUT_DIR / f"green_check_{label}.png"), annotated)

    write_checker_csv(OUTPUT_DIR / "checkerboard_frames.csv", observations)
    if marker_rows:
        with (OUTPUT_DIR / "green_marker_check.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(marker_rows[0].keys()))
            writer.writeheader()
            writer.writerows(marker_rows)

    pivot_error_before_pct = 100.0 * (
        measured_pivot_span_before - expected_pivot_span
    ) / expected_pivot_span
    validation_text = (
        f"Pivot span: {measured_pivot_span_before:.1f} mm before correction "
        f"({pivot_error_before_pct:+.2f}%), {measured_pivot_span:.1f} mm saved"
    )
    save_overview(
        OUTPUT_DIR / "calibration_overview.png",
        reference,
        pivot_points,
        finish_points,
        validation_text,
    )

    np.savez_compressed(
        OUTPUT_DIR / "doe_session_calibration.npz",
        camera_matrix=camera_matrix,
        dist_coeffs=distortion,
        new_camera_matrix=new_camera_matrix,
        image_size=np.asarray([width, height], np.int32),
        image_to_plane_mm=homography,
        plane_normal_camera=plane_normal,
        plane_distance_mm=np.asarray(plane_distance_mm),
        centre_mm_per_pixel_xy=np.asarray([centre_scale_x, centre_scale_y]),
        pivot_points_px=pivot_points,
        pivot_points_mm=pivot_mm,
        expected_pivot_span_mm=np.asarray(expected_pivot_span),
        measured_pivot_span_before_correction_mm=np.asarray(
            measured_pivot_span_before
        ),
        applied_scale_correction=np.asarray(
            scale_correction if APPLY_PIVOT_SCALE_CORRECTION else 1.0
        ),
        finish_line_points_px=finish_points,
        finish_line_abc_px=finish_line,
        finish_line_points_mm=finish_mm,
        green_hsv_lower=np.asarray(GREEN_HSV_LOWER, np.uint8),
        green_hsv_upper=np.asarray(GREEN_HSV_UPPER, np.uint8),
        expected_green_dot_diameter_mm=np.asarray(GREEN_DOT_DIAMETER_MM),
        travel_direction=np.asarray(TRAVEL_DIRECTION),
        reference_frame_index=np.asarray(reference_index),
        video_fps=np.asarray(fps),
    )

    summary = {
        "source_video": str(video_path),
        "video": {
            "fps": fps,
            "duration_s": duration_s,
            "frame_count": frame_count,
            "image_size": [width, height],
        },
        "checkerboard": {
            "internal_corners": list(CHECKERBOARD_CORNERS),
            "square_mm": CHECKERBOARD_SQUARE_MM,
            "windows": [list(item) for item in CHECKERBOARD_WINDOWS],
            "detections": len(observations),
            "accepted_detections": sum(item.accepted for item in observations),
            "median_detected_pixels_per_mm": float(np.median(checker_scales)),
            "plane_distance_mm": plane_distance_mm,
            "plane_normal_camera": plane_normal.tolist(),
        },
        "pivot_validation": {
            "cilia": [PIVOT_A_CILIUM, PIVOT_B_CILIUM],
            "spacing_mm": CILIA_SPACING_MM,
            "expected_span_mm": expected_pivot_span,
            "measured_span_before_correction_mm": measured_pivot_span_before,
            "error_before_correction_percent": pivot_error_before_pct,
            "scale_correction_applied": (
                scale_correction if APPLY_PIVOT_SCALE_CORRECTION else 1.0
            ),
            "saved_span_mm": measured_pivot_span,
        },
        "finish": {
            "definition": "centre of leading green dot crosses the clicked left edge",
            "travel_direction": TRAVEL_DIRECTION,
            "line_points_px": finish_points.tolist(),
            "line_abc_px": finish_line.tolist(),
        },
        "green_markers": {
            "expected_diameter_mm": GREEN_DOT_DIAMETER_MM,
            "hsv_lower": list(GREEN_HSV_LOWER),
            "hsv_upper": list(GREEN_HSV_UPPER),
            "validation_rows": len(marker_rows),
        },
        "reference_frame_time_s": REFERENCE_FRAME_TIME_S,
        "reference_frame_index": reference_index,
        "outputs": {
            "machine_calibration": "doe_session_calibration.npz",
            "visual_check": "calibration_overview.png",
            "checkerboard_audit": "checkerboard_frames.csv",
            "green_marker_audit": "green_marker_check.csv",
        },
    }
    with (OUTPUT_DIR / "doe_session_calibration.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    capture.release()
    cv2.destroyAllWindows()
    print("\nCalibration complete.")
    print(f"Machine-readable file: {OUTPUT_DIR / 'doe_session_calibration.npz'}")
    print(f"Visual check:          {OUTPUT_DIR / 'calibration_overview.png'}")
    print(validation_text)


if __name__ == "__main__":
    main()
