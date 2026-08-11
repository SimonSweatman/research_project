"""Create one reusable calibration for the full DOE recording session.

Set VIDEO_PATH, run once to create a timestamp contact sheet, then fill the
stationary timestamp windows and run again. Moving frames between windows are
ignored; remaining unstable checkerboard frames are rejected automatically.

The final run asks for two cilia pivot clicks and two clicks along the left
edge of the finish marker. Enter accepts, Backspace undoes, and Esc cancels.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


# =============================================================================
# EDIT THESE AFTER ADDING THE WIDE-VIEW CALIBRATION VIDEO
# =============================================================================

SCRIPT_DIR = Path(__file__).resolve().parent

# Example: Path(r"C:\full\path\to\calibration_video.mp4")
VIDEO_PATH = Path(r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260807_162630.mp4")

# Only include periods when the complete checkerboard is held still.
CHECKERBOARD_WINDOWS: list[tuple[str, float, float]] = [
    ("left", 3.0, 5.0),
    ("centre", 8.0, 10.0),
    ("right", 14.0, 17.0),
]

# Clear frame after your hand/checkerboard has left the scene.
REFERENCE_FRAME_TIME_S = 55.0

# Optional periods with all three dots visible. If empty, the reference frame
# is checked instead.
GREEN_MARKER_WINDOWS: list[tuple[str, float, float]] = [
    ("markers_at_start", 55.0, 57.0),
    ("markers_at_middle", 63.0, 65.0),
    ("markers_at_start", 74.0, 77.0),
]


# =============================================================================
# PHYSICAL AND DETECTION SETTINGS
# =============================================================================

CAMERA_CALIBRATION_NPZ = (
    SCRIPT_DIR / "checkerboard_calibration_outputs" / "camera_calibration.npz"
)
OUTPUT_DIR = SCRIPT_DIR / "doe_session_calibration_outputs"
CHECKERBOARD_CORNERS = (9, 7)  # internal corners
CHECKERBOARD_SQUARE_MM = 10.0

# Click these two lower pivot centres, in this order.
PIVOT_A_CILIUM = 1
PIVOT_B_CILIUM = 12
CILIA_SPACING_MM = 87.0
APPLY_PIVOT_SCALE_CORRECTION = True

# Change this to the dot diameter actually used on the 50 cm test piece.
GREEN_DOT_DIAMETER_MM = 30.0
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
        "video", nargs="?", type=Path, help="Overrides VIDEO_PATH above."
    )
    parser.add_argument(
        "--contact-sheet-only",
        action="store_true",
        help="Create timestamp sheet and stop.",
    )
    return parser.parse_args()


def open_video(path: Path) -> tuple[cv2.VideoCapture, float, int, int, int]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or count <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Video metadata could not be read reliably.")
    return capture, fps, count, width, height


def read_frame_at(
    capture: cv2.VideoCapture, time_s: float, fps: float
) -> tuple[int, np.ndarray]:
    index = max(0, int(round(time_s * fps)))
    capture.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = capture.read()
    if not ok:
        raise RuntimeError(f"Could not read frame at {time_s:.3f} s.")
    return index, frame


def load_lens_calibration(
    path: Path, image_size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Lens calibration not found: {path}")
    with np.load(path) as data:
        camera = np.asarray(data["camera_matrix"], dtype=np.float64)
        distortion = np.asarray(data["dist_coeffs"], dtype=np.float64)
    new_camera, _ = cv2.getOptimalNewCameraMatrix(
        camera, distortion, image_size, 1.0, image_size
    )
    return camera, distortion, new_camera


def undistort(
    frame: np.ndarray,
    camera: np.ndarray,
    distortion: np.ndarray,
    new_camera: np.ndarray,
) -> np.ndarray:
    return cv2.undistort(frame, camera, distortion, None, new_camera)


def checker_object_points() -> np.ndarray:
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
        flags=(cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE),
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


def canonicalize(corners: np.ndarray) -> np.ndarray:
    columns, rows = CHECKERBOARD_CORNERS
    grid = corners.reshape(rows, columns, 2).copy()
    if np.mean(grid[:, -1, 0] - grid[:, 0, 0]) < 0:
        grid = grid[:, ::-1]
    if np.mean(grid[-1, :, 1] - grid[0, :, 1]) < 0:
        grid = grid[::-1]
    return np.ascontiguousarray(grid.reshape(-1, 2), dtype=np.float32)


def pixels_per_mm(corners: np.ndarray) -> float:
    columns, rows = CHECKERBOARD_CORNERS
    grid = corners.reshape(rows, columns, 2)
    lengths: list[float] = []
    for row in grid:
        lengths.extend(np.linalg.norm(np.diff(row, axis=0), axis=1).tolist())
    for column in range(columns):
        lengths.extend(
            np.linalg.norm(np.diff(grid[:, column], axis=0), axis=1).tolist()
        )
    return float(np.median(lengths) / CHECKERBOARD_SQUARE_MM)


def robust_upper(values: np.ndarray, multiple: float = 3.5) -> float:
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    return median + multiple * max(1.4826 * mad, 1e-9)


def collect_checkerboards(
    capture: cv2.VideoCapture,
    fps: float,
    camera: np.ndarray,
    distortion: np.ndarray,
    new_camera: np.ndarray,
) -> list[CheckerObservation]:
    object_points = checker_object_points()
    observations: list[CheckerObservation] = []
    step = max(1, int(round(CHECKERBOARD_SAMPLE_INTERVAL_S * fps)))
    for label, start_s, end_s in CHECKERBOARD_WINDOWS:
        if end_s <= start_s:
            raise ValueError(f"Window {label!r} has end <= start.")
        for index in range(
            max(0, int(start_s * fps)), int(math.ceil(end_s * fps)) + 1, step
        ):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, raw = capture.read()
            if not ok:
                continue
            frame = undistort(raw, camera, distortion, new_camera)
            corners = detect_checkerboard(frame)
            if corners is None:
                continue
            corners = canonicalize(corners)
            ok_pose, rvec, tvec = cv2.solvePnP(
                object_points,
                corners.reshape(-1, 1, 2),
                new_camera,
                None,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok_pose:
                continue
            rotation, _ = cv2.Rodrigues(rvec)
            normal = rotation[:, 2].astype(float)
            distance = float(normal @ tvec.reshape(3))
            if distance < 0:
                normal, distance = -normal, -distance
            projected, _ = cv2.projectPoints(
                object_points, rvec, tvec, new_camera, None
            )
            error = float(
                np.sqrt(
                    np.mean(
                        np.sum((projected.reshape(-1, 2) - corners) ** 2, axis=1)
                    )
                )
            )
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            observations.append(
                CheckerObservation(
                    label,
                    index,
                    index / fps,
                    corners,
                    float(cv2.Laplacian(gray, cv2.CV_64F).var()),
                    np.mean(corners, axis=0),
                    pixels_per_mm(corners),
                    normal,
                    distance,
                    error,
                )
            )
    if not observations:
        raise RuntimeError(
            "No checkerboard found in the selected windows. Check timestamps and "
            "ensure all 9 x 7 internal corners are visible."
        )

    for label in {item.label for item in observations}:
        group = [item for item in observations if item.label == label]
        median_corners = np.median(np.stack([x.corners for x in group]), axis=0)
        scores = np.asarray(
            [np.median(np.linalg.norm(x.corners - median_corners, axis=1)) for x in group]
        )
        stability_limit = robust_upper(scores)
        sharpness = np.asarray([x.sharpness for x in group])
        sharpness_floor = max(15.0, 0.45 * float(np.median(sharpness)))
        errors = np.asarray([x.reprojection_error_px for x in group])
        error_limit = max(0.75, robust_upper(errors))
        for item, score in zip(group, scores):
            item.stability_score_px = float(score)
            item.accepted = bool(
                score <= stability_limit
                and item.sharpness >= sharpness_floor
                and item.reprojection_error_px <= error_limit
            )
    if sum(x.accepted for x in observations) < 3:
        raise RuntimeError("Fewer than three stable checkerboard frames survived.")
    return observations


def plane_homography(
    observations: list[CheckerObservation], new_camera: np.ndarray
) -> tuple[np.ndarray, float, np.ndarray]:
    accepted = [x for x in observations if x.accepted]
    normals = np.stack([x.normal_camera for x in accepted])
    distances = np.asarray([x.plane_distance_mm for x in accepted])
    preliminary = np.mean(normals, axis=0)
    preliminary /= np.linalg.norm(preliminary)
    angle_error = np.degrees(
        np.arccos(np.clip(normals @ preliminary, -1.0, 1.0))
    )
    distance_error = np.abs(distances - np.median(distances))
    keep = (angle_error <= max(0.35, robust_upper(angle_error))) & (
        distance_error <= max(2.0, robust_upper(distance_error))
    )
    if int(np.sum(keep)) < 3:
        keep[:] = True
    normal = np.mean(normals[keep], axis=0)
    normal /= np.linalg.norm(normal)
    distance = float(np.median(distances[keep]))

    # Global X is image-right; global Y is image-down. Ray-plane intersection
    # gives one perspective-correct mapping valid over the entire test field.
    axis_x = np.array([1.0, 0.0, 0.0])
    axis_x -= normal * float(axis_x @ normal)
    axis_x /= np.linalg.norm(axis_x)
    axis_y = np.cross(normal, axis_x)
    if axis_y @ np.array([0.0, 1.0, 0.0]) < 0:
        axis_y = -axis_y
    inverse_k = np.linalg.inv(new_camera)
    homography = np.vstack(
        [distance * (axis_x @ inverse_k), distance * (axis_y @ inverse_k), normal @ inverse_k]
    )
    homography /= homography[2, 2]
    return homography, distance, normal


def transform(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        np.asarray(points, np.float32).reshape(-1, 1, 2), homography
    ).reshape(-1, 2)


def local_mm_per_px(point: tuple[float, float], homography: np.ndarray) -> tuple[float, float]:
    x, y = point
    mapped = transform(np.asarray([[x, y], [x + 1, y], [x, y + 1]]), homography)
    return float(np.linalg.norm(mapped[1] - mapped[0])), float(
        np.linalg.norm(mapped[2] - mapped[0])
    )


def contact_sheet(
    capture: cv2.VideoCapture, fps: float, count: int, destination: Path
) -> None:
    thumbnails: list[np.ndarray] = []
    for time_s in np.arange(0.0, count / fps, CONTACT_SHEET_INTERVAL_S):
        try:
            _, frame = read_frame_at(capture, float(time_s), fps)
        except RuntimeError:
            continue
        height, width = frame.shape[:2]
        thumb = cv2.resize(
            frame,
            (CONTACT_THUMBNAIL_WIDTH, int(height * CONTACT_THUMBNAIL_WIDTH / width)),
        )
        cv2.rectangle(thumb, (0, 0), (155, 34), (0, 0, 0), -1)
        cv2.putText(
            thumb, f"{time_s:6.1f} s", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
            0.68, (255, 255, 255), 2, cv2.LINE_AA
        )
        thumbnails.append(thumb)
    if not thumbnails:
        raise RuntimeError("No frames available for contact sheet.")
    columns = 3
    rows = math.ceil(len(thumbnails) / columns)
    thumbnails += [np.zeros_like(thumbnails[0])] * (rows * columns - len(thumbnails))
    sheet = np.vstack(
        [np.hstack(thumbnails[r * columns : (r + 1) * columns]) for r in range(rows)]
    )
    if not cv2.imwrite(str(destination), sheet):
        raise RuntimeError(f"Could not save {destination}")


def select_two(
    image: np.ndarray, title: str, instructions: list[str], colour: tuple[int, int, int]
) -> np.ndarray:
    points: list[tuple[int, int]] = []

    def mouse(event: int, x: int, y: int, _flags: int, _data: object) -> None:
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(title, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(title, min(1500, image.shape[1]), min(850, image.shape[0]))
    cv2.setMouseCallback(title, mouse)
    while True:
        display = image.copy()
        cv2.rectangle(display, (0, 0), (display.shape[1], 92), (0, 0, 0), -1)
        for row, line in enumerate(instructions):
            cv2.putText(
                display, line, (14, 29 + row * 28), cv2.FONT_HERSHEY_SIMPLEX,
                0.67, (255, 255, 255), 2, cv2.LINE_AA
            )
        for number, point in enumerate(points, 1):
            cv2.circle(display, point, 8, colour, -1, cv2.LINE_AA)
            cv2.putText(display, str(number), (point[0] + 10, point[1] - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colour, 2, cv2.LINE_AA)
        if len(points) == 2:
            cv2.line(display, points[0], points[1], colour, 3, cv2.LINE_AA)
        cv2.imshow(title, display)
        key = cv2.waitKey(25) & 0xFF
        if key in (8, 127) and points:
            points.pop()
        elif key in (10, 13) and len(points) == 2:
            break
        elif key == 27:
            cv2.destroyWindow(title)
            raise KeyboardInterrupt("Selection cancelled.")
    cv2.destroyWindow(title)
    return np.asarray(points, np.float32)


def line_equation(points: np.ndarray) -> np.ndarray:
    (x1, y1), (x2, y2) = np.asarray(points, float)
    line = np.asarray([y1 - y2, x2 - x1, x1 * y2 - x2 * y1])
    length = float(np.linalg.norm(line[:2]))
    if length < 1e-9:
        raise ValueError("Finish-line points must differ.")
    return line / length


def green_candidates(frame: np.ndarray, homography: np.ndarray) -> list[dict[str, float]]:
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.asarray(GREEN_HSV_LOWER, np.uint8),
                       np.asarray(GREEN_HSV_UPPER, np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    found: list[dict[str, float]] = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        perimeter = float(cv2.arcLength(contour, True))
        moments = cv2.moments(contour)
        if area < 15 or perimeter <= 0 or moments["m00"] == 0:
            continue
        centre = np.asarray([moments["m10"] / moments["m00"],
                             moments["m01"] / moments["m00"]], np.float32)
        diameter_px = math.sqrt(4 * area / math.pi)
        sx, sy = local_mm_per_px(tuple(centre), homography)
        diameter_mm = diameter_px * math.sqrt(sx * sy)
        circularity = 4 * math.pi * area / perimeter**2
        found.append({
            "x_px": float(centre[0]), "y_px": float(centre[1]),
            "diameter_px": diameter_px, "diameter_mm": diameter_mm,
            "circularity": circularity,
            "score": abs(diameter_mm - GREEN_DOT_DIAMETER_MM)
                     + 10 * max(0.0, 0.65 - circularity),
        })
    return sorted(found, key=lambda x: x["score"])[:3]


def save_checker_csv(path: Path, observations: list[CheckerObservation]) -> None:
    fields = ["window", "frame", "time_s", "accepted", "stability_score_px",
              "sharpness", "pixels_per_mm_at_board", "plane_distance_mm",
              "pose_reprojection_error_px", "centre_x_px", "centre_y_px"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for x in observations:
            writer.writerow({
                "window": x.label, "frame": x.frame_index, "time_s": x.time_s,
                "accepted": int(x.accepted), "stability_score_px": x.stability_score_px,
                "sharpness": x.sharpness, "pixels_per_mm_at_board": x.pixels_per_mm,
                "plane_distance_mm": x.plane_distance_mm,
                "pose_reprojection_error_px": x.reprojection_error_px,
                "centre_x_px": float(x.centre_px[0]), "centre_y_px": float(x.centre_px[1]),
            })


def save_overview(
    path: Path, frame: np.ndarray, pivots: np.ndarray, finish: np.ndarray, text: str
) -> None:
    image = frame.copy()
    for number, point in enumerate(pivots.astype(int), 1):
        cv2.circle(image, tuple(point), 9, (255, 80, 0), -1, cv2.LINE_AA)
        cv2.putText(image, f"P{number}", tuple(point + [12, -10]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 80, 0), 2, cv2.LINE_AA)
    cv2.line(image, tuple(finish[0].astype(int)), tuple(finish[1].astype(int)),
             (0, 0, 255), 4, cv2.LINE_AA)
    cv2.putText(image, "FINISH: left edge", tuple(finish[0].astype(int) + [12, -12]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.rectangle(image, (0, 0), (image.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(image, text, (15, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.72,
                (255, 255, 255), 2, cv2.LINE_AA)
    if not cv2.imwrite(str(path), image):
        raise RuntimeError(f"Could not save {path}")


def main() -> None:
    args = parse_args()
    selected = args.video if args.video is not None else VIDEO_PATH
    if selected is None:
        raise SystemExit(
            "Set VIDEO_PATH near the top, or pass the video path after the script name."
        )
    video_path = Path(selected).expanduser().resolve()
    if not video_path.exists():
        raise FileNotFoundError(f"Video does not exist: {video_path}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    capture, fps, count, width, height = open_video(video_path)
    sheet_path = OUTPUT_DIR / "timestamp_contact_sheet.png"
    contact_sheet(capture, fps, count, sheet_path)
    print(f"Timestamp contact sheet: {sheet_path}")
    if args.contact_sheet_only:
        capture.release()
        return
    if not CHECKERBOARD_WINDOWS or REFERENCE_FRAME_TIME_S is None:
        capture.release()
        raise SystemExit(
            "Contact sheet created. Fill CHECKERBOARD_WINDOWS and "
            "REFERENCE_FRAME_TIME_S near the top, then run again."
        )

    camera, distortion, new_camera = load_lens_calibration(
        CAMERA_CALIBRATION_NPZ, (width, height)
    )
    observations = collect_checkerboards(
        capture, fps, camera, distortion, new_camera
    )
    homography, plane_distance, plane_normal = plane_homography(
        observations, new_camera
    )
    reference_index, raw = read_frame_at(capture, REFERENCE_FRAME_TIME_S, fps)
    reference = undistort(raw, camera, distortion, new_camera)

    pivots = select_two(
        reference,
        "Select two lower pivots",
        [f"Click cilium {PIVOT_A_CILIUM}, then cilium {PIVOT_B_CILIUM} lower pivot.",
         "Enter = accept | Backspace = undo | Esc = cancel"],
        (255, 80, 0),
    )
    before_mm = transform(pivots, homography)
    measured_before = float(np.linalg.norm(before_mm[1] - before_mm[0]))
    expected_span = abs(PIVOT_B_CILIUM - PIVOT_A_CILIUM) * CILIA_SPACING_MM
    correction = expected_span / measured_before
    if APPLY_PIVOT_SCALE_CORRECTION:
        homography[:2] *= correction
    pivots_mm = transform(pivots, homography)
    measured_saved = float(np.linalg.norm(pivots_mm[1] - pivots_mm[0]))

    finish = select_two(
        reference,
        "Select finish marker left edge",
        ["Click two separated points on the LEFT EDGE of the vertical wood.",
         "Enter = accept | Backspace = undo | Esc = cancel"],
        (0, 0, 255),
    )
    finish_line = line_equation(finish)
    finish_mm = transform(finish, homography)
    centre_scale = local_mm_per_px((width / 2, height / 2), homography)

    marker_rows: list[dict[str, float | int | str]] = []
    marker_windows = GREEN_MARKER_WINDOWS or [
        ("reference_frame", REFERENCE_FRAME_TIME_S, REFERENCE_FRAME_TIME_S)
    ]
    for label, start_s, end_s in marker_windows:
        times = [start_s] if end_s <= start_s else np.arange(
            start_s, end_s + 1e-9, GREEN_SAMPLE_INTERVAL_S
        )
        best: tuple[float, int, np.ndarray, list[dict[str, float]]] | None = None
        for time_s in times:
            try:
                index, raw = read_frame_at(capture, float(time_s), fps)
            except RuntimeError:
                continue
            frame = undistort(raw, camera, distortion, new_camera)
            detections = green_candidates(frame, homography)
            quality = len(detections) * 1000 - sum(x["score"] for x in detections)
            if best is None or quality > best[0]:
                best = (quality, index, frame, detections)
        if best is None:
            continue
        _, index, annotated, detections = best
        for number, item in enumerate(sorted(detections, key=lambda x: x["x_px"]), 1):
            centre = (round(item["x_px"]), round(item["y_px"]))
            cv2.circle(annotated, centre, 13, (0, 255, 255), 3, cv2.LINE_AA)
            cv2.putText(annotated, f"{number}: {item['diameter_mm']:.1f} mm",
                        (centre[0] + 15, centre[1] - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 255), 2, cv2.LINE_AA)
            marker_rows.append({"window": label, "frame": index,
                                "time_s": index / fps, "marker": number, **item})
        cv2.imwrite(str(OUTPUT_DIR / f"green_check_{label}.png"), annotated)

    save_checker_csv(OUTPUT_DIR / "checkerboard_frames.csv", observations)
    if marker_rows:
        with (OUTPUT_DIR / "green_marker_check.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=list(marker_rows[0]))
            writer.writeheader()
            writer.writerows(marker_rows)

    error_pct = 100 * (measured_before - expected_span) / expected_span
    validation = (
        f"Pivot span: {measured_before:.1f} mm before correction "
        f"({error_pct:+.2f}%), {measured_saved:.1f} mm saved"
    )
    save_overview(
        OUTPUT_DIR / "calibration_overview.png", reference, pivots, finish, validation
    )

    applied_correction = correction if APPLY_PIVOT_SCALE_CORRECTION else 1.0
    np.savez_compressed(
        OUTPUT_DIR / "doe_session_calibration.npz",
        camera_matrix=camera,
        dist_coeffs=distortion,
        new_camera_matrix=new_camera,
        image_size=np.asarray([width, height], np.int32),
        image_to_plane_mm=homography,
        plane_normal_camera=plane_normal,
        plane_distance_mm=np.asarray(plane_distance),
        centre_mm_per_pixel_xy=np.asarray(centre_scale),
        pivot_points_px=pivots,
        pivot_points_mm=pivots_mm,
        expected_pivot_span_mm=np.asarray(expected_span),
        measured_pivot_span_before_correction_mm=np.asarray(measured_before),
        applied_scale_correction=np.asarray(applied_correction),
        finish_line_points_px=finish,
        finish_line_abc_px=finish_line,
        finish_line_points_mm=finish_mm,
        green_hsv_lower=np.asarray(GREEN_HSV_LOWER, np.uint8),
        green_hsv_upper=np.asarray(GREEN_HSV_UPPER, np.uint8),
        expected_green_dot_diameter_mm=np.asarray(GREEN_DOT_DIAMETER_MM),
        travel_direction=np.asarray(TRAVEL_DIRECTION),
        reference_frame_index=np.asarray(reference_index),
        video_fps=np.asarray(fps),
    )

    accepted = [x for x in observations if x.accepted]
    summary = {
        "source_video": str(video_path),
        "video": {"fps": fps, "duration_s": count / fps,
                  "frame_count": count, "image_size": [width, height]},
        "checkerboard": {
            "internal_corners": list(CHECKERBOARD_CORNERS),
            "square_mm": CHECKERBOARD_SQUARE_MM,
            "windows": [list(x) for x in CHECKERBOARD_WINDOWS],
            "detections": len(observations), "accepted_detections": len(accepted),
            "median_detected_pixels_per_mm": float(
                np.median([x.pixels_per_mm for x in accepted])
            ),
            "plane_distance_mm": plane_distance,
            "plane_normal_camera": plane_normal.tolist(),
        },
        "pivot_validation": {
            "cilia": [PIVOT_A_CILIUM, PIVOT_B_CILIUM],
            "spacing_mm": CILIA_SPACING_MM, "expected_span_mm": expected_span,
            "measured_span_before_correction_mm": measured_before,
            "error_before_correction_percent": error_pct,
            "scale_correction_applied": applied_correction,
            "saved_span_mm": measured_saved,
        },
        "finish": {
            "definition": "centre of leading green dot crosses clicked left edge",
            "travel_direction": TRAVEL_DIRECTION,
            "line_points_px": finish.tolist(), "line_abc_px": finish_line.tolist(),
        },
        "green_markers": {
            "expected_diameter_mm": GREEN_DOT_DIAMETER_MM,
            "hsv_lower": list(GREEN_HSV_LOWER), "hsv_upper": list(GREEN_HSV_UPPER),
            "validation_rows": len(marker_rows),
        },
        "reference_frame_time_s": REFERENCE_FRAME_TIME_S,
        "reference_frame_index": reference_index,
    }
    with (OUTPUT_DIR / "doe_session_calibration.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)

    capture.release()
    cv2.destroyAllWindows()
    print("\nCalibration complete.")
    print(f"Machine file: {OUTPUT_DIR / 'doe_session_calibration.npz'}")
    print(f"Visual check: {OUTPUT_DIR / 'calibration_overview.png'}")
    print(validation)


if __name__ == "__main__":
    main()
