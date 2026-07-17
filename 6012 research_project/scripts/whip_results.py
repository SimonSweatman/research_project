#!/usr/bin/env python3
"""
Overlay the intended cilium tip path with the measured green-tip path.

Video timing:
    0-2 s   checkerboard scale calibration
    2-5 s   ignored while the checkerboard is removed
    5 s+    green tip tracking

Outputs:
    desired_vs_measured_tip_path.png
    tracked_tip_positions.csv
"""

from pathlib import Path
import csv
import math
import sys

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ---------------- USER SETTINGS ----------------

VIDEO_PATH = Path(
    r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260717_194941.mp4"
)

# Set to None if you do not want to use checkerboard lens undistortion.
CAMERA_CALIBRATION_PATH = Path(r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\research_project\6012 research_project\scripts\checkerboard_calibration_outputs\camera_calibration.npz")

OUTPUT_DIRECTORY = Path(r"C:\temp\whip_results")

CHECKERBOARD_END_S = 2.0
TRACKING_START_S = 5.0

CHECKERBOARD_INTERNAL_CORNERS = (9, 7)
CHECKERBOARD_SQUARE_SIZE_MM = 10.0

GREEN_HSV_LOWER = np.array([35, 70, 60], dtype=np.uint8)
GREEN_HSV_UPPER = np.array([85, 255, 255], dtype=np.uint8)

MIN_MARKER_AREA_PX = 20.0
MAX_MARKER_AREA_PX = 5000.0
MAX_FRAME_JUMP_MM = 8.0

FLIP_X = False
FLIP_Z = True

# Link lengths and gait settings
L1 = 45.0
L2 = 25.0
N = 720

LOWER_BACK = 70.0
LOWER_FORWARD = 110.0

UPPER_UPRIGHT = 92.0
UPPER_STRIKE = 100.0
UPPER_FOLDED = 160.0


# ---------------- CAMERA CALIBRATION ----------------

def load_camera_calibration(path):
    if path is None or not path.exists():
        print("Camera calibration not found; continuing without undistortion.")
        return None, None

    data = np.load(path)

    camera_matrix = None
    distortion = None

    for key in ("camera_matrix", "mtx", "K"):
        if key in data:
            camera_matrix = data[key]
            break

    for key in ("distortion_coefficients", "dist_coeffs", "dist", "D"):
        if key in data:
            distortion = data[key]
            break

    if camera_matrix is None or distortion is None:
        print("Calibration file opened, but expected arrays were not found.")
        print("Available keys:", list(data.keys()))
        return None, None

    print("Loaded camera calibration:", path)
    return camera_matrix, distortion


def undistort(frame, camera_matrix, distortion):
    if camera_matrix is None or distortion is None:
        return frame
    return cv2.undistort(frame, camera_matrix, distortion)


# ---------------- PIXEL-TO-MM SCALE ----------------

def checkerboard_spacing_px(corners, pattern_size):
    columns, rows = pattern_size
    points = corners.reshape(rows, columns, 2)

    horizontal = np.linalg.norm(
        points[:, 1:, :] - points[:, :-1, :],
        axis=2,
    )
    vertical = np.linalg.norm(
        points[1:, :, :] - points[:-1, :, :],
        axis=2,
    )

    return float(
        np.mean(np.concatenate([horizontal.ravel(), vertical.ravel()]))
    )


def calculate_mm_per_pixel(capture, fps, camera_matrix, distortion):
    end_frame = int(round(CHECKERBOARD_END_S * fps))
    spacings = []

    capture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for _ in range(end_frame):
        ok, frame = capture.read()
        if not ok:
            break

        frame = undistort(frame, camera_matrix, distortion)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        found, corners = cv2.findChessboardCorners(
            gray,
            CHECKERBOARD_INTERNAL_CORNERS,
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE,
        )

        if not found:
            continue

        corners = cv2.cornerSubPix(
            gray,
            corners,
            (11, 11),
            (-1, -1),
            criteria,
        )

        spacings.append(
            checkerboard_spacing_px(
                corners,
                CHECKERBOARD_INTERNAL_CORNERS,
            )
        )

    if not spacings:
        raise RuntimeError(
            "No checkerboard was detected during the first 2 seconds."
        )

    median_spacing_px = float(np.median(spacings))
    mm_per_pixel = CHECKERBOARD_SQUARE_SIZE_MM / median_spacing_px

    print("\nCheckerboard calibration")
    print("Successful frames:", len(spacings))
    print(f"Median spacing: {median_spacing_px:.3f} px")
    print(f"Scale: {mm_per_pixel:.6f} mm/px")
    print(f"Scale: {1.0 / mm_per_pixel:.3f} px/mm")

    return mm_per_pixel


# ---------------- INTENDED GAIT ----------------

def smoothstep(t):
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def smooth_move(start, end, t):
    return start + (end - start) * smoothstep(t)


def generate_desired_gait():
    """
    Generate one intended gait cycle using the same phase timings
    as the lookup-table generator.
    """

    phase = np.linspace(0.0, 1.0, N, endpoint=False)

    lower = np.empty(N)
    upper = np.empty(N)

    # 1) Strike
    strike = phase < 0.40
    u = smoothstep(phase[strike] / 0.40)
    lower[strike] = smooth_move(
        LOWER_BACK,
        LOWER_FORWARD,
        u,
    )
    upper[strike] = smooth_move(
        UPPER_UPRIGHT,
        UPPER_STRIKE,
        u,
    )

    # 2) Quick whip/fold
    whip = (phase >= 0.40) & (phase < 0.47)
    u = smoothstep((phase[whip] - 0.40) / 0.07)
    lower[whip] = LOWER_FORWARD
    upper[whip] = smooth_move(
        UPPER_STRIKE,
        UPPER_FOLDED,
        u,
    )

    # 3) Recovery while folded
    recovery = (phase >= 0.47) & (phase < 0.85)
    u = smoothstep((phase[recovery] - 0.47) / 0.38)
    lower[recovery] = smooth_move(
        LOWER_FORWARD,
        LOWER_BACK,
        u,
    )
    upper[recovery] = UPPER_FOLDED

    # 4) Reset upper joint
    reset = phase >= 0.85
    u = smoothstep((phase[reset] - 0.85) / 0.15)
    lower[reset] = LOWER_BACK
    upper[reset] = smooth_move(
        UPPER_FOLDED,
        UPPER_UPRIGHT,
        u,
    )

    return lower, upper


def forward_kinematics(lower_deg, upper_servo_deg):
    """
    Forward kinematics for the physical cilium.

    The upper servo is approximately straight when commanded to 90 degrees.
    Therefore, the actual relative link angle is:

        upper_servo_deg - 90
    """

    lower_rad = np.deg2rad(lower_deg)

    upper_relative_deg = upper_servo_deg - 90.0

    upper_absolute_rad = np.deg2rad(
        lower_deg + upper_relative_deg
    )

    x = (
        L1 * np.cos(lower_rad)
        + L2 * np.cos(upper_absolute_rad)
    )

    z = (
        L1 * np.sin(lower_rad)
        + L2 * np.sin(upper_absolute_rad)
    )

    return x, z


# ---------------- GREEN TIP TRACKING ----------------

def detect_green_tip(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        GREEN_HSV_LOWER,
        GREEN_HSV_UPPER,
    )

    kernel = np.ones((5, 5), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )

    valid = [
        contour
        for contour in contours
        if MIN_MARKER_AREA_PX
        <= cv2.contourArea(contour)
        <= MAX_MARKER_AREA_PX
    ]

    if not valid:
        return None

    contour = max(valid, key=cv2.contourArea)
    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    x = moments["m10"] / moments["m00"]
    y = moments["m01"] / moments["m00"]
    area = cv2.contourArea(contour)

    return float(x), float(y), float(area)


def remove_large_jumps(x_mm, z_mm):
    if MAX_FRAME_JUMP_MM is None or len(x_mm) < 2:
        return np.ones(len(x_mm), dtype=bool)

    keep = np.zeros(len(x_mm), dtype=bool)
    keep[0] = True

    previous_x = x_mm[0]
    previous_z = z_mm[0]

    for i in range(1, len(x_mm)):
        jump = math.hypot(
            x_mm[i] - previous_x,
            z_mm[i] - previous_z,
        )

        if jump <= MAX_FRAME_JUMP_MM:
            keep[i] = True
            previous_x = x_mm[i]
            previous_z = z_mm[i]

    return keep


def track_green_tip(
    capture,
    fps,
    frame_count,
    mm_per_pixel,
    camera_matrix,
    distortion,
):
    start_frame = int(round(TRACKING_START_S * fps))
    capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    frames = []
    times = []
    x_px = []
    y_px = []
    areas = []

    for frame_index in range(start_frame, frame_count):
        ok, frame = capture.read()
        if not ok:
            break

        frame = undistort(frame, camera_matrix, distortion)
        detection = detect_green_tip(frame)

        if detection is None:
            continue

        x, y, area = detection

        frames.append(frame_index)
        times.append(frame_index / fps)
        x_px.append(x)
        y_px.append(y)
        areas.append(area)

    if not x_px:
        raise RuntimeError(
            "No green marker was detected after 5 seconds."
        )

    frames = np.asarray(frames)
    times = np.asarray(times)
    x_px = np.asarray(x_px)
    y_px = np.asarray(y_px)
    areas = np.asarray(areas)

    x_mm = x_px * mm_per_pixel
    z_mm = y_px * mm_per_pixel

    if FLIP_X:
        x_mm = -x_mm

    if FLIP_Z:
        z_mm = -z_mm

    keep = remove_large_jumps(x_mm, z_mm)

    total_tracking_frames = max(frame_count - start_frame, 1)
    tracking_success = 100.0 * np.count_nonzero(keep) / total_tracking_frames

    print("\nGreen tip tracking")
    print("Accepted detections:", np.count_nonzero(keep))
    print("Rejected jumps:", np.count_nonzero(~keep))
    print(f"Tracking success: {tracking_success:.1f}%")

    return {
        "frame": frames[keep],
        "time_s": times[keep],
        "x_px": x_px[keep],
        "y_px": y_px[keep],
        "area_px2": areas[keep],
        "x_mm": x_mm[keep],
        "z_mm": z_mm[keep],
    }


# ---------------- OUTPUTS ----------------

def save_csv(data):
    path = OUTPUT_DIRECTORY / "tracked_tip_positions.csv"

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "frame",
                "time_s",
                "x_pixel",
                "y_pixel",
                "marker_area_pixel2",
                "x_mm",
                "z_mm",
            ]
        )

        writer.writerows(
            zip(
                data["frame"],
                data["time_s"],
                data["x_px"],
                data["y_px"],
                data["area_px2"],
                data["x_mm"],
                data["z_mm"],
            )
        )

    print("Saved:", path)


def centre_path(x, z):
    return x - np.mean(x), z - np.mean(z)


def plot_overlay(desired_x, desired_z, measured_x, measured_z):
    desired_x, desired_z = centre_path(desired_x, desired_z)
    measured_x, measured_z = centre_path(measured_x, measured_z)

    desired_x = np.append(desired_x, desired_x[0])
    desired_z = np.append(desired_z, desired_z[0])

    fig, ax = plt.subplots(figsize=(8, 7))

    ax.plot(
        desired_x,
        desired_z,
        linewidth=2.2,
        label="Intended tip path",
    )

    ax.scatter(
        measured_x,
        measured_z,
        s=7,
        alpha=0.55,
        label="Measured green-tip positions",
    )

    ax.set_title("Intended and measured cilium tip trajectories")
    ax.set_xlabel("Horizontal displacement, x (mm)")
    ax.set_ylabel("Vertical displacement, z (mm)")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.3)
    ax.legend()

    fig.tight_layout()

    path = OUTPUT_DIRECTORY / "desired_vs_measured_tip_path.png"
    fig.savefig(path, dpi=300, bbox_inches="tight")

    print("Saved:", path)
    plt.show()


# ---------------- MAIN ----------------

def main():
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)

    if not VIDEO_PATH.exists():
        raise FileNotFoundError(
            f"Video not found:\n{VIDEO_PATH}\n"
            "Change VIDEO_PATH at the top of the script."
        )

    camera_matrix, distortion = load_camera_calibration(
        CAMERA_CALIBRATION_PATH
    )

    capture = cv2.VideoCapture(str(VIDEO_PATH))

    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        raise RuntimeError("Could not read the video frame rate.")

    print("Video:", VIDEO_PATH)
    print(f"FPS: {fps:.3f}")
    print("Frames:", frame_count)
    print(f"Duration: {frame_count / fps:.2f} s")

    mm_per_pixel = calculate_mm_per_pixel(
        capture,
        fps,
        camera_matrix,
        distortion,
    )

    measured = track_green_tip(
        capture,
        fps,
        frame_count,
        mm_per_pixel,
        camera_matrix,
        distortion,
    )

    capture.release()

    lower_angles, upper_angles = generate_desired_gait()
    desired_x, desired_z = forward_kinematics(
        lower_angles,
        upper_angles,
    )

    print("\nIntended path dimensions")
    print(f"Width:  {np.ptp(desired_x):.3f} mm")
    print(f"Height: {np.ptp(desired_z):.3f} mm")

    print("\nMeasured path dimensions")
    print(f"Width:  {np.ptp(measured['x_mm']):.3f} mm")
    print(f"Height: {np.ptp(measured['z_mm']):.3f} mm")

    save_csv(measured)

    plot_overlay(
        desired_x,
        desired_z,
        measured["x_mm"],
        measured["z_mm"],
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("\nERROR:", error, file=sys.stderr)
        sys.exit(1)