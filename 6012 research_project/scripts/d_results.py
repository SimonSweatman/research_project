#!/usr/bin/env python3

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# FILE PATHS
# ============================================================

VIDEO_PATH = Path(
    r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260717_122544.mp4"
)

CAMERA_CALIBRATION_PATH = Path(r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\research_project\6012 research_project\scripts\checkerboard_calibration_outputs\camera_calibration.npz")

OUTPUT_FOLDER = Path(r"C:\temp\d_shape_tip_validation")

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

# ============================================================
# VIDEO TIMING
# ============================================================

# Checkerboard is visible during the first two seconds
CHECKERBOARD_START_S = 0.0
CHECKERBOARD_END_S = 2.0

# Checkerboard is removed between 2 and 5 seconds
TRACKING_START_S = 5.0

# None means track until the recording ends
TRACKING_END_S = None

# Test every third checkerboard frame
CHECKERBOARD_FRAME_STEP = 3


# ============================================================
# CHECKERBOARD SETTINGS
# ============================================================

# A 10 × 8 square checkerboard contains 9 × 7 internal corners
CHECKERBOARD_INTERNAL_CORNERS = (9, 7)

# Actual printed square size
CHECKERBOARD_SQUARE_SIZE_MM = 10.0


# ============================================================
# CAMERA-TRACKING SETTINGS
# ============================================================

# OpenCV HSV range for the lime-green marker
LOWER_GREEN_HSV = np.array(
    [35, 70, 60],
    dtype=np.uint8
)

UPPER_GREEN_HSV = np.array(
    [90, 255, 255],
    dtype=np.uint8
)

# Reject contours outside the expected marker size
MIN_MARKER_AREA_PX2 = 30.0
MAX_MARKER_AREA_PX2 = 20000.0

# Remove isolated mask noise
MORPHOLOGY_KERNEL_SIZE = 5


# ============================================================
# CILIUM AND D-SHAPE SETTINGS
# These must match the gait-generator script.
# ============================================================

# Link lengths [mm]
L1 = 45.0
L2 = 25.0

# Number of points used to represent the planned paths
N_PATH_POINTS = 720

# Neutral servo commands
HOME_LOWER_COMMAND_DEG = 90.0
HOME_UPPER_COMMAND_DEG = 90.0

# Mechanical-angle calibration
LOWER_CMD_TO_MECH_OFFSET = 0.0
UPPER_CMD_TO_MECH_OFFSET = -90.0

# Select inverse-kinematics solution
ELBOW_SIGN = 1

# D-shape dimensions
D_HALF_WIDTH_X = 12.0
D_RECOVERY_DEPTH_Z = 8.0

# The flat edge is placed below the neutral tip because the
# 90°/90° position is close to maximum reach.
D_TOP_DROP_FROM_HOME_Z = 2.0

# Percentage of the cycle used for the flat upper stroke
STRAIGHT_STROKE_FRACTION = 0.50

# True: top stroke runs from left to right
TOP_STROKE_LEFT_TO_RIGHT = True


# ============================================================
# PCA9685 MODEL
# These must match the Python generator and Arduino code.
# ============================================================

PWM_FREQUENCY_HZ = 50.0
PCA9685_COUNTS_PER_CYCLE = 4096

MIN_SERVO_PULSE_US = 600.0
MAX_SERVO_PULSE_US = 2400.0

SERVO_ANGLE_RANGE_DEG = 180.0


# ============================================================
# GENERAL HELPERS
# ============================================================

def clamp(value, minimum, maximum):
    return max(
        minimum,
        min(maximum, value)
    )


def smoothstep(value):
    """
    Smooth interpolation between zero and one.
    """
    value = clamp(
        value,
        0.0,
        1.0
    )

    return (
        value
        * value
        * (3.0 - 2.0 * value)
    )


# ============================================================
# CAMERA CALIBRATION AND UNDISTORTION
# ============================================================

if not CAMERA_CALIBRATION_PATH.exists():
    raise FileNotFoundError(
        "Camera calibration file not found:\n"
        f"{CAMERA_CALIBRATION_PATH}"
    )

camera_calibration = np.load(
    CAMERA_CALIBRATION_PATH
)

camera_matrix = camera_calibration[
    "camera_matrix"
]

distortion_coefficients = camera_calibration[
    "dist_coeffs"
]

new_camera_matrix = None


def undistort_frame(frame):
    """
    Apply the saved checkerboard camera calibration.
    """
    return cv2.undistort(
        frame,
        camera_matrix,
        distortion_coefficients,
        None,
        new_camera_matrix
    )


# ============================================================
# CILIUM KINEMATICS
# ============================================================

def forward_kinematics(
    lower_command_deg,
    upper_command_deg
):
    """
    Calculate the tip position from the servo commands.
    """
    lower_mechanical_deg = (
        lower_command_deg
        + LOWER_CMD_TO_MECH_OFFSET
    )

    upper_relative_mechanical_deg = (
        upper_command_deg
        + UPPER_CMD_TO_MECH_OFFSET
    )

    theta_1 = np.deg2rad(
        lower_mechanical_deg
    )

    theta_2 = np.deg2rad(
        upper_relative_mechanical_deg
    )

    x = (
        L1 * np.cos(theta_1)
        + L2 * np.cos(theta_1 + theta_2)
    )

    z = (
        L1 * np.sin(theta_1)
        + L2 * np.sin(theta_1 + theta_2)
    )

    return float(x), float(z)


def inverse_kinematics(x, z):
    """
    Calculate the two mechanical joint angles required to reach
    an x-z coordinate.
    """
    radius_squared = x**2 + z**2
    radius = np.sqrt(radius_squared)

    minimum_reach = abs(L1 - L2)
    maximum_reach = L1 + L2

    if (
        radius < minimum_reach
        or radius > maximum_reach
    ):
        raise ValueError(
            f"Requested point is outside the workspace: "
            f"x={x:.3f} mm, z={z:.3f} mm, "
            f"radius={radius:.3f} mm."
        )

    cos_theta_2 = (
        radius_squared
        - L1**2
        - L2**2
    ) / (
        2.0 * L1 * L2
    )

    cos_theta_2 = clamp(
        cos_theta_2,
        -1.0,
        1.0
    )

    theta_2 = (
        ELBOW_SIGN
        * np.arccos(cos_theta_2)
    )

    k1 = L1 + L2 * np.cos(theta_2)
    k2 = L2 * np.sin(theta_2)

    theta_1 = (
        np.arctan2(z, x)
        - np.arctan2(k2, k1)
    )

    return (
        np.rad2deg(theta_1),
        np.rad2deg(theta_2)
    )


def mechanical_to_servo(
    lower_mechanical_deg,
    upper_relative_mechanical_deg
):
    lower_command_deg = (
        lower_mechanical_deg
        - LOWER_CMD_TO_MECH_OFFSET
    )

    upper_command_deg = (
        upper_relative_mechanical_deg
        - UPPER_CMD_TO_MECH_OFFSET
    )

    return (
        lower_command_deg,
        upper_command_deg
    )


# ============================================================
# D-SHAPED PATH
# ============================================================

home_tip_x_mm, home_tip_z_mm = forward_kinematics(
    HOME_LOWER_COMMAND_DEG,
    HOME_UPPER_COMMAND_DEG
)

D_CENTRE_X = home_tip_x_mm

D_TOP_Z = (
    home_tip_z_mm
    - D_TOP_DROP_FROM_HOME_Z
)


def d_shape_position(phase):
    """
    Produce a sideways D-shape with:

    - flat edge at the top;
    - curved recovery underneath.
    """
    phase = phase % 1.0

    left_x = (
        D_CENTRE_X
        - D_HALF_WIDTH_X
    )

    right_x = (
        D_CENTRE_X
        + D_HALF_WIDTH_X
    )

    if phase < STRAIGHT_STROKE_FRACTION:
        local_phase = (
            phase
            / STRAIGHT_STROKE_FRACTION
        )

        interpolation = smoothstep(
            local_phase
        )

        if TOP_STROKE_LEFT_TO_RIGHT:
            x = (
                left_x
                + (right_x - left_x)
                * interpolation
            )
        else:
            x = (
                right_x
                + (left_x - right_x)
                * interpolation
            )

        z = D_TOP_Z

    else:
        local_phase = (
            phase
            - STRAIGHT_STROKE_FRACTION
        ) / (
            1.0
            - STRAIGHT_STROKE_FRACTION
        )

        theta = (
            np.pi
            * smoothstep(local_phase)
        )

        if TOP_STROKE_LEFT_TO_RIGHT:
            x = (
                D_CENTRE_X
                + D_HALF_WIDTH_X
                * np.cos(theta)
            )
        else:
            x = (
                D_CENTRE_X
                - D_HALF_WIDTH_X
                * np.cos(theta)
            )

        z = (
            D_TOP_Z
            - D_RECOVERY_DEPTH_Z
            * np.sin(theta)
        )

    return x, z


# ============================================================
# PCA9685 QUANTISATION
# ============================================================

def servo_angle_to_pca_count(
    servo_angle_deg
):
    """
    Convert a decimal servo command to the nearest PCA9685 count.
    """
    servo_angle_deg = float(
        np.clip(
            servo_angle_deg,
            0.0,
            SERVO_ANGLE_RANGE_DEG
        )
    )

    pulse_width_us = (
        MIN_SERVO_PULSE_US
        + servo_angle_deg
        / SERVO_ANGLE_RANGE_DEG
        * (
            MAX_SERVO_PULSE_US
            - MIN_SERVO_PULSE_US
        )
    )

    pwm_period_us = (
        1_000_000.0
        / PWM_FREQUENCY_HZ
    )

    pca_count = int(
        round(
            pulse_width_us
            / pwm_period_us
            * PCA9685_COUNTS_PER_CYCLE
        )
    )

    return int(
        np.clip(
            pca_count,
            0,
            PCA9685_COUNTS_PER_CYCLE - 1
        )
    )


def pca_count_to_servo_angle(
    pca_count
):
    """
    Convert the quantised PCA9685 count back into its equivalent
    servo angle for forward-kinematics prediction.
    """
    pwm_period_us = (
        1_000_000.0
        / PWM_FREQUENCY_HZ
    )

    pulse_width_us = (
        pca_count
        / PCA9685_COUNTS_PER_CYCLE
        * pwm_period_us
    )

    servo_angle_deg = (
        pulse_width_us
        - MIN_SERVO_PULSE_US
    ) / (
        MAX_SERVO_PULSE_US
        - MIN_SERVO_PULSE_US
    ) * SERVO_ANGLE_RANGE_DEG

    return float(servo_angle_deg)


# ============================================================
# GENERATE PLANNED AND PCA-LIMITED PATHS
# ============================================================

planned_x_mm = np.zeros(
    N_PATH_POINTS,
    dtype=float
)

planned_z_mm = np.zeros(
    N_PATH_POINTS,
    dtype=float
)

pca_x_mm = np.zeros(
    N_PATH_POINTS,
    dtype=float
)

pca_z_mm = np.zeros(
    N_PATH_POINTS,
    dtype=float
)

lower_float_commands = np.zeros(
    N_PATH_POINTS,
    dtype=float
)

upper_float_commands = np.zeros(
    N_PATH_POINTS,
    dtype=float
)

lower_pwm_counts = np.zeros(
    N_PATH_POINTS,
    dtype=np.uint16
)

upper_pwm_counts = np.zeros(
    N_PATH_POINTS,
    dtype=np.uint16
)

for index in range(N_PATH_POINTS):
    phase = index / N_PATH_POINTS

    desired_x, desired_z = d_shape_position(
        phase
    )

    planned_x_mm[index] = desired_x
    planned_z_mm[index] = desired_z

    (
        lower_mechanical_deg,
        upper_relative_mechanical_deg
    ) = inverse_kinematics(
        desired_x,
        desired_z
    )

    (
        lower_command_deg,
        upper_command_deg
    ) = mechanical_to_servo(
        lower_mechanical_deg,
        upper_relative_mechanical_deg
    )

    lower_float_commands[index] = (
        lower_command_deg
    )

    upper_float_commands[index] = (
        upper_command_deg
    )

    lower_count = servo_angle_to_pca_count(
        lower_command_deg
    )

    upper_count = servo_angle_to_pca_count(
        upper_command_deg
    )

    lower_pwm_counts[index] = lower_count
    upper_pwm_counts[index] = upper_count

    lower_quantised_deg = (
        pca_count_to_servo_angle(
            lower_count
        )
    )

    upper_quantised_deg = (
        pca_count_to_servo_angle(
            upper_count
        )
    )

    (
        pca_x_mm[index],
        pca_z_mm[index]
    ) = forward_kinematics(
        lower_quantised_deg,
        upper_quantised_deg
    )


# ============================================================
# CHECKERBOARD DETECTION
# ============================================================

def detect_checkerboard(frame):
    """
    Detect the 9 × 7 internal corners with sub-pixel accuracy.
    """
    gray = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD_INTERNAL_CORNERS,
        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
    )

    if found:
        return True, corners

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_INTERNAL_CORNERS,
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    )

    if not found:
        return False, None

    termination_criteria = (
        cv2.TERM_CRITERIA_EPS
        + cv2.TERM_CRITERIA_MAX_ITER,
        50,
        0.001
    )

    corners = cv2.cornerSubPix(
        gray,
        corners,
        winSize=(11, 11),
        zeroZone=(-1, -1),
        criteria=termination_criteria
    )

    return True, corners


def scale_and_orientation_from_corners(
    corners
):
    """
    Calculate:

    - pixels per millimetre;
    - checkerboard horizontal direction;
    - checkerboard vertical direction.

    The checkerboard directions are used to remove small camera
    roll from the measured path.
    """
    corner_columns, corner_rows = (
        CHECKERBOARD_INTERNAL_CORNERS
    )

    corner_grid = corners.reshape(
        corner_rows,
        corner_columns,
        2
    )

    scale_estimates = []

    # Every neighbouring pair of corners is one 10 mm interval
    for row in range(corner_rows):
        for column in range(
            corner_columns - 1
        ):
            distance_px = np.linalg.norm(
                corner_grid[row, column + 1]
                - corner_grid[row, column]
            )

            scale_estimates.append(
                distance_px
                / CHECKERBOARD_SQUARE_SIZE_MM
            )

    for column in range(corner_columns):
        for row in range(
            corner_rows - 1
        ):
            distance_px = np.linalg.norm(
                corner_grid[row + 1, column]
                - corner_grid[row, column]
            )

            scale_estimates.append(
                distance_px
                / CHECKERBOARD_SQUARE_SIZE_MM
            )

    pixels_per_mm = float(
        np.median(scale_estimates)
    )

    # Mean direction across all checkerboard rows
    horizontal_vectors = (
        corner_grid[:, -1, :]
        - corner_grid[:, 0, :]
    )

    horizontal_vector = np.mean(
        horizontal_vectors,
        axis=0
    )

    horizontal_unit = (
        horizontal_vector
        / np.linalg.norm(horizontal_vector)
    )

    # In image coordinates, increasing y is downward.
    # The z direction is therefore perpendicular and upward.
    vertical_up_unit = np.array([
        -horizontal_unit[1],
        horizontal_unit[0]
    ])

    # Make sure the vertical vector points upwards on the image
    if vertical_up_unit[1] > 0:
        vertical_up_unit *= -1.0

    return (
        pixels_per_mm,
        horizontal_unit,
        vertical_up_unit,
        np.asarray(scale_estimates)
    )


def obtain_checkerboard_calibration(
    video_capture,
    fps
):
    """
    Process all usable checkerboard frames in the first two
    seconds and use median values for repeatability.
    """
    first_frame = int(
        CHECKERBOARD_START_S * fps
    )

    final_frame = int(
        CHECKERBOARD_END_S * fps
    )

    video_capture.set(
        cv2.CAP_PROP_POS_FRAMES,
        first_frame
    )

    scale_values = []
    horizontal_vectors = []
    vertical_vectors = []
    successful_frames = []

    best_marked_frame = None

    for frame_index in range(
        first_frame,
        final_frame
    ):
        success, frame = video_capture.read()

        if not success:
            break

        if (
            frame_index - first_frame
        ) % CHECKERBOARD_FRAME_STEP != 0:
            continue

        processed_frame = undistort_frame(
            frame
        )

        found, corners = detect_checkerboard(
            processed_frame
        )

        if not found:
            continue

        (
            pixels_per_mm,
            horizontal_unit,
            vertical_up_unit,
            scale_estimates
        ) = scale_and_orientation_from_corners(
            corners
        )

        scale_values.append(
            pixels_per_mm
        )

        horizontal_vectors.append(
            horizontal_unit
        )

        vertical_vectors.append(
            vertical_up_unit
        )

        successful_frames.append(
            frame_index
        )

        marked_frame = processed_frame.copy()

        cv2.drawChessboardCorners(
            marked_frame,
            CHECKERBOARD_INTERNAL_CORNERS,
            corners,
            True
        )

        cv2.putText(
            marked_frame,
            f"{pixels_per_mm:.5f} px/mm",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        best_marked_frame = marked_frame

    if not scale_values:
        raise RuntimeError(
            "The 9 × 7 checkerboard was not detected during "
            "the first two seconds of the video."
        )

    final_pixels_per_mm = float(
        np.median(scale_values)
    )

    final_mm_per_pixel = (
        1.0 / final_pixels_per_mm
    )

    final_horizontal_unit = np.mean(
        horizontal_vectors,
        axis=0
    )

    final_horizontal_unit /= np.linalg.norm(
        final_horizontal_unit
    )

    final_vertical_unit = np.mean(
        vertical_vectors,
        axis=0
    )

    final_vertical_unit /= np.linalg.norm(
        final_vertical_unit
    )

    scale_variation_percent = (
        np.std(scale_values, ddof=1)
        / final_pixels_per_mm
        * 100.0
        if len(scale_values) > 1
        else 0.0
    )

    if best_marked_frame is not None:
        cv2.imwrite(
            str(
                OUTPUT_FOLDER
                / "checkerboard_detected_corners.png"
            ),
            best_marked_frame
        )

    scale_summary = pd.DataFrame([{
        "successful_checkerboard_frames":
            len(scale_values),

        "pixels_per_mm":
            final_pixels_per_mm,

        "mm_per_pixel":
            final_mm_per_pixel,

        "scale_variation_percent":
            scale_variation_percent,

        "horizontal_unit_x":
            final_horizontal_unit[0],

        "horizontal_unit_y":
            final_horizontal_unit[1],

        "vertical_unit_x":
            final_vertical_unit[0],

        "vertical_unit_y":
            final_vertical_unit[1]
    }])

    scale_summary.to_csv(
        OUTPUT_FOLDER
        / "checkerboard_scale_summary.csv",
        index=False
    )

    print("\n--- Checkerboard calibration ---")
    print(
        f"Successful frames: "
        f"{len(scale_values)}"
    )
    print(
        f"Scale: "
        f"{final_pixels_per_mm:.6f} px/mm"
    )
    print(
        f"Scale: "
        f"{final_mm_per_pixel:.6f} mm/px"
    )
    print(
        f"Between-frame scale variation: "
        f"{scale_variation_percent:.4f}%"
    )

    return (
        final_pixels_per_mm,
        final_mm_per_pixel,
        final_horizontal_unit,
        final_vertical_unit
    )


# ============================================================
# GREEN-TIP DETECTION
# ============================================================

def detect_green_tip(frame):
    """
    Detect the lime-green marker and return its centroid.
    """
    hsv_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2HSV
    )

    mask = cv2.inRange(
        hsv_frame,
        LOWER_GREEN_HSV,
        UPPER_GREEN_HSV
    )

    morphology_kernel = np.ones(
        (
            MORPHOLOGY_KERNEL_SIZE,
            MORPHOLOGY_KERNEL_SIZE
        ),
        dtype=np.uint8
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        morphology_kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        morphology_kernel
    )

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None, mask

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:
        contour_area = cv2.contourArea(
            contour
        )

        if not (
            MIN_MARKER_AREA_PX2
            <= contour_area
            <= MAX_MARKER_AREA_PX2
        ):
            continue

        moments = cv2.moments(
            contour
        )

        if moments["m00"] == 0:
            continue

        centroid_x_px = (
            moments["m10"]
            / moments["m00"]
        )

        centroid_y_px = (
            moments["m01"]
            / moments["m00"]
        )

        marker_radius_px = np.sqrt(
            contour_area / np.pi
        )

        return {
            "x_px": float(centroid_x_px),
            "y_px": float(centroid_y_px),
            "area_px2": float(contour_area),
            "radius_px": float(marker_radius_px)
        }, mask

    return None, mask


# ============================================================
# OPEN VIDEO
# ============================================================

if not VIDEO_PATH.exists():
    raise FileNotFoundError(
        f"Video file not found:\n{VIDEO_PATH}"
    )

video_capture = cv2.VideoCapture(
    str(VIDEO_PATH)
)

if not video_capture.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{VIDEO_PATH}"
    )

fps = video_capture.get(
    cv2.CAP_PROP_FPS
)

frame_width = int(
    video_capture.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)

frame_height = int(
    video_capture.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)

total_frames = int(
    video_capture.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)

video_duration_s = (
    total_frames / fps
)

if fps <= 0:
    raise RuntimeError(
        "The video frame rate could not be read."
    )

print("\n--- Video information ---")
print(f"Video: {VIDEO_PATH.name}")
print(
    f"Resolution: "
    f"{frame_width} × {frame_height}"
)
print(f"Frame rate: {fps:.3f} fps")
print(f"Frames: {total_frames}")
print(
    f"Duration: "
    f"{video_duration_s:.3f} s"
)

new_camera_matrix, _ = (
    cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion_coefficients,
        (frame_width, frame_height),
        alpha=1,
        newImgSize=(
            frame_width,
            frame_height
        )
    )
)


# ============================================================
# CALCULATE SCALE FROM FIRST TWO SECONDS
# ============================================================

(
    pixels_per_mm,
    mm_per_pixel,
    checkerboard_x_unit,
    checkerboard_z_unit
) = obtain_checkerboard_calibration(
    video_capture,
    fps
)


# ============================================================
# TRACK TIP FROM FIVE SECONDS ONWARDS
# ============================================================

tracking_start_frame = int(
    TRACKING_START_S * fps
)

if TRACKING_END_S is None:
    tracking_end_frame = total_frames
else:
    tracking_end_frame = min(
        int(TRACKING_END_S * fps),
        total_frames
    )

video_capture.set(
    cv2.CAP_PROP_POS_FRAMES,
    tracking_start_frame
)

overlay_path = (
    OUTPUT_FOLDER
    / "green_tip_tracking_overlay.mp4"
)

video_writer = cv2.VideoWriter(
    str(overlay_path),
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (
        frame_width,
        frame_height
    )
)

if not video_writer.isOpened():
    raise RuntimeError(
        "Could not create the tracking overlay video."
    )

tracking_rows = []

frame_index = tracking_start_frame

while frame_index < tracking_end_frame:
    success, frame = video_capture.read()

    if not success:
        break

    video_time_s = (
        frame_index / fps
    )

    processed_frame = undistort_frame(
        frame
    )

    marker, marker_mask = detect_green_tip(
        processed_frame
    )

    overlay_frame = processed_frame.copy()

    if marker is not None:
        marker_x_px = marker["x_px"]
        marker_y_px = marker["y_px"]
        marker_radius_px = marker[
            "radius_px"
        ]
        marker_area_px2 = marker[
            "area_px2"
        ]

        cv2.circle(
            overlay_frame,
            (
                int(round(marker_x_px)),
                int(round(marker_y_px))
            ),
            int(round(marker_radius_px)),
            (0, 255, 0),
            2
        )

        cv2.circle(
            overlay_frame,
            (
                int(round(marker_x_px)),
                int(round(marker_y_px))
            ),
            4,
            (0, 0, 255),
            -1
        )

        status_text = "Tip detected"
        status_colour = (0, 255, 0)

    else:
        marker_x_px = np.nan
        marker_y_px = np.nan
        marker_radius_px = np.nan
        marker_area_px2 = np.nan

        status_text = "Tip not detected"
        status_colour = (0, 0, 255)

    cv2.putText(
        overlay_frame,
        (
            f"Video time: {video_time_s:.2f} s | "
            f"{status_text}"
        ),
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        status_colour,
        2
    )

    video_writer.write(
        overlay_frame
    )

    tracking_rows.append({
        "frame": frame_index,
        "video_time_s": video_time_s,
        "x_px": marker_x_px,
        "y_px": marker_y_px,
        "marker_radius_px":
            marker_radius_px,
        "marker_radius_mm":
            marker_radius_px * mm_per_pixel
            if not np.isnan(marker_radius_px)
            else np.nan,
        "marker_area_px2":
            marker_area_px2
    })

    frame_index += 1

video_capture.release()
video_writer.release()

tracking_data = pd.DataFrame(
    tracking_rows
)

tracking_data.to_csv(
    OUTPUT_FOLDER
    / "raw_green_tip_tracking.csv",
    index=False
)

valid_tracking = tracking_data.dropna(
    subset=["x_px", "y_px"]
).copy()

valid_detection_percent = (
    len(valid_tracking)
    / len(tracking_data)
    * 100.0
    if len(tracking_data) > 0
    else 0.0
)

print("\n--- Green-tip tracking ---")
print(
    f"Valid detections: "
    f"{len(valid_tracking)} / "
    f"{len(tracking_data)} "
    f"({valid_detection_percent:.2f}%)"
)

if len(valid_tracking) < 20:
    raise RuntimeError(
        "Too few valid marker detections were obtained."
    )


# ============================================================
# CONVERT IMAGE COORDINATES TO CHECKERBOARD x-z COORDINATES
# ============================================================

# Use the first valid tracked point as a temporary image origin
origin_x_px = valid_tracking[
    "x_px"
].iloc[0]

origin_y_px = valid_tracking[
    "y_px"
].iloc[0]

delta_x_px = (
    valid_tracking["x_px"]
    - origin_x_px
)

delta_y_px = (
    valid_tracking["y_px"]
    - origin_y_px
)

image_displacements = np.column_stack((
    delta_x_px.to_numpy(),
    delta_y_px.to_numpy()
))

# Project pixel displacement onto checkerboard-aligned axes
measured_x_relative_mm = (
    image_displacements
    @ checkerboard_x_unit
) * mm_per_pixel

measured_z_relative_mm = (
    image_displacements
    @ checkerboard_z_unit
) * mm_per_pixel

valid_tracking[
    "x_relative_mm"
] = measured_x_relative_mm

valid_tracking[
    "z_relative_mm"
] = measured_z_relative_mm


# ============================================================
# CENTRE-ALIGN PATHS FOR SHAPE COMPARISON
# ============================================================

# No scaling is applied here.
# This preserves the measured width, height and distortion.

planned_centre_x = np.mean(
    planned_x_mm
)

planned_centre_z = np.mean(
    planned_z_mm
)

planned_plot_x = (
    planned_x_mm
    - planned_centre_x
)

planned_plot_z = (
    planned_z_mm
    - planned_centre_z
)

pca_centre_x = np.mean(
    pca_x_mm
)

pca_centre_z = np.mean(
    pca_z_mm
)

pca_plot_x = (
    pca_x_mm
    - pca_centre_x
)

pca_plot_z = (
    pca_z_mm
    - pca_centre_z
)

measured_centre_x = np.mean(
    measured_x_relative_mm
)

measured_centre_z = np.mean(
    measured_z_relative_mm
)

measured_plot_x = (
    measured_x_relative_mm
    - measured_centre_x
)

measured_plot_z = (
    measured_z_relative_mm
    - measured_centre_z
)

valid_tracking[
    "x_centred_mm"
] = measured_plot_x

valid_tracking[
    "z_centred_mm"
] = measured_plot_z

valid_tracking.to_csv(
    OUTPUT_FOLDER
    / "green_tip_tracking_mm.csv",
    index=False
)


# ============================================================
# PATH DIMENSIONS
# ============================================================

def path_dimensions(x_values, z_values):
    return {
        "width_mm": float(
            np.max(x_values)
            - np.min(x_values)
        ),
        "height_mm": float(
            np.max(z_values)
            - np.min(z_values)
        )
    }


planned_dimensions = path_dimensions(
    planned_plot_x,
    planned_plot_z
)

pca_dimensions = path_dimensions(
    pca_plot_x,
    pca_plot_z
)

measured_dimensions = path_dimensions(
    measured_plot_x,
    measured_plot_z
)


def percentage_error(
    measured_value,
    reference_value
):
    if reference_value == 0:
        return np.nan

    return (
        measured_value
        - reference_value
    ) / reference_value * 100.0


summary = pd.DataFrame([
    {
        "path": "Planned D-shape",
        "width_mm":
            planned_dimensions["width_mm"],
        "height_mm":
            planned_dimensions["height_mm"],
        "width_error_vs_planned_percent":
            0.0,
        "height_error_vs_planned_percent":
            0.0
    },
    {
        "path": "PCA9685-limited prediction",
        "width_mm":
            pca_dimensions["width_mm"],
        "height_mm":
            pca_dimensions["height_mm"],
        "width_error_vs_planned_percent":
            percentage_error(
                pca_dimensions["width_mm"],
                planned_dimensions["width_mm"]
            ),
        "height_error_vs_planned_percent":
            percentage_error(
                pca_dimensions["height_mm"],
                planned_dimensions["height_mm"]
            )
    },
    {
        "path": "Measured tip path",
        "width_mm":
            measured_dimensions["width_mm"],
        "height_mm":
            measured_dimensions["height_mm"],
        "width_error_vs_planned_percent":
            percentage_error(
                measured_dimensions["width_mm"],
                planned_dimensions["width_mm"]
            ),
        "height_error_vs_planned_percent":
            percentage_error(
                measured_dimensions["height_mm"],
                planned_dimensions["height_mm"]
            )
    }
])

summary.to_csv(
    OUTPUT_FOLDER
    / "path_dimension_comparison.csv",
    index=False
)


# ============================================================
# CREATE COMPARISON PLOT
# ============================================================

plt.figure(
    figsize=(9, 7)
)

plt.plot(
    planned_plot_x,
    planned_plot_z,
    "--",
    linewidth=2.2,
    label="Planned D-shape"
)

plt.plot(
    pca_plot_x,
    pca_plot_z,
    linewidth=2.0,
    label="PCA9685-limited prediction"
)

plt.plot(
    measured_plot_x,
    measured_plot_z,
    ".",
    markersize=2.5,
    alpha=0.65,
    label="Measured green-tip path"
)

plt.scatter(
    [planned_plot_x[0]],
    [planned_plot_z[0]],
    s=55,
    marker="x",
    label="Commanded start"
)

plt.xlabel(
    "Horizontal tip position, x [mm]"
)

plt.ylabel(
    "Vertical tip position, z [mm]"
)

plt.title(
    "Planned, controller-limited and measured "
    "cilium tip paths"
)

plt.axis("equal")
plt.grid(
    True,
    alpha=0.3
)
plt.legend()
plt.tight_layout()

comparison_plot_path = (
    OUTPUT_FOLDER
    / "planned_pca_measured_d_shape.png"
)

plt.savefig(
    comparison_plot_path,
    dpi=300
)

plt.close()


# ============================================================
# SAVE SEPARATE DATA FOR THEORETICAL PATHS
# ============================================================

theoretical_paths = pd.DataFrame({
    "sample": np.arange(
        N_PATH_POINTS
    ),
    "phase": np.arange(
        N_PATH_POINTS
    ) / N_PATH_POINTS,
    "planned_x_mm":
        planned_plot_x,
    "planned_z_mm":
        planned_plot_z,
    "pca_x_mm":
        pca_plot_x,
    "pca_z_mm":
        pca_plot_z,
    "lower_float_command_deg":
        lower_float_commands,
    "upper_float_command_deg":
        upper_float_commands,
    "lower_pwm_count":
        lower_pwm_counts,
    "upper_pwm_count":
        upper_pwm_counts
})

theoretical_paths.to_csv(
    OUTPUT_FOLDER
    / "planned_and_pca_paths.csv",
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n--- Path dimensions ---")
print(
    summary.to_string(
        index=False
    )
)

print("\n--- Files saved ---")
print(
    "Checkerboard image:\n"
    f"{OUTPUT_FOLDER / 'checkerboard_detected_corners.png'}"
)
print(
    "Tracking overlay:\n"
    f"{overlay_path}"
)
print(
    "Raw tracking data:\n"
    f"{OUTPUT_FOLDER / 'raw_green_tip_tracking.csv'}"
)
print(
    "Tracking data in millimetres:\n"
    f"{OUTPUT_FOLDER / 'green_tip_tracking_mm.csv'}"
)
print(
    "Theoretical path data:\n"
    f"{OUTPUT_FOLDER / 'planned_and_pca_paths.csv'}"
)
print(
    "Dimension summary:\n"
    f"{OUTPUT_FOLDER / 'path_dimension_comparison.csv'}"
)
print(
    "Comparison plot:\n"
    f"{comparison_plot_path}"
)