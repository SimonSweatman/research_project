import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# USER SETTINGS
# ==========================================================

VIDEO_PATH = r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260714_161627.mp4"

CALIBRATION_NPZ = r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\research_project\6012 research_project\scripts\checkerboard_calibration_outputs\camera_calibration.npz"

OUTPUT_FOLDER = Path(r"C:\temp\tracking_method_comparison")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 10 x 8 checkerboard squares = 9 x 7 internal corners
CHECKERBOARD_CORNERS = (9, 7)
SQUARE_SIZE_MM = 10.0

# Recording sequence
SCALE_END_S = 2.0
TRACKING_START_S = 5.0
CHECKERBOARD_FRAME_STEP = 3

# Physical marker size
MARKER_DIAMETER_MM = 8.0

# Lime-green HSV limits
LOWER_GREEN = np.array([35, 70, 60])
UPPER_GREEN = np.array([85, 255, 255])

MIN_MARKER_AREA = 30
MAX_MARKER_AREA = 20000

# Hough settings
HOUGH_DP = 1.2
HOUGH_PARAM1 = 80
HOUGH_PARAM2 = 12
HOUGH_RADIUS_TOLERANCE = 0.45

# Maximum allowed frame-to-frame Hough centre movement.
# This helps stop Hough jumping to another circular object.
MAX_HOUGH_JUMP_MM = 15.0

METHODS = [
    "centroid",
    "min_enclosing_circle",
    "hough"
]


# ==========================================================
# LOAD CAMERA CALIBRATION
# ==========================================================

calibration_path = Path(CALIBRATION_NPZ)

if not calibration_path.exists():
    raise FileNotFoundError(
        f"Camera calibration file not found:\n{calibration_path}"
    )

calibration = np.load(calibration_path)

camera_matrix = calibration["camera_matrix"]
dist_coeffs = calibration["dist_coeffs"]

new_camera_matrix = None


def undistort_frame(frame):
    return cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )


# ==========================================================
# CHECKERBOARD SCALE
# ==========================================================

def detect_checkerboard(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCornersSB(
        gray,
        CHECKERBOARD_CORNERS,
        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
    )

    if found:
        return True, corners, "findChessboardCornersSB"

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD_CORNERS,
        flags=(
            cv2.CALIB_CB_ADAPTIVE_THRESH
            | cv2.CALIB_CB_NORMALIZE_IMAGE
        )
    )

    if not found:
        return False, None, None

    criteria = (
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
        criteria=criteria
    )

    return True, corners, "findChessboardCorners + cornerSubPix"


def scale_from_checkerboard(corners):
    """
    Calculate scale using all complete checkerboard rows and columns.
    """

    columns, rows = CHECKERBOARD_CORNERS
    grid = corners.reshape(rows, columns, 2)

    estimates = []

    horizontal_length_mm = (columns - 1) * SQUARE_SIZE_MM
    vertical_length_mm = (rows - 1) * SQUARE_SIZE_MM

    for row in range(rows):
        distance_px = np.linalg.norm(
            grid[row, -1] - grid[row, 0]
        )

        estimates.append(
            distance_px / horizontal_length_mm
        )

    for column in range(columns):
        distance_px = np.linalg.norm(
            grid[-1, column] - grid[0, column]
        )

        estimates.append(
            distance_px / vertical_length_mm
        )

    estimates = np.asarray(estimates, dtype=float)

    pixels_per_mm = np.median(estimates)
    mm_per_pixel = 1.0 / pixels_per_mm

    return pixels_per_mm, mm_per_pixel, estimates


def obtain_scale(cap, fps):
    """
    Search the first two seconds and use the median scale from all
    successful checkerboard detections.
    """

    end_frame = int(SCALE_END_S * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    scale_rows = []
    detection_images = []

    for frame_index in range(end_frame):
        ret, frame = cap.read()

        if not ret:
            break

        if frame_index % CHECKERBOARD_FRAME_STEP != 0:
            continue

        processed = undistort_frame(frame)

        found, corners, detector = detect_checkerboard(processed)

        if not found:
            continue

        pixels_per_mm, mm_per_pixel, estimates = (
            scale_from_checkerboard(corners)
        )

        scale_rows.append({
            "frame": frame_index,
            "time_s": frame_index / fps,
            "detector": detector,
            "pixels_per_mm": pixels_per_mm,
            "mm_per_pixel": mm_per_pixel,
            "within_frame_scale_std": np.std(estimates, ddof=1)
        })

        marked = processed.copy()

        cv2.drawChessboardCorners(
            marked,
            CHECKERBOARD_CORNERS,
            corners,
            True
        )

        cv2.putText(
            marked,
            f"{pixels_per_mm:.5f} px/mm",
            (30, 45),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        detection_images.append(
            (pixels_per_mm, frame_index, marked)
        )

    if not scale_rows:
        raise RuntimeError(
            "The checkerboard was not detected during the first "
            f"{SCALE_END_S:.1f} seconds."
        )

    scale_df = pd.DataFrame(scale_rows)

    final_pixels_per_mm = scale_df["pixels_per_mm"].median()
    final_mm_per_pixel = 1.0 / final_pixels_per_mm

    best_image_index = np.argmin(
        np.abs(
            scale_df["pixels_per_mm"].to_numpy()
            - final_pixels_per_mm
        )
    )

    _, best_frame, best_image = detection_images[best_image_index]

    cv2.putText(
        best_image,
        f"Final scale: {final_pixels_per_mm:.5f} px/mm",
        (30, 85),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imwrite(
        str(OUTPUT_FOLDER / "checkerboard_detected_corners.png"),
        best_image
    )

    scale_df.to_csv(
        OUTPUT_FOLDER / "checkerboard_scale_detections.csv",
        index=False
    )

    scale_cv_percent = (
        scale_df["pixels_per_mm"].std()
        / final_pixels_per_mm
        * 100
        if len(scale_df) > 1
        else 0.0
    )

    print("\n--- Checkerboard scale ---")
    print(f"Successful frames: {len(scale_df)}")
    print(f"Best detection frame: {best_frame}")
    print(f"Scale: {final_pixels_per_mm:.6f} px/mm")
    print(f"Scale: {final_mm_per_pixel:.6f} mm/px")
    print(f"Between-frame scale variation: {scale_cv_percent:.4f}%")

    return final_pixels_per_mm, final_mm_per_pixel


# ==========================================================
# GREEN MASK AND CONTOUR
# ==========================================================

def get_green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(
        hsv,
        LOWER_GREEN,
        UPPER_GREEN
    )

    kernel = np.ones((5, 5), np.uint8)

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        kernel
    )

    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_CLOSE,
        kernel
    )

    return mask


def get_largest_valid_contour(mask):
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    contours = sorted(
        contours,
        key=cv2.contourArea,
        reverse=True
    )

    for contour in contours:
        area = cv2.contourArea(contour)

        if MIN_MARKER_AREA <= area <= MAX_MARKER_AREA:
            return contour

    return None


# ==========================================================
# THREE TRACKING METHODS
# ==========================================================

def detect_centroid(mask):
    """
    Centre of mass of the detected green pixels.
    """

    contour = get_largest_valid_contour(mask)

    if contour is None:
        return None

    moments = cv2.moments(contour)

    if moments["m00"] == 0:
        return None

    return {
        "x_px": moments["m10"] / moments["m00"],
        "y_px": moments["m01"] / moments["m00"],
        "radius_px": np.sqrt(
            cv2.contourArea(contour) / np.pi
        ),
        "area_px2": cv2.contourArea(contour)
    }


def detect_min_enclosing_circle(mask):
    """
    Smallest circle that contains the complete green contour.
    """

    contour = get_largest_valid_contour(mask)

    if contour is None:
        return None

    (x_px, y_px), radius_px = cv2.minEnclosingCircle(contour)

    return {
        "x_px": float(x_px),
        "y_px": float(y_px),
        "radius_px": float(radius_px),
        "area_px2": cv2.contourArea(contour)
    }


def detect_hough_circle(
    mask,
    expected_radius_px,
    previous_position=None,
    max_jump_px=None
):
    """
    Hough circle detection on the colour-isolated green mask.
    """

    blurred = cv2.GaussianBlur(mask, (9, 9), 2)

    minimum_radius = max(
        2,
        int(
            expected_radius_px
            * (1.0 - HOUGH_RADIUS_TOLERANCE)
        )
    )

    maximum_radius = max(
        minimum_radius + 1,
        int(
            expected_radius_px
            * (1.0 + HOUGH_RADIUS_TOLERANCE)
        )
    )

    circles = cv2.HoughCircles(
        blurred,
        cv2.HOUGH_GRADIENT,
        dp=HOUGH_DP,
        minDist=max(10, int(expected_radius_px * 2)),
        param1=HOUGH_PARAM1,
        param2=HOUGH_PARAM2,
        minRadius=minimum_radius,
        maxRadius=maximum_radius
    )

    if circles is None:
        return None

    circles = circles[0]

    candidates = []

    for x_px, y_px, radius_px in circles:
        radius_error = abs(radius_px - expected_radius_px)

        if previous_position is None:
            movement = 0.0
        else:
            movement = np.hypot(
                x_px - previous_position[0],
                y_px - previous_position[1]
            )

            if (
                max_jump_px is not None
                and movement > max_jump_px
            ):
                continue

        candidates.append({
            "x_px": float(x_px),
            "y_px": float(y_px),
            "radius_px": float(radius_px),
            "area_px2": np.pi * radius_px**2,
            "score": radius_error + movement
        })

    if not candidates:
        return None

    candidates.sort(key=lambda item: item["score"])

    best = candidates[0]
    best.pop("score")

    return best


# ==========================================================
# GEOMETRY FITTING
# ==========================================================

def fit_circle(x, z):
    matrix = np.column_stack((
        2.0 * x,
        2.0 * z,
        np.ones(len(x))
    ))

    target = x**2 + z**2

    solution, _, _, _ = np.linalg.lstsq(
        matrix,
        target,
        rcond=None
    )

    centre_x = solution[0]
    centre_z = solution[1]

    radius = np.sqrt(
        solution[2]
        + centre_x**2
        + centre_z**2
    )

    return centre_x, centre_z, radius


def analyse_method(method_df, mm_per_pixel):
    valid = method_df.dropna(
        subset=["x_px", "y_px"]
    ).copy()

    if len(valid) < 20:
        return None, None

    # Fit in pixels first, then convert relative coordinates to mm.
    centre_x_px, centre_y_px, radius_px = fit_circle(
        valid["x_px"].to_numpy(),
        valid["y_px"].to_numpy()
    )

    valid["x_mm"] = (
        valid["x_px"] - centre_x_px
    ) * mm_per_pixel

    valid["z_mm"] = -(
        valid["y_px"] - centre_y_px
    ) * mm_per_pixel

    radius_mm = radius_px * mm_per_pixel

    valid["measured_radius_mm"] = np.sqrt(
        valid["x_mm"]**2
        + valid["z_mm"]**2
    )

    valid["radial_error_mm"] = (
        valid["measured_radius_mm"]
        - radius_mm
    )

    valid["absolute_radial_error_mm"] = (
        valid["radial_error_mm"].abs()
    )

    valid["radial_error_percent"] = (
        valid["absolute_radial_error_mm"]
        / radius_mm
        * 100.0
    )

    # Frame-to-frame movement, useful for identifying jitter.
    valid["frame_step_mm"] = np.sqrt(
        valid["x_mm"].diff()**2
        + valid["z_mm"].diff()**2
    )

    summary = {
        "method": valid["method"].iloc[0],
        "valid_frames": len(valid),
        "total_frames": len(method_df),
        "detection_percent":
            len(valid) / len(method_df) * 100.0,
        "best_fit_radius_mm": radius_mm,
        "mean_radius_mm":
            valid["measured_radius_mm"].mean(),
        "radius_std_mm":
            valid["measured_radius_mm"].std(),
        "mean_absolute_radial_error_mm":
            valid["absolute_radial_error_mm"].mean(),
        "maximum_absolute_radial_error_mm":
            valid["absolute_radial_error_mm"].max(),
        "mean_radial_error_percent":
            valid["radial_error_percent"].mean(),
        "maximum_radial_error_percent":
            valid["radial_error_percent"].max(),
        "frame_step_std_mm":
            valid["frame_step_mm"].std()
    }

    return valid, summary


# ==========================================================
# OPEN VIDEO
# ==========================================================

video_path = Path(VIDEO_PATH)

if not video_path.exists():
    raise FileNotFoundError(f"Video not found:\n{video_path}")

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(f"Could not open video:\n{video_path}")

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    raise RuntimeError("Invalid video frame rate.")

print("\n--- Video information ---")
print(f"Video: {video_path.name}")
print(f"Resolution: {width} x {height}")
print(f"Frame rate: {fps:.3f} fps")
print(f"Frame count: {frame_count}")
print(f"Duration: {frame_count / fps:.3f} s")


# ==========================================================
# PREPARE UNDISTORTION
# ==========================================================

new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    (width, height),
    alpha=1,
    newImgSize=(width, height)
)


# ==========================================================
# SCALE FROM FIRST TWO SECONDS
# ==========================================================

pixels_per_mm, mm_per_pixel = obtain_scale(
    cap,
    fps
)

expected_marker_radius_px = (
    MARKER_DIAMETER_MM / 2.0
) * pixels_per_mm

max_hough_jump_px = (
    MAX_HOUGH_JUMP_MM * pixels_per_mm
)

print("\n--- Marker settings ---")
print(
    f"Expected marker radius: "
    f"{expected_marker_radius_px:.2f} px"
)


# ==========================================================
# TRACK FROM FIVE SECONDS
# ==========================================================

tracking_start_frame = int(TRACKING_START_S * fps)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    tracking_start_frame
)

overlay_path = OUTPUT_FOLDER / "three_method_overlay.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(overlay_path),
    fourcc,
    fps,
    (width, height)
)

rows = []
frame_index = tracking_start_frame

previous_hough_position = None

while True:
    ret, frame = cap.read()

    if not ret:
        break

    video_time_s = frame_index / fps
    tracking_time_s = video_time_s - TRACKING_START_S

    processed = undistort_frame(frame)
    mask = get_green_mask(processed)

    detections = {
        "centroid": detect_centroid(mask),
        "min_enclosing_circle":
            detect_min_enclosing_circle(mask),
        "hough": detect_hough_circle(
            mask,
            expected_marker_radius_px,
            previous_position=previous_hough_position,
            max_jump_px=max_hough_jump_px
        )
    }

    if detections["hough"] is not None:
        previous_hough_position = (
            detections["hough"]["x_px"],
            detections["hough"]["y_px"]
        )

    overlay = processed.copy()

    overlay_colours = {
        "centroid": (0, 0, 255),
        "min_enclosing_circle": (255, 0, 0),
        "hough": (0, 255, 0)
    }

    for method, detection in detections.items():
        if detection is None:
            rows.append({
                "method": method,
                "frame": frame_index,
                "video_time_s": video_time_s,
                "tracking_time_s": tracking_time_s,
                "x_px": np.nan,
                "y_px": np.nan,
                "radius_px": np.nan,
                "area_px2": np.nan
            })

            continue

        x_px = detection["x_px"]
        y_px = detection["y_px"]
        radius_px = detection["radius_px"]

        colour = overlay_colours[method]

        cv2.circle(
            overlay,
            (int(round(x_px)), int(round(y_px))),
            int(round(radius_px)),
            colour,
            2
        )

        cv2.circle(
            overlay,
            (int(round(x_px)), int(round(y_px))),
            3,
            colour,
            -1
        )

        rows.append({
            "method": method,
            "frame": frame_index,
            "video_time_s": video_time_s,
            "tracking_time_s": tracking_time_s,
            "x_px": x_px,
            "y_px": y_px,
            "radius_px": radius_px,
            "area_px2": detection["area_px2"]
        })

    cv2.putText(
        overlay,
        "Red: centroid | Blue: enclosing circle | Green: Hough",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    cv2.putText(
        overlay,
        f"Tracking time: {tracking_time_s:.2f} s",
        (20, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    out.write(overlay)

    frame_index += 1

cap.release()
out.release()


# ==========================================================
# ANALYSE EACH METHOD
# ==========================================================

all_data = pd.DataFrame(rows)

all_data.to_csv(
    OUTPUT_FOLDER / "all_method_raw_tracking.csv",
    index=False
)

processed_methods = {}
summary_rows = []

for method in METHODS:
    method_data = all_data[
        all_data["method"] == method
    ].copy()

    processed, summary = analyse_method(
        method_data,
        mm_per_pixel
    )

    if processed is None:
        print(f"{method}: too few detections")
        continue

    processed_methods[method] = processed
    summary_rows.append(summary)

    processed.to_csv(
        OUTPUT_FOLDER / f"{method}_processed_tracking.csv",
        index=False
    )

comparison = pd.DataFrame(summary_rows)

# Rank methods primarily by mean radial error.
comparison["accuracy_rank"] = comparison[
    "mean_absolute_radial_error_mm"
].rank(method="min")

comparison = comparison.sort_values(
    "accuracy_rank"
)

comparison.to_csv(
    OUTPUT_FOLDER / "tracking_method_comparison.csv",
    index=False
)

print("\n--- Method comparison ---")
print(
    comparison[
        [
            "method",
            "detection_percent",
            "best_fit_radius_mm",
            "radius_std_mm",
            "mean_absolute_radial_error_mm",
            "mean_radial_error_percent",
            "frame_step_std_mm",
            "accuracy_rank"
        ]
    ].to_string(index=False)
)


# ==========================================================
# PLOT 1: ALL TRAJECTORIES
# ==========================================================

plt.figure(figsize=(7, 7))

for method, data in processed_methods.items():
    plt.plot(
        data["x_mm"],
        data["z_mm"],
        ".",
        markersize=1.5,
        label=method.replace("_", " ")
    )

plt.xlabel("x position [mm]")
plt.ylabel("z position [mm]")
plt.title("Tracking-method trajectory comparison")
plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "all_method_trajectories_mm.png",
    dpi=300
)


# ==========================================================
# PLOT 2: RADIAL ERROR OVER TIME
# ==========================================================

plt.figure(figsize=(10, 5))

for method, data in processed_methods.items():
    plt.plot(
        data["tracking_time_s"],
        data["radial_error_mm"],
        label=method.replace("_", " "),
        linewidth=1
    )

plt.axhline(0.0, linestyle="--")

plt.xlabel("Tracking time [s]")
plt.ylabel("Signed radial error [mm]")
plt.title("Radial-error comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "all_method_radial_error.png",
    dpi=300
)


# ==========================================================
# PLOT 3: MEAN ERROR BAR CHART
# ==========================================================

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["method"].str.replace("_", " "),
    comparison["mean_absolute_radial_error_mm"]
)

plt.ylabel("Mean absolute radial error [mm]")
plt.xlabel("Tracking method")
plt.title("Mean radial-error comparison")
plt.grid(True, axis="y")
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "mean_radial_error_comparison.png",
    dpi=300
)


# ==========================================================
# PLOT 4: DETECTION RATE
# ==========================================================

plt.figure(figsize=(8, 5))

plt.bar(
    comparison["method"].str.replace("_", " "),
    comparison["detection_percent"]
)

plt.ylabel("Valid detections [%]")
plt.xlabel("Tracking method")
plt.title("Detection-rate comparison")
plt.ylim(0, 105)
plt.grid(True, axis="y")
plt.tight_layout()

plt.savefig(
    OUTPUT_FOLDER / "detection_rate_comparison.png",
    dpi=300
)

plt.show()


# ==========================================================
# FINAL OUTPUTS
# ==========================================================

print("\n--- Files saved ---")
print(
    f"Checkerboard image: "
    f"{OUTPUT_FOLDER / 'checkerboard_detected_corners.png'}"
)
print(
    f"Raw tracking: "
    f"{OUTPUT_FOLDER / 'all_method_raw_tracking.csv'}"
)
print(
    f"Comparison table: "
    f"{OUTPUT_FOLDER / 'tracking_method_comparison.csv'}"
)
print(f"Overlay video: {overlay_path}")
print(
    f"Trajectory plot: "
    f"{OUTPUT_FOLDER / 'all_method_trajectories_mm.png'}"
)
print(
    f"Error plot: "
    f"{OUTPUT_FOLDER / 'all_method_radial_error.png'}"
)
print(
    f"Mean-error chart: "
    f"{OUTPUT_FOLDER / 'mean_radial_error_comparison.png'}"
)
print(
    f"Detection-rate chart: "
    f"{OUTPUT_FOLDER / 'detection_rate_comparison.png'}"
)
print(f"\nAll outputs saved to:\n{OUTPUT_FOLDER.resolve()}")