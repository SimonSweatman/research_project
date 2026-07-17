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

OUTPUT_FOLDER = Path(r"C:\temp\green_dot_checkerboard_scale")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

# 10 x 8 squares = 9 x 7 internal corners
CHECKERBOARD_CORNERS = (9, 7)
SQUARE_SIZE_MM = 10.0

# Recording sequence
SCALE_SEARCH_START_S = 0.0
SCALE_SEARCH_END_S = 2.0
TRACKING_START_S = 5.0

# Only test every few frames during checkerboard detection.
# At 30 fps, 3 means approximately 10 checked frames per second.
CHECKERBOARD_FRAME_STEP = 3

# Lime-green marker HSV range
LOWER_GREEN = np.array([35, 80, 80])
UPPER_GREEN = np.array([85, 255, 255])

MIN_MARKER_AREA = 30
MAX_MARKER_AREA = 20000

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
    """Remove lens distortion from one video frame."""
    return cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

# ==========================================================
# CHECKERBOARD SCALE DETECTION
# ==========================================================

def detect_checkerboard(frame):
    """
    Detect the 9 x 7 internal checkerboard corners.

    Returns:
        found: True/False
        corners: detected sub-pixel corner positions
        detector_name: detector used
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Try the stronger detector first.
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

    # Fall back to traditional detection.
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


def calculate_scale_from_corners(corners):
    """
    Calculate pixels/mm from a detected 9 x 7 internal-corner grid.

    The scale is estimated using:
    - every complete horizontal row;
    - every complete vertical column.

    A horizontal row spans:
        8 gaps x 10 mm = 80 mm.

    A vertical column spans:
        6 gaps x 10 mm = 60 mm.
    """
    columns, rows = CHECKERBOARD_CORNERS

    grid = corners.reshape(rows, columns, 2)

    scale_estimates_px_per_mm = []

    # Complete horizontal row measurements
    horizontal_distance_mm = (columns - 1) * SQUARE_SIZE_MM

    for row in range(rows):
        first_corner = grid[row, 0]
        last_corner = grid[row, columns - 1]

        pixel_distance = np.linalg.norm(last_corner - first_corner)

        scale_estimates_px_per_mm.append(
            pixel_distance / horizontal_distance_mm
        )

    # Complete vertical column measurements
    vertical_distance_mm = (rows - 1) * SQUARE_SIZE_MM

    for column in range(columns):
        first_corner = grid[0, column]
        last_corner = grid[rows - 1, column]

        pixel_distance = np.linalg.norm(last_corner - first_corner)

        scale_estimates_px_per_mm.append(
            pixel_distance / vertical_distance_mm
        )

    scale_estimates_px_per_mm = np.asarray(
        scale_estimates_px_per_mm,
        dtype=float
    )

    # Median is resistant to one poor row/column measurement.
    pixels_per_mm = np.median(scale_estimates_px_per_mm)
    mm_per_pixel = 1.0 / pixels_per_mm

    return (
        pixels_per_mm,
        mm_per_pixel,
        scale_estimates_px_per_mm
    )


def obtain_video_scale(cap, fps):
    """
    Search the first two seconds for the checkerboard.

    The scale from every successful checkerboard frame is
    collected, then the median is used as the final scale.
    """
    start_frame = int(SCALE_SEARCH_START_S * fps)
    end_frame = int(SCALE_SEARCH_END_S * fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    successful_scales = []
    successful_images = []
    detection_records = []

    for frame_index in range(start_frame, end_frame):
        ret, frame = cap.read()

        if not ret:
            break

        if (frame_index - start_frame) % CHECKERBOARD_FRAME_STEP != 0:
            continue

        processed = undistort_frame(frame)

        found, corners, detector_name = detect_checkerboard(processed)

        if not found:
            continue

        (
            pixels_per_mm,
            mm_per_pixel,
            individual_scale_estimates
        ) = calculate_scale_from_corners(corners)

        successful_scales.append(pixels_per_mm)

        detection_records.append({
            "frame": frame_index,
            "time_s": frame_index / fps,
            "detector": detector_name,
            "pixels_per_mm": pixels_per_mm,
            "mm_per_pixel": mm_per_pixel,
            "individual_scale_std_px_per_mm":
                np.std(individual_scale_estimates, ddof=1)
        })

        detection_image = processed.copy()

        cv2.drawChessboardCorners(
            detection_image,
            CHECKERBOARD_CORNERS,
            corners,
            True
        )

        cv2.putText(
            detection_image,
            f"Scale = {pixels_per_mm:.5f} px/mm",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2
        )

        successful_images.append(
            (frame_index, detection_image)
        )

    if not successful_scales:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first_frame = cap.read()

        if ret:
            failed_image = undistort_frame(first_frame)

            cv2.imwrite(
                str(OUTPUT_FOLDER / "checkerboard_detection_failed.png"),
                failed_image
            )

        raise RuntimeError(
            "The 9 x 7 internal-corner checkerboard was not detected "
            f"between {SCALE_SEARCH_START_S:.1f} and "
            f"{SCALE_SEARCH_END_S:.1f} seconds.\n\n"
            "Keep the complete checkerboard still and fully visible "
            "for the first two seconds."
        )

    successful_scales = np.asarray(successful_scales)

    final_pixels_per_mm = np.median(successful_scales)
    final_mm_per_pixel = 1.0 / final_pixels_per_mm

    scale_std_px_per_mm = np.std(
        successful_scales,
        ddof=1
    ) if len(successful_scales) > 1 else 0.0

    scale_cv_percent = (
        scale_std_px_per_mm
        / final_pixels_per_mm
        * 100.0
    )

    # Save the detection closest to the final median scale.
    best_index = int(
        np.argmin(
            np.abs(successful_scales - final_pixels_per_mm)
        )
    )

    best_frame_index, best_image = successful_images[best_index]

    cv2.putText(
        best_image,
        f"Final median scale = {final_pixels_per_mm:.5f} px/mm",
        (30, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )

    cv2.imwrite(
        str(OUTPUT_FOLDER / "checkerboard_detected_corners.png"),
        best_image
    )

    pd.DataFrame(detection_records).to_csv(
        OUTPUT_FOLDER / "checkerboard_scale_detections.csv",
        index=False
    )

    pd.DataFrame([{
        "successful_checkerboard_frames": len(successful_scales),
        "best_detection_frame": best_frame_index,
        "pixels_per_mm": final_pixels_per_mm,
        "mm_per_pixel": final_mm_per_pixel,
        "scale_std_px_per_mm": scale_std_px_per_mm,
        "scale_coefficient_of_variation_percent": scale_cv_percent
    }]).to_csv(
        OUTPUT_FOLDER / "checkerboard_scale_summary.csv",
        index=False
    )

    print("\n--- Checkerboard scale ---")
    print(
        f"Successful checkerboard detections: "
        f"{len(successful_scales)}"
    )
    print(
        f"Final scale: "
        f"{final_pixels_per_mm:.6f} px/mm"
    )
    print(
        f"Final scale: "
        f"{final_mm_per_pixel:.6f} mm/px"
    )
    print(
        f"Scale variation between detected frames: "
        f"{scale_cv_percent:.4f}%"
    )
    print(
        "Detection image saved to: "
        f"{OUTPUT_FOLDER / 'checkerboard_detected_corners.png'}"
    )

    return final_pixels_per_mm, final_mm_per_pixel


# ==========================================================
# GREEN MARKER TRACKING
# ==========================================================

def detect_green_marker(frame):
    """Find the centroid of the largest valid lime-green blob."""
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

        if not MIN_MARKER_AREA <= area <= MAX_MARKER_AREA:
            continue

        moments = cv2.moments(contour)

        if moments["m00"] == 0:
            continue

        x_px = moments["m10"] / moments["m00"]
        y_px = moments["m01"] / moments["m00"]

        radius_px = np.sqrt(area / np.pi)

        return {
            "x_px": float(x_px),
            "y_px": float(y_px),
            "radius_px": float(radius_px),
            "area_px2": float(area)
        }

    return None


# ==========================================================
# GEOMETRY FITTING
# ==========================================================

def fit_ellipse(x, y):
    points = np.column_stack((x, y)).astype(np.float32)

    if len(points) < 5:
        raise RuntimeError(
            "At least five tracked points are required."
        )

    ellipse = cv2.fitEllipse(points)

    (centre_x, centre_y), (axis_1, axis_2), angle = ellipse

    semi_major = max(axis_1, axis_2) / 2.0
    semi_minor = min(axis_1, axis_2) / 2.0

    circularity_ratio = semi_minor / semi_major

    eccentricity = np.sqrt(
        max(
            0.0,
            1.0 - (
                semi_minor**2 / semi_major**2
            )
        )
    )

    return {
        "centre_x_px": centre_x,
        "centre_y_px": centre_y,
        "semi_major_px": semi_major,
        "semi_minor_px": semi_minor,
        "angle_deg": angle,
        "circularity_ratio": circularity_ratio,
        "eccentricity": eccentricity
    }


def fit_circle(x, z):
    """Least-squares perfect-circle fit in millimetres."""
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


# ==========================================================
# OPEN VIDEO
# ==========================================================

video_path = Path(VIDEO_PATH)

if not video_path.exists():
    raise FileNotFoundError(
        f"Video not found:\n{video_path}"
    )

cap = cv2.VideoCapture(str(video_path))

if not cap.isOpened():
    raise RuntimeError(
        f"Could not open video:\n{video_path}"
    )

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

image_size = (width, height)

new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
    camera_matrix,
    dist_coeffs,
    image_size,
    alpha=1,
    newImgSize=image_size
)


# ==========================================================
# CALCULATE SCALE FROM FIRST TWO SECONDS
# ==========================================================

pixels_per_mm, mm_per_pixel = obtain_video_scale(
    cap,
    fps
)


# ==========================================================
# TRACK GREEN MARKER FROM FIVE SECONDS
# ==========================================================

tracking_start_frame = int(TRACKING_START_S * fps)

cap.set(
    cv2.CAP_PROP_POS_FRAMES,
    tracking_start_frame
)

overlay_path = OUTPUT_FOLDER / "green_marker_tracking_overlay.mp4"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out = cv2.VideoWriter(
    str(overlay_path),
    fourcc,
    fps,
    (width, height)
)

results = []
frame_index = tracking_start_frame

while True:
    ret, frame = cap.read()

    if not ret:
        break

    time_s = frame_index / fps
    tracking_time_s = time_s - TRACKING_START_S

    processed = undistort_frame(frame)
    marker = detect_green_marker(processed)

    overlay = processed.copy()

    if marker is not None:
        x_px = marker["x_px"]
        y_px = marker["y_px"]
        marker_radius_px = marker["radius_px"]
        area_px2 = marker["area_px2"]

        cv2.circle(
            overlay,
            (int(round(x_px)), int(round(y_px))),
            int(round(marker_radius_px)),
            (0, 255, 0),
            2
        )

        cv2.circle(
            overlay,
            (int(round(x_px)), int(round(y_px))),
            4,
            (0, 0, 255),
            -1
        )

        status = (
            f"x={x_px:.1f}, "
            f"y={y_px:.1f}, "
            f"r={marker_radius_px:.1f}px"
        )

        text_colour = (0, 255, 0)

    else:
        x_px = np.nan
        y_px = np.nan
        marker_radius_px = np.nan
        area_px2 = np.nan

        status = "marker not found"
        text_colour = (0, 0, 255)

    cv2.putText(
        overlay,
        f"Tracking t={tracking_time_s:.2f}s | {status}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        text_colour,
        2
    )

    out.write(overlay)

    results.append({
        "frame": frame_index,
        "video_time_s": time_s,
        "tracking_time_s": tracking_time_s,
        "x_px": x_px,
        "y_px": y_px,
        "dot_radius_px": marker_radius_px,
        "dot_radius_mm": (
            marker_radius_px * mm_per_pixel
            if not np.isnan(marker_radius_px)
            else np.nan
        ),
        "area_px2": area_px2
    })

    frame_index += 1

cap.release()
out.release()


# ==========================================================
# SAVE TRACKING DATA
# ==========================================================

raw_data = pd.DataFrame(results)

raw_data_path = OUTPUT_FOLDER / "green_marker_raw_tracking.csv"
raw_data.to_csv(raw_data_path, index=False)

valid = raw_data.dropna(
    subset=["x_px", "y_px"]
).copy()

valid_percent = (
    len(valid) / len(raw_data) * 100.0
    if len(raw_data) > 0
    else 0.0
)

print("\n--- Marker tracking ---")
print(
    f"Valid detections: {len(valid)} / {len(raw_data)} "
    f"({valid_percent:.2f}%)"
)

if len(valid) < 20:
    raise RuntimeError(
        "Too few valid green-marker detections."
    )


# ==========================================================
# ELLIPSE AND CIRCLE ANALYSIS
# ==========================================================

ellipse = fit_ellipse(
    valid["x_px"].to_numpy(),
    valid["y_px"].to_numpy()
)

centre_x_px = ellipse["centre_x_px"]
centre_y_px = ellipse["centre_y_px"]

valid["x_mm"] = (
    valid["x_px"] - centre_x_px
) * mm_per_pixel

valid["z_mm"] = -(
    valid["y_px"] - centre_y_px
) * mm_per_pixel

valid["radius_from_ellipse_centre_mm"] = np.sqrt(
    valid["x_mm"]**2
    + valid["z_mm"]**2
)

circle_centre_x_mm, circle_centre_z_mm, circle_radius_mm = (
    fit_circle(
        valid["x_mm"].to_numpy(),
        valid["z_mm"].to_numpy()
    )
)

valid["radius_from_circle_centre_mm"] = np.sqrt(
    (
        valid["x_mm"] - circle_centre_x_mm
    )**2
    + (
        valid["z_mm"] - circle_centre_z_mm
    )**2
)

valid["radial_error_mm"] = (
    valid["radius_from_circle_centre_mm"]
    - circle_radius_mm
)

valid["absolute_radial_error_mm"] = (
    valid["radial_error_mm"].abs()
)

valid["radial_error_percent"] = (
    valid["absolute_radial_error_mm"]
    / circle_radius_mm
    * 100.0
)

mean_absolute_error_mm = (
    valid["absolute_radial_error_mm"].mean()
)

maximum_absolute_error_mm = (
    valid["absolute_radial_error_mm"].max()
)

mean_error_percent = (
    valid["radial_error_percent"].mean()
)

maximum_error_percent = (
    valid["radial_error_percent"].max()
)

processed_data_path = (
    OUTPUT_FOLDER
    / "green_marker_processed_tracking.csv"
)

valid.to_csv(
    processed_data_path,
    index=False
)


# ==========================================================
# SAVE SUMMARY
# ==========================================================

summary = pd.DataFrame([{
    "pixels_per_mm": pixels_per_mm,
    "mm_per_pixel": mm_per_pixel,

    "valid_frames": len(valid),
    "total_tracking_frames": len(raw_data),
    "valid_percent": valid_percent,

    "ellipse_semi_major_mm":
        ellipse["semi_major_px"] * mm_per_pixel,

    "ellipse_semi_minor_mm":
        ellipse["semi_minor_px"] * mm_per_pixel,

    "ellipse_circularity_ratio":
        ellipse["circularity_ratio"],

    "ellipse_eccentricity":
        ellipse["eccentricity"],

    "best_fit_circle_radius_mm":
        circle_radius_mm,

    "mean_absolute_radial_error_mm":
        mean_absolute_error_mm,

    "maximum_absolute_radial_error_mm":
        maximum_absolute_error_mm,

    "mean_radial_error_percent":
        mean_error_percent,

    "maximum_radial_error_percent":
        maximum_error_percent,

    "mean_detected_marker_radius_mm":
        valid["dot_radius_mm"].mean(),

    "std_detected_marker_radius_mm":
        valid["dot_radius_mm"].std()
}])

summary_path = OUTPUT_FOLDER / "tracking_summary.csv"
summary.to_csv(summary_path, index=False)

print("\n--- Circle comparison ---")
print(f"Fitted radius: {circle_radius_mm:.3f} mm")
print(
    f"Mean absolute radial error: "
    f"{mean_absolute_error_mm:.3f} mm "
    f"({mean_error_percent:.3f}%)"
)
print(
    f"Maximum absolute radial error: "
    f"{maximum_absolute_error_mm:.3f} mm "
    f"({maximum_error_percent:.3f}%)"
)


# ==========================================================
# PLOT MEASURED PATH VS PERFECT CIRCLE
# ==========================================================

theta = np.linspace(0, 2 * np.pi, 500)

circle_x = (
    circle_centre_x_mm
    + circle_radius_mm * np.cos(theta)
)

circle_z = (
    circle_centre_z_mm
    + circle_radius_mm * np.sin(theta)
)

plt.figure(figsize=(6, 6))

plt.plot(
    valid["x_mm"],
    valid["z_mm"],
    ".",
    markersize=2,
    label="Measured marker path"
)

plt.plot(
    circle_x,
    circle_z,
    "--",
    linewidth=2,
    label="Best-fit circle"
)

plt.scatter(
    [circle_centre_x_mm],
    [circle_centre_z_mm],
    marker="x",
    s=80,
    label="Circle centre"
)

plt.xlabel("x position [mm]")
plt.ylabel("z position [mm]")

plt.title(
    "Measured marker path vs best-fit circle\n"
    f"Radius = {circle_radius_mm:.2f} mm, "
    f"mean error = {mean_absolute_error_mm:.2f} mm "
    f"({mean_error_percent:.2f}%)"
)

plt.axis("equal")
plt.grid(True)
plt.legend()
plt.tight_layout()

circle_plot_path = (
    OUTPUT_FOLDER
    / "measured_path_vs_best_fit_circle.png"
)

plt.savefig(circle_plot_path, dpi=300)


# ==========================================================
# PLOT RADIAL ERROR
# ==========================================================

plt.figure(figsize=(8, 4))

plt.plot(
    valid["tracking_time_s"],
    valid["radial_error_mm"]
)

plt.axhline(0.0, linestyle="--")

plt.xlabel("Tracking time [s]")
plt.ylabel("Signed radial error [mm]")
plt.title("Radial deviation from best-fit circle")
plt.grid(True)
plt.tight_layout()

error_plot_path = (
    OUTPUT_FOLDER
    / "radial_error_over_time.png"
)

plt.savefig(error_plot_path, dpi=300)

plt.show()


# ==========================================================
# FINAL OUTPUTS
# ==========================================================

print("\n--- Files saved ---")
print(
    f"Checkerboard detection image: "
    f"{OUTPUT_FOLDER / 'checkerboard_detected_corners.png'}"
)
print(
    f"Checkerboard scale details: "
    f"{OUTPUT_FOLDER / 'checkerboard_scale_detections.csv'}"
)
print(
    f"Checkerboard scale summary: "
    f"{OUTPUT_FOLDER / 'checkerboard_scale_summary.csv'}"
)
print(f"Overlay video: {overlay_path}")
print(f"Raw tracking data: {raw_data_path}")
print(f"Processed tracking data: {processed_data_path}")
print(f"Tracking summary: {summary_path}")
print(f"Circle comparison plot: {circle_plot_path}")
print(f"Radial error plot: {error_plot_path}")
print(f"\nAll outputs saved to:\n{OUTPUT_FOLDER.resolve()}")