import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# USER SETTINGS
# ==========================================================
VIDEO_PATH = r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260610_185333.mp4"

OUTPUT_FOLDER = Path("tracking_comparison")
OUTPUT_FOLDER.mkdir(exist_ok=True)

KNOWN_DOT_RADIUS_MM = 22.0
RULER_DISTANCE_MM = 100.0  # click 0 cm and 10 cm if possible

LOWER_GREEN = np.array([35, 80, 80])
UPPER_GREEN = np.array([85, 255, 255])

MIN_AREA = 30
MAX_AREA = 20000
MIN_REV_COMPLETENESS = 0.8

METHODS = ["centroid", "min_enclosing_circle", "hough"]


# ==========================================================
# MANUAL RULER SCALE
# ==========================================================
clicked_points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(clicked_points) < 2:
        clicked_points.append((x, y))
        print(f"Clicked point {len(clicked_points)}: x={x}, y={y}")


def get_ruler_scale(first_frame):
    cv2.namedWindow("Click ruler points", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("Click ruler points", mouse_callback)

    while True:
        shown = first_frame.copy()

        cv2.putText(
            shown,
            "Click 0 cm and 10 cm ruler marks, then press any key",
            (30, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        for p in clicked_points:
            cv2.circle(shown, p, 8, (0, 255, 0), -1)

        if len(clicked_points) == 2:
            cv2.line(shown, clicked_points[0], clicked_points[1], (0, 255, 0), 2)

        cv2.imshow("Click ruler points", shown)
        key = cv2.waitKey(20)

        if key != -1 and len(clicked_points) == 2:
            break

    cv2.destroyWindow("Click ruler points")

    p1 = np.array(clicked_points[0], dtype=float)
    p2 = np.array(clicked_points[1], dtype=float)

    ruler_px = np.linalg.norm(p2 - p1)
    pixels_per_mm = ruler_px / RULER_DISTANCE_MM
    mm_per_pixel = 1.0 / pixels_per_mm

    print("\n--- Ruler scale ---")
    print(f"Ruler distance: {RULER_DISTANCE_MM:.2f} mm")
    print(f"Pixel distance: {ruler_px:.2f} px")
    print(f"Scale: {pixels_per_mm:.4f} px/mm")
    print(f"Scale: {mm_per_pixel:.6f} mm/px")

    return pixels_per_mm, mm_per_pixel


# ==========================================================
# HELPERS
# ==========================================================
def fit_circle(x, y):
    A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
    b = x**2 + y**2
    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    xc = c[0]
    yc = c[1]
    r = np.sqrt(c[2] + xc**2 + yc**2)

    return xc, yc, r


def get_green_mask(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return mask


def detect_dot(frame, method):
    mask = get_green_mask(frame)

    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return np.nan, np.nan, np.nan, np.nan

    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    valid_contour = None
    area = np.nan

    for contour in contours:
        area_temp = cv2.contourArea(contour)

        if MIN_AREA <= area_temp <= MAX_AREA:
            valid_contour = contour
            area = area_temp
            break

    if valid_contour is None:
        return np.nan, np.nan, np.nan, np.nan

    if method == "centroid":
        M = cv2.moments(valid_contour)
        if M["m00"] == 0:
            return np.nan, np.nan, np.nan, area

        x = M["m10"] / M["m00"]
        y = M["m01"] / M["m00"]
        radius = np.sqrt(area / np.pi)

        return float(x), float(y), float(radius), float(area)

    if method == "min_enclosing_circle":
        (x, y), radius = cv2.minEnclosingCircle(valid_contour)
        return float(x), float(y), float(radius), float(area)

    if method == "hough":
        x, y, w, h = cv2.boundingRect(valid_contour)

        pad = 20
        x0 = max(x - pad, 0)
        y0 = max(y - pad, 0)
        x1 = min(x + w + pad, frame.shape[1])
        y1 = min(y + h + pad, frame.shape[0])

        roi_mask = mask[y0:y1, x0:x1]
        roi_blur = cv2.GaussianBlur(roi_mask, (9, 9), 2)

        circles = cv2.HoughCircles(
            roi_blur,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=20,
            param1=50,
            param2=10,
            minRadius=3,
            maxRadius=80
        )

        if circles is not None:
            circles = np.round(circles[0, :]).astype(float)

            # Use circle closest to contour centre
            M = cv2.moments(valid_contour)
            if M["m00"] != 0:
                cx = M["m10"] / M["m00"] - x0
                cy = M["m01"] / M["m00"] - y0

                distances = np.sqrt((circles[:, 0] - cx) ** 2 + (circles[:, 1] - cy) ** 2)
                best = circles[np.argmin(distances)]
            else:
                best = circles[0]

            hough_x = best[0] + x0
            hough_y = best[1] + y0
            hough_r = best[2]

            return float(hough_x), float(hough_y), float(hough_r), float(area)

        # fallback if Hough misses frame
        return np.nan, np.nan, np.nan, area

    raise ValueError(f"Unknown method: {method}")


# ==========================================================
# PROCESS ONE METHOD
# ==========================================================
def process_method(method, mm_per_pixel):
    method_folder = OUTPUT_FOLDER / method
    method_folder.mkdir(exist_ok=True)

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    overlay_path = method_folder / f"{method}_tracking_overlay.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(overlay_path), fourcc, fps, (width, height))

    results = []
    frame_index = 0

    print(f"\nProcessing method: {method}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_s = frame_index / fps

        x_px, y_px, marker_radius_px, area_px2 = detect_dot(frame, method)

        overlay = frame.copy()

        if not np.isnan(x_px):
            cv2.circle(
                overlay,
                (int(x_px), int(y_px)),
                int(marker_radius_px) if not np.isnan(marker_radius_px) else 8,
                (0, 255, 0),
                2
            )
            cv2.circle(overlay, (int(x_px), int(y_px)), 4, (0, 0, 255), -1)
            status = f"{method}: x={x_px:.1f}, y={y_px:.1f}"
            colour = (0, 255, 0)
        else:
            status = f"{method}: no detection"
            colour = (0, 0, 255)

        cv2.putText(
            overlay,
            f"Frame {frame_index} | {status}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            colour,
            2
        )

        out.write(overlay)

        results.append({
            "method": method,
            "frame": frame_index,
            "time_s": time_s,
            "x_px": x_px,
            "y_px": y_px,
            "marker_radius_px": marker_radius_px,
            "marker_radius_mm": marker_radius_px * mm_per_pixel if not np.isnan(marker_radius_px) else np.nan,
            "area_px2": area_px2
        })

        frame_index += 1

    cap.release()
    out.release()

    df = pd.DataFrame(results)
    df.to_csv(method_folder / f"{method}_raw_tracking.csv", index=False)

    valid = df.dropna(subset=["x_px", "y_px"]).copy()

    if len(valid) < 50:
        print(f"WARNING: too few valid points for {method}")
        return valid, None

    xc_px, yc_px, motion_radius_px = fit_circle(
        valid["x_px"].to_numpy(),
        valid["y_px"].to_numpy()
    )

    valid["x_mm"] = (valid["x_px"] - xc_px) * mm_per_pixel
    valid["y_mm"] = -(valid["y_px"] - yc_px) * mm_per_pixel

    valid["motion_radius_px"] = np.sqrt(
        (valid["x_px"] - xc_px) ** 2 + (valid["y_px"] - yc_px) ** 2
    )
    valid["motion_radius_mm"] = valid["motion_radius_px"] * mm_per_pixel

    valid["theta_rad"] = np.arctan2(valid["y_mm"], valid["x_mm"])
    valid["theta_unwrapped_rad"] = np.unwrap(valid["theta_rad"].to_numpy())
    valid["theta_unwrapped_deg"] = np.degrees(valid["theta_unwrapped_rad"])

    theta0 = valid["theta_unwrapped_deg"].iloc[0]
    valid["rev_float"] = np.abs(valid["theta_unwrapped_deg"] - theta0) / 360.0
    valid["rev_index"] = np.floor(valid["rev_float"]).astype(int) + 1

    valid.to_csv(method_folder / f"{method}_tracking_with_mm.csv", index=False)

    rev_summaries = []

    for rev, group in valid.groupby("rev_index"):
        rev_span = group["rev_float"].max() - group["rev_float"].min()

        if rev_span < MIN_REV_COMPLETENESS:
            continue

        duration = group["time_s"].max() - group["time_s"].min()

        rev_summaries.append({
            "method": method,
            "rev": rev,
            "frames": len(group),
            "duration_s": duration,
            "rpm": 60.0 / duration if duration > 0 else np.nan,
            "mean_motion_radius_mm": group["motion_radius_mm"].mean(),
            "std_motion_radius_mm": group["motion_radius_mm"].std(),
            "radius_error_mm": group["motion_radius_mm"].mean() - KNOWN_DOT_RADIUS_MM,
            "radius_error_percent": (group["motion_radius_mm"].mean() - KNOWN_DOT_RADIUS_MM) / KNOWN_DOT_RADIUS_MM * 100,
            "mean_marker_radius_mm": group["marker_radius_mm"].mean(),
            "std_marker_radius_mm": group["marker_radius_mm"].std(),
            "mean_area_px2": group["area_px2"].mean(),
            "std_area_px2": group["area_px2"].std()
        })

    rev_summary = pd.DataFrame(rev_summaries)
    rev_summary.to_csv(method_folder / f"{method}_revolution_summary.csv", index=False)

    summary = {
        "method": method,
        "valid_frames": len(valid),
        "total_frames": len(df),
        "valid_percent": len(valid) / len(df) * 100,
        "num_complete_revs": len(rev_summary),
        "fitted_centre_x_px": xc_px,
        "fitted_centre_y_px": yc_px,
        "fitted_motion_radius_px": motion_radius_px,
        "mean_motion_radius_mm": valid["motion_radius_mm"].mean(),
        "std_motion_radius_mm": valid["motion_radius_mm"].std(),
        "radius_error_mm": valid["motion_radius_mm"].mean() - KNOWN_DOT_RADIUS_MM,
        "radius_error_percent": (valid["motion_radius_mm"].mean() - KNOWN_DOT_RADIUS_MM) / KNOWN_DOT_RADIUS_MM * 100,
        "between_rev_std_mean_radius_mm": rev_summary["mean_motion_radius_mm"].std() if len(rev_summary) > 1 else np.nan,
        "mean_marker_radius_mm": valid["marker_radius_mm"].mean(),
        "std_marker_radius_mm": valid["marker_radius_mm"].std(),
        "mean_area_px2": valid["area_px2"].mean(),
        "std_area_px2": valid["area_px2"].std(),
        "overlay_video": str(overlay_path)
    }

    # Plots
    plt.figure()
    plt.plot(valid["x_mm"], valid["y_mm"], ".", markersize=1)
    plt.axis("equal")
    plt.xlabel("x position [mm]")
    plt.ylabel("y position [mm]")
    plt.title(f"{method}: tracked trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(method_folder / f"{method}_trajectory_mm.png", dpi=300)
    plt.close()

    plt.figure()
    plt.plot(valid["time_s"], valid["motion_radius_mm"], linewidth=1)
    plt.axhline(KNOWN_DOT_RADIUS_MM, linestyle="--", label="Known radius = 22 mm")
    plt.xlabel("Time [s]")
    plt.ylabel("Measured radius [mm]")
    plt.title(f"{method}: radius over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(method_folder / f"{method}_radius_over_time.png", dpi=300)
    plt.close()

    if len(rev_summary) > 0:
        plt.figure()
        plt.errorbar(
            rev_summary["rev"],
            rev_summary["mean_motion_radius_mm"],
            yerr=rev_summary["std_motion_radius_mm"],
            fmt="o",
            capsize=4
        )
        plt.axhline(KNOWN_DOT_RADIUS_MM, linestyle="--", label="Known radius = 22 mm")
        plt.xlabel("Revolution number")
        plt.ylabel("Mean radius [mm]")
        plt.title(f"{method}: mean radius per revolution")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(method_folder / f"{method}_mean_radius_per_revolution.png", dpi=300)
        plt.close()

    return valid, summary


# ==========================================================
# MAIN
# ==========================================================
video_path = Path(VIDEO_PATH)

if not video_path.exists():
    raise FileNotFoundError(f"Video not found:\n{video_path}")

cap = cv2.VideoCapture(str(video_path))
ret, first_frame = cap.read()
cap.release()

if not ret:
    raise RuntimeError("Could not read first frame.")

pixels_per_mm, mm_per_pixel = get_ruler_scale(first_frame)

all_valid = []
summaries = []

for method in METHODS:
    valid, summary = process_method(method, mm_per_pixel)

    if summary is not None:
        all_valid.append(valid)
        summaries.append(summary)

comparison_df = pd.DataFrame(summaries)
comparison_df.to_csv(OUTPUT_FOLDER / "method_comparison_summary.csv", index=False)

print("\n" + "=" * 70)
print("METHOD COMPARISON SUMMARY")
print("=" * 70)

columns_to_print = [
    "method",
    "valid_percent",
    "num_complete_revs",
    "mean_motion_radius_mm",
    "std_motion_radius_mm",
    "radius_error_mm",
    "radius_error_percent",
    "between_rev_std_mean_radius_mm",
    "mean_marker_radius_mm",
    "std_marker_radius_mm"
]

print(comparison_df[columns_to_print].to_string(index=False))

# Combined trajectory plot
plt.figure()
for valid in all_valid:
    method = valid["method"].iloc[0]
    plt.plot(valid["x_mm"], valid["y_mm"], ".", markersize=1, label=method)

plt.axis("equal")
plt.xlabel("x position [mm]")
plt.ylabel("y position [mm]")
plt.title("Tracking method comparison: trajectory")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "combined_trajectory_comparison.png", dpi=300)

# Combined radius over time
plt.figure()
for valid in all_valid:
    method = valid["method"].iloc[0]
    plt.plot(valid["time_s"], valid["motion_radius_mm"], linewidth=1, label=method)

plt.axhline(KNOWN_DOT_RADIUS_MM, linestyle="--", label="Known radius = 22 mm")
plt.xlabel("Time [s]")
plt.ylabel("Measured radius [mm]")
plt.title("Tracking method comparison: radius over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "combined_radius_over_time.png", dpi=300)

# Bar chart: radius std
plt.figure()
plt.bar(comparison_df["method"], comparison_df["std_motion_radius_mm"])
plt.xlabel("Tracking method")
plt.ylabel("Radius standard deviation [mm]")
plt.title("Tracking jitter comparison")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "method_radius_std_comparison.png", dpi=300)

# Bar chart: radius error %
plt.figure()
plt.bar(comparison_df["method"], comparison_df["radius_error_percent"])
plt.axhline(0, linestyle="--")
plt.xlabel("Tracking method")
plt.ylabel("Radius error [%]")
plt.title("Tracking accuracy comparison")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "method_radius_error_percent_comparison.png", dpi=300)

plt.show()

print(f"\nAll outputs saved to:\n{OUTPUT_FOLDER.resolve()}")