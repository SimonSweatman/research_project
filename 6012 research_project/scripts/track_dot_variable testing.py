import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# USER SETTINGS
# ==========================================================
VIDEO_PATHS = [
    r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260609_152051.mp4",
    r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260609_140511.mp4",
    r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260609_152034.mp4",
    r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260609_135848.mp4",
    r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260609_155505.mp4",
]

OUTPUT_FOLDER = Path("calibration_outputs")
OUTPUT_FOLDER.mkdir(exist_ok=True)

KNOWN_RADIUS_MM = 22.0

# Lime green HSV range
LOWER_GREEN = np.array([35, 80, 80])
UPPER_GREEN = np.array([85, 255, 255])

MIN_AREA = 30
MAX_AREA = 20000


# ==========================================================
# CIRCLE FIT FUNCTION
# ==========================================================
def fit_circle(x, y):
    A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
    b = x**2 + y**2

    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    xc = c[0]
    yc = c[1]
    r = np.sqrt(c[2] + xc**2 + yc**2)

    return xc, yc, r


# ==========================================================
# VIDEO PROCESSING FUNCTION
# ==========================================================
def process_video(video_path, video_number):
    video_path = Path(video_path)

    if not video_path.exists():
        raise FileNotFoundError(f"Video not found:\n{video_path}")

    cap = cv2.VideoCapture(str(video_path))

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print("\n" + "=" * 60)
    print(f"Processing video {video_number}: {video_path.name}")
    print(f"Resolution: {width} x {height}")
    print(f"FPS: {fps}")
    print(f"Frames: {frame_count}")

    results = []
    frame_index = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        time_s = frame_index / fps

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        x_px = np.nan
        y_px = np.nan
        area = np.nan

        if contours:
            contours = sorted(contours, key=cv2.contourArea, reverse=True)

            for contour in contours:
                area_temp = cv2.contourArea(contour)

                if MIN_AREA <= area_temp <= MAX_AREA:
                    M = cv2.moments(contour)

                    if M["m00"] != 0:
                        x_px = M["m10"] / M["m00"]
                        y_px = M["m01"] / M["m00"]
                        area = area_temp
                        break

        results.append({
            "video": video_number,
            "source_file": video_path.name,
            "frame": frame_index,
            "time_s": time_s,
            "x_px": x_px,
            "y_px": y_px,
            "area_px2": area
        })

        frame_index += 1

    cap.release()

    df = pd.DataFrame(results)

    raw_csv = OUTPUT_FOLDER / f"video_{video_number}_raw_tracking.csv"
    df.to_csv(raw_csv, index=False)

    valid = df.dropna(subset=["x_px", "y_px"]).copy()

    print(f"Valid tracked frames: {len(valid)} / {len(df)}")

    if len(valid) < 10:
        raise RuntimeError(
            f"Too few valid tracked points in video {video_number}. "
            "Adjust HSV range, lighting, or area limits."
        )

    x = valid["x_px"].to_numpy()
    y = valid["y_px"].to_numpy()

    xc, yc, r_px = fit_circle(x, y)

    pixels_per_mm = r_px / KNOWN_RADIUS_MM
    mm_per_pixel = 1 / pixels_per_mm

    valid["x_mm"] = (valid["x_px"] - xc) * mm_per_pixel
    valid["y_mm"] = -(valid["y_px"] - yc) * mm_per_pixel

    valid["radius_px"] = np.sqrt((valid["x_px"] - xc)**2 + (valid["y_px"] - yc)**2)
    valid["radius_mm"] = valid["radius_px"] * mm_per_pixel

    mean_radius_mm = valid["radius_mm"].mean()
    std_radius_mm = valid["radius_mm"].std()
    radius_error_mm = mean_radius_mm - KNOWN_RADIUS_MM
    radius_error_percent = radius_error_mm / KNOWN_RADIUS_MM * 100

    processed_csv = OUTPUT_FOLDER / f"video_{video_number}_tracking_with_mm.csv"
    valid.to_csv(processed_csv, index=False)

    print("--- Calibration result ---")
    print(f"Fitted centre: x={xc:.2f} px, y={yc:.2f} px")
    print(f"Fitted radius: {r_px:.2f} px")
    print(f"Scale: {pixels_per_mm:.2f} px/mm")
    print(f"Mean measured radius: {mean_radius_mm:.2f} mm")
    print(f"Radius error: {radius_error_mm:.2f} mm ({radius_error_percent:.2f}%)")
    print(f"Radius standard deviation: {std_radius_mm:.2f} mm")

    # Pixel trajectory plot
    plt.figure()
    plt.plot(valid["x_px"], valid["y_px"], ".", markersize=2)
    plt.scatter([xc], [yc], marker="x", s=80)
    plt.gca().invert_yaxis()
    plt.axis("equal")
    plt.xlabel("x position [pixels]")
    plt.ylabel("y position [pixels]")
    plt.title(f"Video {video_number}: tracked lime green dot trajectory")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"video_{video_number}_trajectory_pixels.png", dpi=300)
    plt.close()

    # mm trajectory plot
    plt.figure()
    plt.plot(valid["x_mm"], valid["y_mm"], ".", markersize=2)
    plt.axis("equal")
    plt.xlabel("x position [mm]")
    plt.ylabel("y position [mm]")
    plt.title(f"Video {video_number}: tracked dot trajectory converted to mm")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"video_{video_number}_trajectory_mm.png", dpi=300)
    plt.close()

    # radius over time plot
    plt.figure()
    plt.plot(valid["time_s"], valid["radius_mm"])
    plt.axhline(KNOWN_RADIUS_MM, linestyle="--", label="Known radius = 22 mm")
    plt.xlabel("Time [s]")
    plt.ylabel("Measured radius [mm]")
    plt.title(f"Video {video_number}: measured radius over time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUTPUT_FOLDER / f"video_{video_number}_radius_over_time.png", dpi=300)
    plt.close()

    summary = {
        "video": video_number,
        "source_file": video_path.name,
        "fps": fps,
        "frame_count": frame_count,
        "valid_frames": len(valid),
        "valid_percent": len(valid) / len(df) * 100,
        "xc_px": xc,
        "yc_px": yc,
        "fitted_radius_px": r_px,
        "pixels_per_mm": pixels_per_mm,
        "mean_radius_mm": mean_radius_mm,
        "std_radius_mm": std_radius_mm,
        "radius_error_mm": radius_error_mm,
        "radius_error_percent": radius_error_percent
    }

    return valid, summary


# ==========================================================
# RUN ALL VIDEOS
# ==========================================================
all_valid = []
summaries = []

for i, path in enumerate(VIDEO_PATHS, start=1):
    valid, summary = process_video(path, i)
    all_valid.append(valid)
    summaries.append(summary)

summary_df = pd.DataFrame(summaries)
summary_df.to_csv(OUTPUT_FOLDER / "calibration_summary.csv", index=False)

combined_df = pd.concat(all_valid, ignore_index=True)
combined_df.to_csv(OUTPUT_FOLDER / "all_videos_tracking_with_mm.csv", index=False)

print("\n" + "=" * 60)
print("Saved summary:")
print(OUTPUT_FOLDER / "calibration_summary.csv")

print("\nSummary table:")
print(summary_df[[
    "video",
    "source_file",
    "valid_percent",
    "pixels_per_mm",
    "mean_radius_mm",
    "std_radius_mm",
    "radius_error_mm",
    "radius_error_percent"
]])

# ==========================================================
# COMBINED PLOTS
# ==========================================================

# Combined mm trajectory overlay
plt.figure()
for valid, summary in zip(all_valid, summaries):
    plt.plot(
        valid["x_mm"],
        valid["y_mm"],
        ".",
        markersize=2,
        label=f"Video {summary['video']}"
    )

plt.axis("equal")
plt.xlabel("x position [mm]")
plt.ylabel("y position [mm]")
plt.title("Tracked dot trajectories for all videos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "combined_trajectory_mm.png", dpi=300)
plt.show()

# Calibration error comparison
plt.figure()
plt.bar(
    summary_df["video"].astype(str),
    summary_df["radius_error_percent"]
)
plt.axhline(0, linestyle="--")
plt.xlabel("Video")
plt.ylabel("Radius error [%]")
plt.title("Calibration radius error comparison")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_FOLDER / "combined_radius_error_percent.png", dpi=300)
plt.show()