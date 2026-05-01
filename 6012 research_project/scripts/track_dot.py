import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ==========================================================
# USER SETTINGS
# ==========================================================
VIDEO_PATH = r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260501_140732.mp4"

OUTPUT_CSV = "tracked_lime_dot_positions.csv"
OUTPUT_PREVIEW = "tracking_preview.mp4"

KNOWN_RADIUS_MM = 22.0      # dot centre is 22 mm from wheel centre
DOT_DIAMETER_MM = 8.0       # used only as reference/info

# Lime green HSV range
LOWER_GREEN = np.array([35, 80, 80])
UPPER_GREEN = np.array([85, 255, 255])

MIN_AREA = 30
MAX_AREA = 20000

# ==========================================================
# CIRCLE FIT FUNCTION
# ==========================================================
def fit_circle(x, y):
    """
    Fits circle: (x-a)^2 + (y-b)^2 = r^2
    Returns centre x, centre y, radius.
    """
    A = np.column_stack((2 * x, 2 * y, np.ones(len(x))))
    b = x**2 + y**2

    c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    xc = c[0]
    yc = c[1]
    r = np.sqrt(c[2] + xc**2 + yc**2)

    return xc, yc, r

# ==========================================================
# LOAD VIDEO
# ==========================================================
video_path = Path(VIDEO_PATH)

if not video_path.exists():
    raise FileNotFoundError(
        f"Video not found:\n{video_path}\n\n"
        "Check the file extension. It may be .mov instead of .mp4."
    )

cap = cv2.VideoCapture(str(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Loaded: {video_path.name}")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")
print(f"Frames: {frame_count}")

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(OUTPUT_PREVIEW, fourcc, fps, (width, height))

results = []
frame_index = 0

# ==========================================================
# TRACKING LOOP
# ==========================================================
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

                    cv2.circle(frame, (int(x_px), int(y_px)), 10, (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"x={x_px:.1f}, y={y_px:.1f}",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 255, 0),
                        2
                    )
                    break

    results.append({
        "frame": frame_index,
        "time_s": time_s,
        "x_px": x_px,
        "y_px": y_px,
        "area_px2": area
    })

    out.write(frame)
    frame_index += 1

cap.release()
out.release()

# ==========================================================
# SAVE DATA
# ==========================================================
df = pd.DataFrame(results)
df.to_csv(OUTPUT_CSV, index=False)

valid = df.dropna(subset=["x_px", "y_px"]).copy()

print(f"\nSaved CSV: {OUTPUT_CSV}")
print(f"Saved preview video: {OUTPUT_PREVIEW}")
print(f"Valid tracked frames: {len(valid)} / {len(df)}")

if len(valid) < 10:
    raise RuntimeError("Too few valid tracked points. Adjust HSV range or lighting.")

# ==========================================================
# FIT CIRCLE AND CONVERT TO MM
# ==========================================================
x = valid["x_px"].to_numpy()
y = valid["y_px"].to_numpy()

xc, yc, r_px = fit_circle(x, y)

pixels_per_mm = r_px / KNOWN_RADIUS_MM
mm_per_pixel = 1 / pixels_per_mm

valid["x_mm"] = (valid["x_px"] - xc) * mm_per_pixel
valid["y_mm"] = -(valid["y_px"] - yc) * mm_per_pixel  # invert image y-axis

valid["radius_px"] = np.sqrt((valid["x_px"] - xc)**2 + (valid["y_px"] - yc)**2)
valid["radius_mm"] = valid["radius_px"] * mm_per_pixel

mean_radius_mm = valid["radius_mm"].mean()
std_radius_mm = valid["radius_mm"].std()
radius_error_mm = mean_radius_mm - KNOWN_RADIUS_MM
radius_error_percent = radius_error_mm / KNOWN_RADIUS_MM * 100

print("\n--- Calibration result ---")
print(f"Fitted centre: x={xc:.2f} px, y={yc:.2f} px")
print(f"Fitted radius: {r_px:.2f} px")
print(f"Scale: {pixels_per_mm:.2f} px/mm")
print(f"Mean measured radius: {mean_radius_mm:.2f} mm")
print(f"Radius error: {radius_error_mm:.2f} mm ({radius_error_percent:.2f}%)")
print(f"Radius standard deviation: {std_radius_mm:.2f} mm")

valid.to_csv("tracked_lime_dot_positions_with_mm.csv", index=False)

# ==========================================================
# PLOTS
# ==========================================================

# Plot 1: pixel trajectory
plt.figure()
plt.plot(valid["x_px"], valid["y_px"], ".", markersize=2)
plt.scatter([xc], [yc], marker="x", s=80)
plt.gca().invert_yaxis()
plt.axis("equal")
plt.xlabel("x position [pixels]")
plt.ylabel("y position [pixels]")
plt.title("Tracked lime green dot trajectory")
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_pixels.png", dpi=300)

# Plot 2: mm trajectory
plt.figure()
plt.plot(valid["x_mm"], valid["y_mm"], ".", markersize=2)
plt.axis("equal")
plt.xlabel("x position [mm]")
plt.ylabel("y position [mm]")
plt.title("Tracked dot trajectory converted to mm")
plt.grid(True)
plt.tight_layout()
plt.savefig("trajectory_mm.png", dpi=300)

# Plot 3: radius over time
plt.figure()
plt.plot(valid["time_s"], valid["radius_mm"])
plt.axhline(KNOWN_RADIUS_MM, linestyle="--", label="Known radius = 22 mm")
plt.xlabel("Time [s]")
plt.ylabel("Measured radius [mm]")
plt.title("Measured radius over time")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("radius_over_time.png", dpi=300)

plt.show()