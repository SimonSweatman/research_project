import cv2
import numpy as np
from pathlib import Path

# ==========================================================
# USER SETTINGS
# ==========================================================
VIDEO_PATH = r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260713_152205.mp4"

OUTPUT_FOLDER = Path("checkerboard_calibration_outputs")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

OUTPUT_NPZ = OUTPUT_FOLDER / "camera_calibration.npz"

# Your checkerboard:
# 10 x 8 squares = 9 x 7 internal corners
CHECKERBOARD = (9, 7)

# Square size in real units
SQUARE_SIZE_MM = 10.0

# Use every nth frame to avoid using many near-identical frames
FRAME_STEP = 15

# Stop after this many good checkerboard detections
MAX_GOOD_FRAMES = 40


# ==========================================================
# PREPARE OBJECT POINTS
# ==========================================================
# Real-world checkerboard points, e.g.:
# (0,0,0), (10,0,0), (20,0,0), ...
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[
    0:CHECKERBOARD[0],
    0:CHECKERBOARD[1]
].T.reshape(-1, 2)

objp *= SQUARE_SIZE_MM

object_points = []  # 3D real-world points
image_points = []   # 2D image points

# Criteria for sub-pixel corner refinement
criteria = (
    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
    30,
    0.001
)


# ==========================================================
# LOAD VIDEO
# ==========================================================
video_path = Path(VIDEO_PATH)

if not video_path.exists():
    raise FileNotFoundError(f"Video not found:\n{video_path}")

cap = cv2.VideoCapture(str(video_path))

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print(f"Loaded: {video_path.name}")
print(f"Resolution: {width} x {height}")
print(f"FPS: {fps}")
print(f"Frames: {frame_count}")
print(f"Checkerboard internal corners: {CHECKERBOARD}")
print(f"Square size: {SQUARE_SIZE_MM} mm")

preview_folder = OUTPUT_FOLDER / "detected_frames"
preview_folder.mkdir(parents=True, exist_ok=True)

frame_index = 0
good_count = 0

# ==========================================================
# PROCESS VIDEO FRAMES
# ==========================================================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_index % FRAME_STEP != 0:
        frame_index += 1
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    found, corners = cv2.findChessboardCorners(
        gray,
        CHECKERBOARD,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
    )

    if found:
        corners_refined = cv2.cornerSubPix(
            gray,
            corners,
            winSize=(11, 11),
            zeroZone=(-1, -1),
            criteria=criteria
        )

        object_points.append(objp)
        image_points.append(corners_refined)

        good_count += 1

        preview = frame.copy()
        cv2.drawChessboardCorners(preview, CHECKERBOARD, corners_refined, found)

        out_path = preview_folder / f"detected_{good_count:03d}_frame_{frame_index}.png"
        cv2.imwrite(str(out_path), preview)

        print(f"Detected checkerboard {good_count} at frame {frame_index}")

        if good_count >= MAX_GOOD_FRAMES:
            break

    frame_index += 1

cap.release()

print(f"\nGood checkerboard detections: {good_count}")

if good_count < 10:
    raise RuntimeError(
        "Too few checkerboard detections. Try recording again with the full board visible, "
        "better lighting, and moving it around the frame more slowly."
    )


# ==========================================================
# CAMERA CALIBRATION
# ==========================================================
image_size = (width, height)

ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
    object_points,
    image_points,
    image_size,
    None,
    None
)

print("\n--- Calibration result ---")
print(f"RMS reprojection error: {ret:.4f}")
print("\nCamera matrix:")
print(camera_matrix)
print("\nDistortion coefficients:")
print(dist_coeffs.ravel())

# ==========================================================
# CALCULATE MEAN REPROJECTION ERROR
# ==========================================================
total_error = 0
total_points = 0

for i in range(len(object_points)):
    projected_points, _ = cv2.projectPoints(
        object_points[i],
        rvecs[i],
        tvecs[i],
        camera_matrix,
        dist_coeffs
    )

    error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2)

    total_error += error ** 2
    total_points += len(projected_points)

mean_error_px = np.sqrt(total_error / total_points)

print(f"Mean reprojection error: {mean_error_px:.4f} px")

# ==========================================================
# SAVE .NPZ FILE
# ==========================================================
np.savez(
    OUTPUT_NPZ,
    camera_matrix=camera_matrix,
    dist_coeffs=dist_coeffs,
    rvecs=rvecs,
    tvecs=tvecs,
    checkerboard=np.array(CHECKERBOARD),
    square_size_mm=SQUARE_SIZE_MM,
    rms_reprojection_error=ret,
    mean_reprojection_error_px=mean_error_px,
    image_width=width,
    image_height=height
)

print(f"\nSaved calibration file:")
print(OUTPUT_NPZ.resolve())

# ==========================================================
# SAVE UNDISTORTED EXAMPLE IMAGE
# ==========================================================
cap = cv2.VideoCapture(str(video_path))
ret, frame = cap.read()
cap.release()

if ret:
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        image_size,
        alpha=1,
        newImgSize=image_size
    )

    undistorted = cv2.undistort(
        frame,
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix
    )

    cv2.imwrite(str(OUTPUT_FOLDER / "example_original.png"), frame)
    cv2.imwrite(str(OUTPUT_FOLDER / "example_undistorted.png"), undistorted)

    print("Saved example_original.png and example_undistorted.png")