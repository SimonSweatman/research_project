import cv2
from pathlib import Path

# ==========================================================
# USER SETTINGS
# ==========================================================

VIDEO_PATH = r"C:\Users\simon\OneDrive - University of Southampton\Documents\02_Uni\01_Masters\6012 research project\code\camera_calibration\20260714_160528.mp4"

OUTPUT_FOLDER = Path(r"C:\temp\checkerboard_detection_test")
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

PATTERN_SIZE = (9, 7)

# Search settings
SEARCH_DURATION_S = 5.0
CHECK_EVERY_N_FRAMES = 3


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
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

print("--- Video information ---")
print(f"Video: {video_path.name}")
print(f"FPS: {fps:.3f}")
print(f"Frame count: {frame_count}")
print(f"Duration: {frame_count / fps:.3f} s")
print(f"Looking for internal-corner pattern: {PATTERN_SIZE}")


# ==========================================================
# SEARCH VIDEO FRAMES
# ==========================================================

max_search_frames = min(
    frame_count,
    int(SEARCH_DURATION_S * fps)
)

found_successfully = False
successful_frame_index = None
successful_corners = None
successful_frame = None
detector_used = None

for frame_index in range(max_search_frames):
    ret, frame = cap.read()

    if not ret:
        break

    if frame_index % CHECK_EVERY_N_FRAMES != 0:
        continue

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Improve local contrast
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    gray_enhanced = clahe.apply(gray)

    # Try the newer detector first
    found, corners = cv2.findChessboardCornersSB(
        gray_enhanced,
        PATTERN_SIZE,
        flags=(
            cv2.CALIB_CB_NORMALIZE_IMAGE
            | cv2.CALIB_CB_EXHAUSTIVE
            | cv2.CALIB_CB_ACCURACY
        )
    )

    detector_used = "findChessboardCornersSB"

    # Fall back to traditional detector
    if not found:
        found, corners = cv2.findChessboardCorners(
            gray_enhanced,
            PATTERN_SIZE,
            flags=(
                cv2.CALIB_CB_ADAPTIVE_THRESH
                | cv2.CALIB_CB_NORMALIZE_IMAGE
            )
        )

        detector_used = "findChessboardCorners + cornerSubPix"

        if found:
            criteria = (
                cv2.TERM_CRITERIA_EPS
                + cv2.TERM_CRITERIA_MAX_ITER,
                50,
                0.001
            )

            corners = cv2.cornerSubPix(
                gray_enhanced,
                corners,
                winSize=(11, 11),
                zeroZone=(-1, -1),
                criteria=criteria
            )

    if found:
        found_successfully = True
        successful_frame_index = frame_index
        successful_corners = corners
        successful_frame = frame.copy()
        break


cap.release()


# ==========================================================
# SAVE RESULT
# ==========================================================

if not found_successfully:
    failure_path = OUTPUT_FOLDER / "checkerboard_not_detected.png"

    # Save the final checked frame when possible
    if "frame" in locals():
        cv2.imwrite(str(failure_path), frame)

    raise RuntimeError(
        f"Checkerboard was not detected in the first "
        f"{SEARCH_DURATION_S:.1f} seconds.\n\n"
        f"Expected pattern: {PATTERN_SIZE}\n"
        f"Failure image saved to:\n{failure_path}"
    )

result_image = successful_frame.copy()

cv2.drawChessboardCorners(
    result_image,
    PATTERN_SIZE,
    successful_corners,
    True
)

cv2.putText(
    result_image,
    f"Pattern detected: {PATTERN_SIZE}",
    (30, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.9,
    (0, 255, 0),
    2
)

cv2.putText(
    result_image,
    f"Frame: {successful_frame_index}",
    (30, 80),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (0, 255, 0),
    2
)

cv2.putText(
    result_image,
    detector_used,
    (30, 120),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (0, 255, 0),
    2
)

output_path = OUTPUT_FOLDER / "checkerboard_detected_corners.png"

cv2.imwrite(str(output_path), result_image)

print("\n--- Detection successful ---")
print(f"Detector used: {detector_used}")
print(f"Detected frame: {successful_frame_index}")
print(f"Number of corners: {len(successful_corners)}")
print(f"Saved result image to:\n{output_path}")