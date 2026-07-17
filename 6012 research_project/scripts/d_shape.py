#!/usr/bin/env python3

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# USER PARAMETERS
# ============================================================

# Cilium link lengths [mm]
L1 = 45.0
L2 = 25.0

# Number of samples around one complete gait cycle
N = 720

# ============================================================
# ROTATED D-SHAPE PARAMETERS
# ============================================================

# Neutral servo commands used as the reference posture [deg]
HOME_LOWER_COMMAND_DEG = 90.0
HOME_UPPER_COMMAND_DEG = 90.0

# The D-shape is positioned relative to the neutral tip location.
# Because 90/90 places the two links fully extended at maximum reach,
# a horizontal stroke cannot pass exactly through that point: its left
# and right ends would lie outside the reachable workspace. The flat
# edge is therefore placed slightly below neutral.
D_TOP_DROP_FROM_HOME_Z = 2.0

# These are calculated automatically from the neutral posture after
# forward_kinematics() has been defined.
D_CENTRE_X = None
D_TOP_Z = None

# Half the horizontal width of the D-shape [mm]
# Full flat-stroke length = 2 x D_HALF_WIDTH_X
D_HALF_WIDTH_X = 12.0

# Maximum depth of the curved recovery below the flat edge [mm]
D_RECOVERY_DEPTH_Z = 8.0

# Fraction of the complete cycle used for the flat stroke
#
# 0.50 means:
# 50% of cycle = flat top stroke
# 50% of cycle = curved recovery
STRAIGHT_STROKE_FRACTION = 0.50

# True:
# flat top stroke moves from left to right
#
# False:
# flat top stroke moves from right to left
TOP_STROKE_LEFT_TO_RIGHT = True

# ============================================================
# KINEMATIC SETTINGS
# ============================================================

# Select the inverse-kinematics elbow configuration.
# Change to -1 if the cilium bends in the wrong direction.
ELBOW_SIGN = 1

# Mechanical-to-servo calibration offsets
LOWER_CMD_TO_MECH_OFFSET = 0.0
UPPER_CMD_TO_MECH_OFFSET = -90.0

# ============================================================
# PCA9685 COMMAND RESOLUTION MODEL
# ============================================================

# These must match the Arduino servo calibration.
MIN_PULSE_US = 600.0
MAX_PULSE_US = 2400.0
PWM_FREQUENCY_HZ = 50.0
PCA9685_COUNTS = 4096

# ============================================================
# OUTPUT LOCATION
# ============================================================

OUT_DIR = Path(__file__).resolve().parent.parent / "include"
OUT_DIR.mkdir(parents=True, exist_ok=True)

HEADER_PATH = OUT_DIR / "gait_table.h"
PATH_PLOT = OUT_DIR / "tip_path_d_shape_pca9685_xz.png"
ANGLE_PLOT = OUT_DIR / "d_shape_pca9685_pwm_commands.png"


# ============================================================
# GENERAL FUNCTIONS
# ============================================================

def clamp(value, minimum, maximum):
    """Restrict a value to a specified range."""
    return max(minimum, min(maximum, value))


def smoothstep(value):
    """
    Smoothly move from 0 to 1.

    The slope is zero at both ends, which reduces sudden
    acceleration at the ends of the flat and curved sections.
    """
    value = clamp(value, 0.0, 1.0)
    return value * value * (3.0 - 2.0 * value)


# ============================================================
# ROTATED D-SHAPE DEFINITION
# ============================================================

def d_shape_position(phase):
    """
    Return the desired tip position for one point in the cycle.

    The path consists of:

    1. A flat horizontal stroke at the top.
    2. A lower half-ellipse returning underneath it.

    phase must be between 0 and 1.
    """
    phase = phase % 1.0

    left_x = D_CENTRE_X - D_HALF_WIDTH_X
    right_x = D_CENTRE_X + D_HALF_WIDTH_X

    # --------------------------------------------------------
    # PART 1: FLAT UPPER STROKE
    # --------------------------------------------------------

    if phase < STRAIGHT_STROKE_FRACTION:
        local_phase = phase / STRAIGHT_STROKE_FRACTION

        # Smooth progression along the straight line
        u = smoothstep(local_phase)

        if TOP_STROKE_LEFT_TO_RIGHT:
            x = left_x + (right_x - left_x) * u
        else:
            x = right_x + (left_x - right_x) * u

        z = D_TOP_Z

    # --------------------------------------------------------
    # PART 2: CURVED LOWER RECOVERY
    # --------------------------------------------------------

    else:
        local_phase = (
            phase - STRAIGHT_STROKE_FRACTION
        ) / (
            1.0 - STRAIGHT_STROKE_FRACTION
        )

        # Move from 0 to pi around the lower half-ellipse
        theta = np.pi * smoothstep(local_phase)

        if TOP_STROKE_LEFT_TO_RIGHT:
            # Right-hand end back to left-hand end
            x = (
                D_CENTRE_X
                + D_HALF_WIDTH_X * np.cos(theta)
            )
        else:
            # Left-hand end back to right-hand end
            x = (
                D_CENTRE_X
                - D_HALF_WIDTH_X * np.cos(theta)
            )

        z = (
            D_TOP_Z
            - D_RECOVERY_DEPTH_Z * np.sin(theta)
        )

    return x, z


# ============================================================
# INVERSE KINEMATICS
# ============================================================

def ik_2link(x, z):
    """
    Calculate the two mechanical joint angles required to place
    the cilium tip at the requested x-z coordinate.

    Returns:
        lower_mechanical_angle_deg
        upper_relative_mechanical_angle_deg
    """
    radius_squared = x**2 + z**2
    radius = np.sqrt(radius_squared)

    minimum_reach = abs(L1 - L2)
    maximum_reach = L1 + L2

    if radius > maximum_reach or radius < minimum_reach:
        raise ValueError(
            f"Point x={x:.3f}, z={z:.3f} is outside the workspace. "
            f"Distance from base = {radius:.3f} mm. "
            f"Valid reach = {minimum_reach:.3f} to "
            f"{maximum_reach:.3f} mm."
        )

    cos_t2 = (
        radius_squared - L1**2 - L2**2
    ) / (
        2.0 * L1 * L2
    )

    cos_t2 = clamp(cos_t2, -1.0, 1.0)

    upper_relative_rad = (
        ELBOW_SIGN * np.arccos(cos_t2)
    )

    k1 = L1 + L2 * np.cos(upper_relative_rad)
    k2 = L2 * np.sin(upper_relative_rad)

    lower_rad = (
        np.arctan2(z, x)
        - np.arctan2(k2, k1)
    )

    lower_mechanical_deg = np.rad2deg(lower_rad)

    upper_relative_mechanical_deg = np.rad2deg(
        upper_relative_rad
    )

    return (
        lower_mechanical_deg,
        upper_relative_mechanical_deg
    )


# ============================================================
# SERVO ANGLE CONVERSION
# ============================================================

def mechanical_to_servo(
    lower_mechanical_deg,
    upper_relative_mechanical_deg
):
    """
    Convert mechanical model angles into servo command angles.
    """
    lower_command_deg = (
        lower_mechanical_deg
        - LOWER_CMD_TO_MECH_OFFSET
    )

    upper_command_deg = (
        upper_relative_mechanical_deg
        - UPPER_CMD_TO_MECH_OFFSET
    )

    return lower_command_deg, upper_command_deg


# ============================================================
# FORWARD KINEMATICS
# ============================================================

def forward_kinematics(
    lower_command_deg,
    upper_command_deg
):
    """
    Calculate the predicted tip position produced by a pair of
    servo commands.

    This is used to check the path produced by the generated
    floating-point command table.
    """
    lower_mechanical_deg = (
        lower_command_deg
        + LOWER_CMD_TO_MECH_OFFSET
    )

    upper_relative_mechanical_deg = (
        upper_command_deg
        + UPPER_CMD_TO_MECH_OFFSET
    )

    lower_rad = np.deg2rad(lower_mechanical_deg)

    upper_relative_rad = np.deg2rad(
        upper_relative_mechanical_deg
    )

    x = (
        L1 * np.cos(lower_rad)
        + L2 * np.cos(
            lower_rad + upper_relative_rad
        )
    )

    z = (
        L1 * np.sin(lower_rad)
        + L2 * np.sin(
            lower_rad + upper_relative_rad
        )
    )

    return x, z


# ============================================================
# POSITION THE D-SHAPE FROM THE NEUTRAL POSTURE
# ============================================================

HOME_X, HOME_Z = forward_kinematics(
    HOME_LOWER_COMMAND_DEG,
    HOME_UPPER_COMMAND_DEG
)

# Centre the path horizontally on the neutral tip position.
D_CENTRE_X = HOME_X

# Put the flat edge slightly below the fully extended neutral tip.
D_TOP_Z = HOME_Z - D_TOP_DROP_FROM_HOME_Z

# Check that the two ends of the flat stroke are reachable.
flat_end_radius = np.sqrt(
    D_HALF_WIDTH_X**2 + D_TOP_Z**2
)

if flat_end_radius > L1 + L2:
    minimum_required_drop = (
        HOME_Z
        - np.sqrt((L1 + L2)**2 - D_HALF_WIDTH_X**2)
    )

    raise ValueError(
        "The D-shaped flat stroke is outside the cilium workspace. "
        f"With a half-width of {D_HALF_WIDTH_X:.3f} mm, "
        f"D_TOP_DROP_FROM_HOME_Z must be at least "
        f"{minimum_required_drop:.3f} mm. Current value: "
        f"{D_TOP_DROP_FROM_HOME_Z:.3f} mm."
    )


# ============================================================
# PCA9685 QUANTISATION
# ============================================================

def quantise_servo_angle_to_pca9685(angle_deg):
    """
    Convert a floating-point servo command to the nearest output
    that the 12-bit PCA9685 can produce at the selected PWM
    frequency.

    This function is only used to predict the theoretical
    PCA9685-limited tip path. The values exported to gait_table.h
    remain the original floating-point inverse-kinematics commands.
    """
    angle_deg = float(np.clip(angle_deg, 0.0, 180.0))

    # Convert servo angle to pulse width.
    pulse_us = (
        MIN_PULSE_US
        + angle_deg / 180.0
        * (MAX_PULSE_US - MIN_PULSE_US)
    )

    # Convert pulse width to the nearest integer PCA9685 count.
    pwm_period_us = 1_000_000.0 / PWM_FREQUENCY_HZ

    raw_count = (
        pulse_us / pwm_period_us
        * PCA9685_COUNTS
    )

    quantised_count = int(round(raw_count))
    quantised_count = int(
        np.clip(quantised_count, 0, PCA9685_COUNTS - 1)
    )

    # Convert the achievable count back to an equivalent
    # servo command angle for forward-kinematics prediction.
    quantised_pulse_us = (
        quantised_count / PCA9685_COUNTS
        * pwm_period_us
    )

    quantised_angle_deg = (
        (quantised_pulse_us - MIN_PULSE_US)
        / (MAX_PULSE_US - MIN_PULSE_US)
        * 180.0
    )

    return quantised_angle_deg, quantised_count


# ============================================================
# GENERATE THE GAIT TABLE
# ============================================================

# Floating-point servo tables
lower_table = np.zeros(N, dtype=np.float32)
upper_table = np.zeros(N, dtype=np.float32)

# Desired D-shaped coordinates
x_desired = np.zeros(N, dtype=float)
z_desired = np.zeros(N, dtype=float)

# Theoretical path after PCA9685 quantisation
x_pca_quantised = np.zeros(N, dtype=float)
z_pca_quantised = np.zeros(N, dtype=float)

# Raw PCA9685 counts used only for reporting and checking
lower_pca_counts = np.zeros(N, dtype=np.uint16)
upper_pca_counts = np.zeros(N, dtype=np.uint16)

for index in range(N):
    phase = index / N

    # Generate the desired D-shaped tip coordinate
    x, z = d_shape_position(phase)

    x_desired[index] = x
    z_desired[index] = z

    # Calculate mechanical joint angles
    lower_mechanical, upper_relative = ik_2link(x, z)

    # Convert mechanical angles into servo commands
    lower_command, upper_command = mechanical_to_servo(
        lower_mechanical,
        upper_relative
    )

    if not 0.0 <= lower_command <= 180.0:
        raise ValueError(
            f"Lower servo command outside range at sample {index}: "
            f"{lower_command:.4f} degrees."
        )

    if not 0.0 <= upper_command <= 180.0:
        raise ValueError(
            f"Upper servo command outside range at sample {index}: "
            f"{upper_command:.4f} degrees."
        )

    # Clip for safety, but do not round to whole degrees
    lower_command = float(
        np.clip(lower_command, 0.0, 180.0)
    )

    upper_command = float(
        np.clip(upper_command, 0.0, 180.0)
    )

    # Keep the original floating-point commands for analysis only.
    # The Arduino header will contain the corresponding raw PCA9685 counts.
    lower_table[index] = lower_command
    upper_table[index] = upper_command

    # Quantise only for the theoretical hardware-limit plot.
    (
        lower_quantised,
        lower_count
    ) = quantise_servo_angle_to_pca9685(
        lower_command
    )

    (
        upper_quantised,
        upper_count
    ) = quantise_servo_angle_to_pca9685(
        upper_command
    )

    lower_pca_counts[index] = lower_count
    upper_pca_counts[index] = upper_count

    (
        x_pca_quantised[index],
        z_pca_quantised[index]
    ) = forward_kinematics(
        lower_quantised,
        upper_quantised
    )


# ============================================================
# COMMAND CHANGE ANALYSIS
# ============================================================

def command_jump_statistics(commands):
    """
    Calculate command changes between consecutive table entries,
    including the final-to-first transition.
    """
    wrapped_commands = np.r_[
        commands,
        commands[0]
    ]

    differences = np.diff(wrapped_commands)

    absolute_differences = np.abs(differences)

    return {
        "maximum_deg": float(
            np.max(absolute_differences)
        ),
        "mean_deg": float(
            np.mean(absolute_differences)
        ),
        "minimum_nonzero_deg": float(
            np.min(
                absolute_differences[
                    absolute_differences > 1e-9
                ]
            )
        )
        if np.any(absolute_differences > 1e-9)
        else 0.0
    }


lower_jump = command_jump_statistics(lower_table)
upper_jump = command_jump_statistics(upper_table)


# ============================================================
# THEORETICAL PCA9685-LIMITED PATH ERROR
# ============================================================

position_error_mm = np.sqrt(
    (x_pca_quantised - x_desired)**2
    + (z_pca_quantised - z_desired)**2
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print("\n--- Neutral-referenced rotated D-shaped path ---")
print(
    f"Neutral command: lower={HOME_LOWER_COMMAND_DEG:.2f} deg, "
    f"upper={HOME_UPPER_COMMAND_DEG:.2f} deg"
)
print(
    f"Neutral tip position: x={HOME_X:.3f} mm, z={HOME_Z:.3f} mm"
)
print(
    f"Flat edge drop below neutral: "
    f"{D_TOP_DROP_FROM_HOME_Z:.3f} mm"
)
print(
    f"Flat stroke length: "
    f"{2.0 * D_HALF_WIDTH_X:.3f} mm"
)
print(
    f"Recovery depth: "
    f"{D_RECOVERY_DEPTH_Z:.3f} mm"
)
print(
    f"Flat stroke height: "
    f"{D_TOP_Z:.3f} mm"
)
print(
    f"Straight-stroke cycle fraction: "
    f"{STRAIGHT_STROKE_FRACTION:.3f}"
)

print("\n--- Servo command ranges ---")
print(
    f"Lower servo: "
    f"{lower_table.min():.4f} to "
    f"{lower_table.max():.4f} degrees"
)
print(
    f"Upper servo: "
    f"{upper_table.min():.4f} to "
    f"{upper_table.max():.4f} degrees"
)


print("\n--- PCA9685 quantisation ---")
print(
    f"Lower count range: "
    f"{lower_pca_counts.min()} to "
    f"{lower_pca_counts.max()}"
)
print(
    f"Upper count range: "
    f"{upper_pca_counts.min()} to "
    f"{upper_pca_counts.max()}"
)
print(
    f"Unique lower PCA commands: "
    f"{len(np.unique(lower_pca_counts))}"
)
print(
    f"Unique upper PCA commands: "
    f"{len(np.unique(upper_pca_counts))}"
)

print("\n--- Lower command changes ---")
print(
    f"Maximum table-step change: "
    f"{lower_jump['maximum_deg']:.6f} degrees"
)
print(
    f"Mean table-step change: "
    f"{lower_jump['mean_deg']:.6f} degrees"
)
print(
    f"Minimum non-zero table-step change: "
    f"{lower_jump['minimum_nonzero_deg']:.6f} degrees"
)

print("\n--- Upper command changes ---")
print(
    f"Maximum table-step change: "
    f"{upper_jump['maximum_deg']:.6f} degrees"
)
print(
    f"Mean table-step change: "
    f"{upper_jump['mean_deg']:.6f} degrees"
)
print(
    f"Minimum non-zero table-step change: "
    f"{upper_jump['minimum_nonzero_deg']:.6f} degrees"
)

print("\n--- Theoretical PCA9685-limited path error ---")
print(
    f"Mean path error: "
    f"{np.mean(position_error_mm):.9f} mm"
)
print(
    f"Maximum path error: "
    f"{np.max(position_error_mm):.9f} mm"
)


# ============================================================
# SAVE COMMANDED-PATH PNG
# ============================================================

plt.figure(figsize=(8, 6))

plt.plot(
    x_desired,
    z_desired,
    "--",
    linewidth=2,
    label="Desired D-shaped path"
)

plt.plot(
    x_pca_quantised,
    z_pca_quantised,
    linewidth=1.5,
    label="PCA9685-quantised commanded path"
)

plt.scatter(
    [x_pca_quantised[0]],
    [z_pca_quantised[0]],
    s=45,
    label="Start"
)

plt.xlabel("x displacement [mm]")
plt.ylabel("z displacement [mm]")

plt.title(
    "Desired and theoretical PCA9685-limited tip paths"
)

plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    PATH_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# SAVE PCA9685 COMMAND PNG
# ============================================================

phase_percent = np.arange(N) / N * 100.0

plt.figure(figsize=(10, 5))

plt.step(
    phase_percent,
    lower_pca_counts,
    where="post",
    label="Lower PCA9685 command"
)

plt.step(
    phase_percent,
    upper_pca_counts,
    where="post",
    label="Upper PCA9685 command"
)

plt.axvline(
    STRAIGHT_STROKE_FRACTION * 100.0,
    linestyle="--",
    label="Start of curved recovery"
)

plt.xlabel("Cycle phase [%]")
plt.ylabel("PCA9685 off-count [0-4095]")
plt.title("PCA9685 commands for rotated D-shaped gait")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

plt.savefig(
    ANGLE_PLOT,
    dpi=300
)

plt.close()


# ============================================================
# EXPORT RAW PCA9685 COUNTS TO ARDUINO HEADER
# ============================================================

def format_uint16_array(name, values):
    """Format an array as an Arduino PROGMEM uint16_t array."""
    lines = []
    current_line = []

    for index, value in enumerate(values):
        current_line.append(f"{int(value):4d}")

        if len(current_line) == 12 or index == len(values) - 1:
            lines.append("    " + ", ".join(current_line))
            current_line = []

    body = ",\n".join(lines)

    return (
        f"const uint16_t {name}[{len(values)}] "
        f"PROGMEM = {{\n"
        f"{body}\n"
        f"}};\n"
    )


header = [
    "#pragma once",
    "#include <Arduino.h>",
    "",
    f"const uint16_t GAIT_TABLE_SIZE = {N};",
    f"const uint16_t GAIT_PWM_FREQUENCY_HZ = {int(PWM_FREQUENCY_HZ)};",
    f"const uint16_t GAIT_MIN_PULSE_US = {int(MIN_PULSE_US)};",
    f"const uint16_t GAIT_MAX_PULSE_US = {int(MAX_PULSE_US)};",
    "",
    format_uint16_array(
        "LOWER_PWM_TABLE",
        lower_pca_counts
    ),
    "",
    format_uint16_array(
        "UPPER_PWM_TABLE",
        upper_pca_counts
    )
]

HEADER_PATH.write_text(
    "\n".join(header),
    encoding="utf-8"
)


# ============================================================
# FINAL OUTPUTS
# ============================================================

print("\n--- Files saved ---")
print(f"Gait table: {HEADER_PATH}")
print(f"Commanded path PNG: {PATH_PLOT}")
print(f"PCA9685 command PNG: {ANGLE_PLOT}")