#!/usr/bin/env python3
"""
Generate a 360-point raw-PWM gait for one two-link conveyor cilium.

The requested tip path is a rotated D:

    1. A long, straight drive stroke at constant height.
    2. A shorter lower recovery stroke that clears the test object.

The output is an Arduino header containing uint16_t PCA9685 off-counts.
The default PWM conversion deliberately reproduces the Seeed
ServoDriver::setAngle() mapping configured with:

    servo.setServoPulseRange(500, 2500, 180);
    servo.setFrequency(50);

This is important because that library does not use the ideal
4096-count / 20,000-us conversion. With the settings above it sends
counts 122, 302 and 482 for 0, 90 and 180 degrees respectively.

Install requirements:

    python -m pip install numpy matplotlib

Run:

    python generate_cilia_pwm_gait.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


# =============================================================================
# CILIUM AND GAIT SETTINGS
# =============================================================================

# Pivot-to-pivot and upper-pivot-to-tip lengths [mm].
L1_MM = 50.0
L2_MM = 50.0

# Exactly one table entry per degree of gait phase.
TABLE_SIZE = 360

# Neutral servo commands [deg]. In the model this is the straight, upright
# posture around which the drive stroke is centred.
HOME_LOWER_DEG = 90.0
HOME_UPPER_DEG = 90.0

# Rotated-D tip path. The default provides a 24 mm drive stroke and a 15 mm
# recovery clearance. Increase HALF_STROKE_MM for more travel, or
# RECOVERY_DEPTH_MM if the tip does not clear the object during recovery.
HALF_STROKE_MM = 12.0
TOP_DROP_FROM_HOME_MM = 2.0
RECOVERY_DEPTH_MM = 15.0

# 60% slow drive, 40% quicker recovery. Neighbouring cilia can later be
# phase-shifted so their drive strokes overlap this cilium's recovery.
DRIVE_FRACTION = 0.60

# Set False to reverse the conveying direction.
DRIVE_LEFT_TO_RIGHT = True


# =============================================================================
# MECHANICAL / SERVO ANGLE CONVENTION
# =============================================================================

# Use -1 if the real mechanism bends in the opposite direction.
ELBOW_SIGN = 1

# mechanical_angle = servo_command + offset
LOWER_COMMAND_TO_MECHANICAL_OFFSET_DEG = 0.0
UPPER_COMMAND_TO_MECHANICAL_OFFSET_DEG = -90.0

# Safe command limits. Narrow these if the real joints hit mechanical stops
# before the nominal 0-180 degree servo range.
LOWER_COMMAND_MIN_DEG = 0.0
LOWER_COMMAND_MAX_DEG = 180.0
UPPER_COMMAND_MIN_DEG = 0.0
UPPER_COMMAND_MAX_DEG = 180.0


# =============================================================================
# SEEED PCA9685 RAW-PWM MODEL
# =============================================================================

# These must match the Arduino initialisation.
MIN_PULSE_US = 500.0
MAX_PULSE_US = 2500.0
SERVO_RANGE_DEG = 180
PWM_FREQUENCY_HZ = 50

# Reproduce setServoPulseRange() and the integer division in setAngle().
SEEED_MIN_COUNT = int((2.46 * MIN_PULSE_US) / 10.0 - 1.0)
SEEED_MAX_COUNT_INTERNAL = int((2.46 * MAX_PULSE_US) / 10.0 - 1.0)
SEEED_COUNTS_PER_DEG = (
    SEEED_MAX_COUNT_INTERNAL - SEEED_MIN_COUNT
) // SERVO_RANGE_DEG
SEEED_HIGHEST_COMMAND_COUNT = (
    SEEED_MIN_COUNT + SEEED_COUNTS_PER_DEG * SERVO_RANGE_DEG
)

# Defaults exactly match setAngle(). Replace either pair of arrays with
# measured servo angle/count data later to include individual non-linearity.
LOWER_CAL_ANGLES_DEG = np.array([0.0, 180.0])
LOWER_CAL_COUNTS = np.array(
    [float(SEEED_MIN_COUNT), float(SEEED_HIGHEST_COMMAND_COUNT)]
)
UPPER_CAL_ANGLES_DEG = np.array([0.0, 180.0])
UPPER_CAL_COUNTS = np.array(
    [float(SEEED_MIN_COUNT), float(SEEED_HIGHEST_COMMAND_COUNT)]
)

# Search nearby integer count pairs in Cartesian space instead of simply
# rounding each joint independently. The high z weighting prioritises a flat
# drive stroke after PWM quantisation.
COUNT_SEARCH_RADIUS = 3
DRIVE_HEIGHT_ERROR_WEIGHT = 30.0


# =============================================================================
# OUTPUTS
# =============================================================================

OUTPUT_DIR = Path(__file__).resolve().parent / "generated_gait"
HEADER_PATH = OUTPUT_DIR / "gait_table.h"
PATH_PLOT_PATH = OUTPUT_DIR / "gait_tip_path.png"
COMMAND_PLOT_PATH = OUTPUT_DIR / "gait_pwm_commands.png"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def smootherstep(value: float) -> float:
    """Quintic 0-to-1 blend with zero velocity and acceleration at each end."""
    value = clamp(value, 0.0, 1.0)
    return value**3 * (value * (6.0 * value - 15.0) + 10.0)


def forward_kinematics(
    lower_command_deg: float,
    upper_command_deg: float,
) -> tuple[float, float]:
    """Return tip x,z [mm] from the two servo command angles."""
    lower_mechanical_deg = (
        lower_command_deg + LOWER_COMMAND_TO_MECHANICAL_OFFSET_DEG
    )
    upper_relative_mechanical_deg = (
        upper_command_deg + UPPER_COMMAND_TO_MECHANICAL_OFFSET_DEG
    )

    q1 = np.deg2rad(lower_mechanical_deg)
    q2 = np.deg2rad(upper_relative_mechanical_deg)

    x = L1_MM * np.cos(q1) + L2_MM * np.cos(q1 + q2)
    z = L1_MM * np.sin(q1) + L2_MM * np.sin(q1 + q2)
    return float(x), float(z)


HOME_X_MM, HOME_Z_MM = forward_kinematics(
    HOME_LOWER_DEG, HOME_UPPER_DEG
)
DRIVE_HEIGHT_MM = HOME_Z_MM - TOP_DROP_FROM_HOME_MM


def desired_tip_position(phase: float) -> tuple[float, float]:
    """
    Return one point on the rotated-D path.

    The drive is a genuinely straight line: only x changes and z remains
    constant. The recovery is the lower half of an ellipse.
    """
    phase %= 1.0
    left_x = HOME_X_MM - HALF_STROKE_MM
    right_x = HOME_X_MM + HALF_STROKE_MM

    if phase < DRIVE_FRACTION:
        local_phase = phase / DRIVE_FRACTION
        progress = smootherstep(local_phase)
        if DRIVE_LEFT_TO_RIGHT:
            x = left_x + (right_x - left_x) * progress
        else:
            x = right_x + (left_x - right_x) * progress
        z = DRIVE_HEIGHT_MM
    else:
        local_phase = (phase - DRIVE_FRACTION) / (1.0 - DRIVE_FRACTION)
        theta = np.pi * smootherstep(local_phase)
        if DRIVE_LEFT_TO_RIGHT:
            x = HOME_X_MM + HALF_STROKE_MM * np.cos(theta)
        else:
            x = HOME_X_MM - HALF_STROKE_MM * np.cos(theta)
        z = DRIVE_HEIGHT_MM - RECOVERY_DEPTH_MM * np.sin(theta)

    return float(x), float(z)


def inverse_kinematics(x: float, z: float) -> tuple[float, float]:
    """Return lower absolute and upper relative mechanical angles [deg]."""
    radius_squared = x * x + z * z
    radius = np.sqrt(radius_squared)
    minimum_reach = abs(L1_MM - L2_MM)
    maximum_reach = L1_MM + L2_MM

    if not minimum_reach <= radius <= maximum_reach:
        raise ValueError(
            f"Unreachable point x={x:.3f}, z={z:.3f} mm; "
            f"radius={radius:.3f} mm, valid range="
            f"{minimum_reach:.3f} to {maximum_reach:.3f} mm."
        )

    cos_q2 = (
        radius_squared - L1_MM**2 - L2_MM**2
    ) / (2.0 * L1_MM * L2_MM)
    q2 = ELBOW_SIGN * np.arccos(clamp(float(cos_q2), -1.0, 1.0))
    q1 = np.arctan2(z, x) - np.arctan2(
        L2_MM * np.sin(q2),
        L1_MM + L2_MM * np.cos(q2),
    )

    return float(np.rad2deg(q1)), float(np.rad2deg(q2))


def mechanical_to_servo(
    lower_mechanical_deg: float,
    upper_relative_mechanical_deg: float,
) -> tuple[float, float]:
    lower_command = (
        lower_mechanical_deg - LOWER_COMMAND_TO_MECHANICAL_OFFSET_DEG
    )
    upper_command = (
        upper_relative_mechanical_deg
        - UPPER_COMMAND_TO_MECHANICAL_OFFSET_DEG
    )
    return float(lower_command), float(upper_command)


def calibration(servo_name: str) -> tuple[np.ndarray, np.ndarray]:
    if servo_name == "lower":
        return LOWER_CAL_ANGLES_DEG, LOWER_CAL_COUNTS
    if servo_name == "upper":
        return UPPER_CAL_ANGLES_DEG, UPPER_CAL_COUNTS
    raise ValueError(f"Unknown servo: {servo_name}")


def validate_calibration(servo_name: str) -> None:
    angles, counts = calibration(servo_name)
    if len(angles) < 2 or len(angles) != len(counts):
        raise ValueError(f"Invalid {servo_name} calibration array lengths.")
    if not np.all(np.diff(angles) > 0.0):
        raise ValueError(f"{servo_name} calibration angles must increase.")
    if not (
        np.all(np.diff(counts) > 0.0)
        or np.all(np.diff(counts) < 0.0)
    ):
        raise ValueError(
            f"{servo_name} calibration counts must be strictly monotonic."
        )


def servo_angle_to_count(angle_deg: float, servo_name: str) -> float:
    angles, counts = calibration(servo_name)
    angle_deg = clamp(float(angle_deg), float(angles[0]), float(angles[-1]))
    return float(np.interp(angle_deg, angles, counts))


def count_to_servo_angle(count: float, servo_name: str) -> float:
    angles, counts = calibration(servo_name)
    if counts[0] < counts[-1]:
        return float(np.interp(count, counts, angles))
    return float(np.interp(count, counts[::-1], angles[::-1]))


def choose_best_integer_counts(
    lower_command_deg: float,
    upper_command_deg: float,
    target_x: float,
    target_z: float,
    is_drive: bool,
) -> tuple[int, int, float, float]:
    """
    Find the nearby raw-count pair with the smallest tip-position error.

    Vertical error is weighted during the drive stroke, so the discretised
    path remains much flatter than independent joint rounding.
    """
    lower_raw = servo_angle_to_count(lower_command_deg, "lower")
    upper_raw = servo_angle_to_count(upper_command_deg, "upper")
    lower_centre = int(round(lower_raw))
    upper_centre = int(round(upper_raw))

    lower_low = int(np.floor(np.min(LOWER_CAL_COUNTS)))
    lower_high = int(np.ceil(np.max(LOWER_CAL_COUNTS)))
    upper_low = int(np.floor(np.min(UPPER_CAL_COUNTS)))
    upper_high = int(np.ceil(np.max(UPPER_CAL_COUNTS)))
    z_weight = DRIVE_HEIGHT_ERROR_WEIGHT if is_drive else 1.0

    best_score = np.inf
    best = None

    for lower_count in range(
        max(lower_low, lower_centre - COUNT_SEARCH_RADIUS),
        min(lower_high, lower_centre + COUNT_SEARCH_RADIUS) + 1,
    ):
        lower_angle = count_to_servo_angle(lower_count, "lower")

        for upper_count in range(
            max(upper_low, upper_centre - COUNT_SEARCH_RADIUS),
            min(upper_high, upper_centre + COUNT_SEARCH_RADIUS) + 1,
        ):
            upper_angle = count_to_servo_angle(upper_count, "upper")
            actual_x, actual_z = forward_kinematics(
                lower_angle, upper_angle
            )
            x_error = actual_x - target_x
            z_error = actual_z - target_z

            # The tiny term only resolves equal Cartesian scores in favour of
            # the count pair nearest the continuous IK solution.
            score = (
                x_error**2
                + z_weight * z_error**2
                + 1e-10
                * (
                    (lower_count - lower_raw) ** 2
                    + (upper_count - upper_raw) ** 2
                )
            )

            if score < best_score:
                best_score = score
                best = (lower_count, upper_count, actual_x, actual_z)

    if best is None:
        raise RuntimeError("No valid PWM-count pair was found.")
    return best


def format_progmem_array(name: str, values: np.ndarray) -> str:
    rows = []
    for start in range(0, len(values), 12):
        row = ", ".join(f"{int(value):3d}" for value in values[start:start + 12])
        rows.append(f"    {row}")
    return (
        f"const uint16_t {name}[GAIT_TABLE_SIZE] PROGMEM = {{\n"
        + ",\n".join(rows)
        + "\n};"
    )


def generate() -> None:
    validate_calibration("lower")
    validate_calibration("upper")

    if not 0.0 < DRIVE_FRACTION < 1.0:
        raise ValueError("DRIVE_FRACTION must be between 0 and 1.")

    # Check the most distant top corners before generating the whole path.
    top_corner_radius = np.hypot(HALF_STROKE_MM, DRIVE_HEIGHT_MM)
    if top_corner_radius > L1_MM + L2_MM:
        required_drop = HOME_Z_MM - np.sqrt(
            (L1_MM + L2_MM) ** 2 - HALF_STROKE_MM**2
        )
        raise ValueError(
            "The flat drive stroke is outside the workspace. "
            f"TOP_DROP_FROM_HOME_MM must be at least "
            f"{required_drop:.3f} mm for the selected stroke width."
        )

    lower_angles = np.zeros(TABLE_SIZE)
    upper_angles = np.zeros(TABLE_SIZE)
    lower_counts = np.zeros(TABLE_SIZE, dtype=np.uint16)
    upper_counts = np.zeros(TABLE_SIZE, dtype=np.uint16)
    desired_x = np.zeros(TABLE_SIZE)
    desired_z = np.zeros(TABLE_SIZE)
    quantised_x = np.zeros(TABLE_SIZE)
    quantised_z = np.zeros(TABLE_SIZE)

    for index in range(TABLE_SIZE):
        phase = index / TABLE_SIZE
        desired_x[index], desired_z[index] = desired_tip_position(phase)
        lower_mech, upper_relative_mech = inverse_kinematics(
            desired_x[index], desired_z[index]
        )
        lower_angles[index], upper_angles[index] = mechanical_to_servo(
            lower_mech, upper_relative_mech
        )

        if not LOWER_COMMAND_MIN_DEG <= lower_angles[index] <= LOWER_COMMAND_MAX_DEG:
            raise ValueError(
                f"Lower command {lower_angles[index]:.3f} degrees at "
                f"sample {index} exceeds its safe range."
            )
        if not UPPER_COMMAND_MIN_DEG <= upper_angles[index] <= UPPER_COMMAND_MAX_DEG:
            raise ValueError(
                f"Upper command {upper_angles[index]:.3f} degrees at "
                f"sample {index} exceeds its safe range."
            )

        (
            lower_counts[index],
            upper_counts[index],
            quantised_x[index],
            quantised_z[index],
        ) = choose_best_integer_counts(
            lower_angles[index],
            upper_angles[index],
            desired_x[index],
            desired_z[index],
            is_drive=phase < DRIVE_FRACTION,
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    header = "\n".join(
        [
            "#pragma once",
            "#include <Arduino.h>",
            "",
            "// Generated by generate_cilia_pwm_gait.py",
            "// Values are raw PCA9685 OFF counts for setPWM/setPwm.",
            f"constexpr uint16_t GAIT_TABLE_SIZE = {TABLE_SIZE};",
            f"constexpr uint16_t GAIT_PWM_FREQUENCY_HZ = {PWM_FREQUENCY_HZ};",
            f"constexpr float GAIT_DRIVE_FRACTION = {DRIVE_FRACTION:.6f}f;",
            f"constexpr uint16_t GAIT_ZERO_DEG_COUNT = {SEEED_MIN_COUNT};",
            f"constexpr uint16_t GAIT_COUNTS_PER_DEG = {SEEED_COUNTS_PER_DEG};",
            "",
            format_progmem_array("LOWER_PWM_TABLE", lower_counts),
            "",
            format_progmem_array("UPPER_PWM_TABLE", upper_counts),
            "",
        ]
    )
    HEADER_PATH.write_text(header, encoding="utf-8")

    drive_mask = np.arange(TABLE_SIZE) / TABLE_SIZE < DRIVE_FRACTION
    position_error = np.hypot(
        quantised_x - desired_x, quantised_z - desired_z
    )
    drive_height_peak_to_peak = np.ptp(quantised_z[drive_mask])

    fig, axis = plt.subplots(figsize=(8, 6))
    axis.plot(desired_x, desired_z, "--", linewidth=2, label="Desired path")
    axis.step(
        quantised_x,
        quantised_z,
        where="post",
        linewidth=1.4,
        label="Quantised PWM path",
    )
    axis.scatter(
        quantised_x[0], quantised_z[0], s=45, zorder=3, label="Sample 0"
    )
    axis.set(
        xlabel="x [mm]",
        ylabel="z [mm]",
        title="Flat-drive, low-recovery cilium gait",
    )
    axis.axis("equal")
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(PATH_PLOT_PATH, dpi=250)
    plt.close(fig)

    phase_degrees = np.arange(TABLE_SIZE)
    fig, axis = plt.subplots(figsize=(10, 5))
    axis.step(
        phase_degrees,
        lower_counts,
        where="post",
        label="Lower servo",
    )
    axis.step(
        phase_degrees,
        upper_counts,
        where="post",
        label="Upper servo",
    )
    axis.axvline(
        DRIVE_FRACTION * 360.0,
        color="black",
        linestyle="--",
        linewidth=1,
        label="Recovery begins",
    )
    axis.set(
        xlabel="Gait phase [degrees]",
        ylabel="PCA9685 OFF count",
        title="Raw PWM gait commands",
        xlim=(0, 359),
    )
    axis.grid(alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(COMMAND_PLOT_PATH, dpi=250)
    plt.close(fig)

    print("\n--- Generated gait ---")
    print(f"Samples: {TABLE_SIZE}")
    print(f"Links: {L1_MM:.1f} mm + {L2_MM:.1f} mm")
    print(
        f"Drive: {2.0 * HALF_STROKE_MM:.1f} mm at "
        f"z={DRIVE_HEIGHT_MM:.1f} mm for {100.0 * DRIVE_FRACTION:.0f}% "
        "of the cycle"
    )
    print(f"Recovery depth: {RECOVERY_DEPTH_MM:.1f} mm")
    print(
        f"Lower range: {lower_angles.min():.2f} to "
        f"{lower_angles.max():.2f} deg; counts "
        f"{lower_counts.min()} to {lower_counts.max()}"
    )
    print(
        f"Upper range: {upper_angles.min():.2f} to "
        f"{upper_angles.max():.2f} deg; counts "
        f"{upper_counts.min()} to {upper_counts.max()}"
    )
    print(f"Mean quantised tip error: {position_error.mean():.3f} mm")
    print(f"Maximum quantised tip error: {position_error.max():.3f} mm")
    print(
        "Drive-height peak-to-peak variation after quantisation: "
        f"{drive_height_peak_to_peak:.3f} mm"
    )
    print(f"Header: {HEADER_PATH}")
    print(f"Path plot: {PATH_PLOT_PATH}")
    print(f"Command plot: {COMMAND_PLOT_PATH}")


if __name__ == "__main__":
    generate()