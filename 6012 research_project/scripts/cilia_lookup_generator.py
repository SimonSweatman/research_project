
#!/usr/bin/env python3
"""
Generate a baseline cilium gait lookup table for Arduino and plot the expected tip motion.

Assumptions
-----------
1. Motion is planar in the x-z plane.
2. The lower joint angle is measured from +x, counter-clockwise.
3. The upper joint angle is RELATIVE to the lower link.
   So the absolute upper-link angle is theta1 + theta2_rel.
4. The lookup table stores SERVO COMMAND angles (0-180 deg), while the
   forward kinematics use calibrated joint angles derived from those commands.

Why the calibration parameters exist
------------------------------------
Your servo command angle is not automatically the same as the mechanical joint angle.
For example, a command of 90 deg might correspond to a truly upright link, or it might not.
So we map:

    mechanical angle = servo command + offset

using the offsets below.

Outputs
-------
- gait_table.h              Arduino-ready lookup tables
- phase_vs_angle.png        Joint angles vs phase
- tip_path_xz.png           Tip trajectory in x-z
- tip_vs_phase.png          x and z displacement vs phase
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# USER PARAMETERS
# ============================================================

# Link lengths (mm)
L1 = 45.0   # lower cilium segment length
L2 = 25.0   # upper cilium segment length

# Number of discrete phase samples across one full cycle
N = 720

# Baseline servo-command angles (deg)
# These match the earlier "good starting" whip-recovery idea.
LOWER_BACK      = 70.0
LOWER_FORWARD   = 95.0

UPPER_UPRIGHT   = 92.0
UPPER_STRIKE    = 100.0
UPPER_FOLDED    = 160.0

# Mechanical calibration offsets used ONLY for kinematics.
# They convert servo command angles to actual joint angles.
#
# Example meaning:
# if lower_servo_angle = 90 deg is physically vertical, then
# lower_mech_deg = lower_servo_angle + LOWER_CMD_TO_MECH_OFFSET
# should give 90 deg when the cilium is upright.
#
# The defaults below assume:
# - lower command 90 deg corresponds to vertical (90 deg mech)
# - upper command 90 deg corresponds to the upper link aligned with the lower link
LOWER_CMD_TO_MECH_OFFSET = 0.0
UPPER_CMD_TO_MECH_OFFSET = -90.0

# Output folder
OUT_DIR = Path(__file__).resolve().parent.parent / "include"

# ============================================================
# GAIT SHAPE DEFINITION
# ============================================================

def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))

def smoothstep(t: float) -> float:
    t = clamp(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)

def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t

def gait_servo_angles(phase: float) -> tuple[float, float]:
    """
    Return (lower_cmd_deg, upper_cmd_deg) for a phase in [0, 1).

    Shape:
    - Strike: lower moves forward, upper stays mostly upright
    - Whip:   upper snaps quickly to folded position
    - Recovery: lower returns while upper stays folded
    - Reset: upper comes back upright ready for next strike
    """
    phase = phase % 1.0

    # 1) Strike
    if phase < 0.40:
        u = smoothstep(phase / 0.40)
        lower = lerp(LOWER_BACK, LOWER_FORWARD, u)
        upper = lerp(UPPER_UPRIGHT, UPPER_STRIKE, u)

    # 2) Quick whip/fold
    elif phase < 0.47:
        u = smoothstep((phase - 0.40) / 0.07)
        lower = LOWER_FORWARD
        upper = lerp(UPPER_STRIKE, UPPER_FOLDED, u)

    # 3) Recovery while folded
    elif phase < 0.85:
        u = smoothstep((phase - 0.47) / 0.38)
        lower = lerp(LOWER_FORWARD, LOWER_BACK, u)
        upper = UPPER_FOLDED

    # 4) Reset upper joint
    else:
        u = smoothstep((phase - 0.85) / 0.15)
        lower = LOWER_BACK
        upper = lerp(UPPER_FOLDED, UPPER_UPRIGHT, u)

    return lower, upper

# ============================================================
# KINEMATICS
# ============================================================

def servo_to_mech(lower_cmd_deg: float, upper_cmd_deg: float) -> tuple[float, float]:
    """
    Convert servo command angles to mechanical joint angles for forward kinematics.

    lower_mech_deg:
        absolute angle of lower link from +x

    upper_rel_mech_deg:
        angle of upper link relative to lower link
    """
    lower_mech_deg = lower_cmd_deg + LOWER_CMD_TO_MECH_OFFSET
    upper_rel_mech_deg = upper_cmd_deg + UPPER_CMD_TO_MECH_OFFSET
    return lower_mech_deg, upper_rel_mech_deg

def tip_position_xz(lower_cmd_deg: float, upper_cmd_deg: float) -> tuple[float, float]:
    """
    Forward kinematics for a 2-link planar chain.

    x: horizontal displacement (mm)
    z: vertical displacement (mm)
    """
    lower_mech_deg, upper_rel_mech_deg = servo_to_mech(lower_cmd_deg, upper_cmd_deg)

    t1 = np.deg2rad(lower_mech_deg)
    t2 = np.deg2rad(upper_rel_mech_deg)

    x = L1 * np.cos(t1) + L2 * np.cos(t1 + t2)
    z = L1 * np.sin(t1) + L2 * np.sin(t1 + t2)
    return x, z

# ============================================================
# TABLE GENERATION
# ============================================================

phase_deg = np.arange(N, dtype=int)
phase_norm = phase_deg / N

lower_table = np.zeros(N, dtype=np.uint8)
upper_table = np.zeros(N, dtype=np.uint8)
x_tip = np.zeros(N, dtype=float)
z_tip = np.zeros(N, dtype=float)

for i, p in enumerate(phase_norm):
    lower_cmd, upper_cmd = gait_servo_angles(float(p))

    # clamp and round for Arduino table storage
    lower_cmd = int(round(np.clip(lower_cmd, 0, 180)))
    upper_cmd = int(round(np.clip(upper_cmd, 0, 180)))

    lower_table[i] = lower_cmd
    upper_table[i] = upper_cmd

    x_tip[i], z_tip[i] = tip_position_xz(lower_cmd, upper_cmd)

# Ensure the last phase joins smoothly to the first
# (table is 0..359, and phase wraps back to 0)
print("First sample:", lower_table[0], upper_table[0])
print("Last sample :", lower_table[-1], upper_table[-1])

# ============================================================
# PLOTS
# ============================================================

# Tip path in x-z
plt.figure(figsize=(6, 6))
plt.plot(x_tip, z_tip)
plt.scatter([x_tip[0]], [z_tip[0]], label="Start")
plt.xlabel("x displacement (mm)")
plt.ylabel("z displacement (mm)")
plt.title("Expected tip path in x-z plane")
plt.axis("equal")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUT_DIR / "tip_path_xz.png", dpi=200)
plt.close()

# ============================================================
# EXPORT ARDUINO HEADER
# ============================================================

header_path = OUT_DIR / "gait_table.h"

def format_array(name: str, arr: np.ndarray) -> str:
    lines = []
    line = []
    for i, v in enumerate(arr):
        line.append(f"{int(v):3d}")
        if len(line) == 16 or i == len(arr) - 1:
            lines.append("    " + ", ".join(line))
            line = []
    body = ",\n".join(lines)
    return f"const uint8_t {name}[{len(arr)}] PROGMEM = {{\n{body}\n}};\n"

header = []
header.append("#pragma once")
header.append("#include <Arduino.h>")
header.append("")
header.append(f"const uint16_t GAIT_TABLE_SIZE = {N};")
header.append("")
header.append(format_array("LOWER_TABLE", lower_table))
header.append("")
header.append(format_array("UPPER_TABLE", upper_table))

header_path.write_text("\n".join(header), encoding="utf-8")

print(f"Saved: {header_path}")
print(f"Saved: {OUT_DIR / 'phase_vs_angle.png'}")
print(f"Saved: {OUT_DIR / 'tip_path_xz.png'}")
print(f"Saved: {OUT_DIR / 'tip_vs_phase.png'}")
