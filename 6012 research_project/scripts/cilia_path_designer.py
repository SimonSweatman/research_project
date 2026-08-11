#!/usr/bin/env python3
"""Interactive path designer for a two-link artificial cilium.

The program uses the same angle convention as ``d_shape.py``:

* the lower servo command is the absolute lower-link angle;
* an upper servo command of 90 degrees makes both links collinear;
* upper commands from 0 to 180 degrees therefore represent relative joint
  angles from -90 to +90 degrees.

The tip can be moved by dragging it on the canvas, by changing its X/Y
sliders, or by changing the two servo-angle sliders.  Points can be saved
manually or captured continuously while Record Trace is enabled.  Signed-height
curves and tangent circular corner fillets can also be sampled directly into
the path.  The resulting path can be exported as coordinates, joint-angle
tables, or an Arduino-style header.  A second view simulates a phased array of
up to 12 cilia and checks the 50 x 7.5 mm paddles for a 2 mm safety clearance.
"""

from __future__ import annotations

import bisect
import csv
import math
import threading
import time
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


# ---------------------------------------------------------------------------
# Mechanical settings
# ---------------------------------------------------------------------------

L1_MM = 50.0
L2_MM = 50.0
MIN_REACH_MM = math.sqrt(L1_MM * L1_MM + L2_MM * L2_MM)

LOWER_MIN_DEG = 0.0
LOWER_MAX_DEG = 180.0
UPPER_MIN_DEG = 0.0
UPPER_MAX_DEG = 180.0

# mechanical relative angle = upper servo command + this offset
UPPER_COMMAND_TO_MECHANICAL_OFFSET_DEG = -90.0

# Experimentally fitted common conversion from the manual measurements of
# lower and upper servos on both PCA9685 boards:
#
#     pwm_count = 304.47 + 2.35018 * (angle_degrees - 90)
#
# Exporting calibrated raw counts lets main.cpp continue using setPwm() and
# lets intermediate IK angles use every available PWM count.
GAIT_PWM_FREQUENCY_HZ = 50
PWM_AT_90_COUNT = 304.47
PWM_COUNTS_PER_DEG = 2.35018
PWM_ZERO_DEG_COUNT = PWM_AT_90_COUNT - 90.0 * PWM_COUNTS_PER_DEG
PWM_MIN_COUNT = 0
PWM_MAX_COUNT = 4095

DESIGNER_X_MIN_MM = -110.0
DESIGNER_X_MAX_MM = 110.0
DESIGNER_Y_MIN_MM = -5.0
DESIGNER_Y_MAX_MM = 110.0

PADDLE_WIDTH_MM = 7.5
COLLISION_MARGIN_MM = 2.0
DRIVE_HEIGHT_TOLERANCE_MM = 0.75
ARRAY_MIN_SPACING_MM = 34.0
ARRAY_MAX_SPACING_MM = 150.0
ARRAY_MAX_CILIA = 12
ARRAY_CYCLE_CHECK_SAMPLES = 360


@dataclass(frozen=True)
class PathPoint:
    """One sampled point and its matching servo commands."""

    x_mm: float
    y_mm: float
    lower_deg: float
    upper_deg: float
    source: str = "trace"


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def forward_kinematics(lower_deg: float, upper_deg: float) -> tuple[float, float]:
    """Return tip X/Y coordinates in millimetres."""

    q1 = math.radians(lower_deg)
    q2 = math.radians(upper_deg + UPPER_COMMAND_TO_MECHANICAL_OFFSET_DEG)
    x_mm = L1_MM * math.cos(q1) + L2_MM * math.cos(q1 + q2)
    y_mm = L1_MM * math.sin(q1) + L2_MM * math.sin(q1 + q2)
    return x_mm, y_mm


def elbow_position(lower_deg: float) -> tuple[float, float]:
    q1 = math.radians(lower_deg)
    return L1_MM * math.cos(q1), L1_MM * math.sin(q1)


def oriented_rectangle(
    start: tuple[float, float],
    end: tuple[float, float],
    width_mm: float = PADDLE_WIDTH_MM,
    expansion_mm: float = 0.0,
) -> list[tuple[float, float]]:
    """Return four corners for a rectangular paddle and optional envelope."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    if length <= 1e-12:
        return [start, start, start, start]

    ux = dx / length
    uy = dy / length
    px = -uy
    py = ux
    half_width = width_mm / 2.0 + expansion_mm
    start_x = start[0] - ux * expansion_mm
    start_y = start[1] - uy * expansion_mm
    end_x = end[0] + ux * expansion_mm
    end_y = end[1] + uy * expansion_mm

    return [
        (start_x + px * half_width, start_y + py * half_width),
        (end_x + px * half_width, end_y + py * half_width),
        (end_x - px * half_width, end_y - py * half_width),
        (start_x - px * half_width, start_y - py * half_width),
    ]


def _polygon_axes(
    polygon: list[tuple[float, float]],
) -> list[tuple[float, float]]:
    axes = []
    for first, second in zip(polygon, polygon[1:] + polygon[:1]):
        dx = second[0] - first[0]
        dy = second[1] - first[1]
        length = math.hypot(dx, dy)
        if length > 1e-12:
            axes.append((-dy / length, dx / length))
    return axes


def rectangles_intersect(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> bool:
    """Separating-axis intersection test for two oriented rectangles."""

    for axis_x, axis_y in _polygon_axes(first) + _polygon_axes(second):
        first_projection = [x * axis_x + y * axis_y for x, y in first]
        second_projection = [x * axis_x + y * axis_y for x, y in second]
        if max(first_projection) < min(second_projection) - 1e-9:
            return False
        if max(second_projection) < min(first_projection) - 1e-9:
            return False
    return True


def _point_segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    denominator = dx * dx + dy * dy
    if denominator <= 1e-12:
        return math.hypot(point[0] - start[0], point[1] - start[1])
    fraction = (
        (point[0] - start[0]) * dx + (point[1] - start[1]) * dy
    ) / denominator
    fraction = clamp(fraction, 0.0, 1.0)
    closest_x = start[0] + fraction * dx
    closest_y = start[1] + fraction * dy
    return math.hypot(point[0] - closest_x, point[1] - closest_y)


def rectangle_distance(
    first: list[tuple[float, float]],
    second: list[tuple[float, float]],
) -> float:
    """Minimum edge distance between two oriented rectangles."""

    if rectangles_intersect(first, second):
        return 0.0

    minimum = float("inf")
    first_edges = list(zip(first, first[1:] + first[:1]))
    second_edges = list(zip(second, second[1:] + second[:1]))
    for point in first:
        for start, end in second_edges:
            minimum = min(minimum, _point_segment_distance(point, start, end))
    for point in second:
        for start, end in first_edges:
            minimum = min(minimum, _point_segment_distance(point, start, end))
    return minimum


def inverse_kinematics(
    x_mm: float,
    y_mm: float,
    reference_angles: tuple[float, float] = (90.0, 90.0),
) -> tuple[float, float] | None:
    """Find a valid IK solution closest to ``reference_angles``.

    Both elbow branches are tested.  A result is returned only when both
    resulting servo commands lie within their nominal 0--180 degree arcs.
    """

    radius_squared = x_mm * x_mm + y_mm * y_mm
    cos_q2 = (
        radius_squared - L1_MM * L1_MM - L2_MM * L2_MM
    ) / (2.0 * L1_MM * L2_MM)

    if cos_q2 < -1.0 - 1e-9 or cos_q2 > 1.0 + 1e-9:
        return None

    cos_q2 = clamp(cos_q2, -1.0, 1.0)
    q2_magnitude = math.acos(cos_q2)
    candidates: list[tuple[float, float]] = []

    for q2 in (q2_magnitude, -q2_magnitude):
        q1 = math.atan2(y_mm, x_mm) - math.atan2(
            L2_MM * math.sin(q2),
            L1_MM + L2_MM * math.cos(q2),
        )
        lower_deg = math.degrees(q1)
        upper_deg = (
            math.degrees(q2) - UPPER_COMMAND_TO_MECHANICAL_OFFSET_DEG
        )

        if (
            LOWER_MIN_DEG - 1e-7 <= lower_deg <= LOWER_MAX_DEG + 1e-7
            and UPPER_MIN_DEG - 1e-7 <= upper_deg <= UPPER_MAX_DEG + 1e-7
        ):
            candidates.append(
                (
                    clamp(lower_deg, LOWER_MIN_DEG, LOWER_MAX_DEG),
                    clamp(upper_deg, UPPER_MIN_DEG, UPPER_MAX_DEG),
                )
            )

    if not candidates:
        return None

    lower_reference, upper_reference = reference_angles
    return min(
        candidates,
        key=lambda angles: (
            (angles[0] - lower_reference) ** 2
            + (angles[1] - upper_reference) ** 2
        ),
    )


def interpolate_path_point(
    first: PathPoint,
    second: PathPoint,
    fraction: float,
    reference_angles: tuple[float, float],
) -> PathPoint:
    x_mm = first.x_mm + (second.x_mm - first.x_mm) * fraction
    y_mm = first.y_mm + (second.y_mm - first.y_mm) * fraction
    angles = inverse_kinematics(x_mm, y_mm, reference_angles)
    if angles is None:
        raise ValueError(
            f"Interpolated point ({x_mm:.3f}, {y_mm:.3f}) mm is unreachable."
        )
    return PathPoint(x_mm, y_mm, angles[0], angles[1], "resampled")


def resample_polyline(points: list[PathPoint], sample_count: int) -> list[PathPoint]:
    """Resample a drawn polyline at approximately equal spatial intervals."""

    if sample_count < 2:
        raise ValueError("Sample count must be at least 2.")
    if len(points) < 2:
        raise ValueError("At least two path points are required.")

    cleaned = [points[0]]
    for point in points[1:]:
        if math.hypot(
            point.x_mm - cleaned[-1].x_mm,
            point.y_mm - cleaned[-1].y_mm,
        ) > 1e-9:
            cleaned.append(point)

    if len(cleaned) < 2:
        raise ValueError("The path has no measurable length.")

    cumulative = [0.0]
    for previous, current in zip(cleaned, cleaned[1:]):
        cumulative.append(
            cumulative[-1]
            + math.hypot(
                current.x_mm - previous.x_mm,
                current.y_mm - previous.y_mm,
            )
        )

    total_length = cumulative[-1]
    closed = math.hypot(
        cleaned[-1].x_mm - cleaned[0].x_mm,
        cleaned[-1].y_mm - cleaned[0].y_mm,
    ) < 1e-6

    if closed:
        targets = [total_length * index / sample_count for index in range(sample_count)]
    else:
        targets = [
            total_length * index / (sample_count - 1)
            for index in range(sample_count)
        ]

    result: list[PathPoint] = []
    reference = (cleaned[0].lower_deg, cleaned[0].upper_deg)

    for target in targets:
        segment_index = max(0, bisect.bisect_right(cumulative, target) - 1)
        segment_index = min(segment_index, len(cleaned) - 2)
        segment_start = cumulative[segment_index]
        segment_end = cumulative[segment_index + 1]
        fraction = (
            0.0
            if segment_end == segment_start
            else (target - segment_start) / (segment_end - segment_start)
        )
        point = interpolate_path_point(
            cleaned[segment_index],
            cleaned[segment_index + 1],
            fraction,
            reference,
        )
        result.append(point)
        reference = (point.lower_deg, point.upper_deg)

    return result


def quadratic_arc_coordinates(
    start: tuple[float, float],
    end: tuple[float, float],
    midpoint_height_mm: float,
    sample_count: int,
) -> list[tuple[float, float]]:
    """Sample a quadratic curve with a signed perpendicular midpoint height."""

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    chord_length = math.hypot(dx, dy)
    if chord_length <= 1e-9:
        raise ValueError("The curved segment needs two different endpoints.")
    sample_count = max(2, sample_count)

    midpoint_x = (start[0] + end[0]) / 2.0
    midpoint_y = (start[1] + end[1]) / 2.0
    normal_x = -dy / chord_length
    normal_y = dx / chord_length

    # For a quadratic Bezier, placing the control point at twice the requested
    # offset makes the curve itself pass through the requested midpoint height.
    control_x = midpoint_x + 2.0 * midpoint_height_mm * normal_x
    control_y = midpoint_y + 2.0 * midpoint_height_mm * normal_y

    coordinates: list[tuple[float, float]] = []
    for index in range(sample_count):
        fraction = index / (sample_count - 1)
        inverse = 1.0 - fraction
        x_mm = (
            inverse * inverse * start[0]
            + 2.0 * inverse * fraction * control_x
            + fraction * fraction * end[0]
        )
        y_mm = (
            inverse * inverse * start[1]
            + 2.0 * inverse * fraction * control_y
            + fraction * fraction * end[1]
        )
        coordinates.append((x_mm, y_mm))
    return coordinates


def circular_fillet_coordinates(
    first: tuple[float, float],
    corner: tuple[float, float],
    last: tuple[float, float],
    requested_radius_mm: float,
    sample_spacing_mm: float,
) -> tuple[list[tuple[float, float]], float]:
    """Return a tangent circular fillet for the corner ``first-corner-last``."""

    if requested_radius_mm <= 0.0:
        raise ValueError("The fillet radius must be greater than zero.")

    first_dx = first[0] - corner[0]
    first_dy = first[1] - corner[1]
    last_dx = last[0] - corner[0]
    last_dy = last[1] - corner[1]
    first_length = math.hypot(first_dx, first_dy)
    last_length = math.hypot(last_dx, last_dy)
    if first_length <= 1e-9 or last_length <= 1e-9:
        raise ValueError("The three fillet points must be different.")

    first_unit = (first_dx / first_length, first_dy / first_length)
    last_unit = (last_dx / last_length, last_dy / last_length)
    dot = clamp(
        first_unit[0] * last_unit[0] + first_unit[1] * last_unit[1],
        -1.0,
        1.0,
    )
    interior_angle = math.acos(dot)
    if interior_angle <= math.radians(1.0):
        raise ValueError("This corner reverses too sharply to create a stable fillet.")
    if interior_angle >= math.radians(179.0):
        raise ValueError("These three points are almost straight; no fillet is needed.")

    tangent_factor = math.tan(interior_angle / 2.0)
    requested_tangent_distance = requested_radius_mm / tangent_factor
    maximum_tangent_distance = 0.49 * min(first_length, last_length)
    tangent_distance = min(requested_tangent_distance, maximum_tangent_distance)
    effective_radius = tangent_distance * tangent_factor

    tangent_start = (
        corner[0] + first_unit[0] * tangent_distance,
        corner[1] + first_unit[1] * tangent_distance,
    )
    tangent_end = (
        corner[0] + last_unit[0] * tangent_distance,
        corner[1] + last_unit[1] * tangent_distance,
    )

    bisector_x = first_unit[0] + last_unit[0]
    bisector_y = first_unit[1] + last_unit[1]
    bisector_length = math.hypot(bisector_x, bisector_y)
    if bisector_length <= 1e-9:
        raise ValueError("These points do not define a usable corner fillet.")
    bisector_x /= bisector_length
    bisector_y /= bisector_length
    centre_distance = effective_radius / math.sin(interior_angle / 2.0)
    centre = (
        corner[0] + bisector_x * centre_distance,
        corner[1] + bisector_y * centre_distance,
    )

    start_angle = math.atan2(
        tangent_start[1] - centre[1], tangent_start[0] - centre[0]
    )
    end_angle = math.atan2(
        tangent_end[1] - centre[1], tangent_end[0] - centre[0]
    )
    incoming = (-first_unit[0], -first_unit[1])
    turn_cross = incoming[0] * last_unit[1] - incoming[1] * last_unit[0]
    if turn_cross > 0.0:
        while end_angle <= start_angle:
            end_angle += 2.0 * math.pi
    else:
        while end_angle >= start_angle:
            end_angle -= 2.0 * math.pi

    sweep = end_angle - start_angle
    arc_length = abs(sweep) * effective_radius
    spacing = max(0.05, sample_spacing_mm)
    sample_count = max(3, int(math.ceil(arc_length / spacing)) + 1)
    coordinates: list[tuple[float, float]] = []
    for index in range(sample_count):
        fraction = index / (sample_count - 1)
        angle = start_angle + sweep * fraction
        coordinates.append(
            (
                centre[0] + effective_radius * math.cos(angle),
                centre[1] + effective_radius * math.sin(angle),
            )
        )
    return coordinates, effective_radius


class CiliaPathDesigner(tk.Tk):
    """Tkinter user interface for interactive cilium path design."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Two-Link Cilia Path Designer")
        self.geometry("1260x820")
        self.minsize(1000, 680)

        self.lower_deg = 90.0
        self.upper_deg = 90.0
        self.tip_x_mm, self.tip_y_mm = forward_kinematics(
            self.lower_deg, self.upper_deg
        )

        self.path_points: list[PathPoint] = []
        self.history: list[list[PathPoint]] = []
        self.recording = False
        self.playing = False
        self.playback_points: list[PathPoint] = []
        self.playback_index = 0
        self.playback_after_id: str | None = None
        self._updating_controls = False
        self.arc_preview_active = False
        self.fillet_preview_active = False

        self.lower_var = tk.DoubleVar(value=self.lower_deg)
        self.upper_var = tk.DoubleVar(value=self.upper_deg)
        self.x_var = tk.DoubleVar(value=self.tip_x_mm)
        self.y_var = tk.DoubleVar(value=self.tip_y_mm)
        self.lower_text = tk.StringVar()
        self.upper_text = tk.StringVar()
        self.x_text = tk.StringVar()
        self.y_text = tk.StringVar()
        self.status_var = tk.StringVar(value="Ready. Drag anywhere on the canvas to move the tip.")
        self.record_button_text = tk.StringVar(value="Start live trace")
        self.path_summary_var = tk.StringVar(value="Path: 0 points")
        self.trace_spacing_var = tk.DoubleVar(value=0.25)
        self.arc_height_var = tk.DoubleVar(value=10.0)
        self.arc_height_text = tk.StringVar(value="10.00")
        self.fillet_radius_var = tk.DoubleVar(value=5.0)
        self.fillet_radius_text = tk.StringVar(value="5.00")
        self.export_format_var = tk.StringVar(value="Coordinates CSV")
        self.sample_count_var = tk.IntVar(value=360)
        self.resample_var = tk.BooleanVar(value=True)
        self.playback_duration_var = tk.DoubleVar(value=5.0)
        self.loop_playback_var = tk.BooleanVar(value=False)

        # Independent view transforms prevent either settings panel from
        # changing the scale of its drawing canvas.
        self.designer_zoom = 1.0
        self.designer_pan_x_mm = 0.0
        self.designer_pan_y_mm = 0.0
        self._designer_pan_anchor: tuple[int, int, float, float] | None = None

        # Array simulator state.  The angle path is copied explicitly from the
        # designer so later edits cannot silently change an active analysis.
        self.array_angle_path: list[PathPoint] = []
        self.array_cilia_count_var = tk.IntVar(value=12)
        self.array_spacing_var = tk.DoubleVar(value=34.0)
        self.array_phase_shift_var = tk.DoubleVar(value=0.0)
        self.array_spacing_text = tk.StringVar(value="34.00")
        self.array_phase_text = tk.StringVar(value="0.00")
        self.array_duration_var = tk.DoubleVar(value=5.0)
        self.array_loop_var = tk.BooleanVar(value=True)
        self.array_show_traces_var = tk.BooleanVar(value=True)
        self.array_show_envelopes_var = tk.BooleanVar(value=False)
        self.array_status_var = tk.StringVar(
            value="Load the current path, then adjust spacing and phase."
        )
        self.array_playing = False
        self.array_after_id: str | None = None
        self.array_global_phase = 0.0
        self.array_playback_started_ms = 0
        self.array_zoom = 1.0
        self.array_pan_x_mm = 0.0
        self.array_pan_y_mm = 0.0
        self._array_pan_anchor: tuple[int, int, float, float] | None = None
        self.safety_map_running = False

        self._build_interface()
        self._update_value_labels()
        self._draw_scene()

        self.bind("<Control-z>", lambda _event: self.undo())
        self.bind("<Control-s>", lambda _event: self.save_coordinate())
        self.bind("<Control-e>", lambda _event: self.export_path())
        self.bind("<Key-r>", lambda _event: self.toggle_recording())

    # ------------------------------------------------------------------ UI

    def _build_interface(self) -> None:
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True)
        designer_page = ttk.Frame(self.notebook)
        array_page = ttk.Frame(self.notebook)
        self.notebook.add(designer_page, text="Path Designer")
        self.notebook.add(array_page, text="Array Simulator")

        outer = ttk.Frame(designer_page, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(0, weight=1)

        canvas_frame = ttk.LabelFrame(outer, text="Side view", padding=5)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(
            canvas_frame,
            background="#f7f8fa",
            highlightthickness=0,
            cursor="crosshair",
        )
        self.canvas.grid(row=0, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", lambda _event: self._draw_scene())
        self.canvas.bind("<Button-1>", self._canvas_move_tip)
        self.canvas.bind("<B1-Motion>", self._canvas_move_tip)
        self.canvas.bind("<MouseWheel>", self._zoom_designer_view)
        self.canvas.bind("<ButtonPress-2>", self._start_designer_pan)
        self.canvas.bind("<B2-Motion>", self._pan_designer_view)
        self.canvas.bind("<ButtonPress-3>", self._start_designer_pan)
        self.canvas.bind("<B3-Motion>", self._pan_designer_view)

        # Keep the settings column at a fixed width so its contents cannot
        # resize the graph.  Only the inner settings frame scrolls vertically.
        sidebar_container = ttk.Frame(outer, width=340)
        sidebar_container.grid(row=0, column=1, sticky="ns")
        sidebar_container.grid_propagate(False)
        sidebar_container.rowconfigure(0, weight=1)
        sidebar_container.columnconfigure(0, weight=1)

        self.sidebar_canvas = tk.Canvas(
            sidebar_container,
            width=318,
            highlightthickness=0,
            borderwidth=0,
            background=self.cget("background"),
        )
        self.sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        sidebar_scrollbar = ttk.Scrollbar(
            sidebar_container,
            orient="vertical",
            command=self.sidebar_canvas.yview,
        )
        sidebar_scrollbar.grid(row=0, column=1, sticky="ns")
        self.sidebar_canvas.configure(yscrollcommand=sidebar_scrollbar.set)

        sidebar = ttk.Frame(self.sidebar_canvas)
        self.sidebar_window_id = self.sidebar_canvas.create_window(
            (0, 0),
            window=sidebar,
            anchor="nw",
        )
        sidebar.bind("<Configure>", self._update_sidebar_scroll_region)
        self.sidebar_canvas.bind("<Configure>", self._resize_sidebar_contents)
        self.bind("<MouseWheel>", self._scroll_sidebar_with_mouse)

        ttk.Label(
            sidebar,
            text="Link lengths: 50 mm + 50 mm",
            font=("Segoe UI", 10, "bold"),
        ).pack(anchor="w", pady=(0, 8))

        angle_frame = ttk.LabelFrame(sidebar, text="Servo commands", padding=10)
        angle_frame.pack(fill="x", pady=(0, 8))
        self._add_slider(
            angle_frame,
            "Lower servo",
            self.lower_var,
            LOWER_MIN_DEG,
            LOWER_MAX_DEG,
            self.lower_text,
            self._angles_changed,
            lambda: self._typed_value_changed("lower"),
            "deg",
        )
        self._add_slider(
            angle_frame,
            "Upper servo",
            self.upper_var,
            UPPER_MIN_DEG,
            UPPER_MAX_DEG,
            self.upper_text,
            self._angles_changed,
            lambda: self._typed_value_changed("upper"),
            "deg",
        )

        coordinate_frame = ttk.LabelFrame(sidebar, text="Tip coordinates", padding=10)
        coordinate_frame.pack(fill="x", pady=(0, 8))
        self._add_slider(
            coordinate_frame,
            "X position",
            self.x_var,
            -100.0,
            100.0,
            self.x_text,
            self._coordinates_changed,
            lambda: self._typed_value_changed("x"),
            "mm",
        )
        self._add_slider(
            coordinate_frame,
            "Y position",
            self.y_var,
            -100.0,
            100.0,
            self.y_text,
            self._coordinates_changed,
            lambda: self._typed_value_changed("y"),
            "mm",
        )

        path_frame = ttk.LabelFrame(sidebar, text="Path recording", padding=10)
        path_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(
            path_frame,
            text="Save coordinate  (Ctrl+S)",
            command=self.save_coordinate,
        ).pack(fill="x", pady=(0, 5))
        ttk.Button(
            path_frame,
            text="Snap to final path point",
            command=self.snap_to_last_point,
        ).pack(fill="x", pady=(0, 5))
        self.record_button = ttk.Button(
            path_frame,
            textvariable=self.record_button_text,
            command=self.toggle_recording,
        )
        self.record_button.pack(fill="x", pady=(0, 5))

        button_row = ttk.Frame(path_frame)
        button_row.pack(fill="x", pady=(0, 5))
        ttk.Button(button_row, text="Undo  (Ctrl+Z)", command=self.undo).pack(
            side="left", fill="x", expand=True, padx=(0, 3)
        )
        ttk.Button(button_row, text="Clear", command=self.clear_path).pack(
            side="left", fill="x", expand=True, padx=(3, 0)
        )

        spacing_row = ttk.Frame(path_frame)
        spacing_row.pack(fill="x")
        ttk.Label(spacing_row, text="Trace spacing (mm)").pack(side="left")
        ttk.Spinbox(
            spacing_row,
            from_=0.05,
            to=10.0,
            increment=0.05,
            width=7,
            textvariable=self.trace_spacing_var,
        ).pack(side="right")
        ttk.Label(path_frame, textvariable=self.path_summary_var).pack(
            anchor="w", pady=(6, 0)
        )

        curve_frame = ttk.LabelFrame(sidebar, text="Curves and corner fillets", padding=10)
        curve_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(
            curve_frame,
            text=(
                "Arc: move the tip to the endpoint, then preview from the "
                "final path point. Signed height selects the curve direction."
            ),
            wraplength=285,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))
        self._add_slider(
            curve_frame,
            "Arc midpoint height",
            self.arc_height_var,
            -50.0,
            50.0,
            self.arc_height_text,
            self._arc_height_changed,
            lambda: self._typed_curve_value_changed("arc"),
            "mm",
        )
        arc_buttons = ttk.Frame(curve_frame)
        arc_buttons.pack(fill="x", pady=(0, 7))
        ttk.Button(
            arc_buttons,
            text="Preview arc",
            command=self.preview_arc_segment,
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(
            arc_buttons,
            text="Add arc",
            command=self.apply_arc_segment,
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        ttk.Separator(curve_frame, orient="horizontal").pack(fill="x", pady=(0, 7))
        ttk.Label(
            curve_frame,
            text=(
                "Fillet: save three consecutive straight-line points. The "
                "middle point is replaced by a tangent circular corner."
            ),
            wraplength=285,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))
        self._add_slider(
            curve_frame,
            "Fillet radius",
            self.fillet_radius_var,
            0.1,
            40.0,
            self.fillet_radius_text,
            self._fillet_radius_changed,
            lambda: self._typed_curve_value_changed("fillet"),
            "mm",
        )
        fillet_buttons = ttk.Frame(curve_frame)
        fillet_buttons.pack(fill="x", pady=(0, 6))
        ttk.Button(
            fillet_buttons,
            text="Preview final corner",
            command=self.preview_final_corner_fillet,
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(
            fillet_buttons,
            text="Apply fillet",
            command=self.apply_final_corner_fillet,
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))
        ttk.Button(
            curve_frame,
            text="Cancel curve preview",
            command=self.cancel_curve_preview,
        ).pack(fill="x")

        playback_frame = ttk.LabelFrame(sidebar, text="Path simulation", padding=10)
        playback_frame.pack(fill="x", pady=(0, 8))
        playback_settings = ttk.Frame(playback_frame)
        playback_settings.pack(fill="x", pady=(0, 6))
        ttk.Label(playback_settings, text="Duration (seconds)").pack(side="left")
        ttk.Spinbox(
            playback_settings,
            from_=0.5,
            to=120.0,
            increment=0.5,
            width=7,
            textvariable=self.playback_duration_var,
        ).pack(side="right")
        ttk.Checkbutton(
            playback_frame,
            text="Loop continuously",
            variable=self.loop_playback_var,
        ).pack(anchor="w", pady=(0, 6))
        playback_buttons = ttk.Frame(playback_frame)
        playback_buttons.pack(fill="x")
        ttk.Button(
            playback_buttons,
            text="Play",
            command=self.play_path,
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(
            playback_buttons,
            text="Stop",
            command=self.stop_playback,
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))
        ttk.Button(
            playback_frame,
            text="Reset zoom and pan",
            command=self._reset_designer_view,
        ).pack(fill="x", pady=(6, 0))
        ttk.Label(
            playback_frame,
            text="Mouse wheel: zoom   |   Middle/right-button drag: pan",
        ).pack(anchor="w", pady=(5, 0))

        export_frame = ttk.LabelFrame(sidebar, text="Export", padding=10)
        export_frame.pack(fill="x", pady=(0, 8))
        ttk.Combobox(
            export_frame,
            state="readonly",
            textvariable=self.export_format_var,
            values=(
                "Coordinates CSV",
                "Joint angles CSV",
                "Arduino PWM header",
            ),
        ).pack(fill="x", pady=(0, 6))

        samples_row = ttk.Frame(export_frame)
        samples_row.pack(fill="x", pady=(0, 4))
        ttk.Label(samples_row, text="Lookup samples").pack(side="left")
        ttk.Spinbox(
            samples_row,
            from_=2,
            to=5000,
            increment=1,
            width=8,
            textvariable=self.sample_count_var,
        ).pack(side="right")
        ttk.Checkbutton(
            export_frame,
            text="Resample uniformly along path",
            variable=self.resample_var,
        ).pack(anchor="w", pady=(0, 6))
        ttk.Button(
            export_frame,
            text="Export path  (Ctrl+E)",
            command=self.export_path,
        ).pack(fill="x")

        status_frame = ttk.LabelFrame(sidebar, text="Status", padding=8)
        status_frame.pack(fill="both", expand=True)
        tk.Label(
            status_frame,
            textvariable=self.status_var,
            wraplength=290,
            justify="left",
            anchor="nw",
            height=5,
            background=self.cget("background"),
        ).pack(fill="x", anchor="nw")

        self._build_array_interface(array_page)

    def _build_array_interface(self, parent: ttk.Frame) -> None:
        outer = ttk.Frame(parent, padding=10)
        outer.pack(fill="both", expand=True)
        outer.columnconfigure(0, weight=1)
        outer.columnconfigure(1, weight=0)
        outer.rowconfigure(0, weight=1)

        canvas_frame = ttk.LabelFrame(outer, text="Cilia array side view", padding=5)
        canvas_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)
        self.array_canvas = tk.Canvas(
            canvas_frame,
            background="#f7f8fa",
            highlightthickness=0,
            cursor="fleur",
        )
        self.array_canvas.grid(row=0, column=0, sticky="nsew")
        self.array_canvas.bind("<Configure>", lambda _event: self._draw_array_scene())
        self.array_canvas.bind("<MouseWheel>", self._zoom_array_view)
        self.array_canvas.bind("<ButtonPress-1>", self._start_array_pan)
        self.array_canvas.bind("<B1-Motion>", self._pan_array_view)
        self.array_canvas.bind("<ButtonPress-2>", self._start_array_pan)
        self.array_canvas.bind("<B2-Motion>", self._pan_array_view)
        self.array_canvas.bind("<ButtonPress-3>", self._start_array_pan)
        self.array_canvas.bind("<B3-Motion>", self._pan_array_view)

        sidebar_container = ttk.Frame(outer, width=350)
        sidebar_container.grid(row=0, column=1, sticky="ns")
        sidebar_container.grid_propagate(False)
        sidebar_container.rowconfigure(0, weight=1)
        sidebar_container.columnconfigure(0, weight=1)
        self.array_sidebar_canvas = tk.Canvas(
            sidebar_container,
            width=328,
            highlightthickness=0,
            borderwidth=0,
            background=self.cget("background"),
        )
        self.array_sidebar_canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            sidebar_container,
            orient="vertical",
            command=self.array_sidebar_canvas.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.array_sidebar_canvas.configure(yscrollcommand=scrollbar.set)
        sidebar = ttk.Frame(self.array_sidebar_canvas)
        self.array_sidebar_window_id = self.array_sidebar_canvas.create_window(
            (0, 0), window=sidebar, anchor="nw"
        )
        sidebar.bind("<Configure>", self._update_array_sidebar_scroll_region)
        self.array_sidebar_canvas.bind(
            "<Configure>", self._resize_array_sidebar_contents
        )

        navigation = ttk.Frame(sidebar)
        navigation.pack(fill="x", pady=(0, 8))
        ttk.Button(
            navigation,
            text="Back to Path Designer",
            command=lambda: self.notebook.select(0),
        ).pack(side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(
            navigation,
            text="Use current path",
            command=self.load_current_path_into_array,
        ).pack(side="left", fill="x", expand=True, padx=(3, 0))

        count_frame = ttk.LabelFrame(sidebar, text="Number of cilia", padding=10)
        count_frame.pack(fill="x", pady=(0, 8))
        count_row = ttk.Frame(count_frame)
        count_row.pack(fill="x")
        ttk.Button(count_row, text="Remove", command=self.remove_array_cilium).pack(
            side="left", fill="x", expand=True, padx=(0, 4)
        )
        ttk.Label(
            count_row,
            textvariable=self.array_cilia_count_var,
            width=5,
            anchor="center",
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")
        ttk.Button(count_row, text="Add", command=self.add_array_cilium).pack(
            side="left", fill="x", expand=True, padx=(4, 0)
        )
        ttk.Label(count_frame, text="Allowed range: 1 to 12").pack(
            anchor="w", pady=(5, 0)
        )

        geometry_frame = ttk.LabelFrame(sidebar, text="Array geometry", padding=10)
        geometry_frame.pack(fill="x", pady=(0, 8))
        self._add_slider(
            geometry_frame,
            "Pivot spacing",
            self.array_spacing_var,
            ARRAY_MIN_SPACING_MM,
            ARRAY_MAX_SPACING_MM,
            self.array_spacing_text,
            self._array_controls_changed,
            lambda: self._typed_array_value_changed("spacing"),
            "mm",
        )
        self._add_slider(
            geometry_frame,
            "Adjacent phase shift",
            self.array_phase_shift_var,
            0.0,
            360.0,
            self.array_phase_text,
            self._array_controls_changed,
            lambda: self._typed_array_value_changed("phase"),
            "deg",
        )
        ttk.Label(
            geometry_frame,
            text=(
                "Spacing is measured between neighbouring lower-pivot centres. "
                "Phase shifts above 180 degrees reverse the apparent wave direction."
            ),
            wraplength=300,
            justify="left",
        ).pack(anchor="w")

        display_frame = ttk.LabelFrame(sidebar, text="Display", padding=10)
        display_frame.pack(fill="x", pady=(0, 8))
        ttk.Checkbutton(
            display_frame,
            text="Show translated tip traces",
            variable=self.array_show_traces_var,
            command=self._draw_array_scene,
        ).pack(anchor="w")
        ttk.Checkbutton(
            display_frame,
            text="Show 2 mm safety envelopes",
            variable=self.array_show_envelopes_var,
            command=self._draw_array_scene,
        ).pack(anchor="w", pady=(3, 6))
        ttk.Button(
            display_frame,
            text="Reset zoom and pan",
            command=self._reset_array_view,
        ).pack(fill="x")
        ttk.Label(
            display_frame,
            text="Mouse wheel: zoom   |   Left, middle or right drag: pan",
        ).pack(anchor="w", pady=(5, 0))
        ttk.Label(
            display_frame,
            text="Green tip: within 0.75 mm of the path's maximum height",
            wraplength=300,
            justify="left",
        ).pack(anchor="w", pady=(3, 0))

        playback_frame = ttk.LabelFrame(sidebar, text="Array playback", padding=10)
        playback_frame.pack(fill="x", pady=(0, 8))
        duration_row = ttk.Frame(playback_frame)
        duration_row.pack(fill="x", pady=(0, 5))
        ttk.Label(duration_row, text="Cycle duration (seconds)").pack(side="left")
        ttk.Spinbox(
            duration_row,
            from_=0.5,
            to=120.0,
            increment=0.5,
            width=7,
            textvariable=self.array_duration_var,
        ).pack(side="right")
        ttk.Checkbutton(
            playback_frame,
            text="Loop continuously",
            variable=self.array_loop_var,
        ).pack(anchor="w", pady=(0, 5))
        playback_buttons = ttk.Frame(playback_frame)
        playback_buttons.pack(fill="x")
        ttk.Button(playback_buttons, text="Play", command=self.play_array).pack(
            side="left", fill="x", expand=True, padx=(0, 3)
        )
        ttk.Button(playback_buttons, text="Pause", command=self.pause_array).pack(
            side="left", fill="x", expand=True, padx=3
        )
        ttk.Button(playback_buttons, text="Stop", command=self.stop_array).pack(
            side="left", fill="x", expand=True, padx=(3, 0)
        )

        collision_frame = ttk.LabelFrame(
            sidebar, text="Collision analysis", padding=10
        )
        collision_frame.pack(fill="x", pady=(0, 8))
        ttk.Label(
            collision_frame,
            text=(
                "Paddles are modelled as 50 x 7.5 mm rectangles. A warning is "
                "raised below 2 mm surface clearance."
            ),
            wraplength=300,
            justify="left",
        ).pack(anchor="w", pady=(0, 6))
        ttk.Button(
            collision_frame,
            text="Check complete cycle (360 positions)",
            command=self.check_complete_array_cycle,
        ).pack(fill="x", pady=(0, 5))
        ttk.Button(
            collision_frame,
            text="Generate phase-spacing safety map",
            command=self.generate_safety_map,
        ).pack(fill="x")

        status_frame = ttk.LabelFrame(sidebar, text="Array status", padding=8)
        status_frame.pack(fill="x")
        tk.Label(
            status_frame,
            textvariable=self.array_status_var,
            wraplength=300,
            justify="left",
            anchor="nw",
            height=7,
            background=self.cget("background"),
        ).pack(fill="x", anchor="nw")

    @staticmethod
    def _add_slider(
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        minimum: float,
        maximum: float,
        value_text: tk.StringVar,
        callback,
        entry_callback,
        unit: str,
    ) -> None:
        header = ttk.Frame(parent)
        header.pack(fill="x")
        ttk.Label(header, text=label).pack(side="left")
        ttk.Label(header, text=unit, width=4, anchor="e").pack(side="right")
        entry = ttk.Entry(
            header,
            textvariable=value_text,
            width=10,
            justify="right",
        )
        entry.pack(side="right", padx=(6, 2))
        entry.bind("<Return>", lambda _event: entry_callback())
        entry.bind("<FocusOut>", lambda _event: entry_callback())
        ttk.Scale(
            parent,
            variable=variable,
            from_=minimum,
            to=maximum,
            command=callback,
        ).pack(fill="x", pady=(0, 7))

    def _update_sidebar_scroll_region(self, _event: tk.Event) -> None:
        """Update the scrollable height without changing the graph layout."""

        bounds = self.sidebar_canvas.bbox("all")
        if bounds is not None:
            self.sidebar_canvas.configure(scrollregion=bounds)

    def _resize_sidebar_contents(self, event: tk.Event) -> None:
        """Match the settings frame width to its fixed-width viewport."""

        self.sidebar_canvas.itemconfigure(
            self.sidebar_window_id,
            width=event.width,
        )

    def _update_array_sidebar_scroll_region(self, _event: tk.Event) -> None:
        bounds = self.array_sidebar_canvas.bbox("all")
        if bounds is not None:
            self.array_sidebar_canvas.configure(scrollregion=bounds)

    def _resize_array_sidebar_contents(self, event: tk.Event) -> None:
        self.array_sidebar_canvas.itemconfigure(
            self.array_sidebar_window_id,
            width=event.width,
        )

    def _scroll_sidebar_with_mouse(self, event: tk.Event) -> str | None:
        """Scroll only when the pointer is over the settings column."""

        widget = self.winfo_containing(event.x_root, event.y_root)
        while widget is not None:
            if widget is self.sidebar_canvas:
                if event.delta:
                    steps = -int(event.delta / 120)
                    if steps == 0:
                        steps = -1 if event.delta > 0 else 1
                    self.sidebar_canvas.yview_scroll(steps, "units")
                return "break"
            if widget is self.array_sidebar_canvas:
                if event.delta:
                    steps = -int(event.delta / 120)
                    if steps == 0:
                        steps = -1 if event.delta > 0 else 1
                    self.array_sidebar_canvas.yview_scroll(steps, "units")
                return "break"
            widget = widget.master
        return None

    # ---------------------------------------------------------- Kinematics

    def _angles_changed(self, _value: str = "") -> None:
        if self._updating_controls:
            return
        self.stop_playback(silent=True)
        self.lower_deg = clamp(
            self.lower_var.get(), LOWER_MIN_DEG, LOWER_MAX_DEG
        )
        self.upper_deg = clamp(
            self.upper_var.get(), UPPER_MIN_DEG, UPPER_MAX_DEG
        )
        self.tip_x_mm, self.tip_y_mm = forward_kinematics(
            self.lower_deg, self.upper_deg
        )
        self._sync_controls()
        self._record_current_if_needed()
        self.status_var.set("Position calculated from the two servo commands.")
        self._draw_scene()

    def _coordinates_changed(self, _value: str = "") -> None:
        if self._updating_controls:
            return
        self.stop_playback(silent=True)
        self._move_tip_to(self.x_var.get(), self.y_var.get())

    def _typed_value_changed(self, control: str) -> None:
        """Apply a number typed beside one of the four sliders."""

        if self._updating_controls:
            return

        text_variables = {
            "lower": self.lower_text,
            "upper": self.upper_text,
            "x": self.x_text,
            "y": self.y_text,
        }
        try:
            value = float(text_variables[control].get().strip())
        except ValueError:
            self.status_var.set("Enter a valid number, for example 72.5.")
            self._update_value_labels()
            return

        if control in ("lower", "upper"):
            if not 0.0 <= value <= 180.0:
                self.status_var.set("Servo commands must be between 0 and 180 degrees.")
                self._update_value_labels()
                return
            if control == "lower":
                self.lower_var.set(value)
            else:
                self.upper_var.set(value)
            self._angles_changed()
            return

        if not -100.0 <= value <= 100.0:
            self.status_var.set("Tip coordinates must be between -100 and 100 mm.")
            self._update_value_labels()
            return
        if control == "x":
            self.x_var.set(value)
        else:
            self.y_var.set(value)
        self._coordinates_changed()

    def _arc_height_changed(self, _value: str = "") -> None:
        self.arc_height_text.set(f"{self.arc_height_var.get():.2f}")
        if self.arc_preview_active:
            self._draw_scene()

    def _fillet_radius_changed(self, _value: str = "") -> None:
        self.fillet_radius_text.set(f"{self.fillet_radius_var.get():.2f}")
        if self.fillet_preview_active:
            self._draw_scene()

    def _typed_curve_value_changed(self, control: str) -> None:
        if control == "arc":
            text_variable = self.arc_height_text
            target_variable = self.arc_height_var
            minimum, maximum = -50.0, 50.0
            callback = self._arc_height_changed
        else:
            text_variable = self.fillet_radius_text
            target_variable = self.fillet_radius_var
            minimum, maximum = 0.1, 40.0
            callback = self._fillet_radius_changed

        try:
            value = float(text_variable.get().strip())
        except ValueError:
            self.status_var.set("Enter a valid curve value, for example 8.5.")
            callback()
            return
        if not minimum <= value <= maximum:
            self.status_var.set(
                f"Curve value must be between {minimum:g} and {maximum:g} mm."
            )
            callback()
            return
        target_variable.set(value)
        callback()

    def _move_tip_to(self, x_mm: float, y_mm: float) -> bool:
        angles = inverse_kinematics(
            x_mm,
            y_mm,
            (self.lower_deg, self.upper_deg),
        )
        if angles is None:
            self.status_var.set(
                "That point is outside the reachable region allowed by the "
                "two 0-180 degree servo arcs. The last valid position was kept."
            )
            self._sync_controls()
            return False

        self.lower_deg, self.upper_deg = angles
        self.tip_x_mm, self.tip_y_mm = forward_kinematics(
            self.lower_deg, self.upper_deg
        )
        self._sync_controls()
        self._record_current_if_needed()
        self.status_var.set("Servo commands calculated using inverse kinematics.")
        self._draw_scene()
        return True

    def _sync_controls(self) -> None:
        self._updating_controls = True
        try:
            self.lower_var.set(self.lower_deg)
            self.upper_var.set(self.upper_deg)
            self.x_var.set(self.tip_x_mm)
            self.y_var.set(self.tip_y_mm)
            self._update_value_labels()
        finally:
            self._updating_controls = False

    def _update_value_labels(self) -> None:
        self.lower_text.set(f"{self.lower_deg:.2f}")
        self.upper_text.set(f"{self.upper_deg:.2f}")
        self.x_text.set(f"{self.tip_x_mm:.2f}")
        self.y_text.set(f"{self.tip_y_mm:.2f}")

    # ------------------------------------------------------------- Canvas

    def _canvas_geometry(self) -> tuple[float, float, float]:
        width = max(self.canvas.winfo_width(), 200)
        height = max(self.canvas.winfo_height(), 200)
        x_span = DESIGNER_X_MAX_MM - DESIGNER_X_MIN_MM
        y_span = DESIGNER_Y_MAX_MM - DESIGNER_Y_MIN_MM
        base_scale = min((width - 50) / x_span, (height - 50) / y_span)
        scale = base_scale * self.designer_zoom
        centre_x = self.designer_pan_x_mm
        centre_y = (
            (DESIGNER_Y_MIN_MM + DESIGNER_Y_MAX_MM) / 2.0
            + self.designer_pan_y_mm
        )
        origin_x = width / 2.0 - centre_x * scale
        origin_y = height / 2.0 + centre_y * scale
        return scale, origin_x, origin_y

    def _world_to_canvas(self, x_mm: float, y_mm: float) -> tuple[float, float]:
        scale, origin_x, origin_y = self._canvas_geometry()
        return origin_x + x_mm * scale, origin_y - y_mm * scale

    def _canvas_to_world(self, canvas_x: float, canvas_y: float) -> tuple[float, float]:
        scale, origin_x, origin_y = self._canvas_geometry()
        return (canvas_x - origin_x) / scale, (origin_y - canvas_y) / scale

    def _canvas_move_tip(self, event: tk.Event) -> None:
        x_mm, y_mm = self._canvas_to_world(event.x, event.y)
        self._move_tip_to(x_mm, y_mm)

    def _zoom_designer_view(self, event: tk.Event) -> str:
        before_x, before_y = self._canvas_to_world(event.x, event.y)
        factor = 1.12 if event.delta > 0 else 1.0 / 1.12
        self.designer_zoom = clamp(self.designer_zoom * factor, 0.35, 8.0)
        after_x, after_y = self._canvas_to_world(event.x, event.y)
        self.designer_pan_x_mm += before_x - after_x
        self.designer_pan_y_mm += before_y - after_y
        self._draw_scene()
        return "break"

    def _start_designer_pan(self, event: tk.Event) -> None:
        self._designer_pan_anchor = (
            event.x,
            event.y,
            self.designer_pan_x_mm,
            self.designer_pan_y_mm,
        )

    def _pan_designer_view(self, event: tk.Event) -> None:
        if self._designer_pan_anchor is None:
            return
        start_x, start_y, pan_x, pan_y = self._designer_pan_anchor
        scale, _origin_x, _origin_y = self._canvas_geometry()
        self.designer_pan_x_mm = pan_x - (event.x - start_x) / scale
        self.designer_pan_y_mm = pan_y + (event.y - start_y) / scale
        self._draw_scene()

    def _reset_designer_view(self) -> None:
        self.designer_zoom = 1.0
        self.designer_pan_x_mm = 0.0
        self.designer_pan_y_mm = 0.0
        self._draw_scene()

    def _draw_scene(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")

        # Grid and axes.
        for coordinate in range(-100, 101, 20):
            x1, y1 = self._world_to_canvas(coordinate, DESIGNER_Y_MIN_MM)
            x2, y2 = self._world_to_canvas(coordinate, DESIGNER_Y_MAX_MM)
            self.canvas.create_line(x1, y1, x2, y2, fill="#e1e5ea")

        for coordinate in range(0, 101, 20):
            x1, y1 = self._world_to_canvas(DESIGNER_X_MIN_MM, coordinate)
            x2, y2 = self._world_to_canvas(DESIGNER_X_MAX_MM, coordinate)
            self.canvas.create_line(x1, y1, x2, y2, fill="#e1e5ea")

        x1, y1 = self._world_to_canvas(DESIGNER_X_MIN_MM, 0.0)
        x2, y2 = self._world_to_canvas(DESIGNER_X_MAX_MM, 0.0)
        self.canvas.create_line(x1, y1, x2, y2, fill="#68727d", width=2)
        x1, y1 = self._world_to_canvas(0.0, DESIGNER_Y_MIN_MM)
        x2, y2 = self._world_to_canvas(0.0, DESIGNER_Y_MAX_MM)
        self.canvas.create_line(x1, y1, x2, y2, fill="#68727d", width=2)

        # Only the useful upper half of the nominal workspace is displayed.
        left, top = self._world_to_canvas(-L1_MM - L2_MM, L1_MM + L2_MM)
        right, bottom = self._world_to_canvas(L1_MM + L2_MM, -L1_MM - L2_MM)
        self.canvas.create_arc(
            left,
            top,
            right,
            bottom,
            start=0,
            extent=180,
            style="arc",
            outline="#aab2bb",
            dash=(5, 4),
            width=1,
        )
        inner_left, inner_top = self._world_to_canvas(
            -MIN_REACH_MM, MIN_REACH_MM
        )
        inner_right, inner_bottom = self._world_to_canvas(
            MIN_REACH_MM, -MIN_REACH_MM
        )
        self.canvas.create_arc(
            inner_left,
            inner_top,
            inner_right,
            inner_bottom,
            start=0,
            extent=180,
            style="arc",
            outline="#8e99a4",
            dash=(2, 4),
            width=1,
        )

        # Designed path.
        if len(self.path_points) >= 2:
            path_coordinates: list[float] = []
            for point in self.path_points:
                canvas_x, canvas_y = self._world_to_canvas(point.x_mm, point.y_mm)
                path_coordinates.extend((canvas_x, canvas_y))
            self.canvas.create_line(
                *path_coordinates,
                fill="#1565c0",
                width=3,
                capstyle="round",
                joinstyle="round",
            )

        for point in self.path_points:
            if point.source != "saved":
                continue
            canvas_x, canvas_y = self._world_to_canvas(point.x_mm, point.y_mm)
            radius = 4
            self.canvas.create_oval(
                canvas_x - radius,
                canvas_y - radius,
                canvas_x + radius,
                canvas_y + radius,
                fill="#1565c0",
                outline="white",
                width=1,
            )

        # Live curve previews are overlays only; the path is not changed until
        # the matching Add/Apply button is pressed.
        if self.arc_preview_active:
            try:
                preview = self._arc_preview_coordinates(81)
                preview_canvas: list[float] = []
                for x_mm, y_mm in preview:
                    preview_canvas.extend(self._world_to_canvas(x_mm, y_mm))
                self.canvas.create_line(
                    *preview_canvas,
                    fill="#ef6c00",
                    width=3,
                    dash=(7, 4),
                    capstyle="round",
                    joinstyle="round",
                )
            except ValueError:
                pass

        if self.fillet_preview_active:
            try:
                first, corner, last = self._final_fillet_points()
                fillet, _effective_radius = circular_fillet_coordinates(
                    (first.x_mm, first.y_mm),
                    (corner.x_mm, corner.y_mm),
                    (last.x_mm, last.y_mm),
                    self.fillet_radius_var.get(),
                    max(0.05, self.trace_spacing_var.get()),
                )
                replacement = [
                    (first.x_mm, first.y_mm),
                    *fillet,
                    (last.x_mm, last.y_mm),
                ]
                preview_canvas = []
                for x_mm, y_mm in replacement:
                    preview_canvas.extend(self._world_to_canvas(x_mm, y_mm))
                self.canvas.create_line(
                    *preview_canvas,
                    fill="#8e24aa",
                    width=4,
                    dash=(7, 4),
                    capstyle="round",
                    joinstyle="round",
                )
            except ValueError:
                pass

        # Two-link cilium.
        base_canvas = self._world_to_canvas(0.0, 0.0)
        elbow_x_mm, elbow_y_mm = elbow_position(self.lower_deg)
        elbow_canvas = self._world_to_canvas(elbow_x_mm, elbow_y_mm)
        tip_canvas = self._world_to_canvas(self.tip_x_mm, self.tip_y_mm)

        lower_rectangle = oriented_rectangle((0.0, 0.0), (elbow_x_mm, elbow_y_mm))
        upper_rectangle = oriented_rectangle(
            (elbow_x_mm, elbow_y_mm), (self.tip_x_mm, self.tip_y_mm)
        )
        for rectangle, colour in (
            (lower_rectangle, "#37474f"),
            (upper_rectangle, "#f57c00"),
        ):
            coordinates: list[float] = []
            for world_x, world_y in rectangle:
                coordinates.extend(self._world_to_canvas(world_x, world_y))
            self.canvas.create_polygon(
                *coordinates, fill=colour, outline="#263238", width=1
            )

        for x_canvas, y_canvas, colour, radius in (
            (*base_canvas, "#263238", 8),
            (*elbow_canvas, "#263238", 7),
            (*tip_canvas, "#d32f2f", 8),
        ):
            self.canvas.create_oval(
                x_canvas - radius,
                y_canvas - radius,
                x_canvas + radius,
                y_canvas + radius,
                fill=colour,
                outline="white",
                width=2,
            )

        tip_label = (
            f"Tip  X={self.tip_x_mm:.2f} mm, Y={self.tip_y_mm:.2f} mm"
        )
        self.canvas.create_text(
            tip_canvas[0] + 12,
            tip_canvas[1] - 14,
            text=tip_label,
            anchor="sw",
            fill="#222222",
            font=("Segoe UI", 9, "bold"),
        )

        if self.recording:
            self.canvas.create_text(
                16,
                16,
                text="LIVE TRACE RECORDING",
                anchor="nw",
                fill="#c62828",
                font=("Segoe UI", 11, "bold"),
            )
        elif self.playing:
            self.canvas.create_text(
                16,
                16,
                text="PATH SIMULATION PLAYING",
                anchor="nw",
                fill="#2e7d32",
                font=("Segoe UI", 11, "bold"),
            )

    # ------------------------------------------------------ Array simulator

    def load_current_path_into_array(self) -> None:
        self.stop_array(reset_phase=False)
        if len(self.path_points) < 2:
            messagebox.showerror(
                "No path available",
                "Save or trace at least two path points in the Path Designer first.",
                parent=self,
            )
            return
        try:
            self.array_angle_path = resample_polyline(
                self.path_points, ARRAY_CYCLE_CHECK_SAMPLES
            )
        except ValueError as error:
            messagebox.showerror("Cannot load path", str(error), parent=self)
            return

        closure_gap = math.hypot(
            self.array_angle_path[-1].x_mm - self.array_angle_path[0].x_mm,
            self.array_angle_path[-1].y_mm - self.array_angle_path[0].y_mm,
        )
        self.array_global_phase = 0.0
        self.array_status_var.set(
            f"Loaded {len(self.array_angle_path)} joint-angle samples. "
            f"Cycle closure gap: {closure_gap:.2f} mm."
        )
        self.notebook.select(1)
        self._draw_array_scene()

    def add_array_cilium(self) -> None:
        count = self.array_cilia_count_var.get()
        if count < ARRAY_MAX_CILIA:
            self.array_cilia_count_var.set(count + 1)
            self._draw_array_scene()
        else:
            self.array_status_var.set("The simulator is limited to 12 cilia.")

    def remove_array_cilium(self) -> None:
        count = self.array_cilia_count_var.get()
        if count > 1:
            self.array_cilia_count_var.set(count - 1)
            self._draw_array_scene()
        else:
            self.array_status_var.set("At least one cilium must remain.")

    def _array_controls_changed(self, _value: str = "") -> None:
        if self._updating_controls:
            return
        spacing = clamp(
            self.array_spacing_var.get(),
            ARRAY_MIN_SPACING_MM,
            ARRAY_MAX_SPACING_MM,
        )
        phase_shift = clamp(self.array_phase_shift_var.get(), 0.0, 360.0)
        self._updating_controls = True
        try:
            self.array_spacing_var.set(spacing)
            self.array_phase_shift_var.set(phase_shift)
            self.array_spacing_text.set(f"{spacing:.2f}")
            self.array_phase_text.set(f"{phase_shift:.2f}")
        finally:
            self._updating_controls = False
        self._draw_array_scene()

    def _typed_array_value_changed(self, control: str) -> None:
        if self._updating_controls:
            return
        variable = (
            self.array_spacing_text if control == "spacing" else self.array_phase_text
        )
        try:
            value = float(variable.get().strip())
        except ValueError:
            self.array_status_var.set("Enter a valid numerical spacing or phase.")
            self._array_controls_changed()
            return

        if control == "spacing":
            if not ARRAY_MIN_SPACING_MM <= value <= ARRAY_MAX_SPACING_MM:
                self.array_status_var.set("Spacing must be between 34 and 150 mm.")
                self._array_controls_changed()
                return
            self.array_spacing_var.set(value)
        else:
            if not 0.0 <= value <= 360.0:
                self.array_status_var.set("Phase shift must be between 0 and 360 degrees.")
                self._array_controls_changed()
                return
            self.array_phase_shift_var.set(value)
        self._array_controls_changed()

    def _array_world_bounds(self) -> tuple[float, float, float, float]:
        count = max(1, self.array_cilia_count_var.get())
        spacing = self.array_spacing_var.get()
        array_width = (count - 1) * spacing
        return (
            -array_width / 2.0 - 105.0,
            array_width / 2.0 + 105.0,
            -5.0,
            110.0,
        )

    def _array_canvas_geometry(self) -> tuple[float, float, float]:
        width = max(self.array_canvas.winfo_width(), 200)
        height = max(self.array_canvas.winfo_height(), 200)
        x_min, x_max, y_min, y_max = self._array_world_bounds()
        base_scale = min(
            (width - 50) / max(1.0, x_max - x_min),
            (height - 50) / max(1.0, y_max - y_min),
        )
        scale = base_scale * self.array_zoom
        centre_x = (x_min + x_max) / 2.0 + self.array_pan_x_mm
        centre_y = (y_min + y_max) / 2.0 + self.array_pan_y_mm
        origin_x = width / 2.0 - centre_x * scale
        origin_y = height / 2.0 + centre_y * scale
        return scale, origin_x, origin_y

    def _array_world_to_canvas(
        self, x_mm: float, y_mm: float
    ) -> tuple[float, float]:
        scale, origin_x, origin_y = self._array_canvas_geometry()
        return origin_x + x_mm * scale, origin_y - y_mm * scale

    def _array_canvas_to_world(
        self, canvas_x: float, canvas_y: float
    ) -> tuple[float, float]:
        scale, origin_x, origin_y = self._array_canvas_geometry()
        return (canvas_x - origin_x) / scale, (origin_y - canvas_y) / scale

    def _zoom_array_view(self, event: tk.Event) -> str:
        before_x, before_y = self._array_canvas_to_world(event.x, event.y)
        factor = 1.12 if event.delta > 0 else 1.0 / 1.12
        self.array_zoom = clamp(self.array_zoom * factor, 0.25, 12.0)
        after_x, after_y = self._array_canvas_to_world(event.x, event.y)
        self.array_pan_x_mm += before_x - after_x
        self.array_pan_y_mm += before_y - after_y
        self._draw_array_scene()
        return "break"

    def _start_array_pan(self, event: tk.Event) -> None:
        self._array_pan_anchor = (
            event.x,
            event.y,
            self.array_pan_x_mm,
            self.array_pan_y_mm,
        )

    def _pan_array_view(self, event: tk.Event) -> None:
        if self._array_pan_anchor is None:
            return
        start_x, start_y, pan_x, pan_y = self._array_pan_anchor
        scale, _origin_x, _origin_y = self._array_canvas_geometry()
        self.array_pan_x_mm = pan_x - (event.x - start_x) / scale
        self.array_pan_y_mm = pan_y + (event.y - start_y) / scale
        self._draw_array_scene()

    def _reset_array_view(self) -> None:
        self.array_zoom = 1.0
        self.array_pan_x_mm = 0.0
        self.array_pan_y_mm = 0.0
        self._draw_array_scene()

    @staticmethod
    def _path_point_at_phase(path: list[PathPoint], phase: float) -> PathPoint:
        if not path:
            return PathPoint(0.0, 100.0, 90.0, 90.0, "array")
        table_position = (phase % 1.0) * len(path)
        index0 = int(math.floor(table_position)) % len(path)
        index1 = (index0 + 1) % len(path)
        fraction = table_position - math.floor(table_position)
        lower = path[index0].lower_deg + (
            path[index1].lower_deg - path[index0].lower_deg
        ) * fraction
        upper = path[index0].upper_deg + (
            path[index1].upper_deg - path[index0].upper_deg
        ) * fraction
        x_mm, y_mm = forward_kinematics(lower, upper)
        return PathPoint(x_mm, y_mm, lower, upper, "array")

    @classmethod
    def _array_poses(
        cls,
        path: list[PathPoint],
        global_phase: float,
        count: int,
        spacing_mm: float,
        phase_shift_deg: float,
    ) -> list[dict]:
        poses = []
        for index in range(count):
            base_x = (index - (count - 1) / 2.0) * spacing_mm
            local_phase = (
                global_phase + index * phase_shift_deg / 360.0
            ) % 1.0
            point = cls._path_point_at_phase(path, local_phase)
            elbow_x, elbow_y = elbow_position(point.lower_deg)
            base = (base_x, 0.0)
            elbow = (base_x + elbow_x, elbow_y)
            tip = (base_x + point.x_mm, point.y_mm)
            rectangles = [
                oriented_rectangle(base, elbow),
                oriented_rectangle(elbow, tip),
            ]
            envelopes = [
                oriented_rectangle(base, elbow, expansion_mm=COLLISION_MARGIN_MM / 2.0),
                oriented_rectangle(
                    elbow, tip, expansion_mm=COLLISION_MARGIN_MM / 2.0
                ),
            ]
            poses.append(
                {
                    "index": index,
                    "phase": local_phase,
                    "point": point,
                    "base": base,
                    "elbow": elbow,
                    "tip": tip,
                    "rectangles": rectangles,
                    "envelopes": envelopes,
                }
            )
        return poses

    @staticmethod
    def _collision_state(
        poses: list[dict],
    ) -> tuple[float, tuple[int, int, int, int] | None, set[tuple[int, int]]]:
        minimum = float("inf")
        closest: tuple[int, int, int, int] | None = None
        colliding_links: set[tuple[int, int]] = set()
        for first_index in range(len(poses)):
            for second_index in range(first_index + 1, len(poses)):
                base_separation = abs(
                    poses[second_index]["base"][0] - poses[first_index]["base"][0]
                )
                if base_separation > 2.0 * (L1_MM + L2_MM) + COLLISION_MARGIN_MM:
                    continue
                for first_link, first_rectangle in enumerate(
                    poses[first_index]["rectangles"]
                ):
                    for second_link, second_rectangle in enumerate(
                        poses[second_index]["rectangles"]
                    ):
                        distance = rectangle_distance(first_rectangle, second_rectangle)
                        if distance < minimum:
                            minimum = distance
                            closest = (
                                first_index,
                                first_link,
                                second_index,
                                second_link,
                            )
                        if distance < COLLISION_MARGIN_MM - 1e-9:
                            colliding_links.add((first_index, first_link))
                            colliding_links.add((second_index, second_link))
        return minimum, closest, colliding_links

    @staticmethod
    def _poses_have_safety_collision(poses: list[dict]) -> bool:
        """Fast conservative envelope test used by the safety-map scan."""

        for first_index in range(len(poses)):
            for second_index in range(first_index + 1, len(poses)):
                if abs(
                    poses[second_index]["base"][0] - poses[first_index]["base"][0]
                ) > 2.0 * (L1_MM + L2_MM) + COLLISION_MARGIN_MM:
                    continue
                for first_rectangle in poses[first_index]["envelopes"]:
                    for second_rectangle in poses[second_index]["envelopes"]:
                        if rectangles_intersect(first_rectangle, second_rectangle):
                            return True
        return False

    @classmethod
    def _sampled_configuration_is_unsafe(
        cls,
        path: list[PathPoint],
        count: int,
        spacing_mm: float,
        phase_shift_deg: float,
        sample_count: int,
    ) -> bool:
        """Check unique cilium separations across a sampled full cycle.

        Every pair with the same index separation repeats the same relative
        motion, only offset in time and X.  Checking one representative pair
        per separation therefore removes duplicate work from the safety map.
        The map's five-degree phase grid is aligned to its 72 cycle samples.
        """

        for index_separation in range(1, count):
            base_separation = index_separation * spacing_mm
            if (
                base_separation
                > 2.0 * (L1_MM + L2_MM) + COLLISION_MARGIN_MM
            ):
                break
            relative_phase = index_separation * phase_shift_deg / 360.0
            for sample in range(sample_count):
                global_phase = sample / sample_count
                first_point = cls._path_point_at_phase(path, global_phase)
                second_point = cls._path_point_at_phase(
                    path, global_phase + relative_phase
                )
                first_elbow = elbow_position(first_point.lower_deg)
                second_elbow_local = elbow_position(second_point.lower_deg)
                second_base = (base_separation, 0.0)
                second_elbow = (
                    base_separation + second_elbow_local[0],
                    second_elbow_local[1],
                )
                first_rectangles = (
                    oriented_rectangle((0.0, 0.0), first_elbow),
                    oriented_rectangle(
                        first_elbow, (first_point.x_mm, first_point.y_mm)
                    ),
                )
                second_rectangles = (
                    oriented_rectangle(second_base, second_elbow),
                    oriented_rectangle(
                        second_elbow,
                        (
                            base_separation + second_point.x_mm,
                            second_point.y_mm,
                        ),
                    ),
                )
                for first_rectangle in first_rectangles:
                    for second_rectangle in second_rectangles:
                        if (
                            rectangle_distance(first_rectangle, second_rectangle)
                            < COLLISION_MARGIN_MM - 1e-9
                        ):
                            return True
        return False

    def _draw_array_polygon(
        self,
        polygon: list[tuple[float, float]],
        **options,
    ) -> int:
        coordinates: list[float] = []
        for world_x, world_y in polygon:
            coordinates.extend(self._array_world_to_canvas(world_x, world_y))
        return self.array_canvas.create_polygon(*coordinates, **options)

    def _draw_array_scene(self) -> None:
        if not hasattr(self, "array_canvas"):
            return
        self.array_canvas.delete("all")
        count = self.array_cilia_count_var.get()
        spacing = self.array_spacing_var.get()
        phase_shift = self.array_phase_shift_var.get()
        path = self.array_angle_path

        x_min, x_max, y_min, y_max = self._array_world_bounds()
        grid_step = 20 if x_max - x_min < 700 else 50
        first_grid_x = math.floor(x_min / grid_step) * grid_step
        x_value = first_grid_x
        while x_value <= x_max:
            start = self._array_world_to_canvas(x_value, y_min)
            end = self._array_world_to_canvas(x_value, y_max)
            self.array_canvas.create_line(*start, *end, fill="#e1e5ea")
            x_value += grid_step
        for y_value in range(0, 101, 20):
            start = self._array_world_to_canvas(x_min, y_value)
            end = self._array_world_to_canvas(x_max, y_value)
            self.array_canvas.create_line(*start, *end, fill="#e1e5ea")
        start = self._array_world_to_canvas(x_min, 0.0)
        end = self._array_world_to_canvas(x_max, 0.0)
        self.array_canvas.create_line(*start, *end, fill="#68727d", width=2)

        if not path:
            self.array_canvas.create_text(
                self.array_canvas.winfo_width() / 2,
                self.array_canvas.winfo_height() / 2,
                text="Load a recorded path from the Path Designer",
                fill="#58636f",
                font=("Segoe UI", 14, "bold"),
            )
            return

        poses = self._array_poses(
            path, self.array_global_phase, count, spacing, phase_shift
        )
        maximum_tip_height = max(point.y_mm for point in path)
        minimum, closest, colliding_links = self._collision_state(poses)

        for pose in poses:
            base_x = pose["base"][0]
            left, top = self._array_world_to_canvas(base_x - 100.0, 100.0)
            right, bottom = self._array_world_to_canvas(base_x + 100.0, -100.0)
            self.array_canvas.create_arc(
                left,
                top,
                right,
                bottom,
                start=0,
                extent=180,
                style="arc",
                outline="#c6ccd2",
                dash=(4, 5),
            )
            inner_left, inner_top = self._array_world_to_canvas(
                base_x - MIN_REACH_MM, MIN_REACH_MM
            )
            inner_right, inner_bottom = self._array_world_to_canvas(
                base_x + MIN_REACH_MM, -MIN_REACH_MM
            )
            self.array_canvas.create_arc(
                inner_left,
                inner_top,
                inner_right,
                inner_bottom,
                start=0,
                extent=180,
                style="arc",
                outline="#a8b0b8",
                dash=(2, 4),
            )

            if self.array_show_traces_var.get():
                trace_coordinates: list[float] = []
                for point in path:
                    trace_coordinates.extend(
                        self._array_world_to_canvas(
                            base_x + point.x_mm, point.y_mm
                        )
                    )
                if len(trace_coordinates) >= 4:
                    self.array_canvas.create_line(
                        *trace_coordinates,
                        fill="#90caf9",
                        width=1,
                        joinstyle="round",
                    )

            if self.array_show_envelopes_var.get():
                for envelope in pose["envelopes"]:
                    coordinates: list[float] = []
                    for world_x, world_y in envelope:
                        coordinates.extend(
                            self._array_world_to_canvas(world_x, world_y)
                        )
                    self.array_canvas.create_polygon(
                        *coordinates,
                        fill="",
                        outline="#ffb74d",
                        dash=(3, 3),
                    )

        for pose in poses:
            for link_index, rectangle in enumerate(pose["rectangles"]):
                collision = (pose["index"], link_index) in colliding_links
                colour = (
                    "#d32f2f"
                    if collision
                    else ("#455a64" if link_index == 0 else "#f57c00")
                )
                self._draw_array_polygon(
                    rectangle,
                    fill=colour,
                    outline="#263238",
                    width=1,
                )
            tip_is_driving = (
                pose["tip"][1]
                >= maximum_tip_height - DRIVE_HEIGHT_TOLERANCE_MM
            )
            for point, radius, colour in (
                (pose["base"], 5, "#263238"),
                (pose["elbow"], 4, "#263238"),
                (pose["tip"], 5 if tip_is_driving else 4,
                 "#2e7d32" if tip_is_driving else "#c62828"),
            ):
                canvas_x, canvas_y = self._array_world_to_canvas(*point)
                self.array_canvas.create_oval(
                    canvas_x - radius,
                    canvas_y - radius,
                    canvas_x + radius,
                    canvas_y + radius,
                    fill=colour,
                    outline="white",
                )
            base_canvas = self._array_world_to_canvas(*pose["base"])
            self.array_canvas.create_text(
                base_canvas[0],
                base_canvas[1] + 14,
                text=str(pose["index"] + 1),
                anchor="n",
                fill="#37474f",
                font=("Segoe UI", 8),
            )

        if self.array_playing:
            clearance_text = (
                "n/a" if math.isinf(minimum) else f"{minimum:.2f} mm"
            )
            warning = "  COLLISION WARNING" if colliding_links else ""
            self.array_status_var.set(
                f"Playing phase {self.array_global_phase * 360.0:.1f} deg. "
                f"Current minimum clearance: {clearance_text}.{warning}"
            )
        if colliding_links:
            self.array_canvas.create_text(
                16,
                16,
                text="CLEARANCE BELOW 2 mm",
                anchor="nw",
                fill="#c62828",
                font=("Segoe UI", 11, "bold"),
            )

    def play_array(self) -> None:
        if not self.array_angle_path:
            messagebox.showerror(
                "No array path",
                "Use the current Path Designer trace before starting playback.",
                parent=self,
            )
            return
        try:
            duration = float(self.array_duration_var.get())
        except (ValueError, tk.TclError):
            duration = 0.0
        if duration <= 0.0:
            messagebox.showerror(
                "Invalid duration",
                "Cycle duration must be greater than zero.",
                parent=self,
            )
            return
        self.pause_array(silent=True)
        self.array_playing = True
        self.array_playback_started_ms = (
            time.perf_counter() * 1000.0
            - self.array_global_phase * duration * 1000.0
        )
        self._array_next_frame()

    def _array_next_frame(self) -> None:
        self.array_after_id = None
        if not self.array_playing:
            return
        duration = max(0.001, float(self.array_duration_var.get()))
        elapsed = (time.perf_counter() * 1000.0 - self.array_playback_started_ms) / 1000.0
        if elapsed >= duration and not self.array_loop_var.get():
            self.array_global_phase = 1.0 - 1e-9
            self.array_playing = False
            self._draw_array_scene()
            self.array_status_var.set("Array playback complete.")
            return
        self.array_global_phase = (elapsed / duration) % 1.0
        self._draw_array_scene()
        self.array_after_id = self.after(33, self._array_next_frame)

    def pause_array(self, silent: bool = False) -> None:
        was_playing = self.array_playing
        self.array_playing = False
        if self.array_after_id is not None:
            try:
                self.after_cancel(self.array_after_id)
            except tk.TclError:
                pass
            self.array_after_id = None
        if was_playing and not silent:
            self.array_status_var.set(
                f"Array paused at phase {self.array_global_phase * 360.0:.1f} degrees."
            )

    def stop_array(self, reset_phase: bool = True) -> None:
        self.pause_array(silent=True)
        if reset_phase:
            self.array_global_phase = 0.0
        self._draw_array_scene()
        self.array_status_var.set("Array playback stopped.")

    def check_complete_array_cycle(self) -> None:
        if not self.array_angle_path:
            messagebox.showerror(
                "No array path",
                "Load the current path before checking the cycle.",
                parent=self,
            )
            return
        self.pause_array(silent=True)
        count = self.array_cilia_count_var.get()
        spacing = self.array_spacing_var.get()
        phase_shift = self.array_phase_shift_var.get()
        minimum = float("inf")
        worst_phase = 0.0
        worst_pair = None
        unsafe_positions = 0
        for sample in range(ARRAY_CYCLE_CHECK_SAMPLES):
            phase = sample / ARRAY_CYCLE_CHECK_SAMPLES
            poses = self._array_poses(
                self.array_angle_path, phase, count, spacing, phase_shift
            )
            clearance, closest, collisions = self._collision_state(poses)
            if clearance < minimum:
                minimum = clearance
                worst_phase = phase
                worst_pair = closest
            if collisions:
                unsafe_positions += 1

        self.array_global_phase = worst_phase
        self._draw_array_scene()
        if count == 1:
            self.array_status_var.set(
                "One cilium has no neighbouring paddle collision to check."
            )
        elif worst_pair is None:
            self.array_status_var.set("No comparable cilium pairs were found.")
        else:
            first, first_link, second, second_link = worst_pair
            result = "UNSAFE" if minimum < COLLISION_MARGIN_MM else "CLEAR"
            self.array_status_var.set(
                f"{result}: minimum clearance {minimum:.3f} mm at global phase "
                f"{worst_phase * 360.0:.1f} deg, between cilium {first + 1} "
                f"{'lower' if first_link == 0 else 'upper'} and cilium "
                f"{second + 1} {'lower' if second_link == 0 else 'upper'}. "
                f"Positions below 2 mm: {unsafe_positions}/360."
            )

    def generate_safety_map(self) -> None:
        if self.safety_map_running:
            self.array_status_var.set("A safety-map calculation is already running.")
            return
        if not self.array_angle_path:
            messagebox.showerror(
                "No array path",
                "Load the current path before generating a safety map.",
                parent=self,
            )
            return
        if self.array_cilia_count_var.get() == 1:
            messagebox.showinfo(
                "Safety map",
                "At least two cilia are required for a spacing/phase collision map.",
                parent=self,
            )
            return

        self.pause_array(silent=True)
        path = list(self.array_angle_path)
        count = self.array_cilia_count_var.get()
        spacing_values = [float(value) for value in range(34, 151, 4)]
        if spacing_values[-1] != 150.0:
            spacing_values.append(150.0)
        phase_values = [float(value) for value in range(0, 361, 5)]
        self.safety_map_running = True
        self._safety_map_result = None
        self.array_status_var.set(
            "Generating sampled safety map: 4 mm spacing steps, 5 degree phase "
            "steps and 72 time positions. The simulator remains responsive."
        )

        def worker() -> None:
            matrix: list[list[int]] = []
            for phase_shift in phase_values:
                row = []
                for spacing in spacing_values:
                    unsafe = self._sampled_configuration_is_unsafe(
                        path,
                        count,
                        spacing,
                        phase_shift,
                        72,
                    )
                    row.append(0 if unsafe else 1)
                matrix.append(row)
            self._safety_map_result = (spacing_values, phase_values, matrix, count)

        self._safety_map_thread = threading.Thread(target=worker, daemon=True)
        self._safety_map_thread.start()
        self.after(250, self._poll_safety_map)

    def _poll_safety_map(self) -> None:
        result = getattr(self, "_safety_map_result", None)
        if result is None:
            if getattr(self, "_safety_map_thread", None) is not None:
                if self._safety_map_thread.is_alive():
                    self.after(250, self._poll_safety_map)
                    return
            self.safety_map_running = False
            self.array_status_var.set("Safety-map calculation ended without a result.")
            return

        self.safety_map_running = False
        spacing_values, phase_values, matrix, count = result
        try:
            import matplotlib.pyplot as plt
            from matplotlib.colors import ListedColormap
        except ImportError:
            self._show_safety_map_canvas(
                spacing_values, phase_values, matrix, count
            )
            self.array_status_var.set(
                "Safety map complete and shown in the built-in figure window. "
                "Green cells passed the coarse scan; verify a selected setting "
                "with the 360-position cycle check."
            )
            return

        figure, axis = plt.subplots(figsize=(9, 5.5))
        image = axis.imshow(
            matrix,
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=[
                spacing_values[0] - 2.0,
                spacing_values[-1] + 2.0,
                phase_values[0] - 2.5,
                phase_values[-1] + 2.5,
            ],
            vmin=0,
            vmax=1,
            cmap=ListedColormap(["#d73027", "#1a9850"]),
        )
        axis.set_xlabel("Neighbouring lower-pivot spacing (mm)")
        axis.set_ylabel("Adjacent phase shift (degrees)")
        axis.set_title(
            f"Sampled paddle-collision safety map ({count} cilia)\n"
            "7.5 mm paddles, 2 mm margin; 72 time positions"
        )
        colourbar = figure.colorbar(image, ax=axis, ticks=[0.25, 0.75])
        colourbar.ax.set_yticklabels(["Collision / <2 mm", "Sampled clear"])
        figure.tight_layout()
        plt.show(block=False)
        self.array_status_var.set(
            "Safety map complete. Green cells passed the coarse 72-position scan; "
            "verify any selected setting with the 360-position cycle check."
        )

    def _show_safety_map_canvas(
        self,
        spacing_values: list[float],
        phase_values: list[float],
        matrix: list[list[int]],
        count: int,
    ) -> None:
        """Render the map using only Tkinter when Matplotlib is unavailable."""

        window = tk.Toplevel(self)
        window.title(f"Phase-spacing safety map - {count} cilia")
        window.geometry("940x650")
        window.minsize(700, 500)
        canvas = tk.Canvas(window, background="white", highlightthickness=0)
        canvas.pack(fill="both", expand=True)

        def draw(_event: tk.Event | None = None) -> None:
            canvas.delete("all")
            width = max(canvas.winfo_width(), 700)
            height = max(canvas.winfo_height(), 500)
            left_margin = 92.0
            right_margin = 38.0
            top_margin = 92.0
            bottom_margin = 88.0
            plot_width = width - left_margin - right_margin
            plot_height = height - top_margin - bottom_margin
            column_width = plot_width / len(spacing_values)
            row_height = plot_height / len(phase_values)

            canvas.create_text(
                width / 2.0,
                25,
                text=f"Sampled paddle-collision safety map ({count} cilia)",
                font=("Segoe UI", 14, "bold"),
            )
            canvas.create_text(
                width / 2.0,
                51,
                text="50 x 7.5 mm paddles, 2 mm clearance; 72 time positions",
                font=("Segoe UI", 10),
                fill="#37474f",
            )

            colours = {0: "#d73027", 1: "#1a9850"}
            for row_index, row in enumerate(matrix):
                canvas_row = len(phase_values) - 1 - row_index
                y1 = top_margin + canvas_row * row_height
                y2 = y1 + row_height
                for column_index, value in enumerate(row):
                    x1 = left_margin + column_index * column_width
                    x2 = x1 + column_width
                    canvas.create_rectangle(
                        x1,
                        y1,
                        x2,
                        y2,
                        fill=colours[value],
                        outline="",
                    )

            plot_right = left_margin + plot_width
            plot_bottom = top_margin + plot_height
            canvas.create_rectangle(
                left_margin,
                top_margin,
                plot_right,
                plot_bottom,
                outline="#263238",
                width=1,
            )

            for index, spacing in enumerate(spacing_values):
                if index % 4 != 0 and index != len(spacing_values) - 1:
                    continue
                x_position = left_margin + (index + 0.5) * column_width
                canvas.create_line(
                    x_position,
                    plot_bottom,
                    x_position,
                    plot_bottom + 5,
                    fill="#263238",
                )
                canvas.create_text(
                    x_position,
                    plot_bottom + 18,
                    text=f"{spacing:.0f}",
                    font=("Segoe UI", 8),
                )

            for index, phase_shift in enumerate(phase_values):
                if int(phase_shift) % 30 != 0:
                    continue
                canvas_row = len(phase_values) - 1 - index
                y_position = top_margin + (canvas_row + 0.5) * row_height
                canvas.create_line(
                    left_margin - 5,
                    y_position,
                    left_margin,
                    y_position,
                    fill="#263238",
                )
                canvas.create_text(
                    left_margin - 10,
                    y_position,
                    text=f"{phase_shift:.0f}",
                    anchor="e",
                    font=("Segoe UI", 8),
                )

            canvas.create_text(
                left_margin + plot_width / 2.0,
                height - 38,
                text="Neighbouring lower-pivot spacing (mm)",
                font=("Segoe UI", 10, "bold"),
            )
            canvas.create_text(
                24,
                top_margin + plot_height / 2.0,
                text="Adjacent phase shift (degrees)",
                angle=90,
                font=("Segoe UI", 10, "bold"),
            )

            legend_y = 70.0
            canvas.create_rectangle(
                width - 270,
                legend_y - 8,
                width - 254,
                legend_y + 8,
                fill=colours[0],
                outline="",
            )
            canvas.create_text(
                width - 248,
                legend_y,
                text="Collision / <2 mm",
                anchor="w",
                font=("Segoe UI", 9),
            )
            canvas.create_rectangle(
                width - 135,
                legend_y - 8,
                width - 119,
                legend_y + 8,
                fill=colours[1],
                outline="",
            )
            canvas.create_text(
                width - 113,
                legend_y,
                text="Sampled clear",
                anchor="w",
                font=("Segoe UI", 9),
            )

        canvas.bind("<Configure>", draw)
        window.after_idle(draw)

    # ------------------------------------------------------- Path actions

    def _arc_preview_coordinates(
        self, sample_count: int
    ) -> list[tuple[float, float]]:
        if not self.path_points:
            raise ValueError("Save a starting path point before creating an arc.")
        start = self.path_points[-1]
        return quadratic_arc_coordinates(
            (start.x_mm, start.y_mm),
            (self.tip_x_mm, self.tip_y_mm),
            self.arc_height_var.get(),
            sample_count,
        )

    def _final_fillet_points(self) -> tuple[PathPoint, PathPoint, PathPoint]:
        if len(self.path_points) < 3:
            raise ValueError("Save three straight-line points before adding a fillet.")
        points = self.path_points[-3:]
        if any(point.source != "saved" for point in points):
            raise ValueError(
                "The final three path points must be manually saved straight-line "
                "points. Undo or save a fresh three-point corner first."
            )
        return points[0], points[1], points[2]

    @staticmethod
    def _coordinates_to_path_points(
        coordinates: list[tuple[float, float]],
        reference_angles: tuple[float, float],
        source: str,
        final_point_saved: bool = False,
    ) -> list[PathPoint]:
        points: list[PathPoint] = []
        reference = reference_angles
        for index, (x_mm, y_mm) in enumerate(coordinates):
            angles = inverse_kinematics(x_mm, y_mm, reference)
            if angles is None:
                raise ValueError(
                    f"Curve point ({x_mm:.2f}, {y_mm:.2f}) mm is outside the "
                    "reachable 0-180 degree servo workspace."
                )
            point_source = (
                "saved"
                if final_point_saved and index == len(coordinates) - 1
                else source
            )
            points.append(PathPoint(x_mm, y_mm, angles[0], angles[1], point_source))
            reference = angles
        return points

    def preview_arc_segment(self) -> None:
        self.stop_playback(silent=True)
        try:
            coordinates = self._arc_preview_coordinates(81)
            start = self.path_points[-1]
            self._coordinates_to_path_points(
                coordinates[1:],
                (start.lower_deg, start.upper_deg),
                "arc",
            )
        except ValueError as error:
            self.arc_preview_active = False
            self.status_var.set(str(error))
            self._draw_scene()
            return
        self.recording = False
        self.record_button_text.set("Start live trace")
        self.arc_preview_active = True
        self.fillet_preview_active = False
        self.status_var.set(
            "Arc preview active. Move the tip or adjust/type the signed midpoint "
            "height, then press Add arc."
        )
        self._draw_scene()

    def apply_arc_segment(self) -> None:
        self.stop_playback(silent=True)
        try:
            dense = self._arc_preview_coordinates(101)
            estimated_length = sum(
                math.hypot(current[0] - previous[0], current[1] - previous[1])
                for previous, current in zip(dense, dense[1:])
            )
            spacing = max(0.05, self.trace_spacing_var.get())
            sample_count = max(2, int(math.ceil(estimated_length / spacing)) + 1)
            coordinates = self._arc_preview_coordinates(sample_count)
            start = self.path_points[-1]
            new_points = self._coordinates_to_path_points(
                coordinates[1:],
                (start.lower_deg, start.upper_deg),
                "arc",
                final_point_saved=True,
            )
        except ValueError as error:
            self.status_var.set(str(error))
            self._draw_scene()
            return

        self._push_history()
        self.path_points.extend(new_points)
        self.arc_preview_active = False
        self.fillet_preview_active = False
        self.recording = False
        self.record_button_text.set("Start live trace")
        final = self.path_points[-1]
        self.lower_deg, self.upper_deg = final.lower_deg, final.upper_deg
        self.tip_x_mm, self.tip_y_mm = final.x_mm, final.y_mm
        self._sync_controls()
        self._update_path_summary()
        self.status_var.set(
            f"Added a {len(new_points)}-sample curved segment with midpoint "
            f"height {self.arc_height_var.get():.2f} mm. Undo removes it as one action."
        )
        self._draw_scene()

    def preview_final_corner_fillet(self) -> None:
        self.stop_playback(silent=True)
        try:
            first, corner, last = self._final_fillet_points()
            coordinates, effective_radius = circular_fillet_coordinates(
                (first.x_mm, first.y_mm),
                (corner.x_mm, corner.y_mm),
                (last.x_mm, last.y_mm),
                self.fillet_radius_var.get(),
                max(0.05, self.trace_spacing_var.get()),
            )
            validation_coordinates = [*coordinates, (last.x_mm, last.y_mm)]
            self._coordinates_to_path_points(
                validation_coordinates,
                (first.lower_deg, first.upper_deg),
                "fillet",
                final_point_saved=True,
            )
        except ValueError as error:
            self.fillet_preview_active = False
            self.status_var.set(str(error))
            self._draw_scene()
            return
        self.recording = False
        self.record_button_text.set("Start live trace")
        self.arc_preview_active = False
        self.fillet_preview_active = True
        radius_note = ""
        if effective_radius < self.fillet_radius_var.get() - 1e-6:
            radius_note = f" (limited by segment length to {effective_radius:.2f} mm)"
        self.status_var.set(
            f"Final-corner fillet preview active at radius {effective_radius:.2f} mm"
            f"{radius_note}. Adjust the slider or type a value, then Apply fillet."
        )
        self._draw_scene()

    def apply_final_corner_fillet(self) -> None:
        self.stop_playback(silent=True)
        try:
            first, corner, last = self._final_fillet_points()
            coordinates, effective_radius = circular_fillet_coordinates(
                (first.x_mm, first.y_mm),
                (corner.x_mm, corner.y_mm),
                (last.x_mm, last.y_mm),
                self.fillet_radius_var.get(),
                max(0.05, self.trace_spacing_var.get()),
            )
            replacement_coordinates = [*coordinates, (last.x_mm, last.y_mm)]
            replacement = self._coordinates_to_path_points(
                replacement_coordinates,
                (first.lower_deg, first.upper_deg),
                "fillet",
                final_point_saved=True,
            )
        except ValueError as error:
            self.status_var.set(str(error))
            self._draw_scene()
            return

        self._push_history()
        # Keep the first of the three points, replace the sharp corner and the
        # old final point with the tangent arc plus a newly solved final point.
        self.path_points = [*self.path_points[:-2], *replacement]
        self.arc_preview_active = False
        self.fillet_preview_active = False
        self.recording = False
        self.record_button_text.set("Start live trace")
        final = self.path_points[-1]
        self.lower_deg, self.upper_deg = final.lower_deg, final.upper_deg
        self.tip_x_mm, self.tip_y_mm = final.x_mm, final.y_mm
        self._sync_controls()
        self._update_path_summary()
        self.status_var.set(
            f"Applied a tangent corner fillet of radius {effective_radius:.2f} mm. "
            "Undo restores the sharp three-point corner."
        )
        self._draw_scene()

    def cancel_curve_preview(self, silent: bool = False) -> None:
        was_active = self.arc_preview_active or self.fillet_preview_active
        self.arc_preview_active = False
        self.fillet_preview_active = False
        if not silent:
            self.status_var.set(
                "Curve preview cancelled." if was_active else "No curve preview is active."
            )
        self._draw_scene()

    def _current_point(self, source: str) -> PathPoint:
        return PathPoint(
            self.tip_x_mm,
            self.tip_y_mm,
            self.lower_deg,
            self.upper_deg,
            source,
        )

    def _push_history(self) -> None:
        self.history.append(list(self.path_points))
        if len(self.history) > 100:
            self.history.pop(0)

    def _update_path_summary(self) -> None:
        if len(self.path_points) < 2:
            length_mm = 0.0
        else:
            length_mm = sum(
                math.hypot(
                    current.x_mm - previous.x_mm,
                    current.y_mm - previous.y_mm,
                )
                for previous, current in zip(
                    self.path_points, self.path_points[1:]
                )
            )
        self.path_summary_var.set(
            f"Path: {len(self.path_points)} points, {length_mm:.1f} mm"
        )

    def save_coordinate(self) -> None:
        self.stop_playback(silent=True)
        self.arc_preview_active = False
        self.fillet_preview_active = False
        self._push_history()
        self.path_points.append(self._current_point("saved"))
        self._update_path_summary()
        self.status_var.set(
            f"Saved coordinate {len(self.path_points)} at "
            f"({self.tip_x_mm:.2f}, {self.tip_y_mm:.2f}) mm."
        )
        self._draw_scene()

    def toggle_recording(self) -> None:
        self.stop_playback(silent=True)
        self.arc_preview_active = False
        self.fillet_preview_active = False
        if self.recording:
            self.recording = False
            self.record_button_text.set("Start live trace")
            self.status_var.set(
                f"Live trace stopped. The path contains {len(self.path_points)} points."
            )
        else:
            self._push_history()
            self.recording = True
            self.record_button_text.set("Stop live trace")
            current = self._current_point("trace")
            if not self.path_points or math.hypot(
                current.x_mm - self.path_points[-1].x_mm,
                current.y_mm - self.path_points[-1].y_mm,
            ) > 1e-9:
                self.path_points.append(current)
            self._update_path_summary()
            self.status_var.set(
                "Live trace started. Drag the tip or move any angle/coordinate slider."
            )
        self._draw_scene()

    def _record_current_if_needed(self) -> None:
        if not self.recording:
            return
        current = self._current_point("trace")
        spacing = max(0.0, self.trace_spacing_var.get())
        if not self.path_points or math.hypot(
            current.x_mm - self.path_points[-1].x_mm,
            current.y_mm - self.path_points[-1].y_mm,
        ) >= spacing:
            self.path_points.append(current)
            self._update_path_summary()

    def undo(self) -> None:
        self.stop_playback(silent=True)
        self.arc_preview_active = False
        self.fillet_preview_active = False
        if not self.history:
            self.status_var.set("Nothing to undo.")
            return
        self.recording = False
        self.record_button_text.set("Start live trace")
        self.path_points = self.history.pop()
        self._update_path_summary()
        self.status_var.set("Restored the path to its previous state.")
        self._draw_scene()

    def clear_path(self) -> None:
        self.stop_playback(silent=True)
        self.arc_preview_active = False
        self.fillet_preview_active = False
        if not self.path_points:
            self.status_var.set("The path is already empty.")
            return
        self._push_history()
        self.path_points.clear()
        self.recording = False
        self.record_button_text.set("Start live trace")
        self._update_path_summary()
        self.status_var.set("Path cleared. Undo is available.")
        self._draw_scene()

    def snap_to_last_point(self) -> None:
        """Return the mechanism exactly to the final recorded path point."""

        self.stop_playback(silent=True)
        self.arc_preview_active = False
        self.fillet_preview_active = False
        if not self.path_points:
            self.status_var.set("There is no path point to snap to yet.")
            return

        self.recording = False
        self.record_button_text.set("Start live trace")
        point = self.path_points[-1]
        self.lower_deg = point.lower_deg
        self.upper_deg = point.upper_deg
        self.tip_x_mm, self.tip_y_mm = forward_kinematics(
            self.lower_deg, self.upper_deg
        )
        self._sync_controls()
        self.status_var.set(
            "Snapped to the final path point. The next saved or traced point "
            "will continue from this exact location."
        )
        self._draw_scene()

    # ----------------------------------------------------------- Playback

    def play_path(self) -> None:
        """Animate the linkage through a uniformly sampled recorded path."""

        self.stop_playback(silent=True)
        self.arc_preview_active = False
        self.fillet_preview_active = False
        if len(self.path_points) < 2:
            messagebox.showerror(
                "Cannot play path",
                "Save or trace at least two path points before playing.",
                parent=self,
            )
            return

        try:
            duration_seconds = float(self.playback_duration_var.get())
        except (ValueError, tk.TclError):
            messagebox.showerror(
                "Invalid duration",
                "Enter a valid playback duration in seconds.",
                parent=self,
            )
            return

        if duration_seconds <= 0.0:
            messagebox.showerror(
                "Invalid duration",
                "Playback duration must be greater than zero.",
                parent=self,
            )
            return

        frames_per_second = 30
        sample_count = max(2, round(duration_seconds * frames_per_second))
        try:
            self.playback_points = resample_polyline(
                self.path_points, sample_count
            )
        except ValueError as error:
            messagebox.showerror("Cannot play path", str(error), parent=self)
            return

        self.recording = False
        self.record_button_text.set("Start live trace")
        self.playing = True
        self.playback_index = 0
        self.status_var.set(
            f"Playing {len(self.playback_points)} frames over "
            f"{duration_seconds:.2f} seconds."
        )
        self._play_next_frame()

    def _play_next_frame(self) -> None:
        self.playback_after_id = None
        if not self.playing or not self.playback_points:
            return

        point = self.playback_points[self.playback_index]
        self.lower_deg = point.lower_deg
        self.upper_deg = point.upper_deg
        self.tip_x_mm, self.tip_y_mm = forward_kinematics(
            self.lower_deg, self.upper_deg
        )
        self._sync_controls()
        self._draw_scene()

        self.playback_index += 1
        if self.playback_index >= len(self.playback_points):
            if self.loop_playback_var.get():
                self.playback_index = 0
            else:
                self.playing = False
                self.status_var.set("Path simulation complete.")
                self._draw_scene()
                return

        self.playback_after_id = self.after(33, self._play_next_frame)

    def stop_playback(self, silent: bool = False) -> None:
        """Stop a running animation without changing the recorded path."""

        was_playing = self.playing
        self.playing = False
        if self.playback_after_id is not None:
            try:
                self.after_cancel(self.playback_after_id)
            except tk.TclError:
                pass
            self.playback_after_id = None
        if was_playing and not silent:
            self.status_var.set("Path simulation stopped.")
            self._draw_scene()

    # -------------------------------------------------------------- Export

    def _points_for_export(self) -> list[PathPoint]:
        if len(self.path_points) < 2:
            raise ValueError("Save or trace at least two path points before exporting.")
        if not self.resample_var.get():
            return list(self.path_points)
        return resample_polyline(self.path_points, self.sample_count_var.get())

    def export_path(self) -> None:
        try:
            points = self._points_for_export()
        except (ValueError, tk.TclError) as error:
            messagebox.showerror("Cannot export path", str(error), parent=self)
            return

        export_format = self.export_format_var.get()
        if export_format == "Arduino PWM header":
            extension = ".h"
            filetypes = [("Arduino/C header", "*.h"), ("All files", "*.*")]
        else:
            extension = ".csv"
            filetypes = [("CSV file", "*.csv"), ("All files", "*.*")]

        if export_format == "Arduino PWM header":
            initial_file = "gait_table.h"
            initial_directory = Path(__file__).resolve().parent.parent / "include"
        else:
            initial_file = "cilia_path" + extension
            initial_directory = Path(__file__).resolve().parent

        destination = filedialog.asksaveasfilename(
            parent=self,
            title="Export cilia path",
            defaultextension=extension,
            initialfile=initial_file,
            initialdir=initial_directory,
            filetypes=filetypes,
        )
        if not destination:
            return

        try:
            destination_path = Path(destination)
            preview_path: Path | None = None
            if export_format == "Coordinates CSV":
                self._write_coordinate_csv(destination_path, points)
            elif export_format == "Joint angles CSV":
                self._write_angle_csv(destination_path, points)
            else:
                self._write_arduino_header(destination_path, points)
                preview_path = destination_path.with_name(
                    f"{destination_path.stem}_tip_path.png"
                )
                self._write_tip_path_png(preview_path, destination_path.name, points)
        except (OSError, ValueError) as error:
            messagebox.showerror("Export failed", str(error), parent=self)
            return

        if preview_path is None:
            self.status_var.set(
                f"Exported {len(points)} lookup points to {destination_path}."
            )
        else:
            self.status_var.set(
                f"Exported {len(points)} lookup points to {destination_path} and "
                f"saved the matching tip-path image as {preview_path.name}."
            )

    @staticmethod
    def _write_coordinate_csv(path: Path, points: list[PathPoint]) -> None:
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(
                ["index", "x_mm", "y_mm", "lower_command_deg", "upper_command_deg"]
            )
            for index, point in enumerate(points):
                writer.writerow(
                    [
                        index,
                        f"{point.x_mm:.6f}",
                        f"{point.y_mm:.6f}",
                        f"{point.lower_deg:.6f}",
                        f"{point.upper_deg:.6f}",
                    ]
                )

    @staticmethod
    def _write_angle_csv(path: Path, points: list[PathPoint]) -> None:
        with path.open("w", newline="", encoding="utf-8") as output_file:
            writer = csv.writer(output_file)
            writer.writerow(["index", "lower_command_deg", "upper_command_deg"])
            for index, point in enumerate(points):
                writer.writerow(
                    [
                        index,
                        f"{point.lower_deg:.6f}",
                        f"{point.upper_deg:.6f}",
                    ]
                )

    @staticmethod
    def _angle_to_pwm_count(angle_deg: float) -> int:
        raw_count = PWM_AT_90_COUNT + PWM_COUNTS_PER_DEG * (angle_deg - 90.0)
        rounded_count = math.floor(raw_count + 0.5)
        return int(clamp(rounded_count, PWM_MIN_COUNT, PWM_MAX_COUNT))

    @staticmethod
    def _format_uint16_array(name: str, values: list[int]) -> str:
        lines = [f"const uint16_t {name}[GAIT_TABLE_SIZE] PROGMEM = {{"]
        for start in range(0, len(values), 12):
            chunk = values[start : start + 12]
            suffix = "," if start + 12 < len(values) else ""
            lines.append("    " + ", ".join(str(value) for value in chunk) + suffix)
        lines.append("};")
        return "\n".join(lines)

    @classmethod
    def _write_arduino_header(cls, path: Path, points: list[PathPoint]) -> None:
        lower_values = [cls._angle_to_pwm_count(point.lower_deg) for point in points]
        upper_values = [cls._angle_to_pwm_count(point.upper_deg) for point in points]
        contents = "\n".join(
            [
                "#pragma once",
                "#include <Arduino.h>",
                "",
                "// Generated by cilia_path_designer.py.",
                "// Values are raw PCA9685 OFF counts for ServoDriver::setPwm().",
                "// Conversion is based on the experimentally fitted servo calibration.",
                f"constexpr uint16_t GAIT_TABLE_SIZE = {len(points)};",
                f"constexpr uint16_t GAIT_PWM_FREQUENCY_HZ = {GAIT_PWM_FREQUENCY_HZ};",
                f"constexpr float GAIT_PWM_AT_90_COUNT = {PWM_AT_90_COUNT:.5f}f;",
                f"constexpr float GAIT_ZERO_DEG_COUNT = {PWM_ZERO_DEG_COUNT:.5f}f;",
                f"constexpr float GAIT_COUNTS_PER_DEG = {PWM_COUNTS_PER_DEG:.5f}f;",
                "",
                cls._format_uint16_array("LOWER_TABLE", lower_values),
                "",
                cls._format_uint16_array("UPPER_TABLE", upper_values),
                "",
            ]
        )
        path.write_text(contents, encoding="utf-8")

    @staticmethod
    def _write_tip_path_png(
        path: Path,
        header_name: str,
        points: list[PathPoint],
    ) -> None:
        """Save a visual record of the exact tip points exported to a header."""

        if len(points) < 2:
            raise ValueError("At least two points are required for the path PNG.")
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
        except ImportError:
            CiliaPathDesigner._write_tip_path_png_with_pillow(
                path, header_name, points
            )
            return

        x_values = [point.x_mm for point in points]
        y_values = [point.y_mm for point in points]
        # Arduino playback interpolates cyclically from the last lookup sample
        # back to the first, so show that closing segment in the reference PNG.
        closed_x = [*x_values, x_values[0]]
        closed_y = [*y_values, y_values[0]]
        phases = [index / len(points) for index in range(len(points))]

        figure = Figure(figsize=(9.0, 5.6), dpi=180, facecolor="white")
        FigureCanvasAgg(figure)
        axes = figure.add_subplot(1, 1, 1)
        axes.plot(
            closed_x,
            closed_y,
            color="#e66b0e",
            linewidth=2.4,
            label="Commanded cyclic tip path",
            zorder=2,
        )
        phase_points = axes.scatter(
            x_values,
            y_values,
            c=phases,
            cmap="viridis",
            s=12,
            edgecolors="none",
            zorder=3,
        )
        axes.scatter(
            [x_values[0]],
            [y_values[0]],
            marker="o",
            s=70,
            color="#2e7d32",
            edgecolors="white",
            linewidths=1.0,
            label="Lookup-table start",
            zorder=4,
        )

        # Add several small arrows so the direction remains obvious even when
        # the first and last samples are close together.
        arrow_count = min(4, len(points) - 1)
        for arrow_index in range(arrow_count):
            index = int(arrow_index * len(points) / arrow_count)
            next_index = (index + 1) % len(points)
            dx = x_values[next_index] - x_values[index]
            dy = y_values[next_index] - y_values[index]
            if math.hypot(dx, dy) <= 1e-9:
                continue
            axes.annotate(
                "",
                xy=(x_values[next_index], y_values[next_index]),
                xytext=(x_values[index], y_values[index]),
                arrowprops={
                    "arrowstyle": "-|>",
                    "color": "#263238",
                    "lw": 1.3,
                    "mutation_scale": 11,
                },
                zorder=5,
            )

        axes.set_title(
            f"Commanded cilium-tip path from {header_name}",
            fontsize=13,
            pad=12,
        )
        axes.set_xlabel("Tip X position (mm)")
        axes.set_ylabel("Tip Y position (mm)")
        axes.set_aspect("equal", adjustable="datalim")
        axes.grid(True, color="#d7dce1", linewidth=0.8, alpha=0.8)
        axes.legend(loc="best", frameon=True)
        colour_bar = figure.colorbar(phase_points, ax=axes, pad=0.025)
        colour_bar.set_label("Normalised gait phase")
        axes.text(
            0.01,
            0.01,
            (
                f"{len(points)} lookup samples\n"
                f"PWM = {PWM_AT_90_COUNT:.2f} + {PWM_COUNTS_PER_DEG:.5f} "
                "x (angle - 90 deg)"
            ),
            transform=axes.transAxes,
            fontsize=8.5,
            va="bottom",
            ha="left",
            bbox={
                "boxstyle": "round,pad=0.35",
                "facecolor": "white",
                "edgecolor": "#b0b7bf",
                "alpha": 0.9,
            },
        )
        figure.tight_layout()
        figure.savefig(path, format="png", dpi=180, facecolor="white")

    @staticmethod
    def _write_tip_path_png_with_pillow(
        path: Path,
        header_name: str,
        points: list[PathPoint],
    ) -> None:
        """Dependency-light PNG fallback when matplotlib is unavailable."""

        try:
            from PIL import Image, ImageDraw, ImageFont
        except ImportError as error:
            raise ValueError(
                "The header was written, but matplotlib or Pillow is required "
                "to create the matching tip-path PNG."
            ) from error

        width, height = 1600, 1000
        plot_left, plot_top = 150, 115
        plot_right, plot_bottom = 1460, 835
        image = Image.new("RGB", (width, height), "white")
        draw = ImageDraw.Draw(image)

        def load_font(size: int, bold: bool = False):
            filename = "arialbd.ttf" if bold else "arial.ttf"
            windows_font = Path("C:/Windows/Fonts") / filename
            try:
                return ImageFont.truetype(str(windows_font), size)
            except OSError:
                try:
                    return ImageFont.truetype(filename, size)
                except OSError:
                    return ImageFont.load_default()

        title_font = load_font(30, bold=True)
        label_font = load_font(22)
        tick_font = load_font(17)
        note_font = load_font(18)

        x_values = [point.x_mm for point in points]
        y_values = [point.y_mm for point in points]
        x_min, x_max = min(x_values), max(x_values)
        y_min, y_max = min(y_values), max(y_values)
        x_span = max(1.0, x_max - x_min)
        y_span = max(1.0, y_max - y_min)
        padding = max(3.0, 0.08 * max(x_span, y_span))
        x_min -= padding
        x_max += padding
        y_min -= padding
        y_max += padding
        x_span = x_max - x_min
        y_span = y_max - y_min

        available_width = plot_right - plot_left
        available_height = plot_bottom - plot_top
        scale = min(available_width / x_span, available_height / y_span)
        drawn_width = x_span * scale
        drawn_height = y_span * scale
        x_offset = plot_left + (available_width - drawn_width) / 2.0
        y_offset = plot_top + (available_height - drawn_height) / 2.0

        def transform(x_mm: float, y_mm: float) -> tuple[float, float]:
            return (
                x_offset + (x_mm - x_min) * scale,
                y_offset + drawn_height - (y_mm - y_min) * scale,
            )

        def nice_step(span: float) -> float:
            rough = span / 8.0
            power = 10.0 ** math.floor(math.log10(max(rough, 1e-9)))
            scaled = rough / power
            multiplier = 1.0 if scaled <= 1.0 else 2.0 if scaled <= 2.0 else 5.0
            return multiplier * power

        x_step = nice_step(x_span)
        y_step = nice_step(y_span)
        x_tick = math.ceil(x_min / x_step) * x_step
        while x_tick <= x_max + 1e-9:
            x_pixel, _ = transform(x_tick, y_min)
            draw.line(
                [(x_pixel, y_offset), (x_pixel, y_offset + drawn_height)],
                fill="#d9dde2",
                width=2,
            )
            label = f"{x_tick:g}"
            box = draw.textbbox((0, 0), label, font=tick_font)
            draw.text(
                (x_pixel - (box[2] - box[0]) / 2, y_offset + drawn_height + 10),
                label,
                fill="#42474d",
                font=tick_font,
            )
            x_tick += x_step

        y_tick = math.ceil(y_min / y_step) * y_step
        while y_tick <= y_max + 1e-9:
            _, y_pixel = transform(x_min, y_tick)
            draw.line(
                [(x_offset, y_pixel), (x_offset + drawn_width, y_pixel)],
                fill="#d9dde2",
                width=2,
            )
            label = f"{y_tick:g}"
            box = draw.textbbox((0, 0), label, font=tick_font)
            draw.text(
                (x_offset - (box[2] - box[0]) - 12, y_pixel - 10),
                label,
                fill="#42474d",
                font=tick_font,
            )
            y_tick += y_step

        draw.rectangle(
            [x_offset, y_offset, x_offset + drawn_width, y_offset + drawn_height],
            outline="#68727d",
            width=3,
        )
        pixel_points = [transform(point.x_mm, point.y_mm) for point in points]
        draw.line([*pixel_points, pixel_points[0]], fill="#e66b0e", width=5)

        def phase_colour(fraction: float) -> tuple[int, int, int]:
            # Compact blue-green-yellow progression similar to viridis.
            stops = (
                (0.0, (68, 1, 84)),
                (0.5, (33, 145, 140)),
                (1.0, (253, 231, 37)),
            )
            first, second = (stops[0], stops[1]) if fraction <= 0.5 else (stops[1], stops[2])
            local = (fraction - first[0]) / (second[0] - first[0])
            return tuple(
                round(first[1][channel] + local * (second[1][channel] - first[1][channel]))
                for channel in range(3)
            )

        for index, (x_pixel, y_pixel) in enumerate(pixel_points):
            fraction = index / max(1, len(pixel_points) - 1)
            radius = 5
            draw.ellipse(
                [x_pixel - radius, y_pixel - radius, x_pixel + radius, y_pixel + radius],
                fill=phase_colour(fraction),
            )

        start_x, start_y = pixel_points[0]
        draw.ellipse(
            [start_x - 11, start_y - 11, start_x + 11, start_y + 11],
            fill="#2e7d32",
            outline="white",
            width=3,
        )

        arrow_count = min(4, len(pixel_points) - 1)
        step = max(1, len(pixel_points) // 50)
        for arrow_index in range(arrow_count):
            index = int(arrow_index * len(pixel_points) / arrow_count)
            next_index = (index + step) % len(pixel_points)
            start = pixel_points[index]
            end = pixel_points[next_index]
            dx, dy = end[0] - start[0], end[1] - start[1]
            length = math.hypot(dx, dy)
            if length <= 1e-6:
                continue
            unit_x, unit_y = dx / length, dy / length
            normal_x, normal_y = -unit_y, unit_x
            tip = end
            base = (tip[0] - 18 * unit_x, tip[1] - 18 * unit_y)
            draw.polygon(
                [
                    tip,
                    (base[0] + 8 * normal_x, base[1] + 8 * normal_y),
                    (base[0] - 8 * normal_x, base[1] - 8 * normal_y),
                ],
                fill="#263238",
            )

        title = f"Commanded cilium-tip path from {header_name}"
        title_box = draw.textbbox((0, 0), title, font=title_font)
        draw.text(
            ((width - (title_box[2] - title_box[0])) / 2, 35),
            title,
            fill="#202428",
            font=title_font,
        )
        x_label = "Tip X position (mm)"
        x_box = draw.textbbox((0, 0), x_label, font=label_font)
        draw.text(
            ((width - (x_box[2] - x_box[0])) / 2, 895),
            x_label,
            fill="#202428",
            font=label_font,
        )
        y_label_image = Image.new("RGBA", (45, 280), (255, 255, 255, 0))
        y_draw = ImageDraw.Draw(y_label_image)
        y_draw.text((5, 5), "Tip Y position (mm)", fill="#202428", font=label_font)
        y_label_image = y_label_image.rotate(90, expand=True)
        image.paste(y_label_image, (25, int((height - y_label_image.height) / 2)), y_label_image)

        note = (
            f"Start: green marker   |   {len(points)} lookup samples   |   "
            f"PWM = {PWM_AT_90_COUNT:.2f} + {PWM_COUNTS_PER_DEG:.5f} x (angle - 90 deg)"
        )
        draw.text((plot_left, 950), note, fill="#42474d", font=note_font)
        image.save(path, format="PNG")


def main() -> None:
    app = CiliaPathDesigner()
    app.mainloop()


if __name__ == "__main__":
    main()
