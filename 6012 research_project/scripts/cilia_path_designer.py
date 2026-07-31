#!/usr/bin/env python3
"""Interactive path designer for a two-link artificial cilium.

The program uses the same angle convention as ``d_shape.py``:

* the lower servo command is the absolute lower-link angle;
* an upper servo command of 90 degrees makes both links collinear;
* upper commands from 0 to 180 degrees therefore represent relative joint
  angles from -90 to +90 degrees.

The tip can be moved by dragging it on the canvas, by changing its X/Y
sliders, or by changing the two servo-angle sliders.  Points can be saved
manually or captured continuously while Record Trace is enabled.  The
resulting path can be exported as coordinates, joint-angle tables, or an
Arduino-style header.
"""

from __future__ import annotations

import bisect
import csv
import math
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk


# ---------------------------------------------------------------------------
# Mechanical settings
# ---------------------------------------------------------------------------

L1_MM = 50.0
L2_MM = 50.0

LOWER_MIN_DEG = 0.0
LOWER_MAX_DEG = 180.0
UPPER_MIN_DEG = 0.0
UPPER_MAX_DEG = 180.0

# mechanical relative angle = upper servo command + this offset
UPPER_COMMAND_TO_MECHANICAL_OFFSET_DEG = -90.0

# This reproduces the effective conversion used by the installed Seeed
# ServoDriver after setServoPulseRange(500, 2500, 180).  The library stores
# its intermediate values as integers, so setAngle() ultimately uses:
#
#     pwm_count = 122 + 2 * whole_angle_degrees
#
# Exporting raw counts lets intermediate IK angles use every available PWM
# count instead of first being truncated to whole degrees.
GAIT_PWM_FREQUENCY_HZ = 50
PWM_ZERO_DEG_COUNT = 122
PWM_COUNTS_PER_DEG = 2.0
PWM_MIN_COUNT = 0
PWM_MAX_COUNT = 4095

WORLD_MIN_MM = -110.0
WORLD_MAX_MM = 110.0


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


class CiliaPathDesigner(tk.Tk):
    """Tkinter user interface for interactive cilium path design."""

    def __init__(self) -> None:
        super().__init__()
        self.title("Two-Link Cilia Path Designer")
        self.geometry("1220x780")
        self.minsize(980, 650)

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
        self.export_format_var = tk.StringVar(value="Coordinates CSV")
        self.sample_count_var = tk.IntVar(value=360)
        self.resample_var = tk.BooleanVar(value=True)
        self.playback_duration_var = tk.DoubleVar(value=5.0)
        self.loop_playback_var = tk.BooleanVar(value=False)

        self._build_interface()
        self._update_value_labels()
        self._draw_scene()

        self.bind("<Control-z>", lambda _event: self.undo())
        self.bind("<Control-s>", lambda _event: self.save_coordinate())
        self.bind("<Control-e>", lambda _event: self.export_path())
        self.bind("<Key-r>", lambda _event: self.toggle_recording())

    # ------------------------------------------------------------------ UI

    def _build_interface(self) -> None:
        outer = ttk.Frame(self, padding=10)
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
        world_span = WORLD_MAX_MM - WORLD_MIN_MM
        scale = min((width - 50) / world_span, (height - 50) / world_span)
        origin_x = width / 2.0
        origin_y = height / 2.0
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

    def _draw_scene(self) -> None:
        if not hasattr(self, "canvas"):
            return
        self.canvas.delete("all")

        # Grid and axes.
        for coordinate in range(-100, 101, 20):
            x1, y1 = self._world_to_canvas(coordinate, WORLD_MIN_MM)
            x2, y2 = self._world_to_canvas(coordinate, WORLD_MAX_MM)
            self.canvas.create_line(x1, y1, x2, y2, fill="#e1e5ea")

            x1, y1 = self._world_to_canvas(WORLD_MIN_MM, coordinate)
            x2, y2 = self._world_to_canvas(WORLD_MAX_MM, coordinate)
            self.canvas.create_line(x1, y1, x2, y2, fill="#e1e5ea")

        x1, y1 = self._world_to_canvas(WORLD_MIN_MM, 0.0)
        x2, y2 = self._world_to_canvas(WORLD_MAX_MM, 0.0)
        self.canvas.create_line(x1, y1, x2, y2, fill="#68727d", width=2)
        x1, y1 = self._world_to_canvas(0.0, WORLD_MIN_MM)
        x2, y2 = self._world_to_canvas(0.0, WORLD_MAX_MM)
        self.canvas.create_line(x1, y1, x2, y2, fill="#68727d", width=2)

        # Nominal maximum-reach circle.
        left, top = self._world_to_canvas(-L1_MM - L2_MM, L1_MM + L2_MM)
        right, bottom = self._world_to_canvas(L1_MM + L2_MM, -L1_MM - L2_MM)
        self.canvas.create_oval(
            left,
            top,
            right,
            bottom,
            outline="#aab2bb",
            dash=(5, 4),
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

        # Two-link cilium.
        base_canvas = self._world_to_canvas(0.0, 0.0)
        elbow_x_mm, elbow_y_mm = elbow_position(self.lower_deg)
        elbow_canvas = self._world_to_canvas(elbow_x_mm, elbow_y_mm)
        tip_canvas = self._world_to_canvas(self.tip_x_mm, self.tip_y_mm)

        self.canvas.create_line(
            *base_canvas,
            *elbow_canvas,
            fill="#37474f",
            width=12,
            capstyle="round",
        )
        self.canvas.create_line(
            *elbow_canvas,
            *tip_canvas,
            fill="#f57c00",
            width=12,
            capstyle="round",
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

    # ------------------------------------------------------- Path actions

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
            if export_format == "Coordinates CSV":
                self._write_coordinate_csv(destination_path, points)
            elif export_format == "Joint angles CSV":
                self._write_angle_csv(destination_path, points)
            else:
                self._write_arduino_header(destination_path, points)
        except (OSError, ValueError) as error:
            messagebox.showerror("Export failed", str(error), parent=self)
            return

        self.status_var.set(
            f"Exported {len(points)} lookup points to {destination_path}."
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
        raw_count = PWM_ZERO_DEG_COUNT + PWM_COUNTS_PER_DEG * angle_deg
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
                "// Conversion matches setServoPulseRange(500, 2500, 180).",
                f"constexpr uint16_t GAIT_TABLE_SIZE = {len(points)};",
                f"constexpr uint16_t GAIT_PWM_FREQUENCY_HZ = {GAIT_PWM_FREQUENCY_HZ};",
                f"constexpr uint16_t GAIT_ZERO_DEG_COUNT = {PWM_ZERO_DEG_COUNT};",
                f"constexpr float GAIT_COUNTS_PER_DEG = {PWM_COUNTS_PER_DEG:.1f}f;",
                "",
                cls._format_uint16_array("LOWER_TABLE", lower_values),
                "",
                cls._format_uint16_array("UPPER_TABLE", upper_values),
                "",
            ]
        )
        path.write_text(contents, encoding="utf-8")


def main() -> None:
    app = CiliaPathDesigner()
    app.mainloop()


if __name__ == "__main__":
    main()
