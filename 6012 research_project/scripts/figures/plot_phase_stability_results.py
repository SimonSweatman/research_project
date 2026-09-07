"""Create a four-panel phase-shift figure with sample-SD error bars.

Repeat-level results are read from ``doe_phase_analysis_outputs/run_summary.csv``.
Only Pillow is required, avoiding a dependency on Matplotlib.
"""

from __future__ import annotations

import csv
import math
import statistics
from html import escape
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


SCRIPT_DIR = Path(__file__).resolve().parent
SCRIPTS_DIR = SCRIPT_DIR.parent
INPUT_CSV = SCRIPTS_DIR / "doe_phase_analysis_outputs" / "run_summary.csv"
OUTPUT_DIR = SCRIPT_DIR / "outputs"

PHASES = [72, 90, 120, 180, 240, 270, 288]
REPEATS = 3
METRICS = [
    ("Mean Net Advance per Cycle (mm)", "Advance per cycle", "Advance per cycle (mm)", False),
    ("Total Backward Travel (mm)", "Total backward travel", "Total backward travel (mm)", True),
    ("Vertical Centroid Peak-to-Peak (mm)", "Vertical peak-to-peak movement", "Vertical peak-to-peak (mm)", False),
    ("Pitch Peak-to-Peak (deg)", "Pitch peak-to-peak movement", "Pitch peak-to-peak (°)", False),
]

WIDTH, HEIGHT = 3200, 2200
WHITE, BLACK, MUTED = "#FFFFFF", "#202020", "#555555"
GRID, BLUE, ERROR = "#D9D9D9", "#4F81BD", "#333333"


def get_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = ["arialbd.ttf", "Arial Bold.ttf"] if bold else ["arial.ttf", "Arial.ttf"]
    for name in names:
        path = Path("C:/Windows/Fonts") / name
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default(size=size)


TITLE_FONT = get_font(50, True)
PANEL_TITLE_FONT = get_font(39, True)
PANEL_LABEL_FONT = get_font(38, True)
AXIS_FONT = get_font(35)
TICK_FONT = get_font(30)
NOTE_FONT = get_font(30)


def load_summary() -> dict[str, dict[str, list[float]]]:
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Phase result file not found: {INPUT_CSV}")
    grouped = {phase: {metric[0]: [] for metric in METRICS} for phase in PHASES}
    with INPUT_CSV.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        required = {"Phase Shift Command (deg)", *(metric[0] for metric in METRICS)}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise KeyError(f"Missing required columns: {', '.join(sorted(missing))}")
        for row in reader:
            try:
                phase = int(float(row["Phase Shift Command (deg)"]))
            except (TypeError, ValueError):
                continue
            if phase not in grouped:
                continue
            for column, *_ in METRICS:
                grouped[phase][column].append(float(row[column]))

    summary: dict[str, dict[str, list[float]]] = {}
    for column, *_ in METRICS:
        means, standard_deviations = [], []
        for phase in PHASES:
            values = grouped[phase][column]
            if len(values) != REPEATS:
                raise ValueError(f"Expected n={REPEATS} for {phase}° {column}; found n={len(values)}")
            means.append(statistics.mean(values))
            standard_deviations.append(statistics.stdev(values))
        summary[column] = {"mean": means, "sd": standard_deviations}
    return summary


def nice_axis(means: list[float], errors: list[float], include_zero: bool) -> tuple[float, float, float]:
    lower = min(mean - error for mean, error in zip(means, errors))
    upper = max(mean + error for mean, error in zip(means, errors))
    spread = max(upper - lower, abs(upper) * 0.08, 1e-9)
    lower -= spread * 0.10
    upper += spread * 0.10
    if include_zero:
        lower = min(0.0, lower)
    raw_step = (upper - lower) / 5
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / 10**exponent
    step = next(value * 10**exponent for value in (1, 2, 2.5, 5, 10) if fraction <= value)
    axis_min = 0.0 if include_zero else math.floor(lower / step) * step
    axis_max = math.ceil(upper / step) * step
    return axis_min, axis_max, step


def axis_ticks(axis_min: float, axis_max: float, step: float) -> list[float]:
    return [axis_min + index * step for index in range(int(round((axis_max - axis_min) / step)) + 1)]


def tick_text(value: float, step: float) -> str:
    if step >= 1:
        return f"{value:.0f}"
    decimals = max(1, -math.floor(math.log10(step)))
    return f"{value:.{decimals}f}"


def centred(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, selected_font: ImageFont.ImageFont, fill: str = BLACK) -> None:
    draw.text(xy, text, font=selected_font, fill=fill, anchor="mm")


def vertical_label(image: Image.Image, xy: tuple[float, float], text: str) -> None:
    box = AXIS_FONT.getbbox(text)
    label = Image.new("RGBA", (box[2] - box[0] + 20, box[3] - box[1] + 20), (0, 0, 0, 0))
    ImageDraw.Draw(label).text((10 - box[0], 10 - box[1]), text, font=AXIS_FONT, fill=BLACK)
    label = label.rotate(90, expand=True, resample=Image.Resampling.BICUBIC)
    image.alpha_composite(label, (int(xy[0] - label.width / 2), int(xy[1] - label.height / 2)))


def draw_panel(
    image: Image.Image,
    draw: ImageDraw.ImageDraw,
    bounds: tuple[int, int, int, int],
    letter: str,
    metric: tuple[str, str, str, bool],
    means: list[float],
    errors: list[float],
) -> None:
    _, title, ylabel, include_zero = metric
    left, top, right, bottom = bounds
    plot_left, plot_right = left + 205, right - 42
    plot_top, plot_bottom = top + 125, bottom - 145
    centred(draw, ((left + right) / 2, top + 42), title, PANEL_TITLE_FONT)
    draw.text((left + 8, top + 15), f"({letter})", font=PANEL_LABEL_FONT, fill=BLACK)

    axis_min, axis_max, step = nice_axis(means, errors, include_zero)

    def sx(phase: float) -> float:
        return plot_left + (phase - PHASES[0]) / (PHASES[-1] - PHASES[0]) * (plot_right - plot_left)

    def sy(value: float) -> float:
        return plot_bottom - (value - axis_min) / (axis_max - axis_min) * (plot_bottom - plot_top)

    for tick in axis_ticks(axis_min, axis_max, step):
        y = sy(tick)
        draw.line((plot_left, y, plot_right, y), fill=GRID, width=2)
        draw.text((plot_left - 18, y), tick_text(tick, step), font=TICK_FONT, fill=MUTED, anchor="rm")
    draw.line((plot_left, plot_top, plot_left, plot_bottom), fill=BLACK, width=3)
    draw.line((plot_left, plot_bottom, plot_right, plot_bottom), fill=BLACK, width=3)

    for phase in PHASES:
        x = sx(phase)
        draw.line((x, plot_bottom, x, plot_bottom + 10), fill=BLACK, width=2)
        draw.text((x, plot_bottom + 20), str(phase), font=TICK_FONT, fill=MUTED, anchor="ma")

    points = [(sx(phase), sy(mean)) for phase, mean in zip(PHASES, means)]
    draw.line(points, fill=BLUE, width=6, joint="curve")
    for (x, y), mean, error in zip(points, means, errors):
        upper, lower = sy(mean + error), sy(mean - error)
        draw.line((x, upper, x, lower), fill=ERROR, width=4)
        draw.line((x - 13, upper, x + 13, upper), fill=ERROR, width=4)
        draw.line((x - 13, lower, x + 13, lower), fill=ERROR, width=4)
        draw.ellipse((x - 10, y - 10, x + 10, y + 10), fill=BLUE, outline=WHITE, width=3)

    centred(draw, ((plot_left + plot_right) / 2, bottom - 47), "Phase shift (°)", AXIS_FONT)
    vertical_label(image, (left + 53, (plot_top + plot_bottom) / 2), ylabel)


def write_svg(summary: dict[str, dict[str, list[float]]], output_path: Path) -> None:
    panel_width, panel_height = 1425, 850
    starts = [(130, 170), (1650, 170), (130, 1110), (1650, 1110)]
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}">',
        f'<rect width="100%" height="100%" fill="{WHITE}"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#202020}.title{font-size:50px;font-weight:700}'
        '.pt{font-size:39px;font-weight:700}.pl{font-size:38px;font-weight:700}'
        '.al{font-size:35px}.tick{font-size:30px;fill:#555}.note{font-size:30px;fill:#555}</style>',
        f'<text x="{WIDTH/2}" y="75" text-anchor="middle" class="title">Effect of phase shift on transport and stability</text>',
    ]
    for index, (metric, (left, top)) in enumerate(zip(METRICS, starts)):
        column, title, ylabel, include_zero = metric
        right, bottom = left + panel_width, top + panel_height
        plot_left, plot_right = left + 205, right - 42
        plot_top, plot_bottom = top + 125, bottom - 145
        means, errors = summary[column]["mean"], summary[column]["sd"]
        axis_min, axis_max, step = nice_axis(means, errors, include_zero)

        def sx(phase: float) -> float:
            return plot_left + (phase - PHASES[0]) / (PHASES[-1] - PHASES[0]) * (plot_right - plot_left)

        def sy(value: float) -> float:
            return plot_bottom - (value - axis_min) / (axis_max - axis_min) * (plot_bottom - plot_top)

        svg.append(f'<text x="{left + 8}" y="{top + 45}" class="pl">({chr(65 + index)})</text>')
        svg.append(f'<text x="{(left + right)/2}" y="{top + 55}" text-anchor="middle" class="pt">{escape(title)}</text>')
        for tick in axis_ticks(axis_min, axis_max, step):
            y = sy(tick)
            svg.append(f'<line x1="{plot_left}" y1="{y:.2f}" x2="{plot_right}" y2="{y:.2f}" stroke="{GRID}" stroke-width="2"/>')
            svg.append(f'<text x="{plot_left - 18}" y="{y + 10:.2f}" text-anchor="end" class="tick">{tick_text(tick, step)}</text>')
        svg.append(f'<line x1="{plot_left}" y1="{plot_top}" x2="{plot_left}" y2="{plot_bottom}" stroke="{BLACK}" stroke-width="3"/>')
        svg.append(f'<line x1="{plot_left}" y1="{plot_bottom}" x2="{plot_right}" y2="{plot_bottom}" stroke="{BLACK}" stroke-width="3"/>')
        for phase in PHASES:
            x = sx(phase)
            svg.append(f'<line x1="{x:.2f}" y1="{plot_bottom}" x2="{x:.2f}" y2="{plot_bottom + 10}" stroke="{BLACK}" stroke-width="2"/>')
            svg.append(f'<text x="{x:.2f}" y="{plot_bottom + 48}" text-anchor="middle" class="tick">{phase}</text>')
        points = " ".join(f"{sx(phase):.2f},{sy(mean):.2f}" for phase, mean in zip(PHASES, means))
        svg.append(f'<polyline points="{points}" fill="none" stroke="{BLUE}" stroke-width="6" stroke-linejoin="round"/>')
        for phase, mean, error in zip(PHASES, means, errors):
            x, y, upper, lower = sx(phase), sy(mean), sy(mean + error), sy(mean - error)
            svg.extend([
                f'<line x1="{x:.2f}" y1="{upper:.2f}" x2="{x:.2f}" y2="{lower:.2f}" stroke="{ERROR}" stroke-width="4"/>',
                f'<line x1="{x-13:.2f}" y1="{upper:.2f}" x2="{x+13:.2f}" y2="{upper:.2f}" stroke="{ERROR}" stroke-width="4"/>',
                f'<line x1="{x-13:.2f}" y1="{lower:.2f}" x2="{x+13:.2f}" y2="{lower:.2f}" stroke="{ERROR}" stroke-width="4"/>',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="10" fill="{BLUE}" stroke="{WHITE}" stroke-width="3"/>',
            ])
        svg.append(f'<text x="{(plot_left + plot_right)/2}" y="{bottom - 38}" text-anchor="middle" class="al">Phase shift (°)</text>')
        y_centre = (plot_top + plot_bottom) / 2
        svg.append(f'<text x="{left + 48}" y="{y_centre}" text-anchor="middle" class="al" transform="rotate(-90 {left + 48} {y_centre})">{escape(ylabel)}</text>')
    svg.append(f'<text x="{WIDTH/2}" y="{HEIGHT - 38}" text-anchor="middle" class="note">Points show the mean of three repeats; error bars show ±1 sample SD (n=3).</text>')
    svg.append("</svg>")
    output_path.write_text("\n".join(svg), encoding="utf-8")


def create_outputs(summary: dict[str, dict[str, list[float]]]) -> tuple[Path, Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGBA", (WIDTH, HEIGHT), WHITE)
    draw = ImageDraw.Draw(image)
    centred(draw, (WIDTH / 2, 72), "Effect of phase shift on transport and stability", TITLE_FONT)
    starts = [(130, 170), (1650, 170), (130, 1110), (1650, 1110)]
    for index, (metric, (left, top)) in enumerate(zip(METRICS, starts)):
        column = metric[0]
        draw_panel(image, draw, (left, top, left + 1425, top + 850), chr(65 + index), metric, summary[column]["mean"], summary[column]["sd"])
    centred(draw, (WIDTH / 2, HEIGHT - 43), "Points show the mean of three repeats; error bars show ±1 sample SD (n=3).", NOTE_FONT, MUTED)

    png_path = OUTPUT_DIR / "phase_secondary_stability_results.png"
    svg_path = OUTPUT_DIR / "phase_secondary_stability_results.svg"
    csv_path = OUTPUT_DIR / "phase_secondary_stability_summary.csv"
    image.convert("RGB").save(png_path, dpi=(400, 400), optimize=True)
    write_svg(summary, svg_path)
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.writer(output)
        writer.writerow(["Phase shift (deg)", *(item for metric in METRICS for item in (f"{metric[1]} mean", f"{metric[1]} sample SD"))])
        for phase_index, phase in enumerate(PHASES):
            row: list[float | int] = [phase]
            for column, *_ in METRICS:
                row.extend([summary[column]["mean"][phase_index], summary[column]["sd"][phase_index]])
            writer.writerow(row)
    return png_path, svg_path, csv_path


def main() -> None:
    summary = load_summary()
    outputs = create_outputs(summary)
    for index, phase in enumerate(PHASES):
        values = [f"{title}={summary[column]['mean'][index]:.2f} ± {summary[column]['sd'][index]:.2f}" for column, title, *_ in METRICS]
        print(f"{phase:>3}° | " + " | ".join(values))
    print("\nCreated:")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
