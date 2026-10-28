"""Render Lecture 08's diagram figures into ../media/.

split_apply_combine.png and result_shapes.png are computed from the lecture's
six-visit table with pandas, and the drawn numbers are asserted against the
values the lecture prints. ssh_laptop_server.png is a plain diagram.
pivot_heatmap.png runs Demo 1's own code cells up to its heatmap cell, so the
figure shows the demo's 100,000 synthetic visits.

Run from the repository root:
    uv run --project 08/demo python 08/scripts/generate_lecture_figures.py
"""

from pathlib import Path

import jupytext
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgba  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

MEDIA_DIR = Path(__file__).resolve().parents[1] / "media"
DEMO1 = Path(__file__).resolve().parents[1] / "demo" / "demo1_groupby_operations.md"

# Okabe-Ito colors, one per clinic
CLINIC_COLORS = {"North": "#0072B2", "South": "#E69F00", "East": "#009E73"}
GREY = "#9A9A9A"
INK = "#222222"

# The lecture's `visits` table (08/README.md, "Aggregating Each Group")
visits = pd.DataFrame({
    "clinic": ["North", "North", "North", "South", "South", "East"],
    "visit_type": ["New", "Follow-up", "Follow-up", "New", "New", "Follow-up"],
    "patient_id": ["P01", "P02", "P01", "P03", "P04", "P05"],
    "wait_min": [12, 15, 6, 20, 30, 9],
    "satisfaction": [4.0, None, 5.0, 3.0, None, 4.0],
})


def draw_table(ax, x, y, widths, header, rows, colors, align, faded=(), row_h=0.55, fontsize=11):
    """Draw a DataFrame-like table with its top-left corner at (x, y); return the row centers."""
    total = sum(widths)
    ax.add_patch(Rectangle((x, y - row_h), total, row_h, facecolor="#EEEEEE", edgecolor="none"))
    edges = [x]
    for w in widths:
        edges.append(edges[-1] + w)
    pad = 0.12
    for i, label in enumerate(header):
        tx = edges[i] + pad if align[i] == "left" else edges[i + 1] - pad
        ax.text(tx, y - row_h / 2, label, ha=align[i], va="center", fontsize=fontsize,
                fontweight="bold", family="monospace", color=INK)
    centers = []
    for r, (row, color) in enumerate(zip(rows, colors)):
        top = y - row_h * (r + 1)
        cy = top - row_h / 2
        centers.append(cy)
        is_faded = r in faded
        fill = "#F4F4F4" if is_faded or color is None else to_rgba(color, 0.18)
        ax.add_patch(Rectangle((x, top - row_h), total, row_h, facecolor=fill, edgecolor="white", linewidth=1.5))
        if color is not None and not is_faded:
            ax.add_patch(Rectangle((x, top - row_h), 0.07, row_h, facecolor=color, edgecolor="none"))
        for i, value in enumerate(row):
            tx = edges[i] + pad if align[i] == "left" else edges[i + 1] - pad
            ax.text(tx, cy, value, ha=align[i], va="center", fontsize=fontsize, family="monospace",
                    color=GREY if is_faded else INK)
        if is_faded:
            ax.plot([x + 0.05, x + total - 0.05], [cy, cy], color=GREY, linewidth=1.2)
    ax.plot([x, x + total], [y - row_h, y - row_h], color=INK, linewidth=0.8)
    return centers


def arrow(ax, start, end, color=INK, style="-|>", lw=1.4, rad=0.0):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=14, color=color,
                                 linewidth=lw, connectionstyle=f"arc3,rad={rad}"))


def blank_canvas(width, height, scale=0.62):
    fig = plt.figure(figsize=(width * scale, height * scale), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])  # data units map to inches at a fixed scale
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.axis("off")
    return fig, ax


def save(fig, filename):
    fig.savefig(MEDIA_DIR / filename, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Generated:", filename)


def split_apply_combine():
    """Mean wait per clinic from the six visits, drawn as split, apply, and combine."""
    groups = {clinic: visits.loc[visits["clinic"] == clinic, "wait_min"].tolist()
              for clinic in ["North", "South", "East"]}
    means = visits.groupby("clinic")["wait_min"].mean()
    assert groups == {"North": [12, 15, 6], "South": [20, 30], "East": [9]}
    assert means.to_dict() == {"East": 9.0, "North": 11.0, "South": 25.0}

    fig, ax = blank_canvas(22, 7.4)
    row_h = 0.55
    top = 6.4
    stage_y = 7.05
    split_x, apply_x, combine_x = 5.7, 11.3, 16.2
    for x, label in [(0.2, "INPUT: one row per visit"), (split_x, "SPLIT by clinic"),
                     (apply_x, "APPLY: mean wait"), (combine_x, "COMBINE: one row per clinic")]:
        ax.text(x, stage_y, label, fontsize=12.5, fontweight="bold", color=INK, va="center")

    rows = visits[["clinic", "wait_min"]].astype(str).values.tolist()
    colors = [CLINIC_COLORS[c] for c in visits["clinic"]]
    input_centers = draw_table(ax, 0.2, top, [1.9, 1.9], ["clinic", "wait_min"], rows, colors,
                               ["left", "right"])

    # Each group sits in its own small table; the gaps make the three groups visible
    combine_rows = sorted(groups)
    group_top = top
    for clinic, waits in groups.items():
        color = CLINIC_COLORS[clinic]
        centers = draw_table(ax, split_x, group_top, [1.9, 1.9], ["clinic", "wait_min"],
                             [[clinic, str(w)] for w in waits], [color] * len(waits), ["left", "right"])
        group_top -= row_h * (len(waits) + 1) + 0.35
        mid = sum(centers) / len(centers)
        first = list(visits["clinic"]).index(clinic)
        arrow(ax, (4.25, input_centers[first]), (split_x - 0.1, centers[0]), color=color)
        arrow(ax, (split_x + 3.95, mid), (apply_x - 0.1, mid), color=color)
        ax.text(apply_x, mid, f"mean = {means[clinic]:.1f} min", fontsize=12, va="center",
                family="monospace", color=color, fontweight="bold")
        target = top - row_h * (1.5 + combine_rows.index(clinic))
        arrow(ax, (apply_x + 3.45, mid), (combine_x - 0.1, target), color=color)

    combined = [[clinic, f"{value:.1f}"] for clinic, value in means.items()]
    draw_table(ax, combine_x, top, [1.9, 2.4], ["clinic", "wait_min"], combined,
               [CLINIC_COLORS[c] for c in means.index], ["left", "right"])
    ax.text(combine_x, top - row_h * 4 - 0.35, "sorted by clinic name;\nwait_min in minutes", fontsize=10.5,
            va="top", color="#555555")
    save(fig, "split_apply_combine.png")


def result_shapes():
    """agg, transform, filter, and apply on the six visits: rows returned and their grain."""
    grouped = visits.groupby("clinic")
    agg = grouped["wait_min"].mean()
    trans = grouped["wait_min"].transform("mean")
    kept = grouped.filter(lambda g: len(g) >= 2)
    top2 = grouped.apply(lambda g: g.nlargest(2, "wait_min"), include_groups=False)
    assert agg.to_dict() == {"East": 9.0, "North": 11.0, "South": 25.0}
    assert trans.tolist() == [11.0, 11.0, 11.0, 25.0, 25.0, 9.0]
    assert kept.index.tolist() == [0, 1, 2, 3, 4]
    assert top2.index.tolist() == [("East", 5), ("North", 1), ("North", 0), ("South", 4), ("South", 3)]
    assert top2["wait_min"].tolist() == [9, 15, 12, 30, 20]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.4))
    fig.patch.set_facecolor("white")
    panels = [
        (".agg('mean') on wait_min", f"{len(agg)} rows: one per clinic"),
        (".transform('mean') on wait_min", f"{len(trans)} rows: one per visit, same index as visits"),
        (".filter(lambda g: len(g) >= 2)", f"{len(kept)} rows: East's single visit dropped"),
        (".apply(lambda g: g.nlargest(2, 'wait_min'))",
         f"{len(top2)} rows: up to two per clinic, indexed by clinic, then row number"),
    ]
    for ax, (code, note) in zip(axes.flat, panels):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 5.2)
        ax.axis("off")
        ax.text(0, 5.05, code, fontsize=12.5, family="monospace", fontweight="bold", color=INK, va="center")
        ax.text(0, 4.6, note, fontsize=11.5, color="#444444", va="center")

    start = 4.15
    draw_table(axes[0, 0], 0, start, [1.9, 2.4], ["clinic", "wait_min"],
               [[c, f"{v:.1f}"] for c, v in agg.items()], [CLINIC_COLORS[c] for c in agg.index],
               ["left", "right"])

    trans_rows = [[str(i), c, str(w), f"{m:.1f}"]
                  for i, c, w, m in zip(visits.index, visits["clinic"], visits["wait_min"], trans)]
    draw_table(axes[0, 1], 0, start, [0.6, 1.9, 1.9, 3.4], ["", "clinic", "wait_min", "clinic_mean_wait"],
               trans_rows, [CLINIC_COLORS[c] for c in visits["clinic"]],
               ["right", "left", "right", "right"])

    filt_rows = visits[["clinic", "patient_id", "wait_min"]].astype(str).values.tolist()
    filt_rows = [[str(i)] + r for i, r in zip(visits.index, filt_rows)]
    east = [i for i, c in enumerate(visits["clinic"]) if c not in kept["clinic"].tolist()]
    centers = draw_table(axes[1, 0], 0, start, [0.6, 1.9, 2.2, 1.9], ["", "clinic", "patient_id", "wait_min"],
                         filt_rows, [CLINIC_COLORS[c] for c in visits["clinic"]],
                         ["right", "left", "left", "right"], faded=east)
    axes[1, 0].text(6.85, centers[east[0]], "dropped: len(g) is 1", fontsize=10.5, va="center", color=GREY)

    apply_rows, apply_colors, previous = [], [], None
    for (clinic, row), patient, wait in zip(top2.index, top2["patient_id"], top2["wait_min"]):
        apply_rows.append([clinic if clinic != previous else "", str(row), patient, str(wait)])
        apply_colors.append(CLINIC_COLORS[clinic])
        previous = clinic
    draw_table(axes[1, 1], 0, start, [1.9, 0.6, 2.2, 1.9], ["clinic", "", "patient_id", "wait_min"],
               apply_rows, apply_colors, ["left", "right", "left", "right"])

    fig.suptitle("Four results from visits.groupby('clinic')", fontsize=14, fontweight="bold",
                 family="monospace", x=0.01, ha="left", color=INK)
    fig.text(0.01, -0.01, "wait_min and clinic_mean_wait in minutes; colors mark the clinic of each row",
             fontsize=10.5, color="#555555")
    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    save(fig, "result_shapes.png")


def ssh_laptop_server():
    """Where ssh and scp move work: commands and files cross, patient records stay on the server."""
    fig, ax = blank_canvas(21, 8.0)
    laptop, server = "#0072B2", "#009E73"

    def box(x, y, w, h, color, title, prompt):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1,rounding_size=0.25",
                                    facecolor=to_rgba(color, 0.07), edgecolor=color, linewidth=2))
        ax.text(x + 0.3, y + h - 0.45, title, fontsize=14, fontweight="bold", color=color, va="center")
        ax.text(x + 0.3, y + h - 1.05, prompt, fontsize=11, family="monospace", color="#444444", va="center")

    def file(x, y, w, name):
        ax.add_patch(Rectangle((x, y - 0.3), w, 0.6, facecolor="white", edgecolor=INK, linewidth=1.2))
        ax.text(x + 0.15, y, name, fontsize=11, family="monospace", color=INK, va="center")

    box(0.3, 0.4, 5.0, 7.2, laptop, "Your laptop", "you@laptop:~$")
    box(14.0, 0.4, 6.6, 7.2, server, "Analysis server", "user@host:~$ python3 analysis.py")

    # Row 1: ssh opens a shell on the server
    y_ssh = 5.2
    arrow(ax, (5.6, y_ssh), (13.8, y_ssh), color=INK, style="<|-|>", lw=1.8)
    ax.text(9.7, y_ssh + 0.45, "ssh user@host", fontsize=12, family="monospace", ha="center",
            fontweight="bold", color=INK)
    ax.text(9.7, y_ssh - 0.45, "encrypted shell: commands run on the server", fontsize=10.5, ha="center",
            color="#444444")

    # Row 2: scp uploads data.csv
    y_up = 3.5
    file(0.9, y_up, 3.8, "data.csv")
    file(14.6, y_up, 5.4, "~/data/data.csv")
    arrow(ax, (4.85, y_up), (14.45, y_up), color="#E69F00", lw=2)
    ax.text(9.65, y_up + 0.4, "scp data.csv user@host:~/data/", fontsize=11, family="monospace",
            ha="center", color=INK)

    # Row 3: scp downloads the permitted result
    y_down = 2.1
    file(0.9, y_down, 3.8, "analysis.csv")
    file(14.6, y_down, 5.4, "~/results/analysis.csv")
    arrow(ax, (14.45, y_down), (4.85, y_down), color="#CC79A7", lw=2)
    ax.text(9.65, y_down + 0.4, "scp user@host:~/results/analysis.csv ./", fontsize=11, family="monospace",
            ha="center", color=INK)

    # Patient records never cross the gap
    ax.add_patch(FancyBboxPatch((14.6, 0.7), 5.4, 0.7, boxstyle="round,pad=0.05,rounding_size=0.15",
                                facecolor=to_rgba("#D55E00", 0.12), edgecolor="#D55E00", linewidth=1.5))
    ax.text(17.3, 1.05, "patient records (PHI) stay here", fontsize=11, ha="center", va="center",
            color="#D55E00", fontweight="bold")
    save(fig, "ssh_laptop_server.png")


def pivot_heatmap():
    """Execute Demo 1's code cells in order and save the figure its heatmap cell draws."""
    env = {"display": lambda *a, **k: None, "__name__": "__demo__"}
    for cell in jupytext.read(DEMO1).cells:
        if cell.cell_type != "code":
            continue
        # Drop notebook magics such as the %pip install line
        src = "\n".join(line for line in cell.source.splitlines() if not line.lstrip().startswith("%"))
        is_heatmap = "sns.heatmap(wait_pivot" in src
        exec(compile(src.replace("plt.show()", ""), str(DEMO1), "exec"), env)
        if is_heatmap:
            break
    else:
        raise SystemExit(f"No sns.heatmap(wait_pivot cell in {DEMO1}")
    print(env["wait_pivot"].round(1))
    plt.gcf().savefig(MEDIA_DIR / "pivot_heatmap.png", dpi=150, facecolor="white")
    plt.close("all")
    print("Generated: pivot_heatmap.png")


if __name__ == "__main__":
    MEDIA_DIR.mkdir(exist_ok=True)
    split_apply_combine()
    result_shapes()
    ssh_laptop_server()
    pivot_heatmap()
