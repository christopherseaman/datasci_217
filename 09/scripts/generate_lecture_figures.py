"""Render Lecture 09's generated figures into ../media/.

Each figure uses a lecture snippet's data or the demo data it rebuilds below,
and the drawn numbers are asserted against the values the lecture and demos print.

Run from the repository root:
    uv run --project 09/demo python 09/scripts/generate_lecture_figures.py [FIGURE_FUNCTION ...]
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.patches import ConnectionPatch, FancyArrowPatch, Rectangle  # noqa: E402
from matplotlib.ticker import FuncFormatter, MultipleLocator  # noqa: E402

MEDIA_DIR = Path(__file__).resolve().parents[1] / "media"

# Okabe-Ito colors
BLUE, ORANGE, GREEN, VERMILLION, PURPLE = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
GREY, INK = "#8C8C8C", "#222222"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titlelocation": "left",
})


def save(fig, name):
    fig.savefig(MEDIA_DIR / name, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", name)


def clock(hours, _pos=None):
    """Format decimal hours as HH:MM for a one-day numeric axis."""
    minutes = int(round(hours * 60)) % (24 * 60)
    return f"{minutes // 60:02d}:{minutes % 60:02d}"


def hours_of(t):
    """Clock time as decimal hours (a Timestamp or a DatetimeIndex)."""
    return t.hour + t.minute / 60


def clock_axis(ax, major=1, minor=None):
    ax.xaxis.set_major_locator(MultipleLocator(major))
    ax.xaxis.set_major_formatter(FuncFormatter(clock))
    if minor:
        ax.xaxis.set_minor_locator(MultipleLocator(minor))


def hide_y(ax):
    ax.spines["left"].set_visible(False)
    ax.set_yticks([])


def arrow(ax, start, end, color=INK, lw=1.4, rad=0.0, style="-|>", **kwargs):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle=style, mutation_scale=13, color=color,
                                 linewidth=lw, connectionstyle=f"arc3,rad={rad}", **kwargs))


# ---------------------------------------------------------------- demo data


def demo2_monitor():
    """Demo 2, section 1: one patient's hourly monitor export with a fever and a disconnection."""
    rng = np.random.default_rng(42)
    hours = pd.date_range("2024-01-01", periods=24 * 28, freq="h")
    monitor = pd.DataFrame({
        "heart_rate": rng.normal(78, 4, len(hours)).round().astype(int),
        "temperature": (98.4 + rng.normal(0, 0.2, len(hours))).round(1),
        "spo2": rng.integers(95, 100, len(hours)),
    }, index=hours)
    fever = (monitor.index >= "2024-01-15") & (monitor.index < "2024-01-18")
    monitor.loc[fever, "temperature"] += 2.5
    monitor.loc[fever, "heart_rate"] += 15
    unplugged = pd.date_range("2024-01-05 08:00", periods=10, freq="h")
    monitor = monitor[~monitor.index.isin(unplugged)]
    assert monitor.shape == (662, 3)
    return monitor


def home_weights():
    """Demo 2, section 10 and Demo 3, section 9: one patient's daily home weights, January to March 2024."""
    rng = np.random.default_rng(42)
    days = pd.date_range("2024-01-01", "2024-03-31", freq="D")
    weights = pd.Series(82 + rng.normal(0, 0.15, len(days)), index=days)
    weights.loc["2024-03-09":"2024-03-15"] += [0.5, 1.9, 2.6, 2.6, 1.9, 1.1, 0.4]
    weights = weights.round(1)
    assert weights.shape == (91,) and weights.max() == 84.4
    return weights


def demo3_vitals():
    """Demo 3, section 1: three patients' charted heart rates, localized in New York and converted to UTC."""
    raw = pd.DataFrame({
        "patient_id": ["P01"] * 5 + ["P02"] * 5 + ["P03"] * 4,
        "recorded_local": [
            "2024-11-02 22:00", "2024-11-02 23:30", "2024-11-03 00:30", "2024-11-03 03:00", "2024-11-03 04:00",
            "2024-11-02 22:30", "2024-11-03 00:00", "2024-11-03 01:30", "2024-11-03 02:00", "2024-11-03 03:30",
            "2024-11-02 23:00", "2024-11-03 00:15", "2024-11-03 02:15", "2024-11-03 04:00",
        ],
        "heart_rate": [88, 94, 101, 112, 118, 72, 70, 68, 71, 69, 104, 99, 95, 92],
    })
    local = pd.to_datetime(raw["recorded_local"], format="%Y-%m-%d %H:%M")
    raw["recorded_at"] = (local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
                          .dt.tz_convert("UTC"))
    vitals = (raw.dropna(subset=["recorded_at"])[["patient_id", "recorded_at", "heart_rate"]]
              .sort_values(["patient_id", "recorded_at"]).reset_index(drop=True))
    assert len(vitals) == 13
    return vitals


def demo3_ili():
    """Demo 3, section 7: three years of daily flu-like illness visits with a February 1 to 14, 2023 outage."""
    rng = np.random.default_rng(42)
    days = pd.date_range("2021-01-01", "2023-12-31", freq="D")
    winter_wave = 20 * np.cos(2 * np.pi * (days.dayofyear - 15) / 365.25)
    noise = rng.normal(0, 4, len(days))
    ili = pd.Series((40 + winter_wave + noise).round().astype(int), index=days)
    ili = ili[~ili.index.isin(pd.date_range("2023-02-01", "2023-02-14", freq="D"))]
    assert len(ili) == 1081
    return ili


# ---------------------------------------------------------------- block 1


def forms_of_time():
    """Forms of Time: a timestamp, a period, and elapsed time on one admission."""
    admitted = pd.Timestamp("2024-02-29 20:15")
    draw = pd.Timestamp("2024-03-01 08:15")
    assert (draw - admitted) / pd.Timedelta(hours=1) == 12
    march = pd.Period("2024-03", freq="M")
    left, right = pd.Timestamp("2024-02-29 16:00"), pd.Timestamp("2024-03-02 12:00")
    x = {name: mdates.date2num(t) for name, t in
         {"admitted": admitted, "draw": draw, "march": march.start_time, "left": left, "right": right}.items()}

    fig, ax = plt.subplots(figsize=(9, 2.9))
    ax.axvspan(x["march"], x["right"], color=BLUE, alpha=0.12, lw=0)
    ax.text(x["right"] - 0.04, 0.9, "Period: March 2024\n(continues to March 31)", ha="right", va="top",
            color=BLUE, fontsize=10, fontweight="bold")
    ax.axvline(x["admitted"], color=INK, lw=1.2)
    ax.text(x["admitted"] - 0.03, 0.9, "Admitted\nFeb 29, 20:15", ha="right", va="top", color=INK)
    ax.axvline(x["draw"], color=VERMILLION, lw=2)
    ax.plot(x["draw"], 0.62, "o", color=VERMILLION, ms=8)
    ax.text(x["draw"] + 0.03, 0.62, "Timestamp: 2024-03-01 08:15\n(blood draw)", va="center",
            color=VERMILLION, fontweight="bold")
    arrow(ax, (x["admitted"], 0.22), (x["draw"], 0.22), color=GREEN, style="<|-|>", lw=1.6)
    ax.text((x["admitted"] + x["draw"]) / 2, 0.29, "Elapsed time: 12 hours", ha="center", va="bottom",
            color=GREEN, fontweight="bold")

    ax.set_xlim(x["left"], x["right"])
    ax.set_ylim(0, 1)
    hide_y(ax)
    ax.xaxis.set_major_locator(mdates.HourLocator(byhour=[0, 12]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d\n%H:%M"))
    ax.set_xlabel("Calendar time")
    hours_axis = ax.secondary_xaxis(-0.42, functions=(lambda d: (d - x["admitted"]) * 24,
                                                      lambda h: h / 24 + x["admitted"]))
    hours_axis.xaxis.set_major_locator(MultipleLocator(12))
    hours_axis.set_xlabel("Hours since admission")
    save(fig, "forms_of_time.png")


def regular_irregular():
    """Regular and Irregular Series: Demo 2's hourly monitor against Demo 3's charted readings."""
    jan5 = demo2_monitor().loc["2024-01-05", "heart_rate"]
    assert len(jan5) == 14
    p01_local = pd.to_datetime(["2024-11-02 22:00", "2024-11-02 23:30", "2024-11-03 00:30",
                                "2024-11-03 03:00", "2024-11-03 04:00"])
    p01_utc = p01_local.tz_localize("America/New_York").tz_convert("UTC").tz_localize(None)
    assert list(p01_utc.strftime("%H:%M")) == ["02:00", "03:30", "04:30", "08:00", "09:00"]
    p01_hr = [88, 94, 101, 112, 118]

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.2), gridspec_kw={"hspace": 0.75})
    ax = axes[0]
    xs = hours_of(jan5.index)
    ax.axvspan(7.5, 17.5, color=GREY, alpha=0.15, lw=0)
    ax.text(12.5, 92, "Unplugged: no rows\n08:00 to 17:00 (10 hours)", ha="center", va="top", color=INK)
    ax.plot(xs, jan5.values, "o", color=BLUE, ms=5)
    ax.plot(xs, np.full(len(xs), 0.06), "|", color=BLUE, ms=14, mew=2, transform=ax.get_xaxis_transform())
    ax.set(title="Regular: one reading every hour, 14 of 24 hours present", xlim=(-0.5, 23.5),
           ylim=(60, 95), xlabel="Time, January 5, 2024", ylabel="Heart rate (bpm)")
    clock_axis(ax, major=2, minor=1)

    ax = axes[1]
    xs = hours_of(p01_utc)
    ax.plot(xs, p01_hr, "o", color=ORANGE, ms=6)
    ax.plot(xs, np.full(len(xs), 0.06), "|", color=ORANGE, ms=14, mew=2, transform=ax.get_xaxis_transform())
    for a, b in zip(xs[:-1], xs[1:]):
        ax.annotate("", (b, 0.2), (a, 0.2), xycoords=ax.get_xaxis_transform(),
                    arrowprops={"arrowstyle": "<->", "color": GREY, "lw": 1})
        ax.text((a + b) / 2, 0.24, f"{b - a:g} h", ha="center", va="bottom", color=INK,
                transform=ax.get_xaxis_transform())
    ax.set(title="Irregular: charted when a nurse took a reading, 1 to 3.5 hours apart", xlim=(1, 10),
           ylim=(60, 125), xlabel="Time (UTC), November 3, 2024", ylabel="Heart rate (bpm)")
    clock_axis(ax, major=1)
    save(fig, "regular_irregular.png")


def icu_panel_hr_utc():
    """Single Series and Panels: Demo 3's three-patient panel in UTC, one line per patient_id."""
    vitals = demo3_vitals()
    assert (vitals["recorded_at"].dt.strftime("%Y-%m-%d") == "2024-11-03").all()
    hr = vitals.groupby("patient_id")["heart_rate"].agg(["first", "last", "min", "max"])
    assert hr.loc["P01", ["first", "last"]].tolist() == [88, 118]
    assert hr.loc["P02", ["min", "max"]].tolist() == [69, 72]
    assert hr.loc["P03", ["first", "last"]].tolist() == [104, 92]

    color = {"P01": BLUE, "P02": ORANGE, "P03": GREEN}
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axhline(100, color=VERMILLION, ls="--", lw=1.2)
    ax.text(1.85, 99, "Tachycardia (100 bpm)", color=VERMILLION, va="top", fontsize=9)
    for pid, group in vitals.groupby("patient_id"):
        xs = hours_of(group["recorded_at"].dt)
        ax.plot(xs, group["heart_rate"], "-o", color=color[pid], ms=6, lw=1.8)
        ax.text(xs.iloc[-1] + 0.12, group["heart_rate"].iloc[-1], pid, color=color[pid], va="center",
                fontweight="bold")
    ax.set(xlim=(1.75, 9.6), ylim=(60, 125), xlabel="Time (UTC), November 3, 2024", ylabel="Heart rate (bpm)")
    clock_axis(ax, major=1)
    save(fig, "icu_panel_hr_utc.png")


def text_vs_datetime_axis():
    """pandas DatetimeIndex: the same four weights on a text axis and a datetime axis."""
    weight = pd.Series([81.9, 82.1, 82.3, 83.4], index=["3/9/2024", "3/10/2024", "3/12/2024", "3/20/2024"])
    as_text = weight.sort_index()
    assert list(as_text.index) == ["3/10/2024", "3/12/2024", "3/20/2024", "3/9/2024"]
    as_dates = weight.set_axis(pd.to_datetime(weight.index, format="%m/%d/%Y")).sort_index()
    assert (as_dates.index[-1] - as_dates.index[0]).days == 11

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4), sharey=True, gridspec_kw={"wspace": 0.12})
    ax = axes[0]
    ax.plot(range(4), as_text.values, "-o", color=GREY, mfc=VERMILLION, mec=VERMILLION, ms=7)
    ax.set_xticks(range(4), [f"'{d}'" for d in as_text.index])
    ax.set(title="As text: sorted character by character,\nevenly spaced", xlim=(-0.4, 3.4),
           xlabel="recorded_at (dtype str)", ylabel="Weight (kg)", ylim=(81.6, 83.7))
    ax = axes[1]
    xd = mdates.date2num(as_dates.index)
    ax.plot(xd, as_dates.values, "-o", color=GREY, mfc=BLUE, mec=BLUE, ms=7)
    ax.set_xticks(xd, as_dates.index.day.astype(str))
    ax.xaxis.set_minor_locator(MultipleLocator(1))
    ax.annotate("", (xd[3], 81.8), (xd[2], 81.8), arrowprops={"arrowstyle": "<->", "color": GREY})
    ax.text((xd[2] + xd[3]) / 2, 81.84, "8 days", ha="center", va="bottom")
    ax.set(title="After pd.to_datetime(): date order,\ntrue spacing", xlim=(xd[0] - 0.8, xd[3] + 0.8),
           xlabel="recorded_at (datetime64), day of March 2024")
    axes[0].yaxis.set_major_locator(MultipleLocator(0.5))
    save(fig, "text_vs_datetime_axis.png")


def date_range_aliases():
    """Date Ranges and Frequencies: W-MON, MS, and ME over January to March 2024."""
    rows = {alias: pd.date_range("2024-01-01", "2024-03-31", freq=alias) for alias in ["W-MON", "MS", "ME"]}
    assert len(rows["W-MON"]) == 13 and rows["W-MON"][:3].day.tolist() == [1, 8, 15]
    assert rows["MS"].day.tolist() == [1, 1, 1] and rows["ME"].day.tolist() == [31, 29, 31]

    fig, ax = plt.subplots(figsize=(9, 2.8))
    colors = {"W-MON": BLUE, "MS": GREEN, "ME": ORANGE}
    names = {"W-MON": "freq='W-MON'\nevery Monday", "MS": "freq='MS'\nmonth start", "ME": "freq='ME'\nmonth end"}
    for month in pd.date_range("2024-01-01", "2024-04-01", freq="MS"):
        ax.axvline(mdates.date2num(month), color=GREY, lw=0.6, alpha=0.5)
    for y, (alias, dates) in zip([2, 1, 0], rows.items()):
        xd = mdates.date2num(dates)
        ax.plot(xd, np.full(len(xd), y), "o", color=colors[alias], ms=7)
        labelled = dates[:3] if alias == "W-MON" else dates
        for t in labelled:
            leap = alias == "ME" and t.month == 2
            ax.text(mdates.date2num(t), y + 0.2, t.strftime("%b %d").replace(" 0", " "), ha="center",
                    va="bottom", fontsize=9, color=VERMILLION if leap else INK,
                    fontweight="bold" if leap else "normal")
    ax.text(mdates.date2num(pd.Timestamp("2024-02-29")), -0.28, "leap day", ha="center", va="top",
            fontsize=9, color=VERMILLION)
    ax.set_yticks([2, 1, 0], [names[a] for a in rows])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_ylim(-0.6, 2.6)
    ax.set_xlim(mdates.date2num(pd.Timestamp("2023-12-29")), mdates.date2num(pd.Timestamp("2024-04-03")))
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b 1"))
    ax.xaxis.set_minor_locator(mdates.DayLocator())
    ax.tick_params(axis="x", which="minor", length=2)
    ax.set_xlabel("pd.date_range('2024-01-01', '2024-03-31', freq=...), one tick per day")
    save(fig, "date_range_aliases.png")


def two_clinics_utc():
    """Basic Time Zone Operations: Demo 1's two clinics, ordered by local clock and by UTC."""
    new_york = pd.DataFrame({"site": "New York", "patient_id": ["NY-01", "NY-02"],
                             "visit_local": ["2024-03-15 09:00", "2024-03-15 10:30"]})
    chicago = pd.DataFrame({"site": "Chicago", "patient_id": ["CH-01", "CH-02"],
                            "visit_local": ["2024-03-15 08:30", "2024-03-15 09:00"]})
    for df, zone in [(new_york, "America/New_York"), (chicago, "America/Chicago")]:
        df["visit_utc"] = (pd.to_datetime(df["visit_local"], format="%Y-%m-%d %H:%M")
                           .dt.tz_localize(zone).dt.tz_convert("UTC"))
    visits = pd.concat([new_york, chicago], ignore_index=True)
    by_utc = visits.sort_values("visit_utc")
    assert by_utc["patient_id"].tolist() == ["NY-01", "CH-01", "CH-02", "NY-02"]
    assert by_utc["visit_utc"].dt.strftime("%H:%M").tolist() == ["13:00", "13:30", "14:00", "14:30"]

    local_order = ["CH-01", "NY-01", "CH-02", "NY-02"]  # lecture's local order; NY-01 and CH-02 tie at 09:00
    offsets = {"CH-01": 0.0, "NY-01": 0.35, "CH-02": -0.35, "NY-02": 0.0}
    color = {"New York": BLUE, "Chicago": ORANGE}
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(9, 4.4), gridspec_kw={"hspace": 1.3})
    for ax, (lo, hi), title, xlabel in [
        (top, (8, 11), "Sorted by local clock reading: CH-01 first",
         "Local clock time, March 15, 2024 (each clinic's own zone)"),
        (bottom, (12.5, 15.5), "Sorted in UTC: NY-01 first", "UTC, March 15, 2024"),
    ]:
        ax.set(xlim=(lo - 0.15, hi + 0.15), ylim=(-1, 1), title=title, xlabel=xlabel)
        hide_y(ax)
        clock_axis(ax, major=0.5)
    for rank, pid in enumerate(local_order, start=1):
        row = visits.loc[visits["patient_id"] == pid].iloc[0]
        c = color[row["site"]]
        x_local = hours_of(pd.Timestamp(row["visit_local"]))
        x_utc = hours_of(row["visit_utc"])
        y_local = offsets[pid]
        top.plot(x_local, y_local, "o", color=c, ms=9)
        top.text(x_local + 0.06, y_local, f"{rank}. {pid}", va="center", fontsize=9, color=c,
                 fontweight="bold")
        utc_rank = by_utc["patient_id"].tolist().index(pid) + 1
        bottom.plot(x_utc, 0, "o", color=c, ms=9)
        bottom.text(x_utc, -0.45, f"{utc_rank}. {pid}", ha="center", va="top", fontsize=9, color=c,
                    fontweight="bold")
        fig.add_artist(ConnectionPatch(xyA=(x_local, y_local), coordsA=top.transData, xyB=(x_utc, 0),
                                       coordsB=bottom.transData, color=c, lw=1.2, alpha=0.7))
    top.plot([], [], "o", color=BLUE, label="New York")
    top.plot([], [], "o", color=ORANGE, label="Chicago")
    top.legend(loc="upper right", frameon=False, ncol=2, bbox_to_anchor=(1, 1.35))
    save(fig, "two_clinics_utc.png")


def clock_changes():
    """Clock Changes: New York wall clock against UTC on both 2024 clock-change nights."""
    local = pd.Series(pd.to_datetime(["2024-03-10 01:00", "2024-03-10 02:30",
                                      "2024-11-03 01:30", "2024-11-03 03:00"]))
    utc = local.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    assert utc.isna().tolist() == [False, True, True, False]
    assert utc[0] == pd.Timestamp("2024-03-10 06:00", tz="UTC")
    assert utc[3] == pd.Timestamp("2024-11-03 08:00", tz="UTC")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, gridspec_kw={"wspace": 0.08})
    panels = [
        (axes[0], "Spring forward: March 10, 2024", "2024-03-10 05:00", (2, 3), "02:00 to 03:00 never happens", 7),
        (axes[1], "Fall back: November 3, 2024", "2024-11-03 04:00", (1, 2), "01:00 to 02:00 (shaded)\nhappens twice", 6),
    ]
    for ax, title, start, band, band_label, jump_at in panels:
        instants = pd.date_range(start, periods=4 * 60 + 1, freq="min", tz="UTC")
        x = np.asarray(hours_of(instants), dtype=float)
        y = np.asarray(hours_of(instants.tz_convert("America/New_York")), dtype=float)
        breaks = np.where(np.abs(np.diff(y)) > 0.5)[0] + 1  # the clock jump, so the line does not join it
        ax.plot(np.insert(x, breaks, np.nan), np.insert(y, breaks, np.nan), color=INK, lw=2)
        ax.axhspan(*band, color=VERMILLION, alpha=0.13, lw=0)
        # spring: inside the skipped band; fall: above it, clear of both line segments
        label_y = band[1] - 0.04 if jump_at == 7 else band[1] + 0.75
        ax.text(x[0] + 0.05, label_y, band_label, va="top", color=VERMILLION, fontweight="bold")
        ax.plot([jump_at, jump_at], [y[breaks[0] - 1], y[breaks[0]]], ":", color=GREY, lw=1.5)
        ax.set(title=title, xlim=(x[0] - 0.15, x[-1] + 0.15), xlabel="Time (UTC)")
        clock_axis(ax, major=1, minor=0.5)
    axes[0].yaxis.set_major_locator(MultipleLocator(1))
    axes[0].yaxis.set_major_formatter(FuncFormatter(clock))
    axes[0].set(ylabel="New York clock", ylim=(-0.3, 5.3))

    ax = axes[0]
    ax.plot(6, 1, "o", color=BLUE, ms=9, zorder=3)
    ax.text(6.12, 0.85, "01:00 → 06:00 UTC", color=BLUE, va="top", fontweight="bold")
    ax.axhline(2.5, color=VERMILLION, ls="--", lw=1.2)
    ax.text(9.1, 2.42, "02:30 meets no\ninstant → NaT", color=VERMILLION, ha="right", va="top",
            fontweight="bold")
    ax = axes[1]
    ax.axhline(1.5, color=VERMILLION, ls="--", lw=1.2)
    ax.plot([5.5, 6.5], [1.5, 1.5], "o", mfc="white", mec=VERMILLION, mew=2, ms=9, zorder=3)
    ax.text(8.1, 0.75, "01:30 meets two instants\n(05:30 and 06:30 UTC) → NaT", color=VERMILLION,
            ha="right", va="top", fontweight="bold")
    ax.plot(8, 3, "o", color=BLUE, ms=9, zorder=3, clip_on=False)
    ax.text(7.9, 3.2, "03:00 → 08:00 UTC", color=BLUE, ha="right", va="bottom", fontweight="bold")
    save(fig, "ny_clock_changes_2024.png")


# ---------------------------------------------------------------- block 2


def time_series_indexing():
    """Basic Time Series Selection: two .loc selections on Demo 2's hourly heart rate."""
    hr = demo2_monitor()["heart_rate"]
    fever = hr.loc["2024-01-15":"2024-01-17"]
    jan5 = hr.loc["2024-01-05"]
    assert len(fever) == 72 and len(jan5) == 14 and len(hr) == 662
    grid = hr.asfreq("h")  # NaN for the unplugged hours, so the line breaks at the gap

    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), gridspec_kw={"hspace": 0.6})
    ax = axes[0]
    ax.plot(mdates.date2num(grid.index), grid.values, color=GREY, lw=0.7)
    ax.plot(mdates.date2num(fever.index), fever.values, color=VERMILLION, lw=0.9)
    ax.axvspan(mdates.date2num(pd.Timestamp("2024-01-15")), mdates.date2num(pd.Timestamp("2024-01-18")),
               color=VERMILLION, alpha=0.12, lw=0)
    ax.text(mdates.date2num(pd.Timestamp("2024-01-18 06:00")), 109,
            "hr.loc['2024-01-15':'2024-01-17']\n72 rows: all of Jan 15, 16, and 17", va="top",
            color=VERMILLION, fontfamily="monospace", fontsize=9)
    ax.axvspan(mdates.date2num(pd.Timestamp("2024-01-05")), mdates.date2num(pd.Timestamp("2024-01-06")),
               color=BLUE, alpha=0.12, lw=0)
    ax.text(mdates.date2num(pd.Timestamp("2024-01-06 06:00")), 109, "hr.loc['2024-01-05']", va="top",
            color=BLUE, fontfamily="monospace", fontsize=9)
    ax.set(title="Hourly heart rate, January 1 to 28, 2024 (662 rows)", ylabel="Heart rate (bpm)",
           ylim=(62, 110), xlim=(mdates.date2num(pd.Timestamp("2024-01-01")),
                                 mdates.date2num(pd.Timestamp("2024-01-29"))))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 29]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))

    ax = axes[1]
    day = jan5.asfreq("h")
    full_day = day.reindex(pd.date_range("2024-01-05", periods=24, freq="h"))
    xs = hours_of(full_day.index)
    ax.plot(xs, full_day.values, "-o", color=BLUE, ms=4, lw=1)
    ax.axvspan(7.5, 17.5, color=GREY, alpha=0.15, lw=0)
    ax.text(12.5, 88, "No rows 08:00 to 17:00", ha="center", va="top")
    ax.set(title="hr.loc['2024-01-05']: one day, 14 rows", xlim=(-0.5, 23.5), ylim=(65, 90),
           xlabel="Time, January 5, 2024", ylabel="Heart rate (bpm)")
    clock_axis(ax, major=2, minor=1)
    save(fig, "time_series_indexing.png")


def resampling_example():
    """Resampling intro: 91 daily weights and their 3 month-end means on one axis."""
    w = home_weights()
    monthly = w.resample("ME").mean()
    assert monthly.index.day.tolist() == [31, 29, 31]
    print("monthly means:", monthly.round(2).tolist(), "max:", w.max())

    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(mdates.date2num(w.index), w.values, "-o", color=GREY, lw=0.8, ms=2.5, label="Daily weight (91 rows)")
    for i, (month_end, mean) in enumerate(monthly.items()):
        start = mdates.date2num(month_end.replace(day=1))
        end = mdates.date2num(month_end)
        ax.hlines(mean, start, end, color=BLUE, lw=3, label="resample('ME').mean() (3 rows)" if i == 0 else None)
        ax.plot(end, mean, "o", color=BLUE, ms=8, zorder=3)
        ax.text(end, 81.6, f"{month_end:%b} {month_end.day}: {mean:.2f} kg", ha="right", va="top",
                color=BLUE, fontweight="bold")
    assert w.min() > 81.6  # month labels sit below every daily point
    peak = w.idxmax()
    ax.text(mdates.date2num(peak) - 1.5, w.max(), f"Daily peak {w.max():.1f} kg", ha="right", va="center")
    ax.set(ylabel="Weight (kg)", xlabel="Date, 2024", ylim=(81.2, 84.8))
    ax.xaxis.set_major_locator(mdates.MonthLocator(bymonthday=[1, 15]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.legend(loc="upper left", frameon=False)
    save(fig, "resampling_example.png")


def resample_bins_2h():
    """How Resampling Draws Bins: five readings in two left-closed, left-labeled 2-hour bins."""
    readings = pd.Series([70, 72, 75, 80, 78], index=pd.to_datetime([
        "2024-03-01 08:00", "2024-03-01 08:30", "2024-03-01 09:45", "2024-03-01 10:00", "2024-03-01 11:15"]))
    bins = readings.resample("2h").agg(["mean", "count"])
    assert bins["count"].tolist() == [3, 2] and bins["mean"].round(1).tolist() == [72.3, 79.0]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    xs = hours_of(readings.index)
    for (lo, color), (label, row) in zip([(8, BLUE), (10, ORANGE)], bins.iterrows()):
        ax.axvspan(lo, lo + 2, color=color, alpha=0.08, lw=0)
        inside = (xs >= lo) & (xs < lo + 2)
        ax.plot(xs[inside], readings.values[inside], "o", color=color, ms=8)
        ax.hlines(row["mean"], lo, lo + 2, color=color, lw=2, ls="--")
        ax.text(lo + 0.25, row["mean"] - (0.9 if lo == 8 else 2.2), f"mean {row['mean']:.1f} from "
                f"{int(row['count'])} readings,\nlabeled {label:%H:%M}", color=color, va="top", fontweight="bold")
        # interval marks: filled circle = edge included, open circle = edge excluded
        y_edge = 64.5 if lo == 8 else 63.5
        ax.plot([lo, lo + 2], [y_edge, y_edge], color=color, lw=2)
        ax.plot(lo, y_edge, "o", color=color, ms=8)
        ax.plot(lo + 2, y_edge, "o", mfc="white", mec=color, mew=2, ms=8)
        ax.text(lo + 1, y_edge + 0.25, f"[{clock(lo)}, {clock(lo + 2)})", ha="center", va="bottom",
                color=color, fontsize=9)
    for x, y in zip(xs, readings.values):
        dy = {10: 0.9, 8.5: 1.1}.get(x, 0)  # lift 80 and 72 above their bin's dashed mean line
        ax.text(x + 0.06, y + dy, str(y), ha="left", va="center", fontsize=9)
    ax.annotate("10:00 opens the second bin\n(closed='left')", (10, 80), (10.6, 83.3), color=INK,
                arrowprops={"arrowstyle": "-|>", "color": INK}, va="center")
    ax.set(xlim=(7.8, 12.2), ylim=(62.8, 85), xlabel="Time, March 1, 2024", ylabel="Heart rate (bpm)")
    clock_axis(ax, major=0.5)
    save(fig, "resample_bins_2h.png")


def daily_reading_counts():
    """Resampling with Different Aggregations: Demo 2's readings per day, short on January 5."""
    counts = demo2_monitor()["heart_rate"].resample("D").count()
    assert len(counts) == 28 and (counts == 24).sum() == 27 and counts["2024-01-05"] == 14

    xd = mdates.date2num(counts.index)
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.bar(xd, counts, width=0.75, color=[VERMILLION if n < 24 else BLUE for n in counts])
    jan5 = mdates.date2num(pd.Timestamp("2024-01-05"))
    ax.text(jan5 - 0.4, 28, "January 5: 14 readings, unplugged 08:00 to 17:00", va="bottom", color=VERMILLION,
            fontweight="bold")
    arrow(ax, (jan5, 27.8), (jan5, 14.4), color=VERMILLION)
    ax.set(ylabel="Readings per day", xlabel="Date, January 2024", ylim=(0, 30),
           xlim=(xd[0] - 0.7, xd[-1] + 0.7), title="resample('D').count(): 24 readings on 27 of 28 days")
    ax.set_yticks([0, 6, 12, 18, 24])
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 5, 8, 15, 22, 28]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    save(fig, "daily_reading_counts.png")


def charted_temp_upsampling():
    """Upsampling: Demo 2's six charted temperatures on an hourly grid, three ways."""
    charted_temp = pd.Series([101.0, 100.8, 100.9, 101.2, 100.6, 100.9],
                             index=pd.date_range("2024-01-15 00:00", periods=6, freq="4h"))
    up = pd.DataFrame({
        "asfreq": charted_temp.resample("h").asfreq(),
        "ffill_2": charted_temp.resample("h").ffill(limit=2),
        "interpolate": charted_temp.resample("h").interpolate(),
    })
    assert len(up) == 21 and up.isna().sum().tolist() == [15, 5, 0]
    assert round(up.loc["2024-01-15 02:00", "interpolate"], 3) == 100.9

    xs = hours_of(up.index)
    xr = hours_of(charted_temp.index)
    fig, axes = plt.subplots(3, 1, figsize=(9, 6.2), sharex=True, sharey=True, gridspec_kw={"hspace": 0.55})
    titles = {
        "asfreq": "resample('h').asfreq(): 15 of 21 hourly slots are NaN",
        "ffill_2": "resample('h').ffill(limit=2): each reading carried 2 hours; 5 slots stay NaN",
        "interpolate": "resample('h').interpolate(): straight lines fill all 21 slots",
    }
    for ax, column in zip(axes, up.columns):
        values = up[column]
        empty = xs[values.isna().to_numpy()]
        if empty.size:  # no legend entry on the interpolate() panel, which has no NaN slots
            ax.plot(empty, np.full(len(empty), 0.08), "x", color=GREY, ms=7, mew=1.5,
                    transform=ax.get_xaxis_transform(), label="NaN slot")
        if column == "ffill_2":
            ax.plot(xs, values, "-", color=ORANGE, lw=1.5)
            ax.plot(xs, values, "s", color=ORANGE, ms=5, label="Filled estimate")
        if column == "interpolate":
            ax.plot(xs, values, "-", color=BLUE, lw=1.5)
            ax.plot(xs, values, "s", color=BLUE, ms=4, label="Filled estimate")
        ax.plot(xr, charted_temp.values, "o", color=INK, ms=7, label="Charted reading", zorder=3)
        ax.set(title=titles[column], ylabel="Temp (°F)", ylim=(100.3, 101.4))
        ax.legend(loc="upper right", frameon=False, ncol=3, fontsize=8, bbox_to_anchor=(1, 1.08))
    axes[-1].set(xlabel="Time, January 15, 2024", xlim=(-0.5, 20.5))
    clock_axis(axes[-1], major=2, minor=1)
    save(fig, "charted_temp_upsampling.png")


def per_patient_bins():
    """Resampling Each Patient Separately: per-patient bin means against whole-table means."""
    vitals = pd.DataFrame({
        "patient_id": ["P1", "P1", "P1", "P2", "P2"],
        "recorded_at": pd.to_datetime(["2024-03-01 08:00", "2024-03-01 09:00", "2024-03-01 10:00",
                                       "2024-03-01 09:00", "2024-03-01 10:00"]),
        "heart_rate": [72, 75, 80, 90, 94],
    })
    indexed = vitals.set_index("recorded_at")
    per_patient = indexed.groupby("patient_id")["heart_rate"].resample("2h").mean()
    whole = indexed["heart_rate"].resample("2h").mean()
    assert per_patient.tolist() == [73.5, 80.0, 90.0, 94.0] and whole.tolist() == [79.0, 87.0]

    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.axvspan(10, 12, color=GREY, alpha=0.08, lw=0)
    color = {"P1": BLUE, "P2": ORANGE}
    for pid, group in vitals.groupby("patient_id"):
        ax.plot(hours_of(group["recorded_at"].dt), group["heart_rate"], "o", color=color[pid], ms=8,
                label=f"{pid} readings")
        for (_, start), mean in per_patient.loc[[pid]].items():
            lo = hours_of(start)
            ax.hlines(mean, lo, lo + 2, color=color[pid], lw=2.5)
            ax.text(lo + 1.97, mean + 0.5, f"{pid} mean {mean:.1f}", ha="right", va="bottom",
                    color=color[pid], fontweight="bold")
    for start, mean in whole.items():
        lo = hours_of(start)
        ax.hlines(mean, lo, lo + 2, color=GREY, lw=2.5, ls="--")
        ax.text(lo + 1.88, mean + 0.5, f"whole table {mean:.1f}", ha="right", va="bottom", color=INK)
    ax.set(xlim=(7.8, 12.2), ylim=(68, 98), xlabel="Time, March 1, 2024 (two-hour bins)",
           ylabel="Heart rate (bpm)")
    clock_axis(ax, major=1)
    ax.legend(loc="upper left", frameon=False)
    save(fig, "per_patient_bins.png")


def gap_run_tiles():
    """Gap Runs: each patient's observed hours, the running count, and the gap runs it numbers."""
    hours = pd.DataFrame({
        "patient_id": ["P1"] * 7 + ["P2"] * 4,
        "source_observed": [True, False, False, True, False, False, False, True, True, False, True],
    })
    hours["run_id"] = hours.groupby("patient_id")["source_observed"].cumsum()
    runs = hours[~hours["source_observed"]].groupby(["patient_id", "run_id"]).size()
    assert runs.to_dict() == {("P1", 1): 2, ("P1", 2): 3, ("P2", 2): 1}

    fig, ax = plt.subplots(figsize=(9, 4.3))
    ax.axis("off")
    x0, w, h = 3.2, 1.0, 0.7
    for top, (pid, group) in zip([4.6, 1.5], hours.groupby("patient_id")):
        ax.text(0, top + h / 2, pid, fontsize=13, fontweight="bold", va="center")
        ax.text(0.6, top + h / 2, "source_observed", va="center", fontfamily="monospace", fontsize=9)
        ax.text(0.6, top - 0.45, "cumsum() = run_id", va="center", fontfamily="monospace", fontsize=9)
        for i, (observed, run_id) in enumerate(zip(group["source_observed"], group["run_id"])):
            x = x0 + i * w
            ax.add_patch(Rectangle((x + 0.04, top), w - 0.08, h, facecolor=BLUE if observed else "white",
                                   edgecolor=BLUE if observed else ORANGE, linewidth=1.5,
                                   hatch=None if observed else "///"))
            ax.text(x + w / 2, top + h / 2, str(observed), ha="center", va="center", fontsize=9,
                    color="white" if observed else INK, fontweight="bold")
            ax.text(x + w / 2, top - 0.45, str(run_id), ha="center", va="center", fontsize=11,
                    color=INK if observed else ORANGE, fontweight="bold")
        positions = np.arange(len(group))
        gap = ~group["source_observed"].to_numpy()
        for run_id, size in runs.loc[pid].items():
            members = positions[gap & (group["run_id"].to_numpy() == run_id)]
            left, right = x0 + members[0] * w + 0.1, x0 + (members[-1] + 1) * w - 0.1
            ax.plot([left, left, right, right], [top - 0.75, top - 0.85, top - 0.85, top - 0.75],
                    color=ORANGE, lw=1.5)
            ax.text((left + right) / 2, top - 0.95, f"gap run {run_id}: {size} h", ha="center", va="top",
                    color=ORANGE, fontweight="bold")
    ax.set_xlim(-0.1, x0 + 7 * w + 0.1)
    ax.set_ylim(-0.2, 5.5)
    save(fig, "gap_run_tiles.png")


# ---------------------------------------------------------------- block 3


def daily_hr_rolling_ewm():
    """Basic Rolling Operations: Demo 2's daily mean heart rate, 7-day rolling mean with ±1 SD, and EWM."""
    daily_hr = demo2_monitor()["heart_rate"].resample("D").mean()
    rolling = daily_hr.rolling(7).mean()
    sd = daily_hr.rolling(7).std()
    ewm = daily_hr.ewm(span=7).mean()
    peak = pd.Timestamp("2024-01-17")
    assert round(ewm[peak], 1) == 86.9 and round(rolling[peak], 1) == 84.7
    assert ewm.idxmax() == peak and rolling.idxmax() == peak
    assert round(ewm["2024-01-21"], 1) == 80.7 and round(rolling["2024-01-21"], 1) == 84.4

    xd = mdates.date2num(daily_hr.index)
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(xd, rolling - sd, rolling + sd, color=BLUE, alpha=0.15, lw=0, label="Rolling mean ± 1 SD")
    ax.plot(xd, daily_hr, "o", color=GREY, ms=5, label="Daily mean")
    ax.plot(xd, rolling, color=BLUE, lw=2, label="7-day rolling mean")
    ax.plot(xd, ewm, "--", color=ORANGE, lw=2.2, label="EWM, span 7")
    xp = mdates.date2num(peak)
    ax.plot([xp, xp], [ewm[peak], rolling[peak]], "o", color=INK, ms=4, zorder=4)
    ax.text(xp + 0.4, ewm[peak] + 0.4, f"EWM {ewm[peak]:.1f} on Jan 17", color=ORANGE, va="bottom",
            fontweight="bold")
    # above the rolling plateau, clear of the falling EWM line
    ax.text(xp + 2.2, rolling[peak] + 0.4, f"rolling {rolling[peak]:.1f} on Jan 17", color=BLUE, va="bottom",
            fontweight="bold")
    ax.set(ylabel="Heart rate (bpm)", xlabel="Date, January 2024", xlim=(xd[0] - 0.7, xd[-1] + 0.7))
    ax.xaxis.set_major_locator(mdates.DayLocator(bymonthday=[1, 8, 15, 22, 28]))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %-d"))
    ax.legend(loc="upper left", frameon=False)
    save(fig, "daily_hr_rolling_ewm.png")


def count_vs_time_window():
    """Count Window vs Time Window: which glucose readings rolling(2) and rolling('2h') average at 11:00."""
    glucose = pd.Series([110, 145, 130, 180], index=pd.to_datetime([
        "2024-03-01 07:00", "2024-03-01 08:00", "2024-03-01 11:00", "2024-03-01 11:30"]))
    by_count, by_time = glucose.rolling(2).mean(), glucose.rolling("2h").mean()
    at = pd.Timestamp("2024-03-01 11:00")
    assert by_count[at] == 137.5 and by_time[at] == 130.0

    xs = hours_of(glucose.index)
    fig, axes = plt.subplots(2, 1, figsize=(9, 3.9), sharex=True, gridspec_kw={"hspace": 0.55})
    rows = [
        (axes[0], "rolling(2): the current reading and the 1 before it, however far back", BLUE,
         (8, 11), "08:00 and 11:00, 3 hours apart:\nmean 137.5", [1, 2]),
        (axes[1], "rolling('2h'): the readings after 09:00, up to and including 11:00", ORANGE,
         (9, 11), "only 11:00:\nmean 130.0", [2]),
    ]
    for ax, title, color, (lo, hi), note, used in rows:
        ax.axvspan(lo, hi, color=color, alpha=0.12, lw=0)
        ax.plot([lo, hi], [0, 0], color=color, lw=3)
        ax.plot(hi, 0, "o", color=color, ms=6)  # 11:00 is inside the window
        if color == ORANGE:
            ax.plot(lo, 0, "o", mfc="white", mec=color, mew=2, ms=6)  # 09:00 itself is outside: (09:00, 11:00]
        for i, (x, y) in enumerate(zip(xs, glucose)):
            inside = i in used
            ax.plot(x, 0.55, "o", ms=11, color=color if inside else "white", mec=color if inside else GREY, mew=2)
            ax.text(x, 1.0, str(y), ha="center", va="bottom", fontsize=9, color=INK if inside else GREY)
        ax.text(11.85, 0.3, note, color=color, fontweight="bold", va="center", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.set(ylim=(-0.4, 1.6))
        hide_y(ax)
    axes[1].set(xlim=(6.7, 13.4), xlabel="Time, March 1, 2024 (glucose in mg/dL; filled = averaged at 11:00)")
    clock_axis(axes[1], major=1)
    save(fig, "count_vs_time_window.png")


def window_weights():
    """Another Smoother: the weight each earlier day gets in rolling(7).mean() and ewm(span=7).mean()."""
    days_back = np.arange(14)
    alpha = 2 / (7 + 1)
    ewm_w = alpha * (1 - alpha) ** days_back
    roll_w = np.where(days_back < 7, 1 / 7, 0.0)
    # check against pandas: weights are what each past reading contributes to today's value
    probe = pd.Series(np.zeros(200))
    probe.iloc[-1 - 3] = 1.0  # a single 1 three days back
    assert abs(probe.ewm(span=7).mean().iloc[-1] - ewm_w[3]) < 1e-9
    assert abs(probe.rolling(7).mean().iloc[-1] - roll_w[3]) < 1e-12

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.bar(days_back - 0.2, roll_w, width=0.4, color=BLUE, label="rolling(7).mean(): 1/7 each for 7 days, then 0")
    ax.bar(days_back + 0.2, ewm_w, width=0.4, color=ORANGE, label="ewm(span=7).mean(): 25% today, each day 3/4 of the next")
    ax.text(0.2, ewm_w[0] + 0.006, "0.25", ha="center", va="bottom", fontsize=9, color=ORANGE)
    ax.text(-0.2, roll_w[0] + 0.006, "0.14", ha="center", va="bottom", fontsize=9, color=BLUE)
    ax.set_xticks(days_back, ["today"] + [str(d) for d in days_back[1:]])
    ax.set(xlabel="Days before the current reading", ylabel="Weight in today's value", ylim=(0, 0.3),
           xlim=(-0.7, 13.7))
    ax.legend(frameon=False, loc="upper right")
    save(fig, "window_weights.png")


def shifting_lagging():
    """Shifting and Lagging: Demo 3's home weights, March 6 to 14, with lag, lead, and diff()."""
    daily = pd.DataFrame({"weight_kg": home_weights()})
    daily["lag_1"] = daily["weight_kg"].shift(1)
    daily["lead_1"] = daily["weight_kg"].shift(-1)
    daily["change_1d"] = daily["weight_kg"].diff().round(1)
    window = daily.loc["2024-03-06":"2024-03-14"]
    assert daily.loc["2024-03-10", "change_1d"] == 1.3 and daily.loc["2024-03-10", "weight_kg"] == 83.9
    assert (window["change_1d"] > 1.0).sum() == 1

    xd = mdates.date2num(window.index)
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True, gridspec_kw={"height_ratios": [3, 2], "hspace": 0.3})
    ax = axes[0]
    ax.plot(xd, window["weight_kg"], "-o", color=INK, ms=7, lw=1.8, label="weight_kg", zorder=3)
    ax.plot(xd, window["lag_1"], "--o", color=BLUE, mfc="white", ms=7, lw=1.2, label="shift(1): lag")
    ax.plot(xd, window["lead_1"], ":s", color=ORANGE, mfc="white", ms=7, lw=1.5, label="shift(-1): lead")
    mar = {d: mdates.date2num(pd.Timestamp(f"2024-03-{d:02d}")) for d in (9, 10, 11)}
    w9, w11 = daily.loc["2024-03-09", "weight_kg"], daily.loc["2024-03-11", "weight_kg"]
    arrow(ax, (mar[9] + 0.08, w9), (mar[10] - 0.12, w9), color=BLUE, lw=1.8)
    ax.text(mar[10] + 0.15, w9 - 0.05, f"Mar 9's {w9} moves\none row later", color=BLUE, va="top", fontsize=9)
    arrow(ax, (mar[11] - 0.08, w11), (mar[10] + 0.12, w11), color=ORANGE, lw=1.8)
    ax.text((mar[10] + mar[11]) / 2, w11 + 0.13, f"Mar 11's {w11} moves one row earlier", color=ORANGE,
            ha="center", va="bottom", fontsize=9)
    ax.set(ylabel="Weight (kg)", ylim=(81.5, 85.0))
    ax.legend(loc="upper left", frameon=False)

    ax = axes[1]
    changes = window["change_1d"]
    ax.bar(xd, changes, width=0.6, color=[VERMILLION if v > 1.0 else GREY for v in changes])
    ax.axhline(1.0, color=VERMILLION, ls="--", lw=1.2)
    ax.text(xd[0] - 0.3, 1.05, "Alert: more than 1.0 kg in a day", color=VERMILLION, va="bottom", fontsize=9)
    ax.axhline(0, color=INK, lw=0.8)
    for x, v in zip(xd, changes):
        ax.text(x, v + (0.05 if v >= 0 else -0.05), "0.0" if v == 0 else f"{v:+.1f}", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=9)
    ax.set(ylabel="diff(): today minus\nprevious day (kg)", ylim=(-1.2, 1.8), xlabel="Date, March 2024")
    ax.set_xticks(xd, window.index.strftime("Mar %d").str.replace(" 0", " "))
    save(fig, "shifting_lagging.png")


def grouped_shift_leak():
    """Grouped Lags and Past-Only Windows: grouped shift, the ungrouped leak, and an empty 2-hour window."""
    vitals = pd.DataFrame({
        "patient_id": ["P1"] * 3 + ["P2"] * 3,
        "recorded_at": pd.to_datetime(["2024-03-01 08:00", "2024-03-01 09:00", "2024-03-01 11:30",
                                       "2024-03-01 09:00", "2024-03-01 10:00", "2024-03-01 11:00"]),
        "heart_rate": [72, 80, 95, 88, 86, 84],
    })
    vitals["wrong_previous"] = vitals["heart_rate"].shift(1)
    vitals["previous_hr"] = vitals.groupby("patient_id")["heart_rate"].shift(1)
    prev_2h = (vitals.set_index("recorded_at").groupby("patient_id")["heart_rate"]
               .rolling("2h", closed="left").mean())
    assert vitals.loc[3, "wrong_previous"] == 95 and np.isnan(vitals.loc[3, "previous_hr"])
    assert np.isnan(prev_2h.loc[("P1", pd.Timestamp("2024-03-01 11:30"))])

    lane = {"P1": 2.0, "P2": 0.0}
    fig, ax = plt.subplots(figsize=(9, 5))
    # P1's past-only 2-hour window at 11:30: [09:30, 11:30), shaded over its lane, bracket below it
    ax.add_patch(Rectangle((9.5, 1.62), 2.0, 0.76, facecolor=PURPLE, alpha=0.12, edgecolor="none"))
    ax.plot([9.5, 11.5], [1.45, 1.45], color=PURPLE, lw=2)
    ax.plot(9.5, 1.45, "o", color=PURPLE, ms=8)
    ax.plot(11.5, 1.45, "o", mfc="white", mec=PURPLE, mew=2, ms=8, zorder=4)
    ax.text(9.42, 1.45, "rolling('2h', closed='left') at 11:30:\n09:30 up to 11:30 holds no P1 reading, so NaN",
            color=PURPLE, ha="right", va="center", fontsize=9, fontweight="bold")
    for pid, group in vitals.groupby("patient_id"):
        xs = hours_of(group["recorded_at"].dt).to_numpy()
        y = lane[pid]
        ax.axhline(y, color=GREY, lw=0.6, alpha=0.6, zorder=0)
        for x_prev, x_next in zip(xs[:-1], xs[1:]):
            # arc3's bulge scales with arrow length, so divide by the gap for one arc height
            arrow(ax, (x_prev + 0.07, y + 0.06), (x_next - 0.07, y + 0.06), color=GREEN, lw=1.5,
                  rad=-0.4 / (x_next - x_prev))
        for x, hr, prev, wrong in zip(xs, group["heart_rate"], group["previous_hr"], group["wrong_previous"]):
            ax.plot(x, y, "o", color=BLUE if pid == "P1" else ORANGE, ms=11, zorder=3)
            ax.text(x, y + 0.17, str(hr), ha="center", va="bottom", fontsize=10, fontweight="bold")
            ax.text(x, y - 0.17, "prev NaN" if np.isnan(prev) else f"prev {prev:g}", ha="center", va="top",
                    fontsize=8.5, color=GREEN)
            if wrong == 95 and pid == "P2":
                ax.text(x, y - 0.4, "plain shift: 95", ha="center", va="top", fontsize=8.5, color=VERMILLION,
                        fontweight="bold")
    arrow(ax, (11.4, 1.97), (9.07, 0.08), color=VERMILLION, lw=2.2)
    ax.text(10.45, 0.62, "plain shift(1): P2's first row\ngets P1's 95, recorded at 11:30", color=VERMILLION,
            fontsize=9, fontweight="bold", ha="left", va="center")
    ax.plot([], [], color=GREEN, lw=1.5, label="groupby('patient_id').shift(1): same patient's previous reading")
    ax.legend(loc="upper left", frameon=False, bbox_to_anchor=(0, 1.04))
    ax.set_yticks([2, 0], ["P1", "P2"])
    ax.tick_params(axis="y", length=0, labelsize=12)
    ax.spines["left"].set_visible(False)
    ax.set(xlim=(7.6, 12.1), ylim=(-0.75, 2.85), xlabel="Time, March 1, 2024")
    clock_axis(ax, major=0.5)
    save(fig, "grouped_shift_leak.png")


def lab_availability_timeline():
    """Availability at Prediction Time and Chronological Holdout: labs and vitals around 11:00."""
    labs = pd.DataFrame({
        "test": ["lactate", "creatinine", "troponin"],
        "collected_at": pd.to_datetime(["2024-03-01 08:00", "2024-03-01 09:30", "2024-03-01 10:30"]),
        "resulted_at": pd.to_datetime(["2024-03-01 08:45", "2024-03-01 11:15", "2024-03-01 10:50"]),
    })
    prediction_time = pd.Timestamp("2024-03-01 11:00")
    labs["available"] = labs["resulted_at"] <= prediction_time
    assert labs["available"].tolist() == [True, False, True]
    vitals = pd.DataFrame({
        "patient_id": ["P1"] * 3 + ["P2"] * 3,
        "recorded_at": pd.to_datetime(["2024-03-01 08:00", "2024-03-01 09:00", "2024-03-01 11:30",
                                       "2024-03-01 09:00", "2024-03-01 10:00", "2024-03-01 11:00"]),
    })
    vitals["block"] = np.where(vitals["recorded_at"] < prediction_time, "earlier", "later_holdout")
    assert vitals["block"].tolist() == ["earlier", "earlier", "later_holdout", "earlier", "earlier", "later_holdout"]

    cut = hours_of(prediction_time)
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, gridspec_kw={"hspace": 0.45})
    ax = axes[0]
    for y, row in zip([2, 1, 0], labs.itertuples()):
        lo, hi = hours_of(row.collected_at), hours_of(row.resulted_at)
        c = GREEN if row.available else VERMILLION
        ax.plot([lo, hi], [y, y], color=c, lw=7, solid_capstyle="butt")
        ax.plot(lo, y, "o", color=INK, ms=6)
        ax.plot(hi, y, "D", color=INK, ms=6)
        status = "available" if row.available else "not available"
        ax.text(hi + 0.08, y, f"resulted {row.resulted_at:%H:%M}: {status}", va="center", color=c,
                fontweight="bold", fontsize=9)
    ax.set_yticks([2, 1, 0], labs["test"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.plot([], [], "o", color=INK, label="collected_at")
    ax.plot([], [], "D", color=INK, label="resulted_at")
    ax.legend(loc="upper left", frameon=False, ncol=2, bbox_to_anchor=(0, 1.02), fontsize=9)
    ax.set(title="Labs: usable only if resulted by prediction_time", ylim=(-0.6, 2.9))

    ax = axes[1]
    ax.axvspan(7.5, cut, color=BLUE, alpha=0.07, lw=0)
    ax.axvspan(cut, 12.25, color=ORANGE, alpha=0.1, lw=0)
    ax.text(7.6, 1.55, "earlier", color=BLUE, fontweight="bold", va="center")
    ax.text(12.15, 1.55, "later_holdout", color="#9A6A00", fontweight="bold", va="center", ha="right")
    for row in vitals.itertuples():
        y = 1 if row.patient_id == "P1" else 0
        early = row.block == "earlier"
        ax.plot(hours_of(row.recorded_at), y, "o", color=BLUE if early else ORANGE, ms=10, zorder=3)
    ax.annotate("P2 at 11:00 is held out:\n'<' keeps the cutoff out of earlier", (cut, 0), (11.3, -0.55),
                fontsize=9, va="center", arrowprops={"arrowstyle": "-|>", "color": INK})
    ax.set_yticks([1, 0], ["P1", "P2"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set(title="Vitals: chronological holdout at the same cutoff", ylim=(-0.9, 1.9),
           xlim=(7.5, 12.25), xlabel="Time, March 1, 2024")
    for a in axes:
        a.axvline(cut, color=INK, ls="--", lw=1.3)
    axes[0].text(cut + 0.04, 2.75, "prediction_time 11:00", fontsize=9, va="top")
    clock_axis(axes[1], major=0.5)
    save(fig, "lab_availability_timeline.png")


def ili_outage_gap():
    """Basic Time Series Plots: Demo 3's winter ILI visits as reported and after asfreq('D')."""
    winter = demo3_ili().loc["2022-12-01":"2023-03-31"]
    gridded = winter.asfreq("D")
    missing = gridded.index[gridded.isna()]
    assert len(winter) == 107 and len(gridded) == 121
    assert missing.equals(pd.date_range("2023-02-01", "2023-02-14", freq="D"))

    out_lo, out_hi = mdates.date2num(missing[0]), mdates.date2num(missing[-1])
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.4), sharex=True, sharey=True, gridspec_kw={"hspace": 0.35})
    for ax, series, title in [
        (axes[0], winter, "As reported: the line bridges the outage"),
        (axes[1], gridded, "After asfreq('D'): the 14 missing days are NaN and the line breaks"),
    ]:
        ax.axvspan(out_lo - 0.5, out_hi + 0.5, color=VERMILLION, alpha=0.1, lw=0)
        ax.plot(mdates.date2num(series.index), series.to_numpy(dtype=float), color=BLUE, lw=1.2)
        ax.set(title=title, ylabel="Visits per day")
    axes[0].text((out_lo + out_hi) / 2, 74, "Feed outage\nFeb 1 to 14", ha="center", va="top", color=VERMILLION,
                 fontweight="bold", fontsize=9)
    axes[1].set(xlabel="Date", ylim=(35, 75),
                xlim=(mdates.date2num(pd.Timestamp("2022-12-01")), mdates.date2num(pd.Timestamp("2023-04-01"))))
    axes[1].xaxis.set_major_locator(mdates.MonthLocator())
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter("%b 1, %Y"))
    save(fig, "ili_outage_gap.png")


if __name__ == "__main__":
    MEDIA_DIR.mkdir(exist_ok=True)
    for make in [forms_of_time, regular_irregular, icu_panel_hr_utc, text_vs_datetime_axis, date_range_aliases,
                 two_clinics_utc, clock_changes, time_series_indexing, resampling_example, resample_bins_2h,
                 daily_reading_counts, charted_temp_upsampling, per_patient_bins, gap_run_tiles,
                 daily_hr_rolling_ewm, count_vs_time_window, window_weights, shifting_lagging,
                 grouped_shift_leak, lab_availability_timeline, ili_outage_gap]:
        if len(sys.argv) == 1 or make.__name__ in sys.argv[1:]:  # optional: name figures to redraw only those
            make()
