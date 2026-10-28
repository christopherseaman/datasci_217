"""Render Lecture 10's generated figures into ../media/.

Every figure is computed from the lecture's snippet data (the 60-patient `clinic`
table) or from the demos' data and code, and the numbers its caption states are
asserted against the values the lecture and demos print.

Run from the repository root. TensorFlow is needed for the four Demo 3 figures
(xgboost_early_stopping, keras_loss_curves, keras_dropout_curves, model_comparison),
since demo3() reruns the demo in order:
    uv run --project 10/demo python 10/scripts/generate_lecture_figures.py
Pass figure function names to render only those, for example `intervals decision_tree`.
"""

import os
import sys
from functools import cache
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import statsmodels.formula.api as smf  # noqa: E402
from cycler import cycler  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402
from sklearn.datasets import load_breast_cancer, load_diabetes  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestClassifier  # noqa: E402
from sklearn.inspection import permutation_importance  # noqa: E402
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression, Ridge  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, mean_absolute_error  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.tree import DecisionTreeRegressor, plot_tree  # noqa: E402

MEDIA_DIR = Path(__file__).resolve().parents[1] / "media"

BLUE, ORANGE, GREEN, VERMILLION, PINK = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"
GREY = "#9A9A9A"
INK = "#222222"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.prop_cycle": cycler(color=[BLUE, ORANGE, GREEN, VERMILLION, PINK]),
    "font.size": 11,
})

DIABETES_NAMES = {"s1": "tc", "s2": "ldl", "s3": "hdl", "s4": "tch", "s5": "ltg", "s6": "glu",
                  "target": "progression"}
DIABETES_FEATURES = ["age", "sex", "bmi", "bp", "tc", "ldl", "hdl", "tch", "ltg", "glu"]


def save(fig, filename):
    fig.savefig(MEDIA_DIR / filename, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Generated:", filename)


# ---------------------------------------------------------------- shared data

@cache
def clinic_fit():
    """The lecture's 60 simulated patients and their `sbp ~ age + bmi` fit."""
    rng = np.random.default_rng(42)
    clinic = pd.DataFrame({"age": rng.integers(30, 70, size=60),
                           "bmi": rng.normal(27, 4, size=60).round(1)})
    clinic["sbp"] = (95 + 0.5 * clinic["age"] + 0.8 * clinic["bmi"]
                     + rng.normal(0, 8, size=60)).round(1)
    results = smf.ols("sbp ~ age + bmi", data=clinic).fit()
    assert results.params.round(2).tolist() == [75.21, 0.66, 1.19]
    assert round(results.rsquared, 3) == 0.536
    return clinic, results


@cache
def diabetes():
    frame = load_diabetes(scaled=False, as_frame=True).frame.rename(columns=DIABETES_NAMES)
    assert frame.shape == (442, 11)
    return frame


@cache
def demo1_readings():
    """Demo 1, Part 8: 210 weekly home readings from 30 patients."""
    rng = np.random.default_rng(217)
    mondays = pd.date_range("2026-01-05", periods=8, freq="W-MON")
    routine_hours = [7, 8, 9, 19, 21, 23, 0]
    records = []
    for patient_id in range(1, 31):
        age = int(rng.integers(35, 80))
        usual_hour = routine_hours[patient_id % len(routine_hours)]
        level = rng.normal(132 + 0.2 * age, 8)
        for monday in mondays:
            hour = (usual_hour + int(rng.integers(-1, 2))) % 24
            level = 0.7 * level + 0.3 * rng.normal(132 + 0.2 * age, 8)
            time_of_day = 5 * np.cos(2 * np.pi * (hour - 10) / 24)
            records.append({
                "patient_id": f"P{patient_id:03d}",
                "reading_time": monday + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60))),
                "age": age,
                "sbp_today": round(level + time_of_day + rng.normal(0, 3), 1),
                "repeat_sbp": round(level + rng.normal(0, 3), 1),
            })
    readings = pd.DataFrame(records).sort_values(["patient_id", "reading_time"])
    readings["sbp_next"] = readings.groupby("patient_id")["sbp_today"].shift(-1)
    readings["target_time"] = readings.groupby("patient_id")["reading_time"].shift(-1)
    readings = readings.dropna(subset=["sbp_next"]).reset_index(drop=True)
    assert len(readings) == 210
    return readings


@cache
def demo2():
    """Demo 2: the diabetes split, the four candidates, and the frozen Lasso's test score."""
    frame = diabetes()
    X, y = frame[DIABETES_FEATURES], frame["progression"]
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.25, random_state=42)
    models = {
        "mean baseline": DummyRegressor(strategy="mean"),
        "linear regression": Pipeline([("scale", StandardScaler()), ("model", LinearRegression())]),
        "ridge (alpha=10)": Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=10.0))]),
        "lasso (alpha=2)": Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=2.0))]),
    }
    valid_mae = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        valid_mae[name] = mean_absolute_error(y_valid, model.predict(X_valid))
    assert [round(v, 2) for v in valid_mae.values()] == [64.13, 42.94, 42.62, 42.45]
    final = Pipeline([("scale", StandardScaler()), ("model", Lasso(alpha=2.0))])
    final.fit(X_train_valid, y_train_valid)
    test_mae = mean_absolute_error(y_test, final.predict(X_test))
    assert round(test_mae, 2) == 42.84
    return dict(X_train=X_train, X_valid=X_valid, y_train=y_train, y_valid=y_valid,
                models=models, valid_mae=valid_mae, test_mae=test_mae)


@cache
def demo3():
    """Demo 3: every candidate in the demo's order, so TensorFlow's seeded run repeats."""
    cancer = load_breast_cancer(as_frame=True)
    X = cancer.frame[cancer.feature_names.tolist()].values
    y = (cancer.target == 0).astype(int).values
    X_train_valid, X_test, y_train_valid, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_train_valid, y_train_valid, test_size=0.25, random_state=42, stratify=y_train_valid)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_valid_scaled = scaler.transform(X_valid)
    accuracy = {}
    majority_class = int(pd.Series(y_train).mode()[0])
    accuracy["Majority-class baseline"] = accuracy_score(y_valid, np.full(len(y_valid), majority_class))
    lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train_scaled, y_train)
    accuracy["Logistic regression"] = accuracy_score(y_valid, lr.predict(X_valid_scaled))
    rf = RandomForestClassifier(n_estimators=100, max_depth=5, min_samples_split=5,
                                random_state=42, n_jobs=-1).fit(X_train, y_train)
    accuracy["Random forest"] = accuracy_score(y_valid, rf.predict(X_valid))

    import xgboost as xgb

    boosted = xgb.XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.1, subsample=0.8,
                                colsample_bytree=0.8, random_state=42, n_jobs=-1).fit(X_train, y_train)
    accuracy["XGBoost (100 rounds)"] = accuracy_score(y_valid, boosted.predict(X_valid))
    early = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.1, subsample=0.8,
                              colsample_bytree=0.8, early_stopping_rounds=10, random_state=42, n_jobs=-1)
    early.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
    accuracy["XGBoost (early stopped)"] = accuracy_score(y_valid, early.predict(X_valid))
    xgb_curve = early.evals_result()["validation_0"]["logloss"]
    assert early.best_iteration == 54 and round(early.best_score, 4) == 0.1168

    import tensorflow as tf
    from tensorflow import keras

    keras.utils.set_random_seed(42)
    tf.config.experimental.enable_op_determinism()
    n_features = X_train.shape[1]

    def fit(layers):
        model = keras.Sequential([keras.layers.Input(shape=(n_features,)), *layers])
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        history = model.fit(X_train_scaled, y_train, epochs=50, batch_size=32,
                            validation_data=(X_valid_scaled, y_valid), verbose=0)
        model.evaluate(X_valid_scaled, y_valid, verbose=0)
        predicted = (model.predict(X_valid_scaled, verbose=0) > 0.5).astype(int).flatten()
        return pd.DataFrame(history.history), accuracy_score(y_valid, predicted)

    dense, l2 = keras.layers.Dense, keras.regularizers.l2
    plain_history, accuracy["Neural net 64-32"] = fit([
        dense(64, activation="relu", name="hidden1"), dense(32, activation="relu", name="hidden2"),
        dense(1, activation="sigmoid", name="output")])
    _, accuracy["Neural net 128-64-32-16"] = fit([
        dense(128, activation="relu"), dense(64, activation="relu"), dense(32, activation="relu"),
        dense(16, activation="relu"), dense(1, activation="sigmoid")])
    reg_history, accuracy["Neural net 64-32 + dropout/L2"] = fit([
        dense(64, activation="relu", kernel_regularizer=l2(0.01)), keras.layers.Dropout(0.3),
        dense(32, activation="relu", kernel_regularizer=l2(0.01)), keras.layers.Dropout(0.3),
        dense(1, activation="sigmoid")])
    correct = [round(a * len(y_valid)) for a in accuracy.values()]
    assert correct == [71, 111, 111, 110, 108, 111, 110, 111], correct
    for history in (plain_history, reg_history):
        history.index = range(1, len(history) + 1)
    return dict(accuracy=accuracy, n_valid=len(y_valid), xgb_curve=xgb_curve,
                best_iteration=early.best_iteration, plain=plain_history, reg=reg_history)


# --------------------------------------------------------- What Is a Model / OLS

def residuals_vs_fitted():
    """Clinic residuals (no pattern) beside a straight line fitted to curved data (U-shape)."""
    clinic, results = clinic_fit()
    rng = np.random.default_rng(10)
    curved = pd.DataFrame({"age": rng.uniform(25, 75, size=60)})
    curved["sbp"] = 108 + 0.018 * (curved["age"] - 25) ** 2 + rng.normal(0, 3, size=60)
    curved_fit = smf.ols("sbp ~ age", data=curved).fit()
    ends = curved_fit.resid[(curved["age"] < 32) | (curved["age"] > 68)].mean()
    middle = curved_fit.resid[curved["age"].between(42, 58)].mean()
    assert ends > 3 and middle < -2, (ends, middle)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, fit, title in [(axes[0], results, "Clinic fit: no pattern"),
                           (axes[1], curved_fit, "Straight line on a curve: U-shape")]:
        ax.scatter(fit.fittedvalues, fit.resid, color=BLUE, s=22)
        ax.axhline(0, color=GREY, linestyle="--")
        ax.set(title=title, xlabel="Fitted SBP (mmHg)", ylabel="Residual, observed - fitted (mmHg)")
    fig.tight_layout()
    save(fig, "ols_residuals_vs_fitted.png")


def ols_summary():
    """The clinic fit's real `results.summary()`, with the columns the lecture reads boxed."""
    _, results = clinic_fit()
    lines = str(results.summary()).splitlines()
    header = next(i for i, line in enumerate(lines) if line.split()[:1] == ["coef"])
    last = max(i for i, line in enumerate(lines) if line.split()[:1] in (["age"], ["bmi"]))
    assert lines[header + 2].startswith("Intercept") and last == header + 4

    def spans(line):
        out, start = [], None
        for i, ch in enumerate(line + " "):
            if ch != " " and start is None:
                start = i
            elif ch == " " and start is not None:
                out.append((start, i))
                start = None
        return out

    def column(label_tokens, row_token):
        """Character span covering a header label and the values printed under it."""
        heads = spans(lines[header])
        words = [lines[header][s:e] for s, e in heads]
        first = words.index(label_tokens[0])
        start, end = heads[first][0], heads[first + len(label_tokens) - 1][1]
        for row in range(header + 2, last + 1):
            s, e = spans(lines[row])[row_token]
            start, end = min(start, s), max(end, e)
        return start, end

    boxes = [
        (column(["coef"], 1), BLUE),
        (column(["std", "err"], 2), ORANGE),
        (column(["P>|t|"], 4), GREEN),
        ((column(["[0.025"], 5)[0], column(["0.975]"], 6)[1]), VERMILLION),
    ]
    r2_line = next(i for i, line in enumerate(lines) if " R-squared:" in line and "Adj." not in line)
    r2_start = lines[r2_line].index("R-squared:")
    assert lines[r2_line].split()[-1] == "0.536"

    fontsize, margin = 10, 0.15
    line_h = 1.3 * fontsize / 72
    # Measure at the save dpi: glyph advances are hinted to whole pixels, so they vary with dpi
    probe_fig = plt.figure(dpi=150)
    probe = probe_fig.text(0, 0, "M" * 100, family="DejaVu Sans Mono", fontsize=fontsize)
    char_w = probe.get_window_extent(probe_fig.canvas.get_renderer()).width / 100 / probe_fig.dpi
    plt.close(probe_fig)
    width = 2 * margin + char_w * max(len(line) for line in lines)
    height = 2 * margin + line_h * len(lines)
    fig = plt.figure(figsize=(width, height), dpi=150)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.axis("off")
    for i, line in enumerate(lines):
        ax.text(margin, margin + line_h * (i + 0.5), line, family="DejaVu Sans Mono",
                fontsize=fontsize, va="center", ha="left", color=INK)

    def box(first_char, last_char, top_row, bottom_row, color):
        pad = 0.04
        ax.add_patch(Rectangle((margin + first_char * char_w - pad, margin + top_row * line_h),
                               (last_char - first_char) * char_w + 2 * pad,
                               (bottom_row - top_row + 1) * line_h,
                               fill=False, edgecolor=color, linewidth=2))

    for (first_char, last_char), color in boxes:
        box(first_char, last_char, header, last, color)
    box(r2_start, len(lines[r2_line].rstrip()), r2_line, r2_line, PINK)
    save(fig, "ols_summary.png")


def intervals():
    """Mean-response and prediction bands at age 55, with the lecture's new patient at BMI 30."""
    clinic, results = clinic_fit()
    grid = pd.DataFrame({"age": 55, "bmi": np.linspace(clinic["bmi"].min(), clinic["bmi"].max(), 200)})
    band = results.get_prediction(grid).summary_frame(alpha=0.05)
    patient = results.get_prediction(pd.DataFrame({"age": [55], "bmi": [30.0]})).summary_frame(alpha=0.05)
    p = patient.iloc[0].round(1)
    assert [p["mean"], p["mean_ci_lower"], p["mean_ci_upper"], p["obs_ci_lower"], p["obs_ci_upper"]] == \
        [147.2, 144.3, 150.2, 131.4, 163.0]

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    # Opaque tints of orange and blue, so the legend swatches match the overlapping bands
    ax.fill_between(grid["bmi"], band["obs_ci_lower"], band["obs_ci_upper"], color="#F9E7BF",
                    linewidth=0, label="95% prediction interval: one patient")
    ax.fill_between(grid["bmi"], band["mean_ci_lower"], band["mean_ci_upper"], color="#A6CEE4",
                    linewidth=0, label="95% mean-response interval: the average patient")
    ax.plot(grid["bmi"], band["mean"], color=BLUE, label="Fitted SBP at age 55")
    ax.errorbar([30], [p["mean"]], yerr=[[p["mean"] - p["obs_ci_lower"]], [p["obs_ci_upper"] - p["mean"]]],
                fmt="none", ecolor=VERMILLION, capsize=5, linewidth=1.5)
    ax.errorbar([30], [p["mean"]], yerr=[[p["mean"] - p["mean_ci_lower"]], [p["mean_ci_upper"] - p["mean"]]],
                fmt="o", color=VERMILLION, ecolor=VERMILLION, capsize=5, linewidth=3, markersize=6)
    ax.annotate(f"New patient, BMI 30: {p['mean']}\nmean {p['mean_ci_lower']} to {p['mean_ci_upper']}\n"
                f"one patient {p['obs_ci_lower']} to {p['obs_ci_upper']}",
                xy=(30, p["obs_ci_lower"]), xytext=(31.2, 124), fontsize=10, color=VERMILLION)
    ax.set(xlabel="BMI (kg/m²)", ylabel="SBP (mmHg)", title="Age 55: two intervals around one fitted line")
    ax.legend(loc="upper left", fontsize=9, frameon=False)
    save(fig, "ols_intervals.png")


def anscombe():
    """Anscombe's quartet: one fitted line and one R², four very different datasets."""
    x = [10, 8, 13, 9, 11, 14, 6, 4, 12, 7, 5]
    x4 = [8, 8, 8, 8, 8, 8, 8, 19, 8, 8, 8]
    ys = {"I": [8.04, 6.95, 7.58, 8.81, 8.33, 9.96, 7.24, 4.26, 10.84, 4.82, 5.68],
          "II": [9.14, 8.14, 8.74, 8.77, 9.26, 8.10, 6.13, 3.10, 9.13, 7.26, 4.74],
          "III": [7.46, 6.77, 12.74, 7.11, 7.81, 8.84, 6.08, 5.39, 8.15, 6.42, 5.73],
          "IV": [6.58, 5.76, 7.71, 8.84, 8.47, 7.04, 5.25, 12.50, 5.56, 7.91, 6.89]}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True, sharey=True)
    line_x = np.array([3, 20])
    for ax, (name, y) in zip(axes, ys.items()):
        data = pd.DataFrame({"x": x4 if name == "IV" else x, "y": y})
        fit = smf.ols("y ~ x", data=data).fit()
        assert round(fit.params["Intercept"], 2) == 3.00 and round(fit.params["x"], 2) == 0.50
        assert round(fit.rsquared, 2) == 0.67
        ax.scatter(data["x"], data["y"], color=BLUE, s=24, zorder=3)
        ax.plot(line_x, fit.params["Intercept"] + fit.params["x"] * line_x, color=VERMILLION)
        ax.set_title(f"{name}: y = 3.00 + 0.50x, R² = 0.67", fontsize=10)
        ax.set_xlabel("x")
    axes[0].set_ylabel("y")
    fig.tight_layout()
    save(fig, "anscombe_quartet.png")


# ---------------------------------------------------- Prediction: honest splits

def leakage_timeline():
    """The lecture's feature-availability table on a time axis, plus the weekly visit's target."""
    rows = [("age", 0, "Keep (known before the visit)", BLUE),
            ("sbp_today", 0, "Keep (measured during the visit)", BLUE),
            ("a1c_result", 24, "Exclude: leakage (lab reports next day)", VERMILLION),
            ("sbp_next_visit (target)", 168, "Target: next weekly visit", GREEN)]
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.axvspan(0, 190, color=GREY, alpha=0.12, linewidth=0)
    ax.axvline(0, color=INK, linestyle="--", linewidth=1.2)
    ax.text(2, len(rows) - 0.35, "prediction time: end of the visit", fontsize=9.5, color=INK)
    ax.text(186, -0.65, "not yet known at prediction time", fontsize=9.5, color="#555555", ha="right")
    for i, (name, hours, label, color) in enumerate(reversed(rows)):
        ax.plot(hours, i, "o", color=color, markersize=9)
        right_side = hours < 100  # the target's label sits to its left, inside the axes
        ax.text(hours + 4 if right_side else hours - 4, i, label, va="center", fontsize=10, color=color,
                ha="left" if right_side else "right")
    ax.set_yticks(range(len(rows)), [name for name, *_ in reversed(rows)])
    ax.set_xlim(-12, 190)
    ax.set_ylim(-0.9, len(rows) - 0.1)
    ax.set_xticks([0, 24, 48, 72, 96, 120, 144, 168])
    ax.set_xlabel("Hours after the visit")
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save(fig, "leakage_timeline.png")


def cyclic_hours():
    """The lecture's hours 0, 1, 12, 23 as plain numbers, then on the (cos, sin) circle."""
    hours = [0, 1, 12, 23]
    angle = {h: 2 * np.pi * h / 24 for h in range(24)}
    assert round(np.sin(angle[23]), 2) == -0.26 and round(np.cos(angle[23]), 2) == 0.97
    chord = 2 * np.sin(np.pi / 24)
    assert round(chord, 2) == 0.26

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4), gridspec_kw={"width_ratios": [1.35, 1]})
    left.axhline(0, color=GREY, linewidth=1)
    left.plot(hours, [0] * len(hours), "o", color=BLUE, markersize=9)
    for h in hours:
        left.text(h, 0.12, str(h), ha="center", fontsize=11, color=BLUE)
    left.annotate("", xy=(23, -0.25), xytext=(0, -0.25), arrowprops=dict(arrowstyle="<->", color=VERMILLION))
    left.text(11.5, -0.4, "23 and 0: 23 apart as numbers", ha="center", fontsize=10, color=VERMILLION)
    left.set(xlim=(-1.5, 24.5), ylim=(-0.6, 0.35), xticks=range(0, 24, 3), yticks=[],
             xlabel="hour", title="Hour as a plain number")
    left.spines["left"].set_visible(False)

    theta = np.linspace(0, 2 * np.pi, 300)
    right.plot(np.cos(theta), np.sin(theta), color=GREY, linewidth=1)
    for h in range(24):
        if h not in hours:
            right.plot(np.cos(angle[h]), np.sin(angle[h]), "o", color=GREY, markersize=3)
    for h in hours:
        cx, sy = np.cos(angle[h]), np.sin(angle[h])
        right.plot(cx, sy, "o", color=BLUE, markersize=9)
        right.text(1.2 * cx, 1.2 * sy, str(h), ha="center", va="center", fontsize=11, color=BLUE)
    right.text(0.55, 0.0, "23, 0, 1:\nneighbors,\n0.26 apart", ha="center", va="center", fontsize=9.5,
               color=VERMILLION)
    right.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal", xlabel="hour_cos (side to side)",
              ylabel="hour_sin (height)", title="Hour on the clock face", xticks=[-1, 0, 1], yticks=[-1, 0, 1])
    fig.tight_layout()
    save(fig, "cyclic_hours.png")


def overfit_polynomial():
    """Degree 0, 1, and 10 curves on 30 diabetes patients: underfit, fit, overfit."""
    frame = diabetes()
    order = np.random.default_rng(0).permutation(len(frame))
    train, rest = frame.iloc[order[:30]], frame.iloc[order[30:]]
    valid = rest[rest["bmi"].between(train["bmi"].min(), train["bmi"].max())]
    assert len(valid) == 402
    expected = {0: (64, 68), 1: (52, 53), 10: (45, 82)}
    grid = np.linspace(train["bmi"].min(), train["bmi"].max(), 400)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    for ax, (degree, label) in zip(axes, [(0, "underfit"), (1, "good fit"), (10, "overfit")]):
        curve = np.polynomial.Polynomial.fit(train["bmi"], train["progression"], degree)
        maes = (mean_absolute_error(train["progression"], curve(train["bmi"])),
                mean_absolute_error(valid["progression"], curve(valid["bmi"])))
        assert tuple(round(m) for m in maes) == expected[degree], maes
        ax.scatter(valid["bmi"], valid["progression"], s=8, color=GREY, alpha=0.35, label="validation patients")
        ax.scatter(train["bmi"], train["progression"], s=22, color=BLUE, label="30 training patients")
        ax.plot(grid, curve(grid), color=VERMILLION, linewidth=2)
        ax.set_title(f"Degree {degree}: {label}\ntrain MAE {maes[0]:.0f}, validation MAE {maes[1]:.0f}",
                     fontsize=10.5)
        ax.set_xlabel("BMI (kg/m²)")
    axes[0].set_ylabel("Progression after one year (score)")
    axes[0].set_ylim(0, 400)
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "overfit_polynomial.png")


def target_time_split():
    """Demo 1's 210 readings by target time, colored by role, with the label-availability cutoff."""
    readings = demo1_readings()
    train = readings[readings["target_time"] < "2026-02-09"]
    valid = readings[(readings["target_time"] >= "2026-02-09") & (readings["target_time"] < "2026-02-23")]
    test = readings[readings["target_time"] >= "2026-02-23"]
    cutoff = valid["reading_time"].min()
    fit_rows = train[train["target_time"] <= cutoff]
    late = train[train["target_time"] > cutoff]
    assert (len(fit_rows), len(late), len(valid), len(test)) == (91, 29, 60, 30)
    assert str(cutoff) == "2026-02-02 01:58:00", cutoff

    fig, ax = plt.subplots(figsize=(10, 4.6))
    for part, color, label in [
        (fit_rows, BLUE, f"training, label known by the cutoff: {len(fit_rows)}"),
        (late, GREY, f"training, label arrives after the cutoff (not fitted): {len(late)}"),
        (valid, ORANGE, f"validation: {len(valid)}"),
        (test, GREEN, f"test: {len(test)}"),
    ]:
        ax.scatter(part["target_time"], part["patient_id"].str[1:].astype(int), s=16, color=color, label=label)
    ax.axvline(cutoff, color=INK, linestyle="--", linewidth=1.2)
    ax.text(cutoff, 31.3, f"  first validation prediction, {cutoff:%b} {cutoff.day} {cutoff:%H:%M}",
            fontsize=9.5, va="bottom")
    ax.set(xlabel="Target time (when sbp_next is measured)", ylabel="Patient", ylim=(0, 33),
           yticks=[1, 10, 20, 30])
    ax.xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%b %d"))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=9, frameon=False)
    save(fig, "target_time_split.png")


# ------------------------------------------------------------- scikit-learn

def ridge_lasso_paths():
    """Coefficient paths on Demo 2's 264 scaled training rows; Lasso at alpha 2 keeps 7 of 10."""
    d = demo2()
    X = StandardScaler().fit_transform(d["X_train"])
    y = d["y_train"]
    alphas = np.logspace(-2, 3, 120)
    lasso2 = Lasso(alpha=2.0).fit(X, y)
    assert (lasso2.coef_ != 0).sum() == 7
    assert lasso2.coef_.round(1)[DIABETES_FEATURES.index("bmi")] == 25.5
    highlight = {"bmi": BLUE, "ltg": ORANGE, "bp": GREEN, "tc": VERMILLION, "ldl": PINK}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharey=True)
    for ax, estimator, title, chosen in [
        (axes[0], Ridge, "Ridge (L2): overall size shrinks, none set to exactly 0", 10),
        (axes[1], Lasso, "Lasso (L1): coefficients reach exactly 0", 2),
    ]:
        paths = np.array([estimator(alpha=a, max_iter=100_000).fit(X, y).coef_ for a in alphas])
        if estimator is Ridge:
            tc, ldl = DIABETES_FEATURES.index("tc"), DIABETES_FEATURES.index("ldl")
            at10 = Ridge(alpha=10).fit(X, y).coef_
            assert round(paths[0, tc]) == -51 and round(at10[tc]) == -12, (paths[0, tc], at10[tc])
            assert round(paths[0, ldl]) == 27 and round(at10[ldl]) == -3, (paths[0, ldl], at10[ldl])
            assert (paths != 0).all()
        for j, name in enumerate(DIABETES_FEATURES):
            color = highlight.get(name, GREY)
            ax.plot(alphas, paths[:, j], color=color, linewidth=2 if name in highlight else 1,
                    label=name if name in highlight else None, zorder=3 if name in highlight else 2)
        ax.axhline(0, color=INK, linewidth=0.6)
        ax.axvline(chosen, color=INK, linestyle=":", linewidth=1.2)
        note = f"alpha={chosen}" + (": 7 of 10 nonzero" if estimator is Lasso else "")
        ax.text(chosen * 1.15, -46, note, fontsize=9.5)
        ax.set(xscale="log", title=title, xlabel="alpha (penalty strength, log scale)")
    axes[1].plot([], [], color=GREY, label="other 5 features")
    axes[0].set_ylabel("Coefficient (score points per SD)")
    axes[1].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9.5, frameon=False)
    fig.tight_layout()
    save(fig, "ridge_lasso_paths.png")


def logistic_threshold():
    """The sigmoid turning a weighted sum into 0.72, flagged 1 at a 0.5 threshold."""
    z = np.linspace(-6, 6, 300)
    point = np.log(0.72 / 0.28)
    assert round(1 / (1 + np.exp(-point)), 2) == 0.72
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.axhspan(0.5, 1.05, color=ORANGE, alpha=0.12, linewidth=0)
    ax.plot(z, 1 / (1 + np.exp(-z)), color=BLUE, linewidth=2)
    ax.axhline(0.5, color=INK, linestyle="--", linewidth=1)
    ax.text(-5.8, 0.53, "threshold 0.5: above it, flag 1", fontsize=9.5)
    ax.text(-5.8, 0.44, "at or below: flag 0", fontsize=9.5)
    ax.plot([point], [0.72], "o", color=VERMILLION, markersize=8)
    ax.annotate(f"weighted sum {point:.2f}\nprobability 0.72, flag 1", xy=(point, 0.72),
                xytext=(1.8, 0.6), fontsize=10, color=VERMILLION,
                arrowprops=dict(arrowstyle="->", color=VERMILLION))
    ax.set(xlabel="Weighted sum of features (log-odds)", ylabel="Predicted probability of readmission",
           ylim=(-0.03, 1.05))
    save(fig, "logistic_threshold.png")


def validation_diagnostics():
    """Demo 2, Part 8: the Lasso's 89 validation predictions against actual, and their residuals."""
    d = demo2()
    predicted = d["models"]["lasso (alpha=2)"].predict(d["X_valid"])
    residual = d["y_valid"] - predicted
    assert len(residual) == 89 and round(residual.mean(), 1) == 14.2
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].scatter(d["y_valid"], predicted, color=BLUE, alpha=0.7, s=22)
    axes[0].plot([25, 346], [25, 346], color=GREY, linestyle="--", label="perfect prediction")
    axes[0].set(xlabel="Actual progression (score)", ylabel="Predicted progression (score)",
                title="Predicted against actual")
    axes[0].legend(loc="upper left", fontsize=9, frameon=False)
    axes[1].scatter(predicted, residual, color=BLUE, alpha=0.7, s=22)
    axes[1].axhline(0, color=GREY, linestyle="--")
    axes[1].axhline(residual.mean(), color=ORANGE, linestyle=":", linewidth=1.5,
                    label=f"mean residual {residual.mean():.1f}")
    axes[1].set(xlabel="Predicted progression (score)", ylabel="Residual, actual - predicted (score)",
                title="Residuals against predictions")
    axes[1].legend(loc="upper left", fontsize=9, frameon=False)
    fig.tight_layout()
    save(fig, "validation_diagnostics.png")


def confusion_matrix_figure():
    """Demo 2, Part 10: the Lasso flag at progression above 200 on 89 validation patients."""
    d = demo2()
    actual = (d["y_valid"] > 200).astype(int)
    flagged = (d["models"]["lasso (alpha=2)"].predict(d["X_valid"]) > 200).astype(int)
    (tn, fp), (fn, tp) = confusion_matrix(actual, flagged)
    assert (tn, fp, fn, tp) == (55, 5, 17, 12)
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    cells = [(0, 1, "TN", tn, "correct"), (1, 1, "FP", fp, "false alarm"),
             (0, 0, "FN", fn, "missed"), (1, 0, "TP", tp, "caught")]
    for col, row, short, count, note in cells:
        color = BLUE if short in ("TN", "TP") else VERMILLION
        ax.add_patch(Rectangle((col, row), 1, 1, facecolor=color, alpha=0.15, edgecolor="white", linewidth=3))
        ax.text(col + 0.5, row + 0.58, f"{short} {count}", ha="center", va="center", fontsize=17,
                fontweight="bold", color=color)
        ax.text(col + 0.5, row + 0.3, note, ha="center", va="center", fontsize=10.5, color=INK)
    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)
    ax.set_xticks([0.5, 1.5], ["not flagged\n(predicted ≤ 200)", "flagged\n(predicted > 200)"])
    ax.set_yticks([0.5, 1.5], ["actual\n> 200", "actual\n≤ 200"])
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)
    ax.set_title("Progression flag, 89 validation patients", fontsize=11.5)
    ax.text(1, -0.42, f"recall = TP / (TP + FN) = {tp} / {tp + fn} = {tp / (tp + fn):.3f}\n"
            f"precision = TP / (TP + FP) = {tp} / {tp + fp} = {tp / (tp + fp):.3f}",
            ha="center", va="top", fontsize=10.5, family="DejaVu Sans Mono")
    save(fig, "confusion_matrix.png")


def permutation_importance_chart():
    """Demo 2, Part 9: increase in validation MAE when each column is shuffled."""
    d = demo2()
    result = permutation_importance(d["models"]["lasso (alpha=2)"], d["X_valid"], d["y_valid"],
                                    scoring="neg_mean_absolute_error", n_repeats=10, random_state=42)
    table = pd.DataFrame({"mean": result.importances_mean, "std": result.importances_std},
                         index=DIABETES_FEATURES).sort_values("mean")
    top = table.round(2).iloc[::-1]
    assert top.loc["bmi"].tolist() == [10.5, 1.98] and top.loc["ltg"].tolist() == [10.43, 2.06]
    assert top.loc["bp"].tolist() == [3.79, 1.07] and top.loc["hdl"].tolist() == [2.9, 1.01]
    assert (top.loc[["age", "ldl", "tch"], "mean"] == 0).all()
    fig, ax = plt.subplots(figsize=(7.5, 4.3))
    colors = [BLUE if m > 0 else GREY for m in table["mean"]]
    ax.barh(table.index, table["mean"], xerr=table["std"], color=colors, capsize=3,
            error_kw=dict(ecolor=INK, linewidth=1))
    for i, (m, s) in enumerate(zip(table["mean"], table["std"])):
        ax.text(m + s + 0.25, i, f"{m:.2f} ± {s:.2f}" if m > 0 else "0.00 (Lasso set it to 0)",
                va="center", fontsize=9)
    ax.set_xlabel("Increase in validation MAE after shuffling (score points)")
    ax.set_xlim(0, 15.5)
    ax.tick_params(axis="y", length=0)
    save(fig, "permutation_importance.png")


def freeze_test_once():
    """Demo 2: four validation MAEs that pick the Lasso, then its single test MAE."""
    d = demo2()
    rows = list(d["valid_mae"].items())
    fig, ax = plt.subplots(figsize=(8, 3.6))
    labels = [name for name, _ in rows] + ["frozen Lasso, test rows"]
    for i, (name, mae) in enumerate(rows):
        y = len(labels) - 1 - i
        ax.plot(mae, y, "o", color=BLUE, markersize=9)
        ax.text(mae + 0.6, y, f"{mae:.2f}", va="center", fontsize=10)
    ax.plot(d["test_mae"], 0, "D", color=VERMILLION, markersize=9)
    ax.text(d["test_mae"] + 0.6, 0, f"{d['test_mae']:.2f}, opened once", va="center", fontsize=10,
            color=VERMILLION)
    ax.axhline(0.5, color=GREY, linewidth=0.8)
    ax.set_yticks(range(len(labels)), labels[::-1])
    ax.text(66.5, 4.35, "validation MAE (choose)", ha="right", fontsize=9.5, color=BLUE)
    ax.text(66.5, 0.25, "test MAE (report)", ha="right", fontsize=9.5, color=VERMILLION, va="top")
    ax.set(xlim=(38, 67), ylim=(-0.6, 4.6), xlabel="Mean absolute error (progression score points)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, "freeze_test_once.png")


# ------------------------------------------------------- trees and boosting

def decision_tree():
    """A depth-3 regression tree on age, BMI, and BP for all 442 diabetes patients."""
    frame = diabetes()
    features = ["age", "bmi", "bp"]
    tree = DecisionTreeRegressor(max_depth=3, random_state=0).fit(frame[features], frame["progression"])
    leaves = tree.tree_.children_left == -1
    leaf_means = sorted(tree.tree_.value[leaves].ravel().round(1))
    assert leaf_means[0] == 101.5 and leaf_means[-1] == 268.5 and leaves.sum() == 8
    assert features[tree.tree_.feature[0]] == "bmi" and round(tree.tree_.threshold[0], 1) == 27.2
    assert "age" not in {features[f] for f in tree.tree_.feature[~leaves]}
    assert sorted(tree.tree_.n_node_samples[leaves]).count(1) == 1
    fig, ax = plt.subplots(figsize=(15, 6))
    plot_tree(tree, feature_names=features, impurity=False, precision=1, rounded=True, fontsize=10, ax=ax)
    save(fig, "decision_tree.png")


def boosting_stages():
    """Depth-1 boosted trees, learning rate 0.1, on progression against BMI after 1, 10, 100 trees."""
    frame = diabetes()
    model = GradientBoostingRegressor(max_depth=1, learning_rate=0.1, n_estimators=100, random_state=0)
    model.fit(frame[["bmi"]], frame["progression"])
    grid = pd.DataFrame({"bmi": np.linspace(frame["bmi"].min(), frame["bmi"].max(), 600)})
    grid_stages = list(model.staged_predict(grid))
    train_stages = list(model.staged_predict(frame[["bmi"]]))
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.9), sharey=True)
    maes = []
    for ax, n in zip(axes, [1, 10, 100]):
        mae = mean_absolute_error(frame["progression"], train_stages[n - 1])
        maes.append(mae)
        ax.scatter(frame["bmi"], frame["progression"], s=7, color=GREY, alpha=0.45)
        ax.axhline(frame["progression"].mean(), color=INK, linestyle=":", linewidth=1)
        ax.plot(grid["bmi"], grid_stages[n - 1], color=VERMILLION, linewidth=2)
        ax.set_title(f"After {n} tree{'s' if n > 1 else ''}: training MAE {mae:.1f}", fontsize=10.5)
        ax.set_xlabel("BMI (kg/m²)")
    assert [round(m, 1) for m in maes] == [64.1, 56.5, 50.1], maes
    axes[0].set_ylabel("Progression after one year (score)")
    axes[0].text(frame["bmi"].max(), frame["progression"].mean() - 20, "starting guess: mean 152.1",
                 ha="right", fontsize=9, bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1))
    assert round(frame["progression"].mean(), 1) == 152.1
    fig.tight_layout()
    save(fig, "boosting_stages.png")
    print("  training MAE after 1, 10, 100 trees:", [round(m, 1) for m in maes])


def xgboost_early_stopping():
    """Demo 3, Part 6: validation log loss per round; best round 54, stopped 10 rounds later."""
    d = demo3()
    curve, best = d["xgb_curve"], d["best_iteration"]
    assert len(curve) == best + 11 and round(curve[best], 4) == 0.1168
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(range(len(curve)), curve, color=BLUE, linewidth=2)
    ax.plot(best, curve[best], "o", color=VERMILLION, markersize=8)
    ax.annotate(f"best round {best}: loss {curve[best]:.4f}", xy=(best, curve[best]), xytext=(best - 24, 0.24),
                fontsize=10, color=VERMILLION, arrowprops=dict(arrowstyle="->", color=VERMILLION))
    ax.annotate(f"stopped at round {len(curve) - 1}\n(10 rounds, no improvement)", xy=(len(curve) - 1, curve[-1]),
                xytext=(len(curve) - 1, 0.36), ha="right", fontsize=9.5,
                arrowprops=dict(arrowstyle="->", color=INK, relpos=(1, 0)))
    ax.set(xlabel="Boosting round (counting from 0)", ylabel="Validation log loss",
           title="XGBoost with early_stopping_rounds=10, budget 500")
    save(fig, "xgboost_early_stopping.png")


# ----------------------------------------------------------- deep learning

def keras_loss_curves():
    """Demo 3, Part 8: the 64-32 network's training and validation loss over 50 epochs."""
    history = demo3()["plain"]
    best = history["val_loss"].idxmin()
    assert best == 20 and round(history["val_loss"][20], 4) == 0.0726 and round(history["val_loss"][50], 4) == 0.0903
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(history.index, history["loss"], color=BLUE, linewidth=2, label="training loss")
    ax.plot(history.index, history["val_loss"], color=ORANGE, linewidth=2, label="validation loss")
    ax.plot(best, history["val_loss"][best], "o", color=VERMILLION, markersize=8)
    ax.annotate(f"lowest validation loss {history['val_loss'][best]:.4f}, epoch {best}",
                xy=(best, history["val_loss"][best]), xytext=(18, 0.3), fontsize=10, color=VERMILLION,
                arrowprops=dict(arrowstyle="->", color=VERMILLION))
    ax.text(50, history["val_loss"][50] + 0.025, f"{history['val_loss'][50]:.4f}", ha="right", fontsize=10,
            color=ORANGE)
    ax.set(xlabel="Epoch", ylabel="Binary cross-entropy loss", title="64-32 network, 341 training biopsies")
    ax.legend(frameon=False)
    save(fig, "keras_loss_curves.png")


def keras_dropout_curves():
    """Demo 3, Part 9: plain 64-32 against 64-32 with dropout and L2, same validation accuracy."""
    d = demo3()
    accuracy = d["accuracy"]
    assert round(accuracy["Neural net 64-32"], 4) == round(accuracy["Neural net 64-32 + dropout/L2"], 4) == 0.9737
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, history, title in [(axes[0], d["plain"], "Plain 64-32"),
                               (axes[1], d["reg"], "64-32 + dropout/L2 (loss includes the L2 penalty)")]:
        ax.plot(history.index, history["loss"], color=BLUE, linewidth=2, label="training loss")
        ax.plot(history.index, history["val_loss"], color=ORANGE, linewidth=2, label="validation loss")
        ax.set(xlabel="Epoch", ylabel="Loss", title=f"{title}\nvalidation accuracy 0.9737 (111 of 114)")
        ax.title.set_fontsize(10.5)
        ax.legend(frameon=False)
    fig.tight_layout()
    save(fig, "keras_dropout_curves.png")


def model_comparison():
    """Demo 3, Part 10: validation accuracy for every candidate on the same 114 biopsies."""
    d = demo3()
    names = list(d["accuracy"])
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    for i, name in enumerate(names):
        acc = d["accuracy"][name]
        y = len(names) - 1 - i
        ax.plot(acc, y, "o", color=GREY if i == 0 else BLUE, markersize=9)
        ax.text(acc - 0.012, y, f"{acc:.4f} ({round(acc * d['n_valid'])}/{d['n_valid']})", ha="right",
                va="center", fontsize=9.5)
    ax.set_yticks(range(len(names)), names[::-1])
    ax.set(xlim=(0.45, 1.0), xlabel="Validation accuracy (114 biopsies)")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    save(fig, "model_comparison.png")


FIGURES = [residuals_vs_fitted, ols_summary, intervals, anscombe, leakage_timeline, cyclic_hours,
           overfit_polynomial, target_time_split, ridge_lasso_paths, logistic_threshold,
           validation_diagnostics, confusion_matrix_figure, permutation_importance_chart, freeze_test_once,
           decision_tree, boosting_stages, xgboost_early_stopping, keras_loss_curves, keras_dropout_curves,
           model_comparison]

if __name__ == "__main__":
    chosen = sys.argv[1:]
    unknown = set(chosen) - {f.__name__ for f in FIGURES}
    if unknown:
        sys.exit(f"Unknown figure names: {sorted(unknown)}")
    for figure in FIGURES:
        if not chosen or figure.__name__ in chosen:
            figure()
