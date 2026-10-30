---
jupyter:
  jupytext:
    notebook_metadata_filter: language_info
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.18.1
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
  language_info:
    name: python
    version: 3.13
---

# Demo 2: Does a Forecast Beat the Nurses' Rule for the Monday Call List?

Cedar Valley Health's care-management director, Dr. Leah Kim, takes three pieces of work to the November 18, 2025, operations committee. Demo 1 measured BMI's association with SBP and built the weekly forecast table for the remote blood-pressure program. This demo builds the forecast and the call list; Demo 3 builds the post-discharge call list.

**This phase's ask.** By November 4, Dr. Kim needs two answers for the committee, which decides whether a model replaces the rules the care team uses today:

- **The dashboard forecast**: each patient's mean SBP for the coming week. Today the dashboard repeats this week's mean. How many mmHg better is a model, and does the gain hold over time?
- **The nurse call list**: the pharmacist already reviews every patient whose week averaged 150 mmHg or more. The nurses want to reach patients still under 150 this week who will average 150 or more next week, before they lose control. They can make 100 such calls a week, and today they call the 100 patients whose week came closest to 150. How many more of the right patients would a model reach?

The data is Demo 1's weekly table of 140,000 patient-weeks. All patients and values are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

## Core walkthrough

### 1. Rebuild the Monday table with what the program knows at 08:00

This supplied cell repeats Demo 1's steps 7 to 9 in one place: download and build the data if missing, clean the readings, build one row per patient-week from the readings uploaded by Monday 08:00, attach next week's mean as the target, and split on target week. It also builds the features the care team named as signs of a coming change, each known at the Monday run:

- **Recent level**: `sbp_week` (this week's mean), `sbp_4wk` (mean of the last four weekly means, a grouped rolling mean from Lecture 09), and `trend` (change from last week, a grouped `diff()`).
- **Readings**: `n_readings` and `n_symptom` (readings taken because the patient felt unwell), plus the week's mean reading time on the clock face (`hour_sin`, `hour_cos`, Demo 1 step 8).
- **Medications**, from the EHR list and pharmacy fills attached to each reading: `bp_med_count`, `days_to_refill` (days until the current fill runs out; negative means overdue), `weeks_since_change`, and `drug_added_recently` (a drug added in the last three weeks, still taking effect).
- **Program schedule**: `review_next_week`. The pharmacist reviews every patient at the end of each four enrolled weeks and adds a drug when the average is high.
- **Two engineered interactions**: `overdue_x_meds` (days overdue, capped at 14, times the number of drugs: a patient off three drugs rebounds more than one off one drug) and `refill_risk` (number of drugs, times a refill due within a week, times the share of days without medication in the year before enrollment, `1 - pdc_prior_year`).

```python
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/10/demo"
TABLES = ["pcp_visits", "rpm_patients", "rpm_readings", "discharges"]
if not Path("generate_data.py").exists():
    urlretrieve(f"{REPO_RAW}/generate_data.py", "generate_data.py")
if not all(Path(f"data/{name}.parquet").exists() for name in TABLES):
    import generate_data
    generate_data.build()

readings = pd.read_parquet("data/rpm_readings.parquet")
enrolled = pd.read_parquet("data/rpm_patients.parquet")
readings = readings.sort_values("upload_time").drop_duplicates(["patient_id", "reading_time"])
readings = readings[readings["sbp"].between(60, 260)].copy()
readings["week"] = readings["reading_time"].dt.floor("D") - readings["reading_time"].dt.dayofweek * pd.Timedelta(days=1)
readings["run_time"] = readings["week"] + pd.Timedelta(days=7, hours=8)
readings["symptom"] = readings["reason"] == "symptoms"
hour = readings["reading_time"].dt.hour + readings["reading_time"].dt.minute / 60
readings["hour_sin"] = np.sin(2 * np.pi * hour / 24)
readings["hour_cos"] = np.cos(2 * np.pi * hour / 24)
readings = readings.sort_values(["patient_id", "reading_time"])

seen = readings[readings["upload_time"] <= readings["run_time"]]
seen_scheduled = seen[~seen["symptom"]]
scheduled = readings[~readings["symptom"]]
keys = ["patient_id", "week"]
weekly = (
    seen_scheduled.groupby(keys).agg(sbp_week=("sbp", "mean"), n_readings=("sbp", "size"),
                                     hour_sin=("hour_sin", "mean"), hour_cos=("hour_cos", "mean"))
    .join(seen.groupby(keys).agg(n_symptom=("symptom", "sum"), bp_med_count=("bp_med_count", "last"),
                                 refill_due=("refill_due_date", "last"), overdue=("refill_days_overdue", "last"),
                                 last_change=("last_med_change", "last"), change_type=("last_change_type", "last")))
    .join(scheduled.groupby(keys).agg(sbp_complete=("sbp", "mean"), last_upload=("upload_time", "max")), how="outer")
    .reset_index()
)

by_patient = weekly.groupby("patient_id")
weekly["sbp_next_week"] = by_patient["sbp_complete"].shift(-1)
weekly["target_week"] = by_patient["week"].shift(-1)
weekly["label_arrival"] = by_patient["last_upload"].shift(-1)
weekly["sbp_4wk"] = by_patient["sbp_week"].transform(lambda s: s.rolling(4, min_periods=1).mean())
weekly["trend"] = by_patient["sbp_week"].diff().fillna(0)
weekly = weekly[(weekly["target_week"] - weekly["week"] == pd.Timedelta(days=7)) & weekly["sbp_week"].notna()]
weekly = weekly.merge(enrolled, on="patient_id", how="left", validate="many_to_one")

weekly["run_time"] = weekly["week"] + pd.Timedelta(days=7, hours=8)
weekly["days_to_refill"] = (weekly["refill_due"] - weekly["run_time"].dt.normalize()).dt.days
days_since_change = (weekly["run_time"] - weekly["last_change"]).dt.days
weekly["weeks_since_change"] = (days_since_change / 7).fillna(52).clip(upper=52)
weekly["drug_added_recently"] = ((weekly["change_type"] == "added") & (days_since_change <= 21)).astype(int)
days_into_cycle = (weekly["target_week"] - weekly["enroll_date"]).dt.days % 28
weekly["review_next_week"] = ((27 - days_into_cycle) % 28 <= 6).astype(int)
weekly["overdue_x_meds"] = weekly["overdue"].clip(upper=14) * weekly["bp_med_count"]
refill_soon = weekly["days_to_refill"] <= 7
weekly["refill_risk"] = weekly["bp_med_count"] * refill_soon * (1 - weekly["pdc_prior_year"])

train = weekly[weekly["target_week"] < "2025-06-01"]
valid = weekly[(weekly["target_week"] >= "2025-06-01") & (weekly["target_week"] < "2025-08-01")]
test = weekly[weekly["target_week"] >= "2025-08-01"]
fit_rows = train[train["label_arrival"] <= valid["run_time"].min()]

numeric = ["sbp_week", "sbp_4wk", "trend", "n_readings", "n_symptom", "hour_sin", "hour_cos",
           "bp_med_count", "days_to_refill", "weeks_since_change", "drug_added_recently", "review_next_week",
           "overdue_x_meds", "refill_risk", "age", "bmi", "egfr", "salt_score", "pdc_prior_year"]
categorical = ["sex", "clinic", "cuff_size", "device", "diabetes", "sleep_apnea", "nsaid_use"]
features = numeric + categorical
print("fit rows:", len(fit_rows), " validation rows:", len(valid), " test rows:", len(test))
display(fit_rows[features].isna().sum().loc[lambda counts: counts > 0].rename("missing"))
```

Expect `fit rows: 68324  validation rows: 37980  test rows: 33785`, the same split as Demo 1. One feature has gaps: `bmi`, missing for patients whose enrollment form left it blank (4,852 fit rows).

### 2. What do the nurses' current rules score?

A model is worth deploying only if it beats the rules the team could use with no model. Score three, with validation **MAE** (the average miss in mmHg) and RMSE:

- **Training mean**: `DummyRegressor` predicts the fit rows' average for everyone.
- **Persistence**, the dashboard today: next week's mean equals this week's.
- **4-week average**: next week's mean equals the mean of the last four weeks.

Every candidate goes into one `results` dictionary keyed by name, so rerunning a cell replaces its row instead of adding a copy.

```python
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

X_fit, y_fit = fit_rows[features], fit_rows["sbp_next_week"]
X_valid, y_valid = valid[features], valid["sbp_next_week"]

def score(name, predicted):
    results[name] = {"MAE": mean_absolute_error(y_valid, predicted),
                     "RMSE": np.sqrt(mean_squared_error(y_valid, predicted))}

results = {}
mean_model = DummyRegressor(strategy="mean").fit(X_fit, y_fit)
score("training mean", mean_model.predict(X_valid))
score("persistence (this week)", valid["sbp_week"])
score("4-week average", valid["sbp_4wk"])
display(pd.DataFrame(results).T.round(2))
```

Expect MAE of about 10.2 mmHg for the training mean, 4.28 for persistence, and 4.29 for the 4-week average. A patient's own recent weeks say far more than the program-wide mean. Four weeks smooth out more noise than one but lag behind real changes, so the two rules tie. Any model has to beat 4.28 to replace the dashboard's rule.

### 3. Does a model beat the rules?

The first attempt fits `LinearRegression` straight on the numeric columns. This intentional error shows what happens with the blank BMIs from step 1.

```python
from sklearn.linear_model import LinearRegression

try:
    LinearRegression().fit(fit_rows[numeric], y_fit)
except ValueError as error:
    print("ValueError:", str(error).splitlines()[0])
```

Expect `ValueError: Input X contains NaN.` The text columns (`sex`, `clinic`, `device`) would fail next, since a linear model needs numbers. The fix is a pipeline that prepares each group of columns: numbers get the training median for missing values (`SimpleImputer`) and then scaling (`StandardScaler`); categories become 0/1 columns (`OneHotEncoder`). `ColumnTransformer` routes each group, and the `Pipeline` learns all of it from the fit rows only. The same preparation feeds three linear models, first on the level columns alone and then on every feature.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_preprocess(numeric_columns=numeric, categorical_columns=categorical):
    numeric_steps = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical_steps = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    return ColumnTransformer([("numeric", numeric_steps, numeric_columns),
                              ("categorical", categorical_steps, categorical_columns)])

level_only = ["sbp_week", "sbp_4wk"]
level_model = Pipeline([("preprocess", make_preprocess(level_only, [])), ("model", LinearRegression())])
level_model.fit(fit_rows[level_only], y_fit)
score("linear, level columns only", level_model.predict(valid[level_only]))

candidates = {"linear regression": LinearRegression(),
              "ridge (alpha=10)": Ridge(alpha=10),
              "lasso (alpha=0.05)": Lasso(alpha=0.05)}
fitted = {}
for name, estimator in candidates.items():
    fitted[name] = Pipeline([("preprocess", make_preprocess()), ("model", estimator)]).fit(X_fit, y_fit)
    score(name, fitted[name].predict(X_valid))
display(pd.DataFrame(results).T.round(3))

mae = pd.DataFrame(results).T["MAE"].drop("training mean").sort_values(ascending=False)
ax = mae.plot.barh(figsize=(7, 3.5), xlim=(0, 5))
ax.bar_label(ax.containers[0], fmt="%.2f", padding=3)
ax.axvline(results["persistence (this week)"]["MAE"], color="gray", linestyle="--")
ax.set(xlabel="Validation MAE (mmHg), lower is better", title="Bars left of the dashed persistence line beat the rule")
plt.show()
plt.close(ax.figure)
```

Expect a linear model on the two level columns alone at 3.91 mmHg: weighting this week and the last four together beats either rule by 0.4. All three linear models on every feature reach 3.72, another 0.19 better and 13% below persistence. The chart leaves out the training mean (10.23) so the small gaps show: four model bars end left of the dashed persistence line at 4.28. Two lessons:

- **The engineered features did real work.** The medication and refill columns tell the model what the level columns cannot: a patient who stopped refilling three drugs is about to rise, and one whose new drug is still taking effect is about to fall.
- **Regularization changes little here.** With 68,324 rows and 36 prepared columns, Ridge and Lasso land within 0.003 mmHg of plain linear regression. Step 4 shows when the penalty matters.

### 4. Marina asks for its own model with the intake questionnaire

The Marina clinic's manager asks whether a clinic could run its own model from one month of data. The cell fits on Marina's January weeks only and scores on its June and July weeks. She also asks to add the whole intake questionnaire: 30 answers from 1 to 5 about sleep, stress, diet, and mood. That makes 49 numeric columns for 229 patient-weeks from about 100 patients, and the questionnaire answers are constant within a patient, so for them only the patients count. Ridge and Lasso were built for this case: many columns for few rows. The cell fits each linear model with and without the questionnaire and scores persistence on the same rows.

```python
intake = [column for column in weekly.columns if column.startswith("intake_q")]
wide = numeric + intake
marina_fit = fit_rows[(fit_rows["clinic"] == "Marina") & (fit_rows["target_week"] < "2025-02-01")]
marina_valid = valid[valid["clinic"] == "Marina"]
print("Marina fit rows:", len(marina_fit), " patients:", marina_fit["patient_id"].nunique())

small = {}
for name, estimator in [("linear regression", LinearRegression()), ("ridge (alpha=10)", Ridge(alpha=10)),
                        ("lasso (alpha=0.5)", Lasso(alpha=0.5))]:
    for label, columns in [("19 core columns", numeric), ("49 with questionnaire", wide)]:
        model = Pipeline([("preprocess", make_preprocess(columns)), ("model", estimator)])
        model.fit(marina_fit[columns + categorical], marina_fit["sbp_next_week"])
        small[(name, label)] = mean_absolute_error(marina_valid["sbp_next_week"],
                                                   model.predict(marina_valid[columns + categorical]))
persistence = mean_absolute_error(marina_valid["sbp_next_week"], marina_valid["sbp_week"])
ax = pd.Series(small).unstack().plot.bar(rot=0, figsize=(8, 3.5), ylim=(0, 7))
for bars in ax.containers:
    ax.bar_label(bars, fmt="%.2f", padding=2, fontsize=8)
ax.axhline(persistence, color="gray", linestyle="--", label="persistence")
ax.set(ylabel="Validation MAE (mmHg)", title="Marina: 229 rows, with and without 30 questionnaire columns")
ax.legend(fontsize=8, ncols=3, loc="upper left")
plt.show()
plt.close(ax.figure)
print("persistence on the same rows:", round(persistence, 2))

kept = pd.Series(model.named_steps["model"].coef_, index=model.named_steps["preprocess"].get_feature_names_out())
print("Lasso kept", (kept != 0).sum(), "of", len(kept), "prepared columns")
display(kept[kept != 0].round(2).rename("kept by Lasso"))
```

Expect 229 fit rows from 103 patients, persistence at 4.19 mmHg on the same rows (the dashed line), and bars with these values:

|  | 19 core columns | 49 with questionnaire |
| --- | --- | --- |
| lasso (alpha=0.5) | 4.15 | 4.26 |
| linear regression | 4.23 | 5.58 |
| ridge (alpha=10) | 4.27 | 5.10 |

- **The questionnaire adds nothing.** Every model gets worse with the 30 answers added: Lasso by 0.11 mmHg, Ridge by 0.83, and plain linear regression by 1.35. In this synthetic data no answer moves blood pressure.
- **The penalty limits the damage.** With about two patients per column, plain linear regression fits noise in the answers and misses by 5.58 mmHg, 1.39 worse than persistence. Ridge shrinks every coefficient and recovers part of the loss (5.10). Lasso sets most coefficients to exactly 0: it keeps 12 of the 62 prepared columns (the two level columns, reading time, BMI, prior-year adherence, and six questionnaire answers it took from noise) and misses by 4.26.
- **No model earns its place here.** On the 19 core columns the three models land between 4.15 and 4.27, within 0.08 of persistence.

**What we tell Marina's manager**: one clinic's first month is too few rows for its own model, and the questionnaire makes every model worse. Marina's patients use the program-wide forecast.

### 5. Were June and July a lucky two months?

A committee member will ask whether the gain holds outside one validation period. `cross_val_score` with `TimeSeriesSplit` answers that: it cuts the fit and validation rows, sorted by target week, into blocks; each fold fits on everything before a block and scores that block. The test rows stay out.

```python
from sklearn.model_selection import TimeSeriesSplit, cross_val_score

dev = pd.concat([fit_rows, valid]).sort_values("target_week")
cv = TimeSeriesSplit(n_splits=4)
ridge_pipeline = Pipeline([("preprocess", make_preprocess()), ("model", Ridge(alpha=10))])
scores = cross_val_score(ridge_pipeline, dev[features], dev["sbp_next_week"], cv=cv,
                         scoring="neg_mean_absolute_error")

folds = pd.DataFrame([{"fit_rows": len(fit_idx),
                       "scored_from": dev["target_week"].iloc[score_idx].min().date(),
                       "scored_to": dev["target_week"].iloc[score_idx].max().date()}
                      for fit_idx, score_idx in cv.split(dev)])
folds["ridge_MAE"] = (-scores).round(2)
folds["persistence_MAE"] = [mean_absolute_error(dev["sbp_next_week"].iloc[score_idx], dev["sbp_week"].iloc[score_idx])
                            for _, score_idx in cv.split(dev)]
display(folds.round(2))

ax = folds.set_index(folds["scored_from"].astype(str))[["persistence_MAE", "ridge_MAE"]].plot(marker="o", figsize=(7, 3.5))
ax.set(xlabel="Fold (first target week scored)", ylabel="MAE (mmHg)", ylim=(0, 5),
       title="Ridge beats persistence in every fold")
plt.show()
plt.close(ax.figure)
```

Expect four folds scored from mid-March to the end of July, with Ridge between 3.51 and 3.78 mmHg and persistence between 4.05 and 4.35. In the chart the two lines run parallel, Ridge about 0.55 mmHg lower in every fold: the model's lead holds in every period. These folds ignore label arrival inside `dev`, so they compare periods retrospectively; the prospective check is still step 3's validation score.

### 6. Who is about to cross 150? Accuracy says the model is 96% right

The call list is a yes/no question, so the target becomes `alert = sbp_next_week >= 150`, and the same preparation feeds a logistic regression fit on patients under 150 this week. `predict_proba` returns one column per class; `[:, 1]` keeps the probability of crossing 150. Before trusting the model's accuracy, compare it with a rule that calls nobody.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

watch_fit = fit_rows[fit_rows["sbp_week"] < 150]
watch_valid = valid[valid["sbp_week"] < 150]
alert_fit = (watch_fit["sbp_next_week"] >= 150).astype(int)
alert_valid = (watch_valid["sbp_next_week"] >= 150).astype(int)
weeks = watch_valid["week"].nunique()
print("patients under 150 per week:", round(len(watch_valid) / weeks),
      " of whom cross 150 next week:", round(alert_valid.sum() / weeks))
print("alert rate: fit rows", round(alert_fit.mean(), 3), " validation", round(alert_valid.mean(), 3))

classifier = Pipeline([("preprocess", make_preprocess()), ("model", LogisticRegression(max_iter=2000))])
classifier.fit(watch_fit[features], alert_fit)
alert_probability = classifier.predict_proba(watch_valid[features])[:, 1]
flag = (alert_probability >= 0.5).astype(int)

print("always no: accuracy", round(accuracy_score(alert_valid, np.zeros(len(alert_valid))), 3))
print("model at 0.5: accuracy", round(accuracy_score(alert_valid, flag), 3),
      " precision", round(precision_score(alert_valid, flag), 3),
      " recall", round(recall_score(alert_valid, flag), 3))
print(confusion_matrix(alert_valid, flag))

# density=True scales each histogram to area 1, so 1,443 crossings and 30,604 others share one axis
fig, ax = plt.subplots(figsize=(7, 3.5))
bins = np.linspace(0, 1, 41)
ax.hist(alert_probability[alert_valid == 0], bins=bins, density=True, alpha=0.5, label="stayed under 150")
ax.hist(alert_probability[alert_valid == 1], bins=bins, density=True, alpha=0.5, label="crossed 150")
ax.axvline(0.5, color="black", linestyle="--", label="threshold 0.5")
ax.set(xlabel="Predicted probability of crossing 150", ylabel="Density (each group sums to 1)",
       title="Most patients who cross score below 0.5")
ax.legend()
plt.show()
plt.close(fig)
```

Expect about 3,561 patients under 150 each week, of whom about 160 average 150 or more the next week. The alert rate falls from 0.077 in the fit rows to 0.045 in validation: the program lowers blood pressure over time, and summer readings run lower. "Always no" scores 0.955 accuracy while calling nobody. At the default 0.5 threshold the model's accuracy is 0.956, its precision 0.535, and its recall only 0.194: the matrix `[[TN, FP], [FN, TP]]` shows 1,163 crossings missed and 280 caught over nine weeks. The histogram shows why: patients who stay under 150 pile up below 0.05, while those who cross spread from 0 to 1, and most of their orange area lies left of the 0.5 line. Accuracy cannot choose the list; recall at the nurses' budget can.

### 7. Which list fills 100 calls a week best?

A threshold of 0.5 is not what the nurses need: they can make 100 calls a week, so each week's list is the 100 patients with the highest probability. `groupby("week").rank(ascending=False)` numbers each week's patients from the highest score down. The rule the team uses today calls the 100 patients whose week came closest to 150. Both lists are scored at five budgets, in case the committee changes the nurses' hours.

```python
def call_list(name, scores, budget):
    rank = pd.Series(scores, index=watch_valid.index).groupby(watch_valid["week"]).rank(ascending=False)
    called = (rank <= budget).astype(int)
    return {"call_list": name, "calls_per_week": budget, "recall": recall_score(alert_valid, called),
            "precision": precision_score(alert_valid, called),
            "caught_per_week": (called & alert_valid).sum() / weeks}

tradeoff = pd.DataFrame([call_list(name, scores, budget)
                         for budget in [25, 50, 100, 150, 200]
                         for name, scores in [("model", alert_probability),
                                              ("rule: highest this week", watch_valid["sbp_week"])]])
display(tradeoff[tradeoff["calls_per_week"] == 100].round(3))

ax = tradeoff.pivot(index="calls_per_week", columns="call_list", values="caught_per_week").plot(marker="o", figsize=(7, 3.5))
ax.axvline(100, color="gray", linestyle="--")
ax.set(xlabel="Calls per week", ylabel="Patients reached who cross 150", title="The model reaches more patients at every budget")
plt.show()
plt.close(ax.figure)
```

Expect the model's line above the rule's at all five budgets, from 25 to 200 calls a week. At 100 calls a week (the dashed line) it reaches 47 of the patients who will cross 150 (recall 0.295, precision 0.473); the rule reaches 28 (recall 0.177, precision 0.288). Even at 150 calls a week the rule catches fewer (41) than the model at 100.

**What we tell Dr. Kim**: for the same 100 calls, the model's list reaches about 19 more patients a week before they lose control. Today's rule would need more than 150 calls a week, 50% more call hours, to match it.

### 8. Freeze the choices and test once for the committee

The frozen choices: Ridge (alpha=10) as the dashboard forecast, and the logistic call list at 100 calls a week. Refit both on every fit and validation row whose label had arrived by the first test run, then score the test rows once, with the same rules beside them. These are the numbers that go on the committee slide.

```python
train_valid = pd.concat([train, valid])
final_rows = train_valid[train_valid["label_arrival"] <= test["run_time"].min()]
final_forecast = Pipeline([("preprocess", make_preprocess()), ("model", Ridge(alpha=10))])
final_forecast.fit(final_rows[features], final_rows["sbp_next_week"])
print("final fit rows:", len(final_rows))
print("test MAE: ridge", round(mean_absolute_error(test["sbp_next_week"], final_forecast.predict(test[features])), 2),
      " persistence", round(mean_absolute_error(test["sbp_next_week"], test["sbp_week"]), 2))

final_watch = final_rows[final_rows["sbp_week"] < 150]
final_classifier = Pipeline([("preprocess", make_preprocess()), ("model", LogisticRegression(max_iter=2000))])
final_classifier.fit(final_watch[features], (final_watch["sbp_next_week"] >= 150).astype(int))

watch_test = test[test["sbp_week"] < 150]
alert_test = (watch_test["sbp_next_week"] >= 150).astype(int)
for name, scores in [("model", final_classifier.predict_proba(watch_test[features])[:, 1]),
                     ("rule: highest this week", watch_test["sbp_week"])]:
    rank = pd.Series(scores, index=watch_test.index).groupby(watch_test["week"]).rank(ascending=False)
    called = (rank <= 100).astype(int)
    print(f"test call list, {name}: recall {recall_score(alert_test, called):.3f}",
          f" precision {precision_score(alert_test, called):.3f}",
          f" caught per week {(called & alert_test).sum() / watch_test['week'].nunique():.1f}")
```

Expect 106,367 final fit rows. On the test weeks, Ridge misses by 4.00 mmHg against 4.55 for persistence, and the call list at 100 calls a week catches 44.8 crossings a week (recall 0.286, precision 0.448) against the rule's 27.9 (recall 0.178, precision 0.284). Both gains hold up on rows that played no part in any choice.

**What we tell the committee**: on August and September, weeks no choice was tuned on, the forecast misses by 0.55 mmHg less than the dashboard's rule, and the model's call list reaches 17 more patients a week who go on to average 150 or more.

## Independent practice

### 1. The care team asks what the forecast leans on

Permutation importance shuffles one column at a time on validation rows and measures how much the MAE worsens. A 10,000-row sample keeps it quick.

```python
from sklearn.inspection import permutation_importance

sample = valid.sample(10000, random_state=42)
importance = permutation_importance(fitted["ridge (alpha=10)"], sample[features], sample["sbp_next_week"],
                                    scoring="neg_mean_absolute_error", n_repeats=5, random_state=42)
ranked = pd.DataFrame({"feature": features, "mae_increase": importance.importances_mean,
                       "spread": importance.importances_std}).sort_values("mae_increase", ascending=False).head(8)
display(ranked.round(3))

ax = ranked.iloc[::-1].plot.barh(x="feature", y="mae_increase", xerr=ranked["spread"].iloc[::-1],
                                  legend=False, figsize=(7, 3.5))
ax.set(xlabel="Validation MAE increase when shuffled (mmHg)", ylabel="", title="What the Ridge forecast relies on")
plt.show()
plt.close(ax.figure)
```

Expect `sbp_4wk` (shuffling it adds 3.1 mmHg to MAE) and `sbp_week` (2.3) far ahead, as the two longest bars, then `overdue_x_meds` at about 0.29; every other column adds 0.04 or less. The categorical columns (cuff size, device) matter little because their effect on a patient's readings is already inside that patient's averages.

### 2. A nurse asks why a patient is on the list

Read the logistic regression's coefficients for the numeric columns with `named_steps` and `get_feature_names_out()`. They are scaled, so each coefficient is the change in log-odds per standard deviation.

```python
names = classifier.named_steps["preprocess"].get_feature_names_out()
coefficients = pd.Series(classifier.named_steps["model"].coef_[0], index=names)
numeric_coefficients = coefficients.filter(like="numeric__")
display(numeric_coefficients.sort_values(key=abs, ascending=False).head(8).round(2).rename("log-odds per SD"))
```

Expect the level columns first (`sbp_4wk` 1.46, `sbp_week` 1.24), then `overdue_x_meds` (0.41) and `weeks_since_change` (0.27). `drug_added_recently` is negative (-0.18): a patient whose new drug is still taking effect is less likely to cross 150. The reply to the nurse: high recent readings first, then lapsed refills on several drugs.

### 3. IT asks which columns the nightly feed could drop

Each column costs a join in the nightly job. Lasso's zero coefficients on the full data show which ones the model ignores.

```python
lasso_pipeline = fitted["lasso (alpha=0.05)"]
names = lasso_pipeline.named_steps["preprocess"].get_feature_names_out()
lasso_coef = pd.Series(lasso_pipeline.named_steps["model"].coef_, index=names).round(3)
print("kept:", (lasso_coef != 0).sum(), "of", len(lasso_coef))
print("dropped:", list(lasso_coef[lasso_coef == 0].index))
```

Expect 15 of 36 coefficients kept. The dropped ones include every category level and `n_symptom`: on 68,324 rows, Lasso at alpha 0.05 finds nothing in them that the level columns do not already carry. Dropping them leaves the Lasso forecast unchanged. Ridge, the frozen choice, never sets a coefficient to exactly 0, so changing the feed means refitting and rescoring it.

### 4. Quality asks for the standard per-class report

```python
from sklearn.metrics import classification_report

rank = pd.Series(alert_probability, index=watch_valid.index).groupby(watch_valid["week"]).rank(ascending=False)
report = classification_report(alert_valid, (rank <= 100).astype(int), output_dict=True)
display(pd.DataFrame(report).T.round(3))
```

Expect class `1` to show precision 0.473, recall 0.295, and support 1,443 (the validation patient-weeks that crossed 150).

### 5. The dashboard vendor's report shows a negative R² for the program-average tile

```python
from sklearn.metrics import r2_score

r2 = {name: r2_score(y_valid, model.predict(X_valid)) for name, model in fitted.items()}
r2["training mean"] = r2_score(y_valid, mean_model.predict(X_valid))
display(pd.Series(r2).round(3))
print("fit-row mean:", round(y_fit.mean(), 1), " validation mean:", round(y_valid.mean(), 1))
```

Expect R² of 0.796 for the linear models and -0.361 for the training mean: the fit rows average 145.6 mmHg and the validation rows 139.2, so predicting the fit rows' mean is worse than predicting the validation rows' own mean. The 6.4 mmHg drop is the program working (added drugs) plus summer.

### 6. The dashboard team wants predicted against actual

```python
plot_rows = valid.sample(3000, random_state=1)
predicted = fitted["ridge (alpha=10)"].predict(plot_rows[features])

fig, ax = plt.subplots(figsize=(5, 5))
ax.scatter(plot_rows["sbp_next_week"], predicted, alpha=0.2, s=8)
ax.plot([100, 200], [100, 200], color="gray", linestyle="--")
ax.set_xlabel("Actual next-week mean SBP (mmHg)")
ax.set_ylabel("Forecast (mmHg)")
plt.show()
plt.close(fig)
```

Expect the cloud to lie along the dashed line but slightly flatter than it: forecasts for very high and very low weeks are pulled toward the patient's longer average, because part of an extreme week is noise.

### 7. A reviewer asks why we did not use a random 80/20 split

`train_test_split` draws the same number of validation rows at random from the fit and validation weeks, so training sees weeks from every month. Compare its MAE with the time split's, and with the test weeks that neither split saw.

```python
from sklearn.model_selection import train_test_split

dev = pd.concat([fit_rows, valid])
random_fit, random_valid = train_test_split(dev, test_size=len(valid), random_state=42)
splits = {"by target week (step 3)": (fit_rows, valid), "random rows": (random_fit, random_valid)}
for name, (fit_part, valid_part) in splits.items():
    model = Pipeline([("preprocess", make_preprocess()), ("model", Ridge(alpha=10))])
    model.fit(fit_part[features], fit_part["sbp_next_week"])
    print(f"{name}: validation MAE {mean_absolute_error(valid_part['sbp_next_week'], model.predict(valid_part[features])):.2f}",
          f" test MAE {mean_absolute_error(test['sbp_next_week'], model.predict(test[features])):.2f}")
```

Expect a validation MAE of 3.72 for the time split and 3.59 for the random split, and a test MAE near 4.0 for both. The random split promises 0.4 mmHg more than the August and September weeks deliver; the time split comes closer. Random rows let training see later weeks of the same patients, which the Monday run never has. The reply to the reviewer: rows that follow each other in time get split in time.

## Optional extension: hand the frozen forecast to the Monday job

Prerequisite: BONUS.md, Model Serialization.

The Monday job runs in a separate session, so it needs the fitted pipeline as a file, with its feature list and versions beside it. Save it with `joblib`, load it back, and check that the loaded copy predicts the last test week exactly as the original does.

```python
import joblib
import sklearn

joblib.dump({"pipeline": final_forecast, "features": features,
             "trained_through": str(final_rows["target_week"].max().date()),
             "sklearn_version": sklearn.__version__}, "rpm_forecast.joblib")

package = joblib.load("rpm_forecast.joblib")
last_week = test[test["week"] == test["week"].max()]
loaded = package["pipeline"].predict(last_week[package["features"]])
print(len(last_week), "patients in the last test week; trained through", package["trained_through"])
print("loaded copy matches:", np.allclose(loaded, final_forecast.predict(last_week[features])))
```

Expect 2,685 patients in the last test week, `trained through 2025-07-28`, and `loaded copy matches: True`.
