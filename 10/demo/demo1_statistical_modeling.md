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

# Demo 1: BMI and Blood Pressure in a Health-System Extract

Two requests reach the analytics team of one health system:

- **Quality team (inference)**: among adults of the same age and sex, how much higher does systolic blood pressure (SBP) run per unit of BMI, and how sure are we? The data is a primary-care extract of about 520,000 visits by 160,000 patients.
- **Remote monitoring team (prediction setup)**: every Monday at 08:00 the program prints a call list. It wants a forecast of each patient's average SBP for the coming week. The data is 590,000 home cuff readings from 5,000 enrolled patients. This demo frames the problem and splits the rows; Demo 2 fits the models.

All patients and values are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas and pyarrow (for Parquet) in Colab (uv sync already did locally); if Colab asks, restart and rerun from the top
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0
```

## Core walkthrough

### 1. Load the primary-care extract

This supplied cell downloads `generate_data.py` when the folder lacks it (`urlretrieve(url, path)` saves the file at a web address to `path`), runs its `build()` when a table in `data/` is missing, and reads the visits table. `build()` writes four tables in under a minute with a fixed seed, so your values match the ones below.

```python
from pathlib import Path
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

REPO_RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main/10/demo"
TABLES = ["pcp_visits", "rpm_patients", "rpm_readings", "discharges"]
if not Path("generate_data.py").exists():
    urlretrieve(f"{REPO_RAW}/generate_data.py", "generate_data.py")
if not all(Path(f"data/{name}.parquet").exists() for name in TABLES):
    import generate_data
    generate_data.build()

visits = pd.read_parquet("data/pcp_visits.parquet")
print(visits.shape, "visits from", visits["patient_id"].nunique(), "patients")
display(visits.head())
display(visits[["age", "bmi", "sbp", "dbp"]].describe().round(1))
```

Expect `(522338, 15) visits from 160000 patients`, then five visits from January 1, 2024. The summary already shows problems: `bmi` has 486,015 values, not 522,338 (missing), its maximum is 484.0 (a dropped decimal point: 48.4 typed as 484), the maximum SBP is 1790 mmHg, and the minimum is 40 (SBP and DBP typed into each other's fields).

### 2. One clean row per patient

OLS standard errors assume independent rows. Repeat visits by the same patient are not independent, so the inference question uses one visit per patient: the first. Three cleaning steps come first:

- **Double-fed rows**: the interface sent some visits twice; the copies match on every column except `visit_id`.
- **Implausible SBP**: below 70 or above 260 mmHg, or not above DBP (swapped fields).
- **Implausible BMI**: above 70 kg/m². Missing BMI stays for now, to see what the model does with it.

```python
same_visit = [column for column in visits.columns if column != "visit_id"]
deduped = visits.drop_duplicates(subset=same_visit)

first = deduped.sort_values(["patient_id", "visit_date"]).drop_duplicates("patient_id")
bad_sbp = ~first["sbp"].between(70, 260) | (first["sbp"] <= first["dbp"])
bad_bmi = first["bmi"] > 70
cohort = first[~bad_sbp & ~bad_bmi]

display(pd.Series({
    "visits in extract": len(visits),
    "double-fed copies removed": len(visits) - len(deduped),
    "patients (first visit each)": len(first),
    "implausible SBP": bad_sbp.sum(),
    "implausible BMI": bad_bmi.sum(),
    "analysis cohort": len(cohort),
    "cohort rows missing BMI": cohort["bmi"].isna().sum(),
}, name="rows"))
```

Expect 2,599 double-fed copies, 160,000 first visits, 474 implausible SBP values and 167 implausible BMIs, a cohort of 159,360 patients, and 10,873 of them missing BMI.

### 3. Fit the model and check `nobs`

`C(sex)` makes sex categorical; `F` is the reference level, so the `C(sex)[T.M]` coefficient compares men with women of the same age and BMI.

```python
results = smf.ols("sbp ~ age + bmi + C(sex)", data=cohort).fit()
print("rows used:", int(results.nobs), "of", len(cohort))
print("R-squared:", round(results.rsquared, 3))

missing = cohort["bmi"].isna()
display(missing.groupby(cohort["clinic"]).mean().sort_values(ascending=False).round(3).rename("share_missing_bmi"))
print("share missing BMI, age 80 and over:", round(missing[cohort["age"] >= 80].mean(), 3),
      " under 80:", round(missing[cohort["age"] < 80].mean(), 3))
```

Expect `rows used: 148487 of 159360`: `statsmodels` silently dropped the 10,873 patients with no BMI. Expect R-squared 0.13: age, BMI, and sex explain a modest share of SBP, which also depends on medication, activity, and a patient's own baseline. The missing BMIs are not spread evenly: Marina is missing 19.7%, every other clinic 5.3% to 7.3%, and patients 80 and over 10.6% against 6.7% for younger ones. The estimates describe patients with a recorded BMI, so Marina's patients and the oldest patients count less than their share; report that beside the result.

### 4. Read the coefficients with their uncertainty

```python
coefficients = pd.DataFrame({
    "coef": results.params,
    "ci_lower": results.conf_int()[0],
    "ci_upper": results.conf_int()[1],
    "p_value": results.pvalues,
}).round(3)
display(coefficients)
```

Expect:

|  | coef | ci_lower | ci_upper | p_value |
| --- | --- | --- | --- | --- |
| Intercept | 82.793 | 82.292 | 83.293 | 0.0 |
| C(sex)[T.M] | 4.077 | 3.920 | 4.235 | 0.0 |
| age | 0.206 | 0.201 | 0.211 | 0.0 |
| bmi | 0.864 | 0.849 | 0.880 | 0.0 |

- **bmi, 0.86**: among patients of the same age and sex, each extra kg/m² goes with about 0.86 mmHg higher SBP (95% CI 0.85 to 0.88). Ten BMI units, say 25 against 35, is about 9 mmHg.
- **age, 0.21**: about 2.1 mmHg per decade, holding BMI and sex fixed. Older patients are treated more often (practice 2's treated patients are about 13 years older), and treatment lowers their readings, so this slope is flatter than the rise in untreated pressure with age.
- **p-values of 0.0**: with 148,487 rows, almost any nonzero association has a tiny p-value. The useful numbers are the size of the coefficient and the width of its interval.
- **Association**: these are observational records. The coefficient does not say that losing weight lowers SBP by 0.86 mmHg per unit; that needs causal evidence.

### 5. Two intervals for one new patient

The quality team asks what the model says about a 55-year-old woman with a BMI of 30.

```python
new_patient = pd.DataFrame({"age": [55], "bmi": [30.0], "sex": ["F"]})
display(results.get_prediction(new_patient).summary_frame(alpha=0.05).round(1))
```

Expect a fitted 120.0 mmHg, a mean-response interval of 119.9 to 120.1, and a prediction interval of 89.8 to 150.2. The 148,487 rows pin down the _average_ for patients like her to within a fraction of a mmHg, but _her_ SBP could sit anywhere in a 60 mmHg range. More rows shrink the first interval, not the second.

Draw both intervals for every BMI from 18 to 45 at age 55, over a sample of real women aged 50 to 60.

```python
grid = pd.DataFrame({"age": 55, "bmi": np.linspace(18, 45, 50), "sex": "F"})
bands = results.get_prediction(grid).summary_frame(alpha=0.05)
women = cohort[(cohort["sex"] == "F") & cohort["age"].between(50, 60)].dropna(subset=["bmi"])
women = women.sample(2000, random_state=1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(women["bmi"], women["sbp"], s=6, alpha=0.25, color="gray", label="women aged 50 to 60")
ax.fill_between(grid["bmi"], bands["obs_ci_lower"], bands["obs_ci_upper"], alpha=0.2, label="95% prediction interval")
ax.fill_between(grid["bmi"], bands["mean_ci_lower"], bands["mean_ci_upper"], color="red", alpha=0.6,
                label="95% mean-response interval")
ax.plot(grid["bmi"], bands["mean"], color="red")
ax.set(xlabel="BMI (kg/m²)", ylabel="SBP (mmHg)", title="Fitted SBP for a 55-year-old woman")
ax.legend(loc="upper left")
plt.show()
plt.close(fig)
```

Expect a red line rising from about 110 mmHg at BMI 18 to 133 at BMI 45. The red mean-response band is so thin it hides under the line. The blue prediction band is about 60 mmHg tall and covers most of the gray dots: the model knows the average well and each woman poorly.

### 6. Check the residuals

A scatter of 148,000 points is a solid blob, so plot a random 5,000 of them, then check the pattern in numbers on every row: `pd.cut` (Lecture 05) groups the fitted values into bands.

```python
diagnostics = pd.DataFrame({"fitted": results.fittedvalues, "residual": results.resid})
sample = diagnostics.sample(5000, random_state=1)

fig, ax = plt.subplots(figsize=(7, 4))
ax.scatter(sample["fitted"], sample["residual"], alpha=0.25, s=8)
ax.axhline(0, color="gray", linestyle="--")
ax.set_xlabel("Fitted SBP (mmHg)")
ax.set_ylabel("Residual (observed - fitted)")
ax.set_title("Residuals against fitted SBP")
fig.savefig("residuals_vs_fitted.png", dpi=150, bbox_inches="tight")
plt.show()
plt.close(fig)

bands = pd.cut(diagnostics["fitted"], [0, 110, 120, 130, 140, 200])
display(diagnostics.groupby(bands, observed=True)["residual"].agg(["mean", "std", "size"]).round(2))
```

Expect a cloud centered on the dashed zero line, a little wider on the right, and a saved `residuals_vs_fitted.png`. Among 5,000 points a widening (a **funnel**) is hard to judge by eye, so the table puts numbers on it: the bands' mean residuals stay within 1 mmHg of 0, but the standard deviation grows from 13.1 mmHg in the lowest band to about 17 in the top two. The spread is not constant (older patients' readings vary more from visit to visit), so the default standard errors and intervals above are approximate. The prediction interval in step 5 is too wide for young patients and too narrow for old ones.

### 7. Frame the forecast: one row per patient-week

Switch to the remote monitoring program. Each patient's cuff takes readings at home and uploads them when the phone syncs, usually within minutes, sometimes days later. Each reading is marked `scheduled` (the daily routine) or `symptoms` (taken because the patient felt unwell); the program judges control on scheduled readings only.

- **Prediction unit**: one patient-week. **Prediction time**: Monday 08:00 after the week ends, when the call list runs. A reading uploaded after that misses the run.
- **Target**: `sbp_next_week`, the mean of the patient's scheduled readings in the following Monday-to-Sunday week. Its **target time** is that week.
- **Label arrival**: the latest `upload_time` among those readings, when the label is complete in the database.

The raw readings need cleaning first: re-synced copies of a reading (same patient and `reading_time`, later `upload_time`) and cuff error codes (SBP 0, 300, 310, 999). Subtracting `.dt.dayofweek` days (Monday is 0) from each reading's date gives the Monday that starts its week.

```python
readings = pd.read_parquet("data/rpm_readings.parquet")
enrolled = pd.read_parquet("data/rpm_patients.parquet")
print("raw readings:", len(readings))

readings = readings.sort_values("upload_time").drop_duplicates(["patient_id", "reading_time"])
readings = readings[readings["sbp"].between(60, 260)].copy()
print("clean readings:", len(readings))
display(readings.groupby("reason")["sbp"].agg(["size", "mean"]).round(1))

readings["week"] = readings["reading_time"].dt.floor("D") - readings["reading_time"].dt.dayofweek * pd.Timedelta(days=1)
readings["run_time"] = readings["week"] + pd.Timedelta(days=7, hours=8)
scheduled = readings[readings["reason"] == "scheduled"]
on_time = scheduled[scheduled["upload_time"] <= scheduled["run_time"]]

seen_by_run = on_time.groupby(["patient_id", "week"]).agg(sbp_week=("sbp", "mean"), n_readings=("sbp", "size"))
complete = scheduled.groupby(["patient_id", "week"]).agg(sbp_complete=("sbp", "mean"),
                                                         last_upload=("upload_time", "max"))
weekly = seen_by_run.join(complete, how="outer").reset_index()

by_patient = weekly.groupby("patient_id")
weekly["sbp_next_week"] = by_patient["sbp_complete"].shift(-1)
weekly["target_week"] = by_patient["week"].shift(-1)
weekly["label_arrival"] = by_patient["last_upload"].shift(-1)
weekly["run_time"] = weekly["week"] + pd.Timedelta(days=7, hours=8)
consecutive = weekly["target_week"] - weekly["week"] == pd.Timedelta(days=7)
weekly = weekly[consecutive & weekly["sbp_week"].notna()].drop(columns=["sbp_complete", "last_upload"])
print("patient-weeks with a target:", len(weekly))
display(weekly.head(3))
```

Expect 587,139 raw readings and 580,698 clean ones: 552,512 scheduled (mean 142.6 mmHg) and 28,186 symptom readings (mean 156.2, taken when the patient felt unwell, so they run high). The weekly table has 140,399 patient-weeks with a target. In the first rows, patient RP00001's `sbp_next_week` is the next row's `sbp_week` (147.2, then 151.7), and each label arrives when the target week's last reading uploads (April 6, 13, and 18).

Plot one patient's first six weeks: every scheduled reading as a dot, each week's mean as a black line across that week, and the target the row carries as a red dashed line across the following week. `ax.hlines(y, xmin, xmax)` draws one horizontal segment per value.

```python
patient = scheduled[scheduled["patient_id"] == "RP00001"]
patient = patient[patient["week"] < patient["week"].min() + pd.Timedelta(weeks=6)]
patient_weeks = weekly[weekly["patient_id"] == "RP00001"].head(5)

fig, ax = plt.subplots(figsize=(8, 3.5))
ax.scatter(patient["reading_time"], patient["sbp"], s=18, alpha=0.6, label="scheduled reading")
ax.hlines(patient_weeks["sbp_week"], patient_weeks["week"], patient_weeks["week"] + pd.Timedelta(days=7),
          color="black", linewidth=2, label="sbp_week (feature)")
ax.hlines(patient_weeks["sbp_next_week"], patient_weeks["target_week"],
          patient_weeks["target_week"] + pd.Timedelta(days=7), color="red", linestyle="--",
          label="sbp_next_week (target)")
ax.set(ylabel="SBP (mmHg)", title="RP00001: each week's mean is the previous row's target")
ax.legend(fontsize=8, loc="upper right")
plt.show()
plt.close(fig)
```

Expect dots scattered up to about 10 mmHg either side of each black line. Each red dashed target sits exactly on the next week's black line: the row for one week carries the following week's mean as its label.

### 8. Audit the features and put reading times on a clock face

Every feature must be known at Monday 08:00. The candidates include two that a dashboard offers but the forecast cannot have:

```python
audit = pd.DataFrame({
    "feature": ["age", "bmi at enrollment", "this week's mean SBP", "mean reading time this week",
                "this week's mean with late uploads", "pharmacist note from next week's review"],
    "becomes_known": ["enrollment", "enrollment", "Monday 08:00", "Monday 08:00",
                      "when the last phone syncs, up to 6 days later", "during the target week"],
    "hours_after_prediction": [0, 0, 0, 0, 144, 96],
})
audit["decision"] = np.where(audit["hours_after_prediction"] <= 0, "keep", "exclude (leakage)")
display(audit)
```

Expect four `keep` rows and two `exclude (leakage)` rows. This week's mean _with_ late uploads looks like the same feature, but at 08:00 on Monday some of those readings are still on a phone.

Most patients measure in the morning, some in the evening, and night-shift workers around midnight, and morning readings run higher. A week's mean reading time is a useful feature, but a plain mean of hours fails at midnight: readings at 23:00 and 01:00 average to 12:00. Averaging the sine and cosine of each hour instead keeps the mean on the clock face: (0, 1) is midnight, (1, 0) 06:00, (0, -1) noon, and (-1, 0) 18:00.

```python
scheduled = scheduled.assign(hour=scheduled["reading_time"].dt.hour + scheduled["reading_time"].dt.minute / 60)
scheduled["hour_sin"] = np.sin(2 * np.pi * scheduled["hour"] / 24)
scheduled["hour_cos"] = np.cos(2 * np.pi * scheduled["hour"] / 24)

display(scheduled.groupby(scheduled["reading_time"].dt.hour)["sbp"].mean().loc[[0, 3, 7, 12, 19, 22]]
        .round(1).rename("mean_sbp"))

straddle = scheduled.groupby("patient_id")["hour"].agg(lambda hours: min((hours >= 23).mean(), (hours < 1).mean()))
night_patient = straddle.idxmax()
night = scheduled[scheduled["patient_id"] == night_patient]
print(night_patient, "plain mean hour:", round(night["hour"].mean(), 1),
      " mean hour_sin:", round(night["hour_sin"].mean(), 2), " mean hour_cos:", round(night["hour_cos"].mean(), 2))
```

Expect the daily rhythm in the means: 145.8 mmHg at 7:00, 136.2 at 19:00, and 138.4 at midnight. For RP04850, who measures around midnight, the plain mean hour is 12.3, a time the patient never measures; the mean sine and cosine are -0.05 and 0.95, just before midnight on the clock face.

Put RP04850's readings on the clock face beside the daily rhythm.

```python
fig, (left, right) = plt.subplots(1, 2, figsize=(10, 4))
scheduled.groupby(scheduled["reading_time"].dt.hour)["sbp"].mean().plot(ax=left, marker="o")
left.set(xlabel="Hour of reading", ylabel="Mean SBP (mmHg)", title="Morning readings run highest")

angle = np.linspace(0, 2 * np.pi, 100)
right.plot(np.sin(angle), np.cos(angle), color="lightgray")
right.scatter(night["hour_sin"], night["hour_cos"], s=12, alpha=0.4, label="RP04850's readings")
right.scatter(night["hour_sin"].mean(), night["hour_cos"].mean(), s=120, marker="*", color="red",
              label="mean of sin and cos")
plain = night["hour"].mean()
right.scatter(np.sin(2 * np.pi * plain / 24), np.cos(2 * np.pi * plain / 24), s=120, marker="X", color="black",
              label=f"plain mean hour ({plain:.1f})")
for label, hour in [("00:00", 0), ("06:00", 6), ("12:00", 12), ("18:00", 18)]:
    right.annotate(label, (1.15 * np.sin(2 * np.pi * hour / 24), 1.15 * np.cos(2 * np.pi * hour / 24)), ha="center")
right.set(xlim=(-1.4, 1.4), ylim=(-1.4, 1.4), aspect="equal", xlabel="hour_sin", ylabel="hour_cos",
          title="One night-shift patient on the clock face")
right.legend(fontsize=8, loc="center")
plt.show()
plt.close(fig)
```

Expect the left curve to peak near 146 mmHg at 07:00 and bottom out near 136 at 20:00. On the right, RP04850's dots cluster at the top of the circle around 00:00. The red star (mean sine and cosine) sits with them; the black X (plain mean hour, 12.3) sits at the bottom near 12:00, where the patient never measures.

### 9. Split on target time, then keep only labels that had arrived

Target weeks starting before June 1 train, June and July validate, and August and September test. A prospective evaluation also requires every training label to have arrived by the first validation run, the earliest `run_time` among the validation rows.

```python
train = weekly[weekly["target_week"] < "2025-06-01"]
valid = weekly[(weekly["target_week"] >= "2025-06-01") & (weekly["target_week"] < "2025-08-01")]
test = weekly[weekly["target_week"] >= "2025-08-01"]

first_valid_run = valid["run_time"].min()
fit_rows = train[train["label_arrival"] <= first_valid_run]
assert fit_rows["label_arrival"].max() <= first_valid_run

display(pd.DataFrame({
    "rows": [len(train), len(fit_rows), len(valid), len(test)],
    "first_target_week": [part["target_week"].min().date() for part in [train, fit_rows, valid, test]],
    "last_target_week": [part["target_week"].max().date() for part in [train, fit_rows, valid, test]],
}, index=["train", "fit_rows (labels arrived)", "valid", "test"]))
print("first validation run:", first_valid_run)
```

Count rows per target week and color each by its role.

```python
roles = pd.concat([fit_rows.assign(role="fit (label arrived)"),
                   train[train["label_arrival"] > first_valid_run].assign(role="train, label late"),
                   valid.assign(role="validation"), test.assign(role="test")])
counts = roles.groupby(["target_week", "role"]).size().unstack(fill_value=0)
counts = counts[["fit (label arrived)", "train, label late", "validation", "test"]]
counts.index = counts.index.strftime("%b %d")

ax = counts.plot.bar(stacked=True, width=1.0, figsize=(10, 3.5), color=["C0", "C3", "C1", "C2"])
ax.set_xticks(range(0, len(counts), 4), counts.index[::4], rotation=0)
ax.set(xlabel="Target week", ylabel="Patient-weeks", title="Roles split on target week", ylim=(0, 6000))
ax.legend(ncols=4, loc="upper center")
plt.show()
plt.close(ax.figure)
```

Expect blue training bars growing from January as patients enroll, a thin red cap on the week of May 26 (the 310 late labels), then orange validation weeks in June and July and green test weeks from August. No target week has two colors except May 26.

Expect 68,634 training, 37,980 validation, and 33,785 test rows with non-overlapping target weeks. The first validation run is Monday, June 2, 2025, at 08:00, when the forecast for the week of June 2 is made. 68,324 training labels had arrived by then; the 310 left out are from the last training week (May 26), whose phones synced after the run. Demo 2 fits its models on `fit_rows`.

## Independent practice

### 1. Count every visit and the intervals shrink for the wrong reason

Fit the same formula to every clean visit instead of one per patient, and compare the standard errors.

```python
all_visits = deduped[deduped["sbp"].between(70, 260) & (deduped["sbp"] > deduped["dbp"]) & ~(deduped["bmi"] > 70)]
every_visit = smf.ols("sbp ~ age + bmi + C(sex)", data=all_visits).fit()
display(pd.DataFrame({"one_per_patient": results.bse, "every_visit": every_visit.bse}).round(4))
print("rows used:", int(every_visit.nobs))
```

Expect 481,614 rows and standard errors about 45% smaller (BMI's falls from 0.0078 to 0.0043). The data did not get more informative: each patient's visits share that patient's usual blood pressure, and the model counts them as unrelated people.

### 2. Does blood-pressure medication "raise" SBP?

Add `C(on_bp_med)` to the step 3 formula and read its coefficient beside the raw difference between treated and untreated patients.

```python
with_med = smf.ols("sbp ~ age + bmi + C(sex) + C(on_bp_med)", data=cohort).fit()
display(with_med.params.round(2).rename("coef"))
display(cohort.groupby("on_bp_med")[["sbp", "age", "bmi"]].mean().round(1))
```

Expect `C(on_bp_med)[T.True]` near +1.1 mmHg: treated patients read _higher_ than untreated patients of the same age, BMI, and sex. The raw means show why: treated patients average 124.1 mmHg against 117.7, and they are 13 years older and 3 BMI units heavier. Medication does not raise blood pressure; clinicians prescribe it to the patients whose pressure was high (**confounding by indication**). An observational coefficient describes who gets treated as much as what treatment does.

### 3. Clinics as a categorical predictor, and the array API

`C(clinic)` gives every clinic but the reference (Bayview, first alphabetically) its own coefficient. Then reproduce step 3's fit with the array API, which needs the missing-BMI rows dropped and the dummy columns built by hand.

```python
by_clinic = smf.ols("sbp ~ age + bmi + C(sex) + C(clinic)", data=cohort).fit()
display(by_clinic.params.filter(like="clinic").round(2).rename("vs Bayview"))
display(cohort.groupby("clinic")["adi_decile"].mean().round(1).rename("mean ADI decile"))

complete = cohort.dropna(subset=["bmi"])
X = sm.add_constant(pd.DataFrame({"age": complete["age"], "bmi": complete["bmi"],
                                  "male": (complete["sex"] == "M").astype(int)}))
array_fit = sm.OLS(complete["sbp"], X).fit()
display(array_fit.params.round(3).rename("coef"))
```

Expect all five clinic coefficients below 0, from -3.5 (Excelsior) to -10.2 (Marina): with age, BMI, and sex fixed, Bayview still reads highest. The ADI deciles point to one candidate reason: Bayview's patients come from the most deprived neighborhoods (mean decile 7.9, Marina 2.3). The coefficients say the clinics differ, not why; staff, cuffs, and neighborhoods all vary by clinic. The array fit gives the same numbers as step 4 (`const` 82.793, `age` 0.206, `bmi` 0.864, `male` 4.077).

### 4. A curve in age, and an interaction

Add `I(age ** 2)` and compare the fits. Then let BMI's slope depend on sex with `bmi * C(sex)`.

```python
curved = smf.ols("sbp ~ age + I(age ** 2) + bmi + C(sex)", data=cohort).fit()
print("R-squared:", round(results.rsquared, 4), "->", round(curved.rsquared, 4))
print("AIC:", round(results.aic), "->", round(curved.aic))
ages = pd.DataFrame({"age": [30, 50, 70, 85], "bmi": 28.0, "sex": "F"})
display(pd.DataFrame({"age": ages["age"], "straight": results.predict(ages),
                      "curved": curved.predict(ages)}).round(1))

interaction = smf.ols("sbp ~ age + bmi * C(sex)", data=cohort).fit()
display(interaction.params.round(3).rename("coef"))
```

Expect R-squared to rise only slightly (0.1299 to 0.1349) and AIC to fall (1233455 to 1232603). The curved model predicts higher SBP than the straight line at 30 and 85 and lower at 50: the straight line understates an 85-year-old by about 3.4 mmHg. In the interaction model, `bmi:C(sex)[T.M]` is -0.041: women's SBP rises 0.883 mmHg per BMI unit and men's 0.883 - 0.041 = 0.842, nearly the same slope.

### 5. Fit a first forecast on the arrived labels and check its intervals

Fit an OLS forecast on `fit_rows` and ask how often the validation weeks land inside their nominal 95% prediction intervals.

```python
fit_rows = fit_rows.merge(enrolled[["patient_id", "age"]], on="patient_id", how="left", validate="many_to_one")
valid = valid.merge(enrolled[["patient_id", "age"]], on="patient_id", how="left", validate="many_to_one")
forecast = smf.ols("sbp_next_week ~ sbp_week + age", data=fit_rows).fit()
display(forecast.params.round(3).rename("coef"))

bounds = forecast.get_prediction(valid).summary_frame(alpha=0.05)
inside = valid["sbp_next_week"].between(bounds["obs_ci_lower"], bounds["obs_ci_upper"])
print("validation weeks inside their 95% interval:", round(inside.mean(), 3))
print("average miss (mmHg):", round((valid["sbp_next_week"] - bounds["mean"]).abs().mean(), 2))
print("average miss of this week's mean as the forecast:", round((valid["sbp_next_week"] - valid["sbp_week"]).abs().mean(), 2))
```

Expect a `sbp_week` coefficient near 0.90: next week's mean moves most of the way toward this week's. About 93% of validation weeks fall inside their intervals, a little short of the nominal 95%, and the average miss is 4.15 mmHg, against 4.28 for simply repeating this week's mean. Demo 2 measures these misses properly and does better.

### 6. Retrospective against prospective

How many training labels arrived after the first validation run, and why?

```python
late = train[train["label_arrival"] > first_valid_run]
print("late labels:", len(late))
display(late["target_week"].value_counts().rename("late labels"))
delay = late["label_arrival"] - first_valid_run
print("median delay after the run:", delay.median().round("h"), " longest:", delay.max().round("h"))
```

Expect 310 late labels, all from the week of May 26, with a median delay of 1 day 11 hours after the June 2 run and the longest about 5.4 days. A **retrospective** split would fit all 68,634 training rows, including labels that did not exist yet when the first validation forecast was made.

## Optional extension: a random intercept per patient

Prerequisite: BONUS.md, Mixed Effects Models.

Practice 1 fit every visit as if each were a different person. A mixed model gives each patient their own intercept, so repeat visits share it. On every clean visit at the Marina clinic:

```python
marina = all_visits[all_visits["clinic"] == "Marina"].dropna(subset=["bmi"])
mixed = smf.mixedlm("sbp ~ age + bmi + C(sex)", data=marina, groups=marina["patient_id"]).fit()
naive = smf.ols("sbp ~ age + bmi + C(sex)", data=marina).fit()
print("visits:", len(marina), "patients:", marina["patient_id"].nunique())
display(pd.DataFrame({"ols_se": naive.bse, "mixed_se": mixed.bse}).dropna().round(4))
```

Expect 24,877 visits from 9,388 patients, and the mixed model's standard errors about 27% larger than plain OLS on the same rows (BMI's 0.0241 against 0.0190): it treats a patient's repeat visits as one patient seen several times.
