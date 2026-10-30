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

# Demo 3: Which Discharged Patients Should the Transitional-Care Nurses Call?

Cedar Valley Health's care-management director, Dr. Leah Kim, takes three pieces of work to the November 18, 2025, operations committee. Demos 1 and 2 built the remote blood-pressure program's forecast and its Monday call list. This demo builds the last piece: the post-discharge call list.

**This phase's ask.** The transitional-care nurses can call about 10% of discharged patients in their first two days at home. Today they call the patients with the most admissions in the past year. By November 11, Dr. Kim needs to know how many more patients readmitted within 30 days a model's list would reach for the same calls, and whether a model the committee can read line by line is enough, or a more complex one reaches enough extra patients to be worth it. The committee decides whether the list replaces today's rule. The data is 235,000 discharges from four hospitals, January 2023 to June 2025, with many patients admitted more than once. Each row carries what the discharge summary and the EHR show at discharge: diagnosis, comorbidity count, labs, prior admissions and ED visits, medications, insurance, and two social flags from the case manager's screen (lives alone, unstable housing). All patients and values are synthetic.

Run the cells from top to bottom.

```python
# Installs the course's pandas, pyarrow (for Parquet), and XGBoost in Colab; if Colab asks, restart and rerun from the top. Intel Mac: run this demo in Colab (no local TensorFlow). Mac XGBoost load error: brew install libomp
%pip install -q --no-warn-conflicts pandas==3.0.5 pyarrow==25.0.0 xgboost==2.1.4
```

## Core walkthrough

### 1. Who counts as readmitted?

The extract has one row per discharge but no readmission column; the label comes from each patient's **next** admission, through a grouped `shift(-1)` as in Demo 1's target. Two groups of rows cannot carry an honest label:

- `readmit_30d` is 1 when the patient's next admission starts within 30 days of this discharge.
- **In-hospital deaths** (`disposition == "expired"`) cannot be readmitted, so they leave the table.
- The label is known 30 days after discharge (its **label time**). The extract ends on June 30, 2025, so discharges after May 31 have incomplete follow-up and leave the table too; keeping them would count unseen readmissions as 0.

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

discharges = pd.read_parquet("data/discharges.parquet").sort_values(["patient_id", "admit_date"])
print(discharges.shape, "discharges from", discharges["patient_id"].nunique(), "patients")

next_admit = discharges.groupby("patient_id")["admit_date"].shift(-1)
discharges["readmit_30d"] = ((next_admit - discharges["discharge_date"]).dt.days <= 30).astype(int)
discharges["label_time"] = discharges["discharge_date"] + pd.Timedelta(days=30)

extract_end = pd.Timestamp("2025-06-30")
cohort = discharges[(discharges["disposition"] != "expired") & (discharges["label_time"] <= extract_end)]
cohort = cohort.sort_values("discharge_date").reset_index(drop=True)
print("eligible discharges:", len(cohort), " readmission rate:", round(cohort["readmit_30d"].mean(), 3))

rates = cohort.groupby("dx_group")["readmit_30d"].mean().sort_values()
ax = rates.plot.barh(figsize=(7, 3.5))
ax.bar_label(ax.containers[0], fmt="%.3f", padding=3)
ax.axvline(cohort["readmit_30d"].mean(), color="gray", linestyle="--")
ax.set(xlabel="30-day readmission rate (dashed: all discharges)", ylabel="", xlim=(0, 0.4))
plt.show()
plt.close(ax.figure)
```

Expect `(235391, 24) discharges from 100799 patients`, then 224,256 eligible discharges with a readmission rate of 0.158. The bars show rates differ by diagnosis: 0.056 after orthopedic stays (mostly planned surgery), 0.322 after heart failure, with only heart failure, COPD, GI bleed, and sepsis right of the dashed overall rate.

### 2. Which details exist when the nurse calls, and which months can train?

The calls happen in the first two days home, so every feature must be known at discharge. `followup_completed` is not: it records whether the patient made the follow-up appointment 7 to 14 days later. Practice 4 shows what it does to the scores.

The split runs on discharge date, and the prospective rule from Demo 1 applies: training rows must have their label (30 days after discharge) by the first validation discharge, July 1, 2024.

```python
numeric = ["age", "los_days", "n_prior_admits_12m", "ed_visits_6m", "n_comorbidities", "n_meds",
           "sodium", "hemoglobin", "creatinine", "adi_decile"]
categorical = ["hospital", "sex", "insurance", "lives_alone", "housing_unstable", "admit_source",
               "dx_group", "disposition", "followup_scheduled"]
features = numeric + categorical

train = cohort[cohort["discharge_date"] < "2024-07-01"]
valid = cohort[(cohort["discharge_date"] >= "2024-07-01") & (cohort["discharge_date"] < "2025-01-01")]
test = cohort[cohort["discharge_date"] >= "2025-01-01"]
fit_rows = train[train["label_time"] <= valid["discharge_date"].min()]
assert fit_rows["label_time"].max() <= valid["discharge_date"].min()

X_fit, y_fit = fit_rows[features], fit_rows["readmit_30d"]
X_valid, y_valid = valid[features], valid["readmit_30d"]
print("train:", len(train), " fit rows:", len(fit_rows), " valid:", len(valid), " test:", len(test))
display(X_fit[numeric].isna().mean().round(3).rename("share_missing").loc[lambda share: share > 0])

roles = pd.concat([fit_rows.assign(role="fit (label known)"),
                   train[train["label_time"] > valid["discharge_date"].min()].assign(role="train, label late"),
                   valid.assign(role="validation"), test.assign(role="test")])
counts = roles.groupby([roles["discharge_date"].dt.strftime("%Y-%m"), "role"]).size().unstack(fill_value=0)
counts = counts[["fit (label known)", "train, label late", "validation", "test"]]
ax = counts.plot.bar(stacked=True, width=1.0, figsize=(10, 3.5), color=["C0", "C3", "C1", "C2"])
ax.set_xticks(range(0, len(counts), 3), counts.index[::3], rotation=0)
ax.set(xlabel="Discharge month", ylabel="Discharges", ylim=(0, 11000),
       title="Split on discharge date; June 2024 labels arrive too late")
ax.legend(ncols=4, loc="upper center")
plt.show()
plt.close(ax.figure)
```

Expect 139,465 training discharges, of which 131,927 have their label by July 1 (the 7,538 left out were discharged in June 2024), 47,219 validation and 37,572 test discharges. Three lab columns have gaps of about 5% to 9%: labs not drawn, more often before planned surgery and for short stays. The chart shows about 6,500 to 8,100 discharges a month from January 2023 to June 2025, blue through May 2024, one red month (June 2024, labels due after July 1), six orange validation months, and five green test months.

### 3. What do a random list and today's rule catch?

Every model gets the same prepared columns: median-filled and scaled numbers, one-hot categories. The comparison rule is fixed before any model is fit, and it follows the nurses' budget:

- Flag the 10% of validation discharges with the highest predicted probability (the 90th percentile of each model's probabilities is its threshold).
- Compare **recall** (share of readmitted patients who got a call), **precision** (share of calls that reached one), and the count of readmitted patients called.

Accuracy is in the table too, at the default 0.5 threshold, to show why it cannot choose. The first row is a random call list: every probability below 0.5, so it predicts "no readmission" for everyone, in random order.

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def make_preprocess(categorical_columns=categorical):
    numeric_steps = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    return ColumnTransformer([("numeric", numeric_steps, numeric),
                              ("categorical", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                               categorical_columns)])

def compare(name, probability, truth=y_valid):
    flag = (probability >= pd.Series(probability).quantile(0.9)).astype(int)
    comparison[name] = {"accuracy_at_0.5": accuracy_score(truth, probability >= 0.5),
                        "recall_at_10pct": recall_score(truth, flag),
                        "precision_at_10pct": precision_score(truth, flag),
                        "readmitted_called": int((flag & truth).sum())}

def show(rows=None):
    table = pd.DataFrame(comparison).T.astype({"readmitted_called": int}).round(3)
    display(table if rows is None else table.loc[rows])

comparison = {}
no_model = np.random.default_rng(0).uniform(0, 0.01, len(y_valid))
compare("no model (random call list)", no_model)
print("validation readmissions:", y_valid.sum())
show()
```

Expect 7,659 validation readmissions, accuracy 0.838 with nobody flagged, and a random call list's recall of 0.102: random calls reach about 10% of readmitted patients (780). 10% of 47,219 validation discharges is about 4,720 calls.

Today's rule ranks discharges by admissions in the past year, then by comorbidity count. Many patients tie, so `rank(method="first")` breaks the remaining ties by row order and the list stays at exactly 10%. A rule gives no probability, so its accuracy cell stays empty (`NaN`).

```python
def todays_rule(name, rows):
    score = rows["n_prior_admits_12m"] * 100 + rows["n_comorbidities"]
    flag = (score.rank(ascending=False, method="first") <= 0.1 * len(rows)).astype(int)
    truth = rows["readmit_30d"]
    comparison[name] = {"accuracy_at_0.5": np.nan, "recall_at_10pct": recall_score(truth, flag),
                        "precision_at_10pct": precision_score(truth, flag),
                        "readmitted_called": int((flag & truth).sum())}

todays_rule("today's rule: most prior admissions", valid)
show()
```

Expect today's rule at a recall of 0.201 and precision of 0.327: 1,542 readmitted patients called, about twice the random list. Every model below has to beat 1,542.

### 4. A list the committee can read line by line

Logistic regression is the simple candidate: one slope per prepared column, and the committee can read its coefficients.

```python
from sklearn.linear_model import LogisticRegression

logistic = Pipeline([("preprocess", make_preprocess()), ("model", LogisticRegression(max_iter=1000))])
logistic.fit(X_fit, y_fit)
compare("logistic regression", logistic.predict_proba(X_valid)[:, 1])
show()
```

Expect accuracy at 0.5 of 0.842, barely above the no-model 0.838, but at the 10% budget a recall of 0.290 and precision of 0.471: 2,224 of the 7,659 readmitted patients get a call, 682 more than today's rule and almost three times the random list's 780.

### 5. Do combinations of risk factors reach more patients?

A random forest can split on combinations of columns that logistic regression gives one slope each. This one has 200 trees, each stopping at 50 rows per leaf (`min_samples_leaf`) so it cannot memorize single discharges; `n_jobs=-1` builds trees on every core. On a two-core Colab runtime this takes under a minute.

```python
from sklearn.ensemble import RandomForestClassifier

forest = Pipeline([("preprocess", make_preprocess()),
                   ("model", RandomForestClassifier(n_estimators=200, min_samples_leaf=50,
                                                    n_jobs=-1, random_state=42))])
forest.fit(X_fit, y_fit)
compare("random forest", forest.predict_proba(X_valid)[:, 1])

names = forest.named_steps["preprocess"].get_feature_names_out()
impurity = pd.Series(forest.named_steps["model"].feature_importances_, index=names)
ax = impurity.sort_values().tail(6).plot.barh(figsize=(7, 3.5))
ax.set(xlabel="Impurity importance (training fit)", title="What the forest's splits used most")
plt.show()
plt.close(ax.figure)
show()
```

Expect a recall of 0.303, a little above logistic regression (2,319 readmitted patients called against 2,224). In the chart, comorbidity count leads the impurity importances (about 0.16), then medication count, hemoglobin, heart failure, age, and creatinine. (`get_feature_names_out()` names the prepared columns; one-hot columns appear as `categorical__dx_group_heart_failure` and so on.) Impurity importance is measured on the training fit and favors columns with many distinct values; practice 1 compares it with permutation importance.

### 6. How many more can the same calls reach, and when does adding trees stop helping?

XGBoost adds small trees one at a time, each aimed at what the trees so far get wrong, so it needs a stopping point. It takes NumPy arrays here, so the preparation step is fit once and applied to both parts. Early stopping watches validation log loss and stops 30 rounds after the best one.

```python
import xgboost as xgb

preprocess = make_preprocess().fit(X_fit)
X_fit_ready = preprocess.transform(X_fit)
X_valid_ready = preprocess.transform(X_valid)

booster = xgb.XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.05,
                            subsample=0.8, colsample_bytree=0.8,
                            early_stopping_rounds=30, random_state=42, n_jobs=-1)
booster.fit(X_fit_ready, y_fit, eval_set=[(X_valid_ready, y_valid)], verbose=False)
print("best round:", booster.best_iteration, " validation log loss:", round(booster.best_score, 4))
compare("XGBoost", booster.predict_proba(X_valid_ready)[:, 1])
show()

curve = booster.evals_result()["validation_0"]["logloss"]
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.plot(curve)
ax.axvline(booster.best_iteration, color="red", linestyle="--", label=f"best round {booster.best_iteration}")
ax.set(xlabel="Boosting round", ylabel="Validation log loss", ylim=(0.375, 0.42),
       title="Early stopping ends 30 rounds after the best")
ax.legend()
plt.show()
plt.close(fig)
```

Expect best round 284 of a 1,000-round budget (validation log loss 0.3793), with training stopped 30 rounds later. The curve drops steeply over the first 50 rounds, flattens near 0.380 after about 150, and ends at round 314 just past the red line. Recall is 0.308, the best so far: 2,362 readmitted patients called, 138 more than logistic regression. Validation picked the number of rounds, so it is no longer an untouched estimate; the test set stays sealed for that.

### 7. The CMIO asks whether "AI" would find more

The CMIO's "AI" is a neural network. It reads the same prepared arrays. `set_random_seed` and `enable_op_determinism` make the run repeat on one machine; another machine can differ in the third decimal. `evaluate()` reports the validation loss and accuracy the network was compiled with.

```python
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
from tensorflow import keras

keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

network = keras.Sequential([
    keras.layers.Input(shape=(X_fit_ready.shape[1],)),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(16, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid"),
])
network.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
history = network.fit(X_fit_ready, y_fit.to_numpy(), epochs=15, batch_size=256,
                      validation_data=(X_valid_ready, y_valid.to_numpy()), verbose=0)
print("weights:", network.count_params())
compare("neural network", network.predict(X_valid_ready, verbose=0).ravel())
loss, accuracy = network.evaluate(X_valid_ready, y_valid.to_numpy(), verbose=0)
print("validation log loss:", round(loss, 4), " accuracy at 0.5:", round(accuracy, 3))

fig, ax = plt.subplots(figsize=(6, 3.5))
ax.plot(range(1, 16), history.history["loss"], "o-", label="training")
ax.plot(range(1, 16), history.history["val_loss"], "s--", label="validation")
ax.set(xlabel="Epoch", ylabel="Log loss", title="Validation loss flattens after a few epochs")
ax.legend()
plt.show()
plt.close(fig)
```

Expect a training loss that falls quickly over the first two epochs and keeps creeping down, and a validation loss that eases from about 0.387 to 0.383 by the fourth epoch and then stays flat, with no late rise: 131,927 rows give a network of 1,857 weights little room to memorize. `evaluate()` gives validation log loss 0.3829 and accuracy 0.844 at 0.5. Its recall at the budget is 0.300, between logistic regression and the forest. Training takes well under a minute.

### 8. Which list goes to the nurses?

```python
show()
called = pd.DataFrame(comparison).T["readmitted_called"].sort_values()
ax = called.plot.barh(figsize=(7, 3.5))
ax.bar_label(ax.containers[0], padding=3)
ax.set(xlabel=f"Readmitted patients called, of {y_valid.sum():,} (10% call budget)", xlim=(0, 3000))
plt.show()
plt.close(ax.figure)
```

Expect, at the 10% budget, recall from 0.290 (logistic regression) to 0.308 (XGBoost), with the network (0.300) and the forest (0.303) between, and today's rule at 0.201. Every model's accuracy at 0.5 sits between 0.842 and 0.845. The bar chart puts the four families within 140 calls of each other (2,224 to 2,362), all of them 682 to 820 above today's rule's 1,542 and near three times the random list's 780: having a model matters far more than which family. XGBoost calls 138 more readmitted patients in six months than logistic regression. Trees find combinations of columns and thresholds within a column by splitting; logistic regression gives each column one straight-line slope unless someone writes the combination as a column. Practice 6 tests whether the gap comes from combinations.

The frozen choice is XGBoost with its validation-chosen 285 rounds: the most readmitted patients called among the four families, and seconds to train. Refit it on every training and validation discharge whose label had arrived by the first test discharge, then score the test rows once at the 10% budget, beside today's rule on the same rows.

```python
train_valid = pd.concat([train, valid])
final_rows = train_valid[train_valid["label_time"] <= test["discharge_date"].min()]
final_preprocess = make_preprocess().fit(final_rows[features])
final_model = xgb.XGBClassifier(n_estimators=booster.best_iteration + 1, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
final_model.fit(final_preprocess.transform(final_rows[features]), final_rows["readmit_30d"])

test_probability = final_model.predict_proba(final_preprocess.transform(test[features]))[:, 1]
compare("XGBoost on test (frozen)", test_probability, truth=test["readmit_30d"])
todays_rule("today's rule on test", test)
print("final fit rows:", len(final_rows), " test readmissions:", test["readmit_30d"].sum())
show(["XGBoost", "XGBoost on test (frozen)", "today's rule on test"])
```

Expect 179,288 final fit rows, 5,789 test readmissions, and a test recall of 0.311 with precision 0.479, close to validation's 0.308 and 0.500. Today's rule on the test discharges reaches a recall of 0.205 and precision of 0.316: 1,186 of the 5,789 readmitted patients, against XGBoost's 1,801.

**What we tell Dr. Kim**: on January to May 2025, discharges no choice was tuned on, the XGBoost list reaches 615 more readmitted patients than today's rule for the same 10% of calls. The four model families differ far less than any of them differs from the rule, so logistic regression stays a fallback if the committee wants coefficients it can read.

## Independent practice

### 1. The chief nursing officer asks which discharge details drive the list

Permutation importance on validation rows measures what the forest relies on for new discharges. `scoring="neg_log_loss"` scores probabilities with log loss, the loss the classifiers were trained to reduce; a 10,000-row sample keeps it quick.

```python
from sklearn.inspection import permutation_importance

sample = valid.sample(10000, random_state=42)
permuted = permutation_importance(forest, sample[features], sample["readmit_30d"],
                                  scoring="neg_log_loss", n_repeats=5, random_state=42)
display(pd.Series(permuted.importances_mean, index=features).sort_values(ascending=False).round(4)
        .rename("log_loss_increase"))
```

Expect diagnosis group ahead (0.0135), then comorbidity count (0.0081), medication count (0.0064), prior admissions (0.0047), hemoglobin, age, and follow-up scheduled, with `hospital`, `adi_decile`, and `sex` near 0. Prior admissions and follow-up scheduled are missing from step 5's impurity top six: impurity rewards columns with many distinct values such as the labs, while permutation shuffles a whole original column and measures the loss on new discharges.

### 2. The nurses want to read "30% risk" at face value

The nurses may read the probability itself to a patient's family. Group validation discharges into tenths by XGBoost's probability and compare the mean prediction with the observed rate.

```python
calibration = pd.DataFrame({"predicted": booster.predict_proba(X_valid_ready)[:, 1], "observed": y_valid.to_numpy()})
calibration["tenth"] = pd.qcut(calibration["predicted"], 10, labels=False) + 1
display(calibration.groupby("tenth")[["predicted", "observed"]].mean().round(3))
```

Expect the two columns to rise together from about 0.02 in the lowest tenth to 0.49 in the highest, within 0.010 of each other in every tenth: a predicted 30% means about 30% of those patients come back.

### 3. A colleague's forest catches 69% of readmissions

The colleague grew 50 trees with no leaf-size limit and reports recall on the rows it was fit on. Compare recall at the 10% budget on fit rows and validation rows.

```python
deep = Pipeline([("preprocess", make_preprocess()),
                 ("model", RandomForestClassifier(n_estimators=50, n_jobs=-1, random_state=42))])
deep.fit(X_fit, y_fit)
for label, X_part, y_part in [("fit rows", X_fit, y_fit), ("validation", X_valid, y_valid)]:
    probability = deep.predict_proba(X_part)[:, 1]
    flag = probability >= pd.Series(probability).quantile(0.9)
    print(label, "recall at 10%:", round(recall_score(y_part, flag), 3))
```

Expect a fit-row recall of 0.693, far above any validation score: the deep trees memorized the training discharges. Validation recall is 0.317, 0.009 above the frozen XGBoost (about 70 more readmitted patients called), because averaging 50 deep trees cancels much of what each one memorized. This forest was not among the candidates in step 8, and choosing it now, after seeing validation, would turn validation into a second test set; it goes on the list for the next model review. The fit-row score says nothing about new discharges; only the validation score does.

### 4. A reviewer proposes adding whether the patient kept the follow-up visit

Add `followup_completed`, known only 7 to 14 days after discharge, to the XGBoost features.

```python
leaky = features + ["followup_completed"]
leaky_preprocess = make_preprocess(categorical + ["followup_completed"]).fit(fit_rows[leaky])
leaky_model = xgb.XGBClassifier(n_estimators=booster.best_iteration + 1, max_depth=4, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, random_state=42, n_jobs=-1)
leaky_model.fit(leaky_preprocess.transform(fit_rows[leaky]), y_fit)
compare("XGBoost + followup_completed (leaky)", leaky_model.predict_proba(leaky_preprocess.transform(valid[leaky]))[:, 1])
show(["XGBoost", "XGBoost + followup_completed (leaky)"])
```

Expect the leaky model's recall about 2.6 points higher (0.334 against 0.308). The gain is not real: readmitted patients often miss the follow-up because they are back in the hospital, and the column does not exist yet when the nurses make their calls. The reply to the reviewer: the column is leakage.

### 5. IT wants the nightly scoring job to finish sooner

Scoring time grows with the number of trees. Refit XGBoost with `learning_rate=0.3` and compare the best round and the recall.

```python
fast = xgb.XGBClassifier(n_estimators=1000, max_depth=4, learning_rate=0.3, subsample=0.8,
                         colsample_bytree=0.8, early_stopping_rounds=30, random_state=42, n_jobs=-1)
fast.fit(X_fit_ready, y_fit, eval_set=[(X_valid_ready, y_valid)], verbose=False)
print("best round:", fast.best_iteration, " validation log loss:", round(fast.best_score, 4))
compare("XGBoost (learning rate 0.3)", fast.predict_proba(X_valid_ready)[:, 1])
show(["XGBoost", "XGBoost (learning rate 0.3)"])
```

Expect best round 39 instead of 284, about 7 times fewer trees to score, with a slightly worse validation log loss (0.3804) and recall (0.304): bigger steps need fewer trees but land less precisely. The tradeoff for IT: 0.004 recall, about 30 readmitted patients in six months, for a faster job.

### 6. The committee prefers a model it can explain

Logistic regression gives each column one slope, so a combination counts only if it is a column. Look for combinations in the fit rows first, with pivot tables (Lecture 08), then write down only what the tables show.

```python
fit_view = fit_rows.assign(age_75_plus=fit_rows["age"] >= 75,
                           sent_home=fit_rows["disposition"].isin(["home", "home_health"]))
display(fit_view.pivot_table(index="dx_group", columns="followup_scheduled", values="readmit_30d").round(3))
display(fit_view.pivot_table(index=["lives_alone", "age_75_plus"], columns="sent_home", values="readmit_30d").round(3))
sodium_band = pd.cut(fit_view["sodium"], [100, 130, 133, 135, 137, 140, 145, 170])
display(fit_view.groupby(sodium_band, observed=True)["readmit_30d"].agg(["mean", "size"]).round(3))
```

Expect three patterns, each worth one column:

- **Heart failure without follow-up**: 0.461 against 0.247 with a scheduled visit. Every other diagnosis gains only 1 to 5 points from follow-up.
- **Living alone, 75 and over, sent home**: 0.290, while the same patients sent to a facility run 0.245 and every other group runs lower at home than elsewhere.
- **Sodium at 133 or below**: 0.241 to 0.300, against 0.119 to 0.162 above 133. The risk steps up below 133 rather than rising evenly, so one straight-line slope fits it poorly.

```python
def add_terms(frame):
    frame = frame.copy()
    frame["hf_no_followup"] = (frame["dx_group"] == "heart_failure") & ~frame["followup_scheduled"]
    frame["alone_75_home"] = frame["lives_alone"] & (frame["age"] >= 75) & frame["disposition"].isin(["home", "home_health"])
    frame["sodium_133_or_less"] = frame["sodium"] <= 133
    return frame

terms = ["hf_no_followup", "alone_75_home", "sodium_133_or_less"]
handmade = Pipeline([("preprocess", make_preprocess(categorical + terms)), ("model", LogisticRegression(max_iter=1000))])
handmade.fit(add_terms(fit_rows)[features + terms], y_fit)
compare("logistic + hand-built terms", handmade.predict_proba(add_terms(valid)[features + terms])[:, 1])
show(["logistic regression", "XGBoost", "logistic + hand-built terms"])
```

Expect the hand-built logistic model at a recall of 0.298 (2,279 readmitted patients called): 55 more than plain logistic regression and 83 fewer than XGBoost, so the three columns close a little under half the gap. The rest sits in patterns these tables did not look at; a pivot table of `housing_unstable` by `ed_visits_6m` is one place to start. XGBoost needed no tables: its splits search for combinations on their own.

## Optional extension: the sodium cutoff the cardiology lead asks about

Prerequisite: BONUS.md, Partial Dependence Plots.

Partial dependence varies one column across a grid while averaging the forest's predictions over everything else. Run on 5,000 fit rows:

```python
from sklearn.inspection import partial_dependence

curve = partial_dependence(forest, X_fit.sample(5000, random_state=0), ["sodium"], grid_resolution=15)
display(pd.Series(curve["average"][0], index=curve["grid_values"][0], name="mean predicted risk").round(3))
```

Expect a curve: predicted risk falls from 0.205 at 131 mmol/L to about 0.155 at 135 and stays nearly flat above it, the same step practice 6's table shows. Logistic regression gives sodium one straight-line slope, so it cannot represent this bend.
