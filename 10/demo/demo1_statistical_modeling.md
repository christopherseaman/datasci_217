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

# Demo 1: Statistical Modeling and Framing a Prediction Problem

A diabetes clinic recorded age, sex, BMI, average blood pressure, and six blood tests for 442 patients, then scored how far each patient's disease had progressed one year later. You fit linear regressions with `statsmodels`, read coefficients with their uncertainty, check residuals, compare models, and add a categorical predictor. Then you switch to a home blood-pressure program and frame a prediction problem: the target and its time, a feature audit, clock-face hour features, and a chronological split. Everything here comes from Lecture 10 up to the first demo break, plus Lectures 01 to 09. The diabetes records are real and de-identified; the blood-pressure readings are synthetic.

**How to run:** open this notebook in Colab from the lecture page's Colab link, or locally in VS Code with the kernel set to a `.venv` made by `uv venv --seed` and `uv pip install -r requirements.txt` in this folder (Lecture 03). Run the cells from top to bottom; after each step, an **Expect** line says what you should see. Colab does not save your changes back to GitHub; use **File → Save a copy in Drive** to keep them. Tested 2026-09-25 with Python 3.13, pandas 3.0.5, NumPy 2.3.3, statsmodels 0.14.6, scikit-learn 1.9.0, and matplotlib 3.11.1; the whole notebook runs in a few seconds.

## Setup

The first cell installs pandas 3.0.5, the course version. Colab ships an older pandas (2.2). A `.venv` made with `uv venv --seed` includes pip, so the same `%pip` cell works locally too.

```python
# Setup: install the course's pandas version (Colab and local)
%pip install -q pandas==3.0.5
```

**Expect:** `Note: you may need to restart the kernel to use updated packages.` Locally, with the requirements already installed, the cell changes nothing and you can go on. In Colab, pip may also print a dependency conflict because some preinstalled packages expect pandas 2.2; that is expected, and this demo does not use them. If Colab asks you to restart the session (or says pandas was previously imported), choose **Runtime → Restart session**, then continue with the next cell. You do not need to rerun the install.

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.datasets import load_diabetes

print('pandas', pd.__version__)
print('statsmodels', statsmodels.__version__)
```

**Expect:** `pandas 3.0.5` and `statsmodels 0.14.6`.

## 1. Load the diabetes records

scikit-learn ships a few small real datasets. `load_diabetes(scaled=False, as_frame=True)` returns this one in its original units as a DataFrame, with the six blood tests named `s1` to `s6`; rename them so the formulas below read clearly.

| Column | Meaning | Units |
| --- | --- | --- |
| `age` | Age at baseline | years |
| `sex` | Sex, coded 1 or 2 (the source does not say which is which) | code |
| `bmi` | Body mass index | kg/m² |
| `bp` | Average blood pressure | mmHg |
| `tc`, `ldl`, `hdl` | Total, LDL, and HDL cholesterol | mg/dL |
| `tch` | Total cholesterol divided by HDL | ratio |
| `ltg` | Serum triglycerides, on a log scale | log units |
| `glu` | Blood glucose | mg/dL |
| `progression` | Disease progression one year after baseline | score, 25 to 346 |

The source lists the blood tests without units; their ranges match mg/dL.

```python
diabetes = load_diabetes(scaled=False, as_frame=True).frame
diabetes = diabetes.rename(columns={'s1': 'tc', 's2': 'ldl', 's3': 'hdl', 's4': 'tch',
                                    's5': 'ltg', 's6': 'glu', 'target': 'progression'})
print(diabetes.shape)
print(diabetes.head(3))
print(diabetes[['age', 'bmi', 'bp', 'progression']].describe().round(1))
```

**Expect:** `(442, 11)`, and these summary rows:

```text
         age    bmi     bp  progression
count  442.0  442.0  442.0        442.0
mean    48.5   26.4   94.6        152.1
std     13.1    4.4   13.8         77.1
min     19.0   18.0   62.0         25.0
max     79.0   42.2  133.0        346.0
```

The printed table also has the 25%, 50%, and 75% rows between `min` and `max`.

Look before you model: plot the outcome against each candidate predictor.

```python
fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharey=True)
for ax, column, label in zip(axes, ['age', 'bmi', 'bp'],
                             ['Age (years)', 'BMI (kg/m²)', 'Average BP (mmHg)']):
    ax.scatter(diabetes[column], diabetes['progression'], alpha=0.4)
    ax.set_xlabel(label)
axes[0].set_ylabel('Progression after one year')
plt.show()
plt.close(fig)
```

**Expect:** three scatter plots sharing one y-axis. Progression climbs with BMI and, more loosely, with blood pressure; against age the cloud is nearly flat.

## 2. The formula API

Fit progression on age, BMI, and blood pressure. Read the formula as "progression is modeled by age, BMI, and BP"; `statsmodels` adds the intercept for you.

```python
results_formula = smf.ols('progression ~ age + bmi + bp', data=diabetes).fit()
print(results_formula.summary())
```

**Expect:** a header that reports `No. Observations:` 442 and `R-squared:` 0.396 (it also shows the date and time you ran it), then this coefficient block:

```text
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
Intercept   -205.1120     22.554     -9.094      0.000    -249.440    -160.784
age            0.0944      0.232      0.407      0.685      -0.362       0.551
bmi            8.5016      0.707     12.031      0.000       7.113       9.890
bp             1.3569      0.235      5.763      0.000       0.894       1.820
```

The same numbers are available as attributes, which is how you use them in later code:

```python
print(f"R-squared: {results_formula.rsquared:.4f}")
print(f"Adjusted R-squared: {results_formula.rsquared_adj:.4f}")
print(pd.DataFrame({'coef': results_formula.params.round(3),
                    'p_value': results_formula.pvalues.round(4)}))
```

**Expect:**

```text
R-squared: 0.3962
Adjusted R-squared: 0.3921
              coef  p_value
Intercept -205.112   0.0000
age          0.094   0.6845
bmi          8.502   0.0000
bp           1.357   0.0000
```

Read each coefficient as an association, holding the other two predictors fixed:

- **bmi:** each 1 kg/m² higher BMI goes with an 8.5-point higher fitted progression score, at the same age and blood pressure.
- **bp:** each 1 mmHg higher average blood pressure goes with a 1.36-point higher fitted score.
- **age:** 0.09 points per year, with p = 0.68. Among patients with the same BMI and blood pressure, these data show no clear association with age. That is not proof that age does not matter.
- **Intercept:** the fitted score for age 0, BMI 0, and BP 0, a patient who cannot exist. It anchors the line and is not interpreted.

These are observational records, so none of this says that lowering a patient's BMI _would_ slow progression; that needs a trial. R² of 0.396 means the three predictors explain about 40% of the variation in progression.

## 3. The array API

The array API asks you to build the table of predictors yourself, including the intercept column that `sm.add_constant()` adds. Both interfaces run the same least-squares fit.

```python
X = sm.add_constant(diabetes[['age', 'bmi', 'bp']])
print(X.head(3))

results_array = sm.OLS(diabetes['progression'], X).fit()

side_by_side = pd.DataFrame({'formula': results_formula.params.values,
                             'array': results_array.params.values},
                            index=results_array.params.index)
side_by_side['difference'] = side_by_side['formula'] - side_by_side['array']
print(side_by_side)
```

**Expect:** the first rows of `X` with a `const` column of 1.0 in front of `age`, `bmi`, and `bp`, then:

```text
          formula       array  difference
const -205.111971 -205.111971         0.0
age      0.094411    0.094411         0.0
bmi      8.501595    8.501595         0.0
bp       1.356921    1.356921         0.0
```

The array API names the intercept `const` instead of `Intercept`; the numbers are identical.

Use the formula API for quick work with DataFrame columns; use the array API when the predictors are already an array, or when you want to build the design table yourself.

## 4. Uncertainty and residuals

A coefficient is an estimate from one sample of 442 patients. Its standard error says how much it would vary from sample to sample, and the 95% confidence interval is roughly the estimate plus or minus two standard errors.

```python
print(f"Rows used: {results_formula.nobs:.0f}")
print(f"Residual degrees of freedom: {results_formula.df_resid:.0f}")
print(f"F-statistic: {results_formula.fvalue:.2f} (p = {results_formula.f_pvalue:.2e})")

coef_summary = pd.DataFrame({
    'coef': results_formula.params,
    'std_err': results_formula.bse,
    'ci_lower': results_formula.conf_int()[0],
    'ci_upper': results_formula.conf_int()[1],
})
print(coef_summary.round(3))
```

**Expect:**

```text
Rows used: 442
Residual degrees of freedom: 438
F-statistic: 95.81 (p = 1.09e-47)
              coef  std_err  ci_lower  ci_upper
Intercept -205.112   22.554  -249.440  -160.784
age          0.094    0.232    -0.362     0.551
bmi          8.502    0.707     7.113     9.890
bp           1.357    0.235     0.894     1.820
```

The residual degrees of freedom are 442 rows minus 4 estimated coefficients. The F-test asks whether all three slopes could be 0 together; p = 1.09e-47 says no. The `age` interval runs from -0.36 to 0.55 and contains 0, which is the same message as its p-value of 0.68. The `bmi` interval, 7.1 to 9.9, is far from 0.

Coefficients only mean something if the straight-line form fits. Plot each patient's residual (observed minus fitted) against the fitted value: a shapeless cloud around zero is what we want, a curve says the form is wrong, and a funnel says the spread is not constant.

```python
print(results_formula.resid.describe().round(1))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(results_formula.fittedvalues, results_formula.resid, alpha=0.5)
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Fitted progression score')
ax.set_ylabel('Residual (observed - fitted)')
plt.show()
plt.close(fig)
```

**Expect:** residuals with mean 0.0 (least squares guarantees it), standard deviation 59.9, and a range of -145.7 to 154.4. In the plot there is no curve, but two things stand out:

- The lower-left edge is a straight diagonal line. Progression cannot fall below 25, so a patient with a low fitted score cannot miss by much on the low side.
- The cloud is narrower at low fitted scores than in the middle, a mild funnel: the spread is not quite constant.

Neither ruins the fit, but both say the straight-line model is an approximation near the bottom of the scale.

## 5. Intervals for new patients

Three new 50-year-olds with rising BMI and blood pressure. `get_prediction(...).summary_frame()` gives both intervals from the lecture: `mean_ci_*` for the average patient like this one, and `obs_ci_*` for one individual patient.

```python
new_patients = pd.DataFrame({'age': [50, 50, 50],
                             'bmi': [22.0, 28.0, 34.0],
                             'bp': [85, 95, 105]})
intervals = results_formula.get_prediction(new_patients).summary_frame(alpha=0.05)
print(new_patients)
print(intervals.drop(columns='mean_se').round(1))
```

**Expect:**

```text
   age   bmi   bp
0   50  22.0   85
1   50  28.0   95
2   50  34.0  105
    mean  mean_ci_lower  mean_ci_upper  obs_ci_lower  obs_ci_upper
0  102.0           93.5          110.4         -16.5         220.4
1  166.6          160.5          172.6          48.3         284.9
2  231.1          219.8          242.4         112.5         349.8
```

The average score for patients like the second one is pinned down to about 160 to 173, but a single such patient could land anywhere from about 48 to 285: the prediction interval adds person-to-person variation. The first prediction interval starts at -16.5, a score no patient can have (the scale starts at 25). The straight line and its normal-shaped errors do not know the scale has a floor, which is the same issue the residual plot showed.

## 6. Compare models

Add predictors one step at a time and compare the fits. All four models use the same 442 rows, so their AIC and BIC are comparable (lower is better).

```python
formulas = {
    'Model 1: bmi': 'progression ~ bmi',
    'Model 2: bmi + bp': 'progression ~ bmi + bp',
    'Model 3: age + bmi + bp': 'progression ~ age + bmi + bp',
    'Model 4: all ten': 'progression ~ age + sex + bmi + bp + tc + ldl + hdl + tch + ltg + glu',
}

rows = []
for name, formula in formulas.items():
    fitted = smf.ols(formula, data=diabetes).fit()
    rows.append({'model': name, 'n_coef': len(fitted.params),
                 'r2': fitted.rsquared, 'adj_r2': fitted.rsquared_adj,
                 'aic': fitted.aic, 'bic': fitted.bic})
comparison = pd.DataFrame(rows)
print(comparison.round(4).to_string(index=False))
```

**Expect:**

```text
                  model  n_coef     r2  adj_r2       aic       bic
           Model 1: bmi       2 0.3439  0.3424 4912.0382 4920.2208
      Model 2: bmi + bp       3 0.3960  0.3932 4877.4878 4889.7618
Model 3: age + bmi + bp       4 0.3962  0.3921 4879.3210 4895.6863
       Model 4: all ten      11 0.5177  0.5066 4793.9857 4838.9901
```

- **R² never falls when a predictor is added.** Adding age (Model 2 to Model 3) nudges it from 0.3960 to 0.3962.
- **Adjusted R², AIC, and BIC charge for each coefficient.** For age, all three get worse: adjusted R² drops, AIC and BIC rise. Age costs a coefficient and explains almost nothing once BMI and BP are in.
- **The blood tests earn their place.** Model 4 is best on all three criteria by a wide margin: adjusted R² 0.507 against 0.393, and AIC and BIC about 80 and 50 lower than Model 2.

Whether six blood tests are worth drawing for every patient is a separate, clinical judgement; the criteria only say the extra columns explain more than they cost.

## 7. A categorical predictor

Clinics often report BMI in bands. `pd.cut()` (Lecture 05) makes the bands; `C()` in the formula treats them as categories, with the first band as the reference. The top edge, 60, sits above the largest BMI (42.2), so every patient lands in a band.

```python
diabetes['bmi_group'] = pd.cut(diabetes['bmi'], bins=[0, 25, 30, 60],
                               labels=['25 or under', 'over 25 to 30', 'over 30'])
print(diabetes['bmi_group'].value_counts())
print(f"Patients without a band: {diabetes['bmi_group'].isna().sum()}")

results_cat = smf.ols('progression ~ age + bp + C(bmi_group)', data=diabetes).fit()
print(f"\nRows used: {results_cat.nobs:.0f}")
print(results_cat.params.round(2))
```

**Expect:**

```text
bmi_group
25 or under      190
over 25 to 30    157
over 30           95
Name: count, dtype: int64
Patients without a band: 0

Rows used: 442
Intercept                       -42.03
C(bmi_group)[T.over 25 to 30]    44.37
C(bmi_group)[T.over 30]          85.01
age                               0.23
bp                                1.57
dtype: float64
```

`25 or under` is the reference band, so it has no row of its own. At the same age and blood pressure, patients in the `over 25 to 30` band average about 44 points higher progression than the reference band, and patients `over 30` about 85 points higher. All 442 patients were used: had the top edge been lower than the largest BMI, the patients above it would have been left without a band, and the formula would have dropped them without a warning. Printing the row count is how you would catch that.

## 8. Framing a prediction problem

Everything so far used all 442 patients at once, which is right when the question is _how_ progression relates to BMI. A prediction question, "what will this patient's next reading be?", is judged on rows the model has never seen, and these baseline records have no time order to split on.

So this part switches to a home blood-pressure program: 30 patients each take one reading a week with a connected cuff, which uploads it with a timestamp. Most measure in the morning or evening; a few night-shift workers measure around midnight. The team wants to predict each patient's next weekly reading as soon as today's reading arrives. The target is the patient's **next** reading, so a grouped `shift(-1)` (Lecture 09) pulls it, and its timestamp, back onto the current row.

```python
rng = np.random.default_rng(217)
mondays = pd.date_range('2026-01-05', periods=8, freq='W-MON')
routine_hours = [7, 8, 9, 19, 21, 23, 0]   # each patient's usual reading hour

records = []
for patient_id in range(1, 31):
    age = int(rng.integers(35, 80))
    usual_hour = routine_hours[patient_id % len(routine_hours)]
    level = rng.normal(132 + 0.2 * age, 8)
    for monday in mondays:
        hour = (usual_hour + int(rng.integers(-1, 2))) % 24   # an hour earlier or later some weeks
        level = 0.7 * level + 0.3 * rng.normal(132 + 0.2 * age, 8)
        time_of_day = 5 * np.cos(2 * np.pi * (hour - 10) / 24)  # higher mid-morning, lower at night
        records.append({
            'patient_id': f'P{patient_id:03d}',
            'reading_time': monday + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60))),
            'age': age,
            'sbp_today': round(level + time_of_day + rng.normal(0, 3), 1),
            'repeat_sbp': round(level + rng.normal(0, 3), 1),  # nurse asks for a repeat the next day
        })

readings = pd.DataFrame(records).sort_values(['patient_id', 'reading_time'])
readings['sbp_next'] = readings.groupby('patient_id')['sbp_today'].shift(-1)
readings['target_time'] = readings.groupby('patient_id')['reading_time'].shift(-1)
readings = readings.dropna(subset=['sbp_next']).reset_index(drop=True)

print('Prediction unit: one reading. Target: sbp_next, measured at target_time.')
print(f'Rows with a target: {len(readings)}')
print(readings[['patient_id', 'reading_time', 'sbp_today', 'sbp_next', 'target_time']].head(3))
```

**Expect:** 210 rows: 30 patients times eight readings, minus each patient's last reading, which has no next one. The first rows show the shift at work: each row's `sbp_next` is the next row's `sbp_today`, and its `target_time` is the next row's `reading_time`.

```text
  patient_id        reading_time  sbp_today  sbp_next         target_time
0       P001 2026-01-05 09:36:00      148.2     143.5 2026-01-12 08:42:00
1       P001 2026-01-12 08:42:00      143.5     140.0 2026-01-19 09:07:00
2       P001 2026-01-19 09:07:00      140.0     146.5 2026-01-26 09:43:00
```

### Audit the candidate features

The prediction is made when today's reading uploads, so anything known later is leakage. The reading's hour comes from its own timestamp, so it is known at once.

```python
candidates = pd.DataFrame({
    'feature': ['age', 'sbp_today', 'reading_hour', 'repeat_sbp'],
    'known': ['at enrollment', 'at upload', 'at upload', 'next day'],
    'hours_after': [0, 0, 0, 24],
})
candidates['available'] = candidates['hours_after'] <= 0
candidates['decision'] = np.where(candidates['available'], 'Keep', 'Exclude (leakage)')
print(candidates)
```

**Expect:**

```text
        feature          known  hours_after  available           decision
0           age  at enrollment            0       True               Keep
1     sbp_today      at upload            0       True               Keep
2  reading_hour      at upload            0       True               Keep
3    repeat_sbp       next day           24      False  Exclude (leakage)
```

`repeat_sbp` would look useful, because it measures the same patient a day later, but it does not exist when the prediction is made. A model trained with it would score well on old data and could not be used on a new reading.

### Put the hour on a clock face

As a plain number, hour 23 and hour 0 look 23 apart. The sine and cosine of `2 * np.pi * hour / 24` place each hour on a circle instead.

```python
readings['reading_hour'] = readings['reading_time'].dt.hour
readings['hour_sin'] = np.sin(2 * np.pi * readings['reading_hour'] / 24)
readings['hour_cos'] = np.cos(2 * np.pi * readings['reading_hour'] / 24)

clock = readings[['reading_hour', 'hour_sin', 'hour_cos']].drop_duplicates()
clock = clock.sort_values('reading_hour')
print(clock.loc[clock['reading_hour'].isin([0, 1, 9, 22, 23])].round(2).to_string(index=False))
```

**Expect:**

```text
 reading_hour  hour_sin  hour_cos
            0      0.00      1.00
            1      0.26      0.97
            9      0.71     -0.71
           22     -0.50      0.87
           23     -0.26      0.97
```

Hour 23 sits at (-0.26, 0.97) and hour 0 at (0.00, 1.00): neighbors on the clock face, exactly as far apart as hours 0 and 1. The 9 o'clock readers sit on the other side of the circle. Both columns go to the model together, because either one alone gives two different hours the same value.

### Split on the target time

Split on when each **target** is measured, not when the features are, so no training outcome is measured during the validation weeks.

```python
train = readings[readings['target_time'] < '2026-02-09']
valid = readings[(readings['target_time'] >= '2026-02-09') & (readings['target_time'] < '2026-02-23')]
test = readings[readings['target_time'] >= '2026-02-23']

split_summary = pd.DataFrame({
    'rows': [len(train), len(valid), len(test)],
    'first_target': [train['target_time'].min(), valid['target_time'].min(), test['target_time'].min()],
    'last_target': [train['target_time'].max(), valid['target_time'].max(), test['target_time'].max()],
}, index=['train', 'valid', 'test'])
print(split_summary)
```

**Expect:**

```text
       rows        first_target         last_target
train   120 2026-01-12 00:07:00 2026-02-02 23:43:00
valid    60 2026-02-09 00:01:00 2026-02-16 23:34:00
test     30 2026-02-23 00:44:00 2026-02-23 23:53:00
```

120, 60, and 30 rows: four target weeks for training, two for validation, one for test. The three target-time ranges do not overlap, and the midnight readers' targets (00:07, 00:01, 00:44) land in the right week because the cutoffs compare full timestamps.

### Fit on training rows, read the validation rows

Fit with the kept features only, on training rows only. Then give each validation row a 95% prediction interval: `conf_int(obs=True)` returns just the individual-prediction bounds, as an array with one row per patient reading.

```python
honest_fit = smf.ols('sbp_next ~ age + sbp_today + hour_sin + hour_cos', data=train).fit()
print(honest_fit.params.round(3))

valid_bounds = honest_fit.get_prediction(valid).conf_int(obs=True)
check = valid[['reading_hour', 'sbp_today', 'sbp_next']].head(3).copy()
check['predicted'] = honest_fit.predict(valid).head(3).round(1)
check['pi_lower'] = valid_bounds[:3, 0].round(1)
check['pi_upper'] = valid_bounds[:3, 1].round(1)
print(check)
```

**Expect:**

```text
Intercept    83.248
age           0.040
sbp_today     0.401
hour_sin      0.934
hour_cos     -1.632
dtype: float64
    reading_hour  sbp_today  sbp_next  predicted  pi_lower  pi_upper
4              9      143.4     146.8      145.3     137.7     152.8
5              7      146.8     149.3      146.2     138.6     153.7
11            10      144.9     141.7      145.6     138.1     153.2
```

- **sbp_today, 0.40:** today's reading carries part of the way to next week's, holding the other features fixed.
- **hour_sin and hour_cos:** read them together. A 9:00 reading gets a prediction about 3.4 mmHg higher than a midnight reading with the same `sbp_today`, because patients who measure mid-morning also read higher the next week.
- **The intervals** are about 15 mmHg wide, and all three validation readings fall inside theirs.

Demo 2 measures how far off predictions like these are, on average, and whether they beat a simple guess.
