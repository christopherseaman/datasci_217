---
notion:
  title_line: "# From Statistics to Deep Learning: The Modern Modeling Landscape"
  role: lecture
  status: mapped
  page_id: "2b0d9fdd-1a1a-80f4-9871-ff3a726e57c3"
  url: "https://app.notion.com/p/2b0d9fdd1a1a80f49871ff3a726e57c3"
---

# From Statistics to Deep Learning: The Modern Modeling Landscape

See [BONUS.md](BONUS.md) for the optional extensions.

**Live notebooks in Colab:** [Demo 1](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/10/demo/demo1_statistical_modeling.ipynb) · [Demo 2](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/10/demo/demo2_sklearn_prediction.ipynb) · [Demo 3](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/10/demo/demo3_trees_boosting_networks.ipynb)

**Run locally:**

```shell
curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/10/demo/setup_demo.sh | sh
```

![xkcd 1838: Machine Learning. Stirring the pile of linear algebra until the answers look right is not the same as checking them.](media/xkcd_1838.png)

<callout icon="🧭" color="blue_bg">
	# This Lecture
	## Covers
	- **Statistical modeling with `statsmodels`**: OLS formulas, coefficients, confidence intervals, p-values, residual plots, and new-patient intervals.
	- **Prediction setup**: features, targets, leakage, cyclic time features, and training, validation, and test rows split on target time.
	- **scikit-learn**: the estimator pattern, linear regression with Ridge and Lasso, logistic regression for classification, baselines, pipelines, metrics, time-ordered cross-validation, permutation importance, and the one final test.
	- **Trees, boosting, and neural networks**: random forests, XGBoost with early stopping, and a small Keras classifier.
	- **Readings**: McKinney, _Python for Data Analysis_ (3rd ed.) 12.1 to 12.4.
	## Enables
	- Fit `sbp ~ age + bmi` with `statsmodels`, read each coefficient with its 95% confidence interval, and give a new patient a prediction interval for SBP.
	- Drop features that arrive after prediction time and split visits into training, validation, and test rows by target time.
	- Compare a baseline, a linear pipeline, and Ridge or Lasso on validation MAE, then freeze the winner and score the test set once.
	- Check a readmission classifier's confusion matrix and recall, since accuracy alone hides missed readmissions.
	- Fit a random forest, XGBoost, and a Keras network on biopsy measurements and compare them on the same validation rows.
</callout>

# What Is a Model?

- **Model**: a simplified mathematical description of how an outcome, such as systolic blood pressure (SBP), relates to other variables, such as age and BMI.
- Like Lecture 08's `groupby('clinic')['wait_min'].mean()`, it estimates an average outcome, but for any combination of inputs.
- The question it serves decides how to judge it:

| Question | Example | Judge it by |
| --- | --- | --- |
| **Inference**: how are the variables related? | Is BMI associated with SBP among patients of the same age, and how sure are we? | Coefficients, their uncertainty, and the model's assumptions |
| **Prediction**: what will the outcome be? | What will this patient's SBP be at the next visit? | Error on held-out observations representing the intended use |

![Tufte, _The Visual Display of Quantitative Information_, p. 47: male lung-cancer deaths per million in 1950 against cigarettes per person in 1930 for 11 countries, with a fitted line and r = 0.73 ± 0.30.](media/tufte_lung_cancer_cigarettes.png)

## The Modeling Landscape

- More flexible models fit more complex patterns but can also **overfit**: fit the noise in the training rows, so predictions for new patients get worse.
- Shortlist libraries by the question, then compare candidates on held-out rows from the workflow you intend to use.

![The usual rule of thumb, not a measured ranking: flexible models such as neural networks and boosted ensembles sit high on predictive power and low on interpretability, linear regression the reverse.](media/interpretability_tradeoff.webp)

### Reference Card: Which Library for Which Question

| Library | Start here when | Key features | Typical use |
| --- | --- | --- | --- |
| **statsmodels** | You need to quantify a relationship and its uncertainty | Statistical inference, model diagnostics | Understanding relationships, research |
| **scikit-learn** | Tabular data and you need predictions | One fit/predict pattern, preprocessing, many models | General prediction tasks |
| **XGBoost** | Candidate for tabular prediction | Gradient-boosted trees, feature-importance summaries | Benchmarking alongside simpler tabular models |
| **TensorFlow/Keras** | Candidate for images, text, audio, or learned representations | Neural-network layers and training loops | Deep-learning workflows |

_Pro tip: Start simple. "But why male models?" Because sometimes the simplest model is the right model!_

# Statistical Modeling with `statsmodels`

- **`statsmodels`**: Python's library for statistical inference.
- **Linear regression**: its workhorse; estimates an outcome as an intercept plus a weighted sum of predictors, as in `sbp = b0 + b1 * age + b2 * bmi`, and reports how uncertain each weight is, so it can say whether higher BMI goes with higher SBP among patients of the same age.

_Think of linear regression as the Derek Zoolander of modeling: simple, reliable, and it can turn left, turn right, or even turn statistically significant._

## Linear Regression by Least Squares

- A **fitted value** is the model's estimated average SBP for patients like this one: their age and BMI plugged into the fitted equation.
- The **intercept** (`b0`) is the fitted SBP when every predictor is 0: an anchor for the line, not a real patient (a newborn with a BMI of 0 would make the journals).
- A **coefficient** (`b2`) is the difference in fitted SBP for a one-unit difference in BMI, _holding age fixed_.
- A **residual** is one row's miss, observed minus fitted. **Ordinary least squares (OLS)** picks the intercept and coefficients that make the sum of squared residuals as small as possible.
- The general form is `y = β₀ + β₁x₁ + β₂x₂ + ... + ε`: here y is SBP, x₁ is age, x₂ is BMI, and ε, the **error term**, is what the predictors do not explain.

![The dashes are vertical: OLS measures each miss straight up or down (`observed y - fitted y`), not as the shortest distance from the point to the line.](media/ols_residuals.png)

## Formulas and Arrays

`statsmodels` has two interfaces. The formula syntax comes from the **Patsy** library. In a formula, read `~` as "is modeled by" and `+` as "also include this predictor", not arithmetic:

```text
Formula: smf.ols('sbp ~ age + bmi', data=clinic)                         intercept added for you
Array:   sm.OLS(clinic['sbp'], sm.add_constant(clinic[['age', 'bmi']]))  intercept column added by you
```

### Reference Card: `statsmodels` Essentials

| Method / attribute | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `smf.ols('sbp ~ age + bmi', data=df)` | Build a formula-based OLS model from DataFrame columns (`import statsmodels.formula.api as smf`). | Unfitted model |
| `sm.OLS(y, sm.add_constant(X))` | Build an array-based OLS model; `sm.add_constant(X)` adds the intercept column (`import statsmodels.api as sm`). | Unfitted model |
| `'sbp ~ ' + ' + '.join(predictors)` | Build a formula string from a list of column names. | `'sbp ~ age + bmi'` |
| `C(column)` in a formula | Treat a column as categorical: each level except the reference gets a coefficient, compared with the reference. Like Lecture 05's `drop_first=True`: the dropped level is the reference, since all-level 0/1 columns would duplicate the intercept. | Extra coefficient rows |
| `np.log(age)`, `I(bmi ** 2)`, `age:bmi`, `age * bmi` in a formula | Transform a column inside the formula; `I()` makes `**` ordinary arithmetic. `a:b` adds only the interaction (product) term, so one feature's slope can depend on the other; `a * b` is shorthand for `a + b + a:b`. | Extra coefficient rows named after each term |
| `model.fit()` | Estimate the coefficients. | Results object |
| `results.summary()` | Print coefficients, uncertainty, fit statistics, and diagnostics. | Formatted text table |

### Reference Card: OLS Results

| Method / attribute | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `results.params` | Fitted intercept and coefficients by term. | Series indexed by term (an array only when the inputs are NumPy arrays) |
| `results.rsquared` | Share of the outcome's variation the fitted model explains in these rows (0 to 1). | Float |
| `results.predict(new_rows)` | Fitted values for new rows with the same columns. | Series |
| `results.rsquared_adj`, `results.fvalue` / `results.f_pvalue`, `results.aic` / `results.bic` | Other numbers in the summary header: R² adjusted for the number of predictors; the F-test that all slopes are 0; information criteria for comparing models fit to the same rows (lower is better). | Float |
| `results.nobs` / `results.df_resid` | Rows used, and residual degrees of freedom (rows minus estimated coefficients). | Float |

### Code Snippet: OLS with the Formula API

`clinic` holds 60 simulated patients' `age`, `bmi`, and `sbp`, generated with true slopes of 0.5 mmHg per year of age and 0.8 mmHg per kg/m² of BMI.

```python
results = smf.ols('sbp ~ age + bmi', data=clinic).fit()
display(results.params.round(2).rename('coef'))
```

|  | coef |
| --- | --- |
| Intercept | 75.21 |
| age | 0.66 |
| bmi | 1.19 |

`results.summary()` shows the same numbers under `coef`, beside `std err`, `P>|t|`, and `[0.025 0.975]` (next section), and reports R-squared, 0.536 here.

![The clinic fit's full `results.summary()` printout, with the parts this lecture reads boxed in color among fit statistics and diagnostics it skips.](media/ols_summary.png)

Transforms and interactions name their own coefficient rows:

```python
print(smf.ols('sbp ~ age * bmi', data=clinic).fit().params.index.tolist())
print(smf.ols('sbp ~ np.log(age) + I(bmi ** 2)', data=clinic).fit().params.index.tolist())
```

```text
['Intercept', 'age', 'bmi', 'age:bmi']
['Intercept', 'np.log(age)', 'I(bmi ** 2)']
```

<callout icon="⚠️" color="yellow_bg">
	## Check `results.nobs` after every fit
	`statsmodels` silently drops rows with a missing value in the formula's columns: blank out three BMIs and the same call fits 57 rows, not 60. Compare `results.nobs` with `len(clinic)`.
</callout>

![xkcd 552: Correlation. "Correlation doesn't imply causation, but it does waggle its eyebrows suggestively and gesture furtively while mouthing 'look over there'."](media/xkcd_552.png)

## Uncertainty and New-Patient Intervals

A coefficient is an estimate from one sample, so it comes with uncertainty:

- The **standard error** measures how much the estimate would vary from sample to sample.
- A **95% confidence interval** is roughly the estimate ± 2 standard errors: repeat the study many times and about 95% of these intervals would contain the true coefficient.
- A **p-value** asks: if the true coefficient were 0 and the model assumptions held, how surprising would an estimate at least this far from 0, relative to its standard error, be? It is not the probability that a hypothesis is true.
- An **association** means two variables move together; **causation** means changing one would change the other. A coefficient from observational records describes an association, so "losing weight would lower SBP by b2" needs additional causal evidence, such as a randomized trial.

A new patient gets two intervals:

- **Mean-response interval**: where the _average_ SBP of all 55-year-olds with BMI 30 probably lies.
- **Prediction interval**: where _one_ such patient's SBP probably lies; it adds person-to-person variation, so it is always wider.

![At age 55 the mean-response band (blue) is far narrower than the prediction band (orange): the BMI 30 patient's 147.2 has a mean interval of 144.3 to 150.2 and a prediction interval of 131.4 to 163.0.](media/ols_intervals.png)

### Reference Card: OLS Uncertainty

| Method / attribute | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `results.bse` | Standard error of each coefficient. | Series indexed by term |
| `results.conf_int(alpha=0.05)` | Lower and upper 95% bounds. | DataFrame with columns `0` and `1` |
| `results.pvalues` | p-value for each coefficient, testing "the true coefficient is 0". | Series |
| `results.get_prediction(new_rows).summary_frame(alpha=0.05)` | Intervals for new rows: `mean`, `mean_ci_lower`, `mean_ci_upper` (mean response) and `obs_ci_lower`, `obs_ci_upper` (individual prediction); `.conf_int(obs=True)` returns only the prediction bounds. | DataFrame |

### Code Snippet: Uncertainty and a New-Patient Interval

```python
display(pd.DataFrame({'std_err': results.bse.round(2),
                      'ci_lower': results.conf_int()[0].round(2),
                      'ci_upper': results.conf_int()[1].round(2),
                      'p_value': results.pvalues.round(4)}))

new_patient = pd.DataFrame({'age': [55], 'bmi': [30.0]})
display(results.get_prediction(new_patient).summary_frame(alpha=0.05).round(1))
```

|  | std_err | ci_lower | ci_upper | p_value |
| --- | --- | --- | --- | --- |
| Intercept | 10.18 | 54.83 | 95.59 | 0.0000 |
| age | 0.09 | 0.48 | 0.84 | 0.0000 |
| bmi | 0.34 | 0.52 | 1.86 | 0.0008 |

|  | mean | mean_se | mean_ci_lower | mean_ci_upper | obs_ci_lower | obs_ci_upper |
| --- | --- | --- | --- | --- | --- | --- |
| 0 | 147.2 | 1.5 | 144.3 | 150.2 | 131.4 | 163.0 |

- `age` row: holding BMI fixed, each extra year is associated with about 0.66 mmHg higher fitted SBP (95% CI 0.48 to 0.84).
- Both intervals contain the slopes the data were simulated with.
- A p-value printed as 0.0000 is below 0.00005, not exactly 0.
- For the new patient the mean-response interval is about 6 mmHg wide, the prediction interval about 32: a clinic can be confident about the average 55-year-old, much less about the one in the exam room.

![xkcd 882: Significant. Test twenty jelly-bean colors at p < 0.05 and one of them will look significant by chance.](media/xkcd_882.png)

## Checking Residuals

The default standard errors and intervals assume independent errors with constant spread around the correct mean; the small-sample intervals also assume normally distributed errors. A **residuals-versus-fitted plot** checks the straight-line form and the constant spread:

- Shapeless cloud around zero: encouraging.
- Curve: the straight-line form is wrong.
- Funnel: the spread is not constant.
- A plot can reveal problems but cannot prove the assumptions. Repeated readings from the same patient break independence even when the cloud looks fine.

![Two residuals-versus-fitted plots. Left: the clinic fit's residuals scatter without pattern around a dashed zero line. Right: a straight line fitted to curved data leaves a U-shaped pattern.](media/ols_residuals_vs_fitted.png)

![Anscombe's quartet: four datasets share one fitted line, y = 3.00 + 0.50x, and R² = 0.67, but only plots show that II curves, III has one point far off the line, and IV rests on a single point.](media/anscombe_quartet.png)

### Reference Card: Residual Diagnostics

| Method / attribute | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `results.fittedvalues` / `results.resid` | One fitted value and one residual (observed minus fitted) per row. | Series |
| `ax.scatter(results.fittedvalues, results.resid)`, `ax.axhline(0, color='gray', linestyle='--')` | Residuals-versus-fitted plot; the reference line sits at 0. | Axes |
| `fig.savefig(path)`, `plt.show()`, `plt.close(fig)` | Save, display, then close; an unclosed figure stays in memory. | PNG on disk |

### Code Snippet: Residuals Versus Fitted Values

```python
diagnostics = pd.DataFrame({'observed': clinic['sbp'], 'fitted': results.fittedvalues,
                            'residual': results.resid})
display(diagnostics.head(3).round(1))

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(results.fittedvalues, results.resid)
ax.axhline(0, color='gray', linestyle='--')
ax.set_xlabel('Fitted SBP (mmHg)')
ax.set_ylabel('Residual (observed - fitted)')
fig.savefig('residuals_vs_fitted.png', dpi=150, bbox_inches='tight')
plt.show()
plt.close(fig)
```

|  | observed | fitted | residual |
| --- | --- | --- | --- |
| 0 | 145.3 | 139.4 | 5.9 |
| 1 | 144.5 | 145.1 | -0.6 |
| 2 | 139.6 | 142.0 | -2.4 |

The plot is the left panel of the residuals figure above: no curve, no funnel.

![xkcd 539: Boyfriend. One clear outlier on a box plot makes a statistically significant other.](media/xkcd_539.png)

# Prediction: Features, Targets, and Honest Splits

- **Prediction**: estimating an unknown outcome, such as _this_ patient's SBP at the next visit. Predicting known patients' future readings and predicting entirely new patients are different tasks; to evaluate the second, keep each held-out patient's rows out of training.
- A model can always describe the rows it was fitted on, so its error there says nothing about the next patient; a prediction model is judged only on rows kept out of its fitting.

## Feature Availability

- The **features** are the input columns the model uses (age, BMI, today's SBP), together called `X`.
- The **target** is the column to predict (next-visit SBP), called `y`; its **target time** is when that value is measured.
- The **prediction unit** receives one prediction (one visit); **prediction time** is when it is made, and every feature must be known by then.
- **Leakage** is information unavailable at prediction time sneaking into training, such as a lab result that arrives 24 hours _after_ the visit.

Keep a candidate feature only if it is known by prediction time: `candidates['resulted_at'] <= prediction_time`, or `candidates['hours_after_visit'] <= 0` for offsets like these, with the prediction made at the end of the visit:

| Candidate feature | Becomes known | Hours after the visit | Decision |
| --- | --- | --- | --- |
| `age` | Before the visit | 0 | Keep |
| `sbp_today` | During the visit | 0 | Keep |
| `a1c_result` | When the lab reports the next day | 24 | Exclude (leakage) |

![Candidate features placed by when they become known: age and sbp_today by the end of the visit (keep), a1c_result 24 hours later (leakage), and the target, sbp_next_visit, 168 hours later.](media/leakage_timeline.png)

![xkcd 2169: Predictive Models. Autocomplete trained on other users' messages gives away the secret meeting; models leak information in ways nobody planned.](media/xkcd_2169.png)

## Cyclic Time Features

- Blood pressure follows a daily rhythm, so a reading's hour is a useful feature, known as soon as the row exists. As a plain number it misleads a model: 23:00 and 00:00 are an hour apart, while 23 and 0 sit at opposite ends of the range.
- The sine and cosine of `2 * np.pi * hour / 24` place each hour on a clock face instead, so hour 23 lands as close to hour 0 as hour 1 does.

![Hours 0, 1, 12, and 23: as plain numbers 23 and 0 sit 23 apart; on the (hour_cos, hour_sin) circle, 23, 0, and 1 are neighbors 0.26 apart and 12 sits opposite.](media/cyclic_hours.png)

### Reference Card: Cyclic Time Features

| Expression | Purpose & arguments | Typical output |
| --- | --- | --- |
| `df['timestamp'].dt.hour` | Hour of day from a datetime column; `.dt.dayofyear` for the yearly cycle in the last row. | Integer Series |
| `np.sin(2 * np.pi * df['hour'] / 24)` | The hour's height on the clock face; `np.pi` is the constant π, and `np.sin()` works column-wise like Lecture 03's `np.sqrt()`. | Float Series, -1 to 1 |
| `np.cos(2 * np.pi * df['hour'] / 24)` | Its side-to-side position. Pair it with the sine: either column alone gives two different hours the same value. | Float Series, -1 to 1 |
| `2 * np.pi * (df['dayofyear'] - 1) / 366` | Yearly cycle: subtract 1 because `dayofyear` starts at 1, then divide by the cycle length (7 for `.dt.dayofweek`, which starts at 0). A fixed 366 leaves a small gap from December 31 to January 1 in non-leap years. | Float Series |

### Code Snippet: Hours on a Clock Face

```python
hours = pd.DataFrame({'hour': [0, 1, 12, 23]})
hours['hour_sin'] = np.sin(2 * np.pi * hours['hour'] / 24).round(2)
hours['hour_cos'] = np.cos(2 * np.pi * hours['hour'] / 24).round(2)
display(hours)
```

|  | hour | hour_sin | hour_cos |
| --- | --- | --- | --- |
| 0 | 0 | 0.00 | 1.00 |
| 1 | 1 | 0.26 | 0.97 |
| 2 | 12 | 0.00 | -1.00 |
| 3 | 23 | -0.26 | 0.97 |

## Training, Validation, and Test Rows

To estimate performance honestly, give rows three roles:

- **Training set**: fits the model.
- **Validation set**: compares candidate models and settings.
- **Test set**: opened once, after the choice is frozen, to report final performance.

**Cross-validation** rotates the validation role through several blocks of the training rows and averages the scores, which helps when rows are scarce. With time-ordered rows, a model fit only on earlier rows scores each block (`TimeSeriesSplit`, in the next topic). The test set stays sealed.

Comparing the errors on these sets shows two failure modes:

- Overfitting: a model memorizing its training rows instead of learning patterns that carry over, like memorizing the practice exam's answers and then failing the real exam. A much lower training error than validation error can warn of it, but a changed patient mix or time period can also create a gap.
- **Underfitting**: a model too simple to capture the pattern, so both errors stay high.

![Curves fit to 30 diabetes patients' progression against BMI and scored on 402 others in the same BMI range: degree 0 underfits (train MAE 64, validation 68), degree 1 fits (52, 53), and degree 10 overfits (45, 82).](media/overfit_polynomial.png)

- Independent rows with no time order: split at random (`train_test_split`, in the next topic).
- A model that will predict the _future_: split into chronological blocks of target times:

| Target weeks (next visit) | Role | Why |
| --- | --- | --- |
| Jan 11 to Feb 8 | Training | Oldest outcomes fit the model |
| Feb 15 to Feb 22 | Validation | Next period chooses between candidates |
| Mar 1 to Mar 15 | Test | Newest outcomes, opened once |

<callout icon="⚠️" color="yellow_bg">
	## Split on the target time, not the visit date
	Assign each row by when its **target** is measured: the Feb 8 visit predicts SBP measured on Feb 15, so it belongs to validation. Splitting on the visit date puts outcomes from the validation weeks into training.
</callout>

Disjoint target periods are only the first check:

- **Prospective evaluation** (made as forecasts would be in use): every training label must already be available when the first validation prediction is made. Remove training rows whose labels arrive after that cutoff, using each label's actual arrival time when it differs from the target time; do the same before refitting for test.
- **Retrospective comparison**: a split that ignores this delay compares historical periods; it is not evidence that the fitted model could have made those forecasts in real time.

![210 weekly home readings from 30 patients by target time: 91 training rows have labels known by the first validation prediction (Feb 2 01:58), 29 arrive later and are not fitted, 60 rows validate, and 30 test.](media/target_time_split.png)

### Reference Card: Splitting on Target Time

- `df['visit_date'] + pd.Timedelta(days=7)`: Target time for a next-week target.
- `df[df['target_date'] < cutoff]`, `df[(df['target_date'] >= start) & (df['target_date'] < end)]`: Rows whose target falls before a cutoff, or inside one period.
- `df['target_utc'] < pd.Timestamp('2024-01-01', tz='America/Chicago')`: Compare aware UTC target times with a local-midnight boundary written in the local zone; pandas compares instants (06:00 UTC here), so nothing needs converting.
- `len(part)`, `part['target_date'].min()`, `part['target_date'].max()`: Each partition's size and target-time range; check before fitting.
- `fit_rows = train[train['target_date'] <= valid['visit_date'].min()]`: Keep only training labels available by the earliest validation prediction; assumes each label arrives at its target time.
- `assert fit_rows['target_date'].max() <= valid['visit_date'].min()`: Check that fitting does not read labels from after the first forecast cutoff.

### Code Snippet: A Chronological Split

`visits` holds ten weekly visits from Sunday, January 4, 2026, each with `sbp_today` and the target, `sbp_next_visit`, measured a week later. `sbp_today` is 138, 142, 135, 150, 147, 139, 144, 152, 141, 137; `sbp_next_visit` is the same list shifted one visit earlier and ends with 145.

```python
visits['target_date'] = visits['visit_date'] + pd.Timedelta(days=7)

train = visits[visits['target_date'] < '2026-02-15']
valid = visits[(visits['target_date'] >= '2026-02-15') & (visits['target_date'] < '2026-03-01')]
test = visits[visits['target_date'] >= '2026-03-01']
print(len(train), len(valid), len(test))
display(valid[['visit_date', 'target_date']])
```

```text
5 2 3
```

|  | visit_date | target_date |
| --- | --- | --- |
| 5 | 2026-02-08 | 2026-02-15 |
| 6 | 2026-02-15 | 2026-02-22 |

![xkcd 1122: Electoral Precedent. Each rule fits every past election until the next one breaks it, like an overfit model meeting new rows.](media/xkcd_1122.png)

# LIVE DEMO!

[Open Demo 1 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/10/demo/demo1_statistical_modeling.ipynb)

# scikit-learn: One Pattern for Every Model

- **scikit-learn**: a widely used Python package for prediction.
- **Estimator**: every scikit-learn model is an object that learns from training rows with `fit()` and predicts new rows with `predict()`, so once you can fit one model you can fit them all. Like a hospital lab analyzer: calibrated against standards of known concentration (`fit`), then measuring new patient samples (`predict`).
- It takes pandas objects directly: `X` is a DataFrame of features such as `df[['age', 'bmi']]`, and `y` a Series such as `df['sbp']`.

## The Estimator Pattern

Create the estimator and choose its settings, `fit()` it on training rows where `X` and `y` are both known, then `predict()` rows where only `X` is known.

![`LinearRegression().fit()` on the 45 training patients learns sbp = 80.9 + 0.59 * age + 1.09 * bmi; `predict()` then gives 3 new patients 153.3, 136.1, and 149.2 mmHg from age and BMI alone.](media/estimator_pattern.png)

### Reference Card: The Estimator Workflow

| Function / method | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `train_test_split(X, y, test_size=0.2, random_state=42)` | Split independent rows at random when they have no time order (`from sklearn.model_selection import train_test_split`); `random_state` makes the split repeat. Call it twice to carve out train, validation, and test. | `X_train, X_test, y_train, y_test` |
| `model.fit(X, y)` | Learn parameters from features and targets. | The fitted estimator (`model`) |
| `model.predict(X)` | Generate predictions for new rows. | NumPy array |
| `model.score(X, y)` | Return the estimator's default score (R² for regressors; for classifiers, accuracy, the fraction predicted correctly; both defined below). | Float |
| `cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=3), scoring=...)` | Cross-validate rows sorted by time (the splitter uses row position, not dates; `df.sort_values('target_date')` first): each fold fits a fresh copy on earlier rows and scores the next block (`from sklearn.model_selection import cross_val_score, TimeSeriesSplit`). | One score per fold (array) |

## Linear Models for Prediction

`LinearRegression` fits the same least-squares line as `statsmodels` without standard errors, p-values, or diagnostics, but it plugs into pipelines beside dozens of other model families.

- **Regularization**: a penalty that shrinks coefficients and can reduce overfitting when features are many or correlated. It treats every coefficient alike, so scale features first (`StandardScaler`, below).
- **Ridge** (L2) penalizes the sum of squared coefficients.
- **Lasso** (L1) penalizes the sum of absolute values, which can set some fitted contributions to exactly zero. The pipeline still expects every original input column.

![Coefficients for 10 diabetes features on 264 scaled training patients as alpha grows: Ridge shrinks their overall size without setting any to exactly 0 (tc from -51 to about -12 at alpha 10, while correlated ldl flips from 27 to -3), and Lasso sets some to exactly 0, keeping 7 of 10 at alpha 2.](media/ridge_lasso_paths.png)

**Regression** predicts a number, such as next-visit SBP. **Classification** predicts a category, such as readmitted or not. Despite its name, **logistic regression** is a classifier: it converts a weighted sum of features to a probability between 0 and 1, then a threshold turns that probability into a label.

```text
Age, BMI, today's SBP → linear regression → 146.2 mmHg
Discharge measurements → logistic regression → 0.72 probability → flag 1 at a 0.5 threshold
```

![Logistic regression's S-shaped curve turns a weighted sum of 0.94 into probability 0.72, which is above the 0.5 threshold, so the patient is flagged 1.](media/logistic_threshold.png)

### Reference Card: Linear Estimators

| Class / attribute | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `LinearRegression()` | Fit an unregularized linear prediction model (`from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression`). | Estimator |
| `Ridge(alpha=...)` | Fit a linear model with an L2 coefficient penalty; larger `alpha` shrinks more. | Estimator |
| `Lasso(alpha=...)` | Fit a linear model with an L1 penalty that may set coefficients to zero. | Estimator |
| `LogisticRegression(max_iter=1000)` | Linear model for yes/no targets; `predict` gives 0/1. `predict_proba(X)` returns one column per class, and `predict_proba(X)[:, 1]` is P(class 1), the probability used for flags and thresholds. | Estimator / 2-D array |
| `model.coef_` / `model.intercept_` | Read fitted slopes and intercept after `.fit(...)`. | Array / float (`LogisticRegression`: 2-D `coef_`, 1-element `intercept_`) |

## Baselines and Pipelines

- **Baseline**: the simplest honest prediction: for a number, the training mean for everyone; for time-ordered rows, **persistence**, the patient's last value again. A model that cannot beat the baseline on validation rows has learned nothing useful.
- **Transformer**: a step, such as `StandardScaler`, that reshapes columns instead of predicting. One that learns its means from validation or test rows leaks information about rows the model should see for the first time.
- **Pipeline**: bundles transformers and a model into one estimator:

```text
X_train --fit-->     [StandardScaler -> LinearRegression]   (learns scaling + coefficients)
X_valid --predict--> [same fitted steps]  --> predictions
```

### Reference Card: Baselines and Pipelines

- `DummyRegressor(strategy='mean')`: Regression baseline; predicts the training mean for every row (`from sklearn.dummy import DummyRegressor`).
- `df.groupby('patient_id')['sbp'].shift(1)`: Persistence baseline: each patient's previous reading, `NaN` on their first row.
- `scaler = StandardScaler()`, then `scaler.fit_transform(X_train)` / `scaler.transform(X_valid)`: Learn each feature's mean and scale from training rows, then reuse them unchanged on other rows; returns a NumPy array without column names (`from sklearn.preprocessing import StandardScaler`).
- `Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])`: Chains named steps; the last is the model (`from sklearn.pipeline import Pipeline`).
- `pipeline.fit(X_train, y_train)` / `pipeline.predict(X_valid)`: Fit every step on training rows only, then reuse that scaling; returns an array.
- `pipeline.named_steps['model'].coef_`: Reach one named step inside a fitted pipeline to read its coefficients, or its settings with `.get_params()`.
- `ColumnTransformer([(name, steps, columns), ...])`: Sends different columns to different preprocessing steps (`from sklearn.compose import ColumnTransformer`).
- `pipeline.named_steps['preprocess'].get_feature_names_out()`: Names of the prepared columns, such as `categorical__clinic_Marina`; use them to label coefficients.
- `SimpleImputer(strategy='median')`: Fills missing numbers with the training median; `strategy='most_frequent'` fills categories (`from sklearn.impute import SimpleImputer`).
- `OneHotEncoder(handle_unknown='ignore', sparse_output=False)`: One 0/1 column per training category; an unseen category becomes all zeros instead of an error (`from sklearn.preprocessing import OneHotEncoder`).

### Code Snippet: A Baseline and a Linear Pipeline

The 60 `clinic` patients from the OLS snippet are already in random order, so the first 45 train and the last 15 validate.

```python
features = ['age', 'bmi']
train, valid = clinic.iloc[:45], clinic.iloc[45:]

baseline = DummyRegressor(strategy='mean')
baseline.fit(train[features], train['sbp'])

pipeline = Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])
pipeline.fit(train[features], train['sbp'])

print(baseline.predict(valid[features])[:3].round(1))  # [140. 140. 140.]
print(pipeline.predict(valid[features])[:3].round(1))  # [153.3 136.1 149.2]
print(valid['sbp'].head(3).values)                     # [144.7 141.2 156.3]
```

### Code Snippet: Variation with Mixed Numeric and Categorical Columns

- `clinic_train` holds four patients' `age`, `ldl` (one missing), and `site`, and `clinic_sbp` their SBP (mmHg).
- `clinic_valid` has another missing `ldl` and a `site`, `west`, that training never saw.
- `ColumnTransformer` routes each group of columns to its own pipeline; for classification, swap the final estimator.

```python
numeric = Pipeline([('impute', SimpleImputer(strategy='median')), ('scale', StandardScaler())])
categorical = Pipeline([('impute', SimpleImputer(strategy='most_frequent')),
                        ('one_hot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocess = ColumnTransformer([('numeric', numeric, ['age', 'ldl']),
                                ('categorical', categorical, ['site'])])
model = Pipeline([('preprocess', preprocess), ('model', Ridge(alpha=1.0))])
model.fit(clinic_train, clinic_sbp)
print(model.predict(clinic_valid).round(1))  # [141.6 147.7]
```

![`display(model)` in Jupyter draws the fitted mixed-column model: 2 numeric features pass SimpleImputer and StandardScaler, 1 categorical feature passes SimpleImputer and OneHotEncoder to become 2, and all 4 feed Ridge.](media/sklearn_pipeline_diagram.png)

![xkcd 2899: Goodhart's Law. A metric that becomes a target stops measuring anything, so choose the metric before comparing models and never tune against the test score.](media/xkcd_2899.png)

## Measuring Prediction Error

- **Metric**: turns a column of errors into one number. Choose it before comparing models, and compute it the same way for the baseline, every candidate, and the final test.
- Numeric target: each error is actual minus predicted.
- For classification (here 1 = readmitted within 30 days), a **confusion matrix** counts the four outcomes:
    - True positives (TP): flagged and readmitted.
    - False positives (FP): flagged but not readmitted.
    - False negatives (FN): readmitted but not flagged.
    - True negatives (TN): neither.
- Lowering a probability threshold flags more patients: it can catch more true cases while also creating more false alarms. Choose that tradeoff on validation data, then freeze the threshold with the model.
- A fixed workload sets the threshold for you: with 100 nurse calls a week, flag the 100 highest probabilities (`probability.rank(ascending=False) <= 100` on a Series) and compare models by recall at that budget.

![A Lasso's predictions for 89 validation diabetes patients: left, predicted against actual, flatter than the dashed perfect line; right, residuals centered 14.2 points above 0.](media/validation_diagnostics.png)

### Reference Card: `sklearn.metrics`

Import each by name: `from sklearn.metrics import mean_absolute_error, r2_score`.

| Metric | Question it answers | Function | Typical output |
| --- | --- | --- | --- |
| **MAE** (mean absolute error) | Average size of a miss | `mean_absolute_error(y_true, y_pred)` | Target units (mmHg); 0 is perfect |
| **RMSE** (root mean squared error) | Like MAE, but big misses count extra | `np.sqrt(mean_squared_error(y_true, y_pred))` | Target units; always >= MAE |
| **R²** | How much better than predicting the mean of these rows? | `r2_score(y_true, y_pred)` | 1 is perfect; 0 ties the mean; **negative is worse than the mean** |
| **Accuracy** | What fraction of predictions were right? (TP + TN) / all | `accuracy_score(y_true, y_pred)` | 0 to 1 |
| **Precision** | Of the patients we flagged, how many were readmitted? TP / (TP + FP) | `precision_score(y_true, y_pred, zero_division=0)` | 0 to 1; `zero_division=0` returns 0 without a warning when nothing is flagged |
| **Recall** | Of the readmitted patients, how many did we flag? TP / (TP + FN) | `recall_score(y_true, y_pred)` | 0 to 1 |
| Confusion matrix | How many of each outcome? | `confusion_matrix(y_true, y_pred)` | 2x2 array `[[TN, FP], [FN, TP]]` |
| Per-class summary | Precision, recall, F1 (one score that balances the two), and support (number of true rows) for each class; `output_dict=True` returns a dict that `pd.DataFrame(...).T` turns into a table | `classification_report(y_true, y_pred)` | Printed table |

![The same Lasso predictions as a flag for progression above 200: TN 55, FP 5, FN 17, TP 12, so recall is 12 / 29 = 0.414 and precision is 12 / 17 = 0.706.](media/confusion_matrix.png)

![A logistic regression flagging progression above 200 on the same 89 patients: at threshold 0.5 it flags 22 and catches 15 of 29; at 0.2 it flags 42 and catches 23, with precision falling from 0.68 to 0.55.](media/threshold_tradeoff.png)

### Code Snippet: Comparing on Validation Rows

```python
for name, fitted in [('mean_baseline', baseline), ('linear_pipeline', pipeline)]:
    pred = fitted.predict(valid[features])
    mae = mean_absolute_error(valid['sbp'], pred)
    rmse = np.sqrt(mean_squared_error(valid['sbp'], pred))
    r2 = r2_score(valid['sbp'], pred)
    print(f'{name}: MAE={mae:.2f} RMSE={rmse:.2f} R2={r2:.3f}')
```

```text
mean_baseline: MAE=11.73 RMSE=14.13 R2=-0.053
linear_pipeline: MAE=7.20 RMSE=9.65 R2=0.510
```

The baseline's R² is negative because it predicts the _training_ mean (140.0), which misses the validation rows' own mean (143.2).

### Code Snippet: Cross-Validation Over Time

Cross-validate only the training and validation rows (`target_date` before 2026-03-01); the test rows stay out. Each fold fits on earlier rows and scores the next one. Scorers are "higher is better", so MAE comes back negated.

```python
dev = visits[visits['target_date'] < '2026-03-01']
cv = TimeSeriesSplit(n_splits=3)
for fit_rows, score_rows in cv.split(dev):
    print(fit_rows, score_rows)
scores = cross_val_score(LinearRegression(), dev[['sbp_today']], dev['sbp_next_visit'],
                         cv=cv, scoring='neg_mean_absolute_error')
print((-scores).round(1))
```

```text
[0 1 2 3] [4]
[0 1 2 3 4] [5]
[0 1 2 3 4 5] [6]
[4.  0.7 9.6]
```

### Code Snippet: Accuracy Hides Missed Readmissions

A model that always says "no" matches this flag's 0.625 accuracy while catching nobody, and the rarer readmissions are, the higher that accuracy climbs. Screening therefore cares about recall.

```python
actual    = [1, 0, 0, 1, 0, 0, 0, 1]
flagged   = [1, 0, 0, 0, 0, 1, 0, 0]
always_no = [0, 0, 0, 0, 0, 0, 0, 0]

for name, predicted in [('flagged', flagged), ('always_no', always_no)]:
    accuracy = accuracy_score(actual, predicted)
    precision = precision_score(actual, predicted, zero_division=0)
    recall = recall_score(actual, predicted)
    print(f'{name}: accuracy={accuracy:.3f} precision={precision:.3f} recall={recall:.3f}')

print(confusion_matrix(actual, flagged))
```

```text
flagged: accuracy=0.625 precision=0.500 recall=0.333
always_no: accuracy=0.625 precision=0.000 recall=0.000
[[4 1]
 [2 1]]
```

The matrix reads `[[TN, FP], [FN, TP]]`: one readmission caught, two missed, one false alarm.

![xkcd 534: Genetic Algorithms. A score can improve while an unmeasured consequence gets much worse; include the costs that matter.](media/xkcd_534.png)

## Permutation Importance: What Does the Model Rely On?

- A metric says how well a model predicts, not which columns it leans on.
- **Permutation importance** measures that for any fitted model or pipeline, tree models included: score it on validation rows, shuffle one feature column to break its link to the target, and score again; the bigger the drop, the more the model relied on that feature.
- It is like scrambling one lab value across a stack of charts to see how much worse a clinician's diagnoses get.
- Correlated features can share or hide importance, and reliance is not causation.

![Shuffling bmi or ltg raises a Lasso's validation MAE by about 10.5 points, bp by 3.79, and hdl by 2.90; age, ldl, and tch add 0.00 because the Lasso set them to 0.](media/permutation_importance.png)

### Reference Card: `permutation_importance`

`permutation_importance(estimator, X, y, scoring=..., n_repeats=10, random_state=...)` shuffles each column of `X` `n_repeats` times and records how much the score drops (`from sklearn.inspection import permutation_importance`).

| Argument | Purpose | Effect |
| --- | --- | --- |
| `estimator` | A fitted model or pipeline | Predictions reuse its training-fitted preprocessing |
| `X`, `y` | Validation features and target | Keeps the test set sealed |
| `scoring` | Metric; scorers are "higher is better", so MAE is `'neg_mean_absolute_error'` | Importance = increase in MAE |
| `n_repeats`, `random_state` | Shuffles per feature and seed | Reproducible mean and spread |

Returns an object whose `importances_mean` and `importances_std` hold one value per column.

### Code Snippet: Permutation Importance on Validation Rows

```python
result = permutation_importance(
    pipeline, valid[features], valid['sbp'],
    scoring='neg_mean_absolute_error', n_repeats=10, random_state=42,
)
importance = pd.DataFrame({
    'feature': features,
    'mae_increase': result.importances_mean.round(2),
    'std': result.importances_std.round(2),
})
display(importance)
```

|  | feature | mae_increase | std |
| --- | --- | --- | --- |
| 0 | age | 4.32 | 1.26 |
| 1 | bmi | 1.31 | 0.38 |

## Freeze, Then Test Once

- **Freeze** the winner validation picks: features, preprocessing, and settings stop changing.
- Refit the frozen pipeline on training + validation rows (same settings, more data), or keep the train-fitted version.
- Either way, evaluate on the test rows exactly once and report that number, disappointing or not. Going back to tweak the model would turn the test set into a second validation set.

![Validation MAE picks the Lasso (mean baseline 64.13, linear 42.94, ridge 42.62, lasso 42.45), and the frozen Lasso then scores 42.84 on the test rows, opened once.](media/freeze_test_once.png)

### Reference Card: Freezing and the Final Test

- `model.get_params(deep=False)`: The settings the model was created with; record them, and fix any `random_state`.
- `final = Pipeline([('scale', StandardScaler()), ('model', LinearRegression())])`: Recreate the frozen pipeline, same steps and settings.
- `train_valid = pd.concat([train, valid])`: Stack training and validation rows.
- `final.fit(train_valid[features], train_valid['sbp'])`: Optional refit on more data; no setting changes.
- `final.predict(test[features])`: The one test prediction, scored with the same metrics as validation.

_"Did you ever think that maybe there's more to life than being really, really, ridiculously good at machine learning?"_

!["I'm not an ambi-turner. I can't turn left. I can't turn right. But I CAN fit, predict, and score!"](media/really_really__really_ridiculously_good_looking.jpg)

# LIVE DEMO!

[Open Demo 2 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/10/demo/demo2_sklearn_prediction.ipynb)

# Trees and Forests

- **Decision tree**: predicts by asking yes/no questions about the features ("Is BMI > 27.2?", then "Is BMI > 33.1?"). The final group is a **leaf**: regression reports its training outcomes' mean; classification uses their class proportions and predicts the most common class.
- It can bend where a linear model's straight line cannot, but one tree is jumpy: change a few training rows and its questions change.

![A depth-3 regression tree for diabetes progression first asks bmi <= 27.2, then splits on BMI and BP but never age; leaf means run from 101.5 to 268.5, and one leaf holds a single patient.](media/decision_tree.png)

## From One Tree to a Forest

- **Random forest**: grows many trees, each on a random resample of the rows, then averages their numeric predictions for regression or class probabilities for classification.
- **Ensemble**: a group of models combined into one prediction; averaging many jumpy trees gives a steadier answer than any one of them: wisdom of crowds.
- Forests capture **nonlinear** (curved) relationships and **interactions**, where one feature's effect depends on another (age might matter more at high BMI), usually without scaling. Categorical text columns still need encoding.

![Decision tree: one model, one prediction. Random forest: trees trained in parallel, predictions averaged. XGBoost: trees trained in sequence, each fit to the previous error.](media/trees.webp)

The figure simplifies one detail: with `max_features` below the column count, a forest draws a fresh random subset of columns at every split, not one fixed subset per tree.

### Reference Card: Random Forests

| Class / method | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `train_test_split(..., stratify=y)` | Split a class label so each part keeps the same share of each class; without it a small split can land lopsided. | Same four parts as before |
| `RandomForestClassifier(n_estimators=..., random_state=...)` | Build a classification forest from randomized trees (`from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor`). | Estimator |
| `RandomForestRegressor(n_estimators=..., random_state=...)` | Build a regression forest from randomized trees. | Estimator |
| `max_features=...` | How many features each question may choose from. Defaults to the square root of the count for `RandomForestClassifier` but to all of them for `RandomForestRegressor`; fewer features make the trees less alike. | Setting |
| `max_depth=...`, `min_samples_split=...` | Limit how deep each tree grows and how many rows a question needs before it splits; smaller trees memorize less. | Settings |
| `n_jobs=-1` | Build trees on all CPU cores at once. | Setting |
| `model.fit(X_train, y_train)` | Fit the trees on training data. | Fitted estimator |
| `model.predict(X_valid)` | Return class labels or numeric predictions. | Array |
| `model.predict_proba(X_valid)` | Return class probabilities (classification only). | 2-D array |
| `model.feature_importances_` | Read impurity-based feature importance scores (how much each feature's questions reduced error while training). Tree models only, and measured on the training fit; `permutation_importance` measures held-out reliance for any model. | Array; not causal evidence |

### Code Snippet: Random-Forest Classification

`X_train` and `X_valid` hold 30 cell-nucleus measurements per breast biopsy (scikit-learn's breast cancer data), split with `stratify=y` into 341 training and 114 validation rows, 37% malignant in each; `y` is 1 for malignant. The test rows stay sealed.

```python
model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)
print(round(accuracy_score(y_valid, model.predict(X_valid)), 4))  # 0.9737: 111 of 114

importance = pd.Series(model.feature_importances_, index=X_train.columns, name='importance')
display(importance.sort_values(ascending=False).head(5).round(3))
```

|  | importance |
| --- | --- |
| worst concave points | 0.140 |
| worst area | 0.130 |
| mean concave points | 0.105 |
| worst perimeter | 0.099 |
| mean concavity | 0.068 |

The forest leans on the size and shape of the most irregular nuclei. `worst area` and `worst perimeter` are near copies of each other, so they split the credit one of them would get alone.

![xkcd 1885: Ensemble Model. Many slightly different runs give a spread of outcomes, like a forest's trees fit on resampled rows, though some members are stranger than others.](media/xkcd_1885.png)

# The Secret Weapon: Gradient Boosting

- **Gradient boosting**: builds trees in sequence instead of in parallel, each small tree aimed at what the ensemble so far still gets wrong (the bottom row of the trees figure).
- Boosted trees are a standard strong candidate for tables such as clinic records, and XGBoost is the widely used library for them.

_Gradient boosting is the Magnum of machine learning: a signature look, built one small correction at a time._

## How Boosting Learns

- **Hyperparameter**: a setting you choose before fitting, such as the number of trees, depth, or learning rate; the model does not learn it.
- **Learning rate**: scales each new tree's correction: at 0.1 add a tenth of that tree's fitted update, so the ensemble learns in smaller steps. The update approximates the remaining error; it does not necessarily remove a tenth of it.
- For squared-error regression each tree fits the ordinary residuals; the "gradient" in the name generalizes that idea to other **loss** functions (measures of how wrong the predictions are, such as log loss for yes/no targets). Step by step, with made-up numbers:

| Step | What Happens | Example |
|------|--------------|---------|
| 1 | The ensemble so far predicts | [5.0, 3.0, 7.0] |
| 2 | Compute the next-step targets | Residuals: [0.5, 0.2, -0.2] |
| 3 | A new tree predicts those targets | Fitted updates: [0.4, 0.3, -0.1] |
| 4 | Add the scaled update to the ensemble | With learning rate 0.1: [5.04, 3.03, 6.99] |
| 5 | Recompute targets and repeat | For N rounds, or until validation stops improving |

![Depth-1 boosted trees with learning rate 0.1 move from the mean, 152.1, toward the BMI trend in small steps: training MAE is 64.1 after 1 tree, 56.5 after 10, and 50.1 after 100.](media/boosting_stages.png)

## `XGBoost` Basics

- **XGBoost** (Extreme Gradient Boosting, "extreme" for its speed and memory engineering): follows the same `fit`/`predict` pattern as `scikit-learn`.
- **Early stopping**: ends training when validation performance stops improving; validation then helps select the model, so keep a separate test set for the one final evaluation.

![XGBoost's validation log loss on biopsy data is lowest at round 54 (0.1168); with early_stopping_rounds=10, training stops at round 64 of a 500-round budget.](media/xgboost_early_stopping.png)

### Reference Card: XGBoost

| Class / parameter | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `xgb.XGBClassifier(...)` | Build a gradient-boosted classifier (`import xgboost as xgb`). | Estimator |
| `xgb.XGBRegressor(...)` | Build a gradient-boosted regressor; accepts `random_state` and `n_jobs` like `scikit-learn` models. | Estimator |
| `model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)` | Fit trees and monitor validation data after each round. | Fitted estimator |
| `model.predict(X)` / `model.predict_proba(X)` | Return predictions or class probabilities. | Array |
| `early_stopping_rounds=10` (in the constructor) | Stop after validation performance fails to improve for 10 rounds. | Fewer trees |
| `model.evals_result()['validation_0']['logloss']` | Validation loss after each round, for plotting the early-stopping curve. | List of floats |
| `model.best_iteration` / `model.best_score` | After early stopping: the round with the best validation score (counting from 0), and that score. | Integer / float |
| `model.feature_importances_` | Summarize fitted tree usage; do not interpret as causation. | Array |

### Reference Card: XGBoost Hyperparameters

| Parameter | What it controls | Starting choice and tradeoff |
| --- | --- | --- |
| `n_estimators` | Boosting rounds | Try 50 to 200; more rounds give more opportunity to fit and overfit |
| `max_depth` | Depth of each tree | Try 3 to 6; deeper trees capture more complexity and can memorize |
| `learning_rate` | Share of each tree's correction added | Try 0.01 to 0.3; smaller steps usually need more rounds |
| `subsample` | Fraction of training rows each tree sees | Try 0.8 to 1.0; sampling can reduce overfitting, but too few rows lose signal |
| `colsample_bytree` | Fraction of features each tree sees | Try 0.8 to 1.0; sampling makes trees less alike, but can omit useful features |

These are toy starting points; choose settings on validation rows for the data and budget.

### Code Snippet: XGBoost with Early Stopping

The same biopsy split as the forest: 341 training and 114 validation rows, test rows sealed.

```python
model = xgb.XGBClassifier(n_estimators=500, max_depth=3, learning_rate=0.1,
                          subsample=0.8, colsample_bytree=0.8,
                          early_stopping_rounds=10, random_state=42)
model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=False)
print(model.best_iteration)  # 54: zero-based round with the lowest validation loss
predicted = model.predict(X_valid)  # one 0/1 label per validation row
```

![It's all about family: a boosted ensemble is a family of trees, each one covering for the last one's mistakes](media/fast_furious_family.jpg)

# Deep Learning: The Modern Frontier

- **Neural network**: a stack of **layers** of simple units.
- **Neuron**: takes a weighted sum of its inputs plus an intercept, a tiny linear regression, and bends the result with an **activation function**.
- "Deep" means several layers, so later layers combine patterns that earlier ones found. This **representation learning** lets a network learn its own features from images, text, or audio.

## How a Network Learns

- **ReLU**: keeps positive values and turns negatives into 0.
- **Sigmoid**: squashes any number into the range 0 to 1, readable as a probability.
- **Hidden layers**: the layers between the input and the output; every unit feeds every unit in the next layer.

![The biopsy network: 30 measurements feed 64 ReLU units, then 32 ReLU units, then one sigmoid unit that outputs P(malignant); right, the two activation functions.](media/neural_network.png)

- Training repeats one loop: predict, measure the loss (`binary_crossentropy` for yes/no targets), and let the **optimizer** (such as Adam) nudge every weight to reduce it.
- **Epoch**: one pass through the training rows, processed in **batches** of, say, 32.
- Representation learning pays off where useful features are hard to write by hand. On a clinic table of a few hundred rows, a linear model or boosted trees usually match a network for far less effort, and a network overfits easily, so watch the loss curves.

![A 64-32 network's training loss keeps falling, while its validation loss bottoms out at 0.0726 at epoch 20 and creeps up to 0.0903 by epoch 50.](media/keras_loss_curves.png)

## `TensorFlow`/`Keras`: The High-Level Approach

- **Keras** (`tf.keras`): TensorFlow's high-level interface: stack layers, `compile()` with a loss and an optimizer, then `fit()` and `predict()` much like `scikit-learn`.
- **Dropout**: randomly masks a fraction of units during training; all units are active when the model predicts. Like L2, it is a regularization choice to validate against a plain network, not a guarantee against overfitting.

![The plain 64-32 network's validation loss turns up after epoch 20, while with dropout and L2 the two curves fall together; both get 111 of 114 validation biopsies right (0.9737).](media/keras_dropout_curves.png)

### Reference Card: `tf.keras`

| Method / class | Purpose & arguments | Typical output |
| :--- | :--- | :--- |
| `import tensorflow as tf`, `from tensorflow import keras` | Load TensorFlow and its Keras interface under the names this card uses. | `tf`, `keras` |
| `keras.utils.set_random_seed(42)` | Seed Python, NumPy, and TensorFlow at once; pair it with `tf.config.experimental.enable_op_determinism()` when the printed numbers must repeat exactly. | None |
| `keras.Sequential([...])` | Build a linear stack of layers. | Keras model |
| `keras.layers.Input(shape=(n_features,))` | Declare the input width as the first item in `Sequential`. | Input placeholder |
| `keras.layers.Dense(units, activation=...)` | Add a fully connected layer. | Layer |
| `keras.layers.Dropout(0.3)` | During training, randomly zero 30% of the previous layer's outputs. | Layer |
| `Dense(..., kernel_regularizer=keras.regularizers.l2(0.01))` | Add an L2 penalty on that layer's weights (the Ridge idea from the linear-estimator card). The loss Keras reports then includes the penalty, so compare such a network with others on accuracy, not loss. | Layer |
| `model.summary()` / `model.count_params()` | Print each layer's output shape and parameter count / return the total number of weights. | Printed table / integer |
| `model.compile(optimizer, loss, metrics)` | Configure optimization, loss, and reported metrics. | Configured model |
| `model.fit(X_train, y_train, epochs=..., batch_size=..., validation_split=...)` | Train for epochs and optionally hold out the last part of the training rows for validation. | `History` object |
| `model.fit(..., validation_data=(X_valid, y_valid))` | Report validation loss and metrics on an explicit validation set after every epoch. | `History` object |
| `history.history` | Per-epoch values such as `loss`, `val_loss`, `accuracy`, `val_accuracy`. | dict of lists |
| `model.predict(X)` | Generate predictions; with a sigmoid output these are probabilities, so `(model.predict(X) > 0.5).astype(int).flatten()` gives 0/1 labels. | NumPy array |
| `model.evaluate(X_test, y_test)` | Calculate loss and configured metrics on held-out data. | Scalar or list |

### Code Snippet: A Small Keras Classifier

`X_train_scaled` and `X_valid_scaled` are the same biopsy measurements, scaled with training-fitted settings, with the same 114 validation rows; one hidden layer is enough to show the API.

```python
model = keras.Sequential([
    keras.layers.Input(shape=(X_train_scaled.shape[1],)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid'),
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(X_train_scaled, y_train, epochs=10, batch_size=32,
                    validation_data=(X_valid_scaled, y_valid), verbose=0)
probability = model.predict(X_valid_scaled, verbose=0).ravel()
print(len(history.history['val_loss']), probability.shape)  # 10 (114,)
```

![xkcd 2173: Trained a Neural Net. Do the task by hand and you can technically say you trained a neural net to do it.](media/xkcd_2173.png)

## Comparing Model Families

- A well-tuned linear regression often beats a poorly tuned neural network.
- Dataset, implementation, hardware, and budget rule out universal rankings, so the same validation rows decide:

![Validation accuracy on the same 114 biopsies: every model beats the majority-class baseline (0.6228, 71 of 114), and logistic regression ties the forest and two networks at 0.9737 (111 of 114).](media/model_comparison.png)

| Model family | Useful role in a shortlist | Potential strengths | Check before choosing |
|---|---|---|---|
| Linear models | Simple baseline or inference model | Fast, compact, often easy to explain | Functional form and statistical assumptions |
| Random forests | Nonlinear tabular candidate | Interactions, limited preprocessing, robust baseline | Latency, **calibration** (whether predicted probabilities match observed rates, e.g. 30% of patients scored 0.3 are readmitted), and explanation needs |
| Gradient-boosted trees | Tabular prediction candidate | Flexible nonlinear fits and strong empirical performance | Tuning, calibration, and validation stability |
| Deep neural networks | Representation-learning candidate | Flexible architectures for images, text, audio, and other complex inputs | Data, compute, deployment, and explanation requirements |

![xkcd 2400: Statistics. When the data are good enough, the answer is obvious without any statistics; better data settles what no model choice can.](media/xkcd_2400.png)

# LIVE DEMO!

[Open Demo 3 in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/10/demo/demo3_trees_boosting_networks.ipynb)
