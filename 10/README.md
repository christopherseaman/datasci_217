# Honest Inference and Predictive Evaluation

A **model** is a simplified mathematical description of a relationship in data. A model can help answer a carefully framed question, but it cannot repair an unclear unit, unavailable information, or an invalid evaluation design. Lecture 10 therefore focuses on one bounded workflow: frame the claim, fit only what the question requires, and keep the final evaluation honest.

Optional hypothesis testing, regularization, split-aware cross-validation, and one bounded tree-ensemble extension are collected in [BONUS.md](BONUS.md). Boosting, deep learning, framework surveys, deployment systems, and time-series forecasting are not required Lecture 10 capabilities.

## Prerequisites

Before starting this lecture, students should be able to:

- build a validated analysis table with a stated row grain and stable row key;
- inspect missingness, types, categories, and provenance without silently changing the unit;
- create and critique a clearly labeled scatter or line chart;
- distinguish a single temporal series from a panel and sort within entity;
- create past-only lags or windows and state whether their inputs are available at a supplied prediction timestamp; and
- restart and run a notebook from top to bottom in Colab or local Jupyter.

Lecture 10 does not assume prior model fitting, statistical inference, causal inference, train/validation/test roles, preprocessing pipelines, or classification-model training.

## Learning objectives

By the end of Lecture 10, students should be able to:

1. Classify a question as descriptive, inferential, or predictive; state the unit and intended claim; and distinguish an observed association from a causal claim.
2. For an inferential question, state a population estimand, fit one OLS association model, interpret one coefficient and confidence interval conditionally, distinguish a mean-response confidence interval from an individual prediction interval, and name the assumptions and residual diagnostic that limit the claim.
3. For a predictive question, define the target, target timestamp, prediction horizon, features, feature cutoff and availability, and a simple baseline before fitting.
4. Create disjoint training, validation, and test partitions using a seeded random split for exchangeable rows or chronological cutoffs for future prediction; fit preprocessing only on training data in one scikit-learn `Pipeline`; and identify target, temporal, preprocessing, or test-set leakage.
5. Use `fit` and `predict` to compare the `Pipeline` with its baseline using MAE, RMSE, and R²; interpret supplied binary-classification accuracy, precision, and recall against a baseline; evaluate the test set once; and report uncertainty and limitations.

## Colab-first execution and evidence

Required Lecture 10 demonstrations are Colab-first and must also run in clean local Jupyter or the VS Code notebook interface. The existing course candidates remain Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. Exact compatible versions of statsmodels, scikit-learn, Matplotlib, notebook execution tools, and their transitive dependencies are not a release lock until the complete local/Colab certification pass succeeds.

Required demonstrations use a pinned small dataset or deterministic supplied data. They do not fetch a dataset at runtime, mount Drive, require a manual upload, or use stored notebook output as evidence. Restart the runtime and run every cell in order. Changes made in a Colab notebook opened from GitHub are not automatically saved back to the repository.

Assignment notebooks must run cleanly in local Jupyter. Colab becomes an assignment submission path only after the repository-save and Classroom 50 pilot is approved.

## Start with the question and the unit

The first decision is not which library to import. It is what claim the analysis is supposed to support.

- A **descriptive question** summarizes the rows actually observed. Example: “What was the median wait in this recorded sample?”
- An **inferential question** uses a sample to learn about a broader population or process. Example: “In the intended participant population, what is the conditional association between study hours and assessment score?”
- A **predictive question** asks for an unknown value for a new case or later time. Example: “Using information available now, how accurately can we predict tomorrow's temperature?”

A **sample** is the set of observed units. A **population** is the broader set or process the inferential claim concerns. An **estimand** is the exact population quantity to be estimated. A **prediction target** is the unknown value a predictive procedure will try to produce for each case.

The same variables can support different questions, so every question contract states:

1. the unit represented by one row;
2. the sample or cases in scope;
3. the descriptive quantity, inferential estimand, or prediction target;
4. the intended claim; and
5. the decisions or uses that remain outside scope.

| Question type | Required contract | Bounded claim |
|---|---|---|
| Descriptive | observed rows, unit, summary | describes only the recorded data |
| Inferential | sample, population, estimand, assumptions | estimates a population quantity under stated conditions |
| Predictive | cases, target, issue time, horizon, available features, metric | estimates performance for the stated prediction setting |

## Association is not causation

An **association** is a pattern in which values of one variable differ with values of another. A **causal claim** says that changing one variable would change an outcome. A fitted regression coefficient can describe a conditional association, but it does not become causal because it is precise, statistically unusual, or produced by sophisticated software.

Causal interpretation requires a causal question plus a design and assumptions that justify the comparison. Randomized assignment can sometimes supply that design. Observational data usually require additional domain knowledge and defensible assumptions about common causes, selection, and measurement. Those topics belong to a dedicated causal-inference treatment.

Lecture 10 uses noncausal language:

- “is associated with,” not “causes”;
- “holding the included variables fixed,” not “all else equal in the world”; and
- “under this model and sampling process,” not “proven.”

## A bounded OLS association model

**Ordinary least squares**, abbreviated **OLS**, fits a linear conditional-mean relationship. A **conditional mean** is the population mean response among units with specified explanatory-variable values. The **response** is the numeric variable being modeled. An **explanatory variable** is an included variable used to describe how that response differs. The **intercept** is the fitted response when all explanatory variables equal zero, if that reference point is meaningful. A **coefficient** is the fitted change in the mean response associated with a one-unit change in one explanatory variable while the other included variables are held fixed.

For two explanatory variables, the model is written:

```text
expected response = intercept
                  + coefficient_1 × explanatory_variable_1
                  + coefficient_2 × explanatory_variable_2
```

A **fitted value** is the model's value for one observed row. A **residual** is `observed response - fitted value` for that row. OLS selects coefficients that minimize the sum of squared residuals. An **error** is the unobserved difference between a response and its population conditional mean; a residual is the sample's fitted estimate of that difference.

The deterministic teaching sample below has grain one synthetic workshop participant. Its inferential question is: in a hypothetical population represented by this sampling process, what is the conditional association between study hours and assessment score after accounting for prior score? The target estimand is the population coefficient on study hours in the stated linear model. Because these are teaching data rather than a probability sample from a real population, the example demonstrates mechanics, not a real-world population conclusion.

```python
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

inference_data = pd.DataFrame(
    {
        "study_hours": [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 8, 9],
        "prior_score": [58, 61, 67, 63, 70, 74, 69, 76, 82, 79, 85, 88],
        "assessment_score": [65, 69, 72, 71, 78, 80, 79, 86, 90, 88, 94, 98],
    }
)

assert inference_data.shape == (12, 3)
```

In the statsmodels formula interface, `response ~ variable_1 + variable_2` places the response to the left of `~` and included explanatory variables to the right. Calling `.fit()` estimates the coefficients from the supplied rows.

### Assumptions come before interpretation

An **assumption** is a condition under which a method's calculation supports its intended interpretation. For the conventional OLS coefficient intervals used here, the important conditions include:

- the conditional mean is reasonably represented by the stated linear form;
- observations are independent, or dependence has been handled by the design and uncertainty method;
- the residual variance is reasonably stable for conventional standard errors;
- no explanatory variable is an exact linear combination of the others;
- the sample and measurement process are relevant to the intended population; and
- a noncausal association is the intended claim.

For exact small-sample conventional intervals, the error distribution also needs an appropriate shape. Large samples can make some calculations less sensitive to that condition, but they do not repair dependence, selection bias, measurement error, misspecified relationships, or causal overclaim.

```python
ols_result = smf.ols(
    "assessment_score ~ study_hours + prior_score",
    data=inference_data,
).fit()

assert list(ols_result.params.index) == [
    "Intercept",
    "study_hours",
    "prior_score",
]
assert np.isclose(ols_result.resid.mean(), 0.0, atol=1e-10)
```

### Coefficient uncertainty

A **standard error** estimates how much a coefficient estimate would vary across repeated samples under the model and sampling assumptions. A **95% confidence interval** comes from a procedure designed to contain the target coefficient in 95% of repeated samples under those assumptions. It is not the probability that this already-computed interval contains a fixed coefficient, and it does not measure causal credibility.

```python
coefficient_intervals = ols_result.conf_int(alpha=0.05)
coefficient_intervals.columns = ["lower", "upper"]

study_hours_summary = pd.Series(
    {
        "estimate": ols_result.params["study_hours"],
        "standard_error": ols_result.bse["study_hours"],
        "lower": coefficient_intervals.loc["study_hours", "lower"],
        "upper": coefficient_intervals.loc["study_hours", "upper"],
    }
)

assert study_hours_summary["lower"] < study_hours_summary["estimate"]
assert study_hours_summary["estimate"] < study_hours_summary["upper"]
study_hours_summary
```

A bounded interpretation is: among units described by this model, and holding prior score fixed, one additional study hour is associated with an estimated `study_hours`-coefficient increase in mean assessment score. The interval describes coefficient uncertainty under the stated assumptions. It is not an intervention effect and should not be generalized beyond a relevant population and measurement process.

### Residual diagnostic

A **residual plot** places residuals against fitted values. Curvature can warn that the mean relationship is not adequately linear. A funnel shape can warn that residual spread changes with the fitted value. Isolated large residuals can identify rows needing a data or influence review. A quiet-looking plot cannot prove that the assumptions hold.

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.scatter(ols_result.fittedvalues, ols_result.resid)
ax.axhline(0, color="black", linewidth=1)
ax.set(
    title="Residual check for the bounded OLS model",
    xlabel="Fitted assessment score",
    ylabel="Residual: observed minus fitted",
)
```

### Mean-response confidence interval versus individual prediction interval

For supplied explanatory values, a **mean-response confidence interval** describes uncertainty about the population mean response at those values. An **individual prediction interval** describes uncertainty for one new individual response at those values. The individual interval is wider because it includes both uncertainty about the mean and individual variation around that mean.

```python
new_case = pd.DataFrame(
    {
        "study_hours": [5.0],
        "prior_score": [75.0],
    }
)

intervals = ols_result.get_prediction(new_case).summary_frame(alpha=0.05)

mean_response_width = (
    intervals.loc[0, "mean_ci_upper"] - intervals.loc[0, "mean_ci_lower"]
)
individual_width = (
    intervals.loc[0, "obs_ci_upper"] - intervals.loc[0, "obs_ci_lower"]
)

assert individual_width > mean_response_width
intervals[
    [
        "mean",
        "mean_ci_lower",
        "mean_ci_upper",
        "obs_ci_lower",
        "obs_ci_upper",
    ]
]
```

Neither interval licenses extrapolation far outside the observed explanatory-variable range. Both remain conditional on the model, data process, and included variables.

## LIVE DEMO 1: Frame a question and bounded inference

[Open the Lecture 10 demo guide](demo/DEMO_GUIDE.md).

The first required demonstration starts from a pinned dataset with a visible generating relationship. It states the unit, population estimand, and noncausal claim; defines coefficient, residual, standard error, confidence interval, and prediction interval; fits one OLS model; interprets one coefficient conditionally; checks one residual plot; and places the mean-response and individual intervals side by side.

## Define a prediction contract before features

A predictive workflow needs more than a target column:

- The **prediction timestamp** is when the prediction is issued.
- The **target** is the unknown value to be predicted.
- The **target timestamp** is when that value is defined or measured.
- The **prediction horizon** is the span between the prediction timestamp and target timestamp.
- A **feature** is an input supplied to the prediction procedure.
- The **feature cutoff** is the latest permitted time for feature inputs.
- A feature is **available** only if every source value and processing step needed to compute it would exist by the cutoff.
- A **metric** is a numeric rule for summarizing prediction errors or decisions; the primary metric is chosen before comparing candidates.

The contract below predicts one station's next-day temperature. Its unit is one station at one daily prediction timestamp. The target is tomorrow's temperature, the horizon is one day, and only information known by today's timestamp is allowed.

```python
prediction_contract = pd.DataFrame(
    {
        "field": [
            "unit",
            "prediction timestamp",
            "target",
            "target timestamp",
            "horizon",
            "feature cutoff",
            "primary metric",
        ],
        "definition": [
            "one station at one daily issue time",
            "current day at 00:00 UTC",
            "next-day temperature in degrees C",
            "prediction timestamp plus one day",
            "one day",
            "prediction timestamp",
            "mean absolute error",
        ],
    }
)

assert prediction_contract.shape == (7, 2)
```

Calendar values, current measurements, and past-only lags can be candidates when they are available by the cutoff. A centered window, a measurement recorded after the cutoff, or a summary fit on the completed dataset is unavailable even if it appears in a historical table.

The deterministic teaching table makes the time contract executable. It is synthetic and represents one station only; it is not evidence about a real forecasting system.

```python
all_timestamps = pd.date_range(
    "2026-01-01",
    periods=42,
    freq="D",
    tz="UTC",
)
all_day_numbers = np.arange(len(all_timestamps), dtype=float)
all_temperatures = (
    10
    + 0.15 * all_day_numbers
    + 2 * np.sin(all_day_numbers / 3)
    + 0.4 * np.cos(all_day_numbers * 1.7)
)

prediction_data = pd.DataFrame(
    {
        "row_id": [
            f"station-a-{timestamp:%Y%m%d}"
            for timestamp in all_timestamps[1:-1]
        ],
        "prediction_timestamp": all_timestamps[1:-1],
        "target_timestamp": all_timestamps[2:],
        "day_number": all_day_numbers[1:-1],
        "current_temperature_c": all_temperatures[1:-1],
        "previous_temperature_c": all_temperatures[:-2],
        "target_next_day_temperature_c": all_temperatures[2:],
    }
)

assert prediction_data.shape == (40, 7)
assert (
    prediction_data["target_timestamp"]
    - prediction_data["prediction_timestamp"]
).eq(pd.Timedelta(days=1)).all()
```

## Assign training, validation, and test roles

The three data roles are different:

- **Training data** fit coefficients and preprocessing state.
- **Validation data** compare candidate approaches or fixed settings during development.
- **Test data** estimate final performance after all choices are fixed. They are evaluated once.

Rows are **exchangeable** for a split when their ordering is not part of the intended prediction setting and no entity, family, location, or time dependence would make a random rearrangement change the problem. Independent one-time records from different units may support a seeded random split. Repeated entities, spatial clusters, or future prediction require a split that preserves those boundaries.

A **random seed** is a fixed input to a pseudorandom procedure that makes the same split membership reproducible. In scikit-learn, `random_state=217` records that seed; it does not make an inappropriate random split valid.

For 30 genuinely exchangeable row IDs, a reproducible two-stage split can create 18 training, 6 validation, and 6 test IDs:

```python
from sklearn.model_selection import train_test_split

exchangeable_ids = np.arange(30)
development_ids, random_test_ids = train_test_split(
    exchangeable_ids,
    test_size=0.20,
    random_state=217,
)
random_train_ids, random_validation_ids = train_test_split(
    development_ids,
    test_size=0.25,
    random_state=217,
)

assert len(random_train_ids) == 18
assert len(random_validation_ids) == 6
assert len(random_test_ids) == 6
assert not (
    set(random_train_ids)
    & set(random_validation_ids)
    | set(random_train_ids)
    & set(random_test_ids)
    | set(random_validation_ids)
    & set(random_test_ids)
)
```

The next-day station question is not exchangeable because the intended use predicts later dates. It uses fixed chronological cutoffs. Splitting on the target timestamp also prevents a training row whose label occurs inside the later evaluation period.

```python
validation_start = pd.Timestamp("2026-01-25", tz="UTC")
test_start = pd.Timestamp("2026-02-01", tz="UTC")

prediction_data["split"] = np.select(
    [
        prediction_data["target_timestamp"].lt(validation_start),
        prediction_data["target_timestamp"].lt(test_start),
    ],
    ["train", "validation"],
    default="test",
)

split_manifest = prediction_data[
    ["row_id", "prediction_timestamp", "target_timestamp", "split"]
].copy()

assert split_manifest["row_id"].is_unique
assert split_manifest["split"].value_counts().to_dict() == {
    "train": 22,
    "validation": 7,
    "test": 11,
}
assert prediction_data.loc[
    prediction_data["split"].eq("train"), "target_timestamp"
].max() < prediction_data.loc[
    prediction_data["split"].eq("validation"), "target_timestamp"
].min()
assert prediction_data.loc[
    prediction_data["split"].eq("validation"), "target_timestamp"
].max() < prediction_data.loc[
    prediction_data["split"].eq("test"), "target_timestamp"
].min()
```

A **split manifest** is a table that records the stable row ID and assigned role. It makes overlap, chronology, and accidental reassignment testable.

## Recognize leakage before fitting

**Leakage** occurs when training, model choice, or evaluation uses information that would not be available in the intended workflow.

- **Target leakage:** a feature directly or indirectly contains the outcome being predicted.
- **Temporal leakage:** a feature uses observations after its feature cutoff.
- **Preprocessing leakage:** a transformation learns means, scales, categories, imputations, or other state from validation, test, or full data before evaluation.
- **Test-set leakage:** test results influence feature choice, model choice, settings, stopping, or repeated revision.

Low correlation does not prove that a feature is available. High correlation does not prove leakage. Availability comes from timestamps, source lineage, and the intended workflow.

## LIVE DEMO 2: Choose a split and audit leakage

[Open the Lecture 10 demo guide](demo/DEMO_GUIDE.md).

The second required demonstration defines a target, prediction timestamp, horizon, feature cutoff, and availability inventory before fitting. It contrasts a seeded random split for genuinely exchangeable rows with fixed chronological cutoffs for future prediction, saves disjoint row IDs, and rejects one post-outcome feature and one full-data preprocessing path as explicit leakage cases.

## Fit preprocessing and a model together

An **estimator** is an object that learns from data. Calling `fit()` learns its state from supplied training rows. Calling `predict()` applies the already-fitted state to new rows. **Preprocessing** transforms inputs before model fitting, such as centering and scaling numeric features.

A scikit-learn **Pipeline** chains preprocessing and an estimator so both receive the correct partitions in the correct order. Fitting the Pipeline on training rows makes `StandardScaler` learn only training means and standard deviations. Calling `transform()` separately on the full dataset before splitting would leak information.

A **baseline** is a simple reference procedure that a learned model must improve on to justify its complexity. For this regression example, `DummyRegressor(strategy="mean")` predicts the training-target mean for every case.

```python
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

feature_columns = [
    "day_number",
    "current_temperature_c",
    "previous_temperature_c",
]
target_column = "target_next_day_temperature_c"

parts = {
    name: prediction_data.loc[prediction_data["split"].eq(name)].copy()
    for name in ["train", "validation", "test"]
}

baseline = DummyRegressor(strategy="mean")
linear_pipeline = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("model", LinearRegression()),
    ]
)

baseline.fit(parts["train"][feature_columns], parts["train"][target_column])
linear_pipeline.fit(
    parts["train"][feature_columns],
    parts["train"][target_column],
)

assert np.allclose(
    linear_pipeline.named_steps["scale"].mean_,
    parts["train"][feature_columns].mean().to_numpy(),
)
```

## Compare on validation with named metrics

**Mean absolute error**, or **MAE**, is the mean absolute difference between targets and predictions. It remains in target units and weights every absolute error linearly.

**Root mean squared error**, or **RMSE**, is the square root of the mean squared error. It remains in target units but gives larger errors more influence than MAE.

**R²** compares squared error with a constant reference based on the evaluation targets. `1` is exact prediction, `0` matches that evaluation-set mean reference, and negative values are possible when predictions are worse. R² is not “percent correct” and should not be the only metric.

Choose a primary metric before comparing candidates. The prediction contract above names MAE.

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(actual, predicted):
    """Return the three named regression metrics."""
    return {
        "mae": mean_absolute_error(actual, predicted),
        "rmse": np.sqrt(mean_squared_error(actual, predicted)),
        "r2": r2_score(actual, predicted),
    }


candidates = {
    "training_mean_baseline": baseline,
    "linear_pipeline": linear_pipeline,
}
validation_metrics = {}

for name, candidate in candidates.items():
    validation_prediction = candidate.predict(
        parts["validation"][feature_columns]
    )
    validation_metrics[name] = regression_metrics(
        parts["validation"][target_column],
        validation_prediction,
    )

validation_metrics = pd.DataFrame(validation_metrics).T
chosen_name = validation_metrics["mae"].idxmin()

assert chosen_name == "linear_pipeline"
validation_metrics
```

The selection is based on validation MAE. The test rows have not been predicted or inspected during this choice.

## Evaluate the chosen approach on test once

An **evaluation** applies fixed choices to data that were not used for fitting or selection. After the model, features, preprocessing, split, and metric are fixed, evaluate the chosen approach on the test set once.

```python
chosen_model = candidates[chosen_name]
final_test_predictions = chosen_model.predict(parts["test"][feature_columns])
final_test_metrics = pd.Series(
    regression_metrics(
        parts["test"][target_column],
        final_test_predictions,
    ),
    name="test",
)

final_predictions = parts["test"][
    ["row_id", "target_timestamp", target_column]
].copy()
final_predictions["prediction"] = final_test_predictions

assert len(final_predictions) == len(parts["test"])
assert final_predictions["row_id"].is_unique
assert final_test_metrics["mae"] < 1.0
final_test_metrics
```

Do not return to development and then report the same test result as if it were untouched. If the test result triggers a redesign, that result becomes development evidence; a new final evaluation requires a genuinely untouched test release.

## Interpret supplied binary metrics

A **binary classification** assigns one of two labels. The **positive class** is the outcome whose detection is being counted. For positive label `1`:

- a **true positive** is an actual positive predicted positive;
- a **false positive** is an actual negative predicted positive;
- a **false negative** is an actual positive predicted negative;
- **accuracy** is the proportion of all predictions that are correct;
- **precision** is `true positives / all predicted positives`; and
- **recall** is `true positives / all actual positives`.

The small table below is supplied prediction output. It is not a second model-fitting exercise. The dummy column represents output from a training-only most-frequent-class baseline.

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score

binary_predictions = pd.DataFrame(
    {
        "actual": [1, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        "supplied_model_prediction": [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        "supplied_dummy_prediction": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    }
)


def binary_metrics(actual, predicted):
    """Return accuracy, precision, and recall for positive label 1."""
    return {
        "accuracy": accuracy_score(actual, predicted),
        "precision": precision_score(actual, predicted, zero_division=0),
        "recall": recall_score(actual, predicted, zero_division=0),
    }


binary_summary = pd.DataFrame(
    {
        "supplied_model": binary_metrics(
            binary_predictions["actual"],
            binary_predictions["supplied_model_prediction"],
        ),
        "dummy_baseline": binary_metrics(
            binary_predictions["actual"],
            binary_predictions["supplied_dummy_prediction"],
        ),
    }
).T

assert np.isclose(binary_summary.loc["supplied_model", "accuracy"], 0.8)
assert np.isclose(binary_summary.loc["dummy_baseline", "accuracy"], 0.8)
assert np.isclose(binary_summary.loc["supplied_model", "precision"], 0.5)
assert np.isclose(binary_summary.loc["supplied_model", "recall"], 0.5)
assert binary_summary.loc["dummy_baseline", "recall"] == 0.0
binary_summary
```

Both approaches have accuracy `0.8`, but the dummy baseline detects none of the actual positives. The supplied model detects one of two positives and half of its positive predictions are correct. Which tradeoff matters depends on the intended decision and the relative consequences of false positives and false negatives.

## LIVE DEMO 3: Compare a baseline and one train-only Pipeline

[Open the Lecture 10 demo guide](demo/DEMO_GUIDE.md).

The third required demonstration fits a training-mean baseline and one `Pipeline(StandardScaler, LinearRegression)` on training rows only. It selects by validation MAE, reports validation MAE/RMSE/R², evaluates test once, saves predictions and one familiar residual plot, and interprets supplied binary accuracy/precision/recall against a dummy baseline without fitting a second classifier.

## Communicate uncertainty and limitations

An honest result separates what was calculated from what remains uncertain.

For bounded inference, report:

- the unit, population, and estimand;
- the coefficient and confidence interval;
- the included variables and residual diagnostic;
- the association-only interpretation; and
- the sampling, measurement, model-form, and generalizability limitations.

For prediction, report:

- the target, target timestamp, horizon, cutoff, and available features;
- the split design and baseline;
- the primary validation metric and one final test result;
- the unit and range of every metric;
- where the evaluation sample differs from intended use; and
- likely failure modes, subgroup or time-slice gaps, and possible distribution change.

A coefficient confidence interval is not an interval for future model accuracy. One held-out metric is not a universal performance guarantee. Both are conditional on the data-generating and use conditions being relevant.

## Handoff to Lecture 11

After this lecture, students should be able to:

- distinguish descriptive, inferential, and predictive questions;
- state a unit and either a population estimand or prediction target;
- use noncausal language for observational associations;
- interpret one bounded OLS coefficient and confidence interval and inspect residuals;
- distinguish mean-response and individual prediction intervals;
- define a prediction timestamp, target timestamp, horizon, feature cutoff, and availability;
- choose a seeded random split only for exchangeable rows and a chronological split for future prediction;
- separate training, validation, and test roles;
- fit preprocessing only on training data inside a Pipeline;
- compare a training-derived baseline with one linear predictor using named metrics;
- interpret supplied binary accuracy, precision, and recall against a baseline;
- recognize target, temporal, preprocessing, and test-set leakage; and
- evaluate test once and communicate limitations.

Lecture 11 may apply this complete modeling vocabulary to a frozen end-to-end project. It must not introduce boosting, deep learning, hyperparameter-search breadth, or feature-importance theory as new required capabilities.

## Core scope boundary

Required Lecture 10 work is limited to question type and unit; estimand or prediction contract; association versus causation; one bounded OLS association model; coefficient, standard error, confidence interval, individual prediction interval, assumptions, and one residual diagnostic; seeded random versus chronological splitting; training/validation/test roles; availability and four leakage types; one training-mean baseline; one train-only linear Pipeline; MAE, RMSE, R²; supplied binary accuracy, precision, and recall; one final test evaluation; and limitations.

Hypothesis testing and p-values, regularization, split-aware cross-validation, one tree ensemble, and held-out permutation importance are optional bonus material. Interactions, broad model-selection catalogues, XGBoost and other boosting libraries, deep learning frameworks, neural-network training, deployment platforms, automated feature engineering, time-series forecasting, and production monitoring are not required demos, assignment capabilities, or Lecture 11 prerequisites.
