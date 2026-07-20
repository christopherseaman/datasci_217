# Optional Extensions for Honest Modeling

This bonus material extends Lecture 10 only after the required question contract, bounded OLS interpretation, split roles, train-only preprocessing, baseline comparison, and untouched final evaluation are secure. None of these topics is a required demonstration, Assignment 10 capability, or Lecture 11 prerequisite.

Treat every executable extension below as an alternative development branch chosen before final evaluation. Do not inspect a test result, return to model development, and then reuse that same test result as final evidence. A redesigned workflow needs a genuinely untouched final test release.

## Scope boundary

The core lecture owns:

- descriptive, inferential, and predictive question contracts;
- population estimand versus prediction target;
- association versus causation;
- one OLS association model with assumptions, coefficient uncertainty, residuals, and mean-response versus individual intervals;
- target timestamp, horizon, cutoff, and information availability;
- training, validation, and test roles;
- train-only preprocessing in one linear Pipeline;
- a baseline, regression metrics, supplied binary metrics, leakage checks, and one final test evaluation.

This file adds bounded extensions:

- null hypotheses and p-values as optional inferential vocabulary;
- Ridge and Lasso regularization inside a Pipeline;
- cross-validation whose folds match the deployment structure; and
- one tree ensemble with held-out permutation importance.

The former surveys of XGBoost, LightGBM, CatBoost, TensorFlow, Keras, PyTorch, JAX, stacking, Bayesian optimization, automated feature engineering, deployment frameworks, drift tests, and time-series forecasting are not retained. Each requires its own prerequisites, evaluation design, environment, and justified course purpose.

## Hypothesis tests and p-values

A **null hypothesis** is a precisely stated reference claim about a population quantity, such as a coefficient being zero in the specified model. A **test statistic** measures how far the sample estimate is from that reference relative to its estimated uncertainty. A **p-value** is the probability, under the null hypothesis and all test assumptions, of obtaining a test statistic at least as incompatible with the null as the observed one.

A p-value is not:

- the probability that the null hypothesis is true;
- the probability that the result happened “by chance”;
- the size or practical importance of an association;
- evidence of causation; or
- a substitute for data-quality, design, and assumption review.

If a project uses a hypothesis test, define the estimand and null before examining the result. Report the coefficient, confidence interval, and context rather than reducing the conclusion to whether a universal threshold was crossed. Multiple testing, selective reporting, dependence, and model selection can invalidate a naive interpretation and require a more advanced design.

The core OLS result exposes a coefficient p-value through `ols_result.pvalues["study_hours"]`, but no required work interprets or grades it.

## Regularization inside a Pipeline

**Regularization** adds a penalty that discourages large fitted coefficients. A **hyperparameter** is a setting chosen outside ordinary coefficient fitting. Ridge uses an L2 squared-coefficient penalty; Lasso uses an L1 absolute-coefficient penalty and can set some fitted coefficients to zero.

Scaling and regularization must remain inside a Pipeline so each candidate learns preprocessing from training rows only. Choose the penalty strength with validation data or a split-aware cross-validation design, never with final test results.

```python
from sklearn.linear_model import Lasso, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ridge_candidate = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=1.0)),
    ]
)
lasso_candidate = Pipeline(
    steps=[
        ("scale", StandardScaler()),
        ("model", Lasso(alpha=0.05, max_iter=10_000)),
    ]
)
```

Coefficient shrinkage does not turn coefficients into causal effects. Lasso selecting a variable does not prove that the variable is important in the world; the selected set can change with correlated features, scaling, sampling, and penalty choice.

## Split-aware cross-validation

**Cross-validation** repeatedly divides development data into training and validation **folds** so performance is summarized across several held-out parts. It can reduce dependence on one validation split, but it does not replace a final untouched test set.

The fold rule must match the intended use:

- `KFold` can be reasonable for independent exchangeable rows;
- `GroupKFold` keeps every supplied entity group in only one fold; and
- `TimeSeriesSplit` preserves order for a single ordered development sequence.

These names are not interchangeable recipes. Repeated entities may require grouping and time order together; a simple built-in splitter may not express both. A gap may also be necessary when feature windows or delayed labels would otherwise cross a fold boundary.

```python
from sklearn.model_selection import GroupKFold, KFold, TimeSeriesSplit

exchangeable_folds = KFold(n_splits=5, shuffle=True, random_state=217)
entity_folds = GroupKFold(n_splits=5)
ordered_folds = TimeSeriesSplit(n_splits=5)
```

Fit the complete Pipeline inside every fold. Fitting a scaler, imputer, feature selector, or encoder once on all development rows before cross-validation leaks fold information.

## One bounded tree-ensemble extension

A **tree ensemble** combines predictions from several decision trees. It can represent nonlinear relationships and interactions that a linear model does not. Optional exploration may fit one `RandomForestRegressor` as a validation candidate after the baseline and linear Pipeline are established.

This extension is not permission to compare an open-ended model catalogue on the test set. The ensemble is fit on training rows, compared on validation rows with the predeclared primary metric, and considered only if its extra complexity serves the stated use.

```python
from sklearn.ensemble import RandomForestRegressor

forest_candidate = RandomForestRegressor(
    n_estimators=200,
    min_samples_leaf=5,
    random_state=217,
    n_jobs=1,
)
```

### Held-out permutation importance

**Permutation importance** measures how much a fitted model's score worsens when one feature column is shuffled on held-out data. Shuffling breaks that feature's relationship with the target while leaving the fitted model unchanged.

```python
from sklearn.inspection import permutation_importance

# Fit only on training rows.
forest_candidate.fit(
    parts["train"][feature_columns],
    parts["train"][target_column],
)

# Inspect only validation rows during development.
permutation_result = permutation_importance(
    forest_candidate,
    parts["validation"][feature_columns],
    parts["validation"][target_column],
    scoring="neg_mean_absolute_error",
    n_repeats=10,
    random_state=217,
)

permutation_summary = pd.DataFrame(
    {
        "feature": feature_columns,
        "mean_score_decrease": permutation_result.importances_mean,
        "repeat_sd": permutation_result.importances_std,
    }
).sort_values("mean_score_decrease", ascending=False)
```

Permutation importance is model-specific and data-specific. Correlated features can share or mask importance, and a feature can be useful for prediction without being causal. Use noncausal language such as “the validation score depended on this feature for this fitted model.”

## Further study, not a framework checklist

Boosting, neural networks, specialized generalized models, mixed-effects models, causal inference, forecasting, uncertainty quantification, deployment, and monitoring can all be valuable. They are not safe next steps merely because a library exposes a short `fit()` call.

Before adopting a specialized method, require:

1. a question and use case that the core baseline/linear workflow cannot answer adequately;
2. the statistical and domain prerequisites for interpreting the method;
3. a split and metric matching real use;
4. a reproducible, certified environment and resource plan; and
5. a communication plan that distinguishes prediction from explanation and association from causation.

That decision rule is more durable than memorizing a list of currently popular frameworks.

## Bonus completion check

Optional work still preserves the core evaluation contract:

- preprocessing is fit within each training split or fold;
- validation or cross-validation chooses settings;
- final test data do not guide the choice;
- permutation importance is held-out and noncausal; and
- no bonus method becomes a Lecture 11 prerequisite without a separate curriculum decision.
