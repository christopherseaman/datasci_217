"""Checks for Assignment 10.

The course keeps these checks in 10/assignment_checks/, which the assignment
workflow downloads on every push, and the handout ships a byte-identical copy
so a local run reports exactly what GitHub will. They read the nine CSV files
and the one PNG a submission saves in output/ and compare them with values
computed from the four Parquet files in data/. The expected values come from
the lecture's own calls (statsmodels OLS, a mean baseline, persistence, and an
imputing, scaling LinearRegression pipeline), worked out by
scripts/build_assignment10_data.py and frozen in the GENERATED block below;
the self-test recomputes them from 10/assignment/data/ and confirms they match.

Each check scores one thing, so one mistake costs only its own points, and no
check waits for another to pass. Values are compared after parsing: spacing,
line endings, quoting, a byte-order mark, column order, row order, a leading
row-number column, number formatting (2 == 2.0 == 2.00), the letter case of
labels and headers, and the way a timestamp is written never cost points, and
cells may be separated by commas, semicolons, or tabs. A number may be
rounded to as few as one decimal, and accuracy, precision, recall, and R² may
be written as percents (85% for 0.85). Timestamps are compared as instants, in
UTC when they carry a zone and read as UTC when they do not. A row label the
checks do not recognize costs the rows check alone: the values checks still
find the row by its other values. Booleans may be written True/False,
true/false, yes/no, or 1/0.

The new patient's intervals and the availability decisions also accept values
calculated correctly from the student's earlier saved artifacts. A recognized
common slip that reaches several checks, such as a leaky feature carried from
Task 2.1 into Tasks 3.1 and 3.2, is charged once, by the first affected check,
and so are rows whose values all sit under each other's labels. Duplicate
headers cost the columns check; every stored copy is considered rather than
silently discarding a column. A value column the file does not hold costs only
its columns check, and so does one saved under another name with the right
values. One renamed with wrong values also costs its values check.

Nothing here imports, runs, or inspects student code.
"""

from __future__ import annotations

import codecs
import csv
import difflib
import re
import zlib
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
from pathlib import Path


# Two decimals are enough for coefficients and metrics; mmHg values may be rounded to one decimal.
FINE = 0.006
MMHG = 0.06
# Any number may also be rounded to fewer decimals, down to one: 3.3 stands for 3.34 (see _close).
# These columns hold proportions, which may also be written as a percent: 85% is 0.85.
PERCENT_COLUMNS = frozenset({"r2", "accuracy", "precision", "recall"})
# Ways pandas and people write an empty cell.
MISSING_SPELLINGS = frozenset({"", "nan", "na", "n/a", "none", "null", "<na>", "nat"})
TRUE_SPELLINGS = frozenset({"true", "t", "yes", "y", "1", "1.0"})
FALSE_SPELLINGS = frozenset({"false", "f", "no", "n", "0", "0.0"})


@dataclass(frozen=True)
class Instant:
    """An expected timestamp, written as UTC text such as 2026-05-09T18:30:00Z."""

    text: str


@dataclass(frozen=True)
class Flag:
    """An expected boolean."""

    value: bool


@dataclass(frozen=True)
class Starts:
    """An expected label: any cell that starts with this word, in any letter case."""

    word: str


@dataclass(frozen=True)
class Either:
    """An expected number with more than one accepted value."""

    values: tuple[float, ...]


@dataclass(frozen=True)
class Slip:
    """A common wrong answer: the values it leaves, by row key and column, and what causes it.

    A values check names the cause when every wrong cell it found holds this slip's value.
    Slips that share a `charge` come from one mistake, so only the first check naming one costs points.
    """

    cause: str
    rows: dict[tuple, dict[str, object]]
    charge: str = ""


@dataclass(frozen=True)
class Artifact:
    """One saved CSV: where it lives, which task writes it, and what it should hold.

    `rows` maps each required key, a tuple of the key columns' values, to the
    expected value of every other column. `optional_columns` may appear without
    counting as extra. `aliases` maps a key column to {other spelling: key value}.
    """

    path: str
    task: str
    columns: tuple[str, ...]
    key: tuple[str, ...]
    rows: dict[tuple, dict[str, object]]
    tolerance: float
    optional_columns: tuple[str, ...] = ()
    aliases: dict[str, dict[str, str]] = field(default_factory=dict)


@dataclass(frozen=True)
class Table:
    """A saved CSV: its casefolded column names and one {column: cell} dict per data row."""

    columns: tuple[str, ...]
    rows: tuple[dict[str, str], ...]
    repeated: tuple[str, ...] = ()


@dataclass(frozen=True)
class Check:
    name: str
    action: Callable[[Path], None]


# The expected values, worked out from data/*.parquet the way the README's steps do. The variants
# are common wrong answers computed from the same data, holding only the cells each one changes.
# BEGIN GENERATED by scripts/build_assignment10_data.py from data/*.parquet; do not edit by hand.
PATIENT_ROWS = 150000
ROWS_USED = 139656
VISIT_ROWS = 515750
DISCHARGE_ROWS = 210000
NEW_PATIENT = (60, 31.0)
AGE_BINS = (17, 39, 49, 59, 69, 79, 120)
AGE_BANDS = ('18-39', '40-49', '50-59', '60-69', '70-79', '80+')
FEATURES = ('age', 'bmi', 'sbp_today', 'prior_sbp', 'n_bp_meds', 'med_change_today')
CANDIDATE_HOURS = {
    'age': 0,
    'bmi': 0,
    'sbp_today': 0,
    'prior_sbp': -672,
    'n_bp_meds': 0,
    'med_change_today': 0,
    'a1c_result': 24,
    'callback_sbp': 72,
    'refill_gap_days': 720,
}
COEFFICIENT_VALUES = {
    ('Intercept',): {'coef': 80.736414, 'std_err': 0.250766, 'ci_lower': 80.244918, 'ci_upper': 81.22791},
    ('age',): {'coef': 0.296145, 'std_err': 0.002319, 'ci_lower': 0.2916, 'ci_upper': 0.30069},
    ('bmi',): {'coef': 0.884268, 'std_err': 0.007482, 'ci_lower': 0.869604, 'ci_upper': 0.898932},
}
NEW_PATIENT_VALUES = {
    (60, 31.0): {'mean': 125.917434, 'mean_ci_lower': 125.828262, 'mean_ci_upper': 126.006606, 'obs_ci_lower': 97.252747, 'obs_ci_upper': 154.582122},
}
RESIDUAL_BAND_VALUES = {
    ('18-39',): {'patients': 24239, 'mean_residual': 1.787162, 'sd_residual': 12.940114},
    ('40-49',): {'patients': 24175, 'mean_residual': -0.593267, 'sd_residual': 13.649724},
    ('50-59',): {'patients': 27508, 'mean_residual': -1.388758, 'sd_residual': 14.174634},
    ('60-69',): {'patients': 26956, 'mean_residual': -1.221691, 'sd_residual': 14.838201},
    ('70-79',): {'patients': 22305, 'mean_residual': 0.179978, 'sd_residual': 15.742256},
    ('80+',): {'patients': 14473, 'mean_residual': 2.635436, 'sd_residual': 16.704558},
}
SPLIT_VALUES = {
    ('train',): {'row_count': 327532, 'first_target_time': '2024-01-15T19:05:00Z', 'last_target_time': '2025-06-30T23:54:00Z'},
    ('fit',): {'row_count': 245361, 'first_target_time': '2024-01-15T19:05:00Z', 'last_target_time': '2025-04-08T18:48:00Z'},
    ('validation',): {'row_count': 97175, 'first_target_time': '2025-07-01T15:00:00Z', 'last_target_time': '2025-09-30T23:54:00Z'},
    ('test',): {'row_count': 91043, 'first_target_time': '2025-10-01T15:00:00Z', 'last_target_time': '2026-01-01T00:54:00Z'},
}
VALIDATION_VALUES = {
    ('mean_baseline',): {'mae': 11.515797, 'rmse': 14.393123, 'r2': -0.045204},
    ('persistence',): {'mae': 10.948958, 'rmse': 13.748351, 'r2': 0.046343},
    ('linear_pipeline',): {'mae': 9.010549, 'rmse': 11.29333, 'r2': 0.35652},
}
TEST_METRIC_VALUES = {
    ('linear_pipeline',): {'mae': 8.956891, 'rmse': 11.244381, 'r2': 0.359695},
}
CLINIC_TEST_VALUES = {
    ('Alameda',): {'test_visits': 15804, 'mae': 8.778114, 'mean_error': 0.489512},
    ('Berkeley',): {'test_visits': 14297, 'mae': 8.818873, 'mean_error': 0.245931},
    ('Eastmont',): {'test_visits': 13207, 'mae': 9.225743, 'mean_error': 3.114131},
    ('Fremont',): {'test_visits': 11995, 'mae': 8.928436, 'mean_error': 0.333324},
    ('Fruitvale',): {'test_visits': 11190, 'mae': 9.220592, 'mean_error': 2.32368},
    ('Hayward',): {'test_visits': 9766, 'mae': 8.86061, 'mean_error': 1.292296},
    ('San Leandro',): {'test_visits': 8509, 'mae': 8.985689, 'mean_error': 0.716793},
    ('Union City',): {'test_visits': 6275, 'mae': 8.8507, 'mean_error': -0.125664},
}
READMISSION_VALUES = {
    ('flag_no_one',): {'flagged': 0, 'accuracy': 0.867581, 'precision': 0, 'recall': 0},
    ('lace_13_plus',): {'flagged': 20414, 'accuracy': 0.83959, 'precision': 0.35603, 'recall': 0.261364},
    ('lace_10_plus',): {'flagged': 62228, 'accuracy': 0.709581, 'precision': 0.2334, 'recall': 0.522296},
}
VARIANTS = {
    'formula_without_age': {
        'coefficients': {('Intercept',): {'coef': 97.579026, 'std_err': 0.225387, 'ci_lower': 97.137271, 'ci_upper': 98.020781}, ('bmi',): {'coef': 0.887296, 'std_err': 0.007906, 'ci_lower': 0.8718, 'ci_upper': 0.902792}},
        'intervals': {(60, 31.0): {'mean': 125.085192, 'mean_ci_lower': 124.99193, 'mean_ci_upper': 125.178455, 'obs_ci_lower': 94.793203, 'obs_ci_upper': 155.377181}},
        'bands': {('18-39',): {'mean_residual': -5.613478, 'sd_residual': 12.87568}, ('40-49',): {'mean_residual': -4.288574, 'sd_residual': 13.661825}, ('50-59',): {'mean_residual': -2.150685, 'sd_residual': 14.197169}, ('60-69',): {'mean_residual': 0.935728, 'sd_residual': 14.876131}, ('70-79',): {'mean_residual': 5.236912, 'sd_residual': 15.789078}, ('80+',): {'mean_residual': 10.838777, 'sd_residual': 16.804808}},
    },
    'formula_without_bmi': {
        'coefficients': {('Intercept',): {'coef': 105.338242, 'std_err': 0.139948, 'ci_lower': 105.063946, 'ci_upper': 105.612538}, ('age',): {'coef': 0.300553, 'std_err': 0.002337, 'ci_lower': 0.295973, 'ci_upper': 0.305133}},
        'intervals': {(60, 31.0): {'mean': 123.371419, 'mean_ci_lower': 123.292622, 'mean_ci_upper': 123.450216, 'obs_ci_lower': 93.196119, 'obs_ci_upper': 153.54672}},
        'bands': {('18-39',): {'patients': 25798, 'mean_residual': 1.382748, 'sd_residual': 13.919447}, ('40-49',): {'patients': 25669, 'mean_residual': -0.436522, 'sd_residual': 14.537567}, ('50-59',): {'patients': 29284, 'mean_residual': -1.016509, 'sd_residual': 15.016498}, ('60-69',): {'patients': 28646, 'mean_residual': -1.054344, 'sd_residual': 15.589958}, ('70-79',): {'patients': 23763, 'mean_residual': 0.090754, 'sd_residual': 16.327769}, ('80+',): {'patients': 16840, 'mean_residual': 1.9802, 'sd_residual': 17.314108}},
    },
    'intervals_at_90': {
        'intervals': {(60, 31.0): {'mean_ci_lower': 125.842599, 'mean_ci_upper': 125.992269, 'obs_ci_lower': 101.861319, 'obs_ci_upper': 149.973549}},
    },
    'bands_size_with_missing': {
        'bands': {('18-39',): {'patients': 25798}, ('40-49',): {'patients': 25669}, ('50-59',): {'patients': 29284}, ('60-69',): {'patients': 28646}, ('70-79',): {'patients': 23763}, ('80+',): {'patients': 16840}},
    },
    'bands_sign_flipped': {
        'bands': {('18-39',): {'mean_residual': -1.787162}, ('40-49',): {'mean_residual': 0.593267}, ('50-59',): {'mean_residual': 1.388758}, ('60-69',): {'mean_residual': 1.221691}, ('70-79',): {'mean_residual': -0.179978}, ('80+',): {'mean_residual': -2.635436}},
    },
    'features_strict_zero': {
        'validation': {('linear_pipeline',): {'mae': 9.893964, 'rmse': 12.420803, 'r2': 0.221622}},
        'test': {('linear_pipeline',): {'mae': 9.711498, 'rmse': 12.201532, 'r2': 0.246046}},
        'clinics': {('Alameda',): {'mae': 9.417094, 'mean_error': 0.26834}, ('Berkeley',): {'mae': 9.547963, 'mean_error': -0.841759}, ('Eastmont',): {'mae': 10.316266, 'mean_error': 4.646131}, ('Fremont',): {'mae': 9.530944, 'mean_error': -0.183841}, ('Fruitvale',): {'mae': 10.086709, 'mean_error': 2.810256}, ('Hayward',): {'mae': 9.543356, 'mean_error': 1.68011}, ('San Leandro',): {'mae': 9.597597, 'mean_error': 0.038786}, ('Union City',): {'mae': 9.644885, 'mean_error': -0.993615}},
    },
    'features_equals_zero': {
        'validation': {('linear_pipeline',): {'mae': 9.435836, 'rmse': 11.823598, 'r2': 0.294673}},
        'test': {('linear_pipeline',): {'mae': 9.362059, 'rmse': 11.753005, 'r2': 0.300458}},
        'clinics': {('Alameda',): {'mae': 9.147821, 'mean_error': -0.24564}, ('Berkeley',): {'mae': 9.218272, 'mean_error': -0.806924}, ('Eastmont',): {'mae': 9.704482, 'mean_error': 3.402747}, ('Fremont',): {'mae': 9.320776, 'mean_error': -0.586719}, ('Fruitvale',): {'mae': 9.642318, 'mean_error': 2.118365}, ('Hayward',): {'mae': 9.213379, 'mean_error': 0.849307}, ('San Leandro',): {'mae': 9.412617, 'mean_error': -0.142474}, ('Union City',): {'mae': 9.250508, 'mean_error': -1.271924}},
    },
    'features_no_audit': {
        'validation': {('linear_pipeline',): {'mae': 8.332521, 'rmse': 10.436147, 'r2': 0.450495}},
        'test': {('linear_pipeline',): {'mae': 8.31909, 'rmse': 10.443836, 'r2': 0.447622}},
        'clinics': {('Alameda',): {'mae': 8.097149, 'mean_error': 0.184181}, ('Berkeley',): {'mae': 8.108692, 'mean_error': 0.411789}, ('Eastmont',): {'mae': 8.778889, 'mean_error': 3.803521}, ('Fremont',): {'mae': 8.250476, 'mean_error': -0.133148}, ('Fruitvale',): {'mae': 8.665942, 'mean_error': 2.942949}, ('Hayward',): {'mae': 8.120193, 'mean_error': 0.941937}, ('San Leandro',): {'mae': 8.286407, 'mean_error': 0.58255}, ('Union City',): {'mae': 8.256188, 'mean_error': -0.664472}},
    },
    'split_on_visit_time': {
        'split': {('train',): {'row_count': 360700, 'last_target_time': '2025-09-22T19:57:00Z'}, ('fit',): {'row_count': 327569, 'last_target_time': '2025-07-01T15:00:00Z'}, ('validation',): {'row_count': 98434, 'first_target_time': '2025-07-15T15:00:00Z', 'last_target_time': '2025-12-24T00:09:00Z'}, ('test',): {'row_count': 56616, 'first_target_time': '2025-10-15T15:00:00Z'}},
        'validation': {('mean_baseline',): {'mae': 11.248582, 'rmse': 14.105883, 'r2': -0.011231}, ('persistence',): {'mae': 11.030721, 'rmse': 13.843167, 'r2': 0.026085}, ('linear_pipeline',): {'mae': 8.99428, 'rmse': 11.276607, 'r2': 0.353741}},
        'test': {('linear_pipeline',): {'mae': 8.976106, 'rmse': 11.275079, 'r2': 0.358481}},
        'clinics': {('Alameda',): {'test_visits': 9804, 'mae': 8.769004, 'mean_error': 1.066627}, ('Berkeley',): {'test_visits': 8835, 'mae': 8.87042, 'mean_error': 0.983315}, ('Eastmont',): {'test_visits': 8301, 'mae': 9.31084, 'mean_error': 3.690172}, ('Fremont',): {'test_visits': 7421, 'mae': 8.960698, 'mean_error': 0.896759}, ('Fruitvale',): {'test_visits': 6995, 'mae': 9.27345, 'mean_error': 2.934862}, ('Hayward',): {'test_visits': 6089, 'mae': 8.799598, 'mean_error': 1.860745}, ('San Leandro',): {'test_visits': 5286, 'mae': 8.926409, 'mean_error': 1.086622}, ('Union City',): {'test_visits': 3885, 'mae': 8.862189, 'mean_error': 0.594844}},
    },
    'ranges_from_visit_time': {
        'split': {('train',): {'first_target_time': '2024-01-01T16:02:00Z', 'last_target_time': '2025-06-16T23:54:00Z'}, ('fit',): {'first_target_time': '2024-01-01T16:02:00Z', 'last_target_time': '2025-03-25T23:54:00Z'}, ('validation',): {'first_target_time': '2025-04-08T18:48:00Z', 'last_target_time': '2025-09-16T23:54:00Z'}, ('test',): {'first_target_time': '2025-07-09T15:22:00Z', 'last_target_time': '2025-12-18T00:54:00Z'}},
    },
    'fit_not_trimmed': {
        'split': {('fit',): {'row_count': 327532, 'last_target_time': '2025-06-30T23:54:00Z'}},
        'validation': {('mean_baseline',): {'mae': 11.40178, 'rmse': 14.277624, 'r2': -0.028497}, ('linear_pipeline',): {'mae': 8.987286, 'rmse': 11.265441, 'r2': 0.359694}},
        'test': {('linear_pipeline',): {'mae': 9.005316, 'rmse': 11.306346, 'r2': 0.352618}},
        'clinics': {('Alameda',): {'mae': 8.811251, 'mean_error': 0.98851}, ('Berkeley',): {'mae': 8.839157, 'mean_error': 0.774074}, ('Eastmont',): {'mae': 9.337862, 'mean_error': 3.593047}, ('Fremont',): {'mae': 8.951372, 'mean_error': 0.842916}, ('Fruitvale',): {'mae': 9.315867, 'mean_error': 2.837603}, ('Hayward',): {'mae': 8.912227, 'mean_error': 1.79083}, ('San Leandro',): {'mae': 9.01567, 'mean_error': 1.243106}, ('Union City',): {'mae': 8.852904, 'mean_error': 0.397517}},
    },
    'baseline_from_validation': {
        'validation': {('mean_baseline',): {'mae': 11.170152, 'rmse': 14.078439, 'r2': 0}},
    },
    'rmse_not_rooted': {
        'validation': {('mean_baseline',): {'rmse': 207.161986}, ('persistence',): {'rmse': 189.017144}, ('linear_pipeline',): {'rmse': 127.539306}},
        'test': {('linear_pipeline',): {'rmse': 126.43611}},
    },
    'r2_args_swapped': {
        'validation': {('mean_baseline',): {'r2': 0}, ('persistence',): {'r2': 0.054446}, ('linear_pipeline',): {'r2': -0.748445}},
        'test': {('linear_pipeline',): {'r2': -0.677188}},
    },
    'validation_on_test_rows': {
        'validation': {('mean_baseline',): {'mae': 11.184855, 'rmse': 14.054485, 'r2': -0.000336}, ('persistence',): {'mae': 10.989895, 'rmse': 13.810772, 'r2': 0.034056}, ('linear_pipeline',): {'mae': 8.956891, 'rmse': 11.244381, 'r2': 0.359695}},
    },
    'refit_on_validation': {
        'test': {('linear_pipeline',): {'mae': 8.975172, 'rmse': 11.267877, 'r2': 0.357016}},
        'clinics': {('Alameda',): {'mae': 8.789702, 'mean_error': 0.728013}, ('Berkeley',): {'mae': 8.824194, 'mean_error': 0.493182}, ('Eastmont',): {'mae': 9.27391, 'mean_error': 3.332162}, ('Fremont',): {'mae': 8.933741, 'mean_error': 0.579229}, ('Fruitvale',): {'mae': 9.259589, 'mean_error': 2.557412}, ('Hayward',): {'mae': 8.88087, 'mean_error': 1.529385}, ('San Leandro',): {'mae': 8.996015, 'mean_error': 0.963515}, ('Union City',): {'mae': 8.848031, 'mean_error': 0.128995}},
    },
    'test_scored_on_validation': {
        'test': {('linear_pipeline',): {'mae': 9.010549, 'rmse': 11.29333, 'r2': 0.35652}},
    },
    'mean_error_sign_flipped': {
        'clinics': {('Alameda',): {'mean_error': -0.489512}, ('Berkeley',): {'mean_error': -0.245931}, ('Eastmont',): {'mean_error': -3.114131}, ('Fremont',): {'mean_error': -0.333324}, ('Fruitvale',): {'mean_error': -2.32368}, ('Hayward',): {'mean_error': -1.292296}, ('San Leandro',): {'mean_error': -0.716793}, ('Union City',): {'mean_error': 0.125664}},
    },
    'lace_strictly_greater': {
        'readmission': {('lace_13_plus',): {'flagged': 12034, 'accuracy': 0.8582, 'precision': 0.418149, 'recall': 0.180955}, ('lace_10_plus',): {'flagged': 43253, 'accuracy': 0.773624, 'precision': 0.271912, 'recall': 0.422936}},
    },
    'flag_passed_first': {
        'readmission': {('lace_13_plus',): {'precision': 0.261364, 'recall': 0.35603}, ('lace_10_plus',): {'precision': 0.522296, 'recall': 0.2334}},
    },
}
# END GENERATED


def _slip(variant: str, artifact: str, cause: str) -> Slip:
    """One variant's wrong cells in one artifact; every artifact of a variant shares its charge."""
    rows = VARIANTS.get(variant, {}).get(artifact, {})
    # Timestamps are frozen as UTC text and compared as instants.
    rows = {key: {column: Instant(value) if column.endswith("_time") else value for column, value in values.items()}
            for key, values in rows.items()}
    return Slip(cause, rows, charge=variant)


CAUSES = {
    "formula_without_age": "Those values come from a formula without age, sbp ~ bmi; the model is sbp ~ age + bmi.",
    "formula_without_bmi": "Those values come from a formula without bmi, sbp ~ age; the model is sbp ~ age + bmi.",
    "intervals_at_90": "Those bounds are 90% intervals, which summary_frame(alpha=0.10) returns; alpha=0.05 gives 95%.",
    "bands_size_with_missing": (
        'Those counts include the patients with no BMI, whose residual is blank, which "size" counts; '
        'count the residuals with "count".'
    ),
    "bands_sign_flipped": (
        "Those mean residuals have the opposite sign, fitted minus observed; a residual is observed minus fitted, "
        "as results.resid holds it."
    ),
    "features_strict_zero": (
        "Those values come from the rule hours_after_visit < 0, which keeps prior_sbp alone; a feature known at 0 "
        "hours, by the end of the visit, is available too, so the rule is <= 0."
    ),
    "features_equals_zero": (
        "Those values come from the rule hours_after_visit == 0, which drops prior_sbp, known 672 hours before the "
        "visit; the rule is <= 0."
    ),
    "features_no_audit": (
        "Those values come from a model that also uses a1c_result, callback_sbp, and refill_gap_days, which are "
        "not known when the visit ends: that is leakage. Use the six features Task 2.1 keeps."
    ),
    "split_on_visit_time": (
        "Those values come from splitting on visit_time; split on followup_time, the target time."
    ),
    "ranges_from_visit_time": (
        "Those times come from visit_time; first_target_time and last_target_time are the min() and max() of "
        "followup_time."
    ),
    "fit_not_trimmed": (
        "Those values come from fitting on every training row; the fit rows are the training rows whose "
        "followup_time is on or before the first validation visit_time."
    ),
    "baseline_from_validation": (
        "The mean baseline was fitted on the validation rows; fit it on the fit rows, so it predicts their mean."
    ),
    "rmse_not_rooted": "Those rmse values are mean squared errors; rmse is np.sqrt() of mean_squared_error().",
    "r2_args_swapped": "Those R² values come from r2_score(predicted, actual); pass the actual values first.",
    "validation_on_test_rows": (
        "Those metrics score the test rows; Task 3.1 scores the validation rows and keeps the test rows sealed."
    ),
    "refit_on_validation": (
        "Those values come from a pipeline refitted on the fit plus validation rows; Task 3.2 predicts with the "
        "pipeline Task 3.1 fitted, unchanged."
    ),
    "test_scored_on_validation": (
        "Those metrics score the validation rows again; predict test[FEATURES] and compare with test[TARGET]."
    ),
    "mean_error_sign_flipped": (
        "Those mean errors have the opposite sign, predicted minus actual; the error is actual minus predicted."
    ),
    "lace_strictly_greater": (
        "Those values flag lace_score > the cutoff; each rule flags lace_score >= its cutoff."
    ),
    "flag_passed_first": (
        "precision and recall have traded places, which is what passing the flag before the actual outcome does; "
        'pass discharges["readmitted_30d"] first.'
    ),
}
# A feature-rule slip shows first in the availability table, then in every model fitted with those features.
FEATURE_RULES = {
    "features_strict_zero": lambda hours: hours < 0,
    "features_equals_zero": lambda hours: hours == 0,
    "features_no_audit": lambda hours: True,
}

FORMULA_SLIPS = tuple(_slip(name, "coefficients", CAUSES[name]) for name in ("formula_without_age", "formula_without_bmi"))
FORMULA_CHARGES = frozenset(slip.charge for slip in FORMULA_SLIPS)
INTERVAL_SLIPS = tuple(
    _slip(name, "intervals", CAUSES[name]) for name in ("intervals_at_90", "formula_without_age", "formula_without_bmi")
)
BAND_SLIPS = tuple(_slip(name, "bands", CAUSES[name]) for name in (
    "formula_without_age", "formula_without_bmi", "bands_size_with_missing", "bands_sign_flipped"))
AVAILABILITY_SLIPS = tuple(
    Slip(CAUSES[name], {(feature,): {"available": Flag(rule(hours))} for feature, hours in CANDIDATE_HOURS.items()
                        if rule(hours) != (hours <= 0)}, charge=name)
    for name, rule in FEATURE_RULES.items()
)
SPLIT_SLIPS = tuple(_slip(name, "split", CAUSES[name]) for name in (
    "split_on_visit_time", "ranges_from_visit_time", "fit_not_trimmed"))
UPSTREAM = ("features_strict_zero", "features_equals_zero", "features_no_audit", "split_on_visit_time",
            "fit_not_trimmed")
VALIDATION_SLIPS = tuple(_slip(name, "validation", CAUSES[name]) for name in (
    *UPSTREAM, "baseline_from_validation", "rmse_not_rooted", "r2_args_swapped", "validation_on_test_rows"))
TEST_SLIPS = tuple(_slip(name, "test", CAUSES[name]) for name in (
    *UPSTREAM, "refit_on_validation", "test_scored_on_validation", "rmse_not_rooted", "r2_args_swapped"))
CLINIC_SLIPS = tuple(_slip(name, "clinics", CAUSES[name]) for name in (
    *UPSTREAM, "refit_on_validation", "mean_error_sign_flipped"))
READMISSION_SLIPS = tuple(_slip(name, "readmission", CAUSES[name]) for name in (
    "lace_strictly_greater", "flag_passed_first"))
SLIPS = (*FORMULA_SLIPS, *INTERVAL_SLIPS, *BAND_SLIPS, *AVAILABILITY_SLIPS, *SPLIT_SLIPS, *VALIDATION_SLIPS,
         *TEST_SLIPS, *CLINIC_SLIPS, *READMISSION_SLIPS)


COEFFICIENTS = Artifact(
    path="output/ols_coefficients.csv",
    task="Task 1.1",
    columns=("term", "coef", "std_err", "ci_lower", "ci_upper"),
    key=("term",),
    rows=COEFFICIENT_VALUES,
    tolerance=FINE,
    # The array interface, sm.OLS with sm.add_constant(), names the intercept const.
    aliases={"term": {"const": "Intercept"}},
)

NEW_PATIENT_INTERVALS = Artifact(
    path="output/new_patient_intervals.csv",
    task="Task 1.2",
    columns=("age", "bmi", "mean", "mean_ci_lower", "mean_ci_upper", "obs_ci_lower", "obs_ci_upper"),
    key=("age", "bmi"),
    rows=NEW_PATIENT_VALUES,
    tolerance=MMHG,
    # summary_frame() also returns mean_se; keeping it is fine.
    optional_columns=("mean_se",),
)

RESIDUAL_BANDS = Artifact(
    path="output/residual_bands.csv",
    task="Task 1.3",
    columns=("age_band", "patients", "mean_residual", "sd_residual"),
    key=("age_band",),
    rows=RESIDUAL_BAND_VALUES,
    tolerance=MMHG,
    # pd.cut() without labels= names each band by its interval, as (17, 39].
    aliases={"age_band": {f"({low}, {high}]": label
                          for low, high, label in zip(AGE_BINS[:-1], AGE_BINS[1:], AGE_BANDS)}},
)

AVAILABILITY = Artifact(
    path="output/availability_decisions.csv",
    task="Task 2.1",
    columns=("candidate_feature", "hours_after_visit", "available", "decision"),
    key=("candidate_feature",),
    rows={
        (feature,): {
            "hours_after_visit": hours,
            "available": Flag(hours <= 0),
            "decision": Starts("keep" if hours <= 0 else "exclude"),
        }
        for feature, hours in CANDIDATE_HOURS.items()
    },
    tolerance=FINE,
    # The supplied candidates table also has becomes_known; a .copy() keeps it.
    optional_columns=("becomes_known",),
)

SPLIT = Artifact(
    path="output/split_summary.csv",
    task="Task 2.2",
    columns=("partition", "row_count", "first_target_time", "last_target_time"),
    key=("partition",),
    rows={
        key: {
            "row_count": values["row_count"],
            "first_target_time": Instant(values["first_target_time"]),
            "last_target_time": Instant(values["last_target_time"]),
        }
        for key, values in SPLIT_VALUES.items()
    },
    tolerance=FINE,
    aliases={"partition": {"training": "train", "fit_rows": "fit", "train_fit": "fit", "valid": "validation",
                           "val": "validation", "testing": "test"}},
)

VALIDATION = Artifact(
    path="output/validation_metrics.csv",
    task="Task 3.1",
    columns=("approach", "mae", "rmse", "r2"),
    key=("approach",),
    rows=VALIDATION_VALUES,
    tolerance=FINE,
)

TEST_METRICS = Artifact(
    path="output/test_metrics.csv",
    task="Task 3.2",
    columns=("approach", "mae", "rmse", "r2"),
    key=("approach",),
    rows=TEST_METRIC_VALUES,
    tolerance=FINE,
)

CLINIC_ERRORS = Artifact(
    path="output/clinic_test_errors.csv",
    task="Task 3.2",
    columns=("clinic", "test_visits", "mae", "mean_error"),
    key=("clinic",),
    rows=CLINIC_TEST_VALUES,
    tolerance=FINE,
)

READMISSION = Artifact(
    path="output/readmission_metrics.csv",
    task="Task 3.3",
    columns=("approach", "flagged", "accuracy", "precision", "recall"),
    key=("approach",),
    rows=READMISSION_VALUES,
    tolerance=FINE,
)

ARTIFACTS = (
    COEFFICIENTS, NEW_PATIENT_INTERVALS, RESIDUAL_BANDS, AVAILABILITY, SPLIT,
    VALIDATION, TEST_METRICS, CLINIC_ERRORS, READMISSION,
)
# The short names VARIANTS uses for ARTIFACTS, in the same order.
VARIANT_ARTIFACTS = ("coefficients", "intervals", "bands", "availability", "split", "validation", "test", "clinics",
                     "readmission")
RESIDUAL_PLOT = "output/residuals_vs_fitted.png"
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_REPORTED_OUTSIDE: set[str] = set()
# The mistakes run_checks() has charged so far in this run.
_CHARGED: set[str] = set()




def _assert(condition: object, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _join(items) -> str:
    items = [str(item) for item in items]
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def _listed(problems: list[str], limit: int = 4) -> str:
    shown = "; ".join(problems[:limit])
    if len(problems) > limit:
        shown += f"; and {len(problems) - limit} more like {'it' if len(problems) - limit == 1 else 'them'}"
    return shown


def _clean(cell: str) -> str:
    return " ".join(cell.split())


def _label(cell: str) -> str:
    """A label compared in any letter case, with spaces, hyphens, and underscores alike."""
    return re.sub(r"[\s_-]+", "_", cell.strip().casefold())


def _fold(column: str) -> str:
    """A column name compared the way labels are."""
    return _label(column)


def _number(cell: str) -> float | None:
    try:
        return float(_clean(cell).replace(",", ""))
    except ValueError:
        return None


def _is_whole_number(cell: str) -> bool:
    number = _number(cell)
    return number is not None and number.is_integer()


def _instant(cell: str) -> datetime | None:
    """A timestamp as a UTC instant; one written without a zone is read as UTC.

    A trailing Z, UTC, or GMT, as strftime("%Z") writes it, names UTC.
    """
    text = _clean(cell)
    zone = re.search(r"\s*(?:z|utc|gmt)$", text, re.IGNORECASE)
    if zone:
        text = text[:zone.start()] + "+00:00"
    try:
        moment = datetime.fromisoformat(text)
    except ValueError:
        moment = None
        dates = ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%m/%d/%Y", "%d/%m/%Y",
                 "%d %b %Y", "%d %B %Y", "%d-%b-%Y", "%d-%B-%Y")
        times = ("%H:%M", "%H:%M:%S", "%H:%M:%S.%f", "%I:%M %p", "%I:%M:%S %p")
        spellings = ("%Y%m%dT%H%M%S", "%Y%m%d %H%M%S", *(f"{date} {time}" for date in dates for time in times))
        for spelling in (*spellings, *(spelling + "%z" for spelling in spellings)):
            try:
                moment = datetime.strptime(text, spelling)
                break
            except ValueError:
                continue
        if moment is None:
            return None
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    try:
        return moment.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def _flag(cell: str) -> bool | None:
    text = _clean(cell).casefold()
    if text in TRUE_SPELLINGS:
        return True
    if text in FALSE_SPELLINGS:
        return False
    return None


def _close(cell: str, expected: float, tolerance: float, percent: bool = False) -> bool:
    """Whether a number cell holds `expected`, within `tolerance` or the rounding its decimals show.

    A number written with one or more decimals is read as rounded there, so it
    passes within half of its last decimal: 3.3 for 3.34, or 0.8 for 0.79. With
    `percent`, a cell ending in % is a percent of the proportion: 85% is 0.85.
    """
    text = _clean(cell).replace(",", "")
    scale = 1
    if percent and text.endswith("%"):
        text, scale = text[:-1].strip(), 100
    try:
        written = Decimal(text)
    except InvalidOperation:
        return False
    if not written.is_finite():
        return False
    decimals = -written.as_tuple().exponent + (2 if scale == 100 else 0)
    allowed = max(tolerance, 0.5 * 10.0 ** -decimals + 1e-9) if decimals >= 1 else tolerance
    return abs(float(written) / scale - expected) <= allowed


def _matches(cell: str, expected: object, tolerance: float, percent: bool = False) -> bool:
    text = _clean(cell)
    if expected is None:
        return text.casefold() in MISSING_SPELLINGS
    if isinstance(expected, Instant):
        return _instant(text) is not None and _instant(text) == _instant(expected.text)
    if isinstance(expected, Flag):
        return _flag(text) is expected.value
    if isinstance(expected, Starts):
        return _label(text).startswith(expected.word)
    if isinstance(expected, Either):
        return any(_close(text, value, tolerance, percent) for value in expected.values)
    if isinstance(expected, (int, float)):
        return _close(text, expected, tolerance, percent)
    return _label(text) == _label(str(expected))


def _key_matches(cell: str, part: object, artifact: Artifact, column: str) -> bool:
    """Whether a saved key cell names this expected key value, allowing the column's other spellings."""
    if isinstance(part, (int, float)):
        return _close(cell, part, artifact.tolerance)
    spelled = _label(cell)
    spelled = {_label(alias): _label(value) for alias, value in artifact.aliases.get(column, {}).items()}.get(spelled, spelled)
    return spelled == _label(str(part))


def _show(value: object) -> str:
    if value is None:
        return "blank"
    if isinstance(value, Instant):
        return value.text
    if isinstance(value, Flag):
        return str(value.value)
    if isinstance(value, Starts):
        return f'a label starting with "{value.word}"'
    if isinstance(value, Either):
        return " or ".join(_show(item) for item in value.values)
    if isinstance(value, float):
        return f"{round(value, 2):g}"
    return str(value)


def _given(cell: str) -> str:
    text = _clean(cell)
    return text if text else "blank"


def _key_name(key: tuple, artifact: Artifact) -> str:
    if len(key) == 1:
        return str(key[0])
    return " and ".join(f"{column} {part}" for column, part in zip(artifact.key, key))


def _expected_keys(names: list[str]) -> str:
    """The expected row keys, shortened past six: 'V001, V002, V003, ... and V015 (15 in all)'."""
    if len(names) <= 6:
        return _join(names)
    return f"{', '.join(names[:3])}, ... and {names[-1]} ({len(names)} in all)"


def _decode(raw: bytes) -> str:
    """An artifact's text without a byte-order mark; UTF-16 is read through its mark."""
    if raw.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return raw.decode("utf-16")
        except UnicodeDecodeError:
            pass
    return raw.decode("utf-8", errors="replace").lstrip("﻿")


# A CSV may separate its cells with commas, semicolons, or tabs.
DELIMITERS = (",", ";", "\t")
# A number written with a decimal comma, as a spreadsheet set to a European locale saves it: 12,5 is 12.5.
DECIMAL_COMMA = re.compile(r"^(\s*[+-]?\d*),(\d+\s*)$")


def _csv_rows(text: str) -> list[list[str]]:
    """The rows of a saved CSV, blank lines left out, split on the separator its header line uses.

    Commas, semicolons, and tabs are all accepted: whichever splits the header
    line into the most cells is used, and a tie keeps commas. A file separated
    by semicolons may write decimal commas, so there 12,5 reads as 12.5.
    """
    if "\x00" in text:
        raise csv.Error("file contains a NUL byte instead of CSV text")
    lines = text.splitlines()
    header = next((line for line in lines if line.strip()), "")
    delimiter = max(DELIMITERS, key=lambda mark: len(next(csv.reader([header], delimiter=mark), [])))
    rows = [row for row in csv.reader(lines, delimiter=delimiter) if any(cell.strip() for cell in row)]
    if delimiter == ";":
        rows = [[DECIMAL_COMMA.sub(r"\1.\2", cell) for cell in row] for row in rows]
    return rows


def _artifact_path(root: Path, name: str) -> Path | None:
    """The artifact's path, matching its file name in any letter case, or None.

    macOS and Windows disks ignore letter case, so matching in any case makes a
    local run there agree with GitHub's Linux runner.
    """
    path = root / name
    if path.is_file() and not path.is_symlink():
        return path
    if not path.parent.is_dir():
        return None
    same_name = [
        other
        for other in path.parent.iterdir()
        if other.name.casefold() == path.name.casefold() and other.is_file() and not other.is_symlink()
    ]
    return same_name[0] if len(same_name) == 1 else None


def _missing(root: Path, path: str, task: str) -> str:
    message = f"{path} is missing; run the {task} cell to write it, then commit it."
    wanted = root / path
    if wanted.parent.is_dir():
        # Another required file, such as validation_metrics.csv beside readmission_metrics.csv, is never a look-alike.
        required = {artifact.path.casefold() for artifact in ARTIFACTS} | {RESIDUAL_PLOT.casefold()}
        look_alikes = sorted(
            other.relative_to(root).as_posix()
            for other in wanted.parent.iterdir()
            if other.is_file()
            and other.relative_to(root).as_posix().casefold() not in required
            and difflib.SequenceMatcher(None, wanted.name.casefold(), other.name.casefold()).ratio() >= 0.75
        )
        if look_alikes:
            message += f" Found {_join(look_alikes)}; save it as {path} instead."
    return message


def _is_index_header(cell: str) -> bool:
    return cell in ("", "index") or cell.startswith("unnamed")


def read_table(root: Path, artifact: Artifact) -> Table:
    """Parse a saved CSV, however it is separated, spaced, quoted, or ended.

    A leading column headed by nothing, `Unnamed: 0`, or `index` that holds only
    whole numbers is the row numbers pandas writes when `index=False` is left
    out, so it is set aside rather than counted as a column. So is an empty
    column with no header, which a trailing comma on every line makes.
    """
    path = _artifact_path(root, artifact.path)
    if path is None:
        outside = _artifact_path(root, Path(artifact.path).name)
        if outside is not None:
            if artifact.path not in _REPORTED_OUTSIDE:
                _REPORTED_OUTSIDE.add(artifact.path)
                raise AssertionError(
                    f"{artifact.path} is missing, but {outside.name} is in the assignment folder; "
                    f"in {artifact.task}, save it in output/. Later checks read the file where it is."
                )
            path = outside
    _assert(path is not None, _missing(root, artifact.path, artifact.task))
    try:
        lines = _csv_rows(_decode(path.read_bytes()))
    except csv.Error as error:
        raise AssertionError(
            f"{artifact.path} cannot be read as a CSV table ({error}); run the {artifact.task} cell again so "
            "to_csv() writes it, then compare it with the checkpoint in README.md."
        ) from None
    _assert(lines, f"{artifact.path} is empty; run the {artifact.task} cell again to write it.")
    header = [_fold(cell) for cell in lines[0]]
    body = [[_clean(cell) for cell in row] for row in lines[1:]]
    width = max(len(header), *(len(row) for row in body)) if body else len(header)
    header += [""] * (width - len(header))
    body = [row + [""] * (width - len(row)) for row in body]

    # Unless the file has no column named for its first key and this column holds those keys instead.
    first_is_key = _fold(artifact.key[0]) not in header and body and all(
        any(_key_matches(row[0], key[0], artifact, artifact.key[0]) for key in artifact.rows) for row in body
    )
    if (len(header) > 1 and _is_index_header(header[0]) and body and not first_is_key
            and all(_is_whole_number(row[0]) for row in body)):
        header = header[1:]
        body = [row[1:] for row in body]
    keep = [i for i, column in enumerate(header) if column or any(row[i] for row in body)]
    header = [header[i] for i in keep]
    body = [[row[i] for i in keep] for row in body]

    positions = {column: [i for i, name in enumerate(header) if name == column] for column in header}
    repeated = tuple(column for column, places in positions.items() if len(places) > 1)
    # Keep every copy available: a duplicate header costs the columns check, while
    # value checks use the copy that best preserves the requested data.
    chosen = {}
    for column, places in positions.items():
        candidates = [values.get(column) for values in artifact.rows.values() if column in values]
        if column in artifact.key:
            candidates = [key[artifact.key.index(column)] for key in artifact.rows]
        def preserved_values(position: int) -> int:
            score = 0
            for row in body:
                matching = [key for key in artifact.rows if all(
                    name in positions and any(_key_matches(row[i], part, artifact, name) for i in positions[name])
                    for name, part in zip(artifact.key, key)
                )]
                expected = [artifact.rows[key][column] for key in matching if column in artifact.rows[key]]
                if column in artifact.key:
                    expected = [key[artifact.key.index(column)] for key in matching]
                score += any(_matches(row[position], value, artifact.tolerance, column in PERCENT_COLUMNS)
                             for value in (expected or candidates))
            return score
        chosen[column] = max(places, key=preserved_values)
    rows = tuple({column: row[i] for column, i in chosen.items()} for row in body)
    return Table(columns=tuple(header), rows=rows, repeated=repeated)


def _resolve(table: Table, artifact: Artifact, root: Path) -> dict[str, str]:
    """Map each expected column to the saved column that holds it.

    A column saved under its own name, in any letter case, is found by name. A
    key column saved under another name, such as a blank header over saved
    index labels, is found by its values. An unexpected column whose values are
    what a values check accepts for exactly one unplaced value column (see
    `_accepted`, which reads the student's earlier artifacts under `root`) is
    taken to hold it. If exactly one expected value column is still unplaced
    and exactly one unexpected column is left, that column is taken to hold it.
    A misnamed column then costs only the columns check, or its values check
    too when its values are wrong.
    """
    expected = [_fold(column) for column in (*artifact.columns, *artifact.optional_columns)]
    found = {column: _fold(column) for column in artifact.columns if _fold(column) in table.columns}
    spare = [column for column in table.columns if column not in expected]
    for position, column in enumerate(artifact.key):
        if column in found:
            continue
        candidates = []
        for other in spare:
            values = [row[other] for row in table.rows if row[other]]
            hits = sum(any(_key_matches(value, key[position], artifact, column) for key in artifact.rows) for value in values)
            if values and hits * 2 > len(values):
                candidates.append(other)
        if len(candidates) == 1:
            found[column] = candidates[0]
            spare.remove(candidates[0])
    # A key column is only ever found by name or by its values, never by elimination.
    unplaced = [column for column in artifact.columns if column not in found and column not in artifact.key]
    slips = list(SLIPS)
    placed = True
    # Repeat while columns are placed: a residual is judged against the fitted column placed before it.
    while placed and unplaced and spare:
        placed = False
        try:
            grouped, _ = _group(table, artifact, found)
        except AssertionError:
            grouped = {}

        def holds(other: str, column: str, slipped: bool) -> bool:
            trial = {**found, column: other}
            return bool(grouped) and all(
                any(_matches(row[other], value, artifact.tolerance, column in PERCENT_COLUMNS) for value in (
                    _accepted(root, artifact, key, rows, column, trial),
                    # A common slip is charged once by its values check, so its values also identify the column.
                    *(slip.rows[key][column] for slip in slips if slipped and column in slip.rows.get(key, {})),
                ))
                for key, rows in grouped.items() for row in rows
            )

        # A slip's values place a column only when the accepted values place none.
        for slipped in (False, True):
            fits = {other: [column for column in unplaced if holds(other, column, slipped)] for other in spare}
            # A spare column that fits two expected columns places neither; of two spares that fit one, the first holds it.
            for other, columns in fits.items():
                if len(columns) == 1 and columns[0] not in found:
                    found[columns[0]] = other
                    placed = True
            if placed:
                break
        spare = [column for column in spare if column not in found.values()]
        unplaced = [column for column in unplaced if column not in found]
    if len(unplaced) == 1 and len(spare) == 1:
        found[unplaced[0]] = spare[0]
    return found


def _describe(row: dict[str, str], columns: list[str]) -> str:
    return "a row reading " + ", ".join(_given(row[column]) for column in columns)


def _by_values(row: dict[str, str], artifact: Artifact, found: dict[str, str], claimed, checked: tuple[str, ...]):
    """The expected key whose values this row holds, for a row its key cells do not name, or None.

    The row is recognized by its value columns other than `checked`, so a wrong
    value in a checked column is still compared instead of hiding the row. When
    those values fit several keys, as three kept features do, the row goes to a
    key it fits in every column, one not yet claimed if there is one.
    """
    values = [column for column in artifact.columns if column not in artifact.key and column in found]
    others = [column for column in values if column not in checked] or values

    def fits(key: tuple, columns: list[str]) -> bool:
        return all(
            _key_matches(row[found[column]], key[position], artifact, column)
            for position, column in enumerate(artifact.key)
            if column in found
        ) and all(
            _matches(row[found[column]], artifact.rows[key][column], artifact.tolerance, column in PERCENT_COLUMNS)
            for column in columns
        )

    candidates = [key for key in artifact.rows if fits(key, others)]
    if len(candidates) == 1:
        return candidates[0]
    exact = [key for key in candidates if fits(key, values)]
    unclaimed = [key for key in exact if key not in claimed]
    return (unclaimed or exact or [None])[0]


def _group(table: Table, artifact: Artifact, found: dict[str, str], checked: tuple[str, ...] | None = None):
    """({expected key: [rows]}, [rows naming no expected key]) for the saved table.

    When every key column is saved, each row is matched by its key. When one is
    not, each row is matched by the values it does hold, so the lost column
    costs the columns check alone. A values check passes the columns it
    compares as `checked`; rows whose key cells name no expected key are then
    matched by their other values too, so a wrong or unexpected label costs the
    rows check alone.
    """
    grouped: dict[tuple, list[dict[str, str]]] = {}
    unknown = []
    if len(artifact.rows) == 1 and len(table.rows) == 1:
        # A one-row answer saved as one row is compared as it stands; a wrong or missing
        # key value there costs the rows check, not the values checks.
        only, row = next(iter(artifact.rows)), table.rows[0]
        named = all(
            _key_matches(row[found[column]], only[position], artifact, column)
            for position, column in enumerate(artifact.key)
            if column in found
        )
        if not named:
            unknown.append("a row for " + " ".join(row[found[column]] or "blank" for column in artifact.key if column in found))
        return {only: [row]}, unknown
    if all(column in found for column in artifact.key):
        unnamed = []
        for row in table.rows:
            match = next(
                (
                    key for key in artifact.rows
                    if all(
                        _key_matches(row[found[column]], key[position], artifact, column)
                        for position, column in enumerate(artifact.key)
                    )
                ),
                None,
            )
            if match is None:
                unnamed.append(row)
            else:
                grouped.setdefault(match, []).append(row)
        if checked is None:
            unknown = ["a row for " + " ".join(row[found[column]] or "blank" for column in artifact.key)
                       for row in unnamed]
            return grouped, unknown
        # The rows check has already charged for these labels; read the values beneath them anyway.
        found = {column: saved for column, saved in found.items() if column not in artifact.key}
        if not found:
            # A table of keys alone has no values to match its other rows by.
            return grouped, []
        rows = unnamed
    else:
        rows = list(table.rows)

    values = [column for column in artifact.columns if column not in artifact.key and column in found]
    absent = [column for column in artifact.key if column not in found]
    _assert(
        values,
        f"{artifact.path} has no {_join(absent)} column (its columns are {_join(table.columns) or 'none'}), "
        f"so its rows cannot be matched; {artifact.task} saves the header line {','.join(artifact.columns)}.",
    )
    for row in rows:
        match = _by_values(row, artifact, found, grouped, checked or ())
        if match is None:
            unknown.append(_describe(row, [found[column] for column in artifact.columns if column in found]))
        else:
            grouped.setdefault(match, []).append(row)
    return grouped, unknown


def columns_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """The saved header holds exactly the expected columns, in any order."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact, root)
        # A saved index with no name leaves its column unheaded; holding the keys, it is the key column.
        first = table.columns[0] if table.columns else None
        unheaded = {
            column: found[column]
            for column in artifact.key
            if _fold(column) not in table.columns and first is not None and found.get(column) == first
            and _is_index_header(first)
        }
        expected = [_fold(column) for column in (*artifact.columns, *artifact.optional_columns)]
        missing = [
            column for column in artifact.columns
            if _fold(column) not in table.columns and column not in unheaded
        ]
        extra = [
            column or "a column with no header"
            for column in table.columns
            if column not in expected and column not in unheaded.values()
        ]
        problems = []
        if table.repeated:
            problems.append("names columns more than once: " + _join(table.repeated))
        if missing:
            problems.append(f"is missing {_join(missing)}")
        if extra:
            problems.append(f"also has {_join(extra)}")
        owner = _header_owner(table, artifact) if problems else None
        cause = (
            f" The saved header is the one {owner.task} writes to {owner.path}; check the path each cell passes "
            "to to_csv()." if owner else ""
        )
        _assert(
            not problems,
            f"{artifact.path} " + " and ".join(problems) + f"; {artifact.task} saves the header line "
            f"{','.join(artifact.columns)} (any column order).{cause} {hint}",
        )

    return check


def _header_owner(table: Table, artifact: Artifact) -> Artifact | None:
    """Another artifact whose header this table holds, as when a cell saves to the wrong path, or None."""
    saved = set(table.columns)
    for other in ARTIFACTS:
        required = {_fold(column) for column in other.columns}
        allowed = required | {_fold(column) for column in other.optional_columns}
        if other.columns != artifact.columns and required <= saved <= allowed:
            return other
    return None


def rows_check(artifact: Artifact, hint: str) -> Callable[[Path], None]:
    """Each required key appears on exactly one row, and no unexpected key appears."""

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        grouped, unknown = _group(table, artifact, _resolve(table, artifact, root))
        missing = [key for key in artifact.rows if key not in grouped]
        repeated = [key for key, rows in grouped.items() if len(rows) > 1]
        problems = []
        if missing:
            problems.append(f"is missing {_join(_key_name(key, artifact) for key in missing)}")
        if repeated:
            problems.append("lists " + _join(f"{_key_name(key, artifact)} {len(grouped[key])} times" for key in repeated))
        if unknown:
            shown = _join(unknown[:4]) + (f" and {len(unknown) - 4} more" if len(unknown) > 4 else "")
            problems.append(f"also has {shown}")
        names = [_key_name(key, artifact) for key in artifact.rows]
        wanted = f"one row for each of {_expected_keys(names)}" if len(names) > 1 else f"one row, for {names[0]}"
        cause = ""
        if artifact is COEFFICIENTS and problems:
            # A coefficient table missing one predictor's row, with the other terms' coef from that one-predictor fit.
            cause = next((
                f" {slip.cause}" for slip in FORMULA_SLIPS
                if _holds(root, COEFFICIENTS, Slip(slip.cause, {
                    key: {"coef": values["coef"]} for key, values in slip.rows.items() if key in COEFFICIENTS.rows
                }))
            ), "")
        _assert(
            not problems,
            f"{artifact.path} " + "; ".join(problems) + f"; it should hold {wanted}.{cause} Fix it in {artifact.task}: "
            f"{hint}",
        )

    return check


def _cause(misses: list[tuple[tuple, str, str]], artifact: Artifact, slips: tuple[Slip, ...]) -> str:
    """The cause of the first slip whose values every wrong (key, column, cell) holds, or "".

    Values that fit two slips go first to one an earlier check already charged, so one mistake costs points
    once, and last to one an earlier artifact would show but did not.
    """
    here = ARTIFACTS.index(artifact)

    def rank(slip: Slip) -> int:
        if slip.charge in _CHARGED:
            return 0
        shown_earlier = any(VARIANT_ARTIFACTS.index(name) < here for name in VARIANTS.get(slip.charge, {}))
        return 2 if shown_earlier else 1

    for slip in sorted(slips, key=rank):
        if all(
            column in slip.rows.get(key, {})
            and _matches(cell, slip.rows[key][column], artifact.tolerance, column in PERCENT_COLUMNS)
            for key, column, cell in misses
        ):
            return f" {slip.cause}"
    return ""


# Rows whose values all belong under other labels: one mistake, charged by the artifact's first values check.
RELABELED = "check the labels each row is saved under"
# A column left out beside a spare column that matches nothing is a second mistake, so its values check still fails.
ABSENT = "no saved column holds"


def _relabeled(grouped: dict[tuple, list[dict[str, str]]], artifact: Artifact, found: dict[str, str]) -> str:
    """Where every expected row's values sit under another present label, the cause naming them, or "".

    Only a whole permutation counts: each saved row holds, in every value column,
    the values of its own key or of exactly one other, and every key is used once.
    """
    columns = [column for column in artifact.columns if column not in artifact.key and column in found]
    if len(artifact.rows) < 2 or not columns or any(len(grouped.get(key, ())) != 1 for key in artifact.rows):
        return ""

    def fits(row: dict[str, str], key: tuple) -> bool:
        return all(
            _matches(row[found[column]], artifact.rows[key][column], artifact.tolerance, column in PERCENT_COLUMNS)
            for column in columns
        )

    holder = {}
    for key in artifact.rows:
        row = grouped[key][0]
        if fits(row, key):
            holder[key] = key
            continue
        others = [other for other in artifact.rows if other != key and fits(row, other)]
        if len(others) != 1:
            return ""
        holder[key] = others[0]
    if sorted(holder.values()) != sorted(artifact.rows) or all(label == values for label, values in holder.items()):
        return ""
    moved = [
        f"the values for {_key_name(values, artifact)} sit in {_key_name(label, artifact)}'s row"
        for label, values in holder.items() if label != values
    ]
    shown = _listed(moved)
    return f" {shown[0].upper()}{shown[1:]}; {RELABELED}."



def _saved_fit(root: Path, age: float, bmi: float) -> float | None:
    """The fitted SBP from the saved coefficients; a term left out of the formula counts as 0."""
    try:
        table = read_table(root, COEFFICIENTS)
        found = _resolve(table, COEFFICIENTS, root)
        grouped, _ = _group(table, COEFFICIENTS, found)
        saved = {
            term: _number(grouped[(term,)][0][found["coef"]])
            for term in ("Intercept", "age", "bmi") if (term,) in grouped
        }
        if len(saved) >= 2 and all(value is not None for value in saved.values()):
            return saved.get("Intercept", 0.0) + age * saved.get("age", 0.0) + bmi * saved.get("bmi", 0.0)
    except (AssertionError, OSError, KeyError, ValueError):
        pass
    return None


def _accepted(root: Path, artifact: Artifact, key: tuple, rows: list[dict[str, str]], column: str,
              found: dict[str, str]) -> object:
    """What a values check accepts in `column` for this key's saved `rows`.

    That is the expected value, or also one calculated from the student's own
    earlier artifact or from the row's other saved columns, so an earlier
    mistake costs points only once.
    """
    expected = artifact.rows[key][column]
    if artifact is NEW_PATIENT_INTERVALS:
        age, bmi = (_number(rows[0][found[name]]) if name in found else None for name in ("age", "bmi"))
        own = _saved_fit(root, age, bmi) if age is not None and bmi is not None else None
        if own is not None:
            expected = Either((expected, expected + own - NEW_PATIENT_VALUES[NEW_PATIENT]["mean"]))
    if artifact is AVAILABILITY and column == "available" and "hours_after_visit" in found:
        hours = _number(rows[0][found["hours_after_visit"]])
        if hours is not None and not _matches(rows[0][found[column]], expected, artifact.tolerance):
            expected = Flag(hours <= 0)
    if artifact is AVAILABILITY and column == "decision" and "available" in found:
        available = _flag(rows[0][found["available"]])
        if available is not None and not _matches(rows[0][found[column]], expected, artifact.tolerance):
            expected = Starts("keep" if available else "exclude")
    return expected


def _holds(root: Path, artifact: Artifact, slip: Slip) -> bool:
    """Whether the saved artifact holds this slip's value in every cell the slip names."""
    try:
        table = read_table(root, artifact)
        found = _resolve(table, artifact, root)
        grouped, _ = _group(table, artifact, found)
        return all(
            grouped.get(key) and column in found and all(
                _matches(row[found[column]], value, artifact.tolerance) for row in grouped[key]
            )
            for key, values in slip.rows.items() for column, value in values.items()
        )
    except (AssertionError, OSError, KeyError, ValueError, UnicodeDecodeError, csv.Error):
        return False


def values_check(
    artifact: Artifact, columns: tuple[str, ...], hint: str, slips: tuple[Slip, ...] = ()
) -> Callable[[Path], None]:
    """Every saved row for an expected key holds the expected values in `columns`.

    A missing or repeated row costs the rows check, not this one: this check
    compares whatever rows the file does hold for the expected keys. A column
    in `columns` the file does not hold is charged once, by the columns check,
    so this check compares only the columns it can place, unless a spare
    column is left that could have held it: then its values are wrong too.
    When the wrong values are those a common slip leaves, the message names
    its cause.
    """

    def check(root: Path) -> None:
        table = read_table(root, artifact)
        found = _resolve(table, artifact, root)
        present = tuple(column for column in columns if column in found)
        grouped, unknown = _group(table, artifact, found, present)
        # A lone row with neither a recognizable key nor any of these columns gives nothing to credit.
        _assert(
            grouped and (present or not unknown or len(artifact.rows) > 1),
            f"{artifact.path} has no row for {_join(_key_name(key, artifact) for key in artifact.rows)}, so its "
            f"{_join(columns)} values cannot be compared; the rows check says what to fix in {artifact.task}.",
        )
        # A formula slip is named downstream only when the saved coefficients show it: a fit without bmi
        # counts every patient in its bands, as "size" does, and only the coefficients tell the two apart.
        named = tuple(slip for slip in slips if slip.charge not in FORMULA_CHARGES or any(
            _holds(root, COEFFICIENTS, formula) for formula in FORMULA_SLIPS if formula.charge == slip.charge))
        wrong = []
        misses = []
        for key, rows in grouped.items():
            for column in present:
                expected = _accepted(root, artifact, key, rows, column, found)
                cells = [
                    row[found[column]] for row in rows
                    if not _matches(row[found[column]], expected, artifact.tolerance, column in PERCENT_COLUMNS)
                ]
                misses += [(key, column, cell) for cell in cells]
                given = sorted({_given(cell) for cell in cells})
                if given:
                    wrong.append(f"{_key_name(key, artifact)} has {column} {_join(given)}, expected {_show(artifact.rows[key][column])}")
        # A column left out is charged once, by the columns check. One renamed with values that match nothing stays
        # unplaced beside a spare column: its wrong values are a second mistake, charged here.
        allowed = {_fold(column) for column in (*artifact.columns, *artifact.optional_columns)}
        spare = [column for column in table.columns if column not in found.values() and column not in allowed]
        absent = [column for column in columns if column not in found] if spare else []
        if absent:
            first = next(key for key in artifact.rows if key in grouped)
            one = len(absent) == 1
            _assert(
                False,
                f"{artifact.path}: {ABSENT} {_join(absent)}; "
                f"{_join(column or 'a column with no header' for column in spare)} "
                f"{'does' if len(spare) == 1 else 'do'} not match {'it' if one else 'them'} (expected "
                + _join(f"{column} {_show(artifact.rows[first][column])}" for column in absent)
                + f" for {_key_name(first, artifact)}, for example)."
                + (f" Also, {_listed(wrong)}." + (_relabeled(grouped, artifact, found) or _cause(misses, artifact, named))
                   if wrong else "")
                + f" Fix it in {artifact.task}: {hint}",
            )
        _assert(
            not wrong,
            f"{artifact.path}: " + _listed(wrong) + "."
            + (_relabeled(grouped, artifact, found) or _cause(misses, artifact, named))
            + f" Fix it in {artifact.task}: {hint}",
        )

    return check


def _png_complete(data: bytes) -> bool:
    """Check PNG chunks and compressed pixels without an optional image library."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        return False
    position, chunks, compressed = 8, [], []
    palette = None
    while position + 12 <= len(data):
        size = int.from_bytes(data[position:position + 4], "big")
        kind = data[position + 4:position + 8]
        payload = data[position + 8:position + 8 + size]
        end = position + 12 + size
        if end > len(data) or zlib.crc32(kind + payload) != int.from_bytes(data[end - 4:end], "big"):
            return False
        chunks.append(kind)
        if kind == b"IHDR":
            if len(chunks) != 1 or size != 13:
                return False
            header = payload
        if kind == b"PLTE":
            if palette is not None or compressed or not size or size % 3 or size > 768:
                return False
            palette = payload
        if kind == b"IDAT":
            compressed.append(payload)
        position = end
        if kind == b"IEND":
            if size != 0 or position != len(data):
                return False
            break
    if not chunks or chunks[0] != b"IHDR" or chunks[-1] != b"IEND" or not compressed:
        return False
    width, height = int.from_bytes(header[:4], "big"), int.from_bytes(header[4:8], "big")
    depth, color, compression, filtering, interlace = header[8:]
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}
    allowed_depths = {0: (1, 2, 4, 8, 16), 2: (8, 16), 3: (1, 2, 4, 8), 4: (8, 16), 6: (8, 16)}
    if not width or not height or depth not in allowed_depths.get(color, ()) or compression or filtering or interlace > 1:
        return False
    if color == 3 and (palette is None or len(palette) // 3 > 2 ** depth):
        return False
    try:
        decoder = zlib.decompressobj()
        pixels = decoder.decompress(b"".join(compressed)) + decoder.flush()
        if not decoder.eof or decoder.unused_data:
            return False
    except zlib.error:
        return False
    # Adam7 interlacing stores seven smaller images, each with its own filtered rows.
    passes = ((0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8), (2, 0, 4, 4),
              (0, 2, 2, 4), (1, 0, 2, 2), (0, 1, 1, 2)) if interlace else ((0, 0, 1, 1),)
    position = 0
    for x, y, dx, dy in passes:
        across, down = max(0, (width - x + dx - 1) // dx), max(0, (height - y + dy - 1) // dy)
        if not across or not down:
            continue
        stride = 1 + (across * channels[color] * depth + 7) // 8
        end = position + down * stride
        if end > len(pixels) or any(pixels[offset] > 4 for offset in range(position, end, stride)):
            return False
        position = end
    return position == len(pixels)


def png_check(root: Path) -> None:
    """The residual plot is saved as a PNG image."""
    path = _artifact_path(root, RESIDUAL_PLOT)
    _assert(path is not None, _missing(root, RESIDUAL_PLOT, "Task 1.3"))
    data = path.read_bytes()
    start = data[:8]
    _assert(
        _png_complete(data),
        f"{RESIDUAL_PLOT} is not a complete PNG image (its first bytes are {start!r}); in Task 1.3 save the figure with "
        f'fig.savefig(RESIDUAL_PLOT_PATH), keeping the .png ending so matplotlib writes a PNG.',
    )



FIT_HINT = 'fit smf.ols("sbp ~ age + bmi", data=patients) on data/bp_patients.parquet.'
FEATURE_HINT = (
    "fit on the fit rows of Task 2.2 with the six features Task 2.1 keeps and the target sbp_followup"
)
TEST_HINT = (
    "predict test[FEATURES] once with the pipeline Task 3.1 fitted, unchanged, and compare with test[TARGET]."
)

CHECKS = (
    # Task 1.1: output/ols_coefficients.csv
    Check(
        "coefficients: columns",
        columns_check(
            COEFFICIENTS,
            "Build the table from results.params, results.bse, and the two columns of results.conf_int(), name "
            'its index "term", and save it with the index so the terms become the first column.',
        ),
    ),
    Check(
        "coefficients: one row per term",
        rows_check(COEFFICIENTS, "the model sbp ~ age + bmi has three terms: Intercept, age, and bmi."),
    ),
    Check(
        "coefficients: coef values",
        values_check(COEFFICIENTS, ("coef",), f"coef is results.params; {FIT_HINT}", FORMULA_SLIPS),
    ),
    Check(
        "coefficients: std_err values",
        values_check(COEFFICIENTS, ("std_err",), f"std_err is results.bse; {FIT_HINT}", FORMULA_SLIPS),
    ),
    Check(
        "coefficients: confidence interval values",
        values_check(
            COEFFICIENTS, ("ci_lower", "ci_upper"),
            "ci_lower and ci_upper are columns 0 and 1 of results.conf_int(alpha=0.05), the 95% interval.",
            FORMULA_SLIPS,
        ),
    ),
    # Task 1.2: output/new_patient_intervals.csv
    Check(
        "new-patient intervals: columns",
        columns_check(
            NEW_PATIENT_INTERVALS,
            "Put the new patient's age and bmi beside summary_frame(alpha=0.05) with pd.concat([...], axis=1); "
            "keeping mean_se is fine.",
        ),
    ),
    Check(
        "new-patient intervals: one row for the new patient",
        rows_check(NEW_PATIENT_INTERVALS, "predict for one new patient, age 60 with BMI 31.0."),
    ),
    Check(
        "new-patient intervals: mean",
        values_check(
            NEW_PATIENT_INTERVALS, ("mean",),
            "mean is the fitted SBP from results.get_prediction(new_patient).summary_frame(alpha=0.05).",
            INTERVAL_SLIPS,
        ),
    ),
    Check(
        "new-patient intervals: mean-response interval",
        values_check(
            NEW_PATIENT_INTERVALS, ("mean_ci_lower", "mean_ci_upper"),
            "mean_ci_lower and mean_ci_upper bound the average SBP of patients like this one.",
            INTERVAL_SLIPS,
        ),
    ),
    Check(
        "new-patient intervals: prediction interval",
        values_check(
            NEW_PATIENT_INTERVALS, ("obs_ci_lower", "obs_ci_upper"),
            "obs_ci_lower and obs_ci_upper bound one such patient's SBP; this interval is the wider one.",
            INTERVAL_SLIPS,
        ),
    ),
    # Task 1.3: output/residual_bands.csv and output/residuals_vs_fitted.png
    Check(
        "residual bands: columns",
        columns_check(
            RESIDUAL_BANDS,
            "Group the residuals by age_band with named aggregation: patients, mean_residual, and sd_residual.",
        ),
    ),
    Check(
        "residual bands: one row per age band",
        rows_check(RESIDUAL_BANDS, "cut age with AGE_BINS and AGE_LABELS from the scaffold; that makes six bands."),
    ),
    Check(
        "residual bands: patients values",
        values_check(
            RESIDUAL_BANDS, ("patients",),
            'patients is the "count" of residuals in the band: the patients the model used.', BAND_SLIPS,
        ),
    ),
    Check(
        "residual bands: mean_residual values",
        values_check(
            RESIDUAL_BANDS, ("mean_residual",),
            f'mean_residual is the "mean" of results.resid in the band; {FIT_HINT}', BAND_SLIPS,
        ),
    ),
    Check(
        "residual bands: sd_residual values",
        values_check(
            RESIDUAL_BANDS, ("sd_residual",), 'sd_residual is the "std" of results.resid in the band.', BAND_SLIPS,
        ),
    ),
    Check("residual plot: PNG image", png_check),
    # Task 2.1: output/availability_decisions.csv
    Check(
        "availability: columns",
        columns_check(
            AVAILABILITY,
            "Start from a .copy() of the supplied candidates table and add the columns available and decision.",
        ),
    ),
    Check(
        "availability: one row per candidate",
        rows_check(AVAILABILITY, "keep one row for each candidate in data/feature_candidates.parquet."),
    ),
    Check(
        "availability: hours_after_visit values",
        values_check(
            AVAILABILITY, ("hours_after_visit",),
            "hours_after_visit is copied unchanged from data/feature_candidates.parquet.",
        ),
    ),
    Check(
        "availability: available values",
        values_check(
            AVAILABILITY, ("available",),
            "available is hours_after_visit <= 0: True when the feature is known by the end of the visit.",
            AVAILABILITY_SLIPS,
        ),
    ),
    Check(
        "availability: decision values",
        values_check(
            AVAILABILITY, ("decision",),
            'decision is "Keep" where available is True and "Exclude (leakage)" where it is False; any label '
            "starting with keep or exclude counts.",
        ),
    ),
    # Task 2.2: output/split_summary.csv
    Check(
        "split summary: columns",
        columns_check(SPLIT, "Build one row per partition with its len() and the min() and max() of its followup_time."),
    ),
    Check(
        "split summary: one row per partition",
        rows_check(SPLIT, "name the four partitions train, fit, validation, and test."),
    ),
    Check(
        "split summary: row_count values",
        values_check(
            SPLIT, ("row_count",),
            "split on followup_time: train is before VALIDATION_START, validation runs up to TEST_START, test is "
            "from TEST_START on, and fit keeps the training rows whose followup_time is on or before "
            "valid['visit_time'].min().",
            SPLIT_SLIPS,
        ),
    ),
    Check(
        "split summary: first_target_time values",
        values_check(
            SPLIT, ("first_target_time",),
            "first_target_time is the partition's followup_time.min(), the full timestamp.", SPLIT_SLIPS,
        ),
    ),
    Check(
        "split summary: last_target_time values",
        values_check(
            SPLIT, ("last_target_time",),
            "last_target_time is the partition's followup_time.max(), the full timestamp.", SPLIT_SLIPS,
        ),
    ),
    # Task 3.1: output/validation_metrics.csv
    Check(
        "validation metrics: columns",
        columns_check(VALIDATION, "Save one row per approach with its MAE, RMSE, and R² on the validation rows."),
    ),
    Check(
        "validation metrics: one row per approach",
        rows_check(VALIDATION, "name the three approaches mean_baseline, persistence, and linear_pipeline."),
    ),
    Check(
        "validation metrics: mae values",
        values_check(
            VALIDATION, ("mae",),
            f"mae is mean_absolute_error(valid[TARGET], predicted); {FEATURE_HINT}, and persistence predicts "
            "sbp_today.",
            VALIDATION_SLIPS,
        ),
    ),
    Check(
        "validation metrics: rmse values",
        values_check(
            VALIDATION, ("rmse",), f"rmse is np.sqrt(mean_squared_error(valid[TARGET], predicted)); {FEATURE_HINT}.",
            VALIDATION_SLIPS,
        ),
    ),
    Check(
        "validation metrics: r2 values",
        values_check(
            VALIDATION, ("r2",), f"r2 is r2_score(valid[TARGET], predicted), actual first; {FEATURE_HINT}.",
            VALIDATION_SLIPS,
        ),
    ),
    # Task 3.2: output/test_metrics.csv and output/clinic_test_errors.csv
    Check(
        "test metrics: columns",
        columns_check(TEST_METRICS, "Save one row with the frozen approach's MAE, RMSE, and R² on the test rows."),
    ),
    Check(
        "test metrics: one row for the frozen approach",
        rows_check(TEST_METRICS, "validation chose linear_pipeline, so the test rows are scored once, for it only."),
    ),
    Check(
        "test metrics: mae value",
        values_check(TEST_METRICS, ("mae",), f"mae is mean_absolute_error(test[TARGET], test_predicted); {TEST_HINT}",
                     TEST_SLIPS),
    ),
    Check(
        "test metrics: rmse value",
        values_check(TEST_METRICS, ("rmse",), f"rmse is the square root of the mean squared error; {TEST_HINT}",
                     TEST_SLIPS),
    ),
    Check(
        "test metrics: r2 value",
        values_check(TEST_METRICS, ("r2",), f"r2 is r2_score(test[TARGET], test_predicted); {TEST_HINT}", TEST_SLIPS),
    ),
    Check(
        "clinic errors: columns",
        columns_check(
            CLINIC_ERRORS,
            "Group the test errors by clinic with named aggregation: test_visits, mae, and mean_error.",
        ),
    ),
    Check(
        "clinic errors: one row per clinic",
        rows_check(CLINIC_ERRORS, "group the test rows by clinic: eight clinics."),
    ),
    Check(
        "clinic errors: test_visits values",
        values_check(CLINIC_ERRORS, ("test_visits",), 'test_visits is the "size" of each clinic\'s test rows.',
                     CLINIC_SLIPS),
    ),
    Check(
        "clinic errors: mae values",
        values_check(CLINIC_ERRORS, ("mae",), f"mae is the mean of the absolute errors in the clinic; {TEST_HINT}",
                     CLINIC_SLIPS),
    ),
    Check(
        "clinic errors: mean_error values",
        values_check(
            CLINIC_ERRORS, ("mean_error",),
            f"mean_error is the mean of the errors, actual minus predicted, in the clinic; {TEST_HINT}", CLINIC_SLIPS,
        ),
    ),
    # Task 3.3: output/readmission_metrics.csv
    Check(
        "readmission metrics: columns",
        columns_check(READMISSION, "Save one row per rule with its flagged count, accuracy, precision, and recall."),
    ),
    Check(
        "readmission metrics: one row per rule",
        rows_check(READMISSION, "name the rows flag_no_one, lace_13_plus, and lace_10_plus."),
    ),
    Check(
        "readmission metrics: flagged values",
        values_check(READMISSION, ("flagged",), "flagged is the number of discharges the rule flags, flag.sum().",
                     READMISSION_SLIPS),
    ),
    Check(
        "readmission metrics: accuracy values",
        values_check(READMISSION, ("accuracy",), "accuracy is accuracy_score(readmitted_30d, flag).",
                     READMISSION_SLIPS),
    ),
    Check(
        "readmission metrics: precision values",
        values_check(
            READMISSION, ("precision",),
            "precision is precision_score(readmitted_30d, flag, zero_division=0), which is 0 for a rule that "
            "flags no one.",
            READMISSION_SLIPS,
        ),
    ),
    Check(
        "readmission metrics: recall values",
        values_check(READMISSION, ("recall",), "recall is recall_score(readmitted_30d, flag).", READMISSION_SLIPS),
    ),
)

def run_checks(root: Path) -> list[tuple[str, str | None]]:
    """Return one (check name, problem or None) pair per check; every check runs."""
    _REPORTED_OUTSIDE.clear()
    _CHARGED.clear()
    results = []
    charged = _CHARGED
    slips = SLIPS
    for check in CHECKS:
        try:
            check.action(Path(root))
        except (AssertionError, OSError, ValueError, UnicodeDecodeError, csv.Error) as error:
            message = str(error)
            slip = next((slip for slip in slips if slip.cause in message), None)
            mistake = slip and (slip.charge or slip.cause)
            if not mistake and RELABELED in message:
                mistake = message.split()[0].rstrip(":") + RELABELED
            if mistake and mistake in charged and ABSENT not in message:
                results.append((check.name, None))
            else:
                results.append((check.name, message))
                if mistake:
                    charged.add(mistake)
        else:
            results.append((check.name, None))
    return results
