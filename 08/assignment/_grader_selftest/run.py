# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "ipykernel==6.29.5",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.0.2",
#   "pandas==3.0.3",
# ]
# ///

"""Adversarial author release harness for Assignment 08."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from classroom50_grader import (
    ARTIFACT_SHA256,
    REQUIRED_CONTEXT_ENV,
    _execute_notebook,
    _execute_setup_only,
    grade_submission,
)


ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
UTC_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


LOAD_SOURCE = r'''REQUEST_DTYPES = {
    'request_id': 'string',
    'agent_id': 'string',
    'resolution_minutes': 'int64',
    'satisfaction_score': 'Int64',
}
support_requests = pd.read_csv(FIXTURE_PATH, dtype=REQUEST_DTYPES)
CENTER_LEVELS = list(fixture_manifest['center_levels'])
CHANNEL_LEVELS = list(fixture_manifest['channel_levels'])
support_requests['center'] = pd.Categorical(
    support_requests['center'], categories=CENTER_LEVELS, ordered=True
)
support_requests['channel'] = pd.Categorical(
    support_requests['channel'], categories=CHANNEL_LEVELS, ordered=True
)
assert support_requests.shape == (15, 6)
assert support_requests.columns.tolist() == EXPECTED_COLUMNS
assert support_requests['request_id'].notna().all()
assert not support_requests['request_id'].duplicated().any()
assert support_requests[['center', 'channel']].notna().all().all()
assert int(support_requests['satisfaction_score'].isna().sum()) == 3
assert str(support_requests['request_id'].dtype) == 'string'
assert str(support_requests['center'].dtype) == 'category'
assert str(support_requests['agent_id'].dtype) == 'string'
assert str(support_requests['channel'].dtype) == 'category'
assert str(support_requests['resolution_minutes'].dtype) == 'int64'
assert str(support_requests['satisfaction_score'].dtype) == 'Int64'
source_snapshot = support_requests.copy(deep=True)
'''


TASK1_MARKDOWN = """## Task 1 — grain and count contract

The input row grain is one synthetic support request. The grouping key is
`center`, so the grouping unit is one observed support center. I predict three
observed groups—Central, Harbor, and Ridge—because the unused Valley category
has no input rows and `observed=True` omits it. The output row grain is one row
per observed support center.

`size` answers the request question because it counts source rows even when a
different field is missing. Selected `satisfaction_score.count()` answers the
recorded-score question because it excludes missing scores in that field.
Selected `agent_id.nunique()` answers the distinct-agent question because an
agent can appear on more than one request.
"""


TASK1_VALUES_SOURCE = r'''input_row_grain = 'one row per synthetic support request'
grouping_key = ['center']
grouping_unit = 'one observed support center'
predicted_group_identities = ['Central', 'Harbor', 'Ridge']
predicted_group_count = 3
observed_category_policy = True
output_row_grain = 'one row per observed support center'
count_plan = {
    'request_count': {
        'question': 'How many support-request rows were recorded?',
        'operation': 'size',
    },
    'satisfaction_count': {
        'question': 'How many requests have a recorded satisfaction score?',
        'operation': 'count',
    },
    'unique_agent_count': {
        'question': 'How many distinct agents appear?',
        'operation': 'nunique',
    },
}
'''


COUNT_FUNCTION_SOURCE = r'''def build_count_summary(request_table):
    """Return three center count semantics without mutating the input."""
    grouped = request_table.groupby(
        'center', observed=True, sort=True, dropna=True
    )
    result = pd.concat(
        [
            grouped.size().rename('request_count'),
            grouped['satisfaction_score'].count().rename('satisfaction_count'),
            grouped['agent_id'].nunique(dropna=True).rename('unique_agent_count'),
        ],
        axis='columns',
    ).reset_index()
    return result[
        ['center', 'request_count', 'satisfaction_count', 'unique_agent_count']
    ]
'''


TASK1_RUN_SOURCE = r'''diagnostic_groupby = support_requests.groupby(
    grouping_key, observed=True, sort=True, dropna=True
)
observed_group_identities = diagnostic_groupby.size().index.tolist()
assert observed_group_identities == predicted_group_identities
assert diagnostic_groupby.ngroups == predicted_group_count == 3
center_count_summary = build_count_summary(support_requests)
assert center_count_summary.columns.tolist() == [
    'center', 'request_count', 'satisfaction_count', 'unique_agent_count'
]
assert center_count_summary.shape == (3, 4)
assert center_count_summary['center'].tolist() == predicted_group_identities
assert center_count_summary['request_count'].tolist() == [5, 5, 5]
assert center_count_summary['satisfaction_count'].tolist() == [4, 3, 5]
assert center_count_summary['unique_agent_count'].tolist() == [3, 2, 3]
assert str(center_count_summary['request_count'].dtype) == 'int64'
assert str(center_count_summary['satisfaction_count'].dtype) == 'Int64'
assert str(center_count_summary['unique_agent_count'].dtype) == 'int64'
assert int(center_count_summary['request_count'].sum()) == len(support_requests) == 15
assert 'Valley' not in center_count_summary['center'].tolist()
pd.testing.assert_frame_equal(support_requests, source_snapshot)
'''


TASK1_SAVE_SOURCE = r'''center_count_summary.to_csv(
    CENTER_COUNT_OUTPUT,
    index=False,
    encoding='utf-8',
    lineterminator='\n',
    na_rep='',
)
center_count_round_trip = pd.read_csv(
    CENTER_COUNT_OUTPUT,
    dtype={
        'center': 'string',
        'request_count': 'int64',
        'satisfaction_count': 'Int64',
        'unique_agent_count': 'int64',
    },
)
expected_count_serialized = center_count_summary.copy()
expected_count_serialized['center'] = expected_count_serialized['center'].astype('string')
pd.testing.assert_frame_equal(center_count_round_trip, expected_count_serialized)
'''


TASK1_EXPLAIN = """### Task 1 explanation

Three satisfaction values are missing, so selected-column `count` reports fewer
records than `size`, while `size` still counts every request row. Agents repeat
within centers, so distinct-agent `nunique` is smaller than request count.
Valley is a declared category but has no input row; `observed=True` therefore
keeps it out of the observed-group result.
"""


CENTER_SUMMARY_FUNCTION_SOURCE = r'''def build_center_summary(request_table):
    """Return one flat named-aggregation row per observed center."""
    return request_table.groupby(
        'center',
        as_index=False,
        observed=True,
        sort=True,
        dropna=True,
    ).agg(
        request_count=('request_id', 'size'),
        satisfaction_count=('satisfaction_score', 'count'),
        unique_agent_count=('agent_id', 'nunique'),
        total_resolution_minutes=('resolution_minutes', 'sum'),
        mean_resolution_minutes=('resolution_minutes', 'mean'),
    )
'''


CONTEXT_FUNCTION_SOURCE = r'''def add_center_context(request_table):
    """Return a copied request table with same-index center mean context."""
    result = request_table.copy(deep=True)
    center_means = request_table['resolution_minutes'].groupby(
        request_table['center'], observed=True, sort=True, dropna=True
    ).transform('mean')
    result['center_mean_resolution_minutes'] = center_means
    result['difference_from_center_mean'] = (
        result['resolution_minutes'] - center_means
    )
    return result
'''


TWO_KEY_FUNCTION_SOURCE = r'''def build_center_channel_summary(request_table):
    """Return one flat row per observed center-channel group."""
    return request_table.groupby(
        ['center', 'channel'],
        as_index=False,
        observed=True,
        sort=True,
        dropna=True,
    ).agg(
        request_count=('request_id', 'size'),
        mean_resolution_minutes=('resolution_minutes', 'mean'),
    )
'''


TASK2_RUN_SOURCE = r'''center_summary = build_center_summary(support_requests)
requests_with_context = add_center_context(support_requests)
center_channel_summary = build_center_channel_summary(support_requests)

assert center_summary.columns.tolist() == [
    'center', 'request_count', 'satisfaction_count', 'unique_agent_count',
    'total_resolution_minutes', 'mean_resolution_minutes',
]
assert center_summary.shape == (3, 6)
assert center_summary['center'].tolist() == ['Central', 'Harbor', 'Ridge']
assert center_summary['request_count'].tolist() == [5, 5, 5]
assert center_summary['satisfaction_count'].tolist() == [4, 3, 5]
assert center_summary['unique_agent_count'].tolist() == [3, 2, 3]
assert center_summary['total_resolution_minutes'].tolist() == [180, 200, 210]
assert center_summary['mean_resolution_minutes'].tolist() == [36.0, 40.0, 42.0]
assert str(center_summary['request_count'].dtype) == 'int64'
assert str(center_summary['satisfaction_count'].dtype) == 'Int64'
assert str(center_summary['unique_agent_count'].dtype) == 'int64'
assert str(center_summary['total_resolution_minutes'].dtype) == 'int64'
assert str(center_summary['mean_resolution_minutes'].dtype) == 'float64'

assert requests_with_context.columns.tolist() == [
    *EXPECTED_COLUMNS,
    'center_mean_resolution_minutes',
    'difference_from_center_mean',
]
assert requests_with_context.shape == (15, 8)
assert requests_with_context.index.equals(support_requests.index)
assert requests_with_context['center_mean_resolution_minutes'].tolist() == (
    [36.0] * 5 + [40.0] * 5 + [42.0] * 5
)
assert requests_with_context['difference_from_center_mean'].tolist() == [
    -6.0, -16.0, 9.0, 24.0, -11.0,
    0.0, -5.0, 10.0, 5.0, -10.0,
    13.0, -17.0, 23.0, -7.0, -12.0,
]
assert str(requests_with_context['center_mean_resolution_minutes'].dtype) == 'float64'
assert str(requests_with_context['difference_from_center_mean'].dtype) == 'float64'

assert center_channel_summary.columns.tolist() == [
    'center', 'channel', 'request_count', 'mean_resolution_minutes'
]
assert center_channel_summary.shape == (8, 4)
assert center_channel_summary[['center', 'channel']].values.tolist() == [
    ['Central', 'Email'], ['Central', 'Phone'], ['Central', 'Chat'],
    ['Harbor', 'Email'], ['Harbor', 'Chat'],
    ['Ridge', 'Email'], ['Ridge', 'Phone'], ['Ridge', 'Chat'],
]
assert center_channel_summary['request_count'].tolist() == [2, 1, 2, 3, 2, 1, 2, 2]
assert center_channel_summary['mean_resolution_minutes'].tolist() == [
    37.5, 60.0, 22.5, 40.0, 40.0, 35.0, 60.0, 27.5
]
assert int(center_channel_summary['request_count'].sum()) == 15
assert ['Harbor', 'Phone'] not in center_channel_summary[['center', 'channel']].values.tolist()
assert 'Valley' not in center_channel_summary['center'].tolist()
assert not isinstance(center_channel_summary.index, pd.MultiIndex)

aggregate_rows = len(center_summary)
source_rows = len(support_requests)
transform_rows = len(requests_with_context)
transform_index_preserved = requests_with_context.index.equals(support_requests.index)
assert aggregate_rows == 3
assert source_rows == transform_rows == 15
assert transform_index_preserved is True
pd.testing.assert_frame_equal(support_requests, source_snapshot)
'''


TASK2_SAVE_SOURCE = r'''center_summary.to_csv(
    CENTER_SUMMARY_OUTPUT, index=False, encoding='utf-8', lineterminator='\n', na_rep=''
)
requests_with_context.to_csv(
    CONTEXT_OUTPUT, index=False, encoding='utf-8', lineterminator='\n', na_rep=''
)
center_channel_summary.to_csv(
    CENTER_CHANNEL_OUTPUT, index=False, encoding='utf-8', lineterminator='\n', na_rep=''
)

center_summary_round_trip = pd.read_csv(
    CENTER_SUMMARY_OUTPUT,
    dtype={
        'center': 'string', 'request_count': 'int64',
        'satisfaction_count': 'Int64', 'unique_agent_count': 'int64',
        'total_resolution_minutes': 'int64', 'mean_resolution_minutes': 'float64',
    },
)
context_round_trip = pd.read_csv(
    CONTEXT_OUTPUT,
    dtype={
        'request_id': 'string', 'center': 'string', 'agent_id': 'string',
        'channel': 'string', 'resolution_minutes': 'int64',
        'satisfaction_score': 'Int64', 'center_mean_resolution_minutes': 'float64',
        'difference_from_center_mean': 'float64',
    },
)
center_channel_round_trip = pd.read_csv(
    CENTER_CHANNEL_OUTPUT,
    dtype={
        'center': 'string', 'channel': 'string', 'request_count': 'int64',
        'mean_resolution_minutes': 'float64',
    },
)
expected_center_serialized = center_summary.copy()
expected_center_serialized['center'] = expected_center_serialized['center'].astype('string')
expected_context_serialized = requests_with_context.copy()
expected_context_serialized['center'] = expected_context_serialized['center'].astype('string')
expected_context_serialized['channel'] = expected_context_serialized['channel'].astype('string')
expected_two_key_serialized = center_channel_summary.copy()
expected_two_key_serialized['center'] = expected_two_key_serialized['center'].astype('string')
expected_two_key_serialized['channel'] = expected_two_key_serialized['channel'].astype('string')
pd.testing.assert_frame_equal(center_summary_round_trip, expected_center_serialized)
pd.testing.assert_frame_equal(context_round_trip, expected_context_serialized)
pd.testing.assert_frame_equal(center_channel_round_trip, expected_two_key_serialized)
'''


TASK2_EXPLAIN = """### Task 2 explanation

`center_summary` changes the output grain to one row per observed center, while
`requests_with_context` stays at one row per request. A transform must match both
the source length and exact index labels so each request receives its own
center's mean. One row of the two-key result represents one observed
center--channel group. Positional assignment of three center aggregates to
fifteen requests would neither supply fifteen values nor identify which request
belongs to which center.
"""


PIVOT_VALUES_SOURCE = r'''pivot_spec = {
    'index': 'center',
    'columns': 'channel',
    'values': 'resolution_minutes',
    'aggfunc': 'mean',
    'observed': True,
    'sort': True,
    'dropna': True,
}
pivot_display_row_grain = 'one observed support center'
pivot_cell_grain = 'one observed center-channel group'
absent_combination = ['Harbor', 'Phone']
absent_combination_meaning = 'no input row for this center-channel combination'
'''


PIVOT_FUNCTION_SOURCE = r'''def build_resolution_pivot(request_table):
    """Return the required unfilled mean-resolution aggregating pivot."""
    return pd.pivot_table(
        request_table,
        index='center',
        columns='channel',
        values='resolution_minutes',
        aggfunc='mean',
        observed=True,
        sort=True,
        dropna=True,
    )
'''


TASK3_RUN_SOURCE = r'''resolution_pivot = build_resolution_pivot(support_requests)
pivot_reference = build_center_channel_summary(support_requests)
assert resolution_pivot.index.tolist() == ['Central', 'Harbor', 'Ridge']
assert resolution_pivot.columns.tolist() == ['Email', 'Phone', 'Chat']
assert resolution_pivot.shape == (3, 3)
assert int(resolution_pivot.notna().sum().sum()) == 8
assert pd.isna(resolution_pivot.loc['Harbor', 'Phone'])
assert 'Valley' not in resolution_pivot.index.tolist()
assert resolution_pivot.loc['Central', 'Email'] == 37.5
assert resolution_pivot.loc['Ridge', 'Phone'] == 60.0
for reference_row in pivot_reference.itertuples(index=False):
    assert (
        resolution_pivot.loc[reference_row.center, reference_row.channel]
        == reference_row.mean_resolution_minutes
    )
assert not (resolution_pivot == 0).any().any()
pd.testing.assert_frame_equal(support_requests, source_snapshot)
'''


TASK3_SAVE_SOURCE = r'''pivot_serialized = resolution_pivot.rename_axis(
    index='center', columns=None
).reset_index()
pivot_serialized.to_csv(
    PIVOT_OUTPUT,
    index=False,
    encoding='utf-8',
    lineterminator='\n',
    na_rep='',
)
pivot_round_trip = pd.read_csv(
    PIVOT_OUTPUT,
    dtype={'center': 'string', 'Email': 'float64', 'Phone': 'float64', 'Chat': 'float64'},
)
expected_pivot_serialized = pivot_serialized.copy()
expected_pivot_serialized['center'] = expected_pivot_serialized['center'].astype('string')
pd.testing.assert_frame_equal(pivot_round_trip, expected_pivot_serialized)
assert int(pivot_round_trip.isna().sum().sum()) == 1
assert pd.isna(pivot_round_trip.loc[pivot_round_trip['center'].eq('Harbor'), 'Phone'].item())
assert not (pivot_round_trip[['Email', 'Phone', 'Chat']] == 0).any().any()
'''


TASK3_EXPLAIN = """### Task 3 explanation

The pivot uses center as its row index, channel as its columns, resolution
minutes as its values, mean as its aggregation function, and `observed=True` as
its category policy. A displayed row is one observed center, but each populated
cell is one observed center--channel group. Cell-for-cell equality with the flat
GroupBy result verifies that the wide layout did not change the grouped values.
Harbor--Phone is absent because no request has that combination; replacing the
missing cell with zero would invent a measured zero-minute result.
"""


SYNTHESIS_MARKDOWN = """## Synthesis

`size`, `count`, `nunique`, and named aggregation reduce many request rows to one
row per group, so they change output grain. `transform` broadcasts group context
back to the same index and therefore preserves request grain. An aggregating
pivot displays one center per row, but each populated cell still represents the
two-key center--channel group made explicit in the flat GroupBy reference.
"""


CORRECT_SOURCES = {
    "a08-load": LOAD_SOURCE,
    "a08-task1-contract": TASK1_MARKDOWN,
    "a08-task1-values": TASK1_VALUES_SOURCE,
    "a08-count-function": COUNT_FUNCTION_SOURCE,
    "a08-task1-run": TASK1_RUN_SOURCE,
    "a08-task1-save": TASK1_SAVE_SOURCE,
    "a08-task1-explain": TASK1_EXPLAIN,
    "a08-center-summary-function": CENTER_SUMMARY_FUNCTION_SOURCE,
    "a08-context-function": CONTEXT_FUNCTION_SOURCE,
    "a08-two-key-function": TWO_KEY_FUNCTION_SOURCE,
    "a08-task2-run": TASK2_RUN_SOURCE,
    "a08-task2-save": TASK2_SAVE_SOURCE,
    "a08-task2-explain": TASK2_EXPLAIN,
    "a08-pivot-values": PIVOT_VALUES_SOURCE,
    "a08-pivot-function": PIVOT_FUNCTION_SOURCE,
    "a08-task3-run": TASK3_RUN_SOURCE,
    "a08-task3-save": TASK3_SAVE_SOURCE,
    "a08-task3-explain": TASK3_EXPLAIN,
    "a08-synthesis": SYNTHESIS_MARKDOWN,
}


def _copy_starter(destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {
            "_grader_selftest", ".venv", "__pycache__", ".ipynb_checkpoints",
            ".pytest_cache", "result.json",
        }.intersection(names)

    shutil.copytree(ASSIGNMENT_DIR, destination, ignore=ignore)


def _load_notebook(root: Path) -> dict:
    return json.loads((root / "assignment.ipynb").read_text(encoding="utf-8"))


def _write_notebook(root: Path, notebook: dict) -> None:
    (root / "assignment.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _set_cell_source(notebook: dict, cell_id: str, source: str) -> None:
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    cell["source"] = source.splitlines(keepends=True)
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    if cell["cell_type"] == "code":
        cell["execution_count"] = None
        cell["outputs"] = []


def _materialize_correct(root: Path) -> None:
    notebook = _load_notebook(root)
    for cell_id, source in CORRECT_SOURCES.items():
        _set_cell_source(notebook, cell_id, source)
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    _write_notebook(root, notebook)
    _execute_notebook(root, root)


def _replace_cell_text(root: Path, cell_id: str, old: str, new: str) -> None:
    notebook = _load_notebook(root)
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    assert old in source, f"mutation target not found in {cell_id}: {old}"
    _set_cell_source(notebook, cell_id, source.replace(old, new))
    _write_notebook(root, notebook)


def _run_public(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "check_assignment.py")],
        cwd=root,
        env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def _assert_result_schema(result: dict, expected_score: int | None = None) -> None:
    required = {
        "schema", "classroom", "assignment", "submission", "commit",
        "release", "review", "datetime", "score", "max-score", "tests",
    }
    optional = {"owner", "assignment_type", "submitted_by"}
    assert required.issubset(result)
    assert set(result).issubset(required | optional)
    assert result["schema"] == "classroom50/result/v1"
    assert all(
        isinstance(result[field], str) and result[field]
        for field in (
            "classroom", "assignment", "submission", "commit", "release",
            "review", "datetime",
        )
    )
    assert isinstance(result["score"], int) and result["max-score"] == 90
    assert UTC_DATETIME.fullmatch(result["datetime"])
    assert isinstance(result["tests"], list) and len(result["tests"]) == 5
    assert sum(test["score"] for test in result["tests"]) == result["score"]
    assert sum(test["max-score"] for test in result["tests"]) == 90
    for test in result["tests"]:
        assert set(test) == {"test-name", "passed", "score", "max-score"}
        assert isinstance(test["test-name"], str) and test["test-name"]
        assert isinstance(test["passed"], bool)
        assert test["score"] in {0, test["max-score"]}
    if "owner" in result:
        assert isinstance(result["owner"], str) and result["owner"]
    if "assignment_type" in result:
        assert isinstance(result["assignment_type"], str) and result["assignment_type"]
    if "submitted_by" in result:
        assert isinstance(result["submitted_by"], dict)
    if expected_score is not None:
        assert result["score"] == expected_score, result


def _clone(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)


def _assert_delivery_inventory(correct: Path, temporary: Path) -> None:
    accepted = temporary / "accepted-delivery"
    _clone(correct, accepted)
    (accepted / ".classroom50.yaml").write_text("version: 1\n", encoding="utf-8")
    workflow = accepted / ".github/workflows/autograde.yaml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text("name: autograde\n", encoding="utf-8")
    git_config = accepted / ".git/config"
    git_config.parent.mkdir(parents=True)
    git_config.write_text("[core]\n", encoding="utf-8")
    assert _run_public(accepted).returncode == 0
    assert grade_submission(accepted)["score"] == 90
    for label, relative in (
        ("extra-root", "notes.txt"),
        ("extra-workflow", ".github/workflows/extra.yaml"),
        ("grader-tree", "_grader_selftest/copied.py"),
        ("nested-git", "ordinary/.git/nested.txt"),
    ):
        rejected = temporary / f"inventory-{label}"
        _clone(accepted, rejected)
        path = rejected / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unexpected\n", encoding="utf-8")
        assert _run_public(rejected).returncode == 1
        assert grade_submission(rejected)["score"] < 90


def _grade_rejected(root: Path, label: str, rejection_log: list[str]) -> dict:
    result = grade_submission(root)
    _assert_result_schema(result)
    assert result["score"] < 90, f"{label} unexpectedly earned full credit"
    rejection_log.append(f"{label}={result['score']}/90")
    return result


def main() -> int:
    rejection_log: list[str] = []
    runner_env = {
        "CLASSROOM": "datasci-217-local",
        "ASSIGNMENT": "assignment-08",
        "SUBMISSION_TAG": "submit/local-correct",
        "COMMIT_URL": "https://example.invalid/commit/correct",
        "RELEASE_URL": "https://example.invalid/release/correct",
        "REVIEW_URL": "https://example.invalid/review/correct",
    }
    os.environ.update(runner_env)
    with tempfile.TemporaryDirectory(prefix="a08-selftest-") as temporary_name:
        temporary = Path(temporary_name)

        starter = temporary / "starter"
        _copy_starter(starter)
        starter_public = _run_public(starter)
        assert starter_public.returncode == 1
        assert "Complete every TODO" in starter_public.stdout
        starter_result = grade_submission(starter)
        _assert_result_schema(starter_result, 0)
        rejection_log.append("untouched-starter=0/90")

        correct = temporary / "correct"
        _copy_starter(correct)
        _materialize_correct(correct)
        correct_public = _run_public(correct)
        assert correct_public.returncode == 0, correct_public.stdout + correct_public.stderr
        assert "All public checks passed." in correct_public.stdout
        correct_result = grade_submission(correct)
        _assert_result_schema(correct_result, 90)
        _assert_delivery_inventory(correct, temporary)
        assert "owner" not in correct_result and "assignment_type" not in correct_result
        assert "submitted_by" not in correct_result

        runner_stamped = dict(correct_result)
        runner_stamped.update({
            "owner": "course-staff",
            "assignment_type": "individual",
            "submitted_by": {"username": "local-author", "id": 12345},
        })
        _assert_result_schema(runner_stamped, 90)

        emitted_env = {
            **os.environ,
            **runner_env,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        emitted = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "autograder.py")],
            cwd=correct,
            env=emitted_env,
            text=True,
            capture_output=True,
            timeout=360,
            check=False,
        )
        assert emitted.returncode == 0, emitted.stdout + emitted.stderr
        emitted_result = json.loads((correct / "result.json").read_text(encoding="utf-8"))
        _assert_result_schema(emitted_result, 90)
        assert emitted_result["submission"] == "submit/local-correct"
        assert emitted_result["review"] == "https://example.invalid/review/correct"
        (correct / "result.json").unlink()

        for label, review_value in (("missing", None), ("empty", "   ")):
            fallback_cwd = temporary / f"review-{label}"
            fallback_cwd.mkdir()
            fallback_env = dict(emitted_env)
            if review_value is None:
                fallback_env.pop("REVIEW_URL", None)
            else:
                fallback_env["REVIEW_URL"] = review_value
            fallback = subprocess.run(
                [sys.executable, str(Path(__file__).parent / "autograder.py"), str(correct)],
                cwd=fallback_cwd,
                env=fallback_env,
                text=True,
                capture_output=True,
                timeout=360,
                check=False,
            )
            assert fallback.returncode == 0, fallback.stdout + fallback.stderr
            fallback_result = json.loads((fallback_cwd / "result.json").read_text())
            _assert_result_schema(fallback_result, 90)
            assert fallback_result["review"] == fallback_result["commit"]

        for environment_name in REQUIRED_CONTEXT_ENV.values():
            for label, replacement in (("missing", None), ("empty", "   ")):
                failure_root = temporary / f"context-{environment_name.lower()}-{label}"
                failure_root.mkdir()
                broken_env = dict(emitted_env)
                if replacement is None:
                    broken_env.pop(environment_name, None)
                else:
                    broken_env[environment_name] = replacement
                infrastructure = subprocess.run(
                    [sys.executable, str(Path(__file__).parent / "autograder.py"), str(correct)],
                    cwd=failure_root,
                    env=broken_env,
                    text=True,
                    capture_output=True,
                    timeout=60,
                    check=False,
                )
                assert infrastructure.returncode != 0
                assert not (failure_root / "result.json").exists()

        captured_failure = temporary / "captured failure"
        _copy_starter(captured_failure)
        failed_cli = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "autograder.py"), str(captured_failure)],
            cwd=captured_failure,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        assert failed_cli.returncode == 0
        failed_result = json.loads((captured_failure / "result.json").read_text(encoding="utf-8"))
        _assert_result_schema(failed_result, 0)
        rejection_log.append("captured-failure-cli=0/90-exit0")

        expected_bytes = {
            name: (correct / "output" / name).read_bytes() for name in ARTIFACT_SHA256
        }

        stored_output = temporary / "stored output untrusted"
        _clone(correct, stored_output)
        stored_nb = _load_notebook(stored_output)
        first_code = next(cell for cell in stored_nb["cells"] if cell["cell_type"] == "code")
        first_code["execution_count"] = 999
        first_code["outputs"] = [{"output_type": "stream", "name": "stdout", "text": "fake pass\n"}]
        _write_notebook(stored_output, stored_nb)
        _assert_result_schema(grade_submission(stored_output), 90)

        broken_with_outputs = temporary / "broken source with outputs"
        _clone(stored_output, broken_with_outputs)
        _replace_cell_text(
            broken_with_outputs,
            "a08-count-function",
            "grouped.size().rename('request_count')",
            "grouped['satisfaction_score'].count().rename('request_count')",
        )
        _grade_rejected(broken_with_outputs, "stored-output-broken-source", rejection_log)

        missing_fixture = temporary / "missing fixture"
        _clone(correct, missing_fixture)
        fixture_backup = (missing_fixture / "data" / "support_requests.csv").read_bytes()
        (missing_fixture / "data" / "support_requests.csv").unlink()
        sentinel_artifact = missing_fixture / "output" / "center_summary.csv"
        sentinel_artifact.write_text("must,remain\n", encoding="utf-8")
        try:
            _execute_notebook(missing_fixture, missing_fixture)
        except Exception:
            pass
        else:
            raise AssertionError("missing fixture unexpectedly executed")
        assert sentinel_artifact.read_text() == "must,remain\n"
        _grade_rejected(missing_fixture, "missing-fixture-before-cleanup", rejection_log)
        (missing_fixture / "data" / "support_requests.csv").write_bytes(fixture_backup)

        corrupt_fixture = temporary / "corrupt fixture"
        _clone(correct, corrupt_fixture)
        (corrupt_fixture / "data" / "support_requests.csv").write_text("corrupt\n", encoding="utf-8")
        corrupt_sentinel = corrupt_fixture / "output" / "center_summary.csv"
        corrupt_sentinel.write_text("must,remain\n", encoding="utf-8")
        try:
            _execute_notebook(corrupt_fixture, corrupt_fixture)
        except Exception:
            pass
        else:
            raise AssertionError("corrupt fixture unexpectedly executed")
        assert corrupt_sentinel.read_text() == "must,remain\n"
        _grade_rejected(corrupt_fixture, "corrupt-fixture-before-cleanup", rejection_log)

        missing_manifest = temporary / "missing manifest"
        _clone(correct, missing_manifest)
        (missing_manifest / "data" / "fixture.json").unlink()
        _grade_rejected(missing_manifest, "missing-fixture-manifest", rejection_log)

        edited_manifest = temporary / "edited manifest categories"
        _clone(correct, edited_manifest)
        manifest_path = edited_manifest / "data" / "fixture.json"
        manifest_path.write_text(
            manifest_path.read_text(encoding="utf-8").replace('"Valley"', '"Changed"'),
            encoding="utf-8",
        )
        _grade_rejected(edited_manifest, "edited-manifest-category-order", rejection_log)

        changed_line_endings = temporary / "changed fixture line endings"
        _clone(correct, changed_line_endings)
        changed_fixture = changed_line_endings / "data" / "support_requests.csv"
        changed_fixture.write_bytes(changed_fixture.read_bytes().replace(b"\n", b"\r\n"))
        _grade_rejected(changed_line_endings, "fixture-line-ending-change", rejection_log)

        extra_fixture = temporary / "extra fixture file"
        _clone(correct, extra_fixture)
        (extra_fixture / "data" / "unexpected.csv").write_text("unexpected\n", encoding="utf-8")
        _grade_rejected(extra_fixture, "unexpected-fixture-file", rejection_log)

        sentinel_case = temporary / "foreign sentinel"
        _clone(correct, sentinel_case)
        foreign = sentinel_case / "output" / "foreign-sentinel.txt"
        foreign.write_text("preserve me\n", encoding="utf-8")
        for name in ARTIFACT_SHA256:
            (sentinel_case / "output" / name).write_text("stale\n", encoding="utf-8")
        _execute_setup_only(sentinel_case, sentinel_case / "output")
        assert foreign.read_text() == "preserve me\n"
        assert all(not (sentinel_case / "output" / name).exists() for name in ARTIFACT_SHA256)
        _grade_rejected(sentinel_case, "foreign-sentinel-final-inventory", rejection_log)
        foreign.unlink()
        _execute_notebook(sentinel_case, sentinel_case)
        _assert_result_schema(grade_submission(sentinel_case), 90)

        stale = temporary / "stale outputs"
        _clone(correct, stale)
        (stale / "output" / "center_count_summary.csv").write_text("stale\n", encoding="utf-8")
        _grade_rejected(stale, "stale-committed-output", rejection_log)
        _execute_notebook(stale, stale)
        _assert_result_schema(grade_submission(stale), 90)

        deleted = temporary / "deleted output"
        _clone(correct, deleted)
        (deleted / "output" / "mean_resolution_pivot.csv").unlink()
        _grade_rejected(deleted, "deleted-committed-output", rejection_log)

        extra = temporary / "extra output"
        _clone(correct, extra)
        (extra / "output" / "q1_legacy.csv").write_text("legacy\n", encoding="utf-8")
        _grade_rejected(extra, "legacy-extra-output", rejection_log)

        malformed = temporary / "malformed notebook"
        _clone(correct, malformed)
        (malformed / "assignment.ipynb").write_text("{bad json\n", encoding="utf-8")
        _grade_rejected(malformed, "malformed-notebook", rejection_log)

        protected = temporary / "protected edit"
        _clone(correct, protected)
        protected_nb = _load_notebook(protected)
        header = next(cell for cell in protected_nb["cells"] if cell["id"] == "a08-header")
        header["source"] = "edited protected header"
        _write_notebook(protected, protected_nb)
        _grade_rejected(protected, "protected-cell-edit", rejection_log)

        duplicate_id = temporary / "duplicate cell id"
        _clone(correct, duplicate_id)
        duplicate_nb = _load_notebook(duplicate_id)
        duplicate_nb["cells"][5]["id"] = duplicate_nb["cells"][4]["id"]
        _write_notebook(duplicate_id, duplicate_nb)
        _grade_rejected(duplicate_id, "duplicate-cell-id", rejection_log)

        reordered_cells = temporary / "reordered cells"
        _clone(correct, reordered_cells)
        reordered_nb = _load_notebook(reordered_cells)
        reordered_nb["cells"][4], reordered_nb["cells"][5] = (
            reordered_nb["cells"][5], reordered_nb["cells"][4]
        )
        _write_notebook(reordered_cells, reordered_nb)
        _grade_rejected(reordered_cells, "reordered-notebook-cells", rejection_log)

        checker_edit = temporary / "checker edit"
        _clone(correct, checker_edit)
        with (checker_edit / "check_assignment.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# edited checker\n")
        _grade_rejected(checker_edit, "public-checker-edit", rejection_log)

        wrong_count = temporary / "wrong request count"
        _clone(correct, wrong_count)
        _replace_cell_text(
            wrong_count,
            "a08-count-function",
            "grouped.size().rename('request_count')",
            "grouped['satisfaction_score'].count().rename('request_count')",
        )
        _grade_rejected(wrong_count, "count-used-for-request-rows", rejection_log)

        wrong_satisfaction = temporary / "wrong satisfaction count"
        _clone(correct, wrong_satisfaction)
        _replace_cell_text(
            wrong_satisfaction,
            "a08-count-function",
            "grouped['satisfaction_score'].count().rename('satisfaction_count')",
            "grouped.size().rename('satisfaction_count')",
        )
        _grade_rejected(wrong_satisfaction, "size-used-for-recorded-satisfaction", rejection_log)

        wrong_unique = temporary / "wrong distinct count"
        _clone(correct, wrong_unique)
        _replace_cell_text(
            wrong_unique,
            "a08-count-function",
            "grouped['agent_id'].nunique(dropna=True).rename('unique_agent_count')",
            "grouped['agent_id'].count().rename('unique_agent_count')",
        )
        _grade_rejected(wrong_unique, "count-used-for-distinct-agents", rejection_log)

        wrong_unique_column = temporary / "wrong nunique column"
        _clone(correct, wrong_unique_column)
        _replace_cell_text(
            wrong_unique_column,
            "a08-count-function",
            "grouped['agent_id'].nunique(dropna=True)",
            "grouped['request_id'].nunique(dropna=True)",
        )
        _grade_rejected(wrong_unique_column, "nunique-used-on-request-id", rejection_log)

        hardcoded_function = temporary / "hardcoded canonical function"
        _clone(correct, hardcoded_function)
        _replace_cell_text(
            hardcoded_function,
            "a08-count-function",
            "    grouped = request_table.groupby(",
            "    canonical_centers = ['Central', 'Harbor', 'Ridge']\n    grouped = request_table.groupby(",
        )
        _grade_rejected(hardcoded_function, "canonical-label-hardcoding", rejection_log)

        implicit_policy = temporary / "implicit policy"
        _clone(correct, implicit_policy)
        _replace_cell_text(
            implicit_policy,
            "a08-count-function",
            "'center', observed=True, sort=True, dropna=True",
            "'center'",
        )
        _grade_rejected(implicit_policy, "implicit-group-policy", rejection_log)

        false_observed = temporary / "false observed"
        _clone(correct, false_observed)
        _replace_cell_text(false_observed, "a08-count-function", "observed=True", "observed=False")
        _grade_rejected(false_observed, "unused-category-materialized", rejection_log)

        indexed_aggregation = temporary / "indexed aggregation"
        _clone(correct, indexed_aggregation)
        _replace_cell_text(indexed_aggregation, "a08-center-summary-function", "as_index=False", "as_index=True")
        _grade_rejected(indexed_aggregation, "indexed-named-aggregation", rejection_log)

        wrong_named_source = temporary / "wrong named aggregation source"
        _clone(correct, wrong_named_source)
        _replace_cell_text(
            wrong_named_source,
            "a08-center-summary-function",
            "total_resolution_minutes=('resolution_minutes', 'sum')",
            "total_resolution_minutes=('satisfaction_score', 'sum')",
        )
        _grade_rejected(wrong_named_source, "wrong-named-aggregation-source", rejection_log)

        rounded_aggregation = temporary / "rounded aggregation"
        _clone(correct, rounded_aggregation)
        _replace_cell_text(
            rounded_aggregation,
            "a08-center-summary-function",
            "        mean_resolution_minutes=('resolution_minutes', 'mean'),\n    )",
            "        mean_resolution_minutes=('resolution_minutes', 'mean'),\n    ).round(0)",
        )
        _grade_rejected(rounded_aggregation, "rounded-aggregation", rejection_log)

        mutated_context = temporary / "mutated context"
        _clone(correct, mutated_context)
        _replace_cell_text(mutated_context, "a08-context-function", "result = request_table.copy(deep=True)", "result = request_table")
        _grade_rejected(mutated_context, "context-input-mutation", rejection_log)

        positional_context = temporary / "positional context"
        _clone(correct, positional_context)
        _replace_cell_text(
            positional_context,
            "a08-context-function",
            ").transform('mean')",
            ").transform('mean').reset_index(drop=True)",
        )
        _grade_rejected(positional_context, "positional-transform-alignment", rejection_log)

        wrong_transform = temporary / "wrong transform column"
        _clone(correct, wrong_transform)
        _replace_cell_text(
            wrong_transform,
            "a08-context-function",
            "request_table['resolution_minutes'].groupby(",
            "request_table['satisfaction_score'].groupby(",
        )
        _grade_rejected(wrong_transform, "wrong-transform-variable", rejection_log)

        multilevel_two_key = temporary / "multiindex two key"
        _clone(correct, multilevel_two_key)
        _replace_cell_text(multilevel_two_key, "a08-two-key-function", "as_index=False", "as_index=True")
        _grade_rejected(multilevel_two_key, "multiindex-two-key-result", rejection_log)

        wrong_pivot = temporary / "wrong pivot aggregation"
        _clone(correct, wrong_pivot)
        _replace_cell_text(wrong_pivot, "a08-pivot-function", "aggfunc='mean'", "aggfunc='sum'")
        _grade_rejected(wrong_pivot, "wrong-pivot-aggregation", rejection_log)

        filled_pivot = temporary / "filled pivot"
        _clone(correct, filled_pivot)
        _replace_cell_text(
            filled_pivot,
            "a08-pivot-function",
            "    )",
            "    ).fillna(0)",
        )
        _grade_rejected(filled_pivot, "absent-combination-filled-zero", rejection_log)

        extra_pivot = temporary / "extra pivot"
        _clone(correct, extra_pivot)
        _replace_cell_text(
            extra_pivot,
            "a08-task3-run",
            "resolution_pivot = build_resolution_pivot(support_requests)",
            "unused_extra_pivot = pd.pivot_table(support_requests, index='center', columns='channel', values='resolution_minutes', aggfunc='mean', observed=True, sort=True, dropna=True)\nresolution_pivot = build_resolution_pivot(support_requests)",
        )
        _grade_rejected(extra_pivot, "multiple-pivot-table-calls", rejection_log)

        missing_comparison = temporary / "missing cell comparison"
        _clone(correct, missing_comparison)
        missing_comparison_nb = _load_notebook(missing_comparison)
        task3_cell = next(
            cell for cell in missing_comparison_nb["cells"]
            if cell["id"] == "a08-task3-run"
        )
        task3_source = "".join(task3_cell["source"])
        comparison_start = task3_source.index("for reference_row")
        comparison_end = task3_source.index("assert not (resolution_pivot == 0)")
        task3_source = (
            task3_source[:comparison_start]
            + "assert len(pivot_reference) == 8\n"
            + task3_source[comparison_end:]
        )
        _set_cell_source(missing_comparison_nb, "a08-task3-run", task3_source)
        _write_notebook(missing_comparison, missing_comparison_nb)
        _grade_rejected(missing_comparison, "missing-cell-for-cell-comparison", rejection_log)

        structural_pivot = temporary / "structural pivot"
        _clone(correct, structural_pivot)
        structural_nb = _load_notebook(structural_pivot)
        _set_cell_source(
            structural_nb,
            "a08-pivot-function",
            "def build_resolution_pivot(request_table):\n"
            "    return request_table.pivot(index='center', columns='channel', values='resolution_minutes')\n",
        )
        _write_notebook(structural_pivot, structural_nb)
        _grade_rejected(structural_pivot, "structural-pivot-instead-of-aggregation", rejection_log)

        remote = temporary / "remote source"
        _clone(correct, remote)
        _replace_cell_text(remote, "a08-load", "support_requests = pd.read_csv", "remote_url = 'https://example.invalid/data.csv'\nsupport_requests = pd.read_csv")
        _grade_rejected(remote, "remote-data-code", rejection_log)

        advanced = temporary / "advanced group apply"
        _clone(correct, advanced)
        _replace_cell_text(
            advanced,
            "a08-task2-run",
            "center_summary = build_center_summary(support_requests)",
            "unused_advanced = support_requests.groupby('center', observed=True, sort=True, dropna=True).apply(lambda part: part)\ncenter_summary = build_center_summary(support_requests)",
        )
        _grade_rejected(advanced, "advanced-group-apply", rejection_log)

        joined = temporary / "out of scope join"
        _clone(correct, joined)
        _replace_cell_text(
            joined,
            "a08-task2-run",
            "center_summary = build_center_summary(support_requests)",
            "unused_join = support_requests.merge(support_requests, on='request_id')\ncenter_summary = build_center_summary(support_requests)",
        )
        _grade_rejected(joined, "out-of-scope-join", rejection_log)

        plotted = temporary / "out of scope plot"
        _clone(correct, plotted)
        _replace_cell_text(
            plotted,
            "a08-task2-run",
            "center_summary = build_center_summary(support_requests)",
            "unused_plot = support_requests.plot()\ncenter_summary = build_center_summary(support_requests)",
        )
        _grade_rejected(plotted, "out-of-scope-visualization", rejection_log)

        timed = temporary / "out of scope time"
        _clone(correct, timed)
        _replace_cell_text(
            timed,
            "a08-task2-run",
            "center_summary = build_center_summary(support_requests)",
            "unused_time = pd.to_datetime(['2026-01-01'])\ncenter_summary = build_center_summary(support_requests)",
        )
        _grade_rejected(timed, "out-of-scope-time-series", rejection_log)

        modeled = temporary / "out of scope modeling import"
        _clone(correct, modeled)
        _replace_cell_text(
            modeled,
            "a08-load",
            "support_requests = pd.read_csv",
            "import sklearn\nsupport_requests = pd.read_csv",
        )
        _grade_rejected(modeled, "out-of-scope-modeling", rejection_log)

        random_case = temporary / "random mutable data"
        _clone(correct, random_case)
        _replace_cell_text(random_case, "a08-load", "support_requests = pd.read_csv", "unused_random = np.random.default_rng()\nsupport_requests = pd.read_csv")
        _grade_rejected(random_case, "random-data-code", rejection_log)

        absolute_path = temporary / "absolute path"
        _clone(correct, absolute_path)
        _replace_cell_text(absolute_path, "a08-load", "support_requests = pd.read_csv", "unused_path = '/tmp/support.csv'\nsupport_requests = pd.read_csv")
        _grade_rejected(absolute_path, "absolute-path-code", rejection_log)

        corrected = temporary / "corrected resubmission"
        _clone(wrong_pivot, corrected)
        corrected_nb = _load_notebook(corrected)
        correct_nb = _load_notebook(correct)
        corrected_nb["cells"] = correct_nb["cells"]
        _write_notebook(corrected, corrected_nb)
        _assert_result_schema(grade_submission(corrected), 90)

        # Recheck the exact accepted bytes after all author-side mutations.
        assert {
            name: (correct / "output" / name).read_bytes() for name in ARTIFACT_SHA256
        } == expected_bytes

    print("Assignment 08 grader self-test passed.")
    print("correct-submission=90/90; corrected-resubmission=90/90; public-check=pass")
    print(f"rejected-cases={len(rejection_log)}")
    for evidence in rejection_log:
        print(f"REJECT {evidence}")
    print("result-schema=classroom50/result/v1; result-file=result.json; captured-failure-exit=0")
    print("runner-optional-fields=accepted-but-not-grader-emitted; review-url=context-supplied")
    print("layouts=flattened+course-root+nested+relocated+spaces; sentinel=preserved-then-extra-rejected")
    print("alternate-functions=5/5; pivot-equivalences=8/8; repeat=deterministic")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
