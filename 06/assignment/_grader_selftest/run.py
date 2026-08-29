# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "ipykernel==7.1.0",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "numpy==2.0.2",
#   "pandas==3.0.5",
# ]
# ///

"""Adversarial release harness for the Assignment 06 grader contract."""

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
    grade_submission,
)


ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
UTC_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

LOAD_SOURCE = r'''SPECIMEN_DTYPES = {
    'specimen_id': 'string',
    'collector_id': 'string',
    'collection_number': 'int64',
    'station_code': 'string',
    'material': 'string',
    'mass_g': 'float64',
}
STATION_DTYPES = {
    'station_code': 'string',
    'station_name': 'string',
    'region': 'string',
    'record_status': 'string',
}
REVIEW_DTYPES = {'specimen_id': 'string', 'review_score': 'float64'}
SENSOR_DTYPES = {
    'sensor_id': 'string',
    'station_code': 'string',
    'baseline_value': 'float64',
    'followup_value': 'float64',
}

specimens = pd.read_csv(DATA_DIR / 'specimens.csv', dtype=SPECIMEN_DTYPES)
stations_history = pd.read_csv(DATA_DIR / 'stations_history.csv', dtype=STATION_DTYPES)
batch_a = pd.read_csv(DATA_DIR / 'specimens_batch_a.csv', dtype=SPECIMEN_DTYPES)
batch_b = pd.read_csv(DATA_DIR / 'specimens_batch_b.csv', dtype=SPECIMEN_DTYPES)
review_scores = pd.read_csv(DATA_DIR / 'review_scores.csv', dtype=REVIEW_DTYPES)
sensor_scores_wide = pd.read_csv(DATA_DIR / 'sensor_scores_wide.csv', dtype=SENSOR_DTYPES)

assert specimens.shape == (7, 6)
assert specimens.columns.tolist() == list(SPECIMEN_DTYPES)
assert stations_history.shape == (5, 4)
assert stations_history.columns.tolist() == list(STATION_DTYPES)
assert batch_a.shape == (4, 6) and batch_a.columns.tolist() == list(SPECIMEN_DTYPES)
assert batch_b.shape == (3, 6) and batch_b.columns.tolist() == list(SPECIMEN_DTYPES)
assert review_scores.shape == (3, 2) and review_scores.columns.tolist() == list(REVIEW_DTYPES)
assert sensor_scores_wide.shape == (4, 4) and sensor_scores_wide.columns.tolist() == list(SENSOR_DTYPES)
'''

TASK1_MARKDOWN = """## Task 1 — contract-first validated merge

The specimen table has one row per specimen. Its primary key is `specimen_id`,
its candidate key is (`collector_id`, `collection_number`), and `station_code`
is a foreign key. Station history has one row per history record; after selecting
current records it should have one row per station. I predict a many-to-one left
join with seven rows because the goal is to keep every specimen. An unmatched
foreign key should remain as a specimen row with missing station metadata and a
`left_only` diagnostic.
"""

CONTRACT_VALUES_SOURCE = r'''specimen_grain = 'one row per specimen'
station_history_grain = 'one row per station-history record'
primary_key = ['specimen_id']
candidate_key = ['collector_id', 'collection_number']
foreign_key = ['station_code']
predicted_cardinality = 'many_to_one'
preservation_goal = 'keep every specimen row'
join_type = 'left'
predicted_rows = 7
'''

KEY_CHECKS_SOURCE = r'''assert specimens['specimen_id'].notna().all()
assert not specimens['specimen_id'].duplicated().any()
assert specimens[candidate_key].notna().all().all()
assert not specimens.duplicated(candidate_key).any()
station_duplicate_mask = stations_history.duplicated(['station_code'], keep=False)
duplicated_station_rows = stations_history.loc[station_duplicate_mask].copy()
assert duplicated_station_rows['station_code'].tolist() == ['R', 'R']
'''

DUPLICATE_FAILURE_SOURCE = r'''duplicate_contract_failed = False
duplicate_error_name = None
try:
    specimens.merge(
        stations_history,
        on='station_code',
        how='left',
        validate='many_to_one',
    )
except pd.errors.MergeError as error:
    duplicate_contract_failed = True
    duplicate_error_name = type(error).__name__
assert duplicate_contract_failed and duplicate_error_name == 'MergeError'
'''

TASK1_FUNCTIONS_SOURCE = r'''def select_current_stations(history_table):
    """Return the validated current station lookup without mutating the input."""
    required = ['station_code', 'station_name', 'region', 'record_status']
    missing = [column for column in required if column not in history_table.columns]
    if missing:
        raise ValueError(f'missing station columns: {missing}')
    selected = history_table.loc[
        history_table['record_status'].eq('current'),
        ['station_code', 'station_name', 'region'],
    ].copy()
    if selected['station_code'].isna().any():
        raise ValueError('current station key must be nonmissing')
    if selected['station_code'].duplicated().any():
        raise ValueError('current station key must be unique')
    return selected.reset_index(drop=True)


def validated_station_merge(specimen_table, station_table):
    """Return an explicit, validated, diagnostic specimen-to-station merge."""
    if 'station_code' not in specimen_table.columns:
        raise ValueError('specimen table needs station_code')
    required_lookup = ['station_code', 'station_name', 'region']
    missing = [column for column in required_lookup if column not in station_table.columns]
    if missing:
        raise ValueError(f'missing lookup columns: {missing}')
    return specimen_table.merge(
        station_table,
        on='station_code',
        how='left',
        validate='many_to_one',
        indicator=True,
    )
'''

TASK1_RUN_SOURCE = r'''current_stations = select_current_stations(stations_history)
assert current_stations['station_code'].notna().all()
assert not current_stations['station_code'].duplicated().any()
specimen_merge_audit = validated_station_merge(specimens, current_stations)
assert specimen_merge_audit.columns.tolist() == [
    'specimen_id', 'collector_id', 'collection_number', 'station_code',
    'material', 'mass_g', 'station_name', 'region', '_merge',
]
assert specimen_merge_audit['specimen_id'].tolist() == specimens['specimen_id'].tolist()
canonical_merge_counts = specimen_merge_audit['_merge'].astype('string').value_counts().reindex(
    ['both', 'left_only', 'right_only'], fill_value=0
).astype('int64').to_dict()
assert canonical_merge_counts == {'both': 6, 'left_only': 1, 'right_only': 0}
unmatched_specimens = specimen_merge_audit.loc[
    specimen_merge_audit['_merge'].astype('string').eq('left_only')
]
assert unmatched_specimens[['specimen_id', 'station_code']].values.tolist() == [['SP106', 'X']]
assert not specimen_merge_audit['station_code'].eq('U').any()
'''

TASK1_SAVE_SOURCE = r'''specimen_merge_audit.to_csv(
    ARTIFACT_PATHS['specimen_merge_audit.csv'],
    index=False,
    encoding='utf-8',
    lineterminator='\n',
)
specimen_merge_round_trip = pd.read_csv(
    ARTIFACT_PATHS['specimen_merge_audit.csv'],
    dtype={
        'specimen_id': 'string',
        'collector_id': 'string',
        'collection_number': 'int64',
        'station_code': 'string',
        'material': 'string',
        'mass_g': 'float64',
        'station_name': 'string',
        'region': 'string',
        '_merge': 'string',
    },
)
expected_merge_serialized = specimen_merge_audit.copy()
expected_merge_serialized['_merge'] = expected_merge_serialized['_merge'].astype('string')
pd.testing.assert_frame_equal(specimen_merge_round_trip, expected_merge_serialized)
'''

TASK2_MARKDOWN = """## Task 2 — vertical concatenation and horizontal alignment

Each batch row is one specimen, with the same base schema, so the two partitions
may be stacked into seven rows. `source_partition` must identify which input
supplied each row. Row stacking aligns column labels across schemas; I predict
that a field absent from one partition will be structurally missing in that
partition's rows. Horizontal feature concatenation instead aligns named
`specimen_id` index labels, so nonoverlapping identifiers produce missing feature
values rather than new row-partition provenance.
"""

STACK_FUNCTION_SOURCE = r'''def stack_specimen_partitions(partition_map):
    """Stack copied specimen partitions with ordinary-column provenance."""
    items = list(partition_map.items())
    if not items:
        raise ValueError('partition_map must not be empty')
    prepared = []
    for label, table in items:
        if 'source_partition' in table.columns:
            raise ValueError('source_partition is reserved for provenance')
        copied = table.copy(deep=True)
        copied['source_partition'] = pd.Series(
            [str(label)] * len(copied),
            index=copied.index,
            dtype='string',
        )
        prepared.append(copied)
    return pd.concat(prepared, axis=0, ignore_index=True, sort=False)
'''

STACK_RUN_SOURCE = r'''combined_specimens = stack_specimen_partitions(
    {'batch_a': batch_a, 'batch_b': batch_b}
)
assert len(combined_specimens) == len(batch_a) + len(batch_b) == 7
assert combined_specimens['source_partition'].value_counts().to_dict() == {'batch_a': 4, 'batch_b': 3}
assert combined_specimens['specimen_id'].tolist() == specimens['specimen_id'].tolist()
assert isinstance(combined_specimens.index, pd.RangeIndex)
assert combined_specimens.index.tolist() == list(range(7))
pd.testing.assert_frame_equal(
    combined_specimens.drop(columns='source_partition'),
    specimens,
)
'''

SCHEMA_DRIFT_SOURCE = r'''batch_a_drift = batch_a.copy(deep=True)
batch_b_drift = batch_b.drop(columns='mass_g').copy()
batch_b_drift['review_note'] = pd.Series(
    ['manual review'] * len(batch_b_drift),
    index=batch_b_drift.index,
    dtype='string',
)
schema_drift_preview = stack_specimen_partitions(
    {'batch_a': batch_a_drift, 'batch_b': batch_b_drift}
)
assert schema_drift_preview.columns.tolist() == [
    'specimen_id', 'collector_id', 'collection_number', 'station_code',
    'material', 'mass_g', 'source_partition', 'review_note',
]
assert int(schema_drift_preview['mass_g'].isna().sum()) == 3
assert int(schema_drift_preview['review_note'].isna().sum()) == 4
'''

ALIGN_FUNCTION_SOURCE = r'''def align_specimen_features(mass_table, review_table):
    """Outer-align mass and review features by a validated named specimen index."""
    contracts = (
        (mass_table, ['specimen_id', 'mass_g']),
        (review_table, ['specimen_id', 'review_score']),
    )
    for table, required in contracts:
        missing = [column for column in required if column not in table.columns]
        if missing:
            raise ValueError(f'missing feature columns: {missing}')
        if table['specimen_id'].isna().any() or table['specimen_id'].duplicated().any():
            raise ValueError('specimen_id must be nonmissing and unique')
    mass_indexed = mass_table.set_index('specimen_id')[['mass_g']].copy()
    review_indexed = review_table.set_index('specimen_id')[['review_score']].copy()
    union_order = []
    seen = set()
    for value in [*mass_indexed.index.tolist(), *review_indexed.index.tolist()]:
        if value not in seen:
            seen.add(value)
            union_order.append(value)
    aligned = pd.concat(
        [mass_indexed, review_indexed],
        axis=1,
        join='outer',
        sort=False,
    ).reindex(union_order)
    aligned.index = aligned.index.astype(pd.StringDtype(na_value=pd.NA))
    aligned.index.name = 'specimen_id'
    return aligned
'''

ALIGN_RUN_SOURCE = r'''mass_features = specimens.loc[
    specimens['specimen_id'].isin(['SP101', 'SP102', 'SP103']),
    ['specimen_id', 'mass_g'],
].copy()
aligned_features = align_specimen_features(mass_features, review_scores)
assert aligned_features.index.tolist() == ['SP101', 'SP102', 'SP103', 'SP108']
assert aligned_features.index.name == 'specimen_id'
assert aligned_features.columns.tolist() == ['mass_g', 'review_score']
assert aligned_features['review_score'].isna().index[aligned_features['review_score'].isna()].tolist() == ['SP101']
assert aligned_features['mass_g'].isna().index[aligned_features['mass_g'].isna()].tolist() == ['SP108']
assert aligned_features.loc['SP102'].tolist() == [8.0, 7.0]
assert aligned_features.loc['SP103'].tolist() == [10.5, 9.0]
'''

TASK2_SAVE_SOURCE = r'''combined_specimens.to_csv(
    ARTIFACT_PATHS['combined_specimens.csv'],
    index=False,
    encoding='utf-8',
    lineterminator='\n',
)
aligned_features.to_csv(
    ARTIFACT_PATHS['aligned_features.csv'],
    index=True,
    encoding='utf-8',
    lineterminator='\n',
)
combined_round_trip = pd.read_csv(
    ARTIFACT_PATHS['combined_specimens.csv'],
    dtype={**SPECIMEN_DTYPES, 'source_partition': 'string'},
)
aligned_round_trip = pd.read_csv(
    ARTIFACT_PATHS['aligned_features.csv'],
    dtype={'specimen_id': 'string', 'mass_g': 'float64', 'review_score': 'float64'},
    index_col='specimen_id',
)
pd.testing.assert_frame_equal(combined_round_trip, combined_specimens)
pd.testing.assert_frame_equal(aligned_round_trip, aligned_features)
'''

TASK3_MARKDOWN = """## Task 3 — nonaggregating wide/long reshape

The wide grain is one row per (`sensor_id`, `station_code`) key. The long grain
is one row per (`sensor_id`, `station_code`, `measurement_label`) key. Melting
two value columns across four wide rows should create eight long rows. Each long
key must identify exactly one value; otherwise structural `pivot` cannot choose
one cell value without an aggregation rule.
"""

RESHAPE_FUNCTIONS_SOURCE = r'''def wide_to_long_scores(wide_table):
    """Melt a validated wide sensor table to its structural long form."""
    id_columns = ['sensor_id', 'station_code']
    value_columns = ['baseline_value', 'followup_value']
    missing = [column for column in id_columns + value_columns if column not in wide_table.columns]
    if missing:
        raise ValueError(f'missing wide columns: {missing}')
    if wide_table[id_columns].isna().any().any() or wide_table.duplicated(id_columns).any():
        raise ValueError('wide structural key must be nonmissing and unique')
    long_table = wide_table.melt(
        id_vars=id_columns,
        value_vars=value_columns,
        var_name='measurement_label',
        value_name='value',
    )
    long_table['measurement_label'] = long_table['measurement_label'].astype('string')
    long_table['value'] = long_table['value'].astype('float64')
    if long_table[id_columns + ['measurement_label']].isna().any().any():
        raise ValueError('long structural key must be nonmissing')
    if long_table.duplicated(id_columns + ['measurement_label']).any():
        raise ValueError('melt produced a duplicate structural key')
    return long_table


def long_to_wide_scores(long_table, ordered_columns):
    """Pivot a uniquely keyed long table and restore caller-supplied order."""
    id_columns = ['sensor_id', 'station_code']
    structural_columns = id_columns + ['measurement_label', 'value']
    missing = [column for column in structural_columns if column not in long_table.columns]
    if missing:
        raise ValueError(f'missing long columns: {missing}')
    if long_table[id_columns + ['measurement_label']].isna().any().any():
        raise ValueError('long structural key must be nonmissing')
    row_order = []
    seen = set()
    for pair in long_table[id_columns].itertuples(index=False, name=None):
        if pair not in seen:
            seen.add(pair)
            row_order.append(pair)
    pivoted = long_table.pivot(
        index=id_columns,
        columns='measurement_label',
        values='value',
    ).reset_index()
    pivoted.columns.name = None
    positions = {pair: position for position, pair in enumerate(row_order)}
    pivoted['_a06_row_order'] = [
        positions[pair]
        for pair in pivoted[id_columns].itertuples(index=False, name=None)
    ]
    pivoted = pivoted.sort_values('_a06_row_order', kind='stable').drop(
        columns='_a06_row_order'
    ).reset_index(drop=True)
    ordered_columns = list(ordered_columns)
    if ordered_columns[:2] != id_columns or set(ordered_columns) != set(pivoted.columns):
        raise ValueError('ordered_columns must name the exact reconstructed wide schema')
    result = pivoted.loc[:, ordered_columns].copy()
    result.columns = pd.Index(ordered_columns)
    return result
'''

RESHAPE_RUN_SOURCE = r'''sensor_scores_long = wide_to_long_scores(sensor_scores_wide)
sensor_scores_round_trip = long_to_wide_scores(
    sensor_scores_long,
    sensor_scores_wide.columns.tolist(),
)
assert sensor_scores_long.shape == (8, 4)
assert sensor_scores_long.columns.tolist() == ['sensor_id', 'station_code', 'measurement_label', 'value']
assert not sensor_scores_long.duplicated(['sensor_id', 'station_code', 'measurement_label']).any()
assert sensor_scores_long['measurement_label'].value_counts().to_dict() == {'baseline_value': 4, 'followup_value': 4}
pd.testing.assert_frame_equal(sensor_scores_round_trip, sensor_scores_wide)
'''

DUPLICATE_PIVOT_SOURCE = r'''duplicate_long = pd.concat(
    [sensor_scores_long, sensor_scores_long.iloc[[0]].copy()],
    ignore_index=True,
)
duplicate_long_mask = duplicate_long.duplicated(
    ['sensor_id', 'station_code', 'measurement_label'],
    keep=False,
)
duplicate_long_rows = duplicate_long.loc[duplicate_long_mask].copy()
duplicate_pivot_failed = False
try:
    long_to_wide_scores(duplicate_long, sensor_scores_wide.columns.tolist())
except ValueError:
    duplicate_pivot_failed = True
assert duplicate_pivot_failed and len(duplicate_long_rows) == 2
'''

TASK3_SAVE_SOURCE = r'''sensor_scores_long.to_csv(
    ARTIFACT_PATHS['sensor_scores_long.csv'],
    index=False,
    encoding='utf-8',
    lineterminator='\n',
)
sensor_scores_round_trip.to_csv(
    ARTIFACT_PATHS['sensor_scores_round_trip.csv'],
    index=False,
    encoding='utf-8',
    lineterminator='\n',
)
long_round_trip = pd.read_csv(
    ARTIFACT_PATHS['sensor_scores_long.csv'],
    dtype={
        'sensor_id': 'string',
        'station_code': 'string',
        'measurement_label': 'string',
        'value': 'float64',
    },
)
wide_round_trip = pd.read_csv(
    ARTIFACT_PATHS['sensor_scores_round_trip.csv'],
    dtype=SENSOR_DTYPES,
)
pd.testing.assert_frame_equal(long_round_trip, sensor_scores_long)
pd.testing.assert_frame_equal(wide_round_trip, sensor_scores_round_trip)
'''

REFLECTION_MARKDOWN = """## Final reflection

1. The status field is supplied source evidence about which history record is
   authoritative. Keeping the first or last duplicate would depend on incidental
   row order instead.
2. The left merge preserved every specimen and the indicator exposed `SP106`/`X`
   as `left_only` with missing lookup metadata.
3. Vertical schema drift created missing cells in rows from the partition that
   never supplied a column. Horizontal alignment created missing features where
   one named specimen index was absent from the other feature table.
4. A duplicate long key maps more than one value to a single wide cell, so
   structural pivot is ambiguous. Aggregation would make a new analytical choice
   and is deferred to Lecture 08.
"""

CORRECT_SOURCES = {
    "a06-load": LOAD_SOURCE,
    "a06-task1-contract": TASK1_MARKDOWN,
    "a06-contract-values": CONTRACT_VALUES_SOURCE,
    "a06-key-checks": KEY_CHECKS_SOURCE,
    "a06-duplicate-failure": DUPLICATE_FAILURE_SOURCE,
    "a06-task1-functions": TASK1_FUNCTIONS_SOURCE,
    "a06-task1-run": TASK1_RUN_SOURCE,
    "a06-task1-save": TASK1_SAVE_SOURCE,
    "a06-task2-contract": TASK2_MARKDOWN,
    "a06-stack-function": STACK_FUNCTION_SOURCE,
    "a06-stack-run": STACK_RUN_SOURCE,
    "a06-schema-drift": SCHEMA_DRIFT_SOURCE,
    "a06-align-function": ALIGN_FUNCTION_SOURCE,
    "a06-align-run": ALIGN_RUN_SOURCE,
    "a06-task2-save": TASK2_SAVE_SOURCE,
    "a06-task3-contract": TASK3_MARKDOWN,
    "a06-reshape-functions": RESHAPE_FUNCTIONS_SOURCE,
    "a06-reshape-run": RESHAPE_RUN_SOURCE,
    "a06-duplicate-pivot": DUPLICATE_PIVOT_SOURCE,
    "a06-task3-save": TASK3_SAVE_SOURCE,
    "a06-reflection": REFLECTION_MARKDOWN,
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
    assert set(result) == {
        "schema", "classroom", "assignment", "submission", "commit",
        "release", "review", "datetime", "score", "max-score", "tests",
    }
    assert result["schema"] == "classroom50/result/v1"
    assert all(
        isinstance(result[field], str) and result[field]
        for field in (
            "classroom", "assignment", "submission", "commit", "release",
            "review", "datetime",
        )
    )
    assert isinstance(result["score"], int)
    assert UTC_DATETIME.fullmatch(result["datetime"])
    assert result["max-score"] == 90
    assert isinstance(result["tests"], list) and len(result["tests"]) == 3
    assert sum(test["score"] for test in result["tests"]) == result["score"]
    assert sum(test["max-score"] for test in result["tests"]) == 90
    for test in result["tests"]:
        assert set(test) == {"test-name", "passed", "score", "max-score"}
        assert isinstance(test["test-name"], str) and test["test-name"]
        assert isinstance(test["passed"], bool)
        assert test["score"] in {0, test["max-score"]}
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
        "ASSIGNMENT": "assignment-06",
        "SUBMISSION_TAG": "submit/local-correct",
        "COMMIT_URL": "https://example.invalid/commit/correct",
        "RELEASE_URL": "https://example.invalid/release/correct",
        "REVIEW_URL": "https://example.invalid/review/correct",
    }
    os.environ.update(runner_env)
    with tempfile.TemporaryDirectory(prefix="a06-selftest-") as temporary_name:
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
        assert emitted_result["review"] == runner_env["REVIEW_URL"]
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

        stored = temporary / "stored output"
        _copy_starter(stored)
        stored_notebook = _load_notebook(stored)
        for cell in stored_notebook["cells"]:
            if cell["cell_type"] == "code":
                cell["execution_count"] = 999
                cell["outputs"] = [{"name": "stdout", "output_type": "stream", "text": ["fake success\n"]}]
        _write_notebook(stored, stored_notebook)
        (stored / "output").mkdir(exist_ok=True)
        for name in ARTIFACT_SHA256:
            shutil.copy2(correct / "output" / name, stored / "output" / name)
        _grade_rejected(stored, "stored-output-broken-code", rejection_log)

        stale = temporary / "stale artifact"
        _clone(correct, stale)
        (stale / "output" / "combined_specimens.csv").write_text("stale\n", encoding="utf-8")
        _grade_rejected(stale, "stale-committed-artifact", rejection_log)

        missing_output = temporary / "missing output"
        _clone(correct, missing_output)
        (missing_output / "output" / "aligned_features.csv").unlink()
        _grade_rejected(missing_output, "missing-required-artifact", rejection_log)

        corrupt_fixture = temporary / "corrupt fixture"
        _clone(correct, corrupt_fixture)
        (corrupt_fixture / "data" / "specimens.csv").write_bytes(b"corrupt\n")
        corrupt_result = _grade_rejected(corrupt_fixture, "corrupt-fixture", rejection_log)
        assert corrupt_result["score"] == 0

        missing_fixture = temporary / "missing fixture"
        _clone(correct, missing_fixture)
        (missing_fixture / "data" / "fixture.json").unlink()
        missing_result = _grade_rejected(missing_fixture, "missing-manifest", rejection_log)
        assert missing_result["score"] == 0

        extra_fixture = temporary / "extra fixture"
        _clone(correct, extra_fixture)
        (extra_fixture / "data" / "extra.csv").write_text("x\n1\n", encoding="utf-8")
        extra_result = _grade_rejected(extra_fixture, "extra-fixture", rejection_log)
        assert extra_result["score"] == 0

        malformed = temporary / "malformed notebook"
        _clone(correct, malformed)
        (malformed / "assignment.ipynb").write_text("{not json\n", encoding="utf-8")
        _grade_rejected(malformed, "malformed-notebook", rejection_log)

        reordered = temporary / "reordered cells"
        _clone(correct, reordered)
        reordered_nb = _load_notebook(reordered)
        reordered_nb["cells"][3], reordered_nb["cells"][4] = reordered_nb["cells"][4], reordered_nb["cells"][3]
        _write_notebook(reordered, reordered_nb)
        _grade_rejected(reordered, "reordered-cells", rejection_log)

        missing_cell = temporary / "missing cell"
        _clone(correct, missing_cell)
        missing_cell_nb = _load_notebook(missing_cell)
        missing_cell_nb["cells"] = [
            cell for cell in missing_cell_nb["cells"] if cell["id"] != "a06-key-checks"
        ]
        _write_notebook(missing_cell, missing_cell_nb)
        _grade_rejected(missing_cell, "missing-cell", rejection_log)

        protected = temporary / "protected edit"
        _clone(correct, protected)
        _replace_cell_text(protected, "a06-setup", "pd.set_option", "pd.set_option")
        protected_nb = _load_notebook(protected)
        setup = next(cell for cell in protected_nb["cells"] if cell["id"] == "a06-setup")
        setup["source"].append("\n# edited")
        _write_notebook(protected, protected_nb)
        _grade_rejected(protected, "protected-cell-edit", rejection_log)

        checker_edit = temporary / "checker edit"
        _clone(correct, checker_edit)
        with (checker_edit / "check_assignment.py").open("a", encoding="utf-8") as stream:
            stream.write("\n# edited checker\n")
        _grade_rejected(checker_edit, "public-checker-edit", rejection_log)

        wrong_validate = temporary / "wrong validate"
        _clone(correct, wrong_validate)
        _replace_cell_text(wrong_validate, "a06-task1-functions", "validate='many_to_one'", "validate='one_to_one'")
        _grade_rejected(wrong_validate, "wrong-merge-cardinality", rejection_log)

        no_indicator = temporary / "no indicator"
        _clone(correct, no_indicator)
        _replace_cell_text(no_indicator, "a06-task1-functions", "indicator=True", "indicator=False")
        _grade_rejected(no_indicator, "missing-merge-indicator", rejection_log)

        inner_join = temporary / "inner join"
        _clone(correct, inner_join)
        _replace_cell_text(inner_join, "a06-task1-functions", "how='left'", "how='inner'")
        _grade_rejected(inner_join, "orphan-dropping-inner-join", rejection_log)

        implicit_key = temporary / "implicit merge key"
        _clone(correct, implicit_key)
        _replace_cell_text(
            implicit_key,
            "a06-task1-functions",
            "on='station_code',",
            "suffixes=('_specimen', '_station'),",
        )
        _grade_rejected(implicit_key, "implicit-merge-key", rejection_log)

        missing_validate = temporary / "missing validate"
        _clone(correct, missing_validate)
        _replace_cell_text(
            missing_validate,
            "a06-task1-functions",
            "validate='many_to_one',",
            "sort=False,",
        )
        _grade_rejected(missing_validate, "missing-merge-validation", rejection_log)

        fake_merge_failure = temporary / "fake merge failure"
        _clone(correct, fake_merge_failure)
        fake_merge_nb = _load_notebook(fake_merge_failure)
        _set_cell_source(
            fake_merge_nb,
            "a06-duplicate-failure",
            "duplicate_contract_failed = True\nduplicate_error_name = 'MergeError'\n",
        )
        _write_notebook(fake_merge_failure, fake_merge_nb)
        _grade_rejected(fake_merge_failure, "manufactured-merge-failure", rejection_log)

        arbitrary_dedupe = temporary / "arbitrary dedupe"
        _clone(correct, arbitrary_dedupe)
        _replace_cell_text(
            arbitrary_dedupe,
            "a06-task1-functions",
            "selected = history_table.loc[\n        history_table['record_status'].eq('current'),\n        ['station_code', 'station_name', 'region'],\n    ].copy()",
            "selected = history_table.drop_duplicates('station_code', keep='first')[['station_code', 'station_name', 'region']].copy()",
        )
        _grade_rejected(arbitrary_dedupe, "arbitrary-station-deduplication", rejection_log)

        hardcoded_selector = temporary / "hardcoded selector"
        _clone(correct, hardcoded_selector)
        _replace_cell_text(
            hardcoded_selector,
            "a06-task1-functions",
            "selected = history_table.loc[\n        history_table['record_status'].eq('current'),\n        ['station_code', 'station_name', 'region'],\n    ].copy()",
            "selected = history_table.loc[history_table['station_code'].isin(['R', 'S', 'T', 'U']) & history_table['record_status'].eq('current'), ['station_code', 'station_name', 'region']].copy()",
        )
        _grade_rejected(hardcoded_selector, "hardcoded-canonical-selector", rejection_log)

        stack_mutation = temporary / "stack mutation"
        _clone(correct, stack_mutation)
        _replace_cell_text(stack_mutation, "a06-stack-function", "copied = table.copy(deep=True)", "copied = table")
        _grade_rejected(stack_mutation, "stack-input-mutation", rejection_log)

        cleaned_schema_gap = temporary / "cleaned schema gap"
        _clone(correct, cleaned_schema_gap)
        _replace_cell_text(
            cleaned_schema_gap,
            "a06-schema-drift",
            "assert int(schema_drift_preview['mass_g'].isna().sum()) == 3",
            "schema_drift_preview = schema_drift_preview.fillna(0)\nassert int(schema_drift_preview['mass_g'].isna().sum()) == 3",
        )
        _grade_rejected(cleaned_schema_gap, "cleaned-structural-missingness", rejection_log)

        positional_align = temporary / "positional alignment"
        _clone(correct, positional_align)
        _replace_cell_text(
            positional_align,
            "a06-align-function",
            "mass_indexed = mass_table.set_index('specimen_id')[['mass_g']].copy()",
            "mass_indexed = mass_table.set_index('specimen_id')[['mass_g']].copy().reset_index(drop=True)",
        )
        _replace_cell_text(
            positional_align,
            "a06-align-function",
            "review_indexed = review_table.set_index('specimen_id')[['review_score']].copy()",
            "review_indexed = review_table.set_index('specimen_id')[['review_score']].copy().reset_index(drop=True)",
        )
        _grade_rejected(positional_align, "positional-feature-alignment", rejection_log)

        aggregating_pivot = temporary / "aggregating pivot"
        _clone(correct, aggregating_pivot)
        _replace_cell_text(aggregating_pivot, "a06-reshape-functions", "long_table.pivot(", "long_table.pivot_table(")
        _grade_rejected(aggregating_pivot, "aggregating-pivot-table", rejection_log)

        predeleted_duplicate = temporary / "predeleted duplicate"
        _clone(correct, predeleted_duplicate)
        _replace_cell_text(
            predeleted_duplicate,
            "a06-reshape-functions",
            "pivoted = long_table.pivot(",
            "long_table = long_table.loc[~long_table.duplicated(id_columns + ['measurement_label'])].copy()\n    pivoted = long_table.pivot(",
        )
        _grade_rejected(predeleted_duplicate, "predeleted-duplicate-long-key", rejection_log)

        fake_pivot_failure = temporary / "fake pivot failure"
        _clone(correct, fake_pivot_failure)
        fake_pivot_nb = _load_notebook(fake_pivot_failure)
        _set_cell_source(
            fake_pivot_nb,
            "a06-duplicate-pivot",
            "duplicate_long_rows = sensor_scores_long.iloc[[0, 0]].copy()\nduplicate_pivot_failed = True\n",
        )
        _write_notebook(fake_pivot_failure, fake_pivot_nb)
        _grade_rejected(fake_pivot_failure, "manufactured-pivot-failure", rejection_log)

        hardcoded_reshape = temporary / "hardcoded reshape"
        _clone(correct, hardcoded_reshape)
        _replace_cell_text(
            hardcoded_reshape,
            "a06-reshape-functions",
            "row_order = []\n    seen = set()\n    for pair in long_table[id_columns].itertuples(index=False, name=None):\n        if pair not in seen:\n            seen.add(pair)\n            row_order.append(pair)",
            "row_order = [('SN01', 'R'), ('SN02', 'S'), ('SN03', 'T'), ('SN04', 'R')]\n    seen = set(row_order)",
        )
        _grade_rejected(hardcoded_reshape, "hardcoded-canonical-reshape-order", rejection_log)

        remote = temporary / "remote path"
        _clone(correct, remote)
        _replace_cell_text(remote, "a06-load", "specimens = pd.read_csv", "remote_url = 'https://example.invalid/data.csv'\nspecimens = pd.read_csv")
        _grade_rejected(remote, "remote-data-code", rejection_log)

        absolute_path = temporary / "absolute path"
        _clone(correct, absolute_path)
        _replace_cell_text(
            absolute_path,
            "a06-load",
            "specimens = pd.read_csv(DATA_DIR / 'specimens.csv'",
            "absolute_fixture = '/tmp/specimens.csv'\nspecimens = pd.read_csv(DATA_DIR / 'specimens.csv'",
        )
        _grade_rejected(absolute_path, "absolute-path-code", rejection_log)

        corrected = temporary / "corrected resubmission"
        _clone(wrong_validate, corrected)
        corrected_nb = _load_notebook(corrected)
        correct_nb = _load_notebook(correct)
        corrected_nb["cells"] = correct_nb["cells"]
        _write_notebook(corrected, corrected_nb)
        corrected_result = grade_submission(corrected)
        _assert_result_schema(corrected_result, 90)

    print("Assignment 06 grader self-test passed.")
    print("correct-submission=90/90; corrected-resubmission=90/90; public-check=pass")
    print(f"rejected-cases={len(rejection_log)}")
    for evidence in rejection_log:
        print(f"REJECT {evidence}")
    print("result-schema=classroom50/result/v1; result-file=result.json; captured-failure-exit=0")
    print("layouts=flattened+course-root; relocation=spaces; rerun=deterministic; alternate-functions=6/6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
