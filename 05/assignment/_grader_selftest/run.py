# /// script
# requires-python = "==3.12.13"
# dependencies = ["numpy==2.0.2", "pandas==3.0.5"]
# ///

"""Adversarial self-test for the Assignment 05 public and central graders."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from grader import (
    OUTPUT_FILES,
    REQUIRED_CONTEXT_ENV,
    _execute,
    grade_submission,
)


ASSIGNMENT_DIR = Path(__file__).resolve().parents[1]
UTC_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

CONTRACT_MARKDOWN = """One row means **one submitted person record**. `record_id` is the candidate
identifier: it is expected to identify a person record, but that expectation is
tested rather than assumed. **Raw data** are the source tokens kept unchanged;
**clean data** are a derived copy that satisfies the stated contract.

A **schema** specifies column names, order, and data types. A **sentinel** is a
documented token such as `unknown`, `-9`, or status `NA` that represents a
missing value rather than its literal text. A **duplicate** is either an exact
repeated raw row or, separately, a repeated candidate identifier. A **missing
value** is an absent or invalid value represented with the appropriate pandas
nullable value. A **validation invariant** is a condition the clean result must
satisfy before export. **Provenance** records where the source came from and its
verified checksum.
"""

EXPLANATION_MARKDOWN = """Forward fill and backward fill are not justified because adjacent rows are
different person records and there is no meaningful within-person sequence.
Reviewable missing values therefore remain instead of being invented. Clean
means that the table satisfies a documented contract; it does not mean the data
are perfect or that uncertainty has disappeared.
"""

TASK1_SOURCE = r'''row_meaning = manifest['row_meaning']
candidate_identifier = manifest['candidate_identifier']
raw = pd.read_csv(DATA_PATH, keep_default_na=False)
raw_snapshot = raw.copy(deep=True)

def audit_person_records(raw_table):
    names = raw_table['full_name'].astype('string').str.strip()
    sites = raw_table['site'].astype('string')
    site_normalized = sites.str.strip().str.lower()
    statuses = raw_table['status'].astype('string')
    status_normalized = statuses.str.strip().str.lower()
    ages = raw_table['age_text'].astype('string').str.strip()
    dates = raw_table['visit_date'].astype('string').str.strip()

    age_sentinel = ages.str.lower().isin(['unknown', '-9'])
    status_sentinel = status_normalized.eq('na')
    numeric_age = pd.to_numeric(ages.mask(age_sentinel | ages.eq('')), errors='coerce')
    finite_age = numeric_age.notna() & np.isfinite(numeric_age)
    date_lexical = dates.str.fullmatch(EXACT_DATE_PATTERN)
    parsed_dates = pd.to_datetime(
        dates.where(date_lexical), format='%Y-%m-%d', errors='coerce'
    )

    issues = [
        ('schema mismatch', int(raw_table.columns.tolist() != EXPECTED_RAW_COLUMNS)),
        ('empty full-name tokens', int(names.eq('').sum())),
        ('empty date tokens', int(dates.eq('').sum())),
        ('age sentinel tokens', int(age_sentinel.sum())),
        ('status sentinel tokens', int(status_sentinel.sum())),
        ('age parse failures', int((~age_sentinel & ages.ne('') & numeric_age.isna()).sum())),
        ('numeric but noninteger age values', int((finite_age & numeric_age.mod(1).ne(0)).sum())),
        ('age values outside 0 through 120', int((finite_age & ~numeric_age.between(0, 120)).sum())),
        ('date parse failures', int((dates.ne('') & (~date_lexical | parsed_dates.isna())).sum())),
        ('rows in exact duplicate sets', int(raw_table.duplicated(keep=False).sum())),
        ('rows with repeated candidate IDs', int(raw_table['record_id'].duplicated(keep=False).sum())),
        ('site values needing format normalization', int((sites.ne('') & sites.ne(site_normalized)).sum())),
        ('status values needing format normalization', int((~status_sentinel & statuses.ne('') & statuses.ne(status_normalized)).sum())),
        ('unexpected site values', int((site_normalized.ne('') & ~site_normalized.isin(['north', 'south', 'west'])).sum())),
        ('unexpected non-sentinel status values', int((~status_sentinel & status_normalized.ne('') & ~status_normalized.isin(['active', 'pending', 'complete'])).sum())),
    ]
    result = pd.DataFrame(issues, columns=['issue', 'count'])
    result['issue'] = result['issue'].astype('string')
    result['count'] = result['count'].astype('Int64')
    return result

issue_audit = audit_person_records(raw)
issue_counts = dict(zip(issue_audit['issue'], issue_audit['count'], strict=True))
assert raw.equals(raw_snapshot), 'Auditing must not mutate raw.'
'''

DECISIONS_SOURCE = r'''decision_table = pd.DataFrame(
    [
        ['full_name', 'empty optional name', 'retain as missing', 'A name is optional, so absence is reviewable rather than a reason to invent or remove a record.'],
        ['full_name, site, status', 'surrounding whitespace and case variants', 'strip surrounding whitespace and normalize bounded field case', 'These formatting changes preserve meaning and make documented categories comparable.'],
        ['status', 'NA sentinel', 'convert the documented sentinel to missing', 'The manifest exercise defines NA as absence here, not as a literal status category.'],
        ['age_text', 'unknown and -9 sentinels', 'convert the documented sentinels to missing', 'Both tokens explicitly encode an unavailable age and are not plausible ages to retain.'],
        ['age_text', 'nonnumeric, fractional, or out-of-range values', 'coerce invalid values to missing without rounding', 'Age is contracted as a whole number from 0 through 120; rounding would invent a value.'],
        ['visit_date', 'empty, lexically invalid, or calendar-invalid values', 'coerce invalid values to missing after an exact-format check', 'A visit date must use exact YYYY-MM-DD text and identify a real calendar date.'],
        ['all raw columns', 'exact duplicate submissions', 'keep the first exact raw row only', 'An identical repeated submission adds no distinct information; candidate conflicts are retained separately.'],
        ['all fields', 'adjacent-row filling', 'do not forward-fill or backward-fill', 'Adjacent rows represent different people and have no meaningful within-entity order.'],
    ],
    columns=['field', 'issue', 'action', 'reason'],
).astype('string')
'''

CLEAN_SOURCE = r'''def clean_person_records(raw_table):
    raw_duplicate_keep = ~raw_table.duplicated(keep='first')
    work = raw_table.copy(deep=True)

    names = work['full_name'].astype('string').str.strip()
    work['full_name'] = names.mask(names.eq('')).str.title()
    work['site'] = work['site'].astype('string').str.strip().str.lower()
    status_tokens = work['status'].astype('string').str.strip()
    status_normalized = status_tokens.str.lower()
    work['status'] = status_normalized.mask(
        status_tokens.eq('') | status_normalized.eq('na')
    )

    age_tokens = work['age_text'].astype('string').str.strip()
    age_missing = age_tokens.eq('') | age_tokens.str.lower().isin(['unknown', '-9'])
    age_numeric = pd.to_numeric(age_tokens.mask(age_missing), errors='coerce')
    age_valid = (
        age_numeric.notna()
        & np.isfinite(age_numeric)
        & age_numeric.mod(1).eq(0)
        & age_numeric.between(0, 120)
    )
    work['age'] = age_numeric.where(age_valid).astype('Int64')

    date_tokens = work['visit_date'].astype('string').str.strip()
    date_lexical = date_tokens.str.fullmatch(EXACT_DATE_PATTERN)
    work['visit_date'] = pd.to_datetime(
        date_tokens.where(date_lexical), format='%Y-%m-%d', errors='coerce'
    ).astype('datetime64[us]')
    work['record_id'] = work['record_id'].astype('string')

    clean_columns = [
        'record_id', 'full_name', 'site', 'status', 'age', 'visit_date'
    ]
    cleaned_table = work.loc[raw_duplicate_keep, clean_columns].copy()
    cleaned_table = cleaned_table.reset_index(drop=True)
    cleaned_table['needs_review'] = (
        cleaned_table['age'].isna() | cleaned_table['visit_date'].isna()
    ).astype('boolean')
    return cleaned_table

cleaned = clean_person_records(raw)
review_queue = cleaned.loc[cleaned['needs_review']].copy()
'''

VALIDATION_SOURCE = r'''def validate_clean_records(raw_table, raw_before, cleaned_table):
    expected_columns = [
        'record_id', 'full_name', 'site', 'status', 'age', 'visit_date',
        'needs_review',
    ]
    candidate_present = (
        cleaned_table['record_id'].notna()
        & cleaned_table['record_id'].str.strip().ne('')
    )
    expected_review = (
        cleaned_table['age'].isna() | cleaned_table['visit_date'].isna()
    ).astype('boolean')
    nonmissing_age = cleaned_table['age'].dropna()
    results = pd.Series(
        {
            'raw preserved': raw_table.equals(raw_before),
            'clean columns exact': cleaned_table.columns.tolist() == expected_columns,
            'row count follows raw exact duplicates': len(cleaned_table) == len(raw_table) - int(raw_table.duplicated(keep='first').sum()),
            'candidate identifier present': bool(candidate_present.all()),
            'candidate identifier unique': bool(cleaned_table['record_id'].is_unique),
            'site categories allowed': bool(cleaned_table['site'].dropna().isin(['north', 'south', 'west']).all()),
            'status categories allowed': bool(cleaned_table['status'].dropna().isin(['active', 'pending', 'complete']).all()),
            'record_id dtype string': str(cleaned_table['record_id'].dtype) == 'string',
            'full_name dtype string': str(cleaned_table['full_name'].dtype) == 'string',
            'site dtype string': str(cleaned_table['site'].dtype) == 'string',
            'status dtype string': str(cleaned_table['status'].dtype) == 'string',
            'age dtype Int64': str(cleaned_table['age'].dtype) == 'Int64',
            'age finite and in range': bool((np.isfinite(nonmissing_age) & nonmissing_age.between(0, 120)).all()),
            'visit_date dtype datetime64[us]': str(cleaned_table['visit_date'].dtype) == 'datetime64[us]',
            'review flag dtype boolean': str(cleaned_table['needs_review'].dtype) == 'boolean',
            'review flag nonmissing': bool(cleaned_table['needs_review'].notna().all()),
            'review flag exact': cleaned_table['needs_review'].equals(expected_review),
        },
        dtype='boolean',
    )
    return results

validation_results = validate_clean_records(raw, raw_snapshot, cleaned)
assert bool(validation_results.all()), validation_results[~validation_results].index.tolist()
'''

SAVE_SOURCE = r'''OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
decision_log = decision_table.copy(deep=True)
decision_log['source'] = pd.Series(
    ['data/people_raw.csv'] * len(decision_log), dtype='string'
)
decision_log['source_sha256'] = pd.Series(
    [EXPECTED_SOURCE_SHA256] * len(decision_log), dtype='string'
)
decision_log['rows_before'] = pd.Series(
    [len(raw)] * len(decision_log), dtype='Int64'
)
decision_log['rows_after'] = pd.Series(
    [len(cleaned)] * len(decision_log), dtype='Int64'
)
decision_log = decision_log[
    ['field', 'issue', 'action', 'reason', 'source', 'source_sha256',
     'rows_before', 'rows_after']
]

issue_audit.to_csv(AUDIT_PATH, index=False)
cleaned.to_csv(CLEANED_PATH, index=False, date_format='%Y-%m-%d')
decision_log.to_csv(DECISION_LOG_PATH, index=False)

round_trip = pd.read_csv(
    CLEANED_PATH,
    dtype={
        'record_id': 'string',
        'full_name': 'string',
        'site': 'string',
        'status': 'string',
        'age': 'Int64',
        'visit_date': 'string',
        'needs_review': 'boolean',
    },
)
round_trip_lexical = round_trip['visit_date'].str.fullmatch(
    EXACT_DATE_PATTERN, na=True
)
assert bool(round_trip_lexical.all())
round_trip['visit_date'] = pd.to_datetime(
    round_trip['visit_date'], format='%Y-%m-%d', errors='coerce'
).astype('datetime64[us]')
audit_round_trip = pd.read_csv(
    AUDIT_PATH, dtype={'issue': 'string', 'count': 'Int64'}
)
decision_round_trip = pd.read_csv(
    DECISION_LOG_PATH,
    dtype={
        'field': 'string',
        'issue': 'string',
        'action': 'string',
        'reason': 'string',
        'source': 'string',
        'source_sha256': 'string',
        'rows_before': 'Int64',
        'rows_after': 'Int64',
    },
)
pd.testing.assert_frame_equal(round_trip, cleaned.reset_index(drop=True))
pd.testing.assert_frame_equal(audit_round_trip, issue_audit.reset_index(drop=True))
pd.testing.assert_frame_equal(decision_round_trip, decision_log.reset_index(drop=True))
'''

CORRECT_SOURCES = {
    "a05-task1-contract": CONTRACT_MARKDOWN,
    "a05-task1-code": TASK1_SOURCE,
    "a05-task2-decisions": DECISIONS_SOURCE,
    "a05-task2-explanation": EXPLANATION_MARKDOWN,
    "a05-task2-clean": CLEAN_SOURCE,
    "a05-task3-validation": VALIDATION_SOURCE,
    "a05-task3-save": SAVE_SOURCE,
}


def _copy_starter(destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {"_grader_selftest", ".venv", "__pycache__", ".ipynb_checkpoints"}.intersection(names)

    shutil.copytree(ASSIGNMENT_DIR, destination, ignore=ignore)


def _load_notebook(root: Path) -> dict:
    return json.loads((root / "assignment.ipynb").read_text(encoding="utf-8"))


def _write_notebook(root: Path, notebook: dict) -> None:
    (root / "assignment.ipynb").write_text(
        json.dumps(notebook, indent=1, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _materialize_correct(root: Path) -> None:
    notebook = _load_notebook(root)
    for cell in notebook["cells"]:
        if cell["id"] in CORRECT_SOURCES:
            cell["source"] = CORRECT_SOURCES[cell["id"]].splitlines(keepends=True)
            if cell["source"]:
                cell["source"][-1] = cell["source"][-1].rstrip("\n")
        if cell["cell_type"] == "code":
            cell["execution_count"] = None
            cell["outputs"] = []
    _write_notebook(root, notebook)
    _execute(root, root)


def _assert_result_schema(result: dict) -> None:
    assert set(result) == {
        "schema",
        "assignment",
        "submission",
        "commit",
        "release",
        "review",
        "datetime",
        "score",
        "max-score",
        "tests",
    }
    assert result["schema"] == "datasci217/grading-result/v1"
    assert all(
        isinstance(result[field], str) and result[field]
        for field in (
            "assignment",
            "submission",
            "commit",
            "release",
            "review",
            "datetime",
        )
    )
    assert isinstance(result["score"], int)
    assert UTC_DATETIME.fullmatch(result["datetime"])
    assert result["max-score"] == 85
    assert isinstance(result["tests"], list) and len(result["tests"]) == 3
    assert sum(test["score"] for test in result["tests"]) == result["score"]
    assert sum(test["max-score"] for test in result["tests"]) == 85
    for test in result["tests"]:
        assert set(test) == {"test-name", "passed", "score", "max-score"}
        assert isinstance(test["test-name"], str) and test["test-name"]
        assert isinstance(test["passed"], bool)
        assert test["score"] in {0, test["max-score"]}


def _run_public(root: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, str(root / "check_assignment.py")],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=120,
        check=False,
    )


def _clone(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)


def _assert_delivery_inventory(correct: Path, temporary: Path) -> None:
    accepted = temporary / "accepted-delivery"
    _clone(correct, accepted)
    git_config = accepted / ".git/config"
    git_config.parent.mkdir(parents=True)
    git_config.write_text("[core]\n", encoding="utf-8")
    assert _run_public(accepted).returncode == 0
    assert grade_submission(accepted)["score"] == 85
    for label, relative in (
        ("legacy-delivery-metadata", ".classroom50.yaml"),
        ("autograde-workflow", ".github/workflows/autograde.yaml"),
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
        assert grade_submission(rejected)["score"] < 85


def _replace_cell_text(root: Path, cell_id: str, old: str, new: str) -> None:
    notebook = _load_notebook(root)
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    source = "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]
    assert old in source, f"self-test mutation target not found: {old}"
    cell["source"] = source.replace(old, new).splitlines(keepends=True)
    if cell["source"]:
        cell["source"][-1] = cell["source"][-1].rstrip("\n")
    _write_notebook(root, notebook)


def main() -> int:
    runner_env = {
        "ASSIGNMENT": "assignment-05",
        "SUBMISSION_TAG": "submit/local-correct",
        "COMMIT_URL": "https://example.invalid/commit/correct",
        "RELEASE_URL": "https://example.invalid/release/correct",
        "REVIEW_URL": "https://example.invalid/review/correct",
    }
    os.environ.update(runner_env)
    with tempfile.TemporaryDirectory(prefix="a05-selftest-") as temporary_name:
        temporary = Path(temporary_name)

        starter = temporary / "starter"
        _copy_starter(starter)
        starter_public = _run_public(starter)
        assert starter_public.returncode == 1
        assert "Complete every TODO" in starter_public.stdout

        correct = temporary / "correct"
        _copy_starter(correct)
        _materialize_correct(correct)
        correct_public = _run_public(correct)
        assert correct_public.returncode == 0, correct_public.stdout + correct_public.stderr
        assert "All public checks passed." in correct_public.stdout
        correct_result = grade_submission(correct)
        _assert_result_schema(correct_result)
        assert correct_result["score"] == 85, correct_result
        _assert_delivery_inventory(correct, temporary)

        emitted_env = {
            **dict(os.environ),
            **runner_env,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        emitted = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "autograder.py")],
            cwd=correct,
            env=emitted_env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        assert emitted.returncode == 0, emitted.stdout + emitted.stderr
        emitted_result = json.loads((correct / "result.json").read_text(encoding="utf-8"))
        _assert_result_schema(emitted_result)
        assert emitted_result["score"] == emitted_result["max-score"] == 85
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
                timeout=180,
                check=False,
            )
            assert fallback.returncode == 0, fallback.stdout + fallback.stderr
            fallback_result = json.loads((fallback_cwd / "result.json").read_text())
            _assert_result_schema(fallback_result)
            assert fallback_result["review"] == fallback_result["commit"]

        captured_failure = temporary / "captured student failure"
        _copy_starter(captured_failure)
        failure_cwd = temporary / "captured failure result"
        failure_cwd.mkdir()
        failed = subprocess.run(
            [sys.executable, str(Path(__file__).parent / "autograder.py"), str(captured_failure)],
            cwd=failure_cwd,
            env=emitted_env,
            text=True,
            capture_output=True,
            timeout=180,
            check=False,
        )
        assert failed.returncode == 0, failed.stdout + failed.stderr
        failed_result = json.loads((failure_cwd / "result.json").read_text())
        _assert_result_schema(failed_result)
        assert failed_result["score"] == 0

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

        stored_output = temporary / "stored-output"
        _copy_starter(stored_output)
        notebook = _load_notebook(stored_output)
        for cell in notebook["cells"]:
            if cell["id"] in CORRECT_SOURCES and cell["cell_type"] == "markdown":
                cell["source"] = CORRECT_SOURCES[cell["id"]].splitlines(keepends=True)
            elif cell["id"] in CORRECT_SOURCES:
                source = "".join(cell["source"]).replace("TODO", "unfinished")
                cell["source"] = source.splitlines(keepends=True)
            if cell["cell_type"] == "code":
                cell["execution_count"] = 999
                cell["outputs"] = [
                    {
                        "name": "stdout",
                        "output_type": "stream",
                        "text": ["Assignment 05 fresh-run verification passed\n"],
                    }
                ]
        _write_notebook(stored_output, notebook)
        output = stored_output / "output"
        for name in OUTPUT_FILES:
            (output / name).write_text("fake,stored,success\n", encoding="utf-8")
        stored_result = grade_submission(stored_output)
        _assert_result_schema(stored_result)
        assert stored_result["score"] < 85

        stale = temporary / "stale-committed"
        _clone(correct, stale)
        (stale / "output" / "cleaned_people.csv").write_text("stale\n", encoding="utf-8")
        stale_result = grade_submission(stale)
        _assert_result_schema(stale_result)
        assert stale_result["score"] < 85

        missing_output = temporary / "missing-output"
        _clone(correct, missing_output)
        (missing_output / "output" / "decision_log.csv").unlink()
        missing_output_result = grade_submission(missing_output)
        _assert_result_schema(missing_output_result)
        assert missing_output_result["score"] < 85

        corrupt = temporary / "corrupt-fixture"
        _clone(correct, corrupt)
        (corrupt / "data" / "people_raw.csv").write_bytes(b"corrupt\n")
        corrupt_result = grade_submission(corrupt)
        _assert_result_schema(corrupt_result)
        assert corrupt_result["score"] == 0

        missing_fixture = temporary / "missing-fixture"
        _clone(correct, missing_fixture)
        (missing_fixture / "data" / "fixture.json").unlink()
        missing_fixture_result = grade_submission(missing_fixture)
        _assert_result_schema(missing_fixture_result)
        assert missing_fixture_result["score"] == 0

        forbidden = temporary / "forbidden-api"
        _clone(correct, forbidden)
        _replace_cell_text(
            forbidden,
            "a05-task2-clean",
            "cleaned = clean_person_records(raw)",
            "_forbidden_result = raw.ffill()\ncleaned = clean_person_records(raw)",
        )
        forbidden_result = grade_submission(forbidden)
        _assert_result_schema(forbidden_result)
        assert forbidden_result["score"] == 0

        mutation = temporary / "input-mutation"
        _clone(correct, mutation)
        _replace_cell_text(
            mutation,
            "a05-task2-clean",
            "work = raw_table.copy(deep=True)",
            "work = raw_table",
        )
        mutation_result = grade_submission(mutation)
        _assert_result_schema(mutation_result)
        assert mutation_result["score"] < 85

        collision = temporary / "normalization-collision"
        _clone(correct, collision)
        _replace_cell_text(
            collision,
            "a05-task2-clean",
            "cleaned_table = work.loc[raw_duplicate_keep, clean_columns].copy()",
            "cleaned_table = work.drop_duplicates(keep='first').loc[:, clean_columns].copy()",
        )
        collision_result = grade_submission(collision)
        _assert_result_schema(collision_result)
        assert collision_result["score"] < 85
        task2 = next(
            test
            for test in collision_result["tests"]
            if test["test-name"] == "Task 2 automated"
        )
        assert not task2["passed"]

    print("Assignment 05 grader self-test passed.")
    print("Covered starter, correct, stored-output distrust, relocated and stale/deleted outputs,")
    print("corrupt/missing fixtures, mutation, normalization collision, alternate values/schema,")
    print("forbidden APIs, public-check independence, and datasci217/grading-result/v1 structure.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
