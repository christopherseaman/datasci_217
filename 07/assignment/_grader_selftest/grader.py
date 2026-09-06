"""Independent artifact grader for Assignment 07; never imports student code."""

from __future__ import annotations

import csv
import datetime
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import sys


TEST_SPECS = (("Fixtures and reproducibility", 10), ("Task 1 bounded exploration", 15), ("Task 2 critique and redesign", 25), ("Task 3 explanatory evidence", 25), ("Artifact integrity", 5))
REQUIRED_CONTEXT_ENV = {"assignment": "ASSIGNMENT", "submission": "SUBMISSION_TAG", "commit": "COMMIT_URL", "release": "RELEASE_URL"}
OUTPUT_NAMES = {"exploratory_spec.json", "critique_redesign.png", "pathway_explanatory.png", "explanatory_supporting_data.csv", "visualization_evidence.json", "explanatory_text_alternative.txt"}
SESSION_COLUMNS = ("session_id", "pathway", "activities_completed", "reflection_score")
PATHWAY_COLUMNS = ("pathway", "checkpoint_number", "completion_percent")
VARIABLE_ROLES = {"pathway": "categorical", "checkpoint_number": "ordered", "completion_percent": "quantitative"}
NUMERIC_FIELDS = {"activities_completed", "reflection_score", "checkpoint_number", "completion_percent"}
PROTECTED_FILE_SHA256 = {"check_assignment.py": "c4ae3864045f948ea6976f42f38ab9e3e0aca295dfd413802b57a2b912d1be7e"}


class InfrastructureError(RuntimeError):
    pass


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _context() -> dict[str, str]:
    context = {key: os.environ.get(environment, "").strip() for key, environment in REQUIRED_CONTEXT_ENV.items()}
    missing = [REQUIRED_CONTEXT_ENV[key] for key, value in context.items() if not value]
    if missing:
        raise InfrastructureError("missing required grading context: " + ", ".join(missing))
    context["review"] = os.environ.get("REVIEW_URL", "").strip() or context["commit"]
    context["datetime"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return context


def _json(path: Path):
    _assert(path.is_file() and not path.is_symlink(), f"missing {path.name}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"invalid JSON: {path.name}: {error}") from error


def _row(row: dict, columns: tuple[str, ...]) -> tuple:
    values = []
    for column in columns:
        value = row.get(column)
        if column in NUMERIC_FIELDS:
            try:
                value = round(float(value), 6)
            except (TypeError, ValueError) as error:
                raise AssertionError(f"{column} must be numeric") from error
            _assert(math.isfinite(value), f"{column} must be finite")
        else:
            _assert(isinstance(value, str), f"{column} must be text")
        values.append(value)
    return tuple(values)


def _rows(path: Path, columns: tuple[str, ...]) -> set[tuple]:
    _assert(path.is_file() and not path.is_symlink(), f"missing {path.name}")
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _assert(set(reader.fieldnames or ()) == set(columns), f"wrong columns: {path.name}")
        rows = [_row(row, columns) for row in reader]
    _assert(len(rows) == len(set(rows)), f"duplicate rows: {path.name}")
    return set(rows)


def _png(path: Path) -> None:
    _assert(path.is_file() and not path.is_symlink(), f"missing {path.name}")
    _assert(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"not a PNG: {path.name}")


def _text(value, label: str) -> str:
    _assert(isinstance(value, str) and value.strip(), f"{label} must be nonblank text")
    return value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def _values(spec: dict) -> list | None:
    data = spec.get("data")
    if isinstance(data, dict) and isinstance(data.get("values"), list):
        return data["values"]
    if isinstance(data, dict) and isinstance(data.get("name"), str) and isinstance(spec.get("datasets"), dict):
        value = spec["datasets"].get(data["name"])
        return value if isinstance(value, list) else None
    return None


def _fixtures(root: Path) -> None:
    _assert((root / "assignment.ipynb").is_file(), "student notebook source is missing")
    _assert(_rows(root / "data" / "session_observations.csv", SESSION_COLUMNS), "empty session fixture")
    _assert(_rows(root / "data" / "pathway_checkpoints.csv", PATHWAY_COLUMNS), "empty pathway fixture")
    manifest = _json(root / "data" / "fixture.json")
    _assert(isinstance(manifest, dict) and manifest.get("fixture_set_id") == "a07-visualization-v1", "fixture manifest changed")


def _task1(root: Path) -> None:
    spec = _json(root / "output" / "exploratory_spec.json")
    _assert(isinstance(spec, dict), "spec is not an object")
    mark = spec.get("mark")
    _assert((mark.get("type") if isinstance(mark, dict) else mark) == "point", "exploration must use point marks")
    expected = _rows(root / "data" / "session_observations.csv", SESSION_COLUMNS)
    values = _values(spec)
    _assert(isinstance(values, list), "spec does not embed data")
    seen = {_row(row, SESSION_COLUMNS) for row in values if isinstance(row, dict)}
    _assert(len(values) == len(seen) and seen == expected, "spec data differs from session fixture")
    enc = spec.get("encoding")
    _assert(isinstance(enc, dict), "spec needs encodings")
    for channel, field, types in (("x", "activities_completed", {"q", "quantitative"}), ("y", "reflection_score", {"q", "quantitative"}), ("color", "pathway", {"n", "nominal"}), ("shape", "pathway", {"n", "nominal"})):
        value = enc.get(channel)
        _assert(isinstance(value, dict) and value.get("field") == field and str(value.get("type", "")).lower() in types, f"wrong {channel} encoding")


def _task2(root: Path) -> None:
    _png(root / "output" / "critique_redesign.png")
    evidence = _json(root / "output" / "visualization_evidence.json")
    critique = evidence.get("critique") if isinstance(evidence, dict) else None
    _assert(isinstance(critique, list) and len(critique) == 5, "five critique entries required")
    _assert({entry.get("category") for entry in critique if isinstance(entry, dict)} == {"unsupported claim", "truncated baseline", "missing unit", "color-only encoding", "distracting decoration"}, "critique categories incomplete")
    for entry in critique:
        _assert(isinstance(entry, dict), "critique entry is not an object")
        _text(entry.get("problem"), "critique problem")
        _text(entry.get("repair"), "critique repair")


def _task3(root: Path) -> None:
    _png(root / "output" / "pathway_explanatory.png")
    _assert(_rows(root / "output" / "explanatory_supporting_data.csv", PATHWAY_COLUMNS) == _rows(root / "data" / "pathway_checkpoints.csv", PATHWAY_COLUMNS), "supporting values differ from fixture")
    evidence = _json(root / "output" / "visualization_evidence.json")
    _assert(isinstance(evidence, dict), "evidence is not an object")
    for key in ("question", "audience", "intended_claim", "displayed_unit", "grain", "text_alternative"):
        _text(evidence.get(key), f"evidence {key}")
    _assert(evidence.get("variable_roles") == VARIABLE_ROLES, "evidence roles wrong")
    text = root / "output" / "explanatory_text_alternative.txt"
    _assert(text.is_file() and not text.is_symlink(), "text alternative missing")
    _assert(_text(text.read_text(encoding="utf-8"), "text alternative") == _text(evidence["text_alternative"], "evidence text_alternative"), "text alternative differs from evidence")


def _integrity(root: Path) -> None:
    output = root / "output"
    _assert(output.is_dir() and not output.is_symlink(), "output directory missing")
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert(OUTPUT_NAMES <= actual, "required artifact missing")
    _assert(not any((output / name).is_symlink() for name in OUTPUT_NAMES), "artifact symlink used")
    for relative, expected in PROTECTED_FILE_SHA256.items():
        path = root / relative
        _assert(path.is_file() and sha256(path.read_bytes()).hexdigest() == expected, f"protected file changed: {relative}")


def _record(name: str, maximum: int, error: Exception | None) -> dict:
    return {"test-name": name, "passed": error is None, "score": maximum if error is None else 0, "max-score": maximum}


def grade_submission(submission_root: str | Path) -> dict:
    root = Path(submission_root).resolve()
    context = _context()
    checks = (_fixtures, _task1, _task2, _task3, _integrity)
    tests = []
    for (name, maximum), check in zip(TEST_SPECS, checks, strict=True):
        try:
            check(root)
        except Exception as error:
            print(f"[FAIL] {name}: {error}")
            tests.append(_record(name, maximum, error))
        else:
            print(f"[PASS] {name}")
            tests.append(_record(name, maximum, None))
    return {"schema": "datasci217/grading-result/v1", **context, "score": sum(test["score"] for test in tests), "max-score": 80, "tests": tests}


def main() -> int:
    try:
        result = grade_submission(Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd())
    except InfrastructureError as error:
        print(f"[INFRASTRUCTURE] {error}", file=sys.stderr)
        return 2
    Path("result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
