"""Independent artifact grader for Assignment 07; never imports student code."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import argparse


TEST_SPECS = (("Task 1 bounded exploration", 25), ("Task 2 critique and redesign", 37), ("Task 3 explanatory evidence", 38))
REFERENCE_ROOT = Path(__file__).resolve().parent
SESSION_COLUMNS = ("session_id", "pathway", "activities_completed", "reflection_score")
PATHWAY_COLUMNS = ("pathway", "checkpoint_number", "completion_percent")
VARIABLE_ROLES = {"pathway": "categorical", "checkpoint_number": "ordered", "completion_percent": "quantitative"}
NUMERIC_FIELDS = {"activities_completed", "reflection_score", "checkpoint_number", "completion_percent"}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


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


def _task1(root: Path) -> None:
    spec = _json(root / "output" / "exploratory_spec.json")
    _assert(isinstance(spec, dict), "spec is not an object")
    mark = spec.get("mark")
    _assert((mark.get("type") if isinstance(mark, dict) else mark) == "point", "exploration must use point marks")
    expected = _rows(REFERENCE_ROOT / "data" / "session_observations.csv", SESSION_COLUMNS)
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
    _assert(_rows(root / "output" / "explanatory_supporting_data.csv", PATHWAY_COLUMNS) == _rows(REFERENCE_ROOT / "data" / "pathway_checkpoints.csv", PATHWAY_COLUMNS), "supporting values differ from fixture")
    evidence = _json(root / "output" / "visualization_evidence.json")
    _assert(isinstance(evidence, dict), "evidence is not an object")
    for key in ("question", "audience", "intended_claim", "displayed_unit", "grain", "text_alternative"):
        _text(evidence.get(key), f"evidence {key}")
    _assert(evidence.get("variable_roles") == VARIABLE_ROLES, "evidence roles wrong")
    text = root / "output" / "explanatory_text_alternative.txt"
    _assert(text.is_file() and not text.is_symlink(), "text alternative missing")
    _assert(_text(text.read_text(encoding="utf-8"), "text alternative") == _text(evidence["text_alternative"], "evidence text_alternative"), "text alternative differs from evidence")


def _record(name: str, maximum: int, error: Exception | None) -> dict:
    return {"test-name": name, "passed": error is None, "score": maximum if error is None else 0, "max-score": maximum,
            "detail": "" if error is None else str(error)}


def grade_submission(submission_root: str | Path) -> dict:
    root = Path(submission_root).resolve()
    checks = (_task1, _task2, _task3)
    tests = []
    for (name, maximum), check in zip(TEST_SPECS, checks, strict=True):
        try:
            check(root)
        except Exception as error:
            tests.append(_record(name, maximum, error))
        else:
            tests.append(_record(name, maximum, None))
    return {"schema": "datasci217/grading-result/v1", "score": sum(test["score"] for test in tests), "max-score": 100, "tests": tests}

def _format_result(result: dict) -> str:
    return "\n".join(
        f"[{'PASS' if test['passed'] else 'FAIL'}] {test['test-name']}: "
        f"{test['score']}/{test['max-score']}"
        + (f" ({test['detail']})" if test.get("detail") else "")
        for test in result["tests"]
    ) + f"\nScore: {result['score']}/{result['max-score']}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Grade committed assignment artifacts without executing submission code.")
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args(argv)
    result = grade_submission(args.submission_dir)
    if args.as_json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(_format_result(result))
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
