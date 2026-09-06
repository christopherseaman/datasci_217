"""Public, dependency-free artifact checks for Assignment 07."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

ASSIGNMENT_DIR = Path(__file__).resolve().parent
OUTPUT_NAMES = {"exploratory_spec.json", "critique_redesign.png", "pathway_explanatory.png", "explanatory_supporting_data.csv", "visualization_evidence.json", "explanatory_text_alternative.txt"}
SESSION_COLUMNS = ("session_id", "pathway", "activities_completed", "reflection_score")
PATHWAY_COLUMNS = ("pathway", "checkpoint_number", "completion_percent")
VARIABLE_ROLES = {"pathway": "categorical", "checkpoint_number": "ordered", "completion_percent": "quantitative"}
NUMERIC_FIELDS = {"activities_completed", "reflection_score", "checkpoint_number", "completion_percent"}


def _assert(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _json(path: Path, label: str):
    _assert(path.is_file() and not path.is_symlink(), f"Missing {label}.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise AssertionError(f"{label} must be valid JSON: {error}") from error


def _row(row: dict, columns: tuple[str, ...]) -> tuple:
    values = []
    for column in columns:
        value = row.get(column)
        if column in NUMERIC_FIELDS:
            try:
                value = round(float(value), 6)
            except (TypeError, ValueError) as error:
                raise AssertionError(f"{column} must be numeric.") from error
            _assert(math.isfinite(value), f"{column} must be finite.")
        else:
            _assert(isinstance(value, str), f"{column} must be text.")
        values.append(value)
    return tuple(values)


def _rows(path: Path, columns: tuple[str, ...], label: str) -> set[tuple]:
    _assert(path.is_file() and not path.is_symlink(), f"Missing {label}.")
    try:
        with path.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            _assert(set(reader.fieldnames or ()) == set(columns), f"{label} has wrong columns.")
            rows = [_row(row, columns) for row in reader]
    except (UnicodeDecodeError, csv.Error, KeyError) as error:
        raise AssertionError(f"{label} is not a readable CSV: {error}") from error
    _assert(len(rows) == len(set(rows)), f"{label} has duplicate rows.")
    return set(rows)


def _png(path: Path) -> None:
    _assert(path.is_file() and not path.is_symlink(), f"Missing output/{path.name}.")
    _assert(path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", f"output/{path.name} is not a PNG file.")


def _text(value, label: str) -> str:
    _assert(isinstance(value, str) and value.strip(), f"{label} must be nonblank text.")
    return value.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def _values(spec: dict) -> list[dict] | None:
    data = spec.get("data")
    if isinstance(data, dict) and isinstance(data.get("values"), list):
        return data["values"]
    if isinstance(data, dict) and isinstance(data.get("name"), str) and isinstance(spec.get("datasets"), dict):
        values = spec["datasets"].get(data["name"])
        return values if isinstance(values, list) else None
    return None


def check_task1() -> None:
    spec = _json(ASSIGNMENT_DIR / "output" / "exploratory_spec.json", "output/exploratory_spec.json")
    _assert(isinstance(spec, dict), "Exploratory specification must be a JSON object.")
    mark = spec.get("mark")
    _assert((mark.get("type") if isinstance(mark, dict) else mark) == "point", "Exploratory specification needs point marks.")
    expected = _rows(ASSIGNMENT_DIR / "data" / "session_observations.csv", SESSION_COLUMNS, "session fixture")
    values = _values(spec)
    _assert(isinstance(values, list), "Exploratory specification must embed plotted data.")
    observed = {_row(row, SESSION_COLUMNS) for row in values if isinstance(row, dict)}
    _assert(len(values) == len(observed) and observed == expected, "Exploratory specification data must match the session fixture.")
    encoding = spec.get("encoding")
    _assert(isinstance(encoding, dict), "Exploratory specification needs encodings.")
    for channel, field, types in (("x", "activities_completed", {"q", "quantitative"}), ("y", "reflection_score", {"q", "quantitative"}), ("color", "pathway", {"n", "nominal"}), ("shape", "pathway", {"n", "nominal"})):
        value = encoding.get(channel)
        _assert(isinstance(value, dict) and value.get("field") == field and str(value.get("type", "")).lower() in types, f"Exploratory {channel} encoding is missing or wrong.")


def check_task2() -> None:
    _png(ASSIGNMENT_DIR / "output" / "critique_redesign.png")
    evidence = _json(ASSIGNMENT_DIR / "output" / "visualization_evidence.json", "output/visualization_evidence.json")
    critique = evidence.get("critique") if isinstance(evidence, dict) else None
    _assert(isinstance(critique, list) and len(critique) == 5, "Evidence needs five critique entries.")
    categories = {entry.get("category") for entry in critique if isinstance(entry, dict)}
    _assert(categories == {"unsupported claim", "truncated baseline", "missing unit", "color-only encoding", "distracting decoration"}, "Evidence critique categories are incomplete.")
    for entry in critique:
        _assert(isinstance(entry, dict), "Each critique entry must be an object.")
        _text(entry.get("problem"), "Critique problem")
        _text(entry.get("repair"), "Critique repair")


def check_task3() -> None:
    _png(ASSIGNMENT_DIR / "output" / "pathway_explanatory.png")
    expected = _rows(ASSIGNMENT_DIR / "data" / "pathway_checkpoints.csv", PATHWAY_COLUMNS, "pathway fixture")
    observed = _rows(ASSIGNMENT_DIR / "output" / "explanatory_supporting_data.csv", PATHWAY_COLUMNS, "output/explanatory_supporting_data.csv")
    _assert(observed == expected, "Supporting CSV values must match the pathway fixture.")
    evidence = _json(ASSIGNMENT_DIR / "output" / "visualization_evidence.json", "output/visualization_evidence.json")
    _assert(isinstance(evidence, dict), "Evidence must be a JSON object.")
    for key in ("question", "audience", "intended_claim", "displayed_unit", "grain", "text_alternative"):
        _text(evidence.get(key), f"Evidence {key}")
    _assert(evidence.get("variable_roles") == VARIABLE_ROLES, "Evidence variable roles are wrong.")
    text = ASSIGNMENT_DIR / "output" / "explanatory_text_alternative.txt"
    _assert(text.is_file() and not text.is_symlink(), "Text alternative must be a regular file.")
    _assert(_text(text.read_text(encoding="utf-8"), "Text alternative") == _text(evidence["text_alternative"], "Evidence text_alternative"), "Text alternative must match the evidence JSON (apart from line endings).")


def check_artifact_inventory() -> None:
    output = ASSIGNMENT_DIR / "output"
    _assert(output.is_dir() and not output.is_symlink(), "Missing output directory.")
    actual = {path.name for path in output.iterdir() if path.is_file() or path.is_symlink()}
    _assert(OUTPUT_NAMES <= actual, "Missing required output artifacts.")
    _assert(not any((output / name).is_symlink() for name in OUTPUT_NAMES), "Output artifacts must be regular files.")


def main() -> int:
    checks = (("Task 1", check_task1), ("Task 2", check_task2), ("Task 3", check_task3), ("artifact inventory", check_artifact_inventory))
    failures = []
    for label, check in checks:
        try:
            check()
        except Exception as error:
            failures.append(f"[FIX] {label}: {error}")
        else:
            print(f"[OK] {label}")
    if failures:
        print("\n".join(failures))
        return 1
    print("Artifact checks passed. Human chart and prose review remains separate.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
