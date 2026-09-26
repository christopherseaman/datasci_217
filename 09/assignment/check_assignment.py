"""Run the Assignment 09 checks against the saved CSV files and say what to fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from grading import grade_submission


def left_to_fix(tests: list[dict]) -> str:
    """The failing checks, grouped by the part of their name before the colon.

    A group whose checks all fail reads "fridge block (all 4 checks)"; otherwise its
    failing checks are named, as in "selected supplies: line C1833 and line C4105".
    """
    groups: dict[str, list[dict]] = {}
    for test in tests:
        groups.setdefault(test["test-name"].partition(": ")[0], []).append(test)
    parts = []
    for group, members in groups.items():
        failing = [test for test in members if not test["passed"]]
        if len(failing) == 1:
            parts.append(failing[0]["test-name"])
        elif failing and len(failing) == len(members):
            parts.append(f"{group} (all {len(members)} checks)")
        elif failing:
            names = [test["test-name"].partition(": ")[2] for test in failing]
            if len(names) > 4:
                names = names[:3] + [f"{len(names) - 3} more"]
            parts.append(f"{group}: {', '.join(names[:-1])} and {names[-1]}")
    return "; ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("submission_dir", nargs="?", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    result = grade_submission(args.submission_dir)
    if args.json:
        print(json.dumps(result))
    else:
        # A fix shared by the checks right after it, such as a missing file, is printed once.
        previous = None
        for test in result["tests"]:
            status = "PASS" if test["passed"] else "FIX "
            line = f"[{status}] {test['score']:>2}/{test['max-score']:<2} {test['test-name']}"
            if test["detail"] and test["detail"] == previous:
                print(f"{line}  (same fix as above)")
            else:
                print(line)
                if test["detail"]:
                    print(f"         {test['detail']}")
            previous = test["detail"] or None
        print(f"\nScore: {result['score']}/{result['max-score']}")
        if result["score"] == result["max-score"]:
            print("All checks passed.")
        else:
            lost = result["max-score"] - result["score"]
            print(f"Left to fix ({lost} point{'' if lost == 1 else 's'}): {left_to_fix(result['tests'])}.")
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
