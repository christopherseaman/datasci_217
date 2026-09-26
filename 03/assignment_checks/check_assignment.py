"""Run the artifact-only Assignment 03 checks, print the score, and say what to fix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from _public_checks import COMPLETE_NOTE, PUBLIC_CHECKS, SCOPE_NOTE, SCORE_LABEL
from grading import grade_submission


# The file each check reads and the check's name within it.
WHERE = {check.name: (check.artifact, check.label) for check in PUBLIC_CHECKS}


def left_to_fix(tests: list[dict]) -> str:
    """The failing checks, grouped by the file they read.

    A file whose checks all fail reads "output/vitals_summary.txt (all 15 checks)"; otherwise its
    failing checks are named, as in "output/vitals_summary.txt: mean_sbp and sd_sbp".
    """
    groups: dict[str, list[tuple[str, bool]]] = {}
    for test in tests:
        artifact, label = WHERE.get(test["test-name"], (test["test-name"], test["test-name"]))
        groups.setdefault(artifact, []).append((label, test["passed"]))
    parts = []
    for artifact, members in groups.items():
        failing = [label for label, passed in members if not passed]
        if not failing:
            continue
        if len(members) > 1 and len(failing) == len(members):
            parts.append(f"{artifact} ({'both' if len(members) == 2 else f'all {len(members)}'} checks)")
        elif failing == [artifact]:
            parts.append(artifact)
        else:
            if len(failing) > 4:
                failing = failing[:3] + [f"{len(failing) - 3} more"]
            named = failing[0] if len(failing) == 1 else f"{', '.join(failing[:-1])} and {failing[-1]}"
            parts.append(f"{artifact}: {named}")
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
        print(SCOPE_NOTE)
        print()
        # A fix shared by the checks right after it, such as a missing file, is printed once.
        previous = None
        for test in result["tests"]:
            status = "PASS" if test["passed"] else "FIX "
            line = f"[{status}] {test['score']:>3}/{test['max-score']:<3} {test['test-name']}"
            if test["detail"] and test["detail"] == previous:
                print(f"{line}  (same fix as above)")
            else:
                print(line)
                if test["detail"]:
                    print(f"         {test['detail']}")
            previous = test["detail"] or None
        print(f"\n{SCORE_LABEL}: {result['score']}/{result['max-score']}")
        if result["score"] == result["max-score"]:
            print(COMPLETE_NOTE)
        else:
            lost = result["max-score"] - result["score"]
            print(f"Left to fix ({lost} point{'' if lost == 1 else 's'}): {left_to_fix(result['tests'])}.")
    return 0 if all(test["passed"] for test in result["tests"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
