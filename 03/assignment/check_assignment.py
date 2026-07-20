"""Run the supplied public Assignment 03 checks without local pytest."""

from pathlib import Path

from _public_checks import run_public_checks


assignment_dir = Path(__file__).resolve().parent
results = run_public_checks(assignment_dir)
failure_count = 0

for name, error in results:
    if error is None:
        print(f"[PASS] {name}")
    else:
        failure_count += 1
        print(f"[FIX]  {name}: {error}")

if failure_count:
    print(f"\n{failure_count} public check(s) still need attention.")
    raise SystemExit(1)

print("\nAll public checks passed.")
