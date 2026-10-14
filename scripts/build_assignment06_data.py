"""Refresh Assignment 06's data_manifest.json from the handout's generate_data.py.

    python3 scripts/build_assignment06_data.py

The handout ships `06/assignment/generate_data.py`, never the data. This writes
each generated file's row count, columns, and the SHA-256 of its table as CSV
text into `06/assignment/data_manifest.json`, which the notebook's first cell
checks. Hashing the CSV text rather than the file covers the Parquet files too,
whose bytes vary with the pyarrow version. Copy the same SHA-256 values into
`DATA_SHA256` in `06/assignment_checks/grading.py`.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from pathlib import Path
import sys

HANDOUT = Path(__file__).resolve().parents[1] / "06" / "assignment"
sys.path.insert(0, str(HANDOUT))
sys.dont_write_bytecode = True
import generate_data  # noqa: E402


def main() -> None:
    manifest = {"fixture_id": "a06-clinic-midterm-v5",
                "provenance": "course-authored synthetic teaching data; no real patients",
                "generator": "generate_data.py",
                "sha256": "of each table as CSV text with \\n line endings, as to_csv(index=False) writes it",
                "files": {}}
    for name, data in generate_data.files().items():
        header, *rows = csv.reader(io.StringIO(data.decode("utf-8")))
        manifest["files"][name] = {"rows": len(rows), "columns": header,
                                   "sha256": hashlib.sha256(data).hexdigest()}
    path = HANDOUT / "data_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    for name, entry in manifest["files"].items():
        print(f'    "{name}": "{entry["sha256"]}",')


if __name__ == "__main__":
    main()
