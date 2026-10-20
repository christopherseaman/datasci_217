"""Refresh Assignment 06's data_manifest.json from the handout's generate_data.py.

    python3 scripts/build_assignment06_data.py

The handout ships `06/mid-term/generate_data.py`, never the data. This writes
the SHA-256 of each generated file's table as CSV text into
`06/mid-term/data_manifest.json`, which the notebook's first cell checks. Hashing the CSV text rather than the file covers the Parquet files too,
whose bytes vary with the pyarrow version. Copy the same SHA-256 values into
`DATA_SHA256` in `06/mid-term_checks/grading.py`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

HANDOUT = Path(__file__).resolve().parents[1] / "06" / "mid-term"
sys.path.insert(0, str(HANDOUT))
sys.dont_write_bytecode = True
import generate_data  # noqa: E402


def main() -> None:
    manifest = {"fixture_id": "a06-clinic-midterm-v8",
                "provenance": "course-authored synthetic teaching data; no real patients",
                "generator": "generate_data.py",
                "sha256": "of each table as CSV text with \\n line endings, as to_csv(index=False) writes it",
                "files": {}}
    for name, data in generate_data.files().items():
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest()}
    path = HANDOUT / "data_manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    for name, entry in manifest["files"].items():
        print(f'    "{name}": "{entry["sha256"]}",')


if __name__ == "__main__":
    main()
