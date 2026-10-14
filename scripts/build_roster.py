#!/usr/bin/env python3
"""Join a private enrollment workbook to committed Assignment 01 identities.

    python3 scripts/build_roster.py ds217-f26-roster.xlsx --submissions scratch/submissions/01

Reads local clones without changing or fetching them. The output contains private
student information: share it privately, never commit it. Existing output is
never replaced; choose a new --output when regenerating a roster.
"""

import argparse
import csv
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import xml.etree.ElementTree as ET
import zipfile

REPO = Path(__file__).resolve().parents[1]
COLUMNS = ["github_user", "email", "last_name", "first_name", "course", "semester"]
NS = {"s": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
spec = importlib.util.spec_from_file_location("course_process_email", REPO / "01/assignment/process_email.py")
identity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(identity)


def read_students(workbook):
    """Read the workbook's enrollment sheet, rejecting ambiguous identities."""
    with zipfile.ZipFile(workbook) as archive:
        shared = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared = ["".join(node.itertext()) for node in
                      ET.fromstring(archive.read("xl/sharedStrings.xml")).findall("s:si", NS)]
        sheets = [name for name in archive.namelist()
                  if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", name)]
        if len(sheets) != 1:
            raise ValueError("Expected one enrollment sheet; provide a workbook containing only that sheet.")
        rows = ET.fromstring(archive.read(sheets[0])).findall("s:sheetData/s:row", NS)
        headers, students, hashes = {}, [], {}
        for row in rows:
            values = {}
            for cell in row.findall("s:c", NS):
                column = re.sub(r"\d+$", "", cell.attrib["r"])
                value = cell.find("s:v", NS)
                text = value.text if value is not None and value.text else ""
                if cell.attrib.get("t") == "s":
                    text = shared[int(text)]
                elif cell.attrib.get("t") == "inlineStr":
                    text = "".join(cell.find("s:is", NS).itertext())
                values[column] = text.strip()
            if not any(values.values()):
                continue
            if not headers:
                for field, title in [("email", "Email"), ("first_name", "First Name"), ("last_name", "Last Name")]:
                    found = [column for column, value in values.items() if value.casefold() == title.casefold()]
                    if len(found) != 1:
                        raise ValueError(f"Expected exactly one {title} column in the workbook header.")
                    headers[field] = found[0]
                continue
            number = row.attrib["r"]
            student = {field: values.get(column, "") for field, column in headers.items()}
            if not student["first_name"] or not student["last_name"]:
                raise ValueError(f"Workbook row {number} is missing a name; correct it before rebuilding.")
            student["email"] = student["email"].lower()
            if not student["email"]:
                students.append({"github_user": "", **student, "identity_hash": ""})
                continue
            try:
                key = identity.process_email(student["email"])["hash"]
            except ValueError:
                raise ValueError(f"Workbook row {number} needs a valid course email ending in @ucsf.edu.") from None
            if key in hashes:
                raise ValueError(f"Workbook rows {hashes[key]} and {number} have duplicate or colliding identities; resolve them before rebuilding.")
            hashes[key] = number
            students.append({"github_user": "", **student, "identity_hash": key})
    if not students:
        raise ValueError("No enrolled students found in the workbook.")
    return students


def git(clone, *args):
    result = subprocess.run(["git", *args], cwd=clone, capture_output=True, text=True,
                            encoding="utf-8", errors="replace", timeout=30, check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def match_students(students, submissions):
    """Match verified GitHub clone origins and committed hashes, never worktree files."""
    enrolled = {student["identity_hash"]: student for student in students if student["identity_hash"]}
    claims, users = {}, set()
    counts = {"clones": 0, "missing_or_invalid": 0, "outside_roster": 0, "ambiguous": 0}
    for clone in sorted(submissions.iterdir()):
        if not (clone / ".git").exists():
            continue
        counts["clones"] += 1
        origin = git(clone, "config", "--get", "remote.origin.url") or ""
        match = re.fullmatch(r"(?:https://github\.com/|git@github\.com:)([A-Za-z0-9-]+)/[A-Za-z0-9_.-]+/?", origin, re.I)
        if not match:
            raise ValueError(f"Submission clone #{counts['clones']} has no GitHub origin; correct --submissions or that clone's origin.")
        user = match.group(1)
        if user.casefold() in users:
            raise ValueError("Multiple clones claim the same GitHub user; keep one clone per user in --submissions.")
        users.add(user.casefold())
        key = git(clone, "show", "HEAD:output/student_identity.txt")
        if key is None or not re.fullmatch(r"[0-9a-fA-F]{64}", key):
            counts["missing_or_invalid"] += 1
            continue
        key = key.lower()
        if key in claims:
            if claims[key] is not None:
                counts["ambiguous"] += 1
            claims[key] = None
            if key in enrolled:
                enrolled[key]["github_user"] = ""
            continue
        claims[key] = counts["clones"]
        if key in enrolled:
            enrolled[key]["github_user"] = user
        else:
            counts["outside_roster"] += 1
    if not counts["clones"]:
        raise ValueError("No submission clones found; point --submissions to the Assignment 01 clone directory.")
    return counts


def write_roster(students, output):
    semester = json.loads((REPO / "assignments.json").read_text(encoding="utf-8"))["term"]
    output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows({**student, "course": "DataSci 217", "semester": semester} for student in students)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("spreadsheet", type=Path)
    parser.add_argument("--submissions", type=Path, default=REPO / "scratch/submissions/01")
    parser.add_argument("--output", type=Path, default=Path("roster.tsv"))
    args = parser.parse_args(argv)
    try:
        if args.output.exists() or args.output.is_symlink():
            raise ValueError("Output already exists; choose a new --output to preserve the existing roster.")
        students = read_students(args.spreadsheet)
        counts = match_students(students, args.submissions)
        write_roster(students, args.output)
    except ValueError as error:
        parser.exit(1, f"Error: {error}\n")
    except (OSError, zipfile.BadZipFile, ET.ParseError, subprocess.TimeoutExpired):
        parser.exit(1, "Could not read inputs or create output; check file access and workbook format. No existing output was replaced.\n")
    matched = sum(bool(student["github_user"]) for student in students)
    print(f"Created private roster: {len(students)} enrolled, {matched} matched, {len(students) - matched} unmatched.")
    print(f"Read {counts['clones']} committed submissions: {counts['missing_or_invalid']} missing/invalid identities, {counts['outside_roster']} outside roster.")
    print(f"Manual review: {sum(not student['email'] for student in students)} missing emails, {counts['ambiguous']} identities claimed by multiple users (left unmatched).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
