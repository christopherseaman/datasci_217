#!/usr/bin/env python3
"""Exercise private roster generation with synthetic workbooks and local commits."""

import csv
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from xml.sax.saxutils import escape
import zipfile

import build_roster


def workbook(path, rows, shared=False):
    strings = []
    xml_rows = []
    for number, row in enumerate([["Last Name", "First Name", "Email"], *rows], 1):
        cells = []
        for column, text in zip("ABC", row):
            if shared:
                strings.append(text)
                value = f'<c r="{column}{number}" t="s"><v>{len(strings) - 1}</v></c>'
            else:
                value = f'<c r="{column}{number}" t="inlineStr"><is><t>{escape(text)}</t></is></c>'
            cells.append(value)
        xml_rows.append(f'<row r="{number}">{"".join(cells)}</row>')
    namespace = build_roster.NS["s"]
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/worksheets/sheet1.xml", f'<worksheet xmlns="{namespace}"><sheetData>{"".join(xml_rows)}</sheetData></worksheet>')
        if shared:
            items = "".join(f'<si><t>{escape(text)}</t></si>' for text in strings)
            archive.writestr("xl/sharedStrings.xml", f'<sst xmlns="{namespace}">{items}</sst>')


def clone(root, user, key):
    path = root / user
    path.mkdir()
    commands = [["init", "--quiet"], ["remote", "add", "origin", f"https://github.com/{user}/renamed-assignment.git"]]
    for args in commands:
        subprocess.run(["git", *args], cwd=path, check=True, capture_output=True)
    (path / "output").mkdir()
    (path / "output/student_identity.txt").write_text(key + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "output/student_identity.txt"], cwd=path, check=True, capture_output=True)
    subprocess.run(["git", "-c", "user.name=Synthetic", "-c", "user.email=synthetic@example.invalid",
                    "commit", "--quiet", "-m", "Add synthetic identity"], cwd=path, check=True, capture_output=True)
    return path


class RosterTests(unittest.TestCase):
    def setUp(self):
        scratch = build_roster.REPO / "scratch"
        scratch.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="roster-test-", dir=scratch)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.book = self.root / "enrollment.xlsx"
        self.submissions = self.root / "submissions"
        self.submissions.mkdir()
        self.key = build_roster.identity.process_email("test.person@ucsf.edu")["hash"]

    def test_cli_joins_committed_identity_preserves_unmatched_and_output(self):
        workbook(self.book, [["Example", "Test", "TEST.PERSON@ucsf.edu"], ["Other", "Person", "other@ucsf.edu"],
                             ["Incomplete", "Person", ""]], shared=True)
        submission = clone(self.submissions, "synthetic-user", self.key.upper())
        (submission / "output/student_identity.txt").write_text("uncommitted edit", encoding="utf-8")
        output = self.root / "private/roster.tsv"
        command = [sys.executable, str(Path(build_roster.__file__)), str(self.book),
                   "--submissions", str(self.submissions), "--output", str(output)]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("3 enrolled, 1 matched, 2 unmatched", result.stdout)
        self.assertNotIn("TEST.PERSON", result.stdout + result.stderr)
        with output.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle, delimiter="\t"))
        self.assertEqual(list(rows[0]), build_roster.COLUMNS)
        self.assertEqual(rows[0]["github_user"], "synthetic-user")
        self.assertNotIn("identity_hash", rows[0])
        self.assertEqual(rows[0]["course"], "ds217-26f")
        self.assertNotIn("semester", rows[0])
        self.assertEqual(rows[1]["github_user"], "")
        self.assertEqual(rows[2]["email"], "")
        self.assertEqual(output.stat().st_mode & 0o777, 0o600)
        previous = output.read_bytes()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 1)
        self.assertEqual(output.read_bytes(), previous)

    def test_workbook_hash_collision_fails_without_exposing_email(self):
        workbook(self.book, [["One", "Synthetic", "test.person@ucsf.edu"],
                             ["Two", "Synthetic", "test_person@ucsf.edu"]])
        with self.assertRaisesRegex(ValueError, "rows 2 and 3") as error:
            build_roster.read_students(self.book)
        self.assertNotIn("@", str(error.exception))

    def test_ambiguous_claims_are_unmatched_and_invalid_claims_counted(self):
        workbook(self.book, [["Example", "Test", "test.person@ucsf.edu"]])
        clone(self.submissions, "synthetic-one", self.key)
        clone(self.submissions, "synthetic-two", self.key)
        clone(self.submissions, "synthetic-three", "invalid")
        students = build_roster.read_students(self.book)
        counts = build_roster.match_students(students, self.submissions)
        self.assertEqual(students[0]["github_user"], "")
        self.assertEqual(counts["ambiguous"], 1)
        self.assertEqual(counts["missing_or_invalid"], 1)

    def test_invalid_email_and_non_github_origin_fail(self):
        workbook(self.book, [["Example", "Test", "invalid@example.invalid"]])
        with self.assertRaisesRegex(ValueError, "row 2 needs a valid course email"):
            build_roster.read_students(self.book)
        submission = clone(self.submissions, "synthetic-user", self.key)
        subprocess.run(["git", "remote", "set-url", "origin", "https://example.invalid/owner/repo"],
                       cwd=submission, check=True, capture_output=True)
        with self.assertRaisesRegex(ValueError, "no GitHub origin"):
            build_roster.match_students([], self.submissions)


if __name__ == "__main__":
    unittest.main()
