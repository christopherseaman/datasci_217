"""Regression checks for the Assignment 01 checks.

Builds submissions in ignored `scratch/` and confirms that the checks score
each case as the README's contract states, that the handout ships them
unchanged, and that no artifact variant scores lower than it did under the
vendored checker they replaced. Nothing here reads or runs student code.
"""

from collections.abc import Callable
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile


CHECKS = Path(__file__).resolve().parents[1]
REPO = CHECKS.parents[1]
ASSIGNMENT = REPO / "01" / "assignment"
sys.path.insert(0, str(CHECKS))
import _value_checks as value  # noqa: E402
from _value_checks import CHECKS as VALUE_CHECKS, EXPECTED_READINESS, ROSTER_HASHES  # noqa: E402
from grading import POINTS, grade_submission  # noqa: E402

# The last commit whose fork carried the checks as 01/assignment/_assignment_checks.py, with one
# check for the practice files (20) and one for the report and identity together (80). This
# quarter's forks were first graded with it, so no submission may score lower now.
REPLACED_COMMIT = "f39598b76433ae1e626f574eff684c8d48561380"
REPLACED_FILES = ("_assignment_checks.py", "grading.py")

# A hash on the handout-time roster, so the replaced checker accepts it too; students added later
# are only on the new roster.
ROSTER_HASH = "07f66adfe2a38fb3a84a2ab2f55d40bdb5cabd85355ecb9f65290385c5cc76b8"
FIRST_LINE = EXPECTED_READINESS.partition("\n")[0] + "\n"

PRACTICE_TESTS = ["terminal-practice/source.txt", "terminal-practice/path-check.txt"]
LINE_TESTS = ["report: Project", "report: Script", *(f"report: Measurement {n}" for n in range(1, 5)),
              "report: Count", "report: Total", "report: Mean", "report: Review count", "report: Readiness",
              "report: Participant count", "report: Next checkpoint"]
IDENTITY_TEST = "identity hash on the roster"
TEST_NAMES = PRACTICE_TESTS + LINE_TESTS + [IDENTITY_TEST]


def task2_double_spaced(report: str) -> str:
    """Each Task 2 line printed with a space inside the label's quotes, so two follow its colon."""
    lines = report.splitlines(keepends=True)
    lines[3:11] = [line.replace(": ", ":  ", 1) for line in lines[3:11]]
    return "".join(lines)


def task2_unspaced(report: str) -> str:
    """Each Task 2 line printed with its label joined to the value, as `print("Count:" + str(count))` does."""
    lines = report.splitlines(keepends=True)
    lines[3:11] = [line.replace(": ", ":", 1) for line in lines[3:11]]
    return "".join(lines)


# A real submission: the running total compared with the threshold instead of each measurement,
# so 19 is labelled review and three readings are counted, printed with double spaces.
RUNNING_TOTAL = task2_double_spaced(
    EXPECTED_READINESS.replace("Measurement: 19 within range", "Measurement: 19 review")
    .replace("Review count: 2", "Review count: 3"))
MISSING_REPORT = "Run make_output.py (Task 3.2), then commit output/readiness.txt as a regular file."
# Report variants that lose no points: whitespace anywhere, and blank or extra lines anywhere.
FULL_MARKS_REPORTS = ("expected", "trailing space", "tab after colon", "no-break space", "no space after a colon",
                      "space before a colon", "no space after any Task 2 colon", "leading blank line",
                      "blank line inside", "blank lines between the scripts", "leftover debug line",
                      "extra final line", "double spaces after Task 2 colons", "leading and trailing spaces")


def load(name: str, path: Path):
    """Import a file under its own module name, as the file it is."""
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module  # dataclasses look their own module up by name
    specification.loader.exec_module(module)
    return module


def scratch(prefix: str) -> tempfile.TemporaryDirectory:
    (REPO / "scratch").mkdir(exist_ok=True)
    return tempfile.TemporaryDirectory(dir=REPO / "scratch", prefix=prefix)


def practice(root: Path) -> None:
    (root / "terminal-practice").mkdir(parents=True, exist_ok=True)
    for name in ("source.txt", "path-check.txt"):
        (root / "terminal-practice" / name).write_text("", encoding="utf-8")


def outputs(root: Path, report: str | bytes | None, identity: str | bytes | None) -> None:
    (root / "output").mkdir(parents=True, exist_ok=True)
    for name, content in (("readiness.txt", report), ("student_identity.txt", identity)):
        path = root / "output" / name
        if isinstance(content, str):
            path.write_bytes(content.encode("utf-8"))
        elif isinstance(content, bytes):
            path.write_bytes(content)


def complete(root: Path) -> None:
    practice(root)
    outputs(root, EXPECTED_READINESS, ROSTER_HASH + "\n")


def family(version: str) -> str:
    return EXPECTED_READINESS.replace(FIRST_LINE, f"Python family: {version}\n", 1)


# --------------------------------------------------------------------------
# Artifact variants for the equivalence run
# --------------------------------------------------------------------------


def _readiness_variants() -> dict[str, Callable[[Path], None]]:
    report = EXPECTED_READINESS
    texts: dict[str, str | bytes] = {
        "expected": report,
        "wrong total": report.replace("Total: 82", "Total: 83"),
        "mean written 20.50": report.replace("Mean: 20.5", "Mean: 20.50"),
        "label in lower case": report.replace("Project:", "project:"),
        "trailing space": report.replace("Total: 82\n", "Total: 82 \n"),
        "tab after colon": report.replace("Total: 82", "Total:\t82"),
        "no-break space": report.replace("Total: 82", "Total:\u00a082"),
        "no space after a colon": report.replace("Measurement: 18", "Measurement:18"),
        "space before a colon": report.replace("Count: 4", "Count : 4"),
        "no space after any Task 2 colon": task2_unspaced(report),
        "leading blank line": "\n" + report,
        "blank line inside": report.replace("Count: 4\n", "\nCount: 4\n"),
        "blank lines between the scripts": report.replace("Measurement: 18", "\nMeasurement: 18", 1)
        .replace("Readiness:", "\n\nReadiness:", 1),
        "leftover debug line": report.replace("Measurement: 18 within range\n",
                                              "Measurement: 18 within range\ntotal so far: 18\n"),
        "extra final line": report + "extra\n",
        "line missing": report.replace("Review count: 2\n", ""),
        "lines reordered": report.replace("Count: 4\nTotal: 82\n", "Total: 82\nCount: 4\n"),
        "CRLF line endings": report.replace("\n", "\r\n"),
        "CR line endings": report.replace("\n", "\r"),
        "CRLF on the first line only": report.replace("\n", "\r\n", 1),
        "no final newline": report[:-1],
        "two final newlines": report + "\n",
        "form feed separator": report.replace("Count: 4\n", "Count: 4\x0c"),
        "line separator": report.replace("Count: 4\n", "Count: 4\u2028"),
        "byte-order mark": "\ufeff" + report,
        "not UTF-8": report.encode("utf-8") + "caf\xe9\n".encode("latin-1"),
        "UTF-16": report.encode("utf-16"),
        "empty": "",
        "newline only": "\n",
        "first line only": FIRST_LINE,
        "Python version written out": report.replace(FIRST_LINE, "Python 3.13.7\n", 1),
        "Python version unlabelled, a line missing": report.replace(FIRST_LINE, "3.13\n", 1)
        .replace("Next checkpoint: 5\n", ""),
        "Python version unlabelled, a blank line inside": report.replace(FIRST_LINE, "3.13\n", 1)
        .replace("Count: 4\n", "\nCount: 4\n"),
        "family label in lower case": report.replace("Python family", "python family", 1),
        "double spaces after Task 2 colons": task2_double_spaced(report),
        "running total compared, double spaced": RUNNING_TOTAL,
        "leading and trailing spaces": "".join(f"  {line} \t\n" for line in report.splitlines()),
        "a Task 2 line wrong": report.replace("Measurement: 24 review", "Measurement: 24 within range"),
        "a Task 3.1 line wrong": report.replace("Next checkpoint: 5", "Next checkpoint: 41"),
    }
    for version in ("3.14", "3.12", "3.9", "3.10", "10.20", "0.0", "\u0663.\u0661\u0663", "3", "three",
                    "3.14.4", "", " 3.13", "3.13 ", "3.13\t", "v3.13", "3,13", "3.13\r"):
        texts[f"Python family {version!r}"] = family(version)
    for version in ("3.14", "3.9", "3"):
        texts[f"Python family {version!r} with a wrong total"] = family(version).replace("Total: 82", "Total: 83")
        texts[f"Python family {version!r} with CRLF"] = family(version).replace("\n", "\r\n")

    variants: dict[str, Callable[[Path], None]] = {
        name: (lambda root, text=text: outputs(root, text, None)) for name, text in texts.items()
    }

    def missing(root: Path) -> None:
        outputs(root, None, None)

    def directory(root: Path) -> None:
        (root / "output" / "readiness.txt").mkdir(parents=True)

    def symlink(root: Path) -> None:
        outputs(root, None, None)
        (root / "output" / "real-report.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
        (root / "output" / "readiness.txt").symlink_to("real-report.txt")

    def dangling(root: Path) -> None:
        outputs(root, None, None)
        (root / "output" / "readiness.txt").symlink_to("nowhere.txt")

    def fifo(root: Path) -> None:
        outputs(root, None, None)
        os.mkfifo(root / "output" / "readiness.txt")

    def unreadable(root: Path) -> None:
        outputs(root, EXPECTED_READINESS, None)
        (root / "output" / "readiness.txt").chmod(0)

    variants |= {"missing": missing, "a directory": directory, "symlink to the expected report": symlink,
                 "dangling symlink": dangling, "named pipe": fifo}
    if os.geteuid() != 0:  # root reads a file whatever its permissions
        variants["unreadable"] = unreadable
    return variants


def _identity_variants() -> dict[str, Callable[[Path], None]]:
    outsider = hashlib.sha256(b"teststudent").hexdigest()
    assert outsider not in ROSTER_HASHES
    hashes: dict[str, str | bytes] = {
        "roster hash": ROSTER_HASH + "\n",
        "no final newline": ROSTER_HASH,
        "upper case": ROSTER_HASH.upper() + "\n",
        "surrounding tabs and spaces": "\t " + ROSTER_HASH.upper() + " \n",
        "CRLF": ROSTER_HASH + "\r\n",
        "no-break spaces around": "\u00a0" + ROSTER_HASH + "\u00a0\n",
        "byte-order mark": "\ufeff" + ROSTER_HASH + "\n",
        "zeros": "0" * 64 + "\n",
        "hash of an address off the roster": outsider + "\n",
        "63 characters": ROSTER_HASH[:-1] + "\n",
        "65 characters": ROSTER_HASH + "0\n",
        "two hashes": ROSTER_HASH + "\n" + ROSTER_HASH,
        "space inside": ROSTER_HASH[:32] + " " + ROSTER_HASH[32:] + "\n",
        "0x prefix": "0x" + ROSTER_HASH + "\n",
        "labelled": "sha256: " + ROSTER_HASH + "\n",
        "full-width digits": ROSTER_HASH.translate({ord(c): ord(c) + 0xFEE0 for c in "0123456789"}) + "\n",
        "not a hash": "not-a-hash\n",
        "empty": "",
        "newline only": "\n",
        "not UTF-8": ROSTER_HASH.encode() + b"\xff\n",
        "UTF-16": (ROSTER_HASH + "\n").encode("utf-16"),
    }
    variants: dict[str, Callable[[Path], None]] = {
        name: (lambda root, text=text: outputs(root, None, text)) for name, text in hashes.items()
    }

    def missing(root: Path) -> None:
        outputs(root, None, None)

    def directory(root: Path) -> None:
        (root / "output" / "student_identity.txt").mkdir(parents=True)

    def symlink(root: Path) -> None:
        outputs(root, None, None)
        (root / "output" / "real-identity.txt").write_text(ROSTER_HASH + "\n", encoding="utf-8")
        (root / "output" / "student_identity.txt").symlink_to("real-identity.txt")

    variants |= {"missing": missing, "a directory": directory, "symlink to a roster hash": symlink}
    return variants


def _practice_variants() -> dict[str, Callable[[Path], None]]:
    def files(*names: str, content: str = "") -> Callable[[Path], None]:
        def build(root: Path) -> None:
            (root / "terminal-practice").mkdir()
            for name in names:
                (root / "terminal-practice" / name).write_text(content, encoding="utf-8")
        return build

    def linked_folder(root: Path) -> None:
        practice(root / "elsewhere")
        (root / "terminal-practice").symlink_to(Path("elsewhere") / "terminal-practice")

    def linked_file(root: Path) -> None:
        files("path-check.txt", "real.txt")(root)
        (root / "terminal-practice" / "source.txt").symlink_to("real.txt")

    def dangling(root: Path) -> None:
        files("path-check.txt")(root)
        (root / "terminal-practice" / "source.txt").symlink_to("nowhere.txt")

    def folder_for_file(root: Path) -> None:
        files("source.txt")(root)
        (root / "terminal-practice" / "path-check.txt").mkdir()

    def file_for_folder(root: Path) -> None:
        (root / "terminal-practice").write_text("", encoding="utf-8")

    return {
        "both files": files("source.txt", "path-check.txt"),
        "no folder": lambda root: None,
        "empty folder": files(),
        "source.txt only": files("source.txt"),
        "path-check.txt only": files("path-check.txt"),
        "files with contents": files("source.txt", "path-check.txt", content="notes\n"),
        "remove-me.txt left behind": files("source.txt", "path-check.txt", "remove-me.txt"),
        "misnamed path_check.txt": files("source.txt", "path_check.txt"),
        "capitalized Source.txt": files("Source.txt", "path-check.txt"),
        "symlinked folder": linked_folder,
        "symlinked file": linked_file,
        "dangling symlink": dangling,
        "folder where a file belongs": folder_for_file,
        "file where the folder belongs": file_for_folder,
    }


def _output_folder_variants() -> dict[str, Callable[[Path], None]]:
    def as_file(root: Path) -> None:
        (root / "output").write_text("", encoding="utf-8")

    def as_symlink(root: Path) -> None:
        outputs(root / "elsewhere", EXPECTED_READINESS, ROSTER_HASH + "\n")
        (root / "output").symlink_to(Path("elsewhere") / "output")

    def empty(root: Path) -> None:
        (root / "output").mkdir()

    def gitkeep_only(root: Path) -> None:
        (root / "output").mkdir()
        (root / "output" / ".gitkeep").write_text("", encoding="utf-8")

    return {"no output folder": lambda root: None, "output is a file": as_file,
            "output is a symlink": as_symlink, "empty output folder": empty, "only .gitkeep": gitkeep_only}


def artifact_variants() -> dict[str, Callable[[Path], None]]:
    """Every submission the comparisons grade, keyed by a readable description."""
    readiness, identity, practices = _readiness_variants(), _identity_variants(), _practice_variants()
    cases: dict[str, Callable[[Path], None]] = {"starter": lambda root: None, "complete": complete}

    def combine(*steps: Callable[[Path], None]) -> Callable[[Path], None]:
        def build(root: Path) -> None:
            for step in steps:
                step(root)
        return build

    for report_name, report in readiness.items():
        for identity_name, saved in identity.items():
            cases[f"report {report_name} + identity {identity_name}"] = combine(practice, report, saved)
    for hash_value in sorted(ROSTER_HASHES):
        for report_name in ("expected", "Python family '3.14'", "wrong total", "CRLF line endings"):
            cases[f"report {report_name} + roster hash {hash_value[:8]}"] = combine(
                practice, readiness[report_name], lambda root, h=hash_value: outputs(root, None, h + "\n"))
    for practice_name, build in practices.items():
        cases[f"practice {practice_name} + complete outputs"] = combine(
            build, lambda root: outputs(root, EXPECTED_READINESS, ROSTER_HASH + "\n"))
        cases[f"practice {practice_name} + no outputs"] = build
    for folder_name, build in _output_folder_variants().items():
        cases[f"{folder_name} + practice files"] = combine(practice, build)
    cases["extra files everywhere"] = combine(
        complete, lambda root: (root / "notes.txt").write_text("extra\n", encoding="utf-8"),
        lambda root: (root / "output" / "extra.txt").write_text("extra\n", encoding="utf-8"))
    return cases


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------


def graded(root: Path) -> dict[str, dict]:
    result = grade_submission(root)
    assert result["schema"] == "datasci217/grading-result/v1", result["schema"]
    assert result["max-score"] == sum(POINTS) == 100, result["max-score"]
    assert [test["test-name"] for test in result["tests"]] == TEST_NAMES, result["tests"]
    assert result["score"] == sum(test["score"] for test in result["tests"]), result
    return {test["test-name"]: test for test in result["tests"]}


def score(root: Path) -> int:
    return sum(test["score"] for test in graded(root).values())


def failing(root: Path) -> dict[str, str]:
    """Each failing check's detail; a failing check always says why."""
    failed = {name: test["detail"] for name, test in graded(root).items() if not test["passed"]}
    assert all(failed.values()), failed
    return failed


def line_detail(expected: str, yours: str, script: str) -> str:
    return f"The line should read `{expected}`; yours reads `{yours}`. Fix {script}, then rerun make_output.py."


def missing_detail(expected: str, script: str) -> str:
    return (f"output/readiness.txt is missing the line `{expected}`, or has it out of order. Fix {script}, "
            "then rerun make_output.py.")


def run() -> None:
    """The checks score each documented case as the README's contract states."""
    process_email = load("process_email", ASSIGNMENT / "process_email.py").process_email
    expected_identity = hashlib.sha256(b"alicesmith").hexdigest()
    for address in ("Alice.Smith@ucsf.edu", " alice-smith@ucsf.edu ", "\tALICE.SMITH@UCSF.EDU\n", "alice_smith@ucsf.edu"):
        assert process_email(address)["hash"] == expected_identity, address
    for address in ("alice.smith@example.com", "alice.smith@ucsf.edu.example.com", "alice.smith@notucsf.edu",
                    "alice.smith@sub.ucsf.edu", "alice.smith", "alice@@ucsf.edu", "@ucsf.edu", "...@ucsf.edu",
                    "alice smith@ucsf.edu", "alice@ ucsf.edu"):
        try:
            process_email(address)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid roster email was accepted: {address!r}")
    assert ROSTER_HASH in ROSTER_HASHES and len(ROSTER_HASHES) >= 40  # 40 at handout; students may be added
    assert all(value.IDENTITY_HASH.fullmatch(identity) for identity in ROSTER_HASHES)
    assert len(EXPECTED_READINESS.splitlines()) == 14 and EXPECTED_READINESS.endswith("\n")

    # One check per practice file, per graded report line, and for the identity, scored 10, 5, and 15.
    assert [check.name for check in VALUE_CHECKS] == TEST_NAMES
    assert dict(zip(TEST_NAMES, POINTS, strict=True)) == (
        dict.fromkeys(PRACTICE_TESTS, 10) | dict.fromkeys(LINE_TESTS, 5) | {IDENTITY_TEST: 15})
    assert sum(POINTS) == 100

    with scratch("a01-selftest-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()
        output = root / "output"

        # An untouched handout scores 0, and every check says why.
        assert score(root) == 0 and set(failing(root)) == set(TEST_NAMES)
        practice(root)
        assert score(root) == 20

        # The identity check stands apart from the report: a roster hash with no report earns its 15,
        # in any case and with any surrounding whitespace, for every student on the roster.
        outputs(root, None, None)
        for identity_hash in ROSTER_HASHES:
            for text in (identity_hash, identity_hash + "\n", "\t " + identity_hash.upper() + " \n"):
                (output / "student_identity.txt").write_text(text, encoding="utf-8")
                assert set(failing(root)) == set(LINE_TESTS), text
                assert score(root) == 35
        # A missing report fails every report line with the same advice, which says how to make it.
        assert set(failing(root).values()) == {MISSING_REPORT}
        (output / "readiness.txt").mkdir()
        assert set(failing(root).values()) == {MISSING_REPORT}
        (output / "readiness.txt").rmdir()
        (output / "readiness.txt").write_bytes(EXPECTED_READINESS.encode("utf-16"))
        assert set(failing(root).values()) == {"output/readiness.txt must be UTF-8 text."} and score(root) == 35
        (output / "readiness.txt").write_bytes(EXPECTED_READINESS.encode() + b"caf\xe9\n")
        assert set(failing(root).values()) == {"output/readiness.txt must be UTF-8 text."} and score(root) == 35

        # A hash off the roster, or not a hash at all, costs the identity's 15 and nothing else.
        (output / "readiness.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
        (root / "notes.txt").write_text("extra files are allowed\n", encoding="utf-8")
        (output / "student_identity.txt").write_text(ROSTER_HASH + "\n", encoding="utf-8")
        assert score(root) == 100 and failing(root) == {}
        for text, advice in (("0" * 64 + "\n", "course roster"),
                             (hashlib.sha256(b"teststudent").hexdigest() + "\n", "course roster"),
                             ("not-a-hash\n", "one SHA-256 hash"),
                             (ROSTER_HASH + "\n" + ROSTER_HASH, "one SHA-256 hash")):
            (output / "student_identity.txt").write_text(text, encoding="utf-8")
            failed = failing(root)
            assert list(failed) == [IDENTITY_TEST] and advice in failed[IDENTITY_TEST], failed
            assert score(root) == 85
        (output / "student_identity.txt").unlink()
        assert failing(root) == {IDENTITY_TEST: "Run capture_identity.py and commit output/student_identity.txt."}
        (output / "student_identity.txt").write_text(ROSTER_HASH + "\n", encoding="utf-8")

        def report_scores(text: str) -> int:
            (output / "readiness.txt").write_text(text, encoding="utf-8")
            return score(root)

        # The first line records the Python version and is never graded: any text there, or none at all,
        # earns full marks; every other line is still graded.
        for text in [family(version) for version in ("3.13", "3.14", "3.12", "3.9", "3", "three", "3.14.4", "")] + [
                EXPECTED_READINESS.replace(FIRST_LINE, "Python 3.13.7\n", 1),
                EXPECTED_READINESS.replace(FIRST_LINE, "3.13\n", 1),
                EXPECTED_READINESS.replace(FIRST_LINE, "", 1),
                EXPECTED_READINESS.replace(FIRST_LINE, "anything at all\n", 1),
                family("3.14") + "an extra final line\n"]:
            assert report_scores(text) == 100, text[:30]
        # Whatever the first line says, a missing line costs only itself.
        for first_line in ("Python family: 3.12\n", "3.13\n", ""):
            assert report_scores(EXPECTED_READINESS.replace(FIRST_LINE, first_line, 1)
                                 .replace("Next checkpoint: 5\n", "")) == 95, first_line
            assert list(failing(root)) == ["report: Next checkpoint"], first_line

        # Whitespace is never graded: not inside a line, not around it, not blank lines, not line endings.
        for text in (task2_double_spaced(EXPECTED_READINESS),
                     task2_unspaced(EXPECTED_READINESS),
                     EXPECTED_READINESS.replace(": ", ":\t"),
                     EXPECTED_READINESS.replace("Measurement: 18", "Measurement:18"),
                     EXPECTED_READINESS.replace("Count: 4", "Count : 4"),
                     EXPECTED_READINESS.replace("Count: 4", "Count:4"),
                     EXPECTED_READINESS.replace("Total: 82", "Total:\u00a082"),
                     "".join(f"  {line} \t\n" for line in EXPECTED_READINESS.splitlines()),
                     EXPECTED_READINESS.replace("within range", "within    range"),
                     EXPECTED_READINESS.replace("\n", "\r\n"),
                     EXPECTED_READINESS[:-1],
                     EXPECTED_READINESS + "\n",
                     "\n" + EXPECTED_READINESS,
                     EXPECTED_READINESS.replace("Count: 4\n", "\nCount: 4\n"),
                     EXPECTED_READINESS.replace("Measurement: 18", "\nMeasurement: 18", 1)
                     .replace("Readiness:", "\n\nReadiness:", 1),
                     EXPECTED_READINESS.replace(FIRST_LINE, "3.13\n", 1).replace("Count: 4\n", "\nCount: 4\n")):
            assert report_scores(text) == 100, text
        # Characters are graded: a letter's case counts, and the message shows the line with that label.
        assert report_scores(EXPECTED_READINESS.replace("Project:", "project:")) == 95
        assert failing(root) == {"report: Project": line_detail("Project: DataSci 217 Assignment 01",
                                                                "project: DataSci 217 Assignment 01", "readiness.py")}

        # One wrong value costs only its own line, and says what the line should read.
        assert report_scores(EXPECTED_READINESS.replace("Total: 82", "Total: 83")) == 95
        assert failing(root) == {"report: Total": line_detail("Total: 82", "Total: 83", "measurement_summary.py")}
        assert report_scores(EXPECTED_READINESS.replace("Script: readiness.py", "Script: TODO")) == 95
        assert failing(root) == {"report: Script": line_detail("Script: readiness.py", "Script: TODO", "readiness.py")}

        # The real submission that compared the running total, double spaced: two lines wrong, 90 points.
        assert report_scores(RUNNING_TOTAL) == 90
        assert failing(root) == {
            "report: Measurement 4": line_detail("Measurement: 19 within range", "Measurement: 19 review",
                                                 "measurement_summary.py"),
            "report: Review count": line_detail("Review count: 2", "Review count: 3", "measurement_summary.py"),
        }

        # A missing line costs only itself, and an extra line, such as a leftover debug print, costs nothing:
        # the lines after either one are still matched.
        assert report_scores(EXPECTED_READINESS.replace("Review count: 2\n", "")) == 95
        assert failing(root) == {"report: Review count": missing_detail("Review count: 2", "measurement_summary.py")}
        assert report_scores(EXPECTED_READINESS.replace("Count: 4\n", "Count: 4\ntotal so far: 18\n")) == 100
        assert report_scores(EXPECTED_READINESS.replace("Total: 82", "Total: 83").replace(
            "Count: 4\n", "Count: 4\ndebug\n")) == 95
        assert failing(root) == {"report: Total": line_detail("Total: 82", "Total: 83", "measurement_summary.py")}
        # Two lines swapped: one of them is out of order, and only it fails.
        assert report_scores(EXPECTED_READINESS.replace("Count: 4\nTotal: 82\n", "Total: 82\nCount: 4\n")) == 95
        assert len(failing(root)) == 1 and "or has it out of order" in next(iter(failing(root).values()))
        # Several wrong lines in a row are each shown beside the line with the same label, in order.
        assert report_scores(EXPECTED_READINESS.replace("21 review", "21 within range")
                             .replace("24 review", "24 within range")) == 90
        assert failing(root) == {
            "report: Measurement 2": line_detail("Measurement: 21 review", "Measurement: 21 within range",
                                                 "measurement_summary.py"),
            "report: Measurement 3": line_detail("Measurement: 24 review", "Measurement: 24 within range",
                                                 "measurement_summary.py"),
        }
        # A report with none of the lines fails each with the same kind of message.
        assert report_scores("wrong\n") == 35
        assert failing(root)["report: Project"] == missing_detail("Project: DataSci 217 Assignment 01",
                                                                    "readiness.py")
        assert report_scores("") == 35
        assert failing(root)["report: Next checkpoint"] == missing_detail("Next checkpoint: 5", "debug_report.py")

        # A practice file costs its own 10 points.
        (output / "readiness.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
        (root / "terminal-practice" / "source.txt").unlink()
        assert failing(root) == {"terminal-practice/source.txt": "terminal-practice/source.txt must be a regular "
                                 "file; create it with the Task 1.1 commands."}
        assert score(root) == 90

        # An output folder that is a symlink holds no artifact of this submission.
        practice(root)
        os.rename(output, root / "elsewhere")
        output.symlink_to("elsewhere")
        assert set(failing(root)) == set(LINE_TESTS + [IDENTITY_TEST]) and score(root) == 20
        assert set(failing(root).values()) == {"Create a regular output/ directory."}

    print("Assignment 01 checks: every roster hash, identity and report failures on their own, the running-total "
          "submission at 90, whitespace, blank and extra lines, and the Python version line ungraded, a missing "
          "line costing only itself, and each failing line saying what it should read, all score as the contract "
          "states.")


def checks_files() -> list[str]:
    """The files the workflow downloads from the course, in its own words."""
    workflow = (ASSIGNMENT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    listed = re.search(r"^  CHECKS_FILES: \|\n((?:    \S.*\n)+)", workflow, re.M)
    assert listed, "tests.yml lists no CHECKS_FILES"
    assert '  CHECKS_PATH: "01/assignment_checks"\n' in workflow
    assert "shape" not in workflow.lower(), "the workflow still describes shape-only checks"
    return listed.group(1).split()


def checker(copy: Path, root: Path, *options: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", str(copy / "check_assignment.py"), str(root), *options],
                          capture_output=True, text=True, check=False)


def run_handout() -> None:
    """The handout carries the course's checks unchanged, so a local run gives what GitHub gives."""
    files = checks_files()
    course_owned = {path.relative_to(CHECKS).as_posix() for path in CHECKS.rglob("*")
                    if path.is_file() and not {"__pycache__", ".pytest_cache", "_grader_selftest"} & set(path.parts)}
    assert sorted(files) == sorted(course_owned - {"README.md"}), (files, course_owned)
    for name in files:
        assert (ASSIGNMENT / name).read_bytes() == (CHECKS / name).read_bytes(), f"01/assignment/{name} differs"
    assert not (ASSIGNMENT / "_shape_checks.py").exists(), "the shape-only checks are back in the handout"

    # The README shows every expected line and agrees with POINTS, row by row.
    readme = (ASSIGNMENT / "README.md").read_text(encoding="utf-8")
    for line in EXPECTED_READINESS.splitlines():
        assert f"\n{line}\n" in readme, line
    contract = readme.partition("### Completion contract\n")[2].partition("\n## ")[0]
    assert f"Grading totals {sum(POINTS)} points" in contract
    rows = re.findall(r"^\| `.+ \| (\d+)(?: \((\d+) each\))? \|$", contract, re.M)
    points = dict(zip(TEST_NAMES, POINTS, strict=True))
    expected_rows = []
    for group in (PRACTICE_TESTS, LINE_TESTS, [IDENTITY_TEST]):
        each = {points[name] for name in group}
        assert len(each) == 1, group
        expected_rows.append((str(sum(points[name] for name in group)), str(*each) if len(group) > 1 else ""))
    assert rows == expected_rows, rows

    with scratch("a01-handout-") as temporary:
        submissions = {name: Path(temporary) / name for name in ("empty", "running total", "complete")}
        for root in submissions.values():
            root.mkdir()
        practice(submissions["running total"])
        outputs(submissions["running total"], RUNNING_TOTAL, "0" * 64 + "\n")
        practice(submissions["complete"])
        outputs(submissions["complete"], family("3.14"), ROSTER_HASH + "\n")

        # Both copies report the same JSON, and the handout's needs nothing from the course directory.
        for name, root in submissions.items():
            local, course = checker(ASSIGNMENT, root, "--json"), checker(CHECKS, root, "--json")
            assert local.stdout == course.stdout and local.returncode == course.returncode, (name, local, course)
            assert json.loads(local.stdout) == grade_submission(root), name
        shown = checker(ASSIGNMENT, submissions["running total"])
        assert shown.returncode == 1 and "Score: 75/100\n" in shown.stdout, shown.stdout
        assert "yours reads `Measurement: 19 review`" in shown.stdout, shown.stdout

        # A clean local run ends the way the README promises.
        shown = checker(ASSIGNMENT, submissions["complete"])
        assert shown.returncode == 0, shown.stdout + shown.stderr
        promised = re.search(r"A clean local run ends with:\n\n```text\n(.*?)```", readme, re.S)
        assert promised and shown.stdout.endswith(promised.group(1)), (shown.stdout, promised)

    print(f"Assignment 01 handout: {len(files)} check files byte-identical to 01/assignment_checks/, no shape "
          "checks left, the README's contract matches POINTS, and the local checker reports what GitHub reports "
          "and ends as the README shows.")


def with_graded_first_line(root: Path) -> None:
    """Give the report the first line the replaced checker required, so only the ungraded line changes."""
    report = root / "output" / "readiness.txt"
    if not report.is_file() or report.is_symlink():
        return
    try:
        text = report.read_text(encoding="utf-8")  # read as the checks read it, newlines normalized
    except (OSError, UnicodeDecodeError):
        return
    graded_lines = EXPECTED_READINESS.partition("\n")[2]
    report.write_bytes((FIRST_LINE + (text if text == graded_lines else text.partition("\n")[2])).encode("utf-8"))


def run_equivalence() -> None:
    """No artifact variant scores lower than under the checker these replaced, whatever its first line."""
    with scratch("a01-replaced-") as temporary:
        replaced = Path(temporary) / "replaced"
        replaced.mkdir()
        for name in REPLACED_FILES:
            shown = subprocess.run(["git", "show", f"{REPLACED_COMMIT}:01/assignment/{name}"], cwd=REPO,
                                   capture_output=True, text=True, check=False)
            if shown.returncode != 0:
                raise SystemExit(f"The equivalence run needs commit {REPLACED_COMMIT[:7]} of the course repository "
                                 f"(a full clone, not a shallow one): {shown.stderr.strip()}")
            (replaced / name).write_text(shown.stdout, encoding="utf-8")
        before = load("_assignment_checks", replaced / "_assignment_checks.py")
        before_grading = load("_replaced_grading", replaced / "grading.py")
        del sys.modules["_assignment_checks"]
        assert before.ROSTER_HASHES <= ROSTER_HASHES  # the roster only grows
        before.ROSTER_HASHES = ROSTER_HASHES  # compare the checking logic, with students added since
        assert before.EXPECTED_READINESS == EXPECTED_READINESS
        assert before_grading.POINTS == (20, 80)

        cases = artifact_variants()
        assert all(f"report {report} + identity roster hash" in cases for report in FULL_MARKS_REPORTS)
        outcomes = {"full": 0, "partial": 0, "zero": 0}
        raised = 0
        for index, (name, build) in enumerate(cases.items()):
            root = Path(temporary) / f"case-{index}"
            root.mkdir()
            build(root)
            now, unchanged = graded(root), before_grading.grade_submission(root)
            with_graded_first_line(root)
            then = before_grading.grade_submission(root)
            new_score = sum(test["score"] for test in now.values())
            # Nobody scores lower, and the first line never matters: each variant scores at least what the
            # replaced checker gives it as submitted and once its first line reads the way that checker required.
            assert new_score >= unchanged["score"], (name, new_score, unchanged)
            assert new_score >= then["score"], (name, new_score, then)
            raised += new_score > unchanged["score"]
            # Whatever the replaced checker passed, the matching new checks pass.
            earlier = {test["test-name"]: test["passed"] for test in then["tests"]}
            if earlier["terminal practice evidence"]:
                assert all(now[test]["passed"] for test in PRACTICE_TESTS), name
            if earlier["committed readiness and identity artifacts"]:
                assert all(now[test]["passed"] for test in LINE_TESTS + [IDENTITY_TEST]), name
            if name in ("complete", "extra files everywhere") or name.startswith("report expected + roster hash"):
                assert new_score == 100, (name, now)
            if name == "report running total compared, double spaced + identity roster hash":
                assert new_score == 90, (name, now)
            if name in (f"report {report} + identity roster hash" for report in FULL_MARKS_REPORTS):
                assert new_score == 100, (name, now)
            if name.endswith("unreadable + identity roster hash"):
                assert "Permission denied" in now["report: Project"]["detail"], now
            outcomes["full" if new_score == 100 else "zero" if new_score == 0 else "partial"] += 1
        assert all(outcomes.values()), outcomes

    print(f"Assignment 01 equivalence: {len(cases)} artifact variants ({outcomes['full']} full, "
          f"{outcomes['partial']} partial, {outcomes['zero']} zero) each score at least what the checker from "
          f"{REPLACED_COMMIT[:7]} gives them, with or without the ungraded Python line set aside; {raised} "
          "score higher, and every one it passed passes the matching checks here.")


if __name__ == "__main__":
    run()
    run_handout()
    run_equivalence()
