"""Regression checks for the Assignment 01 checks.

Builds submissions in ignored `scratch/` and confirms that the value checks
score what they should, that the shape checks shipped in the fork agree with
them without knowing any answer, and that the value checks grade every artifact
variant exactly as the vendored checker they replaced. Nothing here reads or
runs student code.
"""

from collections.abc import Callable
import hashlib
import importlib.util
import inspect
import os
from pathlib import Path
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

# The last commit whose fork carried the value checks itself, as
# 01/assignment/_assignment_checks.py; the value checks must grade exactly as it did.
REPLACED_COMMIT = "f39598b76433ae1e626f574eff684c8d48561380"
REPLACED_FILES = ("_assignment_checks.py", "grading.py")

ROSTER_HASH = sorted(ROSTER_HASHES)[0]
# The checker files that ship in the fork; the scaffolds legitimately print report lines.
CHECKER_FILES = {"_shape_checks.py", "grading.py", "check_assignment.py", "test_assignment.py"}
FIRST_LINE = EXPECTED_READINESS.partition("\n")[0] + "\n"


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
# Artifact variants, shared by the equivalence and shape runs
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
        "leading blank line": "\n" + report,
        "blank line inside": report.replace("Count: 4\n", "\nCount: 4\n"),
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
        "family label in lower case": report.replace("Python family", "python family", 1),
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


def scores(root: Path) -> dict[str, int]:
    result = grade_submission(root)
    assert result["schema"] == "datasci217/grading-result/v1", result["schema"]
    assert result["max-score"] == sum(POINTS) == 100, result["max-score"]
    return {test["test-name"]: test["score"] for test in result["tests"]}


def run() -> None:
    """The value checks score each documented case as the README's contract states."""
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
    assert len(ROSTER_HASHES) == 40
    assert all(value.IDENTITY_HASH.fullmatch(identity) for identity in ROSTER_HASHES)
    assert len(EXPECTED_READINESS.splitlines()) == 14 and EXPECTED_READINESS.endswith("\n")

    with scratch("a01-selftest-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()
        assert grade_submission(root)["score"] == 0
        assert len(grade_submission(root)["tests"]) == len(POINTS)
        practice(root)
        output = root / "output"
        outputs(root, EXPECTED_READINESS, None)
        assert grade_submission(root)["score"] == 20
        for identity_hash in ROSTER_HASHES:
            for text in (identity_hash, identity_hash + "\n", "\t " + identity_hash.upper() + " \n"):
                (output / "student_identity.txt").write_text(text, encoding="utf-8")
                assert scores(root) == {"terminal practice evidence": 20,
                                        "committed readiness and identity artifacts": 80}
        (output / "student_identity.txt").write_text("0" * 64 + "\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 20
        (output / "student_identity.txt").write_text(ROSTER_HASH + "\n", encoding="utf-8")
        (root / "notes.txt").write_text("extra files are allowed\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 100
        (output / "readiness.txt").write_text("wrong\n", encoding="utf-8")
        assert grade_submission(root)["score"] == 20
        (output / "readiness.txt").unlink()
        assert grade_submission(root)["score"] == 20
        (output / "readiness.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
        for text in ("not-a-hash\n", ROSTER_HASH + "\n" + ROSTER_HASH):
            (output / "student_identity.txt").write_text(text, encoding="utf-8")
            assert grade_submission(root)["score"] == 20, text
        (output / "student_identity.txt").write_text(ROSTER_HASH + "\n", encoding="utf-8")
        # The first line records whichever Python ran the script, and any version earns the points;
        # every other line, and the shape of that one, is still graded.
        for version, score in (("3.13", 100), ("3.14", 100), ("3.12", 100), ("3.9", 100),
                               ("3", 20), ("three", 20), ("3.14.4", 20), ("", 20)):
            (output / "readiness.txt").write_text(family(version), encoding="utf-8")
            assert grade_submission(root)["score"] == score, version
        (output / "readiness.txt").write_text(EXPECTED_READINESS.replace("Total: 82", "Total: 83"), encoding="utf-8")
        assert grade_submission(root)["score"] == 20
        (output / "readiness.txt").write_text(EXPECTED_READINESS, encoding="utf-8")
        (root / "terminal-practice" / "source.txt").unlink()
        assert scores(root) == {"terminal practice evidence": 0, "committed readiness and identity artifacts": 80}

    print("Assignment 01 value checks: all 40 roster hashes, non-roster hashes, starter, extra-file, missing, "
          "wrong-artifact, and any-Python-version cases score as the contract states.")


def run_equivalence() -> None:
    """The value checks grade every variant exactly as the checker they replaced."""
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
        assert before.ROSTER_HASHES == ROSTER_HASHES
        assert before.EXPECTED_READINESS == EXPECTED_READINESS
        assert before_grading.POINTS == POINTS

        shape = load_shape()
        cases = artifact_variants()
        outcomes = {"full": 0, "partial": 0, "zero": 0}
        for index, (name, build) in enumerate(cases.items()):
            root = Path(temporary) / f"case-{index}"
            root.mkdir()
            build(root)
            now, then = grade_submission(root), before_grading.grade_submission(root)
            assert now == then, (name, now, then)
            outcomes["full" if now["score"] == 100 else "zero" if now["score"] == 0 else "partial"] += 1
            # A submission the value checks accept always passes the local shape checks too.
            for (check, detail), test in zip(shape.run_checks(root), now["tests"], strict=True):
                assert check == test["test-name"], (check, test["test-name"])
                assert detail is None or not test["passed"], (name, check, detail)
            if name.endswith("unreadable + identity roster hash"):
                assert "Permission denied" in now["tests"][1]["detail"], now
        assert all(outcomes.values()), outcomes

    print(f"Assignment 01 equivalence: {len(cases)} artifact variants ({outcomes['full']} full, "
          f"{outcomes['partial']} partial, {outcomes['zero']} zero) grade identically, detail for detail, "
          f"under the value checks and the checker from {REPLACED_COMMIT[:7]}; every one the value checks "
          "accept passes the shape checks.")


def load_shape():
    return load("_shape_checks", ASSIGNMENT / "_shape_checks.py")


def run_shape() -> None:
    """The checks in the student fork judge shape, and cannot judge an answer."""
    shape = load_shape()
    shape_grading_source = (ASSIGNMENT / "grading.py").read_text(encoding="utf-8")
    assert "from _shape_checks import run_checks" in shape_grading_source
    shape_grading = load("_shape_grading", ASSIGNMENT / "grading.py")
    assert shape_grading.POINTS == POINTS

    # No answer ships in the fork: no roster hash anywhere, and no report line beyond the documented
    # first-line form in any file but the README, which shows each script's output as instructions.
    for path in sorted(ASSIGNMENT.rglob("*")):
        if not path.is_file() or {"__pycache__", ".pytest_cache"} & set(path.parts):
            continue
        content = path.read_bytes()
        leaked = [identity for identity in ROSTER_HASHES if identity.encode() in content.lower()]
        assert not leaked, (path, len(leaked))
        assert EXPECTED_READINESS.encode() not in content, path
        for marker in (b"EXPECTED_READINESS", b"ROSTER_HASHES"):
            assert marker not in content, (path, marker)
        if path.name in CHECKER_FILES:
            text = content.decode("utf-8")
            for line in EXPECTED_READINESS.splitlines()[1:]:
                assert line not in text, (path, line)

    # Both halves read the artifacts with the same code, so they never disagree about a format.
    for name in ("PRACTICE_DIR", "PRACTICE_FILES", "OUTPUT_DIR", "READINESS_FILE", "IDENTITY_FILE",
                 "PYTHON_FAMILY", "IDENTITY_HASH"):
        assert getattr(shape, name) == getattr(value, name), name  # compiled regexes compare pattern and flags
    for name in ("Check", "_assert", "_read_text", "_after_python_family", "_readiness_report", "_identity_hash",
                 "check_terminal_practice", "run_checks"):
        assert inspect.getsource(getattr(shape, name)) == inspect.getsource(getattr(value, name)), name
    assert shape.READINESS_LINES == len(EXPECTED_READINESS.splitlines())
    assert [check.name for check in shape.CHECKS] == [check.name for check in VALUE_CHECKS]
    for shared in ("test_assignment.py", ".github/test/test_assignment.py", ".github/test/requirements.txt"):
        assert (ASSIGNMENT / shared).read_bytes() == (CHECKS / shared).read_bytes(), shared

    with scratch("a01-shape-") as temporary:
        root = Path(temporary) / "submission"
        root.mkdir()

        # An untouched handout fails every shape check.
        assert all(detail for _, detail in shape.run_checks(root))

        # Right and plausibly wrong are indistinguishable: a wrong line and a hash off the roster pass.
        for report, identity in ((EXPECTED_READINESS, ROSTER_HASH),
                                 (family("3.14").replace("Total: 82", "Total: 83"), "0" * 64),
                                 (EXPECTED_READINESS.replace("\n", "\r\n"), ROSTER_HASH.upper())):
            practice(root)
            outputs(root, report, identity + "\n")
            failures = [detail for _, detail in shape.run_checks(root) if detail]
            assert not failures, failures
            assert shape_grading.grade_submission(root)["score"] == 100

        # Malformed artifacts are still caught, each by its own check.
        for report, identity, failing in (
            (EXPECTED_READINESS.replace("Review count: 2\n", ""), ROSTER_HASH, "(yours has 13)"),
            (EXPECTED_READINESS + "extra\n", ROSTER_HASH, "(yours has 15)"),
            (EXPECTED_READINESS[:-1], ROSTER_HASH, "end with a newline"),
            (EXPECTED_READINESS.replace(FIRST_LINE, "Python 3.13.7\n"), ROSTER_HASH, "Python family: 3.13"),
            ("\ufeff" + EXPECTED_READINESS, ROSTER_HASH, "Python family: 3.13"),
            (EXPECTED_READINESS, "not-a-hash", "one SHA-256 hash"),
            (EXPECTED_READINESS, ROSTER_HASH + "\n" + ROSTER_HASH, "one SHA-256 hash"),
        ):
            outputs(root, report, identity + "\n")
            results = dict(shape.run_checks(root))
            detail = results["committed readiness and identity artifacts"]
            assert detail and failing in detail, (failing, detail)
            assert results["terminal practice evidence"] is None
        outputs(root, EXPECTED_READINESS.encode("utf-16"), ROSTER_HASH)
        assert "UTF-8" in dict(shape.run_checks(root))["committed readiness and identity artifacts"]
        (root / "terminal-practice" / "path-check.txt").unlink()
        assert dict(shape.run_checks(root))["terminal practice evidence"] == "terminal-practice/path-check.txt must be a regular file."

    # The local checker ends the way the README promises.
    with scratch("a01-shape-cli-") as temporary:
        root = Path(temporary)
        practice(root)
        outputs(root, family("3.14"), "0" * 64 + "\n")
        shown = subprocess.run([sys.executable, "-B", str(ASSIGNMENT / "check_assignment.py"), str(root)],
                               capture_output=True, text=True, check=False)
        assert shown.returncode == 0, shown.stdout + shown.stderr
        assert shown.stdout.endswith(f"2 of 2 shape checks passed.\n{shape_grading.REPORT_NOTE}\n"), shown.stdout
        readme = (ASSIGNMENT / "README.md").read_text(encoding="utf-8")
        assert f"```text\n2 of 2 shape checks passed.\n{shape_grading.REPORT_NOTE}\n```" in readme

    print("Assignment 01 shape checks: share the value checks' code, hold no roster hash or report line, "
          "cannot tell a right report or roster hash from a wrong one, and still catch malformed artifacts.")


if __name__ == "__main__":
    run()
    run_shape()
    run_equivalence()
