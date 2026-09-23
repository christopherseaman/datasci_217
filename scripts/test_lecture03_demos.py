"""Run Lecture 03 demos in isolation and check the guide's expected output against real runs."""

from hashlib import sha256
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "03" / "demo"
SCRIPTS = (
    "demo1_cli_pipeline.sh",
    "demo2_python_collections.py",
    "demo2_numpy_performance.py",
    "demo2_numpy_arrays.py",
    "demo3_student_analysis.py",
    "demo3_csv_summary.py",
)
COLLECTIONS = """Python Tools for Collections
==================================================

=== Introspection ===
Original value: 42 Type: <class 'str'>
Converted value: 42 Type: <class 'int'>
Strings have split: True

=== Sequence functions ===
Numbered names:
  Student 1: Alice
  Student 2: Bob
  Student 3: Charlie
Paired records:
  Alice: 85
  Bob: 92
  Charlie: 78
Reverse order: ['Charlie', 'Bob', 'Alice']
Sorted grades: [78, 85, 92]

=== List comprehensions ===
Fevers (100.4 or above): [101.2, 103.1]
Doses in grams: [0.25, 0.5, 0.125]
Curved grades: [90, 97, 83]
Scored 85 or above: ['Alice', 'Bob']
"""


def undate(text):
    """Replace run timestamps so two runs compare equal."""
    return re.sub(r"\d{8}_\d{6}", "TIMESTAMP", text)


def guide_blocks():
    """Every ```text block in DEMO_GUIDE.md; each one quotes real demo output."""
    guide = (DEMOS / "DEMO_GUIDE.md").read_text(encoding="utf-8")
    return re.findall(r"^```text\n(.*?)^```$", guide, re.M | re.S)


def contains(output, block):
    """True when the block's non-empty lines appear in order as exact lines of output."""
    lines = iter(output.splitlines())
    return all(any(line == produced for produced in lines)
               for line in block.splitlines() if line.strip())


def sources():
    """Hash the tracked demo sources so a run cannot change them unnoticed."""
    return {
        path.relative_to(DEMOS).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(DEMOS.rglob("*"))
        if path.is_file() and "__pycache__" not in path.parts and "output" not in path.parts
    }


def run():
    before = sources()
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch", prefix="lecture03-") as temporary:
        demo = Path(temporary)
        for name in (*SCRIPTS, "students.csv"):
            shutil.copy2(DEMOS / name, demo / name)

        def python(*arguments, cwd=demo):
            return subprocess.run(
                [sys.executable, *arguments], cwd=cwd, check=True,
                capture_output=True, text=True,
            ).stdout

        def shell(command, cwd):
            return subprocess.run(
                command, cwd=cwd, check=True, shell=True,
                capture_output=True, text=True,
            ).stdout

        # Every script is import-safe: importing runs no analysis and prints nothing.
        assert python("-c", "import demo2_python_collections, demo2_numpy_performance, "
                            "demo2_numpy_arrays, demo3_student_analysis, demo3_csv_summary") == ""

        # Demo 1 runs from a disposable directory and writes only below it.
        pipeline = demo / "cli-run"
        pipeline.mkdir()
        first = shell(f"bash {demo / 'demo1_cli_pipeline.sh'}", pipeline)
        summary = sorted((pipeline / "results").glob("summary_*.txt"))
        log = (pipeline / "logs" / "processing.log").read_text(encoding="utf-8")
        assert len(summary) == 1, summary
        assert undate(summary[0].read_text(encoding="utf-8")) == (
            "run timestamp: TIMESTAMP\nrecords: 6\nsubject counts:\n"
            "      1 English\n      3 Math\n      2 Science\n"
        )
        assert undate(log).splitlines() == [
            "TIMESTAMP pipeline started",
            "TIMESTAMP wrote results/summary_TIMESTAMP.txt",
        ]

        # A second run keeps the first result and appends to the log.
        time.sleep(1.1)  # the run timestamp has one-second resolution
        assert undate(shell(f"bash {demo / 'demo1_cli_pipeline.sh'}", pipeline)) == undate(first)
        assert len(sorted((pipeline / "results").glob("summary_*.txt"))) == 2
        assert len((pipeline / "logs" / "processing.log").read_text(encoding="utf-8").splitlines()) == 4

        # Demo 2 and Demo 3 print the same text on every run.
        collections = python("demo2_python_collections.py")
        assert collections == COLLECTIONS
        arrays = python("demo2_numpy_arrays.py")
        analysis = python("demo3_student_analysis.py")
        csv_summary = python("demo3_csv_summary.py")
        performance = python("demo2_numpy_performance.py")
        for name, text in (("demo2_numpy_arrays.py", arrays),
                           ("demo3_student_analysis.py", analysis),
                           ("demo3_csv_summary.py", csv_summary)):
            assert python(name) == text, name
        assert "Result sample: [0, 2, 4, 6, 8]" in performance
        assert "Result sample: [0 2 4 6 8]" in performance
        assert analysis.endswith("NumPy analysis complete.\n")

        # The shell checks the guide asks students to compare with the CSV summary.
        previews = shell("head -n 3 students.csv", demo)
        counts = shell("cut -d',' -f4 students.csv | tail -n +2 | sort | uniq -c", demo)

        outputs = [undate(text) for text in
                   (first, summary[0].read_text(encoding="utf-8"), log, collections,
                    performance, arrays, analysis, csv_summary, previews, counts)]
        for block in guide_blocks():
            assert any(contains(output, undate(block)) for output in outputs), block

    after = sources()
    assert before == after, "a demo changed the course sources"
    print("Lecture 03: demo outputs, repeat runs, and every guide expectation matched real runs.")


if __name__ == "__main__":
    run()
