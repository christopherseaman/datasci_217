"""Run Lecture 02 demos in isolation; keep generated files out of course sources."""

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[1]


def run():
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch", prefix="lecture02-") as temporary:
        demo = Path(temporary)
        for name in ("functions_demo.py", "student_tools.py", "module_usage_demo.py"):
            shutil.copy2(ROOT / "02" / "demo" / name, demo / name)

        def python(*arguments):
            return subprocess.run(
                [sys.executable, *arguments], cwd=demo, check=True,
                capture_output=True, text=True,
            ).stdout

        assert python("-c", "import student_tools; import module_usage_demo") == ""
        assert not (demo / "grade_report.txt").exists()
        assert python("functions_demo.py") == (
            "=== Demo 2: Functions ===\n"
            "Before: every script would repeat this loop.\n"
            "Before loop extracted: [85, 92, 78]\n"
            "After: reuse helpers from student_tools.\n"
            "After get_grades() extracted: [85, 92, 78]\n"
            "Average grade: 85.0\nHighest grade: 92\n"
        )
        assert not (demo / "grade_report.txt").exists()
        expected = "Alice: 85\nBob: 92\nCharlie: 78\nAverage grade: 85.0\nHighest grade: 92\n"
        for _ in range(2):
            assert python("module_usage_demo.py") == (
                "Read back from grade_report.txt:\n" + expected + "Saved report matches: True\n"
            )
            assert (demo / "grade_report.txt").read_text(encoding="utf-8") == expected
        python("-c", "from student_tools import calculate_average, get_grades; "
               "assert calculate_average([]) == 0; assert get_grades([]) == []")
    print("Lecture 02: demo outputs, repeat execution, and silent imports passed.")


if __name__ == "__main__":
    run()
