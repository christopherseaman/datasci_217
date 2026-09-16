"""Run Lecture 01 Python demos as scripts and as REPL entries."""

from code import InteractiveConsole
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "01" / "demo"


def output(name):
    return subprocess.run(
        [sys.executable, DEMOS / name], check=True, capture_output=True, text=True
    ).stdout


def repl_output(name):
    console = InteractiveConsole()
    stream = StringIO()
    errors = StringIO()
    with redirect_stdout(stream), redirect_stderr(errors):
        for line in (DEMOS / name).read_text(encoding="utf-8").splitlines():
            console.push(line)
        console.push("")
    assert errors.getvalue() == ""
    return stream.getvalue()


def run():
    expected = {
        "03a_values.py": "name: Alice type: <class 'str'>\nage: 22 type: <class 'int'>\nscore: 87.5 type: <class 'float'>\nenrolled: True type: <class 'bool'>\nnext score: 89.5\n",
        "03b_strings.py": "Welcome to Data Science 217\ncourse length: 16\nupper case: DATA SCIENCE 217\ntrimmed text: ready\n",
        "03c_calculations.py": "hours: 3\npoints per hour: 10\npoints earned: 30\nBMI: 22.857142857142858\n",
        "04a_decisions.py": "grade: B\ncandidate meets both requirements\n",
        "04b_for_loops.py": "score: 87\nscore: 92\nscore: 78\nscore: 95\nscore: 88\ntotal: 440\ncount: 5\naverage: 88.0\nassignment 1 score 87\nassignment 2 score 92\nassignment 3 score 78\nassignment 4 score 95\nassignment 5 score 88\n",
        "04c_loop_control.py": "counter: 1\ncounter: 2\ncounter: 3\nStop at the first score above 90:\nfound: 92\nSkip scores below 80:\nprocessing: 87\nprocessing: 92\nprocessing: 95\nprocessing: 88\n",
        "04d_debugging.py": "Corrected version: 440\nCorrected version: 26\nCorrected version: 42\n",
        "04e_measurement_workflow.py": "Student 1 score: 92 PASS\nStudent 2 score: 76 PASS\nStudent 3 score: 88 PASS\nStudent 4 score: 64 REVIEW\ntotal: 320\ncount: 4\naverage: 80.0\npassing: 3\n",
    }
    for name, text in expected.items():
        assert output(name) == text
        assert repl_output(name) == text

    exceptions = (
        ("# print(total_socre)", "print(total_socre)", NameError),
        ("# print(age_text + 1)", "age_text + 1", TypeError),
        ('# invalid_number = int("hello")', 'int("hello")', ValueError),
    )
    debugging_source = (DEMOS / "04d_debugging.py").read_text(encoding="utf-8")
    for commented_line, expression, error in exceptions:
        assert commented_line in debugging_source
        try:
            eval(expression, {"age_text": "25"})
        except error:
            pass
        else:
            raise AssertionError(expression + " did not raise " + error.__name__)
    print("Lecture 01: Python demos run as scripts and REPL entries.")


if __name__ == "__main__":
    run()
