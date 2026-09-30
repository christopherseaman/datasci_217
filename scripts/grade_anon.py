# /// script
# requires-python = ">=3.13,<3.14"
# dependencies = ["numpy==2.3.3", "pandas==3.0.5", "scikit-learn==1.9.0"]
# ///
"""Grade every student fork like grade_submissions.py, printing no GitHub user names.

    uv run scripts/grade_anon.py 02

Takes the same arguments and writes the same grades.csv, which still names
each fork. On screen each fork is `student N` in grading order, and error
details have the user name replaced with <student>. The closing star line
names the earliest full-marks submission.
"""

import sys

from grade_submissions import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:], show_names=False))
