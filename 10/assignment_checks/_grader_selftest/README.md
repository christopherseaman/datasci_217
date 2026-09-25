# Assignment 10 checks self-test

Course-side QA for the checks, not a second grading mode. It answers the
assignment with statsmodels, scikit-learn, and pandas from
`10/assignment/data/`, the way the lecture does, builds submissions in ignored
`scratch/`, and confirms that:

- every expected value in `_value_checks.py` matches a fresh computation from
  the data, for both refits of the frozen pipeline;
- a correct submission scores 100, and so do one refitted on training plus
  validation rows, one saved with every row index, and one written
  differently: columns and rows in another order, CRLF, a byte-order mark,
  UTF-16, quoted or padded cells, labels and headers in another letter case or
  with spaces for underscores, rounded numbers, a leading row-number column,
  timestamps written with `Z`, another UTC offset, or no zone, `1`/`0`
  booleans, `keep`/`exclude` decisions, the array interface's `const`, Demo 1's
  `valid` label, `mean_se` left out, every file separated by semicolons (with
  decimal commas) or by tabs, every number rounded to one decimal, and
  accuracy, precision, recall, and R² written as percents such as `85%`,
  where a one-decimal value that rounds something else, a wrong percent, or a
  percent in a column that is not a proportion still costs only its own
  check;
- an empty directory and the untouched handout score 0;
- each of 45 single mistakes, one per check, costs exactly that check,
  including a misnamed column; a coefficient table saved without its terms
  costs only the columns check; approaches saved under other labels cost only
  the rows check; a missing file costs only its own artifact's
  checks; leaky features and a split on the visit time land on the checks
  that name them, and the feedback says what was expected and what was found;
- the checks read only `output/`: poisoned data and code in a submission
  change nothing;
- the handout README's checkpoint header lines, completion contract, and task
  numbering match the checks, `POINTS`, and the notebook, and neither uses an
  em dash;
- every file the workflow lists in `CHECKS_FILES` is byte-identical in
  `10/assignment/` and here, the handout has no other Python file and an empty
  `output/`, and its notebook ships with outputs cleared; both
  `check_assignment.py` copies and pytest report the same result.

```bash
uv run --python 3.13 --with-requirements 10/assignment/requirements.txt --with 'pytest>=8,<9' python 10/assignment_checks/_grader_selftest/run.py
```
