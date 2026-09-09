# Platform check

Use CPython 3.14 and the exact direct versions in `requirements.txt`.

Before submission:

- open the entire assignment directory, not only the notebook;
- confirm `data/fixture.json` and all four CSV fixtures are present;
- complete the notebook and commit its ten output artifacts (graders do not rerun the notebook);
- run `python check_assignment.py` from the assignment directory;
- confirm the ten required generated files appear in the Git GUI and are not ignored.

This assignment is local-only. The notebook searches only the current directory and its parents for either this assignment marker or `10/assignment`. It never searches the whole filesystem, fetches data, mounts a drive, or deletes files other than its ten owned output paths.

The optional Actions workflow is feedback only; it does not replace instructor review.
