# Platform check

Use CPython 3.12.13 and the exact direct versions in `requirements.txt`.

Before submission:

- open the entire assignment directory, not only the notebook;
- confirm `data/fixture.json` and all four CSV fixtures are present;
- restart the kernel, clear output, and run all 30 cells in order;
- run `python check_assignment.py` from the assignment directory;
- run all cells and the checker a second time;
- confirm the nine required generated files appear in the Git GUI and are not ignored.

The notebook searches only the current directory and its parents for either this assignment marker or `10/assignment`. It never searches the whole filesystem, fetches data, mounts a drive, or deletes files other than its nine owned output paths.

Colab is conditional on the course launch route copying the complete assignment tree into the runtime. A standalone notebook upload is not supported.

This candidate is not release-certified until the course-wide exact transitive dependency lock is supplied and certified. The public learner workflow can be tested with the direct pins. The optional Actions workflow is feedback only; it does not replace instructor review.
