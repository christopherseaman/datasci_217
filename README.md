# Introduction to Python & Data Science Tools

## Lecture authoring

The course runtime is Python 3.13, with pandas 3.0.5 where used. Supplied projects ship `pyproject.toml` and `uv.lock`: run `uv venv --seed`, activate the environment, and run `uv sync`. Add packages with `uv add`; `requirements.txt` is the labeled alternative. Artifact checks also support Python 3.9 and never grade a reported Python version.

Notion is the primary publishing surface. Write lectures and bonus pages with `#` for major sections, `##` for subsections, and `###` for deeper topics; multiple H1 sections are intentional. Keep each prose paragraph on one physical line, with no line-length limit. Preserve code-block formatting and list nesting. Do not add horizontal rules or prose that merely repeats a heading. Preserve humor.

When syncing, preserve heading levels exactly. Resolve local links and media URLs for Notion, preserve native child pages, and omit navigation links already represented by those child pages. YAML mapping metadata stays local. Notebook demos remain Markdown sources generated into `.ipynb` files, with Colab links targeting the notebooks.

Keep course navigation inside Notion: link to mapped Markdown sources, not website routes. Outbound links are for external references (including documentation and software), Colab notebook demos, and assignments. Embedded media and attached demo scripts remain inline. `index.md`, `references.md`, and `shell_workout.md` are mapped course pages alongside lectures, bonus pages, and the Lecture 01–03 demo guides.

Use bare live-demo markers without descriptions beneath them. The final demo comes after all lecture content, including any closing humor. Do not append summaries, worked examples, key takeaways, or meta-content after it; integrated walkthroughs belong in the demos, and optional reference material belongs in the bonus page.

Publish only when the instructor asks. Fetch the live page and reconcile its content and child pages with the local source first, preserving unrecognized edits. From the repository root, run `python3 scripts/notion_push.py SOURCE.md --dry-run` to rehearse, then `python3 scripts/notion_push.py SOURCE.md` to publish. The command preserves child pages, uploads and attaches local media, and re-reads the result to verify it. The optional `notion.title_line` identifies the exact local title to omit because Notion already displays its page title.

## Resources

- GitHub repo - https://github.com/christopherseaman/datasci_217

#### References

- [Python for Data Analysis](https://wesmckinney.com/book/) (rough basis for Python content)
- [The Missing Semester](https://missing.csail.mit.edu/) (command line, git, data wrangling)
- [The Linux Command Line book](http://linuxcommand.org/tlcl.php) (command line in-depth)
- [Markdown Guide](https://www.markdownguide.org/) (you can probably figure this one out)

#### Development Tools (free!)

- [VS Code](https://code.visualstudio.com/) (it’s pretty good)
- [Python](https://www.python.org/) (great docs & tutorials, too!)
- [GitHub Codespaces](https://github.com/features/codespaces) (IDE in a browser; free usage quota)
- [Google Cloud Shell](https://cloud.google.com/shell/docs) (practice command line anywhere)

## Assignments

Each lecture directory contains the source for its assignment. The instructor
supplies the term-specific assignment repository URLs separately.

Term-specific repository URLs: `#FIXME:ASSIGNMENT_URLS`

The Fall 2026 repositories are listed in [assignments.json](assignments.json) for graders and future fork collection. Each public `UCSF-DataSci/ds217-26f-##` repository contains the corresponding assignment directory, excluding development-only `_grader_selftest` fixtures. Students fork the repository and commit their completed artifacts. GitHub Actions runs the checks automatically on every push, pull request, or manual dispatch; incomplete scaffold submissions are expected to fail. Students may need to enable Actions once in their fork's Actions tab. Course lecture links remain term-neutral placeholders.

### Grading

Homework handouts ship the course's artifact checks. From a homework directory, run `python3 check_assignment.py`; GitHub Actions downloads the current whole checker set and runs it through pytest. Automated grading reads saved submission artifacts without running notebooks or inspecting how students wrote their solutions. Exam handouts ship no checks; their course-owned checks run after the deadline.

Points are awarded only for documented student grading materials, not supplied notebooks, input files, grader files, or repository bookkeeping. A grading target can contain just the required outputs and written responses; extra files are ignored. Expected results use the trusted grader's input data. Untouched scaffolds earn zero. Run `uv run scripts/test_assignment_grading.py` to verify the empty/scaffold contract across all eleven assignments; each course-owned `_grader_selftest` exercises completed and incorrect artifacts.

For grading another submission, run the trusted assignment's `check_assignment.py /path/to/submission --json`. Use the published assignment version and its dependencies, not checker code supplied by the submission. The JSON report contains the same milestone results and automated score shown to students; batch collection and reporting do not change grading criteria.

For TA batch grading, run `uv run scripts/grade_submissions.py 02`. To prepare LMS identities, build the private roster once from the class spreadsheet and the collected Assignment 01 clones:

```bash
python3 scripts/build_roster.py ds217-f26-roster.xlsx --submissions scratch/submissions/01
```

This creates `roster.tsv` in the current directory, with `github_user`, `email`, `last_name`, `first_name`, `course`, and `exclude` columns. The course includes the semester, such as `ds217-26f`, matching the assignment repository prefix in `assignments.json`; keep one roster per course offering. Matching uses committed Assignment 01 identity hashes internally, with the same email normalization as `process_email.py`; hashes are omitted from the export. Unmatched or ambiguous students keep a blank `github_user`; fill missing logins or emails manually in the TSV. Add TA logins with `exclude` set to `yes` to skip their forks, including explicit `--fork` requests; their other identity fields may be blank. Existing CSV rows are preserved, but excluded users cannot receive the star. The builder refuses to overwrite an existing map, preserving manual corrections.

Share the TSV with TAs through an access-restricted channel. The workbook and repo-root `roster.tsv` are Git-ignored; neither belongs in a public repository. The grader expects `roster.tsv` in the directory where grading is invoked, or accepts `--roster /path/to/roster.tsv`. It reads that edited file directly, shows aligned email, last-name, and first-name columns beside terminal results, and appends `email`, `last_name`, and `first_name` after the grading columns in `grades.csv`. Missing mappings leave identity fields blank without warnings. Existing rows also get refreshed identities when any fork is graded. Grade exports contain personal information and must also stay private.

TAs can also run the grader without cloning this repository. In the directory containing their privately shared `roster.tsv`, run:

```bash
uv run https://raw.githubusercontent.com/christopherseaman/datasci_217/main/scripts/grade_submissions.py 02
```

This downloads the current course checks and saves submission clones and `grades.csv` under `./ds217-26f-02/`.

For the lab music choice, optionally keep `music.lst` alongside `roster.tsv`, one email per line for people already offered a turn. The ⭐ goes to the earliest full-score row whose user is not excluded and whose email is not in this list. Matching ignores email case and surrounding whitespace. Music exclusions do not change grades or hide students from results; the script never adds winners to the list automatically. With no list, every non-excluded full-score row is eligible. Selection uses saved submission commit dates, not the first historical commit that passed. Both private files are Git-ignored.

Assignments 01–05 and 07–10 have 100 automated points. Both exams, Assignments 06 and 11, have 75 automated points plus 25 human-review points. Exam handouts ship no checks: after the deadline the course grades each exam's committed files with the course-owned checks in `06/mid-term_checks/` and `11/assignment_checks/`, and each exam README gives the full rubric, including the human-review criteria. Passing automated checks does not award human-review points.

## Lectures

1. [1. Command Line + Python](01/README.md)
    - Command line navigation and file operations
    - Python installation and basic setup
    - VS Code setup and basic workflows
    - Introduction to development tools
2. [2. Python + Git](02/README.md)
    - Python syntax, variables, and data types
    - Control structures and functions basics
    - Git through VS Code/GitHub (GUI focus)
    - Project organization and collaboration
3. [3. NumPy + Virtual Environments](03/README.md)
    - N-dimensional arrays and array creation
    - Array operations and universal functions
    - Boolean indexing and fancy indexing
    - Basic mathematical operations
4. [4. Jupyter + Pandas](04/README.md)
    - Jupyter notebooks
    - Series and DataFrame creation
    - Data selection and filtering
    - Reading CSV, Excel, and JSON files; inspecting data types, missingness, and duplicate rows
5. [5. Data Cleaning](05/README.md)
    - Data transformation techniques
    - String operations for data cleaning
    - Handling missing data strategies
    - Data validation and quality assessment
6. [6. Joins, Combining + Reshaping](06/README.md)
    - Merge, join, and concatenate operations
    - Reshaping data (pivot, melt, stack/unstack)
    - DataFrame indexes and basic MultiIndex structures
    - Combining and reshaping data with explicit shape and key contracts
7. [7. Data Visualization](07/README.md)
    - Matplotlib fundamentals
    - Pandas plotting interface
    - Seaborn for statistical visualization
    - Creating effective data visualizations
8. [8. Aggregation + Group Operations](08/README.md)
    - GroupBy mechanics and advanced techniques
    - Aggregation functions and transformations
    - Pivot tables and cross-tabulation
    - Result-shape choices: aggregation, transform, filter, and apply
9. [9. Time Series Analysis](09/README.md)
    - Time series data handling and analysis
    - Date/time operations and resampling
10. [10. Statistics, Machine Learning + Deep Learning Models](10/README.md)
    - Statistical modeling, scikit-learn, gradient boosting, and neural-network foundations
    - Leakage-safe splitting, preprocessing, comparison, and evaluation
11. [11. From Question to Defensible Result](11/README.md)
    - A question-led, iterative capstone workflow
    - Data contracts, provenance, and defensible conclusions
