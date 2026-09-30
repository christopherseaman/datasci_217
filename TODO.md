# TODO

- [x] Standardize lectures, demos, assignments, and grading on Python 3.13.

## After Assignment 03 ships (Lectures 04-11)

Decided on 2026-09-29 while reworking Lecture 03; deferred so 03 goes out first.

- [ ] **`pyproject.toml` is the course default.** Lecture 03 now teaches `pyproject.toml` (`uv venv --seed`, `uv add`, `uv sync`, `uv run`) with `requirements.txt` as the alternative. Bring 04-11 in line:
    - Lecture 04: `uv pip install ipykernel` becomes `uv add ipykernel`; check every setup line in 04-11 READMEs and BONUS pages (`grep -rn "requirements.txt\|uv pip install" 0[4-9] 1[01]`).
    - Assignment handouts 04-11 (`NN/assignment/requirements.txt`): replace with `pyproject.toml` plus a generated `uv.lock`, and rewrite each README's Setup to match Lecture 03. Update anything that reads those files (selftests, `scripts/test_assignment_grading.py`, `scripts/publish_assignment.sh`, 11's `download_data.sh`).
    - Demos 04-11 (`NN/demo/requirements.txt`): same switch for local runs; keep each notebook's Colab `%pip install` cell (a seeded `.venv` keeps pip through `uv add` and `uv sync`, verified with uv 0.12.1).
- [ ] **Lecture 04 installs with `uv pip install`.** Lecture 03 now says to add packages with `uv add` because `uv sync` removes anything `pyproject.toml` does not list, but 04/README.md (lines ~30, 41, 631) still says `uv pip install ipykernel` and `uv pip install pyarrow`. Use `uv add` for a pyproject project (found by the 2026-09-29 Lecture 03 review).
- [ ] **Reading chapters in every lecture.** Lecture 03 now opens with a bulleted list of the chapters it covers (instructor, 2026-09-29; rule in AGENTS.md "Styling"). Add the same list to Lectures 01, 02, and 04-11 (Lecture 02 has only a one-line McKinney note).
- [ ] **Topic intros in 04-10.** Same instructor note as Lecture 03: each major topic's intro should actually explain the topic (what it is, why it matters) per AGENTS.md "Topic organization" (updated 2026-09-29 from ../datasci_223/AGENTS.md); subsections need no filler intros. An unverified first pass from 2026-09-29 is saved as `scratch/deferred/intro-tightening-04-10_2026-09-29.patch` (not applied); review it against HEAD before using it, or redo the pass.
- [ ] **Humor check in 04-10.** Lecture 03 had lost comics over the edits; check each lecture's history for dropped comics and long stretches without one.
- [ ] **Graders on an old system Python.** Assignment 04's `grading.py` no longer uses `zip(strict=True)`, which crashes with a traceback on Python 3.9 (macOS system `python3`); 01-03 and 05-10 still do. Apply the same explicit length check, keeping handout copies byte-identical.
- [ ] **Republish to Notion** (only when the instructor asks): Lecture 03 after its rework, and 04-10 after the intro pass.
