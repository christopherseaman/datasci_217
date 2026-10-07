# TODO

- [ ] **Review AGENTS.md once Lecture 04 (lecture, demos, assignment) is put to bed:** a full pass to trim and consolidate the rules, starting with the 2026-10-06 notebook-setup rule ("A notebook needs almost no setup text..."), which is longer than it needs to be.

- [ ] **Validate classroom pacing:** rehearse the selected route against 90 minutes total, aiming for 60 minutes of lecture and 30 minutes of demos, including setup, questions, and transitions. Lecture 11 retains its project-workflow exception.

## Lectures 05-11: carry over the instructor's Lecture 04 notes (2026-10-06)

Lecture 04 is the current focus; these are applied to 04 first and then to the rest. The notebook-opening trim (no route table, no local setup prose, one install cell with a one-line caveat comment) is already running for 04-11.

- [ ] **Lists over paragraphs (done 05-11 2026-10-07 except 07 Data-Ink Ratio and Lie Factor, still prose); cut low-value prose (instructor, 2026-10-06, on Lecture 04):** topic intros, callouts, and explanations are too verbose. Prefer a few bullets (terms as bullets starting with a bold label) to paragraphs, keep callout headings short and on the issue with a short body (or none when the heading says it all), let snippets show rather than describe, and drop code snippets that exist only to fill the pattern. Apply to 05-11 after 04 is settled.
- [x] **Readings list format (instructor, Lecture 04 on Notion, 2026-10-07):** "This lecture covers McKinney, _Python for Data Analysis_ (3rd ed.):" followed by one bullet per section or chapter. Apply to 01-03 and 05-11, and fold into the AGENTS.md review.
- [x] **Remove intra-course pointers beyond the top BONUS and demo links:** no sentences advertising BONUS or other lectures (such as "BONUS.md shows how to..."), even unlinked. Apply to 01-03 and 05-11.
- [x] **One-liner jokes of questionable value:** keep only the ones that land (the instructor cut several in 04, such as the Konami-code, screenshot-friend, and dodgeball lines); review 05-11 for the same.
- [x] **`display()` over `print()` in all Jupyter content (04+), lecture snippets included,** except where a script is the point.
- [x] **Assignment README heading order (instructor, 2026-10-07, on Assignment 04):** Overview (a brief summary of the task, then the dataset) first, then Setup, then Files, then the tasks. Review 01-03 and 05-11 for the same order.
- [ ] **Assignment README trims (done 2026-10-07; open: 03 Windows bullet, git identity in 02-03, 02 branching steps) (instructor, 2026-10-07, on Assignment 04):** terminal and git-identity setup steps only in Assignment 01; no Windows or troubleshooting paragraphs; a very brief Check your work (push or run the checker; GitHub runs the latest checks) that also covers committing outputs, with no separate Submit section; the completion contract in a linked CHECKS.md. Apply to 02-03 and 05-11.
- [x] **Local checker always uses the latest course checks (instructor, 2026-10-07):** Assignment 04's check_assignment.py downloads the current checks from the course repo's main (falling back to the bundled copy offline) and exposes run_checks() for the notebook's last cell. Every assignment's local checker should behave this way: roll it out to 01-03 and 06-10, and to the course-side checkers for exams 05 and 11 (their handouts ship none).
- [ ] **Revisit Lecture 07's distribution of visualization content by package** (instructor, 2026-10-07): which topics belong to matplotlib, pandas .plot, seaborn, and Altair, and whether each demo's core route covers its package's block.
- [ ] **Instructor walkthrough of each lecture's demos in Colab**, as was done for 04: links open, the install cell is quiet, and each notebook runs; note what the instructor finds there.
- [ ] **Fuller demo data (done 06-08, 10-11; 05 and 09 Demo 3 not widened because Expect counts would change) and `display()`:** Lecture 04's Demo 1 now uses a fuller clinic table shown with `display()`; check each later demo's first tables for the same (more columns where they help, `display()` rather than `print()` for tables).
- [ ] **Notion republish of 05-11** after the instructor reviews 04: their pages still lack the local setup block at the top and the Colab link under each demo break (both are in the sources since `69dd1cd`).
