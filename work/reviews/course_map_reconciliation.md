# Course map reconciliation

Status: targeted follow-up check of `course_dependency_alignment.md`, `course_refresh_2026.md`, and the three detailed range matrices. No lecture, demo, assignment, or grader source was edited.

## Overall result

**Pass — the documentation contradictions are resolved.** The course map, master refresh plan, and range matrices now describe one consistent instructional sequence and environment policy.

| Check | Result | Current contract |
|---|---|---|
| Canonical concept homes and entry/exit contracts | **Pass** | Lecture 02 supplies minimal dictionaries/text I/O and functions/modules; Lecture 05 supplies raw/schema/provenance and row meaning; Lecture 06 formalizes grain/keys/joins/structural reshape; Lectures 08–10 own grouped result grain, entity/time semantics, and modeling contracts respectively. |
| Lecture 06/08 reshape boundary | **Pass** | Lecture 06 owns structural `melt`/`pivot`; Lecture 08 owns aggregating `pivot_table` after GroupBy and result grain. |
| Terminal/Jupyter boundary | **Pass** | Lectures 01–03 use terminal-executed scripts and shell files. Lecture 04 defines notebook/cell/kernel-runtime/state/order/output before independent notebook work. |
| GUI-first Git | **Pass** | Lecture 01 repository access and first synchronization are guided onboarding. Required independent Git work begins in Lecture 02 through VS Code Source Control or GitHub Desktop; CLI Git remains bonus. |
| Notebook execution/output | **Pass** | Students restart/run-all; graders execute a fresh copy and ignore stored output as execution evidence; sensitive outputs are cleared; ordinary outputs remain only when a human rubric needs them. Generated `output/` files are a separate artifact contract. |
| Colab policy | **Pass** | Compatible demos from Lecture 04 are Colab-first and local-Jupyter compatible. Notebook assignments require clean local-Jupyter execution; Colab becomes an assignment path only after repository-save/submission validation. |
| Cross-range terminology | **Pass** | Lecture 09 now defines entity, entity key, and entity-plus-timestamp ordering. Lecture 10 defines residual and residual plots rather than inheriting those terms from Lecture 07. |
| Lecture 10 modeling depth | **Pass after specialist reconciliation** | Required prediction work is a baseline plus one train-only linear Pipeline. Binary classification is metric literacy from supplied predictions; a tree ensemble is optional bonus, not a second required classifier/model. |
| Lecture 11 project/data contract | **Decision recorded, implementation blocked appropriately** | The narrative may teach the integrated contract, manifest, and evidence workflow. Dataset-specific demos and the final assessment wait for a frozen entity/key/grain, target/horizon/cutoff, metric/baseline, and immutable licensed release; the current Chicago weather feed cannot supply the previously proposed water-temperature target. |
| Candidate runtime | **Pass as a candidate, not a release lock** | Official sources support Python 3.12.13/NumPy 2.0.2 for Colab 2026.04. pandas 3.0.4 was yanked after the earlier reconciliation; pandas 3.0.3 is now the candidate and has passed an isolated import/version smoke test. The full scientific stack, notebooks, graders, and two-environment matrix remain uncertified. |

Runtime sources checked 2026-07-18: [Colab runtime FAQ](https://research.google.com/colaboratory/runtime-version-faq.html), [PyPI pandas release history and yank notice](https://pypi.org/project/pandas/), and [pandas 3.0 support changes](https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v3.0.0.html).

## Remaining substantive decisions

1. **Assignment 01:** confirm implementation of the proposed redesign and its guided, unassessed first GUI synchronization rather than restoring the historical package unchanged.
2. **Lecture 05 assessment role:** confirm whether it remains a 100-point midterm and set the human-rubric weight.
3. **Lecture 11 assessment contract:** confirm its final-assessment role; freeze the entity, target, horizon, availability cutoff, primary metric, baseline, and immutable licensed dataset snapshot before authoring. The current Chicago weather feed contains air temperature rather than the previously proposed water-temperature target and is not an immutable grading release.
4. **Human visualization grading:** approve the concise integrity/accessibility/communication rubric that automated public checks cannot replace.
5. **Supported runtime:** freeze Python and package constraints only after required scripts, notebooks, plotting/stats/modeling dependencies, and graders pass locally and in fresh Colab.
6. **Colab assignments:** approve Colab as a submission path only after the Assignment 04 save-to-repository and Classroom 50 pilot succeeds.
7. **Classroom 50 evidence and grading:** determine whether grading checkouts preserve enough Git history/branch refs for Assignment 02, and confirm that discoverable graders are acceptable for each assessment.
8. **Course operations:** choose the production GitHub organization/Classroom 50 classroom and staff access; individual/group mode, roster, late and authoritative-submission rules; grade-system import and retention policy; canonical repository/release convention; and source-of-truth syllabus.
