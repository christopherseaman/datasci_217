# Lecture 06 demo guide

These three demos treat each table combination as an explicit contract: state row grain, test keys, predict row behavior, execute a validated operation, and verify the result. Colab is the default launch experience; the same notebooks run top-to-bottom in local Jupyter.

## Launch the demos

The development badges point to the `eleventy` branch. Work opened from GitHub in Colab is not automatically saved back to GitHub.

| Demo | Colab | Local notebook | Purpose |
|---|---|---|---|
| 1. Validated merge | [![Open Demo 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/06/demo/demo1_validated_merge.ipynb) | `demo1_validated_merge.ipynb` | Expose duplicate and orphan keys with `validate=` and `indicator=True` |
| 2. Concat alignment | [![Open Demo 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/06/demo/demo2_concat_alignment.ipynb) | `demo2_concat_alignment.ipynb` | Stack same-grain partitions with provenance and align deliberate index keys |
| 3. Structural reshape | [![Open Demo 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/06/demo/demo3_structural_reshape.ipynb) | `demo3_structural_reshape.ipynb` | Melt and pivot a unique structural round trip without aggregation |

Before publication, replace `eleventy` in all three badge targets with one immutable release tag. Open and fresh-run every resulting URL before calling the demos certified.

## Environment candidate

The compatibility candidate is Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. It is not the final course lock until both local-Jupyter and fresh-Colab certification are complete. Do not install pandas 3.0.4.

Each notebook begins with one supplied setup cell. It conditionally installs pandas 3.0.3 before the first pandas import and then prints the actual Python, NumPy, and pandas versions. It does not reinstall the complete Colab package collection.

For local use, start from this directory, create a Python 3.12.13 environment, and install the two deliberate direct course dependencies from `requirements.txt`. Open the notebook with the course Jupyter or VS Code host and select that environment as the Python 3 kernel. Jupyter host and kernel-support packages are platform tooling rather than lecture imports; record their versions during certification.

The portable kernelspec is named `Python 3`, not after a local virtual environment. The notebook is the sole executable teaching source; there are no paired same-stem Markdown copies.

## Pinned synthetic fixtures and paths

All fixtures are course-authored and non-identifying:

| Fixture | SHA-256 | Grain or role |
|---|---|---|
| `data/visits.csv` | `ccff0b9eaab1b6aae702734628db50b5223b04efc0071e1e9b4b9d6796e0c930` | one row per recorded visit; includes one orphan site code |
| `data/sites_history.csv` | `42e3b766ca41024b33883463d49c9be56d3998536e7f00e0ba483dd799fdd935` | one row per site-metadata version; includes two versions of N |
| `data/scores_wide.csv` | `5098b5e9a0165f9f7f6e22bc761f01d5ddb7205af9b88d41876f116cea2d7c38` | one row per participant and site with two repeated-measure columns |

Inside the repository, notebooks search upward for the committed files. When only a notebook is present, they recreate the exact supplied bytes under a runtime-local `data/` directory. Both branches verify checksums before pandas reads the files. No manual upload, Drive mount, credentials, or network data fetch is required.

Repository executions write generated artifacts under `06/demo/output/`. A standalone notebook writes under `output/` relative to its launch directory. Code creates the directory when needed, and `.gitignore` prevents runtime artifacts from becoming teaching inputs.

## Demo 1: validated merge diagnostics

State the starting grains before executing code:

- `visits`: one row per recorded visit; primary key `visit_id`; candidate composite key (`participant_id`, `visit_number`); foreign key `site_code`;
- `sites_history`: one row per metadata version, so `site_code` is deliberately duplicated until the supplied current-version rule is applied.

The intended current-site relationship is many visits to one site. The preservation goal is to keep every visit, including a visit whose metadata is absent, so predict a many-to-one left merge with six output rows.

Rehearse the failure and repair sequence:

1. `validate="many_to_one"` rejects the unfiltered history table.
2. Inspect the two N rows rather than silently dropping one.
3. Apply the supplied `record_status == "current"` source rule.
4. Retest the current-site key for nonmissingness and uniqueness.
5. Merge with explicit `on=`, `how="left"`, `validate="many_to_one"`, and `indicator=True`.
6. Inspect `V006`/`X` as the single `left_only` orphan.
7. Verify six preserved unique visit IDs, five matches, one left-only row, and zero right-only rows.

Do not delete the orphan or invent metadata during this merge demo.

## Demo 2: concat provenance and alignment

Vertically stack two three-row partitions that retain the one-row-per-visit grain and the same schema. Add `source_partition` before concatenation, use a fresh positional index, then verify six unique visit IDs and three rows from each source.

Next, remove `measure` and add `review_note` only in a disposable schema-drift preview. Explain why column-label alignment produces missing `measure` values on one side and missing `review_note` values on the other. This diagnoses a mismatch; it does not authorize filling or deletion.

Finally, horizontally concatenate two one-row-per-visit feature tables whose indexes are named `visit_id`. Their label sets differ deliberately:

- `V001` occurs only in the measure table, so its `review_score` is missing;
- `V006` occurs only in the review table, so its `measure` is missing.

The notebook writes and schema-aware reads back `combined_visits.csv` and `aligned_features.csv`. The vertical output omits its positional index; the horizontal output retains the named visit key intentionally.

## Demo 3: structural melt/pivot round trip

The wide source has one row per (`participant_id`, `site_code`). The long result has one row per (`participant_id`, `site_code`, `visit_label`). Students should predict six long rows before calling `melt()`.

Verify that the long key combination is unique, then use structural `pivot()` to reconstruct the source. Restore ordinary column labels/order, sort deterministically, and compare the reconstructed DataFrame exactly with the original.

A planted repeated identifier-variable combination must make `pivot()` fail. Do not replace it with `pivot_table()` here: aggregating repeated values belongs to Lecture 08 after grouping and result grain are defined.

The notebook writes `scores_long.csv` without a positional index and verifies a schema-aware readback.

## Destructive and repeat-run rehearsal

Use disposable copies for destructive checks.

- Corrupt each fixture separately and confirm that the consuming notebook stops at its checksum assertion.
- Remove a fixture in a disposable repository copy and confirm that the supplied runtime-local fallback produces identical bytes.
- Launch from the repository root, `06/demo/`, and outside the repository.
- Delete generated outputs and restart/run-all to confirm complete recreation.
- Repeat clean runs and compare output bytes to confirm deterministic replacement.

## Scope and privacy policy

The required demos do not teach or use data-cleaning decisions, GroupBy, aggregation, `pivot_table()`, advanced MultiIndex manipulation, plotting, datetime/resampling/rolling work, modeling, databases, or performance engineering. Optional index-based merge/join and advanced concat checks remain in `../BONUS.md`.

- The fixtures contain no real person or protected information.
- Never put credentials, tokens, private records, or identifying data in notebook source or output.
- Stored notebook output is never execution proof. Canonical notebooks have cleared outputs and null execution counts.
- Executed certification copies and generated `output/` files are disposable.
- GitHub source opened in Colab is not automatically updated by edits made in the Colab tab.

## Certification record

Do not mark a row as passing without independent evidence from that environment.

| Notebook | Paired Markdown | Local candidate | Fresh Colab | Badge release ref |
|---|---|---|---|---|
| `demo1_validated_merge.ipynb` | none — canonical notebook policy | pending | pending | development: `eleventy` |
| `demo2_concat_alignment.ipynb` | none — canonical notebook policy | pending | pending | development: `eleventy` |
| `demo3_structural_reshape.ipynb` | none — canonical notebook policy | pending | pending | development: `eleventy` |

For each certification run, record the notebook path, environment, Python/NumPy/pandas versions, launch working directory, fixture paths and checksums, generated files, final verification result, tester, date, and immutable release ref. Do not treat this guide or committed notebook output as independent certification.
