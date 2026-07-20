# Lecture 05 demo guide

These three demos use one invented, pinned six-row fixture to practice a documented cleaning sequence: raw → audit → decide → transform → validate → save. Colab is the default launch experience; the same notebooks run top-to-bottom in local Jupyter.

## Launch the demos

The development badges point to the `eleventy` branch. Work opened from GitHub in Colab is not automatically saved back to GitHub.

| Demo | Colab | Local notebook | Purpose |
|---|---|---|---|
| 1. Audit before deciding | [![Open Demo 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/05/demo/demo1_audit_decisions.ipynb) | `demo1_audit_decisions.ipynb` | Measure source issues and record decisions without mutating raw data |
| 2. Targeted transformations | [![Open Demo 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/05/demo/demo2_targeted_transformations.ipynb) | `demo2_targeted_transformations.ipynb` | Apply only the recorded sentinel, normalization, type, date, and duplicate rules |
| 3. Validated pipeline | [![Open Demo 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/eleventy/05/demo/demo3_validated_pipeline.ipynb) | `demo3_validated_pipeline.ipynb` | Reproduce and read back a clean artifact, issue audit, and decision log |

Before publication, replace `eleventy` in all three badge targets with one immutable release tag. Open and fresh-run every resulting URL before calling the demos certified.

## Environment candidate

The compatibility candidate is Python 3.12.13, NumPy 2.0.2, and pandas 3.0.3. It is not the final course lock until both local-Jupyter and fresh-Colab certification are complete. Do not install pandas 3.0.4.

Each notebook begins with one supplied setup cell. It conditionally installs pandas 3.0.3 before the first pandas import and then prints the actual Python, NumPy, and pandas versions. It does not reinstall the complete Colab package collection.

For local use, start from this directory, create a Python 3.12.13 environment, and install the two deliberate direct course dependencies from `requirements.txt`. Open the notebook with the course Jupyter or VS Code host and select that environment as the Python 3 kernel. Jupyter host and kernel-support packages are platform tooling rather than lecture imports; record their versions during certification.

The portable kernelspec is named `Python 3`, not after a local virtual environment. The notebook is the sole executable teaching source; there are no paired same-stem Markdown copies.

## Pinned fixture and portable paths

`data/supplied_people_raw.csv` is a course-authored synthetic fixture. Its SHA-256 checksum is:

```text
7b3223154756aa59f2f00027ddbadaa225eeee51ad75d0df91de1fd8d14abe2d
```

Inside the course repository, each notebook searches upward for the committed fixture. When a Colab session or relocated notebook has no repository checkout, the notebook recreates the exact supplied bytes under a runtime-local `data/` directory. Both paths verify the checksum before pandas reads the file. This is deliberate supplied machinery, not a network download or manual upload.

Repository executions write generated files under `05/demo/output/`. Non-repository executions write under `output/` relative to the launch directory. Code creates the directory when needed, and `.gitignore` prevents generated files from becoming canonical teaching inputs.

## Demo 1: audit and decision checkpoints

The raw table has six rows and six columns. One row represents one submitted person record; `record_id` is the candidate identifier. `keep_default_na=False` preserves the empty strings, `NA`, `unknown`, and `-9` tokens for source-aware auditing.

Rehearse these distinctions in order:

- pandas initially recognizes zero missing values because the source-specific tokens remain text;
- three age sentinel tokens and one status sentinel token are present;
- two rows contain the invalid calendar date `2026-02-30`;
- two rows belong to one exact-duplicate set and the same two rows repeat the candidate identifier;
- four site values and one nonsentinel status value need format normalization; and
- the six-row decision table rejects unsupported imputation and adjacent-row filling.

The final cell asserts the observable counts and proves `raw.equals(raw_snapshot)`. Do not transform the raw table during this demo.

## Demo 2: transformation checkpoints

Begin by previewing the failure of an adjacent-row forward fill. It would copy `R003`'s visit date into `R004`, even though the rows represent different people and there is no within-entity order. The preview is never assigned to the working table.

Then apply the recorded rules:

- convert only the documented empty, `NA`, `unknown`, and `-9` sentinels to missing values;
- strip and normalize the bounded name, site, and status fields;
- retain only finite, integer-valued age parses as nullable `Int64` values;
- require exact ASCII `YYYY-MM-DD` source text before calendar parsing;
- use the raw-derived keep mask to remove one exact repeated submission; and
- retain uncertain rows with `needs_review` rather than filling or dropping them.

The final table has five unique record IDs. `R002` has missing age and visit date, `R004` keeps its missing visit date, and a separate `40.5` probe becomes missing without rounding.

## Demo 3: pipeline and artifact checkpoints

Run the complete notebook from a restarted runtime. The visible cells must reacquire and verify the source, create a nonmutating issue audit, record decisions, clean a copy, validate invariants, create the output directory, replace three CSVs, and read them back.

Expected generated artifacts:

| Artifact | Observable contract |
|---|---|
| `cleaned_people.csv` | five rows; exact clean columns and dtypes after schema-aware readback |
| `issue_audit.csv` | fifteen explicitly labeled issue counts |
| `decision_log.csv` | six decisions plus source checksum and before/after row counts |

The normalization regression must retain two raw-distinct rows that become equal after normalization. The fractional-age regression must count `40.5`, convert it to missing without rounding, retain nullable `Int64`, and flag the row for review. The final cell verifies the readbacks and confirms that the raw fixture checksum is unchanged.

Use disposable copies for destructive rehearsal. Corrupt the fixture in one copy and confirm that execution stops at the checksum assertion. Delete generated outputs and repeat a clean run to confirm deterministic replacement. Launch from the repository root, `05/demo/`, and a directory outside the repository to exercise both path branches.

## Scope and privacy policy

These required demos do not teach or use charting, joins, concatenation, reshape, GroupBy, aggregation, feature encoding, modeling, outlier rules, configuration frameworks, shell notebook automation, Drive mounts, or manual uploads. Bounded optional cleaning extensions live in `../BONUS.md`.

- The synthetic fixture contains no real person or protected information.
- Never put credentials, tokens, private records, or identifying data in notebook source or output.
- Stored notebook output is never execution proof. Canonical notebooks have cleared outputs and null execution counts.
- Executed certification copies and generated `output/` files are disposable.
- GitHub source opened in Colab is not automatically updated by edits made in the Colab tab.

## Certification record

Do not mark a row as passing without independent evidence from that environment.

| Notebook | Paired Markdown | Local candidate | Fresh Colab | Badge release ref |
|---|---|---|---|---|
| `demo1_audit_decisions.ipynb` | none — canonical notebook policy | pending | pending | development: `eleventy` |
| `demo2_targeted_transformations.ipynb` | none — canonical notebook policy | pending | pending | development: `eleventy` |
| `demo3_validated_pipeline.ipynb` | none — canonical notebook policy | pending | pending | development: `eleventy` |

For each certification run, record the notebook path, environment, Python/NumPy/pandas versions, launch working directory, fixture path and checksum, generated files, final verification result, tester, date, and immutable release ref. Do not treat this guide or committed notebook output as independent certification.
