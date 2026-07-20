# Lecture 08 demonstration guide

These three demonstrations teach one progression: predict grouped result grain,
choose counting and aggregation operations from the question, preserve encounter
grain with `transform`, and build one aggregating pivot whose cells are verified
against GroupBy. They use only the fixed synthetic encounter table below.

Required Lecture 08 demos are Colab-first and run equivalently in local Jupyter
or the VS Code notebook interface. Colab storage is ephemeral, and changes made
in a notebook opened from GitHub are not automatically saved back to the
repository. Assignment Colab support remains conditional on the repository-save
and Classroom50 pilot; these demo badges do not establish an assignment workflow.

## Launch the demonstrations

The badges below are development links to `main`, not publication references.
Each must be changed to the same immutable course release tag and fresh-run in
Colab before publication.

1. [![Open Demo 1 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo1_grouping_grain_counts.ipynb) — predict grouping grain and distinguish `size`, `count`, and `nunique`.
2. [![Open Demo 2 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo2_named_aggregation_transform.ipynb) — contrast named aggregation with same-index `transform`, then make a two-key result deliberate.
3. [![Open Demo 3 in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/08/demo/demo3_aggregating_pivot.ipynb) — specify one aggregating pivot and verify every populated cell against GroupBy.

For local use, open the repository in VS Code/Jupyter with Python 3.12.13 and
the exact direct dependencies in `requirements.txt`. The supplied setup cell
checks the environment and installs only a mismatched NumPy or pandas candidate
before importing either package. Start Jupyter from repository root, `08/demo/`,
or a nested directory; the notebooks resolve the same demo root in every layout.
Restart the kernel and run all cells in order. Stored output is not execution
evidence.

## Fixed teaching input

`data/encounters.csv` is course-authored, synthetic, non-identifying, and has
grain one row per recorded encounter. Its SHA-256 is
`24a31904c1371553ff3af627dc21146ed743c8c0c47452ade3628c2fc199c5dc`.
The declared facility categories are North, South, West, and unused Remote; the
declared service categories are Consult, Follow-up, and Procedure. Missing
ratings, repeated providers, and the absent South--Follow-up combination are
intentional teaching properties, not defects to clean or impute.

Every notebook verifies the committed bytes before parsing. If no fixture is
available in an ephemeral standalone/Colab layout, it reconstructs those exact
supplied bytes. A present but corrupt fixture stops execution and is never
silently replaced. No manual upload, Drive mount, credential, network fetch,
random input, or mutable date is required.

## Demo sequence and expected visible results

### Demo 1: grouping grain and count semantics

Before computing, learners predict North, South, and West as three observed
facility groups with output grain one observed facility. A **GroupBy object**
records which input rows belong together. It is not itself a summary.

- GroupBy `size()` counts encounter rows, including rows whose rating is missing.
- selected-column `count()` counts nonmissing ratings.
- selected-column `nunique()` counts distinct nonmissing providers.

The visible result is:

```text
facility,encounter_count,rating_count,unique_provider_count
North,4,3,2
South,4,2,2
West,4,4,2
```

Generated output: `output/count_comparison.csv`.

Learner prediction checkpoints:

| Predict before running | Expected outcome | Why |
|---|---|---|
| Input and output grain | one recorded encounter → one observed facility | Aggregation reduces all encounter rows in each facility group to one result row. |
| Three facility counts | North `4/3/2`, South `4/2/2`, West `4/4/2` for rows/ratings/providers | `size`, selected-column `count`, and selected-column `nunique` count different things. |
| Whether Remote appears | no Remote result row | Remote is declared as a possible category but has no input row, and `observed=True` keeps observed values only. |

### Demo 2: aggregation versus transform

A **named aggregation** reduces each group to deliberately named summary
columns. Its result has one row per observed facility. A GroupBy **transform**
returns a same-index value for every encounter row; it adds group context without
changing the twelve-row encounter grain. A bounded two-key aggregation then has
one row per observed facility--service combination.

Expected visible properties include facility total charges `[560, 545, 545]`,
mean waits `[28.0, 26.0, 28.75]`, twelve aligned encounter-context rows, and
eight two-key summary rows whose encounter counts sum to twelve.

Generated outputs:

- `output/facility_summary.csv`;
- `output/encounters_with_context.csv`;
- `output/facility_service_summary.csv`.

Learner prediction checkpoints:

| Predict before running | Expected outcome | Why |
|---|---|---|
| Positional assignment of three facility totals to twelve encounters | a caught length-mismatch failure and no mutated encounter table | Three aggregate values cannot supply one positional value for each of twelve rows. |
| `transform("mean")` shape and alignment | length 12 with the original encounter index | A selected-Series transform broadcasts one facility mean back to every row in that facility. |
| Two-key output grain and row count | one observed facility--service combination per row; eight rows | Three North, two South, and three West combinations occur in the input. |

### Demo 3: one aggregating pivot

A structural `pivot` from Lecture 06 requires unique row/column combinations and
does not aggregate. An aggregating `pivot_table` groups repeated combinations.
Learners name its index, columns, values, aggregation function, and observed-
category policy before execution.

The visible pivot has rows North, South, West and columns Consult, Follow-up,
Procedure. North--Consult is 135.0, South--Procedure is 172.5, and exactly one
cell, South--Follow-up, is missing. Missing means no input row for that
combination; it is not a measured zero. All eight populated cells equal the
independently built two-key GroupBy means.

Generated output: `output/mean_charge_pivot.csv`.

Learner prediction checkpoints:

| Predict before running | Expected outcome | Why |
|---|---|---|
| Displayed row and populated-cell meaning | one observed facility per row; one observed facility--service group per populated cell | The wide display has a row axis and a second grouping axis across columns. |
| Populated-cell count | eight | It must match the eight observed two-key GroupBy combinations. |
| South--Follow-up value | missing, not zero | No encounter has that combination; zero would claim that a measured mean charge exists and equals zero. |

## Rehearsal and likely failure modes

Run the rehearsal from repository root, `08/demo/`, a nested directory, and a
standalone copy. For each notebook:

1. delete its owned generated output and confirm a fresh run recreates it;
2. replace that output with stale or corrupt text and confirm a rerun replaces it;
3. run again and confirm same-platform output hashes are unchanged;
4. temporarily remove the fixture and confirm exact-byte reconstruction;
5. temporarily corrupt a present fixture and confirm checksum failure occurs
   before parsing or replacement; and
6. restore the committed fixture and restart/run all.

Common instructional failures are confusing declared categories with observed
groups, using `count` for a row-count question, assigning a short aggregation to
encounter rows, omitting explicit GroupBy policies, treating a categorical index
as an ordinary column, or reading an absent pivot combination as zero.

Each notebook owns only the outputs listed above. Its setup removes stale copies
of those files without deleting another notebook's artifacts. Runtime output is
ignored and never used as a teaching input.

## Privacy and scope boundaries

The fixture is synthetic and non-identifying. Do not substitute credentials,
private records, or sensitive stored output in a shared notebook. Required demos
do not clean, impute, join, structurally reshape, filter groups, use
`GroupBy.apply`, teach advanced MultiIndex manipulation, analyze time, model,
visualize, use remote computing, or optimize performance. Optional categorical
and indexed-result extensions remain in `../BONUS.md`.

## Certification record

Authorship and stored notebook output are not independent certification. Record
measured values during certification rather than estimating them now.

| Gate | Execution date | Runtime | Maximum observed memory | Warnings | Result/reference |
|---|---|---:|---:|---|---|
| Independent local candidate | 2026-07-18 | not reported | not reported | 0 notebook-code warnings | PASS: four-layout fresh-kernel, strict-warning, and adversarial matrices in `../../work/2026_refresh_audit.md` |
| Fresh Colab runtime | pending | pending | pending | pending | pending |
| Immutable release-tag badge | pending | pending | pending | pending | pending |

Independent local evidence and the earlier author-side checks are recorded in
`../../work/2026_refresh_audit.md`. Fresh Colab execution and immutable release-
tag badges remain separate publication gates.
