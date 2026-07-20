# Lecture 08 demo implementation blueprint

Status: independently reviewed, implementation-ready design handoff; the Lecture
08 narrative and its Lecture 07 to 08 dependency boundary have passed independent
verification. This document authorizes only demo implementation, not Assignment
08 redesign.

## Accepted demo role

The required demonstrations must teach one cumulative progression:

1. predict grouping identities, count, and output grain before computing, then
   distinguish `size`, `count`, and `nunique` by the question each answers;
2. create a flat named aggregation, contrast it with a same-index `transform`,
   and make a bounded two-key result's columns and ordering deliberate; and
3. build exactly one aggregating `pivot_table`, compare every populated cell to
   the equivalent two-key GroupBy result, and distinguish an absent combination
   from a measured zero.

All three notebooks use the same small encounter fixture so students can focus
on result grain rather than repeatedly learning new data. The demos do not
clean, join, impute, reshape structurally, analyze time, model, fetch network
data, or introduce a new visualization objective.

## Exact package

Replace the current required demo package atomically with:

```text
08/demo/
|-- .gitignore
|-- .python-version
|-- DEMO_GUIDE.md
|-- requirements.txt
|-- data/
|   `-- encounters.csv
|-- demo1_grouping_grain_counts.ipynb
|-- demo2_named_aggregation_transform.ipynb
`-- demo3_aggregating_pivot.ipynb
```

Delete the three legacy notebooks, their paired same-stem Markdown copies,
`live_demo_guide.md`, and any committed generated output. In particular, the
remote-computing/performance notebook is not a required aggregation demo.
Generated demonstration evidence belongs under ignored `08/demo/output/` and
is never a teaching input.

`.gitignore` must ignore `output/`, notebook checkpoints, Python caches, and
common local virtual-environment directories. It must not hide the notebooks,
guide, requirements, or committed fixture.

Use exact candidate records:

```text
.python-version: 3.12.13
requirements.txt:
numpy==2.0.2
pandas==3.0.3
```

Jupyter, notebook execution tools, and image inspection tools used only during
certification are platform/test dependencies, not lecture runtime imports.

## Pinned encounter fixture

`data/encounters.csv` is course-authored, synthetic, complete except for the
intentional missing ratings, and non-identifying. Its grain is one row per
recorded encounter. Its exact SHA-256 is
`24a31904c1371553ff3af627dc21146ed743c8c0c47452ade3628c2fc199c5dc`.
Use UTF-8, comma delimiters, `\n` line endings, the exact column order below,
and a final newline:

```csv
encounter_id,facility,provider_id,service,charge,wait_minutes,rating
E001,North,P01,Consult,120,20,4
E002,North,P01,Follow-up,80,12,
E003,North,P02,Consult,150,30,5
E004,North,P02,Procedure,210,50,5
E005,South,P03,Consult,110,18,4
E006,South,P03,Consult,90,16,
E007,South,P04,Procedure,220,45,
E008,South,P04,Procedure,125,25,4
E009,West,P05,Consult,130,25,3
E010,West,P05,Procedure,200,40,4
E011,West,P06,Consult,140,35,3
E012,West,P06,Follow-up,75,15,4
```

After parsing, every notebook must apply the same semantic dtypes and category
orders:

```python
FACILITY_LEVELS = ["North", "South", "West", "Remote"]
SERVICE_LEVELS = ["Consult", "Follow-up", "Procedure"]
```

- `encounter_id` and `provider_id`: pandas string dtype;
- `facility`: ordered categorical with `FACILITY_LEVELS`;
- `service`: ordered categorical with `SERVICE_LEVELS`;
- `charge` and `wait_minutes`: NumPy `int64`; and
- `rating`: nullable integer (`Int64`).

The unused `Remote` level and absent South--Follow-up combination are deliberate
and must remain distinct. Missing ratings make `size` differ from `count`, and
repeated providers make `count` differ from `nunique`.

Each notebook must resolve one demo root before reading or writing. Search the
current directory and its ancestors for a directory containing the supplied
`DEMO_GUIDE.md` and `.python-version`, recognizing both the course layout
`08/demo/` and a flattened standalone demo directory. If no such root exists,
use the current directory as the ephemeral standalone/Colab root. Read
`data/encounters.csv` below that root, reconstruct the exact supplied bytes only
when that path is absent, and verify its checksum before parsing. A present but
corrupted fixture must stop execution rather than be overwritten or silently
replaced. Write only below `<resolved-demo-root>/output/`. Repository root,
`08/demo/`, a directory nested below `08/demo/`, and standalone launch
directories must therefore be equivalent. No manual upload, Drive mount,
credential, randomness, mutable date, or network fetch is allowed.

## Notebook-wide contract

Every notebook must have:

- portable `Python 3` kernelspec metadata, stable globally unique cell IDs,
  null execution counts, and zero stored outputs;
- a first Markdown cell stating the learning question, input and output grain,
  Colab-first/local-Jupyter equivalence, ephemeral filesystem, privacy rule,
  fresh-execution rule, that Colab changes are not automatically saved to the
  repository, and the assignment-Colab pilot boundary;
- one supplied setup cell that conditionally installs only mismatched course
  packages before their first import, then prints and asserts Python 3.12.13,
  NumPy 2.0.2, and pandas 3.0.3;
- deterministic fixture/bootstrap logic with checksum verification and the
  exact dtype/category contract above;
- explicit output-directory creation and deterministic replacement of stale
  files that the notebook owns;
- UTF-8 CSV writes with `lineterminator="\n"` and `index=False` after moving any
  meaningful index labels into ordinary columns; the pivot output must also use
  an explicit empty `na_rep` so an absent combination is never serialized as
  zero;
- concise instructional Markdown that defines each newly demanding term before
  its first code use: GroupBy/grouping unit/output grain in Demo 1, named
  aggregation and `transform` in Demo 2, and aggregating `pivot_table` plus its
  structural-`pivot` contrast in Demo 3;
- explicit `observed=True`, `sort=True`, and `dropna=True` on required GroupBy
  calls, plus deliberate `as_index=` where layout matters;
- executable assertions for group identities, group count, row count, index,
  columns, order, values, paths, and exact CSV readback; and
- a final verification cell that passes only after all prior source executes
  freshly.

The notebooks may display tables during instruction. Stored display output is
not execution evidence. Each notebook is self-contained and must not depend on
variables or generated files from another notebook.

## Demo 1: grouping grain and counts

Canonical filename: `demo1_grouping_grain_counts.ipynb`.

Before creating a GroupBy object, require learners to state this exact contract:

- input row grain: one recorded encounter;
- grouping key: `facility`;
- grouping unit: one observed facility;
- predicted group identities: North, South, and West;
- predicted number of groups: three;
- category policy: observed category values only; and
- aggregated output grain: one observed facility.

Create one reusable `facility_groups` object and verify `ngroups`, exact group
identities, group sizes, and conservation of all 12 rows. Then answer three
plain-language questions:

1. How many encounter rows were recorded? Use `size`.
2. How many encounters have a recorded rating? Use `rating.count`.
3. How many distinct providers appear? Use `provider_id.nunique`.

Combine the three results into the exact flat table:

```text
facility,encounter_count,rating_count,unique_provider_count
North,4,3,2
South,4,2,2
West,4,4,2
```

Write only `output/count_comparison.csv` with `index=False`. Assertions must
cover exact input shape and dtypes, the unused Remote category, three observed
groups, the three distinct counting semantics, exact schema/order/values,
encounter conservation, schema-aware exact readback, and identical output bytes
on a deterministic repeat run. The explanation must state why the three
operations can all be correct without being interchangeable.

## Demo 2: named aggregation and transform

Canonical filename: `demo2_named_aggregation_transform.ipynb`.

Use one `facility` key twice to make the grain contrast visible.

First create a flat named aggregation with `as_index=False` and exact columns:

```text
facility,encounter_count,rating_count,unique_provider_count,total_charge,mean_wait_minutes
```

Its exact facility-order values include total charges `[560, 545, 545]` and
mean waits `[28.0, 26.0, 28.75]`. State and verify that its grain is one observed
facility and its three rows conserve all 12 encounters through
`encounter_count`.

Then compute facility mean charge with a selected-Series `transform("mean")`
and add `facility_mean_charge` plus `difference_from_facility_mean` to the
encounter table. State and prove that the result retains one encounter per row,
the original 12-row count, and the exact original index. Include a bounded
diagnostic showing why the three-row aggregation has the wrong grain for direct
positional assignment to 12 encounter rows. Use a three-value NumPy array or an
equally unambiguous positional value, verify the `3 != 12` length mismatch, catch
the expected `ValueError`, and leave the encounter table unchanged. Do not use a
three-row pandas Series for this diagnostic: its label alignment would demonstrate
a different failure mode. Explain the failure without leaving an expected
exception uncaught or matching a version-sensitive exception string.

End with one two-key named aggregation on `facility` and `service`, using
`as_index=False`, exact flat columns
`facility,service,encounter_count,mean_charge`, eight observed combinations,
and an encounter-count sum of 12. Assert that South--Follow-up is absent and
Remote is absent. This is not a lesson in MultiIndex manipulation.

Write exactly:

- `output/facility_summary.csv`;
- `output/encounters_with_context.csv`; and
- `output/facility_service_summary.csv`.

Use `index=False`, preserve deterministic category ordering in the serialized
rows, and perform schema-aware exact readback checks. Do not round results to
make assertions pass.

## Demo 3: one aggregating pivot

Canonical filename: `demo3_aggregating_pivot.ipynb`.

Before execution, require the learner to name all five pivot choices:

- `index="facility"`;
- `columns="service"`;
- `values="charge"`;
- `aggfunc="mean"`; and
- `observed=True`.

Build exactly one `pivot_table` with explicit `sort=True` and `dropna=True`.
Build the equivalent two-key GroupBy mean-charge result independently in the
same notebook, and compare every populated group value with its pivot cell.
The pivot must have row order North, South, West; column order Consult,
Follow-up, Procedure; value 135.0 at North--Consult; value 172.5 at
South--Procedure; exactly eight populated cells; South--Follow-up as the only
missing cell; and no Remote row.

State both meanings: the displayed row grain is one observed facility, while a
populated cell summarizes one observed facility--service group. Explicitly
reject interpreting the absent South--Follow-up combination as a measured zero.

Write only `output/mean_charge_pivot.csv` with the facility labels serialized
as an ordinary first column. Read it back with a schema-aware contract that
preserves the distinction between missing and zero. No chart is required: the
table comparison is the learning objective, and Lecture 07 visualization need
not become a second objective here.

## Guide and publication contract

`DEMO_GUIDE.md` must identify the exact three notebooks, objectives, fixture and
checksum, generated outputs, expected visible tables, launch paths, actionable
learner prediction checkpoints with their expected outcomes and concise
explanations, likely failure modes, destructive/repeat-run rehearsal, privacy,
and scope boundaries. The guide must not use instructor talking points or
"now discuss" meta-instructions. It must explain `size` versus `count` versus
`nunique`, aggregation versus `transform`, and structural `pivot` versus
aggregating `pivot_table` without introducing later concepts.

Link each notebook through a development Colab badge and state that every badge
must move to one immutable release tag and be fresh-run before publication. The
development link must use the official course-repository Colab URL form; it may
not imply that Colab edits save back to GitHub. The guide's certification table
starts with local candidate, fresh Colab, and immutable badge reference all
pending, and has fields for execution date, runtime, maximum observed memory,
warnings, and result. Authorship or stored output is not independent
certification. Record measured runtime and memory during certification rather
than inventing publication values during implementation.

## Independent QA matrix

After implementation, a reviewer who did not author the notebooks must verify:

- actual fresh-kernel execution of all three notebooks from repository root,
  `08/demo/`, a directory nested below `08/demo/`, and standalone layouts;
- a separate progressive execution with lecture warnings promoted to errors;
- missing fixture reconstruction from the exact embedded bytes and present-but-
  corrupted fixture failure before parsing or replacement;
- stale/deleted/corrupt generated outputs, deterministic repeat replacement,
  and stable repeat hashes for every owned CSV;
- exact GroupBy policies, identities, counts, grain, index, columns, ordering,
  values, pivot equivalence, missing-combination semantics, and CSV readback;
- portable metadata/state/IDs, exact dependencies, guide claims/checksum, no
  paired Markdown, and no committed generated artifacts; and
- absence from code cells of cleaning/imputation, merge/join/concat-as-a-new-
  lesson, structural reshape, `GroupBy.apply`, filtering groups, advanced
  MultiIndex manipulation, crosstabs, statistics/modeling, periods/resampling/
  rolling, visualization instruction, remote computing, performance work,
  network/upload/Drive, randomness, and mutable dates.

Fresh Colab execution and immutable release-tag badges remain separate
publication gates even after local independent QA passes.
