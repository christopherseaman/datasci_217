# Assignment 06 Blueprint: Validated Combination and Structural Reshaping

Status: accepted technical implementation handoff. The 90/10 scoring overlay remains provisional pending the revised syllabus; it does not block implementation of the fixture, tasks, artifacts, or grader behavior.

## Evidence audit

The current Assignment 06 package is not aligned with the accepted Lecture 06
scope or with the rebuilt demonstrations.

- The current README describes GroupBy and aggregation as Lecture 05 material,
  while the accepted course map places those concepts in Lecture 08.
- Current Question 3 requires dates, monthly grouping, `pivot_table`, and
  GroupBy aggregation. Those are later-course operations and do not assess the
  Lecture 06 structural-wrangling spine.
- The schemas documented in the current README do not match the schemas created
  by its data generator: customer and product columns differ, and the promised
  stock field is absent.
- The README requires `q1_validation.txt`, but the current test suite explicitly
  deletes it. The written submission contract and executable contract therefore
  disagree.
- The current merge work does not require `validate=`, a merge indicator, an
  observed duplicate-key failure, or exact unmatched-key invariants. The tests
  mostly accept nonempty or approximately shaped results; one missing-value
  check computes a Boolean but never asserts it.
- The current package creates large random fixtures at grading time, uses broad
  dependency lower bounds, pins Python 3.11 in its GitHub workflow, and downloads
  a mutable generator from an older course repository. Results are neither
  small nor fully reproducible under the accepted course environment.
- The paired Markdown/notebook assignment, paired Markdown/notebook generator,
  GitHub Classroom workflow, and lookalike tests multiply sources of truth.
- The accepted Lecture 06 narrative and demos instead teach five bounded
  capabilities: state grain/key/cardinality/preservation; validate a merge and
  inspect unmatched rows; concatenate with provenance and observe alignment;
  align columns by index; and perform nonaggregating wide/long reshaping with a
  duplicate-key failure.

This blueprint replaces that package with one local-Jupyter notebook and small,
checksum-pinned course fixtures. It is an implementation specification, not a
claim that the future assignment has passed review.

## Decisions fixed by this blueprint

- Assignment 06 is one notebook, `assignment.ipynb`, run in local Jupyter.
- It assesses exactly the five accepted Lecture 06 capabilities and no Lecture
  07 or later material.
- All data are course-authored, deterministic, nonidentifying, and committed to
  the assignment repository. Students do not generate data.
- The main relationship is specimens to station metadata. The unfiltered station
  history deliberately violates a many-to-one contract; a supplied domain rule
  selects current records before the validated merge.
- The join is a left join because the declared preservation goal is to retain
  every specimen, including an intentionally unmatched station code.
- Vertical concatenation uses an ordinary `source_partition` column. Horizontal
  concatenation demonstrates named-index alignment.
- Reshaping uses `melt` and `pivot`, never `pivot_table` or an aggregation.
- The notebook must create five exact CSV artifacts. A clean restart-and-run must
  reproduce them.
- Classroom50 is the course-wide assignment system, including Assignments 01--03.
  This design requires no GitHub Classroom export work.
- Colab remains a conditional platform pilot. Assignment 06 must not advertise a
  Colab badge until the course pilot records a pass for this package; local
  Jupyter remains mandatory.
- Scoring below is a provisional review model pending the revised syllabus. It
  does not establish a pass threshold, grade conversion, late policy, or timing.

## Learning and scope contract

After completing the assignment, a student must be able to:

1. state row grain, primary/candidate/foreign keys, predicted cardinality, and
   row-preservation intent before combining tables;
2. make a duplicate-key contract violation observable, apply a supplied
   deterministic record-selection rule, and perform an explicitly keyed,
   validated, diagnostic left merge;
3. stack row partitions with ordinary-column provenance while explaining schema
   alignment, and align feature columns by named index;
4. convert a wide table to long form without aggregation while preserving a
   declared long-form key; and
5. reconstruct the exact wide table with `pivot` and explain why duplicate
   identifier/variable pairs must be rejected rather than silently aggregated.

The following are out of scope:

- choosing missing-data treatments, filling, dropping, recoding, or general
  cleaning from Lecture 05;
- arbitrary deduplication; the only record selection is the supplied
  `record_status == "current"` source rule;
- `groupby`, `agg`, `aggregate`, `transform`, `pivot_table`, crosstabs, summary
  statistics, plots, dates, time series, modeling, or performance comparisons;
- advanced MultiIndex work, database operations, remote data, APIs, Drive
  mounting, network downloads, and manual file uploads; and
- assessments of terminal Python. Lecture 04 introduced Jupyter, so this
  assignment uses a notebook even though Assignments 01--03 remain terminal
  assignments.

`reset_index` is permitted only as a bounded structural operation after the
horizontal alignment or pivot. It is not a license to replace label alignment
with positional logic.

## Student repository contract

The implemented student repository must contain exactly this instructional
surface, plus Classroom50-owned metadata outside the protected student files:

```text
06/assignment/
├── .gitignore
├── .python-version
├── PLATFORM_CHECK.md
├── README.md
├── assignment.ipynb
├── check_assignment.py
├── requirements.txt
├── data/
│   ├── fixture.json
│   ├── specimens.csv
│   ├── stations_history.csv
│   ├── specimens_batch_a.csv
│   ├── specimens_batch_b.csv
│   ├── review_scores.csv
│   └── sensor_scores_wide.csv
└── output/
    └── .gitkeep
```

There must be no student-facing data generator, paired Markdown copy of the
notebook, `.github/` grader, grading secrets, or completed output in the starter.
`.gitignore` must ignore notebook checkpoints, caches, and local
virtual-environment directories. It must **not** ignore the five required CSV
artifacts: students need them to appear in VS Code Source Control or GitHub
Desktop so they can commit and submit them. `output/.gitkeep` preserves the
empty starter directory.

### Runtime contract

`.python-version` must contain:

```text
3.12.13
```

`requirements.txt` must contain only the assignment's direct runtime
dependencies:

```text
numpy==2.0.2
pandas==3.0.3
```

Jupyter is the host application, not an assignment import. `pytest`, `nbclient`,
and `nbformat` are central-grader dependencies, not student runtime
dependencies. `PLATFORM_CHECK.md` must distinguish the interpreter launching
Jupyter from the notebook kernel and show students how to verify both. The
notebook kernelspec must be portable Python 3 metadata and must not name a local
environment path.

## Canonical fixture set

All CSV files use UTF-8, comma delimiters, `\n` line endings, the exact column
order shown below, and a final newline. Identifier columns are strings on read.
The manifest's `fixture_set_id` is
`a06-structural-wrangling-v1`. Its provenance text must identify the records as
course-authored synthetic specimens, stations, reviews, and sensor readings;
it must not imply that they are human-subject records.

These are assignment-only fixtures. They must not copy the accepted Lecture 06
demo visit/site/score records, so success cannot come from replaying demo values.

The manifest must record, for every file, the relative path, row grain, row
count, ordered columns, and SHA-256 digest below. The setup cell validates the
manifest and bytes before pandas reads any fixture.

`fixture.json` must use this exact structure and content (JSON indentation is not
part of its semantic contract):

```json
{
  "fixture_set_id": "a06-structural-wrangling-v1",
  "provenance": "Course-authored synthetic specimen, station, review, and sensor records; no human-subject data.",
  "files": [
    {
      "path": "specimens.csv",
      "row_grain": "one row per specimen",
      "row_count": 7,
      "columns": ["specimen_id", "collector_id", "collection_number", "station_code", "material", "mass_g"],
      "sha256": "26eeae8d64a2870dc94195a45f924058b777eb1c97f96d2310e86f06403ba605"
    },
    {
      "path": "stations_history.csv",
      "row_grain": "one row per station-history record",
      "row_count": 5,
      "columns": ["station_code", "station_name", "region", "record_status"],
      "sha256": "dc6f75e588183d5291abd69b4d5aa856472a711f6ff546b015dd21610d55708c"
    },
    {
      "path": "specimens_batch_a.csv",
      "row_grain": "one row per specimen in source partition A",
      "row_count": 4,
      "columns": ["specimen_id", "collector_id", "collection_number", "station_code", "material", "mass_g"],
      "sha256": "1aaa71d01d141bf45dd65ba1ec7c28286536c8ee8aa72834c18bcf0b54af2943"
    },
    {
      "path": "specimens_batch_b.csv",
      "row_grain": "one row per specimen in source partition B",
      "row_count": 3,
      "columns": ["specimen_id", "collector_id", "collection_number", "station_code", "material", "mass_g"],
      "sha256": "8506512a4cef07d7918817e8d8dc15c7230f2923bd28d531326c997995dd58bc"
    },
    {
      "path": "review_scores.csv",
      "row_grain": "one row per reviewed specimen",
      "row_count": 3,
      "columns": ["specimen_id", "review_score"],
      "sha256": "d7a1c9570d463a006cec838a4557581467ffb7459d315f57cbfb3cf73274ad22"
    },
    {
      "path": "sensor_scores_wide.csv",
      "row_grain": "one row per sensor and station pair",
      "row_count": 4,
      "columns": ["sensor_id", "station_code", "baseline_value", "followup_value"],
      "sha256": "6eb9bfb9561fc7c55708bc0038b77b99e6d383f85843dff3e735c5962abe8701"
    }
  ]
}
```

### `specimens.csv`

- Grain: one row per specimen.
- Primary key: `specimen_id`.
- Candidate key: (`collector_id`, `collection_number`).
- Foreign key for the merge: `station_code`.
- Rows: 7.
- SHA-256:
  `26eeae8d64a2870dc94195a45f924058b777eb1c97f96d2310e86f06403ba605`

```csv
specimen_id,collector_id,collection_number,station_code,material,mass_g
SP101,C01,1,R,soil,12.5
SP102,C01,2,R,water,8.0
SP103,C02,1,S,soil,10.5
SP104,C03,1,T,water,9.0
SP105,C04,1,R,soil,11.5
SP106,C05,1,X,air,4.0
SP107,C06,1,S,water,7.5
```

### `stations_history.csv`

- Grain before selection: one row per station-history record.
- Intended lookup grain after the supplied rule: one row per station code.
- Rows: 5.
- SHA-256:
  `dc6f75e588183d5291abd69b4d5aa856472a711f6ff546b015dd21610d55708c`

```csv
station_code,station_name,region,record_status
R,River Station Old,north,retired
R,River Station,north,current
S,Shore Station,south,current
T,Trail Station,west,current
U,Upland Station,east,current
```

The deterministic selection rule is exactly
`record_status == "current"`. It yields unique keys `R`, `S`, `T`, and `U`.
Students must not solve the duplicate with `drop_duplicates`, row order, or an
arbitrary keep-first/keep-last rule.

### `specimens_batch_a.csv`

- Grain: one row per specimen in source partition A.
- Rows: 4.
- SHA-256:
  `1aaa71d01d141bf45dd65ba1ec7c28286536c8ee8aa72834c18bcf0b54af2943`

```csv
specimen_id,collector_id,collection_number,station_code,material,mass_g
SP101,C01,1,R,soil,12.5
SP102,C01,2,R,water,8.0
SP103,C02,1,S,soil,10.5
SP104,C03,1,T,water,9.0
```

### `specimens_batch_b.csv`

- Grain: one row per specimen in source partition B.
- Rows: 3.
- SHA-256:
  `8506512a4cef07d7918817e8d8dc15c7230f2923bd28d531326c997995dd58bc`

```csv
specimen_id,collector_id,collection_number,station_code,material,mass_g
SP105,C04,1,R,soil,11.5
SP106,C05,1,X,air,4.0
SP107,C06,1,S,water,7.5
```

### `review_scores.csv`

- Grain: one row per reviewed specimen.
- Key: `specimen_id`.
- Rows: 3.
- SHA-256:
  `d7a1c9570d463a006cec838a4557581467ffb7459d315f57cbfb3cf73274ad22`

```csv
specimen_id,review_score
SP102,7.0
SP103,9.0
SP108,6.0
```

### `sensor_scores_wide.csv`

- Grain: one row per (`sensor_id`, `station_code`) pair.
- Rows: 4.
- SHA-256:
  `6eb9bfb9561fc7c55708bc0038b77b99e6d383f85843dff3e735c5962abe8701`

```csv
sensor_id,station_code,baseline_value,followup_value
SN01,R,10.0,12.5
SN02,S,8.5,9.0
SN03,T,11.0,13.5
SN04,R,9.5,10.5
```

## Portable setup and fixture validation

The first code cell is supplied and protected. It must:

1. import only the standard library, NumPy, and pandas;
2. search upward from the current working directory for either the flattened
   Classroom50 layout `data/fixture.json` or the course layout
   `06/assignment/data/fixture.json`;
3. define `ASSIGNMENT_ROOT`, `DATA_DIR`, and `OUTPUT_DIR` from the discovered
   manifest, create only `OUTPUT_DIR`, and print the resolved assignment root;
4. require the exact `fixture_set_id`, required manifest keys, required relative
   file set, and safe paths contained within `DATA_DIR`;
5. recompute every fixture's SHA-256 digest and reject a missing, extra, or
   corrupted fixture before any `pd.read_csv` call; and
6. stop with an actionable exception on failure.

The setup must not contain an embedded-data fallback, network request, absolute
workspace path, Google Drive mount, Colab upload prompt, or silent checksum
bypass. The implementation may accept a grader-owned manifest with the same
schema when central tests explicitly invoke student functions on alternate
in-memory tables; it must never rewrite the canonical manifest.

The load cell must use explicit dtypes:

- all identifiers, labels, status, names, regions, and materials: pandas
  `string` dtype;
- `collection_number`: `int64`; and
- mass, review, baseline, and follow-up values: `float64`.

## Notebook contract

The notebook has exactly the following 25 top-level cells in this order. Each
cell has the stable cell ID shown. Supplied/protected cells may contain small
helpers and assertions; student cells contain prompts and `TODO` scaffolds, not
solutions. There are no hidden prerequisite cells.

### Cells 0--3: orientation and load

0. `a06-header` (supplied Markdown): title, learning objectives, local-Jupyter
   requirement, fixture privacy statement, restart-and-run submission rule, and
   exact artifact list.
1. `a06-setup` (supplied protected code): portable path discovery, manifest and
   checksum validation, imports, display settings, output directory creation.
2. `a06-data-contract` (supplied Markdown): compact data dictionary for all five
   tables, distinction between grain and key, and supplied current-record rule.
3. `a06-load` (student code scaffold): explicit-dtype reads into
   `specimens`, `stations_history`, `batch_a`, `batch_b`, `review_scores`, and
   `sensor_scores_wide`; exact shape and column-order assertions.

### Cells 4--10: Task 1, contract-first validated merge

4. `a06-task1-contract` (student Markdown): in the student's own words, state
   specimen and station-history grain, the intended post-selection station
   grain, keys, predicted cardinality, preservation goal, join choice, and the
   consequence of an unmatched foreign key.
5. `a06-contract-values` (student code): assign exactly these machine-checkable
   values:

   ```python
   specimen_grain = "one row per specimen"
   station_history_grain = "one row per station-history record"
   primary_key = ["specimen_id"]
   candidate_key = ["collector_id", "collection_number"]
   foreign_key = ["station_code"]
   predicted_cardinality = "many_to_one"
   preservation_goal = "keep every specimen row"
   join_type = "left"
   predicted_rows = 7
   ```

6. `a06-key-checks` (student code): assert nonmissing/unique specimen primary and
   candidate keys; inspect duplicated right-side station keys; create
   `duplicated_station_rows` containing exactly the two `R` history rows.
7. `a06-duplicate-failure` (student code): attempt the unfiltered explicit-key
   left merge with `validate="many_to_one"`; catch only the relevant pandas merge
   error; set `duplicate_contract_failed = True` and
   `duplicate_error_name = "MergeError"`. The failure must be caused by pandas,
   not assigned or raised manually.
8. `a06-task1-functions` (student code): define
   `select_current_stations(history_table)` and
   `validated_station_merge(specimen_table, station_table)`. The selector applies
   only the supplied status rule and returns the ordered lookup columns
   `station_code`, `station_name`, and `region`; it validates a unique,
   nonmissing station key. The merge function explicitly uses
   `on="station_code"`, `how="left"`, `validate="many_to_one"`, and
   `indicator=True` without mutating either input.
9. `a06-task1-run` (student code): create `current_stations`, rerun key checks,
   create `specimen_merge_audit`, and assert the exact output columns, seven
   preserved specimen IDs, `_merge` counts `both=6`, `left_only=1`,
   `right_only=0`, and one unmatched row `SP106`/`X`. Also assert that unused
   station `U` does not create a row in a left join.
10. `a06-task1-save` (student code): write
    `output/specimen_merge_audit.csv` with `index=False`, read it back with
    explicit dtypes, normalize the serialized `_merge` column to string for the
    comparison, and assert exact frame equality with the in-memory result.

### Cells 11--17: Task 2, concatenation and alignment

11. `a06-task2-contract` (student Markdown): state the batch row grain, explain
    why batch A and B may be stacked, predict the stacked row count, define
    provenance, and distinguish row stacking from index-aligned feature columns.
12. `a06-stack-function` (student code): define
    `stack_specimen_partitions(partition_map)`. It accepts an insertion-ordered
    mapping from source label to DataFrame, copies each input, adds an ordinary
    string `source_partition` column, concatenates on rows with `ignore_index`,
    preserves first-seen column order, and rejects an input that already contains
    the reserved provenance column. It must not mutate inputs or use concat keys
    to create a MultiIndex.
13. `a06-stack-run` (student code): call the function with labels `batch_a` and
    `batch_b`; create `combined_specimens`; assert 4 + 3 = 7 rows, exact source
    counts, exact specimen order `SP101` through `SP107`, a default RangeIndex,
    and equality of all non-provenance columns to `specimens`.
14. `a06-schema-drift` (student code): on disposable copies only, drop `mass_g`
    from batch B and add a `review_note` string column to batch B. Concatenate
    them with the same function and observe exact structural missingness:
    `mass_g` is missing in 3 rows and `review_note` in 4 rows. Do not fill, drop,
    recode, or save this preview. The Task 2 contract at cell 11 predicts
    column-label alignment as the cause, and the final reflection at cell 23
    explains it after observing the result.
15. `a06-align-function` (student code): define
    `align_specimen_features(mass_table, review_table)`. It accepts two tables
    with a unique, nonmissing `specimen_id`, builds named indexes, horizontally
    concatenates the single feature columns with outer label alignment, preserves
    the union's first-seen order, and returns a DataFrame whose index is named
    `specimen_id`. It must not merge, reset before alignment, or align by row
    position.
16. `a06-align-run` (student code): use specimen mass rows `SP101`, `SP102`, and
    `SP103` plus the three review rows. Create `aligned_features`; assert index
    `SP101`, `SP102`, `SP103`, `SP108`; exactly one missing review at `SP101` and
    one missing mass at `SP108`; and exact overlapping values `(8.0, 7.0)` for
    `SP102` and `(10.5, 9.0)` for `SP103`.
17. `a06-task2-save` (student code): write
    `combined_specimens.csv` with `index=False` and `aligned_features.csv` with
    its intentionally named index; read both back with explicit dtypes and assert
    exact equality, including index name and order.

### Cells 18--22: Task 3, nonaggregating reshape

18. `a06-task3-contract` (student Markdown): state wide and long grains; identify
    the wide key (`sensor_id`, `station_code`) and long key (`sensor_id`,
    `station_code`, `measurement_label`); predict eight long rows; explain that
    each long key must identify one value for `pivot` to be reversible.
19. `a06-reshape-functions` (student code): define
    `wide_to_long_scores(wide_table)` and
    `long_to_wide_scores(long_table, ordered_columns)`. The first uses `melt` with
    ID columns `sensor_id`, `station_code`, value columns `baseline_value`,
    `followup_value`, variable name `measurement_label`, and value name `value`.
    The second uses `pivot` with those exact roles, never an aggregating reshaper,
    then restores the caller-supplied column order and original row order by the
    two identifiers. Both functions reject missing or duplicated structural keys
    and do not mutate inputs.
20. `a06-reshape-run` (student code): create `sensor_scores_long` and
    `sensor_scores_round_trip`; assert 8 rows, exact ordered columns, unique
    three-column long key, four rows per measurement label, and exact frame
    equality between round-trip and original wide data including dtypes, row
    order, and column order.
21. `a06-duplicate-pivot` (student code): append a copy of the first long row to a
    disposable long table; show that its duplicated structural-key subset has
    exactly two rows; call the student's wide function; catch the natural
    `ValueError`; set `duplicate_pivot_failed = True`. It must not delete the
    duplicate or aggregate it.
22. `a06-task3-save` (student code): write
    `sensor_scores_long.csv` and `sensor_scores_round_trip.csv` with
    `index=False`; read them back with explicit dtypes and assert exact equality.

### Cells 23--24: synthesis and final verification

23. `a06-reflection` (student Markdown): concise responses to four prompts:
    why the status rule is evidence-based while arbitrary deduplication is not;
    how the left merge implements the preservation goal and exposes the orphan;
    why vertical schema drift and horizontal label mismatch create different
    structural missingness; and why duplicate long keys make structural pivot
    ambiguous and aggregation is deferred.
24. `a06-final-verify` (supplied protected code): independently recheck exact
    canonical invariants and artifact paths, print a single completion summary,
    and instruct the student to restart the kernel and run all cells before
    running `python check_assignment.py`. It must not award points or substitute
    for central grading.

## Exact canonical behavior

### Merge contract

The selected right table has four rows and a unique `station_code`. The validated
left merge has seven rows in original specimen order. Its ordered columns are:

```text
specimen_id, collector_id, collection_number, station_code, material, mass_g,
station_name, region, _merge
```

The result has six `both` rows and one `left_only` row. The only orphan is
`SP106` with station code `X`; its `station_name` and `region` are missing. There
are no `right_only` rows. The selected metadata for every `R` row is
`River Station`, `north`.

Its serialized bytes are:

```csv
specimen_id,collector_id,collection_number,station_code,material,mass_g,station_name,region,_merge
SP101,C01,1,R,soil,12.5,River Station,north,both
SP102,C01,2,R,water,8.0,River Station,north,both
SP103,C02,1,S,soil,10.5,Shore Station,south,both
SP104,C03,1,T,water,9.0,Trail Station,west,both
SP105,C04,1,R,soil,11.5,River Station,north,both
SP106,C05,1,X,air,4.0,,,left_only
SP107,C06,1,S,water,7.5,Shore Station,south,both
```

### Vertical-concat contract

`combined_specimens` has seven rows and the specimen columns followed by
`source_partition`. `SP101`--`SP104` are `batch_a`; `SP105`--`SP107` are
`batch_b`. It has a zero-based RangeIndex. Removing `source_partition` produces
the exact canonical `specimens` frame.

Its serialized bytes are:

```csv
specimen_id,collector_id,collection_number,station_code,material,mass_g,source_partition
SP101,C01,1,R,soil,12.5,batch_a
SP102,C01,2,R,water,8.0,batch_a
SP103,C02,1,S,soil,10.5,batch_a
SP104,C03,1,T,water,9.0,batch_a
SP105,C04,1,R,soil,11.5,batch_b
SP106,C05,1,X,air,4.0,batch_b
SP107,C06,1,S,water,7.5,batch_b
```

The disposable schema-drift preview has seven rows and the first-seen ordered
columns:

```text
specimen_id, collector_id, collection_number, station_code, material, mass_g,
source_partition, review_note
```

It has exactly three missing `mass_g` values and four missing `review_note`
values. Those are alignment observations, not cleaning prompts.

### Horizontal-alignment contract

`aligned_features` has named index `specimen_id`, ordered index
`SP101`, `SP102`, `SP103`, `SP108`, and ordered columns `mass_g`,
`review_score`. Its serialized bytes are:

```csv
specimen_id,mass_g,review_score
SP101,12.5,
SP102,8.0,7.0
SP103,10.5,9.0
SP108,,6.0
```

### Reshape contract

`sensor_scores_long` has ordered columns `sensor_id`, `station_code`,
`measurement_label`, `value`. Melt's canonical order is the four baseline rows
in original sensor order followed by the four follow-up rows in original sensor
order. The exact bytes are:

```csv
sensor_id,station_code,measurement_label,value
SN01,R,baseline_value,10.0
SN02,S,baseline_value,8.5
SN03,T,baseline_value,11.0
SN04,R,baseline_value,9.5
SN01,R,followup_value,12.5
SN02,S,followup_value,9.0
SN03,T,followup_value,13.5
SN04,R,followup_value,10.5
```

The round-trip is byte-for-byte identical to `sensor_scores_wide.csv`. Adding a
copy of the first long row makes two rows visible in the duplicated-key subset
and causes the structural pivot to raise `ValueError`.

## Artifact contract

All five artifacts are deterministic, replace stale files, and use UTF-8,
comma delimiters, `\n` line endings, ordered columns, and a final newline.
Four use `index=False`; `aligned_features.csv` intentionally serializes its
named index. Under CPython 3.12.13, NumPy 2.0.2, and pandas 3.0.3, the canonical
digests are:

| Artifact | Rows | SHA-256 |
| --- | ---: | --- |
| `specimen_merge_audit.csv` | 7 | `1bc33aeecbae2483e314399784bbcaf8b8847798fe3ca5b7662908053615e98c` |
| `combined_specimens.csv` | 7 | `78cbd883bea393fb84d699cdb9923a9d71d7c045d002eebe88cc84c9da61c666` |
| `aligned_features.csv` | 4 | `19cb5d07f7ae51ce0347876802a44eadf48490076bb24dbdcae547d9388775e7` |
| `sensor_scores_long.csv` | 8 | `989affb14d49ecd0e144e23a6b53ab4a093edd6211656390144869ecaa3126dd` |
| `sensor_scores_round_trip.csv` | 4 | `6eb9bfb9561fc7c55708bc0038b77b99e6d383f85843dff3e735c5962abe8701` |

Canonical in-memory dtypes are:

- merge: string identifiers/labels/metadata, `collection_number` `int64`,
  `mass_g` `float64`, and pandas categorical `_merge`;
- combined: the specimen dtypes plus string `source_partition`;
- aligned: named string index and two `float64` columns;
- long: three string columns and one `float64` value; and
- round-trip: two string identifier columns and two `float64` value columns.

Because CSV does not preserve categoricals, readback checks compare `_merge` as
normalized string after verifying its exact allowed labels. Central grading must
validate both behavior and values rather than relying only on file hashes.

## Protected and student-editable surfaces

Classroom50's assignment template must record implementation-time hashes for the
following protected files and surfaces:

- `.python-version`, `requirements.txt`, `.gitignore`, `README.md`, and
  `PLATFORM_CHECK.md`;
- `check_assignment.py`;
- `data/fixture.json` and all six canonical CSV fixtures; and
- notebook cells `a06-header`, `a06-setup`, `a06-data-contract`, and
  `a06-final-verify`, including their stable cell IDs and source.

Only the designated student Markdown/code cells in `assignment.ipynb` and the
five regenerated CSVs are student work products. `output/.gitkeep` is template
structure, not an answer. The public checker must report protected edits, and
the central grader must independently enforce them; an editable public checker
cannot weaken that enforcement. Staff must regenerate the recorded hashes when
they intentionally publish a new assignment version rather than asking students
to bypass a mismatch.

## Student-visible checker

`check_assignment.py` must be dependency-free apart from the Python standard
library so it can explain setup failures before importing the notebook. It must:

- validate the canonical manifest, fixture inventory, safe relative paths,
  byte hashes, schemas, row counts, and final newlines;
- parse the notebook as JSON and require the 25 stable cell IDs in exact order,
  portable kernelspec metadata, valid cell structure, and no missing cells;
- protect supplied cell sources by course-owned hashes and report edited cells by
  ID;
- detect untouched `TODO` scaffolds and missing required function names;
- reject source use of banned later-scope APIs and obvious absolute paths,
  network access, Drive mounts, upload calls, or embedded fixture fallbacks;
- validate the five artifact paths, CSV headers, row order, exact canonical
  values, missing-value positions, and intended index serialization; and
- return nonzero with actionable messages for starter, partial, corrupt, and
  malformed submissions, or print a concise readiness message on success.

The public checker must not execute arbitrary notebook code, trust notebook
outputs, read grader secrets, claim a grade, or serve as the production grading
authority. Source scanning is only an early warning; central behavioral tests
decide correctness.

## Classroom50 central grader

The production grader is a teacher-controlled, discoverable Classroom50 bundle,
separate from the student repository and documented for future maintainers. It
contains no student credentials or opaque external dependency. It must:

1. verify protected fixture and notebook-cell hashes independently of
   `check_assignment.py`;
2. copy the submission to an isolated temporary directory, remove generated CSV
   artifacts, strip notebook outputs and execution counts, append a grader-owned
   cell, and execute the notebook from a fresh kernel in the pinned environment;
3. test both flattened Assignment 06 repositories and the course-root layout,
   including relocation to a path containing spaces;
4. assert every canonical namespace value, natural failure flag, function
   signature, DataFrame schema, dtype, key invariant, row-preservation invariant,
   unmatched-key invariant, and exact regenerated artifact;
5. independently call the six reusable student functions on discoverable
   alternate in-memory tables that vary IDs, row order, values, row counts,
   partition labels, overlapping indexes, and measurement values;
6. include alternate merge data with more than one orphan and an unfiltered
   duplicate right key; verify that the selector follows status rather than row
   order and that the validated merge rejects invalid cardinality;
7. include alternate partitions with reordered columns and schema differences;
   verify first-seen alignment, provenance, input immutability, and ordinary
   columns rather than MultiIndex keys;
8. include alternate horizontal tables with asymmetric index overlap; verify
   label-based outer alignment and named index preservation;
9. include alternate wide data with different identifier values and row order;
   verify exact reversible melt/pivot behavior and rejection of duplicate long
   keys;
10. run once with a missing fixture and once with a corrupted fixture and require
    setup to stop before analysis;
11. seed stale output files, confirm replacement, rerun, and verify deterministic
    results; and
12. write `./result.json`, exit zero when grading completed even if tests fail,
    and emit the official `classroom50/result/v1` shape: required submission
    metadata; total `score` and hyphenated `max-score`; and per-test
    `test-name`, `passed`, `score`, and `max-score`. Concise failure detail belongs
    in the grading log or release body, not in an incompatible result schema.

The grader must not infer correctness from hard-coded canonical output alone.
Alternate behavioral calls must be discoverable from the student prompt; hidden
tests may vary data but may not introduce an undisclosed API or domain rule.

### Adversarial QA matrix

Before release, course staff must exercise at least these cases:

- untouched starter, correct solution, partial solution, and corrected
  resubmission;
- stored correct-looking outputs with broken or unexecuted code;
- malformed notebook JSON, deleted/reordered/duplicated cell IDs, missing cell,
  and edited protected setup or verification cells;
- missing, extra, renamed, or byte-corrupted fixture and edited manifest;
- implicit merge keys, inner/right/outer join, missing `validate`, wrong
  `validate="one_to_one"`, missing indicator, or indicator dropped before audit;
- arbitrary `drop_duplicates`, keep-first/last dependence, retired station
  selected, orphan lost, right-only row added, duplicate-expanded merge, and
  canonical IDs or row counts hard-coded inside reusable functions;
- concat on the wrong axis, provenance in an index, provenance inferred by row
  position, input mutation, reordered output, or schema gaps filled/dropped;
- horizontal inputs reset before alignment, positional concatenation, wrong join
  of indexes, unnamed/serialized extra index, or overlapping rows hard-coded;
- melt with missing ID variables, wrong value variables or labels, wrong row
  count/order, or loss of dtype;
- use of `pivot_table`, GroupBy, or another aggregation; silently accepted or
  deleted duplicate long key; incorrect round-trip order or schema;
- Lecture 05 cleaning actions applied to structural missingness;
- plots, dates, time grouping, modeling, remote downloads, absolute paths,
  content-dependent path guesses, Drive mounts, or manual uploads; and
- stale artifacts, repeat execution, alternate working directories, flattened
  Classroom50 checkout, relocated checkout, and a path containing spaces.

## Assessment boundary and provisional scoring

The behavior should remain stable if syllabus weights change. For review and
grader design, use this provisional 100-point diagnostic allocation:

| Area | Automated | Human | Total |
| --- | ---: | ---: | ---: |
| Task 1: contracts and validated diagnostic merge | 40 | 5 | 45 |
| Task 2: vertical concat and horizontal alignment | 27 | 3 | 30 |
| Task 3: reversible structural reshape | 23 | 2 | 25 |
| **Total** | **90** | **10** | **100** |

Automation owns environment/fixture integrity, natural failure evidence,
function behavior, nonmutation, keys, schemas, dtypes, exact invariants,
artifacts, portability, and deterministic reruns. The human review is limited to:

- whether grain, key, cardinality, and preservation reasoning is coherent;
- whether the current-record rule is distinguished from arbitrary
  deduplication;
- whether structural missingness from the two alignment modes is explained;
- whether duplicate pivot ambiguity and the no-aggregation boundary are
  explained; and
- whether the notebook is readable, concise, and contains no identifying data.

The revised syllabus must still decide whether the historical regular-assignment
competence/pass-fail policy remains, how this diagnostic maps to that policy, the
pass threshold if any, grade conversion, and late/resubmission rules. No such
policy may be encoded in the notebook or public checker before that decision.

## Full legacy disposition

Implementation must make these explicit replacements rather than preserving
contradictory paths:

- rewrite `06/assignment/README.md` to describe the single notebook, exact local
  setup, fixture contract, artifacts, public check, restart-and-run procedure,
  local-Jupyter requirement, Classroom50 submission surface, and conditional
  Colab status;
- replace `06/assignment/assignment.ipynb` with the 25-cell notebook above;
- delete `06/assignment/assignment.md`;
- delete `06/assignment/data_generator.ipynb` and
  `06/assignment/data_generator.md`;
- delete the entire legacy `06/assignment/.github/workflows/` and
  `06/assignment/.github/test/` trees, including their separate requirements;
- replace broad dependency lower bounds with the exact environment contract;
- remove all legacy generated customer, product, purchase, pricing, store,
  satisfaction, loyalty, quarterly, category, date, and month data/output names;
- remove the obsolete `q1_validation.txt` contract and every random-data path;
- remove GroupBy, aggregation, `pivot_table`, date parsing, monthly report, and
  plotting requirements from Assignment 06; and
- add only the repository surface and canonical fixtures specified above.

Classroom50 course configuration and its central grader live in the course-wide
platform area, not in `.github/` and not as a second assignment implementation.
No classroom export is part of this work.

## Implementation gate

The specimen/station domain, supplied `record_status == "current"` rule, exact
five-artifact contract, and six reusable function interfaces are accepted for
technical implementation. The 90/10 automation-human review boundary is an
explicitly provisional grader design and must remain separable from the fixture,
task, and artifact behavior until the syllabus resolves course policy.

After implementation, a separate reviewer must run the public-check self-tests,
central grader self-tests, fresh-kernel canonical and alternate-data executions,
fixture corruption tests, relocation tests, and scope scan. This blueprint does
not self-certify those future results.

Production-contract correction (2026-07-19): Classroom50 invokes the teacher
bundle's standard-library `autograder.py` with plain Python; it installs exact
sibling requirements before importing the central grader. The accepted student
repository may additionally contain only delivery-owned `.classroom50.yaml` and
`.github/workflows/autograde.yaml`; only the top-level `.git/**` repository
metadata tree is ignored, while every other root/workflow/grader-tree file,
including a nested `ordinary/.git/**` tree, is rejected.
