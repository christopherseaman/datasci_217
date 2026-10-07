# Assignment 07 checks

The report prints one `PASS` or `FIX` line per check with its points. Before Task 1, for example, the first check reports:

```text
[FIX ]  0/4  exploratory spec: point mark
         output/exploratory_spec.json is missing; run the Task 1.1 cell to write it, then commit it.
```

A clean run ends with:

```text
[PASS]  2/2  text alternative file

Score: 100/100
All checks passed.
```

How the files are read:

- Each check is scored on its own, so one mistake costs only that check's points.
- Spacing, line endings, quoting, key order, and column order never cost points, and extra keys in the JSON files are ignored.
- Numbers are compared as numbers, so `79`, `79.0`, and `79.00` are the same value.
- Program names, categories, keys, column names, and data types are compared in any letter case. A data type may carry a note, as in `ordinal (visit order)`. `nominal` and `qualitative` count as categorical, `numeric` and `continuous` as quantitative, and `ordered` as ordinal, alone or beside categorical, as in `categorical (ordered)`.
- A leading column of row numbers, which `to_csv()` writes when `index=False` is left out, is ignored.
- The exploratory spec is compared on the three plotted columns only, so leaving out `patient_id` or adding a column costs nothing.
- The PNG checks confirm that each chart was saved as a PNG image that is not blank, and the text checks confirm that each answer is filled in. The wording and the look are yours, so open both PNG files and check them against Tasks 2.3 and 3.2 yourself.

Every push also runs GitHub Actions, which downloads the course's current copy of the checks and reruns them on the files you committed and pushed. That run is what counts, and a check corrected after handout reaches you there on your next push.

## Completion contract

Grading totals 100 points and reads these files relative to the assignment root.

| Artifact | Complete when | Check | Points |
| --- | --- | --- | ---: |
| `output/exploratory_spec.json` | Its mark is `point`. | exploratory spec: point mark | 4 |
| `output/exploratory_spec.json` | Its embedded rows hold the twelve patients' `program`, `sessions_attended`, and `walk_distance_m`. | exploratory spec: embedded patient rows | 5 |
| `output/exploratory_spec.json` | It encodes `sessions_attended` as quantitative `x`. | exploratory spec: x encoding | 4 |
| `output/exploratory_spec.json` | It encodes `walk_distance_m` as quantitative `y`. | exploratory spec: y encoding | 4 |
| `output/exploratory_spec.json` | It encodes `program` as nominal `color`. | exploratory spec: color encoding | 4 |
| `output/exploratory_spec.json` | It encodes `program` as nominal `shape`. | exploratory spec: shape encoding | 4 |
| `output/critique_redesign.png` | It is a PNG image, not a blank one. | critique redesign: PNG image | 12 |
| `output/visualization_evidence.json` | Its `critique` has an `unsupported claim` entry with a `problem` and a `repair`. | critique: unsupported claim | 5 |
| `output/visualization_evidence.json` | Its `critique` has a `truncated baseline` entry with a `problem` and a `repair`. | critique: truncated baseline | 5 |
| `output/visualization_evidence.json` | Its `critique` has a `missing unit` entry with a `problem` and a `repair`. | critique: missing unit | 5 |
| `output/visualization_evidence.json` | Its `critique` has a `color-only encoding` entry with a `problem` and a `repair`. | critique: color-only encoding | 5 |
| `output/visualization_evidence.json` | Its `critique` has a `distracting decoration` entry with a `problem` and a `repair`. | critique: distracting decoration | 5 |
| `output/explanatory_chart.png` | It is a PNG image, not a blank one. | explanatory chart: PNG image | 8 |
| `output/explanatory_supporting_data.csv` | Its columns are `program`, `visit_number`, and `goal_met_pct`. | supporting data: columns | 4 |
| `output/explanatory_supporting_data.csv` | It holds the eight rows of `data/followup_goals.csv`, unchanged. | supporting data: rows and values | 6 |
| `output/visualization_evidence.json` | Its `question` is filled in. | evidence: question | 2 |
| `output/visualization_evidence.json` | Its `audience` is filled in. | evidence: audience | 2 |
| `output/visualization_evidence.json` | Its `intended_claim` is filled in. | evidence: intended_claim | 2 |
| `output/visualization_evidence.json` | Its `y_measure` is filled in. | evidence: y_measure | 2 |
| `output/visualization_evidence.json` | Its `grain` is filled in. | evidence: grain | 2 |
| `output/visualization_evidence.json` | Its `text_alternative` is filled in. | evidence: text_alternative | 2 |
| `output/visualization_evidence.json` | Its `data_types` gives `program` as categorical. | data type: program | 2 |
| `output/visualization_evidence.json` | Its `data_types` gives `visit_number` as ordinal. | data type: visit_number | 2 |
| `output/visualization_evidence.json` | Its `data_types` gives `goal_met_pct` as quantitative. | data type: goal_met_pct | 2 |
| `output/explanatory_text_alternative.txt` | It holds the same text as `text_alternative` in the JSON. | text alternative file | 2 |

Extra files are ignored.
