# Lectures 08–11 content and alignment review

> Historical review snapshot. Any Classroom50 or GitHub Classroom language
> below records a superseded intermediate delivery plan, not current course
> policy. The release design is one repository per assignment with repo-local
> pytest/GitHub Actions; no Classroom service is used.

Status: evidence-backed design proposal; no lecture, demo, assignment, or grader source has been edited.

Reviewed scope:

- Lecture, bonus, demo, assignment, dependency, and current public-test files under 08/ through 11/.
- The course contracts in work/lecture_review_workflow.md, work/course_dependency_alignment.md, and work/colab_standard.md.
- Current pandas behavior where the repository permits an unbounded package upgrade.

The matrices below describe the alignment of the current artifacts against the proposed measurable objectives. A proposed objective is not treated as already taught merely because a current notebook contains code that uses it.

## Range decision

Lectures 08–11 should form one continuous reasoning chain:

1. Lecture 08 changes row grain deliberately through grouping and explains the resulting shape.
2. Lecture 09 adds ordered time and entity boundaries to those grouped operations.
3. Lecture 10 adds a question, target or estimand, information boundaries, and valid evaluation.
4. Lecture 11 integrates those capabilities without introducing another model family or a new analysis framework.

The current range has four release blockers:

- Lecture 08 gives an incorrect history of DataFrameGroupBy.apply behavior while allowing any pandas version at or above 2.0 (08/README.md:166-178; 08/demo/requirements.txt:1). Pandas 3 changes the default and no longer permits include_groups=True.
- Lecture 09 contains multiple examples that create a Series and then assign or select named columns, so the published examples cannot execute as written (09/README.md:233-245, 466-478, 533-544).
- Lecture 10 teaches test-set isolation and then uses the test set for XGBoost early stopping and final reporting (10/README.md:268-279, 599-615; 10/demo/demo2_ml_boosting.md:394-428).
- Lecture 11 requires feature engineering before students define the prediction target or horizon and cleans multi-station time-series data before establishing station/time order (11/assignment/q2_data_cleaning.md:22-35; 11/assignment/q4_feature_engineering.md:31-35; 11/assignment/q6_modeling_preparation.md:128-168).

## Lecture 08: aggregation with an explicit result grain

### Intended role

Teach students to state what one input row and one output row represent, then use GroupBy aggregation, transform, and one pivot without silently changing the analysis unit.

### Proposed measurable objectives

By the end of Lecture 08, a student can:

1. Given a DataFrame and one or more grouping keys, state the input row grain, predict the number and identity of output groups, and verify the result.
2. Produce a flat, explicitly named summary table with GroupBy named aggregation, choosing size, count, or nunique to match the question.
3. Distinguish aggregation from transform by output grain and shape, and use transform to add a group statistic without changing row count or index.
4. Produce and interpret one pivot table, identifying its index, columns, values, aggregation function, missing combinations, and totals.

These replace the demo guide's non-measurable objectives to “master” advanced GroupBy, remote computing, and performance optimization (08/demo/DEMO_GUIDE.md:38-43).

### Exact incoming prerequisites

| Required on entry | Source capability | Why Lecture 08 needs it |
|---|---|---|
| Select columns and rows; distinguish DataFrame index from columns | Lecture 04 exit contract | GroupBy selection and result-index reasoning depend on it. |
| Recognize missing values and choose whether they should contribute to a summary | Lecture 05 exit contract | count, size, nunique, and pivot missing combinations have different meanings. |
| State row grain, identify keys, validate join cardinality, and verify post-merge grain | Lecture 06 exit contract | The current assignment merges three tables before aggregation (08/assignment/assignment.md:40-60). |
| Read long and wide tabular forms | Lecture 06 exit contract | A pivot is an aggregated reshape, not a new data type. |
| Make or read one honest chart | Lecture 07 exit contract | Charts may communicate an aggregate, but visualization is not a new Lecture 08 objective. |

Lecture 08 should state these prerequisites explicitly. The current assignment lists only “groupby operations, pivot tables, and aggregation functions from Lecture 08” (08/assignment/README.md:69), so it does not warn students that the merge requires Lecture 06 grain/cardinality skills.

### Exact outgoing prerequisites

Lecture 09 may assume that a student can:

- define a grouping key and the unit represented by each group;
- distinguish an input row from an aggregated output row;
- distinguish aggregation, which reduces or changes grain, from transform, which returns one value per input row;
- make grouped results explicit with named columns and a deliberate index;
- explain size versus count versus nunique;
- read one pivot as an aggregated table.

Lecture 09 may not assume that students have already learned periods, frequency, resampling, or grouped rolling operations. Current Lecture 08 demos use Date.dt.to_period('M') before Lecture 09 establishes period/frequency semantics (08/demo/demo1_groupby_operations.md:175-203; 08/demo/demo2_pivot_tables.md:149-183).

### Current alignment matrix

| Proposed objective | Current lecture section | Current required demo | Current assignment | Current test/rubric | Status |
|---|---|---|---|---|---|
| 8.1 State and verify input/output grain | Split–apply–combine is defined at 08/README.md:22-67, but “unit of analysis,” input grain, output grain, and expected group count are absent. | Demo 1 performs many groupings but does not require a before/after grain prediction (08/demo/demo1_groupby_operations.md:42-68). | Students merge encounter, provider, and facility data and immediately summarize provider experience and encounter charges (08/assignment/assignment.md:53-78). The encounter join repeats provider attributes, so a mean/count can answer the wrong question. | Q1 checks only that a nonempty CSV has a facility_name column (08/assignment/.github/test/test_assignment.py:27-32). | **partially aligned; P0 grain defect** |
| 8.2 Named aggregation and correct counting semantics | Basic operations introduce mean, sum, count, size, and dictionary aggregation (08/README.md:69-103), but not named aggregation or a question-driven choice among count/size/nunique. | Demo 1 uses dictionary aggregation and later a time-based grouping (08/demo/demo1_groupby_operations.md:48-68, 175-203). | Q1 asks for mean, sum, and count of years_experience after an encounter-level merge (08/assignment/assignment.md:69-78); this can count encounters rather than unique providers. | No expected totals, group keys, column names beyond facility_name, or unique-provider invariant are checked (08/assignment/.github/test/test_assignment.py:27-45). | **untaught but assessed** |
| 8.3 Aggregation versus transform | Transform is defined as same-shape at 08/README.md:108-129, but it is not contrasted with aggregate output grain or index. | Demo 1 uses transform, filter, apply, MultiIndex, dates, and performance in one notebook (08/demo/DEMO_GUIDE.md:8-17; 08/demo/demo1_groupby_operations.md:76-118). | Q1 requires transform-generated means, standard deviations, normalization, and totals (08/assignment/assignment.md:81-92). | The transform output is not tested; the grader only checks the separately written report exists and contains a label (08/assignment/.github/test/test_assignment.py:34-45). | **partially aligned** |
| 8.4 One interpretable pivot | Pivot tables are motivated and demonstrated at 08/README.md:240-299, followed immediately by advanced options and crosstab (08/README.md:301-330). | Demo 2 includes multiple aggregations, totals, missing-value fill, crosstabs, time periods, several analyses, and visualization (08/demo/demo2_pivot_tables.md:55-143, 145-228). | Q3 requires pivot, crosstab, totals, missing handling, custom aggregation, and visualization (08/assignment/README.md:117-128). | Tests verify only that CSVs are readable and the PNG is nonempty (08/assignment/.github/test/test_assignment.py:92-124). | **partially aligned** |

### Current high-priority findings

- The apply compatibility explanation reverses the history. In pandas 2.2, including grouping columns was deprecated; pandas 3 defaults to include_groups=False and disallows True. The current text says old pandas excluded groups and pandas 2.2 began including them (08/README.md:166-178). The official pandas 3 apply reference and pandas 2.2 release notes confirm the opposite:
  - https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.api.typing.DataFrameGroupBy.apply.html
  - https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v2.2.0.html
- The paired demo still calls DataFrameGroupBy.apply without include_groups=False (08/demo/demo1_groupby_operations.md:101-118), and the assignment does the same (08/assignment/assignment.md:120-138). Because requirements specify pandas>=2.0.0 with no upper bound (08/demo/requirements.txt:1; 08/assignment/requirements.txt:1), behavior and warnings depend on the installation date.
- Hierarchical grouping and MultiIndex are advertised as bonus at 08/README.md:3-9 but are also required core material and assessed in Question 2 (08/README.md:209-238; 08/assignment/README.md:105-115).
- Remote computing, SSH, screen, tmux, multiprocessing, chunking, and profiling occupy the second half of the lecture and a required demo (08/README.md:335-544; 08/demo/DEMO_GUIDE.md:28-43). They do not support the aggregation objectives.
- The remote demo prints simulated shell commands inside Python rather than giving students a real remote-computing capability (08/demo/demo3_remote_performance.md:138-203).
- The assignment is local-environment-first and assumes local data paths (08/assignment/README.md:7-58). The redesigned analysis notebook must be portable and clean-local-Jupyter executable; Colab becomes an assignment path only after repository-save and submission validation is approved. A separate optional SSH/tmux lab, if retained, remains terminal-only because the terminal tool is its objective.

### Recommended required demos

All three analysis demos use a small deterministic dataset, run top-to-bottom in a fresh Colab runtime, and also run in local Jupyter.

1. **Predict the grouped result**
   - Display a small table with a stated row grain.
   - Ask students to identify grouping keys and predict group count before executing.
   - Compare size, count, and nunique with a deliberately missing value and a repeated entity.
   - Produce a flat named-aggregation result and verify its totals.

2. **Aggregate versus transform**
   - Use the same table and the same grouping key.
   - Show that aggregation produces one row per group.
   - Add a within-group mean and deviation with transform and assert unchanged length/index.
   - Include one wrong result whose grain students diagnose.

3. **From long records to one pivot**
   - State index, columns, values, and aggregation before the call.
   - Compare the pivot to an equivalent GroupBy result.
   - Include one missing combination and distinguish “no row” from a measured zero.
   - End with one already-familiar Lecture 07 chart only if it clarifies the table.

### Assignment redesign contract

| Contract element | Required design |
|---|---|
| Format | One portable restartable notebook; clean local-Jupyter execution is required. Colab becomes required only if the course-level assignment workflow is approved; otherwise it remains a compatibility target. |
| Data | A committed, deterministic, small table whose row grain is one healthcare encounter. Either supply the analysis table already joined or require a validated Lecture 06 merge with explicit pre/post row-count checks. |
| Task 1 | State input grain and grouping keys; create a facility summary with total charges, encounter_count using size, nonmissing_measurement_count using count where appropriate, and unique_provider_count using nunique. |
| Task 2 | Add one group statistic to encounter rows with transform; verify unchanged row count and index; explain why this is not an aggregation. |
| Task 3 | Create one pivot with an explicit aggregation and interpret one missing combination and one total. |
| Student evidence | Notebook reasoning cells plus three compact machine-readable outputs: group_summary.csv, transformed_rows.csv, and pivot_summary.csv. |
| Automated checks | Recompute exact summaries from the pinned fixture; check named schema, group keys, total conservation, unique-provider count, unchanged transform row count/index, pivot cell values, and restart-and-run-all. |
| Human check | Read the grain statements and missing-combination interpretation. |
| Classroom 50 rule | Public tests are treated as discoverable; correctness depends on behavioral invariants, not hidden filenames or secret expected prose. |
| Excluded | GroupBy.apply, group filtering, MultiIndex manipulation, crosstab, performance reports, remote commands, and a new visualization objective. |

### Content disposition

| Current material | Action | Evidence and rationale |
|---|---|---|
| Split–apply–combine visual and basic GroupBy | **Keep/revise** | Strong conceptual anchor at 08/README.md:22-103; add row-grain prediction and named outputs. |
| count versus size | **Keep/revise** | Both appear at 08/README.md:77-79; add nunique and question-based selection. |
| transform | **Keep/reorder** | Same-shape definition at 08/README.md:108-129 belongs directly after aggregation. |
| filter and apply | **Move to bonus** | They expand the API before students can reliably predict output shape (08/README.md:131-205). Prefer agg/transform first; pandas itself recommends specific methods over apply. |
| Hierarchical grouping and MultiIndex operations | **Move to bonus** | Current core/bonus contradiction at 08/README.md:3-9, 209-238 and assignment requirement at 08/assignment/README.md:105-115. |
| One pivot_table | **Keep/revise** | Retain 08/README.md:240-299 but tie it to equivalent GroupBy and explicit output grain. |
| Advanced pivot options and multidimensional crosstab | **Move to bonus** | Current scope at 08/README.md:301-330 and 08/demo/demo2_pivot_tables.md:69-143 is not required for the handoff. |
| Time-period groupings inside demos | **Move later** | Date.dt.to_period('M') appears before Lecture 09 definitions (08/demo/demo1_groupby_operations.md:175-203; 08/demo/demo2_pivot_tables.md:149-183). |
| SSH, scp, screen, tmux | **Move to a separate optional local-terminal bonus lab** | The material at 08/README.md:335-467 is operationally useful but not an aggregation prerequisite and cannot be a Colab demo. |
| Chunking, multiprocessing, profiling, memory tuning | **Move to bonus/reduce** | Broad performance scope at 08/README.md:468-544 and 08/demo/demo3_remote_performance.md:23-136, 205-340. |
| Simulated remote-computing notebook | **Drop** | Printing commands and sleeping does not teach or validate remote operation (08/demo/demo3_remote_performance.md:138-203). |

### Ordered content outline

1. Reconnect row grain and keys from Lecture 06.
2. Define grouping key, group, GroupBy object, aggregation, and output grain.
3. Predict groups and output rows before executing.
4. Single-key GroupBy and one aggregation.
5. size, count, and nunique.
6. Named aggregation and flat output columns.
7. Multi-key grouping with explicit result index/columns.
8. Aggregation versus transform.
9. One pivot_table as an aggregated reshape.
10. Missing combinations, totals, and verification invariants.
11. Bonus boundary: filter/apply, MultiIndex, advanced pivots, remote/performance.

## Lecture 09: time structure, entity boundaries, and safe windows

### Intended role

Teach students to represent ordered temporal data correctly and perform resampling, lags, differences, and trailing windows without pooling entities accidentally or using future information.

### Proposed measurable objectives

By the end of Lecture 09, a student can:

1. Classify a dataset as timestamp- or period-based, regular or irregular, and single-series or panel; state the row grain and sort keys.
2. Parse timestamps, distinguish naive from timezone-aware values, localize or convert one series correctly, and create a sorted datetime index within entity.
3. Use asfreq or resample with an aggregation justified by measurement meaning, preserving entity boundaries and explaining newly introduced missing values.
4. Create a lag, difference, and trailing observation-count or elapsed-time window without crossing entity boundaries or using future observations.
5. State what information is available at a prediction timestamp, reject centered or future-derived features, and construct a plausible chronological holdout for Lecture 10.

### Exact incoming prerequisites

| Required on entry | Source capability | Why Lecture 09 needs it |
|---|---|---|
| Select, sort, index, and save pandas data | Lecture 04 exit contract | DatetimeIndex operations build on ordinary pandas indexing. |
| Reason about missing values and imputation consequences | Lecture 05 exit contract | Resampling and regularization create structural missing values. |
| State row grain and identify candidate/grouping keys | Lectures 06 and 08 exit contracts | Lecture 09 defines entity, entity key, and entity-plus-timestamp ordering before panel operations. |
| Group and aggregate with explicit output grain | Lecture 08 exit contract | Resample is time-based grouping and grouped resampling inherits GroupBy shape/index behavior. |
| Create a clear line chart with labels | Lecture 07 exit contract | Lecture 09 adds temporal semantics, not foundational plotting syntax. |

### Exact outgoing prerequisites

Lecture 10 may assume that a student can:

- distinguish a single series from a panel of entity-specific series;
- identify the timestamp associated with each row and sort by entity plus timestamp;
- state whether a series is regular or irregular and what a frequency means;
- distinguish asfreq from resample;
- create lags, differences, and trailing windows that use only current/past information;
- explain why centered windows and future values are unavailable for prediction at time t;
- make a chronological holdout plausible, although model-validation terminology is introduced in Lecture 10.

### Current alignment matrix

| Proposed objective | Current lecture section | Current required demo | Current assignment | Current test/rubric | Status |
|---|---|---|---|---|---|
| 9.1 Classify timestamp/period, regular/irregular, single/panel, and row grain | Regular/irregular and other “types” appear at 09/README.md:30-51; timestamp versus period and single versus panel are not defined. | Demo 1 covers datetime, frequency, shift, and timezone in one long notebook; Demo 3 creates a two-site panel (09/demo/demo3_visualization_automation.md:63-89) without first defining panel semantics. | Q1 has 200 patients and Q2 has 75 ICU patients (09/assignment/q1_datetime.md:42-66; 09/assignment/q2_resampling.md:40-72), but tasks often collapse them. | Q1 validates per-patient days_since_start minima (09/assignment/.github/test/test_assignment.py:49-61), but no test checks sort order, row grain, or panel boundaries. | **partially aligned** |
| 9.2 Parse, sort, localize, and convert timestamps | Datetime parsing/index/sort are shown at 09/README.md:98-145; timezone operations are a separate late section (09/README.md:549-594). | Demo 1 practices localization/conversion. | Q1 requires a long timezone report and DST discussion (09/assignment/q1_datetime.md:125-167). | The grader checks only that the report exceeds 50 characters (09/assignment/.github/test/test_assignment.py:63-79). | **partially aligned** |
| 9.3 asfreq/resample with entity and missingness semantics | Frequency inference and asfreq precede shifting (09/README.md:185-211); resampling is defined later at 09/README.md:339-403. Entity boundaries are not integrated. | Demo 2 resamples pooled ICU rows, and calls forward fill simpler but interpolation “more sophisticated” (09/demo/demo2_indexing_resampling.md:118-177), an unreliable decision rule. | Q2 resamples all patients together (09/assignment/q2_resampling.md:143-160), then averages patients by date for comparison (09/assignment/q2_resampling.md:173-202), changing the estimand without naming it. | Tests check three frequency labels and column presence, not expected grouped values or entity preservation (09/assignment/.github/test/test_assignment.py:99-113). | **partially aligned; P0 panel-grain defect** |
| 9.4 Lag/difference/trailing windows without leakage | The lag example creates a Series then assigns named columns (09/README.md:233-245); rolling repeats the same invalid pattern (09/README.md:466-478). Observation-count versus elapsed-time windows is not defined. | Demo 2 repairs the container by using DataFrames but treats pooled daily ICU means as one patient-like series (09/demo/demo2_indexing_resampling.md:179-215). | Q3 averages all patients by date and calls that handling missing data “naturally” (09/assignment/q3_rolling.md:40-78), then requires centered/custom/expanding/EWM operations (09/assignment/q3_rolling.md:121-168). | Tests require 365 rows and only one “basic” plus one “advanced” column, not the correct rolling values or absence of future/entity leakage (09/assignment/.github/test/test_assignment.py:144-177). | **untaught but assessed** |
| 9.5 Information availability and chronological handoff | Information availability is not defined, and centered windows appear without a prediction-time warning (09/README.md:481-512). | Required demos do not establish a timestamp cutoff or contrast past-only with future-derived features. | Q3 requires centered/custom windows without asking whether their inputs would exist at prediction time (09/assignment/q3_rolling.md:121-168). | Tests do not reject future leakage or check a chronological cutoff. | **untaught but assessed** |

### Current high-priority findings

- The lecture calls uppercase H the canonical hourly alias (09/README.md:153-162), while pandas 2.2 deprecated H and pandas 3 removed it in favor of h. Requirements permit pandas 3 because they specify pandas>=2.0.0 without an upper bound (09/demo/requirements.txt:1; 09/assignment/requirements.txt:1). Official release evidence:
  - https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v2.2.0.html
  - https://pandas.pydata.org/pandas-docs/version/3.0/whatsnew/v3.0.0.html
- Quarterly and annual examples still use Q and A at 09/README.md:355-362; pandas 3 replaces those end-frequency aliases with QE and YE.
- Shifting appears before time-series selection and before a stable single/panel distinction (09/README.md:213-250). The operation should follow parse, entity, sort, index, and frequency.
- The assignment manufactures roughly 96% missingness by converting monthly aggregates to a daily grid and then asks students to compare imputation methods (09/assignment/q2_resampling.md:208-260). This teaches an artificial reconstruction problem rather than missing sensor/visit semantics.
- Q3 aggregates 200 patients into one daily mean (09/assignment/q3_rolling.md:63-78). That is a valid population-level series only if the estimand is explicitly “daily mean among observed patients”; it is not a patient trajectory, and changing patient composition can change the value.
- The test labels q3_visualization.png as bonus but requires it in both a test and the all-files list (09/assignment/.github/test/test_assignment.py:191-223).

### Recommended required demos

1. **What kind of time series is this?**
   - Compare a single regular series, a single irregular series, and a two-entity panel.
   - Define timestamp, period, frequency, entity, panel, naive timestamp, and timezone-aware timestamp.
   - Parse, localize/convert, sort by entity and time, and assert monotonic order within entity.

2. **Frequency and resampling with meaning**
   - Contrast asfreq with resample on one measurement variable and one count variable.
   - Ask students to choose sum, mean, last, or no fill based on the data meaning.
   - Perform a grouped resample that retains entity identity.
   - Show which missing values are introduced by a new grid and which were present in source data.

3. **Past-only features and information availability**
   - Create lag/difference and compare rolling(3) with rolling('3h') on irregular data.
   - Use groupby entity before shift/rolling and prove the first value in each group is missing rather than borrowed from another entity.
   - Demonstrate why center=True uses future observations.
   - Mark an example prediction timestamp, inventory the values available then, and construct a chronological holdout without introducing modeling terminology beyond the Lecture 10 handoff.
   - Optionally plot raw and trailing-smoothed values for one entity using already-established Lecture 07 skills; plotting is not a new Lecture 09 objective.

### Assignment redesign contract

| Contract element | Required design |
|---|---|
| Format | One portable restartable notebook with a deterministic setup/data cell and final verification cell; clean local-Jupyter execution is required. Colab becomes required only if the course-level assignment workflow is approved. |
| Data | A small pinned two-station sensor panel containing irregular timestamps, one timezone conversion, a known missing interval, and no protected data. |
| Task 1 | State row grain, entity key, timestamp meaning, regularity, and timezone; parse, localize/convert, and sort within station. |
| Task 2 | Produce an hourly station panel using a justified aggregation/asfreq choice; report source missingness separately from grid-created missingness. |
| Task 3 | Create one lag, difference, and trailing elapsed-time window within station; identify what is available at a supplied prediction timestamp; reject a centered/future-derived candidate; and define a chronological holdout. |
| Student evidence | prepared_panel.csv, hourly_summary.csv, temporal_features.csv, an availability decision table, and short reasoning cells. A familiar temporal chart may be included as ungraded reinforcement, not as a new objective. |
| Automated checks | Exact timestamp dtype/zone representation, station retention, within-station monotonicity, expected hourly values, no cross-station lag/window, expected first-row missingness per station, no centered window, and restart-and-run-all. |
| Human check | Aggregation/fill rationale and explanation of prediction-time availability and chronological ordering. |
| Excluded | Artificial monthly-to-daily 96% missingness, broad visualization survey, custom rolling functions, expanding windows, decomposition, forecasting, and independent advanced timezone/DST cases. |

### Content disposition

| Current material | Action | Evidence and rationale |
|---|---|---|
| Regular versus irregular data | **Keep/revise** | Useful table at 09/README.md:30-51; add timestamp/period and single/panel axes. |
| Python datetime recap | **Reduce/consolidate** | 09/README.md:53-96 can be a short bridge to pandas rather than a parallel API lesson. |
| DatetimeIndex parsing and sorting | **Keep/reorder** | 09/README.md:98-145 should precede all lag/resample/window work and include entity sorting. |
| Date range and frequency aliases | **Keep/revise** | Update aliases and define frequency as a grid/offset, not merely a code list (09/README.md:147-211). |
| Shift, difference, percentage change | **Keep/reorder** | Repair the Series/DataFrame example and place after structure/index/frequency (09/README.md:213-245). |
| between_time, at_time, truncate | **Move to bonus/reduce** | Advanced selection at 09/README.md:296-337 is useful reference material but not essential to the handoff. |
| Resample and explicit aggregation | **Keep/revise** | Core at 09/README.md:339-434; add measurement semantics and grouped-panel example. |
| Trailing rolling windows | **Keep/revise** | Core at 09/README.md:440-478; distinguish observation-count and elapsed-time windows and repair invalid examples. |
| EWM, centered, custom, and expanding windows | **Move to bonus** | Current assignment treats them as required advanced operations (09/assignment/q3_rolling.md:121-168); centered windows conflict with prediction availability unless explicitly labeled descriptive. EWM adds API surface without strengthening the required Lecture 09 → 10 handoff. |
| Basic timezone localization/conversion | **Keep/reduce** | Retain one practical UTC/local example from 09/README.md:549-594; advanced DST ambiguity belongs in bonus. |
| Basic temporal line chart | **Keep as bounded reinforcement** | 09/README.md:597-643 may reinforce Lecture 07, but chart production and interpretation are not new Lecture 09 objectives. |
| Synthetic decomposition components and broad plotting survey | **Move to bonus/reduce** | 09/README.md:650-694 and Demo 3 expand beyond the temporal-data core. |
| Forecasting, ARIMA, STL, high-frequency/tick analysis | **Keep in bonus only** | Already isolated in 09/BONUS.md; do not leak into required demos or assessment. |

### Ordered content outline

1. Why temporal order changes analysis.
2. Timestamp versus period.
3. Single series versus panel; row grain and entity key.
4. Regular versus irregular observations; frequency.
5. Parse datetime; naive versus timezone-aware; localize versus convert.
6. Sort and index by entity plus timestamp.
7. Select a bounded time interval.
8. asfreq versus resample; downsampling/upsampling and aggregation meaning.
9. Structural missingness created by a grid.
10. Lag, lead as a warning example, and difference.
11. Observation-count versus elapsed-time trailing windows.
12. Information availability at a supplied prediction timestamp.
13. Past-only features, a chronological holdout, and the handoff to Lecture 10.
14. Optional bounded reinforcement: one already-familiar temporal chart; EWM and other window variants remain bonus.

## Lecture 10: honest inference and predictive evaluation

### Intended role

Teach a bounded modeling workflow: classify a question as descriptive, inferential, or predictive; fit and interpret one association-focused linear model with assumptions; then compare a baseline and one train-only linear prediction Pipeline using honest validation and untouched evaluation data.

### Proposed measurable objectives

By the end of Lecture 10, a student can:

1. Classify a question as descriptive, inferential, or predictive; state the unit and intended claim; and distinguish an observed association from a causal claim.
2. For an inferential question, state a population estimand, fit one OLS association model, interpret one coefficient and confidence interval conditionally, distinguish a mean-response confidence interval from an individual prediction interval, and name the assumptions and residual diagnostic that limit the claim.
3. For a predictive question, define the target, target timestamp, prediction horizon, features, feature cutoff and availability, and a simple baseline before fitting.
4. Create disjoint training, validation, and test partitions using a seeded random split for exchangeable rows or chronological cutoffs for future prediction; fit preprocessing only on training data in one scikit-learn Pipeline; and identify target, temporal, preprocessing, or test-set leakage.
5. Use `fit` and `predict` to compare the Pipeline with its baseline using MAE, RMSE, and R²; interpret supplied binary-classification accuracy, precision, and recall against a baseline; evaluate the test set once; and report uncertainty and limitations.

### Exact incoming prerequisites

| Required on entry | Source capability | Why Lecture 10 needs it |
|---|---|---|
| Build a validated analysis table with a stated row grain | Lectures 05–08 exit contracts | A model's unit and features must be explicit before fitting. |
| Make and critique a scatter or line chart | Lecture 07 exit contract | Lecture 10 defines residual and residual plot before diagnostic use. |
| Distinguish single versus panel data and chronological order | Lecture 09 exit contract | Splitting must match the data structure. |
| Create past-only lags/windows and reason about availability | Lecture 09 exit contract | Prediction features must exist at the prediction timestamp. |
| Use notebooks reproducibly | Lecture 04 exit contract | Modeling demos are Colab-first notebooks and must restart-and-run-all. |

### Exact outgoing prerequisites

Lecture 11 may assume that a student can:

- distinguish an inferential estimand from a prediction target;
- state unit, target timestamp, prediction horizon, and feature availability;
- explain association versus causation and avoid causal language without a design/assumptions;
- interpret an OLS coefficient and uncertainty conditionally and check residuals;
- separate training, validation, and test roles;
- fit preprocessing only on training data;
- compare a baseline and one course-supported linear Pipeline with named regression metrics;
- interpret supplied binary-classification accuracy, precision, and recall against a baseline without fitting a second classifier;
- reserve the test set for one final evaluation;
- report limitations and recognize leakage.

Lecture 11 may not introduce XGBoost, deep learning, hyperparameter search, or feature-importance theory as new required capabilities.

### Current alignment matrix

| Proposed objective | Current lecture section | Current required demo | Current assignment | Current test/rubric | Status |
|---|---|---|---|---|---|
| 10.1 Frame inference/prediction, unit, estimand/target/horizon, association/causation | The outline distinguishes inference and prediction (10/README.md:17-23), but the lecture says statistical models explain “why” (10/README.md:101-116, 229-237) before defining estimand, assumptions, or causal identification. | Demo 1 asks for inference and significance but no question/estimand contract (10/demo/DEMO_GUIDE.md:5-21). | Q1 says p-values/AIC help explain “why” variables are related (10/assignment/assignment.md:43-49); no causal caveat or estimand is required. | Tests look for output files and tokens such as R-squared/Observations (10/assignment/.github/test/test_assignment.py:42-89). | **untaught but assessed** |
| 10.2 OLS assumptions, coefficient/CI interpretation, residuals, CI versus PI | OLS syntax, coefficients, p-values, and confidence intervals are listed (10/README.md:120-180), but key assumptions and a real diagnostic sequence are absent. The synthetic y is independent of x1/x2 despite the teaching purpose (10/README.md:190-205). | Demo 1 calls get_prediction(...).conf_int() a prediction interval although the default interval is for the mean response (10/demo/demo1_statistical_modeling.md:179-203); coefficient language uses “effect” without a causal qualification (10/demo/demo1_statistical_modeling.md:167-177). | Q1 assesses p-values, AIC, interactions, and full-data fitted predictions (10/assignment/assignment.md:64-151), including concepts not established in the main lecture. | Tests do not check coefficient interpretation, assumptions, intervals, residuals, AIC, or interaction work (10/assignment/.github/test/test_assignment.py:75-89). | **untaught but assessed** |
| 10.3 Train/validation/test, train-only preprocessing, baseline, availability leakage | The lecture defines only train/test and correctly says test data are unseen (10/README.md:268-279); validation, train-only preprocessing, baseline, and availability are missing. | Demo 2 repeatedly compares models on test data and uses it for XGBoost early stopping (10/demo/demo2_ml_boosting.md:64-83, 139-157, 194-261, 394-428). | Q2 uses one train/test split and lets the test set drive comparisons (10/assignment/assignment.md:157-274). | Tests validate output shape and that two prediction columns differ, not split isolation or train-only fitting (10/assignment/.github/test/test_assignment.py:91-136). | **partially aligned; P0 evaluation leakage** |
| 10.4 Baseline, linear, nonlinear, metrics, final limitations | Linear regression and Random Forest are taught at 10/README.md:301-435, followed by a broad model survey (10/README.md:435-486). No simple baseline or MAE appears in the core sequence. | Demo 2 includes linear, Ridge, Lasso, Random Forest, XGBoost, feature importance, Altair, and early stopping (10/demo/DEMO_GUIDE.md:25-46). | Q2 compares linear/RF; Q3 requires XGBoost (10/assignment/README.md:46-68). | Tests require XGBoost outputs and only shallow schema/content checks (10/assignment/.github/test/test_assignment.py:138-192). | **partially aligned** |

### Current high-priority findings

- The lecture repeatedly equates statistical modeling with explaining why and machine learning with predicting what (10/README.md:101-116, 229-237, 303-316). Inference can estimate conditional associations under assumptions; causal claims require a causal question, design, and identification assumptions.
- p-values, confidence intervals, hypothesis tests, model diagnostics, and statistical significance are used before plain-language definitions or assumptions (10/README.md:25-32, 97-129).
- The OLS example constructs y independently of x1 and x2 (10/README.md:190-196), which undermines the intended coefficient lesson.
- The lecture says the test set is never seen during training (10/README.md:268-279), then passes X_test/y_test as eval_set for early stopping and predicts on that same test set (10/README.md:599-615).
- Demo 1 confuses confidence intervals for the expected mean response with prediction intervals for a new individual observation (10/demo/demo1_statistical_modeling.md:198-203). The Statsmodels reference confirms that `PredictionResults.conf_int(obs=False)` defaults to the mean-response interval, while `obs=True` requests the interval for a new observation: <https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.PredictionResults.conf_int.html>.
- Demo 2 compares feature_importances_ values across Random Forest and XGBoost as though their scales/meaning were directly comparable and explanatory (10/demo/demo2_ml_boosting.md:364-391).
- The required environment combines pandas, statsmodels, scikit-learn, XGBoost, TensorFlow, and Altair with only lower bounds (10/demo/requirements.txt:1-9), while the guide requires a specific local Python because of TensorFlow (10/demo/DEMO_GUIDE.md:71-89). That is inconsistent with a Colab-first core.
- The assignment fetches California Housing at runtime (10/assignment/README.md:3-14), so a fresh run depends on network/cache availability. The redesign should pin a small copy or provide a deterministic fallback for Colab, local grading, and Classroom 50.
- Assignment Q1 requires interactions and AIC although the lecture's core example does not teach their interpretation (10/assignment/assignment.md:121-151).

### Recommended required demos

1. **Question contract and bounded inference**
   - Start with a synthetic or pinned dataset whose generating relationship is visible.
   - Define unit, estimand, association, causation, coefficient, residual, standard error, confidence interval, and prediction interval.
   - Fit OLS, interpret one coefficient conditionally, inspect a residual plot, and display mean-response and individual-observation intervals side by side.

2. **Split choice, availability, and leakage**
   - Define target, target timestamp, horizon, feature cutoff, and availability.
   - Contrast a seeded random split for a genuinely exchangeable cross-sectional task with fixed chronological cutoffs for future prediction.
   - Save disjoint train/validation/test IDs.
   - Identify a post-outcome feature and preprocessing performed before splitting as explicit failure cases; do not use either in the model.

3. **Train-only Pipeline, baseline, and honest evaluation**
   - Fit a training-mean `DummyRegressor` and one `Pipeline(StandardScaler, LinearRegression)`.
   - Compare validation MAE, RMSE, and R², then evaluate the chosen approach on test once.
   - Save predictions, metrics, and one residual plot.
   - Calculate accuracy, precision, and recall from a tiny supplied binary-prediction table and `DummyClassifier` output without fitting a second classification model.

All use a pinned small dataset or deterministic fallback and run in Colab/local Jupyter. XGBoost and deep learning are not required demos.

### Assignment redesign contract

| Contract element | Required design |
|---|---|
| Format | One portable restartable notebook and one short interpretation response; clean local-Jupyter execution is required. Colab becomes required only if the course-level assignment workflow is approved. |
| Data | Pinned committed dataset or deterministic generator with stable row IDs and no runtime-only fetch dependency. |
| Task 1 | State an inferential question and estimand; fit OLS; report one coefficient with CI, one residual check, and a noncausal interpretation. |
| Task 2 | State a prediction target and availability; use provided train/validation/test row IDs; fit train-only preprocessing, `DummyRegressor`, and one linear Pipeline. |
| Task 3 | Select using validation metrics, evaluate once on test, interpret a supplied binary prediction table against a dummy baseline, and report uncertainty and limitations. |
| Student evidence | inference_summary.csv, split_manifest.csv, validation_metrics.csv, final_test_metrics.csv, final_predictions.csv, one diagnostics figure, and short reasoning cells. |
| Automated checks | Exact split membership/no overlap, stable target/features, Pipeline fit contract where inspectable, baseline and metric recomputation, one final prediction per test row, and restart-and-run-all. |
| Human check | Estimand, coefficient/CI interpretation, association/causation language, availability rationale, and limitation. |
| Classroom 50 rule | Tests are public-safe and invariant-based; no reference solution or confidential data enters the published bundle. |
| Excluded | AIC/model-selection survey, interaction terms, Ridge/Lasso, XGBoost, deep learning, repeated test-set comparison, and causal “effect” claims. |

### Content disposition

| Current material | Action | Evidence and rationale |
|---|---|---|
| Inference versus prediction distinction | **Keep/revise** | Core role at 10/README.md:17-38, but replace “why versus what” with question/estimand/target definitions. |
| OLS formula/API and coefficients | **Keep/revise** | 10/README.md:101-208 is the right foundation; add assumptions, real signal, uncertainty distinctions, and diagnostics. |
| GLM/time-series method list | **Move to bonus/reduce to orientation** | Survey at 10/README.md:212-237 is not needed for the exit capability. |
| scikit-learn fit/predict pattern | **Keep/revise** | Good practical anchor at 10/README.md:245-299; add validation, baseline, Pipeline, metrics, and leakage. |
| Linear regression | **Keep** | 10/README.md:301-364 supports the baseline comparison. |
| Ridge/Lasso | **Move to bonus** | Regularization at 10/README.md:318-339 is not assessed in the proposed core and needs scaling/context. |
| Random Forest | **Move to bonus** | One bounded tree ensemble may be an optional nonlinear comparison after the required baseline/linear workflow; it is not a second required model. |
| Broad scikit-learn method catalogue | **Move to bonus/drop from core** | 10/README.md:435-486 adds names without practice. |
| XGBoost | **Move to bonus** | Large required section at 10/README.md:488-652 creates evaluation leakage and environment burden. |
| TensorFlow/Keras and PyTorch code | **Drop from required path; optional orientation only** | 10/README.md:654-891 and Demo 3 add a separate modeling paradigm without prerequisites or a justified dataset. |
| Altair modeling visualizations | **Consolidate with Lecture 07 tools** | Demos should use the established plotting stack rather than introduce another library as incidental machinery (10/demo/DEMO_GUIDE.md:9-21, 29-45). |
| Existing advanced BONUS.md survey | **Reduce and curate** | Keep a small pathway for tuning/interpretability/deployment; remove unsupported library grab-bag examples. |

### Ordered content outline

1. Modeling starts with a question and a unit.
2. Inference versus prediction.
3. Estimand versus prediction target; target timestamp/horizon and feature availability.
4. Association versus causation.
5. Linear relationship, OLS, residuals, and assumptions.
6. Coefficient, standard error, confidence interval, and prediction interval.
7. One residual diagnostic and limitations.
8. Train, validation, and test roles.
9. Train-only preprocessing and Pipeline.
10. Dummy baseline and named metrics.
11. Linear predictor.
12. Validation-based choice and one untouched test evaluation.
13. Supplied binary predictions, positive class, accuracy, precision, and recall.
14. Communicate performance, uncertainty, leakage checks, and limitations.
15. Bonus boundary: regularization, split-aware cross-validation, and one bounded tree ensemble with held-out permutation importance.

## Lecture 11: a compact, reproducible end-to-end workflow

### Intended role

Integrate prior skills around one predeclared prediction question and one reproducible pipeline; Lecture 11 should add project coordination and communication, not new analytical techniques.

### Proposed measurable objectives

By the end of Lecture 11, a student can:

1. Write one project contract stating the question, intended use and claim, entity, unit and row grain, key, target timestamp, prediction horizon, availability cutoff, primary metric, and baseline.
2. Acquire an immutable licensed data release and verify its checksum, schema, row count, attribution, and provenance before analysis.
3. Reuse validated cleaning, joining, reshaping, aggregation, and time-safe feature skills to build one analysis table while preserving entity/time grain and information availability.
4. Create fixed train/validation/test partitions, fit one reproducible train-only `Pipeline`, compare the baseline and model using validation data, evaluate the test set once, and inspect errors by meaningful entity/time slices.
5. Deliver one restartable analysis and coherent report that links every bounded claim to evidence and states limitations and reproducibility details without causal overclaim.

### Exact incoming prerequisites

| Required on entry | Source capability | Why Lecture 11 needs it |
|---|---|---|
| Paths, modules, environments, and reproducible execution | Lectures 01–04 | The final workflow must restart cleanly in Colab/local Jupyter and Classroom 50. |
| Data-quality profile, justified cleaning, and invariants | Lecture 05 | Final cleaning must be evidence-based and auditable. |
| Grain, keys, merge validation, reshaping | Lecture 06 | Entity and timestamp keys must survive joins. |
| Honest accessible charts and communication | Lecture 07 | The final report should apply, not reteach, visualization. |
| Grouped result grain, named aggregation, transform | Lecture 08 | Per-entity summaries and features depend on it. |
| Panel/frequency semantics, grouped resampling, past-only windows | Lecture 09 | Sensor data are an irregular multi-station panel. |
| Question/estimand/target/horizon, availability, split roles, train-only Pipeline, baseline, metrics, leakage | Lecture 10 | Lecture 11 integrates the modeling workflow rather than inventing it. |

### Exact outgoing capabilities

At course completion, a student can independently:

- frame a reproducible analysis or prediction project before manipulating data;
- preserve and verify unit/entity/time meaning across cleaning, grouping, merging, and feature construction;
- separate exploratory work from final evaluation;
- run the complete notebook sequence from a fresh Colab or local-Jupyter environment;
- compare a baseline and supported model on data not used for fitting or selection;
- communicate a bounded claim with evidence, limitations, and provenance;
- submit the complete repository through the course's supported Classroom 50/Git workflow.

### Current alignment matrix

| Proposed objective | Current lecture section | Current required demo | Current assignment | Current test/rubric | Status |
|---|---|---|---|---|---|
| 11.1 Predeclare question/unit/target timestamp/horizon/availability/metric | The lecture opens with a generic nine-phase workflow and dataset (11/README.md:7-31); objectives appear only after the checklist at 11/README.md:168-177. Target selection first appears in Phase 7, after feature engineering (11/README.md:122-141). | Demo 3 defines target only during modeling preparation, after cleaning, wrangling, aggregation, and patterns (11/demo/03_model_prep.md:301-425). Prediction horizon is absent. | Q4 creates features before Q6 lets the student choose a target (11/assignment/q4_feature_engineering.md:31-123; 11/assignment/q6_modeling_preparation.md:128-168). | No test can validate a target/horizon contract because none is required. | **untaught but assessed** |
| 11.2 Pinned provenance and entity-aware cleaning/invariants | The lecture provides a live NYC download workflow (11/README.md:179-203) and a checklist, not a data contract. | Notebook 1 is a broad exploration/cleaning treatment; Notebook 2 reparses datetime and sorts later (11/demo/01_setup.md:119-130; 11/demo/02_wrangling.md:99-123). | Q2 loads raw sensor data and recommends forward fill before Q3 establishes datetime index/order (11/assignment/q2_data_cleaning.md:22-35; 11/assignment/q3_data_wrangling.md:105-109). The live download has no immutable snapshot or checksum (11/assignment/download_data.sh:15-24). | The fixture only changes directory and creates output; it does not execute notebooks or verify source provenance (11/assignment/.github/test/test_assignment.py:18-25). | **untaught but assessed; P0 cross-entity risk** |
| 11.3 Past-only features, train-only prep, chronological train/validation/test | Lecture checklist mentions temporal train/test and leakage (11/README.md:136-141) but no validation set, horizon, or train-only preprocessing. | Demo 3 uses a temporal train/test split and train-derived medians (11/demo/03_model_prep.md:301-359, 499-517), but it initially includes fare_per_mile, derived from the target, as a candidate feature (11/demo/03_model_prep.md:419-450). | Q4 calls observation-count rolling windows “7-hour/24-hour” on irregular multi-station data and does not require station grouping (11/assignment/q4_feature_engineering.md:31-71, 107-124). Q6 uses only train/test and chooses the target late (11/assignment/q6_modeling_preparation.md:31-35, 117-168). | Tests verify only X/y lengths, train larger than test, and that a text file says temporal/time; they do not inspect timestamps, overlap, preprocessing, or feature availability (11/assignment/.github/test/test_assignment.py:412-458). | **untaught but assessed** |
| 11.4 Baseline, one model, untouched test, entity/time error audit | Lecture requires at least two models and train/test evaluation (11/README.md:143-148), with no naive baseline or validation gate. | Demo 4 adds Linear Regression, Random Forest, and XGBoost (11/demo/04_modeling.md:47-61, 253-424) and selects using test RMSE (11/demo/04_modeling.md:424-487). | Q7 requires at least two models and emphasizes R²/feature importance (11/assignment/q7_modeling.md:78-188), not a persistence baseline or station/time error audit. | Tests require two prediction columns and the token R², but do not recompute metrics or protect test isolation (11/assignment/.github/test/test_assignment.py:474-548). | **partially aligned; P0 test-selection defect** |
| 11.5 Restartable analysis and bounded report | The lecture emphasizes integration and results communication (11/README.md:18-31, 150-177), but four notebooks are described as “independent (after previous notebooks)” (11/README.md:81-85). | Four notebooks generate a large set of intermediate outputs. | The README promises one assignment.ipynb (11/assignment/README.md:52-60), but the actual workflow uses nine files (11/assignment/README.md:124-142) and 26 machine artifacts while claiming 24 (11/assignment/README.md:229-289). The manual report rubric is substantive (11/assignment/README.md:300-340). | Automated tests check artifact presence/schema; the report is manual. | **partially aligned; communication aligned, execution contract not aligned** |

### Current high-priority findings

- The final assignment says the live download ensures the exact same dataset as grading (11/assignment/README.md:34-50), but the script downloads the mutable current Socrata export and verifies only that it has at least ten lines (11/assignment/download_data.sh:19-49). Exact grading cannot depend on that claim.
- The assignment says the dataset size varies by download date (11/assignment/README.md:26), directly contradicting the exact-data assertion at lines 36-50.
- Q2's default guidance treats forward fill as generally appropriate for sensor time series before timestamps are parsed/sorted and without grouping by station (11/assignment/q2_data_cleaning.md:22-35, 123-137). A global fill can carry a value across stations.
- Q4 says rolling(window=7) represents seven hours (11/assignment/q4_feature_engineering.md:31-35, 51-71, 120-124). On irregular data, it means seven observations. On a panel, an ungrouped window can cross stations.
- The blanket rule “do not create any features that use your target” (11/assignment/q4_feature_engineering.md:107-113; 11/assignment/q6_modeling_preparation.md:139-161) is incorrect. A lag of the observed target can be valid when it is available before the prediction cutoff; the correct rule is information availability at the stated horizon.
- Q7 treats coefficient magnitude as feature importance without requiring scaling (11/assignment/q7_modeling.md:153-164), so magnitudes are not comparable across differently scaled features.
- Current leakage heuristics equate high correlation or nearly identical train/test performance with leakage (11/assignment/q7_modeling.md:44-74, 212-229). These are warning signals, not definitions; low correlation does not prove availability and high correlation does not prove leakage.
- Q8 includes example “findings” with exact performance and importance values, including a target-like most-important feature (11/assignment/q8_results.md:81-107). This steers students toward reproducing an answer rather than defending a valid workflow.
- The final grader does not execute any student notebook (11/assignment/.github/test/test_assignment.py:18-25). Pre-created outputs can pass many checks without demonstrating a restartable pipeline.

### Recommended required demos

Use one small immutable licensed release that runs in Colab and local Jupyter. The exact source, entity, target, horizon, metric, and baseline remain an instructor decision; no demo or assignment may be authored against the current mutable feed as though that contract were frozen.

1. **Contract, release manifest, and data audit**
   - Freeze the approved question, intended claim, entity, grain, key, target timestamp, horizon, availability cutoff, metric, and baseline before manipulating data.
   - Record source release, license/attribution, checksum, row count, and schema.
   - Parse and sort by entity/time; verify the key and source invariants before analysis.

2. **Validated preparation, availability, and split manifest**
   - Reuse cleaning/joining/reshaping/grouping skills while asserting the approved entity/time grain.
   - Create only features available by the frozen cutoff; current or lagged target values are permitted only when the observed value is available by then.
   - Create chronological train/validation/test partitions and fit preprocessing on training only.
   - Save a compact data audit and split manifest.

3. **Baseline, final model, error slices, and evidence table**
   - Use the baseline frozen in the project contract.
   - Fit one course-supported model selected using validation data.
   - Evaluate once on test with the frozen metrics, then slice error by one meaningful entity/time grouping.
   - Produce two or three accessible figures and a concise claim/evidence/limitations report.

### Assignment redesign contract

| Contract element | Required design |
|---|---|
| Fixed question | **Instructor decision required before authoring:** freeze the intended use and claim, entity, unit/grain, key, target, target timestamp, horizon, cutoff, primary metric, and baseline. |
| Fixed unit | Derived from and verified against the approved immutable release; do not assume station-hour until the source contract supports it. |
| Fixed availability | Features must be computable at the approved cutoff. Current observations and past target lags are permitted only when available by then; future, centered, full-dataset, and post-target aggregates are not. |
| Format | Two portable restartable notebooks—01_contract_prepare.ipynb and 02_model_evaluate.ipynb—plus report.md. Clean local-Jupyter execution is required; Colab becomes required only if the course-level assignment workflow is approved. A third report notebook is optional only if it materially improves accessibility. |
| Data | Immutable reduced CSV/Parquet with source URL, retrieval date, explicit license/terms and attribution, schema, row count, and checksum. A mutable live feed may be optional enrichment, never the grading source. |
| Cleaning | Parse timestamps first; sort by entity/time; deduplicate on the declared key; clean/fill only within an entity with a documented limit; assert no key duplicates and no cross-entity propagation. |
| Features | Only approved, meaningfully defined predictors available by the cutoff; every feature has an availability statement. |
| Split | Fixed chronological train/validation/test cutoffs. Preprocessing fits on train only; validation selects the supported model/settings; test is evaluated once. |
| Models | The baseline frozen in the contract plus one course-supported model. XGBoost and deep learning are not required. |
| Metrics | Use the frozen primary metric and approved secondary metrics; report overall and by a meaningful entity/time slice. R² is never the only metric. |
| Compact artifacts | data_audit.json, split_manifest.csv, validation_metrics.csv, test_metrics.csv, test_predictions.csv, two or three figures, and report.md. |
| Automated checks | Dataset hash/schema, uniqueness of the declared key, chronological/entity boundaries, target shift/horizon, feature availability manifest, disjoint fixed splits, train-only preprocessing evidence, baseline recomputation, metric recomputation, and notebook restart/run-all. |
| Human rubric | Question/contract, cleaning rationale, availability/leakage reasoning, chart integrity/accessibility, error interpretation, claim/evidence, limitations, and reproducibility. |
| Classroom 50 rule | Grader bundle remains valid when inspected. Machine checks use invariants and fixed public data; instructor reference solution remains outside the published bundle. |

### Content disposition

| Current material | Action | Evidence and rationale |
|---|---|---|
| Iterative end-to-end workflow | **Keep/revise** | Strong motivation at 11/README.md:18-31; begin with a specific project contract. |
| Four phase notebooks | **Consolidate to 2–3 restartable notebooks** | Current structure at 11/README.md:33-85 duplicates phase scaffolding and conflicts with “independent after previous.” |
| Objectives | **Move to opening and make measurable** | Current objectives appear at 11/README.md:168-177 after most of the lecture contract. |
| NYC Taxi full workflow | **Move to optional/local extension or use a reduced pinned sample** | Download is dated but not checksum-pinned (11/demo/download_data.sh:14-40) and the project differs materially from the sensor final. |
| Generic cleaning/join/GroupBy/pivot instruction | **Consolidate as reminders/checkpoints** | Lecture 11 currently reteaches earlier APIs; integration should point to prior homes and focus on decisions/invariants. |
| Time-series component | **Keep/revise** | 11/README.md:87-95 is necessary, but entity, frequency, horizon, and availability must be explicit. |
| Correlation-first feature selection | **Revise** | Demo 3 treats correlation as a primary selection heuristic (11/demo/03_model_prep.md:390-417); add availability, domain meaning, and validation. |
| Linear/RF/XGBoost model suite | **Reduce to baseline plus one supported model** | Demo 4 adds breadth and selects on test data (11/demo/04_modeling.md:253-487). |
| Feature importance as “drivers” | **Revise** | Demo 4 frames importance as what “drives predictions” (11/demo/04_modeling.md:29-37, 340-358); keep noncausal, model-specific language. |
| Nine assignment files and 26 machine artifacts | **Drop/consolidate** | Current mismatch at 11/assignment/README.md:52-60, 124-142, 229-289 rewards artifact compliance over a coherent pipeline. |
| Manual communication rubric | **Keep/revise** | 11/assignment/README.md:300-340 appropriately assesses rationale, visuals, interpretation, and limitations; align it to the fixed question. |
| Mutable live download for exact grading | **Drop** | Contradiction at 11/assignment/README.md:26, 34-50 and unpinned URL at 11/assignment/download_data.sh:19-24. |
| Holiday links and unrelated terminal games | **Drop from core** | 11/README.md:213-220 is extraneous to the capstone role. |

### Ordered content outline

1. Measurable objectives and exact prerequisites.
2. One fixed question and intended decision/claim.
3. Entity, unit, row grain, key, target timestamp, horizon, availability cutoff, primary metric, and baseline.
4. Data release, license, checksum, schema, and reproducible acquisition.
5. Parse and sort by entity/time.
6. Entity-aware cleaning and validation invariants.
7. Apply the approved entity/time preparation and missingness rules without reteaching prior APIs.
8. Available feature construction and availability manifest.
9. Chronological train/validation/test design.
10. Train-only preprocessing.
11. The frozen baseline and one supported model.
12. Validation selection and one final test evaluation.
13. Error audit by meaningful entity/time slices.
14. Accessible figures, evidence table, and concise report.
15. Restart-and-run-all and submission verification.

## First-definition and first-independent-use ledger

“Target first definition” is the required revised location. “First independent use” is the earliest point at which students may be asked to produce or debug the capability.

| Term/capability | Current first mention/definition | Current first demanding use | Target first definition | First independent use rule | Action |
|---|---|---|---|---|---|
| unit of analysis / row grain | Not defined in Lecture 08; inherited implicitly | Merge then group in 08/assignment/assignment.md:53-78 | Reconnect from Lecture 06 at Lecture 08 opening | Before any GroupBy or pivot and again after any merge | **move earlier/reinforce** |
| grouping key | Split by category shown at 08/README.md:22-67; term not explicitly defined | Multi-key grouping at 08/README.md:209-238 | Lecture 08 before first GroupBy call | Before student chooses groupby columns | **revise** |
| GroupBy object | API calls begin at 08/README.md:69-103; object not defined | Transform/filter/apply at 08/README.md:108-205 | Lecture 08 basic GroupBy section | Before chained GroupBy methods | **revise** |
| aggregation | Plain-language summary/grouping at 08/README.md:20-26 | Dictionary aggregation at 08/README.md:99-103 | Lecture 08 opening with output-grain change | Before students select aggregation functions | **keep/revise** |
| size / count / nunique | size and count distinguished at 08/README.md:77-78; nunique absent | Provider/facility counts after merge at 08/assignment/assignment.md:69-78 | Lecture 08 counting subsection | Before any assignment count | **add nunique and assess** |
| named aggregation | Absent | Assignment requires multiple differently named results | Lecture 08 after one simple aggregation | Before assignment group_summary.csv | **add** |
| transform | Same-shape definition at 08/README.md:108-110 | Assignment Q1 at 08/assignment/assignment.md:81-92 | Lecture 08 directly after aggregation | After explicit aggregate/transform contrast | **keep/reorder** |
| MultiIndex | Listed as bonus at 08/README.md:3-9; no plain definition before operations | Core at 08/README.md:209-238 and assignment at 08/assignment/assignment.md:130-154 | Lecture 08 bonus only | No required independent use | **move to bonus** |
| pivot table | Defined as multidimensional summary at 08/README.md:240-270 | Advanced pivot/crosstab in assignment at 08/assignment/README.md:117-128 | Lecture 08 after named aggregation | One simple pivot only in required work | **keep/reduce** |
| timestamp | Datetime objects introduced at 09/README.md:53-100, but timestamp versus period is absent | Period/month grouping already appears in Lecture 08 demos | Lecture 09 opening | Before date index, periods, or resampling | **add distinction** |
| period | Only bonus is advertised at 09/README.md:1-7 | Date.dt.to_period used in Lecture 08 demos | Lecture 09 opening, bounded definition | Before any to_period use; remove from Lecture 08 | **move earlier to L09** |
| regular / irregular | Defined at 09/README.md:44-47 | Multi-patient resampling in assignment Q2 | Lecture 09 opening with examples | Before asfreq/resample | **keep/reinforce** |
| single series / panel | Absent | Multi-patient and multi-site data in 09/assignment/README.md:120-158 | Lecture 09 opening | Before grouped time operations | **add** |
| frequency | Inference and code list at 09/README.md:147-211; conceptual meaning weak | Resampling at 09/README.md:339-434 | Lecture 09 before asfreq | Before choosing an offset alias | **revise** |
| naive / timezone-aware | Timezone section begins at 09/README.md:549-594 | Assignment timezone report at 09/assignment/q1_datetime.md:125-167 | Lecture 09 parse/index section | Before tz_localize/tz_convert | **reorder/define** |
| resample / asfreq | asfreq shown at 09/README.md:185-211; resample defined at 09/README.md:339-351 | Assignment Q2 mixes both at 09/assignment/q2_resampling.md:137-205 | One paired Lecture 09 section | Before independent frequency conversion | **consolidate** |
| lag / lead / difference | Defined at 09/README.md:213-229 | Invalid Series example at 09/README.md:233-245 | After entity/sort/frequency | Before student feature creation | **reorder/repair** |
| observation-count / elapsed-time window | Absent | “7-day” rolling by seven rows in 09/README.md:440-478; “7-hour” in 11/assignment/q4_feature_engineering.md:35-60 | Lecture 09 rolling section | Before rolling(window=n) or rolling('nh') | **add; P0** |
| information availability | Not a Lecture 09 term | Centered windows at 09/README.md:481-512 and prediction features later | Lecture 09 window section; formalize in Lecture 10 | Before any modeling feature is created | **add/reinforce** |
| inference / prediction | Mentioned in 10/README.md:17-32; oversimplified at 10/README.md:101-116 | Assignment interpretation at 10/assignment/assignment.md:43-49 | Lecture 10 opening with question examples | Before choosing statsmodels/sklearn | **revise** |
| estimand | Absent | Coefficient/p-value interpretation in Lecture 10 | Lecture 10 opening | Before OLS fitting | **add** |
| association / causation | Correlation joke at 10/README.md:239-241, after “why” claims | Effect language in 10/demo/demo1_statistical_modeling.md:167-177 | Lecture 10 before OLS interpretation | Before any coefficient claim | **move earlier/define** |
| p-value | API/property listed at 10/README.md:25-32, 120-130 | Printed as significance at 10/README.md:198-205 | Lecture 10 uncertainty section | After null/hypothesis/sampling context; optional in bounded core | **define or remove from assessment** |
| confidence interval | Listed at 10/README.md:25-32 | Demo coefficient/prediction interpretation | Lecture 10 OLS section | Before interpreting interval output | **define** |
| prediction interval | Not correctly distinguished | Mislabel at 10/demo/demo1_statistical_modeling.md:198-203 | Lecture 10 beside mean-response CI | Before student interval interpretation | **add/correct** |
| residual / assumption | Residual appears as an error in boosting at 10/README.md:535; OLS assumptions not defined | Demo interpretation and assignment inference | Lecture 10 before diagnostics | Before model adequacy claims | **add** |
| baseline | Absent from current Lecture 10/11 core | Multiple model comparison begins without one | Lecture 10 prediction workflow | Before first trained predictor is judged | **add** |
| train / validation / test | Train/test defined at 10/README.md:268-279; validation absent | Test reused for selection/early stopping | Lecture 10 before any fit | Validation selects; test used once | **add validation/correct use** |
| leakage | Implied by unseen test rule; not defined in Lecture 10 | Current demos leak test information | Lecture 10 split/availability section | Before feature construction and model selection | **add** |
| target timestamp / prediction horizon | Absent | Lecture 11 features precede target selection | Lecture 10 for temporal prediction; restate in Lecture 11 contract | Before any target/feature creation | **add; P0** |
| persistence baseline | Absent | Lecture 11 compares trained models only | Lecture 11 project contract/model section | Before learned model comparison | **add** |
| provenance / immutable data release | Absent | Live downloads used for exact grading | Lecture 11 acquisition section | Before data analysis starts | **add** |

## Handoff verification checks

| Boundary | Must be true before the next lecture begins | Current evidence/gap | Release check |
|---|---|---|---|
| Lecture 08 → 09 | Students can state grouping keys and result grain, distinguish agg/transform, and make grouped output columns/index explicit. No required Lecture 08 demo uses periods or resampling. | Current Lecture 08 demos use to_period('M') (08/demo/demo1_groupby_operations.md:175-203; 08/demo/demo2_pivot_tables.md:149-183), while basic output-grain reasoning is not assessed. | Give students an unseen grouped table; they predict group count/schema, choose size/count/nunique, and explain transform row preservation. Search required Lecture 08 artifacts for to_period, resample, rolling, shift; expected zero teaching uses. |
| Lecture 09 → 10 | Students can identify single/panel structure, sort within entity, distinguish asfreq/resample, and construct past-only lag/windows with no entity crossover. | Current assignment pools patients (09/assignment/q2_resampling.md:143-202; 09/assignment/q3_rolling.md:63-78) and invalid Series examples obscure the basic operation (09/README.md:233-245, 466-478, 533-544). | On a two-entity fixture, first lag/window value for each entity is missing, results are chronological, centered/future features are absent, and the student can state availability at time t. |
| Lecture 10 → 11 | Students can define estimand or target/horizon, distinguish association/causation, use train-only preprocessing, select on validation, compare against a baseline, and evaluate once on test. | Current Lecture 10 reuses test for model comparison/early stopping (10/demo/demo2_ml_boosting.md:394-428) and current Lecture 11 again selects by test RMSE (11/demo/04_modeling.md:424-487). | A split-manifest exercise has disjoint train/validation/test IDs; preprocessing state derives only from train; model choice records validation metrics; final test metrics are computed once and match predictions. |
| Lecture 11 course exit | The end-to-end project is restartable, data are pinned, entity/time grain and horizon are explicit, and every claim maps to an artifact with a limitation. | Current assignment uses mutable data, nine question files, 26 outputs, and tests that do not execute notebooks (11/assignment/README.md:26-60, 124-142, 229-298; 11/assignment/.github/test/test_assignment.py:18-25). | Fresh Colab and clean local runs reproduce hashes, row counts, split manifest, metrics, and figures; Classroom 50 correct/starter/partial/broken fixtures score as specified; human rubric verifies interpretation. |

## Core range dependency graph

The range should implement this capability sequence:

    state input row grain and grouping keys
      → predict grouped output grain and schema
      → choose size/count/nunique and named aggregation
      → distinguish aggregation from transform
      → create one aggregated pivot
      → add entity and timestamp keys
      → distinguish single/panel and regular/irregular data
      → resample within entity with measurement-appropriate semantics
      → create past-only lag/difference/windows
      → state inference or prediction question
      → define estimand or target timestamp/horizon/availability
      → split train/validation/test and fit preprocessing on train only
      → compare a baseline and one train-only linear Pipeline
      → interpret supplied binary metrics against a baseline
      → evaluate once on untouched test data
      → integrate a pinned, entity-aware, reproducible workflow
      → communicate claim, evidence, uncertainty, and limitations

## Ordered implementation gates

1. Approve the proposed objectives and the fixed canonical homes for GroupBy, time-series structure, modeling/evaluation, and integration.
2. Choose and pin the supported pandas/Python package matrix. The current lower-bound-only requirements cannot support one stable API narrative.
3. Rewrite Lecture 08 narrative; remove period/time and remote-performance leakage from required content.
4. Rewrite Lecture 09 narrative and repair every Series/DataFrame example before demo redesign.
5. Rewrite Lecture 10 around descriptive/inferential/predictive questions, assumptions, baseline, validation, one linear Pipeline, and supplied classification metrics; move nonlinear modeling out of core.
6. Freeze Lecture 11's intended claim, entity, key/grain, target, horizon, availability cutoff, pinned licensed dataset release, primary metric, and baseline before any final demo/assignment work.
7. Rebuild the required demos in lecture order against the approved narratives.
8. Rebuild assignments and public-safe Classroom 50 graders against the matrices.
9. Certify all compatible demos in fresh Colab and clean local Jupyter; keep only genuine terminal-tool objectives outside Colab.
10. Run independent boundary verification for 08→09, 09→10, and 10→11.

## Blockers requiring instructor or course-level decision

- **Supported pandas/Python versions:** pandas 3 behavior already conflicts with current Lecture 08/09 prose, while requirements have no upper bounds.
- **Lecture 11 assessment status:** confirm the syllabus decision that this remains the broad final assessment; the redesign assumes a capstone with manual reasoning/communication review.
- **Lecture 11 fixed prediction contract:** select and approve the intended claim, entity, key/grain, target, target timestamp, horizon, availability cutoff, primary metric, and baseline. The current Chicago weather feed contains air temperature, not the previously proposed water-temperature target.
- **Pinned final dataset:** create and license-review an immutable teaching snapshot with checksum; the live Socrata export cannot be the exact grader dataset.
- **Colab assignment support:** demos are Colab-first by existing decision. If assignments are also allowed in Colab, the repository save/submission workflow must be validated before student instructions are finalized.

These blockers do not prevent rewriting the lecture narratives in the proposed order, but they do prevent final assignment/grader certification.

## Independent Lecture 09 narrative verification — 2026-07-18

Scope: post-edit verification of `09/README.md` and `09/BONUS.md` only. This
entry does not certify the still-legacy demos, assignment, grader, or a fresh
Colab run, and it does not rely on the narrative author's implementation notes.
The earlier Lecture 09 alignment matrix remains historical pre-rewrite evidence.

**Result: PASS after one narrow prerequisite correction.** The prerequisite
list had claimed that students arrived knowing an entity key, although the
accepted Lecture 08 exit supplies candidate/grouping keys and Lecture 09 owns
the entity/entity-key definition. `09/README.md` now states the accepted
incoming capability and defines entity and entity key before panel operations.

Verification evidence:

- The five learning objectives match the accepted Lecture 09 contract exactly
  in capability and order. Exactly three H2 `LIVE DEMO` contracts cover
  classification/preparation, measurement-aware resampling, and past-only
  features/availability.
- The narrative proceeds from timestamp/period, single/panel, entity/grain, and
  regularity/frequency through parse/localize/convert/sort/index; grouped
  `asfreq`/`resample` and source-versus-grid missingness; entity-scoped
  lag/difference and observation-count-versus-elapsed-time trailing windows;
  then prediction-time availability and a chronological holdout.
- Every core operation retains station identity. First lags/differences are
  missing within each station, both rolling forms are grouped, the elapsed-time
  window is left-closed to exclude the current observation, and no centered or
  future-derived value is computed as a core feature.
- All 11 core Python fences and all 7 bonus Python fences executed progressively
  with Python 3.12.13, NumPy 2.0.2, pandas 3.0.3, and warnings treated as errors.
  The lowercase `h` and pandas 3 `ME`/`QE`/`YE` alias path produced no warnings.
- One H1, non-skipping heading levels, unique H2 demo headings, and all local
  links passed structural checks. The three linked official pandas pages were
  reachable during verification.
- Expanding, EWM, centered/custom computation, advanced selection/calendar/DST,
  decomposition/STL, forecasting/ARIMA/exponential smoothing, and high-frequency
  material remain bonus-only. Core mentions a centered candidate only to define
  and reject its future-information dependency. No model is fit or evaluated,
  and visualization is explicitly limited to optional reuse of a Lecture 07
  chart rather than a new objective.
- The Lecture 08 handoff is limited to grain/keys, aggregation/transform, and
  explicit grouped output. The Lecture 10 handoff supplies single/panel order,
  past-only windows, and availability reasoning while deferring target/horizon,
  formal split roles, baselines, evaluation, and broader leakage terminology to
  Lecture 10.

Remaining gate: rebuild and independently verify the three required demos and
the Lecture 09 assignment/grader against this narrative. Their existence or
stored outputs were not treated as narrative evidence here.
