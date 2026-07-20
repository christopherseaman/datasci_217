# From a Project Decision to Reproducible Evidence

Lecture 11 brings the course together around one end-to-end predictive project. The emphasis is not a new model or another collection of APIs. It is the order of decisions that makes an analysis reproducible, an evaluation honest, and a claim supportable.

## Learning objectives

By the end of this lecture, students should be able to:

1. Write a project contract that states the question, intended use, bounded claim, entity, unit and row grain, row key, prediction and target timestamps, horizon, availability cutoff, primary metric, and baseline before manipulating data.
2. Verify an immutable licensed data release by its provenance, attribution, checksum, schema, and row count before using it.
3. Build one validated analysis table while preserving entity, time, key, grain, and information-availability rules through cleaning, joining, grouping, reshaping, and feature construction.
4. Use fixed chronological training, validation, and test partitions; fit one train-only course-supported `Pipeline`; select between it and one baseline on validation data; evaluate the chosen approach on the frozen test data exactly once; and inspect one meaningful error slice.
5. Produce a restartable analysis and a concise report in which accessible evidence supports each bounded claim and the limitations are explicit.
6. Complete the supported GUI Git and Classroom 50 submission cycle without treating notebook output stored in the repository as execution evidence.

## Prerequisites

Before starting this lecture, students should be able to:

- resolve portable paths, manage a course environment, and restart and run a notebook from top to bottom;
- profile data quality, justify cleaning decisions, and check explicit invariants;
- state row grain and keys, validate merge cardinality, inspect unmatched rows, and reshape without silent duplication;
- make an honest, labeled, accessible chart;
- predict grouped result grain and distinguish aggregation from a row-preserving transform;
- distinguish a timestamp from a period, a single series from a panel, and an observation-count window from an elapsed-time window;
- sort within entity and construct past-only lags or windows without crossing entity boundaries;
- define a prediction target, target timestamp, horizon, cutoff, and feature availability;
- separate training, validation, and test roles, fit preprocessing on training data only, compare a baseline with one linear `Pipeline`, recognize leakage, and evaluate test data once; and
- distinguish an observed association from a causal claim and report uncertainty and limitations.

Lecture 11 applies these capabilities. It does not reteach every pandas operation or introduce another model family.

## Core scope and current status

The required workflow is one compact path:

1. declare the decision and project contract;
2. verify the approved data release;
3. construct one validated, time-safe analysis table;
4. freeze chronological data roles;
5. compare one baseline with one supported `Pipeline` on validation data;
6. evaluate the selected approach once on frozen test data; and
7. connect claims to accessible evidence, error slices, provenance, and limitations.

The exact dataset, prediction contract, and role of the culminating assessment have not yet been approved. This chapter therefore defines a dataset-agnostic integration framework. It does not authorize dataset-specific demonstrations or an assignment. The materialization gate below must close before those artifacts are designed or described as ready.

Required work does not add categorical-encoding breadth, feature-importance theory, random forests, boosting, deep learning, hyperparameter-search breadth, or a second predictive model. The supported modeling path remains the Lecture 10 baseline and linear scikit-learn `Pipeline`.

## Start with the project contract

A **project contract** is a short, predeclared record of the decision, data meaning, prediction setting, and evaluation rule. It is written before data exploration so that convenient columns or striking patterns do not silently redefine the question.

Restate the contract vocabulary before using it:

- The **intended use** is the decision or action the result may inform.
- A **bounded claim** is the narrow statement the evidence is intended to support. It names the population or cases, conditions, and limits rather than implying universal or causal validity.
- An **entity** is the real-world unit with an ordered history, such as one station. An **entity key** identifies it.
- The **unit** is the case about which one prediction is made. **Row grain** states exactly what one row represents.
- A **row key** is the smallest set of columns that must uniquely identify a row at that grain. In panel data it often includes an entity key and a timestamp, but the approved release must determine the exact key.
- The **prediction timestamp** is when a prediction is issued. The **target** is the unknown value being predicted, and the **target timestamp** is when that value is defined or measured.
- The **prediction horizon** is the span from prediction timestamp to target timestamp.
- The **availability cutoff** is the latest permitted time for information used in a prediction. A feature is **available** only when every source value and processing step needed to compute it exists by that cutoff.
- The **primary metric** is the predeclared numeric rule used to compare observed targets with predictions.
- A **baseline** is a simple reference rule against which a learned model is compared. Added complexity is adopted only if the selection rule chooses the model.

The project contract records all of those fields together with the question, cases in scope, intended users, excluded uses, and the units in which the target and metric are reported. It also states that observational model output describes predictive performance or association; it does not by itself establish causation.

The contract is an operational dependency. No target column, feature, split, or chart should be chosen until the contract is complete and internally consistent.

## Verify an immutable licensed release before analysis

An **immutable release** is a named collection of exact data bytes that does not change after publication. A **license** or documented set of source terms states how those bytes may be used and shared. **Attribution** identifies the source as those terms require.

**Provenance** is the recorded lineage from the original source through any selection or transformation used to create the teaching release. A **checksum** is a digest computed from file bytes; matching the approved digest verifies that the acquired file is byte-for-byte the intended release. A **release manifest** stores the release identifier, source, retrieval date, license or terms, attribution, transformation record, filename, byte size, checksum algorithm and digest, row count, schema, and documented timestamp meaning.

Data work begins only after code verifies the local release against that manifest. The first audit checks:

- the exact filename, byte size, and checksum;
- expected columns, data types, and missing-value encodings;
- exact row count and documented source coverage;
- candidate-key uniqueness and required nonmissing identifiers; and
- parseable timestamps under the approved time-zone interpretation.

A checksum match establishes file identity, not data quality. Schema, keys, time coverage, and source invariants remain separate checks. A mutable live feed may support optional exploration, but it cannot replace the immutable teaching release used for demonstrations, assessment, or grading.

## Preserve entity, time, key, and grain

Every transformation must preserve or deliberately change the meanings declared in the project contract. Before computing a feature or summary, carry forward the entity key, timestamp, row key, and row grain.

For temporal panel data, prepare structure in this order:

1. parse timestamps using the release's documented format and time zone;
2. validate the row key and handle duplicates under the approved rule;
3. sort by entity key and timestamp with a stable order;
4. verify monotonically increasing time within each entity; and
5. distinguish source missingness from missing rows introduced by a requested time grid.

Cleaning does not cross entity boundaries. A fill, lag, difference, rolling window, or resample is grouped by entity whenever entities have separate histories. Observation-count windows are not relabeled as elapsed-time windows. A missing value is not filled merely because a method can fill it; the approved measurement and availability rules must justify the operation and its limit.

A join declares the left and right grain, join keys, and expected cardinality before execution. Afterward, verify unmatched keys, row count, key uniqueness, and output grain. A grouping or aggregation similarly declares the grouping keys and result grain first. If an operation changes grain, record that change rather than allowing it to happen implicitly.

These checks prevent a numerically plausible table from combining different entities, duplicating cases, or assigning information to the wrong time.

## Build one validated analysis table

The **analysis table** is the table at the prediction unit and row grain defined in the project contract. It contains the row key, prediction and target timestamps, target, approved features, and a stable source-row reference needed for auditing.

Reuse the cleaning, joining, reshaping, grouping, and temporal operations from Lectures 05–09 only when the project contract requires them. The goal is not to demonstrate every operation. Each transformation should either establish the declared grain, produce an approved available feature, or create evidence needed to verify the workflow.

The preparation stage ends with explicit invariants:

- the analysis-table row key is unique and required key fields are nonmissing;
- every row has the declared target horizon;
- rows are ordered within entity and remain inside the approved source coverage;
- joins and aggregations have the declared cardinality and result grain;
- missingness decisions are documented and entity-safe;
- target and feature units match the contract; and
- every feature has an availability record.

Save a compact data audit that reports these checks and the resulting row count. Assertions should stop the run when a required invariant fails; a later model must not hide an invalid table.

## Construct only time-available features

A **feature-availability record** states a feature's source columns, source timestamps, computation, latest source time, required cutoff, and approval decision. Meaning and availability come before statistical association.

A past target lag can be valid if that target observation is genuinely recorded by the prediction cutoff. A current measurement can be invalid if it arrives after the cutoff. Centered windows, future rows, post-target measurements, and summaries fit on the completed dataset are unavailable for a prospective prediction.

**Leakage** occurs when information outside the allowed prediction or fitting boundary influences a feature, learned preprocessing state, model choice, or evaluation. In this workflow:

- target leakage uses the outcome or a post-outcome proxy as an input;
- temporal leakage uses information that was not available by the cutoff;
- preprocessing leakage learns imputation, scaling, or other state outside training data; and
- test leakage lets test results affect features, settings, model choice, or reporting decisions.

For every feature, trace the actual source and timestamp through the computation and compare it with the approved cutoff. A numerical pattern alone cannot establish availability or rule out leakage.

## Freeze chronological training, validation, and test roles

The **training partition** fits model coefficients and all learned preprocessing state. The **validation partition** compares the predeclared baseline and supported `Pipeline` during development. The **test partition** estimates final performance only after every choice is frozen.

Because this project represents prediction across time, use fixed chronological boundaries rather than a random split. Assign each row by the approved target-time or issue-time rule, record the rule, and save a **split manifest** that maps every row key to exactly one role. Then verify:

- the three row-key sets are nonempty, disjoint, and exhaustive for eligible rows;
- the chronological ordering required by the prediction setting holds;
- each target and all feature inputs obey the cutoff for its row;
- no entity history is accidentally pooled during lag or window construction; and
- the same immutable manifest is reused by every subsequent run.

The split manifest freezes membership; rerunning the notebook must not silently choose new rows. Test targets may be present so the final metric can be computed, but they remain outside development decisions.

## Select on validation and evaluate test once

A scikit-learn **`Pipeline`** chains learned preprocessing and an estimator so both are fit under the same data boundary. The required model remains the course-supported linear path from Lecture 10: training-fitted scaling followed by linear regression. The materialized project must freeze its exact columns and settings.

Fit the baseline's state from training rows only. Fit the `Pipeline` on the same training rows; its scaler and model must never fit on validation or test rows. Compute the frozen primary metric on validation for exactly those two approaches. Record the values and the deterministic selection rule, including its tie rule, in a validation table.

After validation chooses the approach:

1. freeze the feature list, preprocessing, model or baseline rule, settings, split manifest, metric code, error slice, and report plan;
2. run the chosen approach against the test partition once;
3. save one prediction per test row key; and
4. recompute the primary test metric from those saved predictions as a consistency check.

Do not inspect test results and return to feature construction, alter a setting, replace a chart, or choose a different slice. Any such change turns the test partition into development data and requires a newly justified untouched evaluation set.

## Inspect one predeclared error slice

For row-level inspection, a **signed residual** is the observed target minus its prediction. An **error slice** is a meaningful, predeclared subset or grouping of test cases, such as an approved entity group or calendar segment. Summarize that slice with the frozen primary metric and use signed residuals only to show the direction of individual misses. Together they help identify where aggregate performance hides systematic gaps.

Choose the slice from the intended-use and data contracts before revealing test results. Report its case count, target coverage, and the same error units used for overall performance. Small or sparse slices need an explicit caution; a visually large difference is not automatically stable or actionable.

The error slice is part of the single frozen test evaluation, not a search over many groupings for the most dramatic result. It describes held-out behavior in the observed test cases. It does not prove a cause for the errors or guarantee the same pattern in future data.

## Link bounded claims to accessible evidence

**Evidence** is a reproducible table, metric, figure, or audit result that directly bears on a claim. A **limitation** is a data, design, measurement, modeling, or generalizability condition that narrows what the evidence supports.

The final report uses a compact claim–evidence–limitation table:

| Bounded claim | Required evidence | Limitation to state |
|---|---|---|
| The source used is the approved release | release manifest, checksum, schema, row count | source coverage and measurement constraints |
| The analysis table preserves the project unit | key/grain audit and transformation checks | exclusions and unresolved missingness |
| Features respect the information boundary | availability records and leakage audit | recording delays or unavailable source details |
| The selected approach earned selection | validation metrics for baseline and `Pipeline` | validation-period representativeness |
| Final performance on the held-out cases | one test metric and row-keyed predictions | distribution change and finite test coverage |
| Performance differs across the approved slice | slice counts, metric, and accessible figure or table | sparse groups and descriptive, noncausal interpretation |

Every figure states its question, labels axes and units, uses readable text and sufficient contrast, and does not rely on color alone. A caption or nearby text states the pattern and the relevant limitation. Axes and aggregation choices must represent the data honestly. Alternative text or an equivalent table must carry the essential information for readers who cannot use the visual.

Use noncausal language unless a separate causal design and assumptions justify more. A predictive error difference does not identify its cause, and a model coefficient does not become an intervention effect because it is precise.

The report also records the release identifier, runtime and package lock, notebook execution order, split manifest identifier, metric units, and Classroom 50 submission commit. That evidence makes the result inspectable without inflating the report into a second analysis.

## Materialization gate: freeze before demos or assignment design

The instructor or course team must approve one versioned project specification containing every item below:

1. the culminating assessment's exact role and required student deliverables;
2. the question, intended use, intended users, bounded claim, and explicit excluded uses;
3. the entity, entity key, prediction unit, source row grain, analysis-table grain, and unique row key;
4. the target, target units, prediction timestamp, target timestamp, and prediction horizon;
5. the availability cutoff and an approved source-and-time rule for every candidate feature;
6. the primary metric, any approved secondary metric, and the exact baseline rule;
7. the one supported `Pipeline`, including its fixed preprocessing, estimator, and settings;
8. the exact chronological training, validation, and test boundaries or immutable row-ID manifests;
9. the immutable teaching-data release identifier and bytes, source URL, retrieval date, license or terms, required attribution, and provenance record;
10. the release checksum algorithm and digest, byte size, schema, row count, timestamp/time-zone interpretation, missing-value encodings, and source invariants;
11. the approved entity-aware cleaning, duplicate, missingness, join, aggregation, and time-feature rules;
12. the predeclared test error slice and the minimum evidence needed to interpret it responsibly;
13. the exact output artifacts, accessible figure requirements, human-review criteria, and machine-checkable invariants;
14. the certified Python and direct/transitive dependency lock for fresh Colab and clean local Jupyter; and
15. whether assignment work may use Colab, which remains conditional on a successful save-back and Classroom 50 submission pilot.

Approval means the fields agree with one another and point to immutable artifacts. A mutable live export, a retrieval date without exact bytes, or a proposed target without a horizon does not pass the gate. Until the complete specification is frozen, the existing Lecture 11 demos and assignment remain legacy material and must not be treated as the implementation of this narrative.

## Restartable demonstration and execution contract

After the materialization gate closes, required demonstrations should form one three-part sequence:

1. contract, release manifest, and source audit;
2. validated preparation, availability records, and split manifest; and
3. baseline, supported `Pipeline`, validation choice, one test evaluation, error slice, and evidence table.

Each notebook must run from a fresh Colab runtime and clean local Jupyter environment using the approved immutable release and dependency lock. The sequence may pass only documented files between notebooks; it may not depend on hidden kernel state, a mounted personal drive, a manual upload, or a mutable live download. A complete clean run recreates every derived table, metric, prediction, and figure from the approved source bytes.

At the top of each notebook, identify the expected input artifacts, verify them before use, and print the active Python and direct dependency versions. At the end, assert the required output schema, keys, row counts, and invariants. Restart the runtime and run all cells in order before review.

Stored notebook output is for human reading only; it is not proof that the current code ran. Release verification and Classroom 50 grading must fresh-execute the notebooks in the certified environment and compare the resulting artifacts with their contracts.

Clean local-Jupyter execution is required. Colab is the default path for compatible demonstrations. It is not an assignment submission path until the save-back and Classroom 50 pilot proves that edits and generated files return to the accepted repository without loss.

## Complete the GUI Git and Classroom 50 course-exit workflow

The supported course-exit path applies Classroom 50 to the complete repository and uses VS Code Source Control or GitHub Desktop for required Git work:

1. open the accepted Classroom 50 repository and confirm the expected assignment and data-release files;
2. run the complete analysis in the certified environment and inspect the recreated artifacts;
3. use the Git GUI to review changed and untracked files, excluding environment, cache, and temporary data files;
4. stage a coherent set of source notebooks, required outputs, manifest records, and report changes;
5. commit with a concise message, synchronize through the GUI, and confirm that the commit appears on GitHub;
6. inspect Classroom 50 feedback, distinguish a workflow failure from a substantive result, and make any permitted correction; and
7. rerun from clean state and repeat the GUI review, commit, synchronization, and feedback check.

Command-line Git is optional bonus material, not a requirement for completing this workflow. A successful push alone is not completion: the submitted commit must contain the required files, fresh execution must reproduce the contracted outputs, and every reported claim must remain linked to evidence and limitations.

Completion means that one traceable decision record connects the project contract, release manifest, data and availability audits, split manifest, validation choice, frozen test evidence, accessible report, and submitted commit. That chain—not the number of models, plots, or intermediate files—is the complete data science workflow.
