# From a Question to a Defensible Result

![xkcd 3172: Fifteen Years](media/fifteen_years_2x.png)

This capstone lecture is about putting familiar tools together around a real
question. It is not a claim that every project follows one complete, linear
"data science lifecycle," and it is not a challenge to use every technique from
the course.

The approach follows the spirit of McKinney's data analysis examples: understand
what the records represent, ask a focused question, and let that question determine
the useful cleaning, reshaping, summaries, plots, and—when appropriate—modeling.
The path is usually iterative. A surprising count may send us back to the source
documentation; a plot may reveal that the first question needs revision.

The current live demo uses a compact course release derived from January–June 2023
NYC Yellow Taxi trip records. It is one worked example, not a universal project
template.

## Learning objectives

By the end of the session, given a dataset and an analytical question, you should
be able to:

- state an answerable question and a candidate claim that evidence could support
  or contradict;
- identify the row grain (what each row represents) and candidate key (the field
  or fields expected to identify each row uniquely) of both a source table and an
  analysis table, then perform a check that would expose duplicate or missing keys;
- trace one reported result back through its source, selection rules, and important
  transformations;
- justify a cleaning, wrangling, aggregation, or visualization decision in terms
  of the question it helps answer;
- when prediction is appropriate, specify a target, prediction time, available
  features, comparison baseline, split strategy, and evaluation measure; and
- write a concise conclusion that separates the observed result, the claim it
  supports, and at least one material limitation.

## What is review, and what is new?

Most of today's code is review. We will reuse DataFrame inspection, missing-value
handling, joins, `groupby`, reshaping, datetime operations, visualization, feature
construction, and the basic scikit-learn estimator interface.

The limited new content is integrative rather than a large new API:

- keeping a question, a possible claim, and the available evidence aligned;
- making row grain, keys, and provenance (where data came from and how it was
  constructed) explicit before calculating;
- recognizing the boundary between a DataFrame built for analysis and a matrix
  built for a model;
- deciding whether a model adds anything to the question; and
- carrying evidence and limitations through to the final communication.

McKinney's Chapter 13 examples model this kind of integration. Different questions
call for different moves: missing fields become visible during inspection, related
tables are merged when the desired comparison needs them, and counts are sometimes
normalized before groups can be compared fairly. Chapter 12 supplies the narrower
bridge from pandas to modeling: feature engineering often grows out of aggregation
and transformation, preprocessing must respect the training data, and performance
is assessed on out-of-sample data. Neither chapter presents a quota of techniques.

## A worked capstone: one question, several connected decisions

### Begin with the question and the possible claim

For the taxi example, our working question is:

> Using information available before a target hour, how well can we predict the
> pickup count for each selected taxi zone in the next hour, and where are the
> largest errors?

That question implies a prediction time, an outcome, and a unit of analysis. It
also suggests a testable candidate claim: recent history, weekly history, and
calendar position may predict next-hour pickups better than simply repeating the
count from the same hour one week earlier. The demo does not assume that claim is
true; the comparison supplies the evidence.

Pause for an in-class choice: what should “how well” mean (which evaluation
measure), how should we define the selected zones, and what baseline would make
the comparison meaningful?

A different project might ask a descriptive or comparative question instead. In
that case, a carefully designed table, aggregation, and visualization may be the
right endpoint. A model is useful only when it helps answer the question.

### Establish what one row means

The taxi release offers two useful views of the same setting:

| View | One row represents | Key or identifier | What it can support |
| --- | --- | --- | --- |
| Event sample | one sampled pickup event | `course_row_id` | auditing fields and practicing event-level checks |
| Zone-hour panel | one selected zone at one UTC hour | (`pickup_zone_id`, `target_hour_utc`) | zone-hour counts, time patterns, and the prediction task |

The event sample is not a random sample for estimating citywide totals. The panel
was derived separately from the full documented source files. Confusing those two
grains would produce calculations that run successfully but answer the wrong
question.

Before analysis, we therefore ask:

- What generated these records, and what selection rules were applied?
- Is the proposed key unique and complete at the grain we need?
- Does a zero mean no observed pickups, or could it mean no row was recorded?
- Which timestamp orders events unambiguously, and which is easiest to interpret?

In this example, UTC provides an unambiguous ordering key through daylight-saving
changes, while New York local time is useful for explaining hourly and weekly
patterns.

### Audit evidence before changing it

Provenance and data quality are part of the analysis, not setup trivia. The demo
uses the release manifest—the expected artifact filenames, source identities, and
checksums—to connect course artifacts to documented source files and verify file
identities. We then inspect types, missingness, timestamp
parsing, month membership, zone membership, uniqueness, and coverage.

Cleaning is purposeful. We do not automatically drop every row with a missing
value or remove every large observation. Instead, we ask whether an issue changes
the meaning of a row or threatens the intended calculation. When a row is excluded,
the demo records a reason so that the decision remains auditable.

### Build the table the question needs

The source grain and the question's grain are often different. Here, prediction
needs a complete zone-hour panel rather than a pile of individual trips. Once that
grain is established, calendar fields, lagged counts, and rolling summaries can be
defined with a clear interpretation. Here, a recent lag is the previous hour
(*t*−1), a weekly lag is the same zone and hour 168 hours earlier (*t*−168), and
rolling history is a summary such as the mean of earlier hourly counts over a
specified window.

This is the same general move seen throughout McKinney's examples: merge when facts
are split across tables, aggregate when the question concerns groups, reshape when
the comparison is clearer in another layout, and normalize when raw totals would
be misleading. Those are choices, not mandatory project stages. Each transformation
should have a sentence explaining what question it serves and a check that the
result still has the intended grain.

*Your capstone is not Pokémon for obscure ML—you don't have to catch 'em all.*

### Let summaries and plots refine the question

Simple summaries usually come before a model. Counts by group, distributions, and
time profiles can reveal data problems, plausible patterns, and important
differences between groups. A useful plot has a job: it might compare hourly pickup
profiles, show changes over time, or expose where prediction errors are concentrated.

The analysis may loop here. An unexpected pattern can motivate a subgroup view, a
normalization, a revised cleaning decision, or a narrower claim. Exploration is
productive when each follow-up is tied to the question rather than to a desire to
generate more charts.

![xkcd 2582: Data Trap](https://imgs.xkcd.com/comics/data_trap.png)
*Analysis should produce understanding, not an unbounded pile of artifacts.*

### Model only under a clear prediction contract

For a predictive question, "what would be known at prediction time?" is the key
boundary. A feature for target hour *t* may use earlier observations, but not the
target count or information created afterward. Rolling features therefore begin
with a shift.

The current demo uses a time-ordered training, validation, and test design because
the intended use is prediction of later hours. It compares a weekly-lag baseline
(a deliberately simple comparison prediction) with one transparent scikit-learn
pipeline, makes the choice using validation data reserved for model choice, and
evaluates the selected approach once on the held-out test period. Preprocessing
and model fitting stay together so that information learned from later data does
not leak backward.

These choices fit this question and dataset. Other questions may need a random
split, grouped split, cross-validation, a statistical model, or no model at all.
The transferable principle is to make the evaluation resemble the proposed use
and to compare against a meaningful simple alternative.

### Communicate a result at the strength the evidence allows

A compact conclusion should answer four questions:

1. What did we observe, using what measure or visualization?
2. What claim does that evidence support—and what stronger claim does it not
   support?
3. Where does the result vary or fail?
4. Which provenance, coverage, measurement, or modeling limitation matters most?

For the taxi demo, prediction error can describe performance on the held-out hours
and selected zones in this release. By itself, it does not establish a causal
explanation, guarantee future performance, or describe all NYC taxi activity.

## Session shape: lecture and demo in equal parts

We will spend roughly half the session on the worked reasoning above and half on
the executable taxi example. The three lecture anchors are question/claim,
grain/provenance, and the prediction contract; the other subsections serve as
synthesis and reference for the live example.

| Share of class | Mode | Focus |
| --- | --- | --- |
| First ~50% | worked lecture and discussion | question/claim, grain/provenance, and prediction contract, with the other decisions synthesized around them |
| Second ~50% | live notebook walkthrough | inspect concrete records, build the zone-hour table, prepare past-only features, evaluate a simple comparison, and inspect errors |

The lecture supplies the reasoning that transfers to other datasets. The demo
shows how one particular set of decisions is implemented and checked; it is not a
second pass through a required universal checklist.

## Demo roadmap

The four core notebooks follow the taxi question from evidence to result:

1. **`01_setup.ipynb` — Trust the release before using it.** Verify provenance,
   inspect event-grain records, and make exclusions auditable.
2. **`02_wrangling.ipynb` — Build a past-only model table.** Verify the zone-hour
   key and coverage, then construct interpretable calendar and history features.
3. **`03_model_prep.ipynb` — Analyze training patterns and freeze the split.** Use
   training data for exploratory summaries and keep later periods separate.
4. **`04_modeling.ipynb` — Compare, freeze, and report.** Compare a weekly baseline
   with one pipeline, evaluate held-out performance, and examine error slices.

**`05_geo_bonus.ipynb`** is an optional geographic view of zone-level results. It
is enrichment, not a required part of the capstone pattern.

### Where this connects to earlier lectures

Use this crosswalk as a set of reminders, not as a mandatory workflow. Each
decision should still be justified by the capstone question and data.

| Capstone decision or concept | Earlier canonical lecture | Related demo roadmap stage |
| --- | --- | --- |
| Question, claim, and evidence | Lecture 07, Data Visualization | `01_setup.ipynb` — trust and inspect the release |
| Missingness, row meaning, keys, and joins | Lecture 05, Handling Missing Data; Lecture 06, Database-Style DataFrame Joins | `01_setup.ipynb` — audit records; `02_wrangling.ipynb` — build the analysis table |
| Aggregation and a question-shaped table | Lecture 08, Data Aggregation and Group Operations | `02_wrangling.ipynb` — verify the zone-hour table |
| Time-aware fields and past-only features | Lecture 09, Time Series Analysis | `02_wrangling.ipynb` — construct history features |
| Candidate models, leakage boundaries, and evaluation | Lecture 10, From Statistics to Deep Learning | `03_model_prep.ipynb` — freeze the split; `04_modeling.ipynb` — compare and evaluate |

## Transfer to the later assignment

The later assignment is a somewhat guided mini research project using a different
dataset. The transfer is in the reasoning: formulate a question, understand the
records, choose transformations that serve the question, evaluate claims honestly,
and communicate limitations. It should not be a search-and-replace copy of the
taxi notebooks, and it does not need a technique merely because that technique
appears in today's demo. Follow the assignment's own deliverables while adapting
the analytical choices to its data and question.

## Getting started with the demo

The commands below assume macOS, Linux, or WSL with Bash. On native Windows, open
the repository in WSL because the data downloader is a Bash script and uses Unix
checksum tools.

From the course repository:

```bash
cd 11/demo
uv venv --python 3.12.13 .venv
source .venv/bin/activate
python --version  # should report Python 3.12.13
uv pip install -r requirements.txt
chmod +x download_data.sh
./download_data.sh
```

Then open `01_setup.ipynb` and continue through `04_modeling.ipynb` in order. Each
notebook explains the artifact it reads or rebuilds, so you can pause between them
and inspect the intermediate reasoning—not just the final output.

## Optional practice after class

- [Advent of Code](https://adventofcode.com) — short programming puzzles for
  continued practice.
- [GameShell](https://github.com/phyver/GameShell) — a game for practicing the Unix
  shell.

![xkcd 1513: Code Quality](media/xkcd_1513.png)
