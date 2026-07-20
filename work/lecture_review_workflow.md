# Lecture review workflow

## Purpose

Use parallel reviewers without fragmenting the curriculum into isolated lecture critiques. The workflow reviews each lecture as a content–demo–assignment unit, then checks prerequisite, terminology, and skill progression across lecture boundaries.

The review answers five questions:

1. Is the right material present and accurate?
2. Is anything distracting, redundant, misplaced, or unnecessarily advanced?
3. Are concepts and terms defined before students are expected to use them freely?
4. Does each lecture build deliberately on prior lectures and prepare students for later ones?
5. Do the demos and assignment assess what the lecture actually teaches?

This workflow does not introduce course-calendar or lecture-timing work.

## Source-of-truth rule

Review the current files under `01/` through `11/` as the material that actually exists. Apply explicit user direction and the course-design constraints recorded in the course-specific historical `CLAUDE.md` revision (`070f7b6:CLAUDE.md`). Files under `work/` are context or historical evidence, not proof of the current sequence. Any claim from an older plan or prerequisite map must be checked against the current lecture, demo, and assignment sources.

When current material conflicts with a course-design constraint, record a policy conflict for instructor adjudication rather than silently treating either side as correct.

## Course-design constraints to test

- Lectures 01–05 should form a coherent foundational toolkit; Lectures 06–11 extend it with advanced mastery.
- Lectures 01–03 execute Python and assignment work through scripts and the terminal; Jupyter begins in Lecture 04.
- Terminal-first Python does not imply command-line-first Git. The recorded course design uses VS Code/GitHub GUI workflows for core Git skills and reserves command-line Git for bonus material unless the instructor changes that policy.
- Core lecture material should emphasize practical daily-use tools. Specialized libraries, theoretical depth, and advanced variations belong in bonus material unless a later prerequisite requires them.
- Stabilize lecture content and progression before redesigning its demos or assignment.
- Each lecture should converge on two or three required demos that directly reinforce its content and assignment.
- Regular assignments should test practical competence with incremental complexity; the Lecture 05 and Lecture 11 assessments may use a broader rubric, subject to the current syllabus decision.

## Review unit

A lecture review includes all of the following where present:

- lecture `README.md`;
- bonus material;
- demo guide and every required demo artifact;
- assignment instructions, starter artifacts, requirements, public tests, and grader configuration;
- media, external links, and terminology that affect comprehension;
- the preceding lecture as a prerequisite boundary;
- the following lecture as a preparation boundary.

Absence is a finding. For example, Assignment 01 is not skipped merely because its package is absent from the current tree; its historical package and current lecture objectives must inform a restore, redesign, or retirement decision.

## Team roles

### Range reviewers

Range reviewers own a small adjacent group of lectures. They perform the full per-lecture review and identify dependencies that enter or leave their range. They cite exact paths and line numbers and do not edit course content during the review pass.

Recommended initial lanes:

- Lectures 01–03: terminal foundations;
- Lectures 04–07: Jupyter/pandas through visualization;
- Lectures 08–11: aggregation/time series through modeling and integration.

### Boundary reviewers

Boundary reviewers examine handoffs rather than whole lectures:

- Lecture 03 → 04: terminal scripts/environments/NumPy into Jupyter and pandas;
- Lecture 07 → 08: visualization into aggregation and group operations;
- Lecture 09 → 10: time-series analysis into modeling;
- Lecture 10 → 11: model preparation/evaluation into the complete workflow.

They look for prerequisites assumed but not taught, terms whose meaning shifts, unnecessary reteaching, and assignments that jump ahead.

### Vertical-thread reviewers

Vertical reviewers trace one concept family through the course:

- terminal, paths, environments, Git, and reproducibility;
- Python syntax, functions, modules, and data structures;
- NumPy and vectorized thinking;
- pandas structures, selection, cleaning, joining, reshaping, and aggregation;
- visualization and communication;
- datetime and time series;
- statistical reasoning, features/targets, modeling, evaluation, and leakage;
- notebooks, Jupyter state, Colab, and reproducible execution.

### Synthesizer

The primary reviewer merges evidence, resolves conflicting recommendations, maintains the dependency graph and term ledger, and produces a single prioritized change set. A recommendation is not accepted solely because one reviewer suggested it.

### Verification reviewer

After content changes, a reviewer who did not make the change verifies the affected lecture unit and both adjacent boundaries. Execution testing and pedagogical verification are separate checks.

## Review passes

### Pass 1: inventory

For every lecture, inventory:

- stated objectives and prerequisites;
- major sections and concepts;
- named terms, libraries, commands, methods, and data-science practices;
- required demos and their concepts;
- assignment tasks and grader expectations;
- bonus topics;
- missing or conflicting artifacts.

### Pass 2: parallel lecture review

Each range reviewer evaluates the same dimensions:

| Dimension | Review question |
|---|---|
| Purpose | Is the lecture's central job in the course clear? |
| Completeness | Are the essential concepts, definitions, examples, and practice present? |
| Scope discipline | What is duplicative, distracting, too specialized, or better as bonus? |
| Internal order | Does each section depend only on concepts already established within the lecture? |
| Prerequisites | Are all assumed skills actually taught earlier? |
| Terminology | Is each term defined before unrestricted use? |
| Progression | Does the lecture extend prior knowledge rather than restart or jump ahead? |
| Demo alignment | Do demos practice the stated objectives using already introduced concepts? |
| Assignment alignment | Does the assignment assess the objectives without requiring untaught skills? |
| Pedagogical sensibility | Is the explanation concrete, motivated, appropriately scaffolded, and internally consistent? |

Complete the content/order findings before recommending detailed demo or assignment rewrites. Artifact recommendations must follow the accepted lecture sequence rather than preserve a mismatch.

### Pass 3: first-definition/first-use ledger

Maintain one row per important term or skill:

| Concept or term | First mention | First actual definition | First guided use | First unrestricted use | Finding/action |
|---|---|---|---|---|---|

Use these distinctions:

- **Mention:** a preview that does not require understanding.
- **Definition:** a plain-language meaning plus its role or purpose.
- **Guided use:** scaffolded code or commands whose unfamiliar parts are explained or deliberately treated as a preview.
- **Unrestricted use:** students are expected to read, modify, debug, or produce it independently.

A term may be mentioned early. It may not be used freely before definition and guided practice unless explicitly identified as supplied machinery outside the learning objective.

### Pass 4: dependency graph

Record edges as concrete capabilities, not vague lecture references. For example:

```text
run a Python script
  → define and call a function
  → import a local module
  → create an isolated environment and install NumPy
  → understand an ndarray's shape/dtype/indexing
  → compare ndarray and DataFrame behavior
  → select and transform labeled pandas data
```

For each edge, verify:

- the source capability is taught and practiced;
- the destination lecture names it as a prerequisite where useful;
- a demo reinforces the transition;
- the assignment does not require a stronger version than was taught.

### Pass 5: alignment matrix

Every required objective receives an evidence row:

| Objective | Lecture definition/example | Guided demo | Assignment evidence | Grader/rubric evidence | Status |
|---|---|---|---|---|---|

Allowed statuses:

- **aligned**;
- **partially aligned**;
- **untaught but assessed**;
- **taught but unpracticed**;
- **practiced but unassessed**;
- **orphaned** (material has no clear course purpose).

### Pass 6: content disposition

Every substantial recommendation uses one of these actions:

| Action | Meaning |
|---|---|
| Keep | Correct, necessary, well placed, and adequately supported. |
| Revise | Necessary and correctly placed, but explanation/scaffolding is insufficient or inaccurate. |
| Reorder | Belongs in the lecture but appears before its prerequisites or motivation. |
| Move earlier | A later dependency requires this concept sooner. |
| Move later | The concept currently jumps ahead of the course progression. |
| Move to bonus | Useful enrichment but not required for the core course thread. |
| Consolidate | Repeated material should have one primary definition and later reinforcement. |
| Drop | Redundant, obsolete, distracting, or unrelated to the course outcomes. |

### Pass 7: boundary and vertical-thread review

Range reviewers exchange their findings. Boundary and vertical reviewers then test the proposed progression across the complete course. This pass catches local recommendations that create a problem elsewhere.

### Pass 8: adjudication

The synthesizer records:

- accepted findings with evidence;
- rejected findings with rationale;
- unresolved questions requiring instructor judgment;
- prerequisite-map changes;
- term-ledger changes;
- the resulting ordered change set.

Priorities describe pedagogical impact:

- **P0:** students are assessed on untaught material, a prerequisite chain is broken, or core material is materially incorrect.
- **P1:** organization, duplication, missing definitions, or weak scaffolding significantly harms learning.
- **P2:** polish, consistency, optional enrichment, or repository hygiene.

### Pass 9: post-edit verification

After revisions:

1. Re-read the complete lecture unit.
2. Recheck both adjacent lecture boundaries.
3. Update the first-use ledger and dependency graph.
4. Execute the required demos.
5. Run known starter, correct, partial, and broken assignment fixtures.
6. Confirm the objective alignment matrix has no `untaught but assessed` row.

## Required reviewer output

Each range reviewer returns this structure:

```markdown
## Lecture NN: title

### Intended role
One concise statement of the lecture's job in the course.

### Strengths
- Evidence-backed strengths.

### P0 findings
- Finding, evidence, consequence, recommended action.

### P1 findings
- Finding, evidence, consequence, recommended action.

### P2 findings
- Finding, evidence, consequence, recommended action.

### Definition/order findings
| Term/skill | Current definition | First demanding use | Action |
|---|---|---|---|

### Alignment findings
| Objective | Lecture | Demo | Assignment | Status/action |
|---|---|---|---|---|

### Content disposition
| Material | Keep/revise/reorder/move/consolidate/drop | Rationale |
|---|---|---|

### Recommended concept sequence
An ordered list for the revised lecture.
```

The range report ends with incoming dependencies, outgoing dependencies, unresolved instructor decisions, and a prioritized change list.

## Review quality rules

- Cite current files and line numbers; separate evidence from inference.
- Do not infer readiness from the existence of a file.
- Do not treat technical execution success as proof of pedagogical alignment.
- Do not preserve material solely because it already exists.
- Do not remove a topic solely because it is difficult; explain whether it is necessary and whether its prerequisites are available.
- Distinguish deliberate spiral learning from accidental duplication. A later revisit must clearly deepen or apply the earlier concept.
- Treat demos and assignments as part of the curriculum, not appendices.
- Do not rewrite content during the evidence-gathering pass.
