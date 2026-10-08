# Course material conventions

Adapted from DataSci 223's lecture style guide (`../datasci_223/AGENTS.md`): content organization first, styling second. These rules override 223's heading, runtime, notebook, and grading choices; where this file is silent, follow 223. Lectures 01–04 are the working examples.

## Audience and voice

- Write for health data science master's students who are beginners in programming. Define jargon at first use, in **bold**, and build knowledge incrementally.
- Students should understand a concept on first scan. Teach how to get something working before adding safeguards.
- A lecture is all the content the instructor talks through and students come back to: explanations, visuals, reference cards, snippets. It is presentation material, not a speaking script or an executable document: no narration, transitions, or asides.
- A demo is what students redo alone after class or in place of a missed one: step-by-step instructions, visible output, expected results, and enough context to work through it alone.
- Concise and clear everywhere, assignments included: prefer a few bullets to paragraphs (when listing terms, start each bullet with a bold label), and diagrams, screenshots, and real output over prose. Show rather than describe; drop snippets that exist only to fill the pattern.
- Humor, comics, and emojis, sparingly, between sections rather than inside explanations; core explanations stay clear. Lecture 03 sets the level: a comic or visual joke about every 150 lines, at least one per block, each tied to the topic beside it, plus a few one-line jokes. Keep only one-liners that land. Preserve relevant humor when editing; never reuse a comic another lecture shows.
- No time estimates, instructor instructions, recaps or summaries, or filler that repeats a heading.
- Lecture pages carry no pointers to other course pages beyond their top BONUS and demo links: no links or callouts to other pages, and no sentences promoting BONUS, other lectures, or assignments. Notion's page hierarchy and the course overview connect them. Exceptions: a clause connecting a topic to an earlier lecture, Lecture 11's map back to the lectures that taught each skill, and a demo extension's `Prerequisite: BONUS.md, ...` line.

## Environment and commands

- Python 3.13 and pandas 3.0.5 in lectures, demos, assignments, and CI grading; checker code also stays 3.9-compatible (see Assignments).
- uv with `pyproject.toml` is the default (`uv venv --seed`, `uv add`, `uv sync`, `uv run`), as in Lecture 03; `requirements.txt` is the labeled alternative. Never `uv pip install` into a `pyproject.toml` project: `uv sync` removes whatever `pyproject.toml` does not list. Never `uv init` over supplied files.
- Handouts and local demo folders ship `pyproject.toml` and a generated `uv.lock`; notebooks also keep a Colab `%pip install` cell.
- Commands say `python3`. Windows students work in WSL (Lecture 01): assume macOS, Linux, or WSL, with no PowerShell, Git Bash, or native-terminal variants beyond Lecture 03's activation note.
- Use VS Code's integrated terminal.
- Command lines students paste carry no `#` comments (zsh passes them as arguments). Put the note in prose, or show a listing meant for reading as a `text` block and say it is not for pasting.

## Lectures

### Organization

- Each major topic blends concepts, reference material, practical examples, and hands-on demos.
- Order topics so each uses only what came before. Python follows McKinney's _Python for Data Analysis_ progression.
- A section that develops one tool fits, where it helps: problem → basic tool → options → pitfalls and safeguards.
- Topics form blocks, each ending at a demo break where a conceptual group finishes, not at a word count. Usually three demos near the first third, second third, and end; Lecture 01 keeps four (Git/GitHub setup, shell, Python basics, control structures and debugging); Lecture 11 keeps one.
- Place breaks from the lecture (timing, grouping), then move demo sections so each demo practices the block before it. A break may move and topics may be reordered to balance blocks. Demo placement and content change together; a demo that needs untaught material names the sections to move; it is never a reason to leave the break where it is.
- Class is 90 minutes: about 60 of lecture and 30 of core demo walkthroughs. Reference coverage may exceed what is discussed. Each demo has a compact core route; label independent practice for outside class, since not every variation is part of the live route. Lecture 11 keeps its project-workflow exception.
- Size: about 850 lines and 3,000 prose words (outside code, tables, headings), with breaks roughly evenly spaced and a little more in the first block. Soft targets for spotting bloat, never a reason to cut a sentence that earns its place. Measure with `python3 scripts/lecture_metrics.py`.
- Pacing edits remove repetition and shorten examples. Ask the instructor before removing functionality; absence from an assignment is no reason to remove a method, and a long example is no reason to move it to BONUS. Keep non-repeated Git guidance and Markdown's role in communication. Keep the main-lecture teaching of every required assignment concept, plus explanations, useful cards, and minimal examples of each method; put extended workflows in the demo, with independent practice where it fits.
- Essential daily tools in the lecture; advanced variations, theory, and specialized tools in `BONUS.md`. The main path first, alternatives (such as `venv` or Conda beside uv) after it, labeled.
- Core demos and the assignment need only this lecture and earlier ones, never only `BONUS.md`. An optional extension may use an instructor-approved BONUS topic if it links that prerequisite and supplies nothing the core route or assignment needs.
- The lecture ends at its last demo break.
- Lecture 11 is the exception: few or no new concepts; its demo fills most of the session with a project workflow like Assignment 11 (the final). The lecture frames that workflow and points back to the lectures that taught each skill, using the topic pattern only where it fits. Teach any skill the final needs in its earlier lecture.

### Topic pattern

A major topic (`#`) usually has these parts, in order. A subsection uses only what it needs (a short one may be just a card or a snippet). Skip a part that would be empty or repeat nearby content; write a missing part students need rather than a placeholder.

1. Introduction: what it is, why it matters (ideally in health data), and the key ideas. A few sentences or bullets; an analogy or a connection to an earlier lecture in a clause when that explains it. It introduces _this_ topic: no story, recap, or lead-in to another topic.
2. Visual: a diagram, before-and-after table, screenshot of the actual action, or real output. Visuals before code.
3. Reference card: commands or functions grouped by task, with purpose, key arguments, and typical output. It covers what the demos and assignment use.
4. Code snippet: the smallest example of one concept, in place, with its expected output and nothing untaught. No setup or boilerplate (imports, seeding, building data, environment steps); demos carry full examples.

Adapt intros and examples from the course sources rather than inventing them: McKinney (`work/mckinney_content/`, summarized in `work/mckinney_topics_summary.md`), `work/tlcl_topics.md` and `work/missing_semester_topics.md` for shell and developer tools, and last year's slides and narration in `work/lectures_bkp/` (different lecture numbering).

### Where material belongs

- Lecture: concepts, visuals, reference cards, short single-concept snippets; no full demo walkthroughs.
- Demo: realistic health data, multiple steps, edge cases, visible checkpoints.
- `BONUS.md`: advanced options, theory, specialized uses, further reading.
- Assignment: independent practice of lecture and demo material; no new required concept.

### Page format

````markdown
---
notion:
  title_line: "# Lecture Title"
  role: lecture
  status: mapped
  page_id: "…"
  url: "…"
---

# Lecture Title

See [BONUS.md](BONUS.md) for the optional extensions.

[Live Demo Guide](demo/DEMO_GUIDE.md)

![xkcd 1987: Python Environment. Virtual environments prevent package chaos](media/xkcd_1987.png)

This lecture covers McKinney, _Python for Data Analysis_ (3rd ed.):

- 4.1 (NumPy arrays)

# First Topic

A short introduction, with each **new term** defined at first use.

## Subtopic

A diagram, before-and-after table, screenshot, or real output.

### Reference Card: Task-Oriented Name

- `function(arg)`: What it does; typical output.

### Code Snippet: What the Code Shows

```python
result = function(arg)
print(result)  # expected output
```

# LIVE DEMO!

# Next Topic
````

Opening lines, in order:

1. Notion front matter, then a `#` title line exactly matching `title_line` (the site shows it; publishing omits it; the Notion title is set in Notion).
2. Exactly `See [BONUS.md](BONUS.md) for the optional extensions.`
3. The demo link. Lectures 01–03: `[Live Demo Guide](demo/DEMO_GUIDE.md)`. Lectures 04–11: one `**Live notebooks in Colab:**` line linking each demo, then `**Run locally:**`, the five-line block (`curl -fsSL .../NN/demo/setup_demo.sh | sh`, `cd ~/NN-demo`, `uv venv --seed`, `source .venv/bin/activate`, `uv sync`), and the line “→ Then open the `NN-demo` folder in VS Code.” Publishing replaces the BONUS and demo-guide lines with child pages.
4. Optionally a comic or one-sentence hook.
5. The readings. One source: `This lecture covers <Author, _Title_ (edition)>:` (for example, `This lecture covers McKinney, _Python for Data Analysis_ (3rd ed.):`), then one bullet per section or chapter. Several sources: `This lecture covers:` then one bullet per source with nested bullets per section or chapter.

No outline, learning-objective, or recap lists (Notion's outline lists the headings).

Headings and Markdown (Notion-native):

- `#` topics, `##` subtopics, `###` for cards, snippets, and minor subsections, `####` only under a `###` that needs children. Never skip a level.
- Real headings, not bold labels. No hard wrapping, four-space list nesting, no horizontal rules. Italics as `_italic_`.
- Name subsections `Reference Card: ...` and `Code Snippet: ...` after their task or concept.

Demo breaks:

- Exactly `# LIVE DEMO!` at every break. Lectures 01–03: nothing beneath it. Lectures 04–11: exactly one line, `[Open Demo N in Colab](https://colab.research.google.com/github/christopherseaman/datasci_217/blob/main/NN/demo/FILE.ipynb)`.
- The next heading is a new `#` topic (a `##` would nest under the marker in Notion). Nothing follows the final marker.

Images and comics:

- The link text is the caption (Notion caption and site `figcaption`); never a separate italic caption paragraph. One line: what it shows or the point, not a restatement of the prose. No source citations such as `Screenshots: VS Code documentation`.
- Comics are local images between topics, captioned `xkcd NNNN: Title. The joke or point.`, as in the example above.

Callouts:

- A Notion `<callout icon="..." color="...">` holds a point students must not miss, such as an alias that is not a copy or `.venv/` kept out of Git. Any fitting icon and color works; common ones are ⚠️ yellow for a warning and 💡 blue for a tip. A few per lecture at most.
- First line: a short `##` heading on the issue, as in Lecture 03. Body: a few tab-indented sentences, or none when the heading says it all.

Jupyter content (Lectures 04+, lecture snippets included) shows tables with `display()` or a bare last expression, not `print()`, except where a script is the point.

#### Reference card formats

Cards hold useful reference material; no table where a sentence suffices. Most cards are a list (item: purpose and typical output):

```markdown
### Reference Card: Viewing Files

- `head -n 5 FILE`: Print the first five lines.
- `wc -l FILE`: Count lines; output looks like `42 FILE`.
```

Use a table grouped by task when arguments and outputs both need a column. Escape a literal pipe in a cell as `\|`; the Notion publisher rejects uneven rows.

```markdown
### Reference Card: Loading and Inspecting Data

| Category | Method | Purpose & arguments | Typical output |
| --- | --- | --- | --- |
| Load | `pd.read_csv(path)` | Read a CSV; `na_values=[...]` marks extra missing codes. | `DataFrame` |
| Inspect | `df.info()` | Columns, non-null counts, dtypes. | Printed summary |
```

For one complex function: signature, purpose, key arguments, return value.

```markdown
### Reference Card: `pd.merge()`

`pd.merge(left, right, how="inner", on=None, ..., validate=None)` combines two tables by matching keys.

| Argument | Purpose | Effect |
| --- | --- | --- |
| `how` | `"inner"`, `"left"`, `"right"`, or `"outer"` | Which unmatched keys survive |
| `validate` | Expected relationship, such as `"one_to_one"` | Raises `MergeError` when violated |

Returns a new `DataFrame`; `left` and `right` are unchanged.
```

### Reviewing a lecture

Organization first, styling last. Fix substance (add the missing explanation, visual, or example from the course sources; reorder or move material) rather than only reporting it.

1. Map it: topics in order, the parts each has, and the block leading to each demo.
2. Organization: blended topics; incremental order with the working path before safeguards; problem → tool → options → pitfalls where it fits; each block builds to a demo that uses only earlier material; lean, with advanced material in BONUS; teaches what its demos and assignment need.
3. Topics: intros explain (what, why, key ideas, new terms) without story, recap, or detour; subsections carry no unneeded intros; teaching comes before the card; a visual precedes code; cards cover the task; snippets are minimal and correct for Python 3.13 and pandas 3.0.5, with outputs matching real output.
4. Styling: fix rule violations, not taste. `python3 scripts/lecture_lint.py` checks heading levels, demo markers, pseudo-headings, horizontal rules, and captions, and exits non-zero on a violation.

## Demos

- Lectures 01–03: scripts with a Notion walkthrough (`demo/DEMO_GUIDE.md`); short demos inline their walkthrough and code; longer ones link runnable files with full context and instructions. Lectures 04–11: Markdown-authored notebooks (generated with Jupytext), linked through Colab.
- Subject matter is patients, encounters, vitals, labs, or clinic operations, never rosters, grades, or school subjects (a lecture snippet may use a non-clinical example only to show syntax). Real units, clinically sensible ranges, synthetic identifiers. Recast old student-grade demos rather than extending them.
- Redoable alone: setup, accessible source links, run instructions, expected outputs, and corrections for intentional errors.
- Demo 1 does its own setup from a fresh terminal or Colab runtime (no separate setup section), with a download link or one `curl ... | sh` line rather than cloning the course repo. Lectures 01–03: each later demo opens with the steps back to its folder and environment, marked `<span color="yellow_bg">**In a new terminal**</span>`. Notebooks set themselves up (below).
- Notebook opening: title, short intro, one sentence (run the cells top to bottom), and one install cell, `%pip install -q --no-warn-conflicts ...` (the flag silences Colab's harmless `google-colab requires pandas==...` conflict), whose one-line comment carries any caveat (restart prompt, a platform limit such as Lecture 10's). No route table, local setup commands, or install Expect paragraph; the lecture page carries those.
- A notebook runs alone in a fresh Colab runtime: its first cells install packages and download every non-notebook file it reads (data, and anything an earlier demo writes) when missing, as Lecture 11's demos do.
- Core content uses only material taught before its break. Label optional extensions and link any approved BONUS prerequisite.
- Lecture 01 uses plain `print()`; f-strings start in Lecture 02.
- Build progressive workflows from preceding material, with realistic data and edge cases where they serve the lesson.
- Show intermediate values, tables, diffs, or saved files after meaningful steps, with concrete expected results.
- Label an intentional failure's expected error and show the fix. Otherwise demos run end to end without errors.

## Assignments

### Handout

- README order: Overview (a brief task summary, then the dataset), Setup, Files, the tasks, Check your work.
- Task-first: annotated scaffold tree, numbered subtasks, each step paired with what to expect, prominent artifact checkpoints. No preambles, forbidden-code lists, instructor notes, or repeated explanations.
- Setup follows Lecture 03 (`uv venv --seed`, activate, `uv sync` on the supplied `pyproject.toml` and `uv.lock`). Terminal and git-identity setup appear only in Assignment 01, git identity as a fallback for when a commit or push asks for it. No Windows or troubleshooting paragraphs.
- Check your work is very brief. Homework notebooks list both the manual `python3 check_assignment.py` run and the notebook's final `run_checks()` cell (GitHub runs the latest checks on push), then commit the outputs. Exams: list the expected files, then commit and push; no separate Submit section. The positive, artifact-based completion contract lives in a linked `CHECKS.md`. Say "checks," not "public checks."
- Checks run on every push; a new fork may need Actions enabled once.

### Checks and grading

- Grade committed artifacts only: never inspect source patterns or run student code. Keep artifact instructions, checks, and point totals consistent.
- Never check the Python version: not its value, its presence, or its line format. Ignore such lines.
- Grade generously: accept any artifact that shows the skill, whatever its case, whitespace, final newline, number or date format, column order, or equivalent approach. Charge each mistake once, judging a later artifact against the student's own earlier one.
- A failure message states what was found beside what was expected, and the likely cause and fix (an empty timestamp, `>` overwriting a file, the wrong axis, positions used for labels).
- Checker code runs on any Python 3, including macOS's system 3.9: no `zip(strict=True)`, `match`, or other 3.10+ features.
- Homework is pass/fail and self-tested: the handout ships a byte-identical copy of its course-owned checks (`NN/assignment/` and `NN/assignment_checks/`); correct both.
- `check_assignment.py` downloads the latest course checks from `main` (falling back to the bundled copy offline or on failure). Assignments 02–04 and 06–10 also expose `run_checks()` for a notebook's last cell; 01, 05, and 11 do not. `DS217_LOCAL_CHECKS` (or `--local-checks`) skips the download; CI sets it.
- The CI workflow (`NN/assignment/.github/workflows/tests.yml`: `CHECKS_REPO: christopherseaman/datasci_217`, `CHECKS_REF`, `CHECKS_PATH: NN/assignment_checks`, `CHECKS_FILES`) downloads the current checks each run, so fixes reach forks on their next push. Fetch the set named by `CHECKS_FILES` whole (`grading.py` pairs checks with `POINTS` and raises on a mismatch), validate and smoke-test it, and fall back to the vendored copy; a run that can use neither fails. Never fetch or overwrite student work.
- Judge a checker by its JSON, never its exit status (it exits non-zero on any incomplete submission).
- Exams (Assignments 05 and 11) ship no checker or `.github/` workflow, so their instructions name every path, header, and row count; the course grades them with `NN/assignment_checks/` after the deadline.

## Verification

- Use ignored `scratch/` or `tmp/` in this repo, not the system temporary disk.
- Keep images local, with paths relative to the referring document; verify their contents before embedding. Screenshots show the actual action or state; reuse a relevant local asset over adding decoration.
- Check every demo break against the content before it. Execute changed scripts, or regenerate and execute changed notebooks, and inspect outputs, not just exit status.
- Verify assignments with fresh scratch completions using only preceding lectures and demos, graded by the authoritative checks, plus a valid alternative solution where grading changes.
- Run the Eleventy build for site configuration, template, navigation, or rendering changes; it does not replace content review or demo execution.

## Publishing to Notion

- Publish only when the user asks; a publish replaces the page.
- Reconcile first: fetch the live page and compare it with the local file, including child pages, links, images, and synced blocks. Notion holds edits the repo lacks; never overwrite an unrecognized change. If the fetch is incomplete or they conflict, stop and ask. Notion page titles may differ deliberately.
- Publish with `python3 scripts/notion_push.py <page.md>` (`--dry-run` to rehearse): it pages child blocks to the end, uploads images, refuses unless every child page is in the payload, writes, attaches images, and verifies. Audit with `python3 scripts/notion_check_pages.py` (non-zero on a broken image).
- Why the script: a payload missing a child page makes `ntn pages edit` hang until killed and can wedge the page (a stalled publish means a missing `<page>` tag, not a slow API); and `ntn pages edit` stores `file-upload://` images as external blocks with empty URLs, so images are attached afterwards. An image is broken when its source URL is empty, not merely when the key is missing.
- Upload media from local files, never GitHub links. Uploads expire about an hour after creation: upload and publish in one sitting. Write `![Caption](file-upload://ID)`; `![Caption](<image src="file-upload://ID"></image>)` stores a dead text link.
