# Course material conventions

Adapted from DataSci 223's student-facing lecture and demo style. The course-specific conventions below take precedence over 223's heading, runtime, notebook, and grading choices.

## Audience and presentation

- Write for health data science master's students who are beginners in programming. Define jargon at first use and build knowledge incrementally.
- Students should be able to scan the material and understand a concept on first exposure. Teach how to get something working before adding safeguards.
- Lectures and demos are not the words said in class. A lecture holds all the content the instructor talks through and students come back to for reference: explanations, visuals, reference cards, and snippets. It is not a speaking script, so it needs no narration, transitions, or asides. A demo is what students redo on their own after the instructor has run it in class, or in place of a class they missed. It gives step-by-step instructions, visual output wherever possible, and expected results, with enough teaching context to work through it alone.
- Prefer diagrams, screenshots, concrete outputs, and concise bullets over walls of prose.
- Humor, comics, and emojis are encouraged when used sparingly: spread a few through each lecture, placed between sections or subsections, and keep core explanations clear rather than jokey. Lecture 03 sets the level: a comic or visual joke about every 150 lines, at least one in each block, each tied to the topic beside it, plus a few one-line jokes. Preserve relevant humor and comics when editing, and do not reuse a comic another lecture shows.
- Do not include time estimates, instructor instructions, recap/summary sections, or filler that merely repeats a heading.

## Lecture format

The lecture format comes from DataSci 223's lecture style and formatting guide (`../datasci_223/AGENTS.md`), adapted here: content organization first, styling second. When the two disagree on something this file does not decide, follow 223. Where the two differ, the course-specific rules below take precedence. The reviewed Lectures 01–03 are the working examples.

- Use Python 3.13 throughout lectures, demos, assignments, and grading. Keep pandas at the course's pinned 3.0.5 version.
- Environments follow Lecture 03: uv with `pyproject.toml` is the course default (`uv venv --seed`, `uv add`, `uv sync`, `uv run`), and `requirements.txt` is the labeled alternative. Never tell students to `uv pip install` into a `pyproject.toml` project, because `uv sync` removes whatever `pyproject.toml` does not list. Handouts and local demo folders ship `pyproject.toml` with a generated `uv.lock`; notebooks keep their Colab `%pip install` cell.
- Commands students type say `python3`, which works on macOS and Linux inside and outside an environment. A uv environment on Windows has no `python3.exe`, so where setup happens, tell PowerShell and Git Bash users to type `python` instead.
- Command lines students paste carry no `#` comments: zsh, the macOS default, passes them to the command as arguments. Put the note in prose, or show a listing meant for reading, such as a script's contents, as a `text` block and say it is not for pasting.
- Lectures explain concepts; they are presentation material, not executable documents. Keep prose concise and student-facing, and preserve humor.

### Lecture organization

- Content balance: each major topic blends conceptual foundations, reference material, practical examples, and hands-on demos.
- Build knowledge incrementally. Order topics so each uses only what came before, and teach how to get something working before adding safeguards. Python content follows McKinney's *Python for Data Analysis* progression.
- When a section develops one tool or technique, a good order is: the problem it solves → the basic tool → options and variations → pitfalls and safeguards. Use this where it fits; not every section or subsection needs every step.
- Topics form blocks that each end at a demo break; each demo practices the block before it using only material already taught. A block is a conceptual group of topics, so a break belongs where one group finishes rather than at a fixed word count.
- An ordinary class has 90 minutes for lecture and demos together, planned around 60 minutes of lecture and 30 minutes of core demo walkthroughs. Reference coverage can exceed what is discussed in depth. Give each demo a compact core walkthrough and clearly identify independent practice to complete outside class; do not treat every demo variation as part of the live route. Preserve the main-lecture teaching of every required assignment concept. Lecture 11 retains its project-workflow exception.
- Pacing edits may remove repetition and shorten examples. Discuss removing actual functionality with the instructor before changing the lecture's scope; being absent from an assignment is not a reason to remove a method. Preserve non-repeated Git guidance and Markdown's role in communication. Keep explanations, useful reference cards, and minimal examples that show each method's functionality in the lecture; put extended workflows in the corresponding demo, with independent practice when appropriate. Do not move functionality to BONUS merely because its example is long.
- A break may move, and the topics around it may be reordered, when that balances the blocks and still leaves each topic using only what came before. Place the break from the lecture, for timing and conceptual grouping, and then move demo sections so each demo practices the block that now precedes it. Demo placement and demo content change together; a demo that would practice untaught material names the sections to move, and is not a reason to leave the break where it is.
- Size: aim for about 850 lines and about 3,000 words of prose (everything outside code fences, tables, and headings) per lecture. Both are soft: a lecture that needs the words to explain something should have them, and a sentence that earns its place is never cut to satisfy the count. Use the numbers to spot bloat, not as a gate. Space the demo breaks roughly evenly, with a little more material in the first block than the later ones. Measure with `python3 scripts/lecture_metrics.py`.
- Keep lectures lean and practical: essential daily data-science tools belong in the lecture; advanced variations, theory, and specialized tools go in `BONUS.md`. The main path comes first, and alternatives (such as standard-library `venv` or Conda beside uv) follow it, labeled as alternatives.
- Everything the core demo routes and the assignment need is taught in this lecture or an earlier one, not only in `BONUS.md`. An optional independent extension may use an instructor-approved BONUS topic when it explicitly links that prerequisite; it must supply no state or skill required by the core route or assignment.
- No summary or recap; the lecture ends at its last demo break.
- Lecture 11 is the exception. It introduces few or no new technical concepts; its demo takes most of the session and walks through an example project workflow like the one students complete for Assignment 11, the final exam. The lecture frames that workflow and points back to the lectures that taught each skill. Apply the topic pattern only where it fits, and teach any skill the final needs in the earlier lecture where it belongs rather than as new Lecture 11 content.

### Topic organization

Each major topic (a `#` section) should usually include these parts, in this order. They describe a major topic, not every subsection: a subsection uses only the parts it needs, and a short one may be just a reference card or a snippet. Skip a part that would be empty or would repeat nearby content; when students need a missing part, write it rather than a placeholder.

1. An introduction, where the concept is explained: what it is, why it matters (the problem it solves, ideally in health data), and the key ideas students need before the details. Keep it short, a few sentences or bullets: the cards and snippets carry the details. It may draw an analogy or connect to an earlier lecture in a clause when that explains the new topic. It introduces _this_ topic: not a story or a recap standing in for the explanation, and not a lead-in about some other topic. Define each new term in **bold** at first use. Prefer concise bullets over long prose.
2. A visual, table, or concrete output that shows the idea: a diagram, a before-and-after table, a screenshot of the actual action, or real output. Visuals go before code.
3. A reference card: related commands or functions grouped by task, with their purpose, key arguments, and typical output. Students look things up here after class, so it covers what the demos and assignment use.
4. A short code snippet: the smallest example of one concept, shown in place, with its expected output and nothing untaught. Leave out setup and boilerplate (imports, seeding, building sample data, environment steps) that the concept does not need; the demos carry the full end-to-end examples.

Adapt introductions and examples from the course sources rather than inventing them: McKinney's chapters (`work/mckinney_content/`, summarized in `work/mckinney_topics_summary.md`), `work/tlcl_topics.md` and `work/missing_semester_topics.md` for the shell and developer tools, and the previous year's slides and narration in `work/lectures_bkp/` (its lecture numbering differs from the current sequence).

### Where material belongs

- Lecture: concepts, visuals, reference cards, and short single-concept snippets.
- Demo: realistic complexity with health data, multiple steps, edge cases, and visible checkpoints. The lecture does not embed full demo walkthroughs. The audience is health data science master's students, so a demo's subject matter is patients, encounters, vitals, labs, or clinic operations, not student rosters, grades, or school subjects. A lecture snippet may keep a simple non-clinical example when it exists only to show syntax; a demo may not. Values carry real units and clinically sensible ranges, and identifiers are synthetic. When extending a demo that still runs on the old student-grade material, recast it rather than adding more of it.
- `BONUS.md`: advanced options, theoretical depth, specialized use cases, and further reading.
- Assignment: independent practice of lecture and demo material; it introduces no new required concept.

### Styling

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

![xkcd 1987: Python Environment](media/xkcd_1987.png)

An optional one-sentence hook.

# First Topic

A short introduction: what the thing is, the problem it solves, and how to use it, with each **new term** defined at first use.

## Subtopic

A diagram, before-and-after table, screenshot, or real output that shows the idea.

### Reference Card: Task-Oriented Name

- `function(arg)`: What it does; typical output.
- `function(arg, option=value)`: What the option changes.

### Code Snippet: What the Code Shows

```python
result = function(arg)
print(result)  # expected output
```

# LIVE DEMO!

# Next Topic
````

- Opening lines: the Notion front matter, the local title line as a `#` heading that exactly matches `title_line` (the course site renders it as the page's title heading, and publishing omits it because it matches `title_line`; the Notion page title is set in Notion), the exact line `See [BONUS.md](BONUS.md) for the optional extensions.`, and the demo link: `[Live Demo Guide](demo/DEMO_GUIDE.md)` for Lectures 01–03 or one `**Live notebooks in Colab:**` line for Lectures 04–11. The publishing script replaces the BONUS and demo-guide lines with their Notion child pages. A comic or one-sentence hook may follow, then a short bulleted list of the reading chapters the lecture covers (McKinney's _Python for Data Analysis_, _The Linux Command Line_, or other course sources, by chapter and section), so students know what to read. Do not add outline, learning-objective, or recap lists, because Notion's page outline already lists the headings. Lectures carry no links to assignments, least of all upcoming ones; the course overview page lists them. Do not add links or callouts pointing to other course pages: Notion's page hierarchy connects them, and course content rarely needs to reference itself.
- Notion-native Markdown: multiple `#` topic headings, `##` subtopics, `###` for reference cards, code snippets, and minor subsections, and `####` only when a `###` subsection needs its own children. Do not skip a level. Use real headings rather than bold labels, no hard wrapping within paragraphs, four-space nesting for lists, and no horizontal rules.
- Demo breaks: exactly `# LIVE DEMO!`, identical at every break, with nothing beneath it. The next line starts a new `#` topic because any `##` heading would nest under the demo marker in Notion. Nothing follows the final marker.
- Name subsections `Reference Card: ...` and `Code Snippet: ...` after the task or concept they cover. Put visuals before code.
- Captions: the link text of an image is its caption. Notion renders `![Caption](path)` as a picture captioned `Caption`, and the site renders the same text as a `figcaption`, so never put a caption in a separate italic paragraph beneath the image. Keep it to one line: what the picture shows or the point it makes, not a restatement of the nearby prose. Do not add inline source citations such as `Screenshots: VS Code documentation`.
- Callouts: a Notion `<callout icon="⚠️" color="yellow_bg">` (warning) or `<callout icon="💡" color="blue_bg">` (tip) holds a point students must not miss, such as an alias that is not a copy or `.venv/` kept out of Git. Its first line is a short `##` heading stating the point, as in Lecture 03, and its body is a few sentences indented with a tab. Use a few per lecture at most.
- Show, don't narrate: when an example makes the point (good versus bad commit messages, a format-spec table), use a code block or table rather than prose describing it. Write italics as `_italic_`.
- Comics: local images placed between topics rather than inside an explanation, captioned `xkcd NNNN: Title` followed by the joke or the point as its own sentence, as in `![xkcd 1987: Python Environment. Virtual environments prevent package chaos](media/xkcd_1987.png)`.

#### Reference card formats

Reference cards contain useful reference material, not just formatting; avoid a table where a sentence suffices. Use a two-column list for most cards (item: purpose and typical output):

```markdown
### Reference Card: Viewing Files

- `head -n 5 FILE`: Print the first five lines.
- `wc -l FILE`: Count lines; output looks like `42 FILE`.
```

Use a table, grouped by task, when arguments and outputs both need a column. Escape a literal pipe inside a table cell as `\|`; the Notion publisher rejects rows with uneven cells.

```markdown
### Reference Card: Loading and Inspecting Data

| Category | Method | Purpose & arguments | Typical output |
| --- | --- | --- | --- |
| Load | `pd.read_csv(path)` | Read a CSV file; `na_values=[...]` marks extra missing-value codes. | `DataFrame` |
| Inspect | `df.head(n)` | Show the first `n` rows (default 5). | `DataFrame` |
| Inspect | `df.info()` | List columns, non-null counts, and dtypes. | Printed summary |
| Summarize | `df.describe()` | Summary statistics for numeric columns. | `DataFrame` |
```

For one complex function, give the signature, purpose, important arguments, and return value:

```markdown
### Reference Card: `pd.merge()`

`pd.merge(left, right, how="inner", on=None, ..., validate=None)` combines two tables by matching key values.

| Argument | Purpose | Effect |
| --- | --- | --- |
| `how` | Rows to keep: `"inner"`, `"left"`, `"right"`, or `"outer"` | Controls which unmatched keys survive |
| `on` | Shared key column or columns | Matches rows with equal keys |
| `validate` | Expected key relationship, such as `"one_to_one"` | Raises `MergeError` when violated |

Returns a new `DataFrame`; `left` and `right` are unchanged.
```

### Reviewing a lecture

Review content organization first and styling last.

1. Map the lecture: its topics in order, the parts each topic has, and the topic block that leads to each demo.
2. Lecture organization: Does each topic blend concepts, reference material, and practical examples? Is the order incremental, with nothing used before it is taught and the working path before safeguards? Where a section develops one tool, does it move from problem to tool to options to pitfalls? Does each block build to its demo, and does the demo use only earlier material? Is the lecture lean, with essential tools in the lecture and advanced material in `BONUS.md`? Does it teach what its demos and assignment need?
3. Topics: Does each major topic's introduction actually explain the concept (what it is, why it matters, the key ideas) and define new terms, without a story, recap, or detour standing in for the explanation? Are subsections free of intros they do not need? Does the topic teach before its reference card, rather than jumping straight to an API list? Is there a visual or concrete output before code? Does the reference card cover the task? Are snippets minimal and correct for Python 3.13 and pandas 3.0.5, with stated outputs that match real output?
4. Styling: fix violations of the rules above, not matters of taste. `python3 scripts/lecture_lint.py` checks the mechanical ones (heading levels, demo markers, pseudo-headings, horizontal rules, and captions) and exits non-zero when a page breaks one.

Fix substance: add the missing explanation, visual, or example, adapted from the course sources, and reorder or move material when the organization is wrong.

## Demos and visible checkpoints

- Students must be able to complete demos independently after missing lecture or when repeating them: provide setup, accessible source links, run instructions, expected outputs, and any intentional-error corrections. For short script demos, include the walkthrough and code inline; for longer demos, use linked runnable files or notebooks with the full context and instructions.
- Keep demo markers unadorned. No descriptions underneath, recaps, time estimates, instructor instructions, or content after the final demo break.
- No setup section stands apart from Demo 1: the first demo starts from a fresh terminal or a new Colab runtime and does its own setup, without cloning the course repository when a download link or one `curl ... | sh` line will do. Each later demo opens with the steps to get back to its folder and environment after the terminal or runtime was closed (Lecture 03 marks them `<span color="yellow_bg">**In a new terminal**</span>`).
- Core demo content must use material taught before its corresponding break. Clearly label optional independent extensions and link any instructor-approved BONUS prerequisite. Lecture 01 has four breaks: Git/GitHub setup, shell commands, Python basics, control structures/debugging. Use plain `print()` in Lecture 01; formatting with f-strings starts in Lecture 02.
- Lectures 01–03 use scripts and Notion demo walkthroughs. Lectures 04–11 use Markdown-authored, generated notebooks linked through Colab.
- A notebook's first cells set it up the way `setup_demo.sh` sets up a folder: they install the course packages and download every non-notebook file it reads (data files, and anything an earlier demo writes) when the file is missing, so the notebook runs alone in a fresh Colab runtime. Lecture 11's demos show the pattern.
- Usually use three demos near the first third, second third, and end of the lecture; preserve Lecture 01's four breaks and Lecture 11's single project walkthrough.
- Build progressive, practical workflows from preceding material. Use realistic data and edge cases where they serve the lesson, without introducing untaught complexity.
- Show intermediate values, tables, diffs, or saved artifacts after meaningful steps. Give students concrete expected results so they can tell what changed and whether it worked.
- If a failure is intentional, label the expected error and show the correction. Otherwise demos must run end-to-end without errors.

## Assignments

- Assignments assess committed output artifacts. Graders do not inspect student source patterns or run student code. Keep artifact instructions, checker expectations, and point totals consistent.
- Grading never checks the Python version: not its value, not whether an artifact records one, and not the format of the line that does. Ignore such lines entirely.
- Homework is pass/fail, and students test it themselves: each homework handout ships a byte-identical copy of its course-owned checks, so `python3 check_assignment.py` gives the full score and says what to fix. Exams (Assignments 05 and 11) are the exception: their handouts ship no automated checks (no checker, no `.github/` workflow), so their artifact instructions must name every path, header, and row count; the course grades them with `NN/assignment_checks/` after the deadline. Grade generously: demonstrated understanding outweighs exact syntax. Accept any artifact that shows the skill, whatever its case, whitespace, missing final newline, number or date format, column order, or equivalent approach; charge each mistake once, judging a later artifact against the student's own earlier one rather than cascading the loss; and still grade only the artifacts.
- A failure message says what the check found beside what it expected and names the likely cause and fix, such as an empty timestamp, `>` overwriting a file, the wrong axis, or positions used where labels belong, so the student's next push is a fix rather than a guess. Checker code runs on any Python 3 a student has, including macOS's system `python3` 3.9, so it avoids `zip(strict=True)`, `match`, and other 3.10+ features.
- Setup follows Lecture 03: the handout ships `pyproject.toml` and `uv.lock`, and students run `uv venv --seed`, activate, and `uv sync`, never `uv init` over the supplied files.
- Keep assignments task-first: annotated scaffold trees, numbered subtasks, and prominent artifact checkpoints. Put a positive, artifact-based Completion Contract under Check Your Work. Avoid preambles, forbidden-code lists, and instructor-facing notes. Call the shared grading rules “checks,” not “public checks.”
- Use VS Code's integrated terminal by default, with native-terminal instructions as a fallback. Assignment checks run automatically on every GitHub push; a new fork may need Actions enabled once.
- Each run downloads the current checker files from this course repository, so a check corrected after handout reaches every student on their next push without touching their fork. The workflow names them in `NN/assignment/.github/workflows/tests.yml` (`CHECKS_REPO: christopherseaman/datasci_217`, `CHECKS_REF`, `CHECKS_PATH: NN/assignment_checks`, and `CHECKS_FILES`), so correct a check in both places: the vendored copy under `NN/assignment/` and the course-owned copy under `NN/assignment_checks/`. Fetch the course-owned set whole and never a partial set, because `grading.py` pairs its checks with `POINTS` and raises on a mismatch at run time even though both files parse. Validate what arrives and smoke-test that it runs; judge a checker by the JSON it emits, never by its exit status, because a working checker exits non-zero for any incomplete submission. The vendored copies stay in place as the fallback, and a run that can use neither fails rather than passing with no tests. Student work is never fetched or overwritten.

## Assets and verification

- Use ignored `scratch/` or `tmp/` within this repository for completion tests; do not use the small system temporary disk.
- Keep screenshots and images local and verify their contents before embedding.
- Use paths relative to the referring document for assets. Screenshots should show the actual action or state being explained; prefer reusing a relevant local asset over adding decoration.
- Verify every demo break against the content before it. Execute changed scripts, or regenerate and execute changed notebooks, and inspect meaningful outputs rather than relying only on exit status.
- Verify assignments through fresh scratch-directory completions using only preceding lectures and demos. Check the saved artifacts with the authoritative grader, including a valid alternative solution where grading behavior changes.
- Run the Eleventy build for site configuration, templates, navigation, or rendering-sensitive changes; it is not a substitute for content review or demo execution.
### Publishing to Notion

- Publish only when the user asks for it, never on your own initiative. A publish replaces the page, so it is theirs to call.
- Reconcile before you write. Fetch the live page and compare it with the local file, including child pages, links, images and synced blocks. Notion holds edits the repository does not, so a local file is never permission to overwrite an unrecognized change; if the fetch is incomplete or the two genuinely conflict, stop and ask. Page titles are set in Notion and may differ deliberately.
- One command does it: `python3 scripts/notion_push.py <page.md>`, or `--dry-run` to rehearse. It pages the child blocks to the end, uploads the page's images, builds the payload, refuses to continue unless every child page survived into it, writes the page, attaches the images, then re-reads and verifies. Check any page any time with `python3 scripts/notion_check_pages.py`, which exits non-zero on a broken image.
- It exists because two steps bite when done by hand, both verified on 2026-09-23. A child page missing from the payload does not vanish quietly: `ntn pages edit` hangs until killed and can leave the page wedged, so a publish that stalls means a missing `<page>` tag rather than a slow API. And `ntn pages edit` does not resolve `file-upload://`, storing each image as an external block with an empty URL, so images need attaching afterwards. An image is broken when its source URL is empty; testing only that the source key exists passes a broken block.
- Media is uploaded from the local files, never linked to GitHub, and uploads expire about an hour after they are created, so upload and publish in one sitting. An image link written as `![Caption](file-upload://ID)` is correct; wrapping the tag inside the link, as in `![Caption](<image src="file-upload://ID"></image>)`, stores a dead text link instead of a picture.
