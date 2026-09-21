# Course material conventions

Adapted from DataSci 223's student-facing lecture and demo style. The course-specific conventions below take precedence over 223's heading, runtime, notebook, and grading choices.

## Audience and presentation

- Write for health data science master's students who are beginners in programming. Define jargon at first use and build knowledge incrementally.
- Students should be able to scan the material and understand a concept on first exposure. Teach how to get something working before adding safeguards.
- Prefer diagrams, screenshots, concrete outputs, and concise bullets over walls of prose. The lecturer speaks to the content; the document is not a speaking script.
- Preserve relevant humor and comics. Keep core explanations clear; place jokes between relevant topics rather than letting them obscure the concept.
- Do not include time estimates, instructor instructions, recap/summary sections, or filler that merely repeats a heading.

## Lecture format

The lecture format comes from DataSci 223's lecture style and formatting guide (`../datasci_223/AGENTS.md`), adapted here: content organization first, styling second. Where the two differ, the course-specific rules below take precedence. The reviewed Lectures 01–03 are the working examples.

- Use Python 3.13 throughout lectures, demos, assignments, and grading. Keep pandas at the course's pinned 3.0.5 version.
- Lectures explain concepts; they are presentation material, not executable documents. Keep prose concise and student-facing, and preserve humor.

### Lecture organization

- Content balance: each major topic blends conceptual foundations, reference material, practical examples, and hands-on demos.
- Build knowledge incrementally. Order topics so each uses only what came before, and teach how to get something working before adding safeguards. Python content follows McKinney's *Python for Data Analysis* progression.
- When a section develops one tool or technique, a good order is: the problem it solves → the basic tool → options and variations → pitfalls and safeguards. Use this where it fits; not every section or subsection needs every step.
- Topics form blocks that each end at a demo break; each demo practices the block before it using only material already taught.
- Keep lectures lean and practical: essential daily data-science tools belong in the lecture; advanced variations, theory, and specialized tools go in `BONUS.md`. The main path comes first, and alternatives (such as standard-library `venv` or Conda beside uv) follow it, labeled as alternatives.
- Everything the demos and the assignment need is taught in this lecture or an earlier one, not only in `BONUS.md`.
- No summary or recap; the lecture ends at its last demo break.
- Lecture 11 is the exception. It introduces few or no new technical concepts; its demo takes most of the session and walks through an example project workflow like the one students complete for Assignment 11, the final exam. The lecture frames that workflow and points back to the lectures that taught each skill. Apply the topic pattern only where it fits, and teach any skill the final needs in the earlier lecture where it belongs rather than as new Lecture 11 content.

### Topic organization

Each major topic usually includes these parts, in this order. Omit a part only when it would be empty or would repeat nearby content; when students need a missing part, write it rather than a placeholder.

1. A freeform introduction, which can be several paragraphs. This is where concepts are explained, analogies drawn, and connections to previous lectures made. Motivate the concept with a concrete problem, ideally from health data, and define each new term in **bold** at first use.
2. A visual, table, or concrete output that shows the idea: a diagram, a before-and-after table, a screenshot of the actual action, or real output. Visuals go before code.
3. A reference card: related commands or functions grouped by task, with their purpose, key arguments, and typical output. Students look things up here after class, so it covers what the demos and assignment use.
4. A short code snippet: the smallest example of one concept, with its expected output and nothing untaught.

Adapt introductions and examples from the course sources rather than inventing them: McKinney's chapters (`work/mckinney_content/`, summarized in `work/mckinney_topics_summary.md`), `work/tlcl_topics.md` and `work/missing_semester_topics.md` for the shell and developer tools, and the previous year's slides and narration in `work/lectures_bkp/` (its lecture numbering differs from the current sequence).

### Where material belongs

- Lecture: concepts, visuals, reference cards, and short single-concept snippets.
- Demo: realistic complexity with health data, multiple steps, edge cases, and visible checkpoints. The lecture does not embed full demo walkthroughs.
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

Freeform introduction, as many paragraphs as the idea needs: the problem this topic solves, the concept explained with an analogy where one helps, each **new term** defined at first use, and the connection to earlier lectures.

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

- Opening lines: the Notion front matter, the local title line as a `#` heading that exactly matches `title_line` (the course site renders it as the page's title heading, and publishing omits it because it matches `title_line`; the Notion page title is set in Notion), the exact line `See [BONUS.md](BONUS.md) for the optional extensions.`, and the demo link: `[Live Demo Guide](demo/DEMO_GUIDE.md)` for Lectures 01–03 or one `**Live notebooks in Colab:**` line for Lectures 04–11. The publishing script replaces the BONUS and demo-guide lines with their Notion child pages. A comic or one-sentence hook may follow; do not add outline, learning-objective, or recap lists, because Notion's page outline already lists the headings.
- Notion-native Markdown: multiple `#` topic headings, `##` subtopics, `###` for reference cards, code snippets, and minor subsections, and `####` only when a `###` subsection needs its own children. Do not skip a level. Use real headings rather than bold labels, no hard wrapping within paragraphs, four-space nesting for lists, and no horizontal rules.
- Demo breaks: exactly `# LIVE DEMO!`, identical at every break, with nothing beneath it. The next line starts a new `#` topic because any `##` heading would nest under the demo marker in Notion. Nothing follows the final marker.
- Name subsections `Reference Card: ...` and `Code Snippet: ...` after the task or concept they cover. Put visuals before code.
- Comics: local images with `xkcd NNNN: Title`-style alt text and an optional short italic caption, placed between topics rather than inside an explanation.

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
3. Topics: Does the introduction explain the concept, draw an analogy where one helps, connect it to earlier lectures, and define new terms, or does the topic jump straight to an API list? Is there a visual or concrete output before code? Does the reference card cover the task? Are snippets minimal and correct for Python 3.13 and pandas 3.0.5, with stated outputs that match real output?
4. Styling: fix violations of the rules above, not matters of taste.

Fix substance: add the missing explanation, visual, or example, adapted from the course sources, and reorder or move material when the organization is wrong.

## Demos and visible checkpoints

- Students must be able to complete demos independently after missing lecture or when repeating them: provide setup, accessible source links, run instructions, expected outputs, and any intentional-error corrections. For short script demos, include the walkthrough and code inline; for longer demos, use linked runnable files or notebooks with the full context and instructions.
- Keep demo markers unadorned. No descriptions underneath, recaps, time estimates, instructor instructions, or content after the final demo break.
- Demo content must use material taught before its corresponding break. Lecture 01 has four breaks: Git/GitHub setup, shell commands, Python basics, control structures/debugging. Use plain `print()` in Lecture 01; formatting with f-strings starts in Lecture 02.
- Lectures 01–03 use scripts and Notion demo walkthroughs. Lectures 04–11 use Markdown-authored, generated notebooks linked through Colab.
- Usually use three demos near the first third, second third, and end of the lecture; preserve Lecture 01's four breaks and Lecture 11's single project walkthrough.
- Build progressive, practical workflows from preceding material. Use realistic data and edge cases where they serve the lesson, without introducing untaught complexity.
- Show intermediate values, tables, diffs, or saved artifacts after meaningful steps. Give students concrete expected results so they can tell what changed and whether it worked.
- If a failure is intentional, label the expected error and show the correction. Otherwise demos must run end-to-end without errors.

## Assignments

- Assignments assess committed output artifacts. Graders do not inspect student source patterns or run student code. Keep artifact instructions, checker expectations, and point totals consistent.
- Keep assignments task-first: annotated scaffold trees, numbered subtasks, and prominent artifact checkpoints. Put a positive, artifact-based Completion Contract under Check Your Work. Avoid preambles, forbidden-code lists, and instructor-facing notes. Call the shared grading rules “checks,” not “public checks.”
- Use VS Code's integrated terminal by default, with native-terminal instructions as a fallback. Assignment checks run automatically on every GitHub push; a new fork may need Actions enabled once.

## Assets and verification

- Use ignored `scratch/` or `tmp/` within this repository for completion tests; do not use the small system temporary disk.
- Keep screenshots and images local and verify their contents before embedding. Preserve current Notion edits when syncing.
- Before editing or pushing local changes to Notion-mapped documents, fetch and compare the latest Notion content with local content. Reconcile user edits first, recheck immediately before publishing, and use targeted updates where possible. After syncing, verify the changed sections and their neighbors, not just the presence of new text; check for duplicates and merged headings.
- Include new child pages, links, images, and native synced blocks in that comparison. Capture unmapped pages with their Notion IDs before publishing; never treat a local file as permission to overwrite unrecognized Notion changes. If the fetch is incomplete or edits conflict, pause that page's sync until reconciled.
- Connect through the configured Notion integration: fetch `self` first to confirm the workspace and available access, then fetch each page by its mapped Notion URL or page ID. Use `scripts/notion_publish.py` only to prepare a reviewed payload; publish with the Notion page-update tool, then fetch the page again to verify the result. Do not use a stale local snapshot as the source of truth.
- Use paths relative to the referring document for assets. Screenshots should show the actual action or state being explained; prefer reusing a relevant local asset over adding decoration.
- Verify every demo break against the content before it. Execute changed scripts, or regenerate and execute changed notebooks, and inspect meaningful outputs rather than relying only on exit status.
- Verify assignments through fresh scratch-directory completions using only preceding lectures and demos. Check the saved artifacts with the authoritative grader, including a valid alternative solution where grading behavior changes.
- Run the Eleventy build for site configuration, templates, navigation, or rendering-sensitive changes; it is not a substitute for content review or demo execution.
