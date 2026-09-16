# Course material conventions

Adapted from DataSci 223's student-facing lecture and demo style. The course-specific conventions below take precedence over 223's heading, runtime, notebook, and grading choices.

## Audience and presentation

- Write for health data science master's students who are beginners in programming. Define jargon at first use and build knowledge incrementally.
- Students should be able to scan the material and understand a concept on first exposure. Teach how to get something working before adding safeguards.
- Prefer diagrams, screenshots, concrete outputs, and concise bullets over walls of prose. The lecturer speaks to the content; the document is not a speaking script.
- Preserve relevant humor and comics. Keep core explanations clear; place jokes between relevant topics rather than letting them obscure the concept.
- Do not include time estimates, instructor instructions, recap/summary sections, or filler that merely repeats a heading.

## Lecture structure and reference cards

- Use Python 3.13 throughout lectures, demos, assignments, and grading. Keep pandas at the course's pinned 3.0.5 version.

- Lectures explain concepts; they are presentation material, not executable documents. Keep prose concise, student-facing, and preserve humor.
- Use Notion-native Markdown: multiple `#` topic headings, `##`/`###` subsections, and no hard wrapping within paragraphs. Use actual headings rather than bold labels for subsections.
- Major topics should usually include a short conceptual introduction, a relevant visual/table/output, a reference card, and a short code snippet. Use unordered lists for two-column reference cards (item and purpose); retain tables when additional columns help. These are guidelines: avoid empty templates, duplicated content, or a table where a sentence suffices.
- Reference cards group related commands/functions and explain purpose, arguments, and typical output where useful. Put visuals before code. Use descriptive `Reference Card: ...` and `Code Snippet: ...` subheadings.
- For a related-method card, group methods by task and give their arguments and typical output. For a complex single function, show its signature, purpose, important parameters, and return value. Reference cards contain useful reference material, not just formatting.
- Keep lecture code snippets short and focused on one concept. Put full workflows in demos. Use actual subsections instead of bolded fake headings, four-space nesting for local Markdown lists, and no horizontal rules.

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
- Use paths relative to the referring document for assets. Screenshots should show the actual action or state being explained; prefer reusing a relevant local asset over adding decoration.
- Verify every demo break against the content before it. Execute changed scripts, or regenerate and execute changed notebooks, and inspect meaningful outputs rather than relying only on exit status.
- Verify assignments through fresh scratch-directory completions using only preceding lectures and demos. Check the saved artifacts with the authoritative grader, including a valid alternative solution where grading behavior changes.
- Run the Eleventy build for site configuration, templates, navigation, or rendering-sensitive changes; it is not a substitute for content review or demo execution.
