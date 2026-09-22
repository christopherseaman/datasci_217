# Handoff

## Current state

- The last committed `main` state is clean and pushed. The latest commit is `ad763a4` (`docs: rebalance lectures and capture WSL troubleshooting`).
- The course is standardized on Python 3.13 and pandas 3.0.5.
- Notion is the primary publishing surface. Course Markdown uses Notion-native headings: multiple `#` topic headings, `##` subsections, and unwrapped paragraphs.
- Lectures 01–03 have been recently reviewed and revised. Lecture 01 now includes the four independent demo breaks, beginner setup guidance, VS Code and terminal workflows, fork/clone and submission guidance, WSL troubleshooting, and the introductory Python material needed for Assignment 01.
- Lecture 02 now covers the command-line workflow, shell shortcuts, Git/GitHub, and functions/modules in the intended sequence. Its demo guide and Assignment 02 were checked against the lecture.
- Lecture 03 and its demo guide were aligned with the current Python 3.13 workflow and synced to Notion.
- The Fall 2026 assignment repositories `ds217-26f-01` through `ds217-26f-11` are public and use artifact-based GitHub Actions checks.

## Recent validation

- The Eleventy site build passed.
- Lecture 01 and Lecture 02 demo checks passed.
- The WSL troubleshooting screenshots were checked locally and the replacement screenshot no longer contains the earlier private email address.
- Recent Notion updates were fetched before publishing and refetched afterward to verify structure, child pages, links, and neighboring content.

## Open course work

- `FIXME.md` records the remaining package-management alignment work for later lectures, demos, and assignments: document the shared `uv`/pip workflow for Python 3.13 and regenerate affected notebooks.
- Review every lecture against "Lecture format" in `AGENTS.md` (`CLAUDE.md` is a symlink to it), content organization first: lecture organization, topic organization, where material belongs, and then styling, following its "Reviewing a lecture" checklist. Where the format exposes a missing explanation or example, add the needed content rather than making a cosmetic formatting change.

## Lecture review (started 2026-09-18)

- Audit and skeptic results for every lecture are in `scratch/review/` (`audit_run1.json`, `verify_01_03.json`, `LECTURE_PLAN.md`). The Notion snapshots used for reconciliation are in `scratch/review/NN/notion_current.md`; the edit, review, and fix-up reports are `apply_run1.json` and `fixup_run1.json`.
- Status: all eleven lectures are reviewed and edited locally on branch `cleanup-26` (README, BONUS, media, plus `02/LECTURE_01_CATCHUP.md`), each with an independent review pass and a fix-up pass. The bonus pages of Lectures 02 and 04–11 also had a cleanup pass (real headings, seeded `default_rng`, runnable examples). Nothing is published to Notion.
- Lecture 01 was already delivered, so its additions are repeated in `02/LECTURE_01_CATCHUP.md`, a temporary catch-up page linked from Lecture 02's opening and rendered at `/02/lecture-01-catchup/` (mapped in `.eleventy.js`; front matter marks it `status: unmapped` for Notion).
- Instructor decisions recorded on 2026-09-21: keep the Lecture 02 Command-Line Catalog; keep Ctrl+E and drop Ctrl+K in the shell shortcut card; finish `cat` input with Enter then Ctrl+C, not Ctrl+D; keep `.python-version` (Assignment 03 grades it) but demote it to one line and point to `uv init`/`pyproject.toml`; title lines use the `# Title` form matching `title_line`.
- uv has no Jupyter magic: its docs recommend `!uv pip install` / `!uv add` in a cell and `uv run --with jupyter jupyter lab`, while `%pip install -r requirements.txt` stays the portable form that also works in Colab. Relevant to the `FIXME.md` package-management work.
- Notion images repaired on 2026-09-22 (instructor-approved, images only, nothing else on those pages touched): 16 dangling `file-upload://` links became real uploaded image blocks (Lecture 05's IQR figure; xkcd 1945, 1845, 833, 1138 in 07; 2533 and 2523 in 08; 2048 and 2289 in 09; 1838, 882, 539, 1725, 2400, 2169 in 10; 2582 in 11), and 7 images missing entirely were inserted (the four fork/clone screenshots in the Lecture 01 demo guide, Lecture 02's bonus branch diagram, and `ols_residuals.png` and `trees.webp` in Lecture 10). A full re-scan reports every mapped page clean.
- Still missing from Notion, deliberately: `vscode-jupyter-kernel-picker.png` and `jupyterlab-interface.png` (04), `Nightingale-mortality-1600.jpg`, `distribution_reference.png` and `altair_study_reference.png` (07), `viz_temp_monthly.png` (09), and `ols_residuals_vs_fitted.png` (10). Each belongs to text the review rewrote, so the old Notion pages have no matching place for them (fuzzy anchor scores 0.25-0.64 against 0.9+ for the ones inserted). They arrive when the reviewed lectures are published.
- These images are Notion uploads. A later publish from local Markdown rewrites images to raw.githubusercontent.com URLs, which will replace them; that is fine, since the files are in the repository.
- Tooling: `scratch/review/notion_bad_images.py` (scan), `notion_plan_missing.py` (locate where an image belongs), `notion_repair_images.py` (upload, attach, optionally delete the dangling block; `--apply` to write), `notion_after.py` (inspect a block's neighbours). Plans and results: `bad_images.txt`, `bad_images_after.txt`, `broken_ops.json`, `missing_ops.json`.
- Validation so far: scratch Eleventy build passes and every image and internal link in the 22 built lecture and bonus pages resolves (`scratch/review/check_built_links.py`); `scripts/notion_publish.py` dry runs pass for every edited file; `scripts/test_notion_publish.py`, `test_lecture01_demos.py`, and `test_lecture02_demos.py` pass; changed snippets were run on Python 3.13 / pandas 3.0.5 by the edit and review agents.
- Open lecture items:
    - New media files are untracked and must be committed with the lecture edits: `07/media/Nightingale-mortality-1600.jpg`, `08/media/xkcd_1319.png`, `09/media/viz_temp_monthly.png`, `10/media/ols_residuals_vs_fitted.png`.
    - Now unreferenced: `06/media/xkcd_2083.png` (actually xkcd 2054, which Lecture 05 uses), `06/media/diagram1.svg`, and `06/media/diagram2.svg`.
    - Bonus cleanup done for 02 and 04–11. Lecture 01's and 03's bonus pages had no bold pseudo-headings; Lecture 06 keeps four `**Gotcha:**`-style paragraph lead-ins, and Lecture 03's bonus keeps one labelled legacy `np.random` example on purpose.
    - Lecture 05's Notion-uploaded IQR figure cannot be fetched through the Notion tools (only `file-upload://`); download it by hand if it should live in `05/media/`.
    - Lecture 11's results table shows the baseline's test row (MAE 32.3, RMSE 50.2, recomputed from the demo outputs); Demo 4 does not print it until deferred item 11-07 is done.
    - Break balance that only demo changes can fix: 04's second break at about 81%, 08's first at about 54%, 09's at about 43% and 75%.
    - Not verified: Colab menu labels in Lecture 04; the new `plt.show()` then `plt.close(fig)` order in a live Jupyter kernel (Lecture 10).
    - Out of scope, for the instructor: `07/POINTS.md` and `08/POINTS.md` cue sheets follow the old order; `11/assignment/README.md` runs `./download_data.sh` while the lectures teach `bash file.sh`; the final's Q1 needs a SHA-256 hash, which only the Lecture 11 demo teaches.
- Scope decisions: this pass edits lecture `README.md` and `BONUS.md` only. Gaps between lectures and demos or assignments are fixed on the lecture side (teach the missing material, or move topics ahead of the demo break); published assignments stay unchanged. Lecture 11 frames the project workflow and points back; skills the final needs are taught in the lecture that owns them.
- Notion images: Lecture 05's "uploaded" IQR figure is not an image at all. The block is a paragraph whose text is `!IQR Method for Outlier Detection` linked to the literal string `<image src="file-upload://3dcd9fdd-1a1a-8184-8879-00b2bde41d2e">`, and that upload id returns 404 from the file-uploads API, so it is a dangling link left by an earlier publish. The local `05/media/boxplot_vs_pdf.png` (a boxplot with the 1.5xIQR fences over a normal curve) is the real figure and should win. Lecture 09's xkcd 2048 block is the same: `!xkcd 2048: Curve-Fitting` linked to `<image src="file-upload://3dcd9fdd-1a1a-8187-9789-00b28d066dbe">`. So Notion holds no uploaded course images; these pages currently show broken image links, and publishing from local would repair them.
- Repairing a Notion image with the CLI (working recipe, used for the Lecture 05 IQR figure on 2026-09-22):
    1. `NOTION_KEYRING=0 ntn files create --filename NAME --content-type image/png --json < path/to/file` returns a file_upload `id`; it expires one hour after creation, so upload immediately before attaching.
    2. `NOTION_KEYRING=0 ntn api /v1/blocks/<page-id-no-dashes>/children -X PATCH -d @body.json < /dev/null` where the body is `{"position": {"type": "after_block", "after_block": {"id": "<block>"}}, "children": [{"object": "block", "type": "image", "image": {"type": "file_upload", "file_upload": {"id": "<upload>"}, "caption": [...]}}]}`. Two gotchas: redirect stdin from /dev/null or the CLI blocks waiting for a body, and the parameter is `position`, not `after` (this API version rejects `after`).
    3. `NOTION_KEYRING=0 ntn api /v1/blocks/<broken-block-id> -X DELETE < /dev/null` removes the dangling paragraph once the image is in place.
- The `ntn` CLI (`NOTION_KEYRING=0 ntn ...`) is authenticated as the "Notion CLI" bot and reaches the public API: `ntn api /v1/blocks/<id-without-dashes>/children` walks a page, `ntn files list` shows real uploads. Helper scripts: `scratch/review/notion_images.py`, `notion_find_upload.py`, `notion_context.py`, `notion_block_json.py`. Paging a lecture page takes a few minutes.
- Do not publish or push any of these changes to Notion (instructor instruction, 2026-09-18): a push would lose content that exists only in Notion. All review edits are local; `scripts/notion_publish.py` is used only as a local dry run.
- If the instructor later asks to publish: Lectures 04, 05, 06, and 10 report `page_last_edited_at` about 75 minutes after the fetched content's as-of time, so refetch immediately before publishing. Lecture 04 shows inline code inside italics as bold in the Notion export; confirm in the Notion UI.
- Deferred to the demo pass (after the lectures are settled), including factual demo errors. After the lecture edits, each lecture's remaining demo items that use material not taught before their break are listed in `scratch/review/apply_run1.json` (`demo_alignment` for each lecture).
    - 04-13: Demo 1's core exercise cannot be done independently.
    - 04-14: The deterministic-sort checkpoint shows nothing.
    - 04-15: Demo 3 opens with a 50-line bootstrap built on material no lecture teaches: pathlib, a while-True upward search function, SHA-256 checksums, and urlretrieve.
    - 04-16: The lecture's only demo links open Colab, and the demos call Colab 'the default launch experience'.
    - 05-04: Demo 3 does not practice the skill its block teaches, which is also the skill the midterm grades: state a contract, keep the raw table, record decisions, transform a copy, run checks that stop the save, and read back.
    - 05-16: Two of Demo 1's three 'strategy' checkpoints show no change, so students cannot see what the step did.
    - 06-05: Demo 3 relies on untaught `concat(keys=)`, MultiIndex selection, date slicing, and horizontal `join='inner'`; `keys=` contradicts the assignment's provenance column.
    - 06-06: Interpretation text contradicts output: Monitor (not Laptop) is lowest turnover, January (not March) grows most, and customers→purchases is one-to-many.
    - 06-07: The provenance flag labels June as 'actual', but June comes only from the estimated table.
    - 06-09: Demo 2 never gives an expected result (row count, shape or equality check), and the notebooks ship without outputs, so a student working alone can't tell whether the melt or round trip worked.
    - 07-03: No demo practices the lecture's core skill: stating a contract, critiquing a flawed chart, redesigning it honestly, and writing a text alternative.
    - 07-18: About half of Demo 2 practices material the lecture calls optional or does not teach at all: regression fits (regplot, pairplot/jointplot kind='reg'), barplot means with 95% CI error bars, and four jointplot variants.
    - 07-19: Demo 1 uses untaught tools and contradicts the lecture's design guidance.
    - 07-M2: Demo 2's first cell titles a chart 'Random Data Over Time', but its x-axis is row numbers 0-99 of random normal columns.
    - 08-06: Demo 3 practices none of the lecture's remote skills; its SSH, tmux, and 'remote workflow' sections only print strings or sleep.
    - 08-11: Both demos use material that is not taught before their breaks: Lecture 09 datetime tools, imshow heatmaps with manual annotation loops (Lecture 07 teaches sns.heatmap), and BONUS-only pivot features.
    - 09-04: No demo practices the entity-aware, past-only, and availability topic, even though the assignment's Question 3 and part of Question 2 depend on it.
    - 09-15: All three notebooks open directly on an imports cell, with no title, purpose, setup note, or run instructions.
    - 09-16: Demo 1 labels 49 hourly rows as 'First 3 days' (two days plus one hour), repeating the inclusive-endpoint bug the lecture has already fixed.
    - 10-14: Moving boosting from block 2 to block 3 (lecture reorder) needs Demo 2/3 changes first.
    - 10-15: Demo 1 relies on untaught statsmodels features: `bse`, F-statistic, adjusted R², AIC/BIC, `C()` categorical terms, and `get_prediction().conf_int(obs=True)`.
    - 10-16: Demo 2 prints RMSE in the wrong units, describes R² as bounded 0-1 even though the assignment asks students to interpret a negative validation R², and fits Lasso on unscaled features.
    - 10-18: The demos model house prices and wine cultivars in a health data science course.
    - 10-22: Each demo opens with a learning-objectives list and closes with 'Key Takeaways' and 'Next Steps' recap lists, which course conventions exclude.
    - 10-M2: Assignment 10 Q2 (35 of 100 points) grades a prediction contract, a feature-availability audit, and a chronological train/validation/test manifest, and Assignment 11 Q6 grades fixed chronological splits.
    - 11-06: Demo 4 says Ridge has no `random_state` parameter; false in scikit-learn 1.9.0, and copying it fails the final's check.
    - 11-07: The lecture's candidate claim is that history and calendar features 'may beat a same-hour-last-week baseline', but Demo 4 reports only the selected pipeline on test.
    - 11-11: The single demo block uses several APIs that no lecture teaches, so a student repeating it alone meets unexplained code: Parquet I/O, `json`, `np.select`, MultiIndex construction, `tz_localize(None)`, `np.clip`, and `DataFrame.pop`.
    - 01-15: The capstone 'measurement workflow' demo processes student test scores.
    - 01-M1: Demo 4 tells students to type indentation at the `...` prompt; Python 3.13's REPL auto-indents, so if/elif blocks raise SyntaxError and loop bodies IndentationError (the automated check misses it).
    - 01-M2: Demo 4.4 says to remove the `#` from an error line; that leaves a leading space and gives `IndentationError: unexpected indent` instead of the intended error.
    - 02-19: Demo 2 practices only functions and imports.
    - 02-20: A student repeating the demo alone is told to start 'from the repository root' and to 'open the course's 02/demo folder', but never how to get the course repository.
    - 03-01: The block before break 2 builds up to the lecture's core NumPy skill: creating arrays, reading shape/ndim/size/dtype, converting text with astype(), and element-wise arithmetic.
    - 03-11: The data is seeded, so every value is deterministic, yet the guide gives only prose checkpoints ('Compare row averages...') and no expected values.
    - 03-15: The demo guide ends with an optional section that uses material the lecture never teaches: awk, which appears only in BONUS, and in demo1a/demo1b, os.path, os.makedirs, generator expressions, and raise.
    - Demo parts of 01-03 (IndentationError step in 04d), 01-12 (`rm` practice), 02-01 (GUI branch steps), 02-04 (a demo that produces a conflict), 02-07 (helpers return None for empty input), 02-09 (debugger practice), 02-10 (Ctrl+E step).
    - Colab pandas-version check in the Lecture 04 (Demos 2–3) and Lecture 11 notebooks: add the setup cell that `work/colab_standard.md` requires (with `FIXME.md`).

## Notion-mapped files

Each mapped Markdown file stores its Notion URL and page ID in YAML metadata. Preserve that metadata when editing. The local source is not permission to overwrite newer Notion content; reconcile fetched Notion edits before publishing.
