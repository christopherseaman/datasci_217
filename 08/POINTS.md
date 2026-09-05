# Lecture 08 Presentation Cues

`README.md` is the authoritative lecture reference. Use this page as a short
instructor cue sheet; examples and API details live in the linked sections.

## Split–apply–combine and result shape

- Frame aggregation as split → apply → combine, then ask which shape the question needs.
- One row per group: `agg`; one value per source row: `transform`; whole groups: `filter`; custom output: `apply`.
- `count()` excludes missing values; `size()` includes them. In pandas 3, categorical groupers default to `observed=True`.
- Point to [GroupBy result-shape choices](README.md#groupby-result-shape-choices) and the `include_groups=False` `apply` contract.

## LIVE DEMO!

## Pivot tables and cross-tabulations

- Pivot tables combine grouping and reshaping; make `values`, `index`, `columns`, and `aggfunc` explicit.
- Use `sum`, `mean`, or `count` according to the question; `fill_value` handles absent combinations, and `observed=False` is opt-in for all defined categorical combinations.
- Point to [Pivot Tables and Cross-Tabulations](README.md#pivot-tables-and-cross-tabulations).

## LIVE DEMO!

## Optional bonus: remote and performance cues

- Remote Jupyter: bind the server to loopback, `jupyter notebook --ip=127.0.0.1 --port=8888 --no-browser`, then forward with `ssh -L 8888:localhost:8888 user@server`.
- Use screen/tmux for disconnect-safe jobs; follow local security policy and never expose an unauthenticated notebook publicly.
- Optimize only after measuring. Select numeric columns explicitly before reductions, e.g. `df.groupby('category', observed=True)[['value']].sum()`; do not imply that a generic groupby sum is numeric-only across versions.
- Chunked workflows must handle no non-empty chunks before calling `pd.concat`; raise a clear input error or return an explicitly documented empty result.
- Prefer built-in aggregations, reviewed dtype choices, and package-supported out-of-core tools; parallelism can cost more than it saves.
- Point to [Basic SSH and Remote Workflow](BONUS.md#basic-ssh-and-remote-workflow) and [Performance Optimization](BONUS.md#performance-optimization).

## OPTIONAL BONUS DEMO!

## Closing cues

- Re-state: choose result shape first, then choose the aggregation/reshape that expresses it.
- Invite a quick “group, summarize, reshape” question using the audience’s own data.
