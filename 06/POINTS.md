# Lecture 06 presentation cues

The [README](README.md) is the authoritative explanation and API reference.
Use this page as a compact speaking and demonstration guide; it is not a
second, executable version of the lecture.

## Opening and learning targets

- Frame wrangling as relating, stacking, and reshaping existing observations.
- Preview `merge`, `concat`, and `pivot`/`melt`, then connect each to the
  question it answers.
- Keep the join-key joke brief: a good key makes matching boring; a bad key is
  an enthusiastic photocopier.

## Merge: relate rows through keys

- Start with customers (one row per customer) and purchases (many rows per
  customer); name the row meaning before showing syntax.
- Demonstrate `how='inner'`, `'left'`, `'right'`, and `'outer'` as choices
  about which unmatched rows to retain.
- Emphasize explicit `on=...` and `left_on`/`right_on`; inspect shapes and
  unmatched keys after a merge.
- State cardinality before merging: `one_to_one`, `one_to_many`,
  `many_to_one`, or intentionally `many_to_many`.
- Use `validate=` to enforce the expected relationship and `indicator=True` to
  audit `left_only`, `right_only`, and `both` keys.
- Call out the pandas/SQL null-key difference: pandas matches null keys to
  null keys, whereas SQL generally does not. Filter or sentinelize first when
  nulls must not match.
- If both inputs have non-key columns with the same name, use descriptive
  `suffixes=` or rename before merging.

## LIVE DEMO!

## Composite and index-based keys

- Use `on=['store_id', 'date']` when one column is not sufficiently specific.
- Remind students that index-based combination matches labels, not row
  positions; verify that labels represent the same entities.
- Point to [BONUS.md](BONUS.md) for `left_index`/`right_index`, `join()`, and
  other specialized index operations.

## Concat: stack or align, do not match keys

- Contrast `merge` (relational key matching) with `concat` (stack one axis and
  align labels on the other).
- Vertical concat adds rows from similarly structured batches; mismatched
  columns become missing values. Use `ignore_index=True` when old row labels
  are not meaningful.
- Horizontal concat adds columns by index labels. The default keeps the union
  of labels; `join='inner'` keeps only shared labels.
- Explain that `join=` in `concat` controls column-set alignment, not a
  relational join. Use `verify_integrity=True` when duplicate labels would be
  an error, and `keys=` when source provenance matters.

## Reshape: change table shape for the next task

- Wide format keeps several measurements in columns; long format stores a
  variable name and value in rows. Neither is universally best.
- Use `melt(id_vars=...)` when column names should become values; use `pivot`
  when each index/column pair is expected to be unique.
- Use `pivot_table` only when repeated pairs are valid and an intentional
  aggregation (`sum`, `mean`, `count`, etc.) answers the question; see the
  canonical aggregation treatment in [Lecture 08](../08/README.md).
- Avoid rules such as “wide is for humans” or “long is always for analysis”:
  choose the shape required by the next grouping, plot, report, or model.

## LIVE DEMO!

## Index and close

- `set_index` changes row labels; `reset_index` returns labels to columns.
  Neither guarantees uniqueness or a database primary key.
- A `MultiIndex` can represent hierarchical labels and often comes from
  grouped results; flatten with `reset_index()` when a regular table is easier
  to use.
- Demo cues: merge with an unmatched key, validate cardinality, concat two
  batches with different columns, then reshape a survey table. Keep each demo
  self-contained; the lecture pages are explanatory material, not one program.

## LIVE DEMO!
