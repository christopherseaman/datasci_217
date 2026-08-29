# Database-Style DataFrame Joins

## Introduction

Let me start by explaining what joining really means in the context of data analysis. Imagine you're running a business and you have customer information in one spreadsheet and purchase history in another. Joining is how you connect these separate pieces of information together using shared identifiers. This is the foundation of working with real-world data because information is rarely all in one place.

- You'll use merging multiple times every single day in real data work - it's the bridge between separate data sources
- Think of joins like puzzle pieces: do you want only pieces that fit together (inner), or keep some with missing partners (outer/left/right)?
- The hardest part isn't syntax - it's choosing the right join type and debugging when you accidentally lose data

## The Basics of pd.merge()

Real-world data lives in separate systems - customer records in one database, purchase history in another. pd.merge() is how you intelligently connect these pieces using shared identifiers, but the devil is in the details of how you specify that matching.

- pd.merge() returns a new DataFrame without modifying originals, which prevents accidental data corruption but requires reassignment to capture results
- Never assume pandas will auto-detect merge columns - explicit specification prevents mysterious failures and makes code readable
- Check DataFrame shapes before and after merging against the expected cardinality: one-to-many joins intentionally add rows, while an unplanned many-to-many join can create a data explosion

## Join Types: The Four Horsemen

The type of join you choose fundamentally changes your results and determines which data you keep or lose. This isn't just a technical detail - it's a business decision about whether you care about customers without purchases, purchases without customer records, or both. Understanding this distinction prevents silent data loss that can invalidate your entire analysis.

- When joining customer database with purchase history, left join keeps all customers even those who haven't purchased, while inner join silently drops non-purchasing customers which can bias your analysis
- Inner join feels safe but is actually dangerous - you silently drop customers without purchases and may not notice
- Left joins are your friend for "master list + details" scenarios - every customer appears at least once
- Pro tip: when results look wrong, try outer join first to see ALL the data and identify what's missing

## Merge Cardinality and Integrity

Not all relationships between datasets are simple one-to-one matches. In real data, one customer might have many purchases, or one product might belong to many categories. Understanding these cardinality relationships (how rows in one table relate to rows in another) is crucial because they determine how many rows your result will have - and getting this wrong leads to mysterious data explosions.

- One-to-many describes customers on the left and purchases on the right: each unique customer record can expand to several purchase rows; reversing the inputs makes the same relationship many-to-one
- Many-to-many explosions happen when both tables have duplicates you didn't know about - check for this!
- Quick sanity check: if result has 10x more rows than you started with, something went wrong
- Real scenario: accidentally merging on 'date' when multiple transactions share the same date
- State the expected relationship before merging, then make it executable with `validate='one_to_one'`, `validate='one_to_many'`, or `validate='many_to_one'`
- Use `indicator=True` when you need to audit which keys appeared only on the left, only on the right, or in both tables
- `validate='many_to_many'` permits repeated keys on both sides, so it documents rather than constrains cardinality

# LIVE DEMO!

## Merging on Multiple Columns

Sometimes a single column isn't specific enough to uniquely identify a match. Think about retail data where you might have store 'S01' in Boston and also store 'S01' in Seattle. You need both store ID and region to make a proper match. This is where composite keys come in - matching on multiple columns simultaneously to ensure you're joining the right records.

- Single-column keys work until they don't - what if store 'S01' exists in multiple regions?
- Composite keys prevent "Halloween 2023 sales" accidentally matching with "Halloween 2024 targets"
- Think hierarchically: year+month+day is more specific than just day

## Handling Overlapping Column Names

When both DataFrames have columns with the same name beyond your merge key, pandas needs to disambiguate them in the result. This happens more often than you'd think - both tables might have 'total', 'date', or 'amount' columns. If you don't handle this proactively, you'll end up with cryptic column names that make your code hard to read.

- Pandas silently adds `_x` and `_y` suffixes to conflicting columns, creating cryptic names like `total_x` and `total_y` that require detective work to decipher
- Both DataFrames having 'date' columns creates `date_x` and `date_y` - which date is which? Was `date_x` the order date or delivery date?
- Proactive suffix specification with descriptive names like `_sales` and `_inventory` prevents confusion and makes results immediately interpretable
- Rename columns before merging when you know conflicts exist - it's faster than debugging mysterious suffixes later

# Alternative Data Combination Methods

## Prerequisite: Index Labels and Alignment

Index-based combination matches row labels, not row positions. Labels may appear in different orders or in only one input; the operation decides whether to retain their union or intersection, and retained unmatched labels produce missing values. Confirm that labels carry the same row identity before relying on this behavior.

## DataFrame.join()

There's a second way to join DataFrames in pandas that's simpler but more specialized. The `.join()` method is designed for the case where you're joining based on the index rather than a column. It's less flexible than merge, but that simplicity can make index-based code clearer.

- Use `.join()` when your right DataFrame is already indexed the way you want - it's like merge but defaults to left join on index
- It's less flexible but that's the point - simpler API for a specific use case like time series data
- Mostly you'll see this in older code; `merge()` is more explicit and recommended for most situations

## Patching Missing Data with combine_first()

Real-world data often comes with gaps, overlaps, and inconsistencies that don't fit neatly into standard merge or concat patterns. These specialized methods handle the messy edge cases you'll encounter when working with incomplete datasets, time series with missing periods, or overlapping data sources that need careful reconciliation.

- `combine_first()` acts like a smart patch tool - it fills NaN values in your primary DataFrame with values from a backup DataFrame while preserving existing data
- This fits scenarios where a reviewed fallback source contains the same variables and observations as an incomplete primary source
- `combine_first()` aligns both axes and may add labels found only in the fallback; the caller's non-null values retain priority
- Real use case: combining customer records from old and new CRM systems where neither has complete data but together they're comprehensive
- Think of it as "if primary source has data, use it; if not, check backup source" - it's a fallback mechanism for patching datasets

# Concatenating DataFrames Along an Axis

## Introduction

While merging relates rows through keys, concatenation stacks objects along an axis and aligns labels on the other axis. Understanding when to concat versus when to merge is one of the key mental models in data wrangling.

- Concat does not perform relational key matching; it stacks along one axis and aligns the other
- Ask yourself: am I combining similar things (concat) or relating different things (merge)?
- Most common mistake: forgetting to reset the index after vertical concat

## Vertical Concatenation: Adding More Rows

The most common concatenation pattern is vertical stacking - taking DataFrames with the same column structure and piling them on top of each other. This is your solution when you have monthly data files, multiple CSV exports, or any situation where you're collecting similar observations over time or across groups and need one unified dataset.

- This is your go-to for "I have January data, February data, March data - make one big DataFrame" - the most common real-world concatenation scenario
- Repeated indexes don't break concatenation immediately but cause confusing behavior in subsequent operations like merges and groupby
- Column mismatches create NaN values - decide if this represents missing data or indicates you should use merge instead
- Concatenate multiple DataFrames at once `concat([df1, df2, df3])` rather than chaining - it's faster and cleaner than repeated concatenation

## Horizontal Concatenation: Adding More Columns

Less common than vertical stacking, horizontal concatenation puts DataFrames side-by-side to add more columns. You're relying on index labels to tell pandas which rows correspond, so those labels must represent the same identity in every input.

- Horizontal concat is rarer because adding columns usually means you should be merging on a key rather than relying on index alignment
- Indexes need not match perfectly: the default retains their union and inserts `NaN` for labels missing from an input; `join='inner'` retains only shared labels
- Real use case: combining results from three separate analyses on the same entities where you want to compare outputs side-by-side

## Column-Set Alignment with the join Parameter

For vertical concatenation, `join=` is the **column-set alignment mode**, not a relational join. It controls whether pandas keeps the union or intersection of columns on the non-concatenation axis.

- In this vertical-concat context, `join` is about column-set alignment, not key matching like `merge()`
- `join='outer'` creates a sparse result by keeping all columns and filling missing values with NaN - useful when you want to preserve all available data
- `join='inner'` keeps only common columns; existing missing values within those columns remain

# Reshaping: Wide vs Long Format

## Introduction

One of the most fundamental yet confusing aspects of data wrangling is understanding that the same information can be organized in completely different table structures. The choice between wide and long format isn't about right or wrong - it's about which operations you need to perform. Different analysis tools, plotting libraries, and statistical methods expect different formats, so you need to be fluent in converting between them.

- This is where students get completely lost - not because it's hard, but because they don't recognize which format they have
- Rule of thumb: if you can read it like a spreadsheet, it's wide; if it has a "variable" column, it's long
- Different interfaces favor different formats; many seaborn mappings are convenient with long data, while some analyses and presentation tables use wide data

## Understanding Wide Format

Wide format is the natural way humans organize data in spreadsheets - one row per entity with multiple columns representing different attributes or measurements. This format is intuitive for reading and data entry, which is why raw data often arrives this way. However, this human-friendly structure is often not ideal for computational analysis.

- Wide feels natural because it's how humans organize tables - one row per "thing"
- Problem: hard to do operations across columns (like "what's the average score across all subjects?")
- Wide is the format data often arrives in, but not the format you'll analyze it in

## Understanding Long Format

Long format seems redundant at first glance - why repeat the student name three times when you could write it once? But a dedicated variable column (like `subject`) can make grouping, filtering, and plotting mappings direct when those operations treat the variable values uniformly.

- Long looks weird and repetitive at first - "why is Alice's name repeated three times?"
- The magic: now `groupby('subject')` works naturally - all math scores are in one column
- Many plotting APIs work naturally with long data, but the required shape depends on the specific interface and chart

## Pivoting Long to Wide with pivot()

Pivoting transforms long format into wide format by spreading out one column's unique values to become separate columns. This is typically your final step before presenting results - you've done your analysis in long format, and now you're creating a human-readable summary table. The key constraint is that your index/column combinations must be unique, which trips up beginners when their data has unexpected duplicates.

- Pivot is for *displaying* results, not analyzing them - it's the last step before showing someone
- The uniqueness constraint catches you off guard - "it worked on the example data but fails on real data"
- Mental model: you're spreading out one column's values to become many column headers

## pivot_table(): Aggregation Preview

`pivot()` cannot choose one cell value when an index/column pair identifies repeated observations. If those repeats are valid, `pivot_table()` can aggregate them before reshaping—but `sum`, `mean`, and `count` answer different questions. [Lecture 08](../08/README.md#pivot-tables-and-cross-tabulations) is the canonical treatment of aggregation and pivot tables.

- Use `pivot_table()` when duplicate index/column combinations genuinely require a defined aggregation; use `pivot()` when uniqueness is expected so unexpected duplicates fail loudly
- Choose `aggfunc` deliberately; this is an aggregation decision, not merely a way around an error

## Melting Wide to Long with melt()

Melt is the inverse of pivot - it transforms selected wide columns into variable-value rows. Use it when the next grouping, filtering, or plotting interface expects observations in that structure; do not reshape merely from habit.

- Melt early when variable names stored across columns need to become values in one variable column
- `id_vars` are the columns that identify the entity; everything else gets unpivoted
- Melt before `groupby` when the groups are encoded in wide column names; otherwise grouping first or not melting at all may be appropriate
- The resulting DataFrame looks inefficient (repeated values) but enables powerful operations

# LIVE DEMO

# Working with DataFrame Indexes

## Introduction

The DataFrame index is often ignored by beginners who just stick with the default 0, 1, 2... numbering, but understanding indexes enables label-based selection and alignment. The index is the row-label axis. Moving columns to and from it changes how you identify and combine rows.

With pandas 3 Copy-on-Write, use a returned-object mental model for these transformations: capture the result or deliberately reassign the variable.

- A default `RangeIndex` is fine when rows do not need meaningful labels
- Index labels are not automatically unique or equivalent to database primary keys
- Care about the index when label-based selection or alignment serves the task

## set_index(): Moving Columns to Index

Setting a column as the index is a declaration about how you'll label and access rows. It's the difference between "give me row 7" and "give me the employee labeled E0123". It can make label-based selection and alignment clearer, but it does not guarantee uniqueness or universal performance gains.

- Setting an index is declaring "this column is how I identify these rows"
- Use the returned-object model: `indexed = df.set_index('employee_id')`
- Check or request uniqueness when the operation requires it

## reset_index(): Moving Index to Columns

The opposite of set_index, reset_index converts your index back into a regular column. You'll do this constantly, especially after groupby operations which create hierarchical indexes automatically. The key decision is whether to keep the old index as a column or discard it entirely, which depends on whether that information is meaningful or just sequential numbering.

- You'll use this constantly after `groupby()` because groupby creates weird hierarchical indexes
- `drop=True` when you don't care about the old index (usually when it was just 0, 1, 2...)
- `drop=False` when the index contains information you want to keep as a column
- Capture or reassign the result: `flat = indexed.reset_index()`

## Basic MultiIndex Operations

MultiIndex, or hierarchical indexing, allows multiple levels of row labels - like having categories and subcategories. This structure emerges automatically from groupby operations with multiple columns, and while it's powerful for certain advanced operations, most people find it confusing and immediately flatten it back with reset_index. That's completely fine - MultiIndex is a tool you grow into, not something you need to master immediately.

- MultiIndex is powerful but confusing - it's like nested dictionaries for row labels
- Automatically appears after `groupby(['col1', 'col2'])` - you didn't ask for it but there it is
- Most people immediately do `reset_index()` to get back to normal - that's fine!
- Advanced usage (stack/unstack/swaplevel) exists but you probably won't need it early on

# LIVE DEMO!
