# Lecture 06 Demo Notebooks Review

**Date:** 2025-10-21
**Reviewer:** Code Analyzer Agent
**Scope:** Review of 3 demo notebooks in 06/demo/

---

## Executive Summary

All three demo notebooks are **well-aligned** with Lecture 06 content and demonstrate appropriate progression in complexity. Content match scores range from 8.5-9.5/10. Minor dependency violations were found (primarily pedagogical warnings about visualization), and clarity could be enhanced with additional beginner scaffolding in some areas.

---

## Demo 1: Merge Operations (`01_merge_operations.ipynb`)

### Content Match Score: 9.5/10

**Alignment with Lecture 06:**
- ✅ **Excellent coverage** of `pd.merge()` fundamentals
- ✅ All four join types demonstrated (inner, left, right, outer)
- ✅ Proper handling of duplicate keys (many-to-one relationships)
- ✅ Merging on multiple columns (composite keys)
- ✅ Overlapping column names with custom suffixes
- ✅ Validation with `indicator=True`

**Dependency Audit:**
- ✅ **No violations** - Uses only concepts from lectures 01-06
- ✅ Appropriate use of: DataFrames, indexing, `pd.to_datetime()`, `fillna()`, `groupby()`, `agg()`
- ✅ All operations covered in prior lectures

**Instruction Clarity:**

**Strengths:**
- Excellent real-world scenario (customer purchase analysis)
- Clear explanations of when each join type is appropriate
- Good use of "Key Observations" and "Interpretation" sections
- Practical business insights (marketing opportunities, data quality issues)
- Strong "Pro tip" callouts throughout

**Areas for Improvement:**
1. **Cell 6 observation** states "C006 has a purchase but no customer record" - could add explicit instruction to verify this in the data
2. **Cell 9 "Common pitfall"** - Excellent warning, but could add a concrete example of HOW to check row counts
3. **Cell 28** mentions composite keys preventing "wrong matches" - could show an example of what a wrong match would look like

**Recommended Fixes:**

```python
# Add to Cell 9 after "Common pitfall" warning:
print(f"Expected rows: {len(customers)} customers")
print(f"Actual rows: {len(inner_merge)}")
print(f"Missing customers: {len(customers) - inner_merge['customer_id'].nunique()}")
```

---

## Demo 2: Pivot and Melt (`02_pivot_melt.ipynb`)

### Content Match Score: 9.0/10

**Alignment with Lecture 06:**
- ✅ **Strong coverage** of reshape operations
- ✅ Wide vs long format clearly explained
- ✅ `melt()` for wide→long transformation
- ✅ `pivot()` for long→wide transformation
- ✅ `pivot_table()` for aggregation with duplicates
- ✅ Cleaning question labels for readability

**Dependency Audit:**
- ⚠️ **Minor pedagogical warning** (not a violation)
- Cell 5 and 8 mention "plotting libraries prefer long format" - while factually correct, this references visualization (Lecture 07+)
- However, this is **acceptable** as it's contextual explanation, not actual visualization code

**Instruction Clarity:**

**Strengths:**
- Realistic survey data scenario
- Clear format comparison (wide vs long characteristics)
- Excellent "What happened" explanations after transformations
- Good progression: transform → analyze → transform back
- Strong visual guide in Cell 36 showing format differences

**Areas for Improvement:**
1. **Cell 7** - The `melt()` parameters could benefit from a "parameter guide" showing what each argument controls
2. **Cell 22** - The error message example is excellent, but could show the actual error output
3. **Cell 26** - "Complete workflow pattern" is great but assumes students remember all steps - could add a visual flowchart

**Recommended Fixes:**

```python
# Add to Cell 7 as a separate markdown cell:
"""
### Understanding melt() Parameters:
- `id_vars`: Columns to KEEP as-is (identifying information)
- `value_vars`: Columns to STACK into rows (measurements to melt)
- `var_name`: What to NAME the new column of variable names
- `value_name`: What to NAME the new column of values

Think of it as: "Keep these (id_vars), stack those (value_vars)"
"""
```

---

## Demo 3: Concat and Time Series (`03_concat_timeseries.ipynb`)

### Content Match Score: 8.5/10

**Alignment with Lecture 06:**
- ✅ **Comprehensive coverage** of concatenation
- ✅ Vertical concatenation (`axis=0`)
- ✅ Horizontal concatenation (`axis=1`)
- ✅ `ignore_index=True` for clean sequential indexing
- ✅ `set_index()` and `reset_index()` operations
- ✅ Misaligned index handling
- ✅ Using `keys` parameter to track data sources
- ✅ Combining `concat()` and `merge()` in workflows

**Dependency Audit:**
- ⚠️ **Minor issue** - Cell 15 uses `.resample('Q')` which is a time series method
  - **Lecture 06 README** doesn't explicitly cover `resample()`
  - However, the README mentions "time series" in Demo 3 description
  - **Assessment:** Acceptable as pedagogical demonstration of datetime index power

**Instruction Clarity:**

**Strengths:**
- Excellent quarterly sales scenario (very realistic)
- Clear problem identification (repeated indexes)
- Two solutions presented (ignore_index vs set_index)
- Strong explanations of when to use each approach
- Year-over-year comparison is advanced but well-scaffolded

**Areas for Improvement:**
1. **Cell 15** - `.resample()` appears without explanation - add a brief note
2. **Cell 21-25** - Misaligned indexes section could benefit from a visual diagram
3. **Cell 32-38** - The combined workflow is excellent but quite long - consider breaking into sub-steps
4. **Cell 46-49** - Year-over-year analysis is complex for beginners - might need more intermediate steps

**Recommended Fixes:**

```python
# Add to Cell 15 before resample example:
"""
**Note:** `.resample()` is a powerful time series method that groups by time periods.
While we'll cover time series in detail later, this demonstrates why datetime
indexes are useful - they enable time-based operations like quarterly summaries.
"""

# Add visual diagram to Cell 21:
"""
### Visual: Index Alignment

DataFrame 1:           DataFrame 2:           Result (axis=1):
month  | revenue       month  | ad_spend      month  | revenue | ad_spend
2023-01|  125000       2023-01|   12000       2023-01| 125000  | 12000
2023-02|  132000       2023-02|   15000       2023-02| 132000  | 15000
2023-03|  145000       2023-04|   20000       2023-03| 145000  |   NaN
                                               2023-04|    NaN  | 20000

Only matching indexes are aligned → NaN where indexes don't match!
"""
```

---

## Dependency Violations Summary

### None Found (All Acceptable)

All demos use ONLY concepts from lectures 01-06:
- **Lecture 01-02:** DataFrame basics, indexing, selection
- **Lecture 03:** Data cleaning fundamentals
- **Lecture 04:** Missing data handling (`fillna()`, `isna()`)
- **Lecture 05:** Data types, `pd.to_datetime()`
- **Lecture 06:** Merging, concatenation, reshaping

**Pedagogical References (Not Violations):**
- Demo 2 mentions "plotting libraries prefer long format" (contextual, no code)
- Demo 3 uses `.resample()` to demonstrate datetime index benefits (pedagogical)

Both are acceptable as they enhance understanding without requiring knowledge from future lectures.

---

## Cross-Cutting Clarity Issues

### Common Patterns Needing Improvement:

1. **Parameter Reference Guides**
   - Demos would benefit from "parameter cheat sheets" showing what each argument does
   - Example: `melt(id_vars=?, value_vars=?, var_name=?, value_name=?)`

2. **Visual Diagrams**
   - Index alignment in horizontal concat needs visual representation
   - Wide vs long format comparison could use side-by-side tables
   - Join types could reference lecture diagrams (if they exist)

3. **Beginner Scaffolding**
   - Some complex operations (year-over-year in Demo 3) need more intermediate steps
   - Error messages could be shown explicitly (not just described)
   - "Why this matters" sections could appear more consistently

4. **Verification Steps**
   - Encourage students to check row counts before/after operations
   - Add "sanity checks" showing what to verify
   - Include "common mistakes" examples

---

## Recommended Enhancements

### For All Demos:

1. **Add "Before You Start" Cell:**
```python
"""
### Before You Start - Verification Checklist
After each operation, verify:
- Row count matches expectations
- No unexpected NaN values appeared
- Index is what you expect (numeric vs meaningful)
- Column names are clear
"""
```

2. **Add "Common Mistakes" Sections:**
```python
# Common Mistake: Forgetting to specify join type
# This uses inner join (default) and silently drops non-matching rows
wrong = pd.merge(customers, purchases, on='customer_id')

# Better: Be explicit about join type
right = pd.merge(customers, purchases, on='customer_id', how='left')
```

3. **Add Practice Exercises:**
- Each demo could end with 2-3 practice problems
- "Try it yourself" sections with hints
- "What would happen if..." exploratory questions

---

## Specific Fixes by Demo

### Demo 1 Fixes:

**Add to Cell 9:**
```python
# Pro tip: ALWAYS check row counts!
print(f"Customers table: {len(customers)} rows")
print(f"Purchases table: {len(purchases)} rows")
print(f"Inner merge: {len(inner_merge)} rows")
print(f"Unique customers in result: {inner_merge['customer_id'].nunique()}")
# Expected: 3 unique customers with purchases
```

**Add to Cell 28:**
```python
# What happens if we merge only on store_id? (WRONG!)
wrong_merge = pd.merge(sales_q1, targets, on='store_id')
print(f"Wrong merge rows: {len(wrong_merge)}")  # Too many rows!
print("This incorrectly matches Q1 sales with Q2 targets!")

# Correct: Merge on BOTH store_id AND quarter
correct_merge = pd.merge(sales_q1, targets, on=['store_id', 'quarter'])
print(f"Correct merge rows: {len(correct_merge)}")
```

### Demo 2 Fixes:

**Add to Cell 7 (after melt example):**
```python
# Understanding what happened:
print(f"Wide format shape: {survey_wide.shape}")  # (6 rows, 7 columns)
print(f"Long format shape: {survey_long.shape}")   # (30 rows, 4 columns)
print(f"Calculation: 6 employees × 5 questions = {6 * 5} rows ✓")

# Each employee now appears 5 times (once per question)
print(survey_long[survey_long['employee_id'] == 'E001'])
```

**Add to Cell 22 (show actual error):**
```python
# This FAILS with duplicate index/column combinations:
try:
    survey_duplicates.pivot(index='department', columns='question', values='rating')
except ValueError as e:
    print(f"Error: {e}")
    # Output: "Index contains duplicate entries, cannot reshape"

# Solution: Use pivot_table() to aggregate duplicates
result = survey_duplicates.pivot_table(...)
```

### Demo 3 Fixes:

**Add to Cell 15:**
```python
# Quarterly summary using resample (time series feature)
# Note: We'll cover time series in detail later - this just shows
# why datetime indexes are powerful!
quarterly_totals = year_sales_indexed.resample('Q').sum()
print("What resample('Q') does: Groups by Quarter and sums all numeric columns")
quarterly_totals
```

**Add to Cell 21 (visual diagram):**
```markdown
### Understanding Index Alignment in Horizontal Concat

When you concat horizontally (axis=1), pandas aligns by INDEX:

```
sales_h1:              marketing:             Result (axis=1):
month    | revenue     month    | ad_spend    month    | revenue | ad_spend
2023-01  | 125000      2023-01  | 12000       2023-01  | 125000  | 12000
2023-02  | 132000      2023-02  | 15000       2023-02  | 132000  | 15000
2023-03  | 145000      2023-07  | 25000       2023-03  | 145000  | NaN
                                              2023-07  | NaN     | 25000
```

Matching indexes join, non-matching create NaN!
```

**Add to Cell 33-38 (break into sub-steps):**
```python
# WORKFLOW STEP-BY-STEP (Don't run all at once - run each step separately!)

# Step 1: Concatenate quarterly files
print("=" * 50)
print("STEP 1: Concatenate quarterly sales files")
print("=" * 50)
all_sales = pd.concat([q1_sales, q2_sales, q3_sales], ignore_index=True)
print(f"Combined rows: {len(all_sales)}")
display(all_sales.head())

# Step 2: Merge with product data
print("\n" + "=" * 50)
print("STEP 2: Merge sales with product categories")
print("=" * 50)
sales_enriched = pd.merge(all_sales, products, on='month', how='left')
print(f"After merge: {len(sales_enriched)} rows")
display(sales_enriched.head())

# ... continue breaking down each step with clear headers
```

---

## Overall Recommendations

### Priority 1 (High Impact):
1. Add row count verification examples throughout
2. Show actual error messages (not just describe them)
3. Add "Common Mistakes" sections to each demo
4. Include visual diagrams for index alignment and format transformations

### Priority 2 (Medium Impact):
5. Add parameter reference guides for complex functions
6. Break long workflows into clearly labeled sub-steps
7. Add "Try it yourself" practice problems at the end
8. Include more "Why this matters" context

### Priority 3 (Nice to Have):
9. Add "Before You Start" checklists
10. Include troubleshooting tips for common errors
11. Add cross-references to lecture README sections
12. Consider adding a "Cheat Sheet" cell at the end of each demo

---

## Conclusion

The demo notebooks are **high quality** and well-aligned with Lecture 06 objectives. They provide realistic scenarios, clear explanations, and appropriate complexity progression. With the recommended enhancements, they would be **exemplary** teaching materials.

**Summary Scores:**
- Demo 1 (Merge): 9.5/10 - Excellent, minor clarity additions
- Demo 2 (Pivot/Melt): 9.0/10 - Strong, needs parameter guides
- Demo 3 (Concat): 8.5/10 - Comprehensive, needs visual aids

**Next Steps:**
1. Implement Priority 1 fixes (high impact, low effort)
2. Add visual diagrams referenced in lecture README FIXMEs
3. Create practice exercises for each demo
4. Consider adding a "Demo Summary" notebook tying all three together
