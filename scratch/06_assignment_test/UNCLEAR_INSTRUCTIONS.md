# Unclear Instructions - Specific Quotes from README

## Direct Quotes of Problematic Instructions

### 1. Missing File Reference (Line 8-11)
**Quote:**
```markdown
Run the data generator notebook to create your dataset:

jupyter notebook data_generator.ipynb
```

**Problem:** File `data_generator.ipynb` does not exist

**Student Impact:** Cannot proceed with assignment

**Clarity Rating:** 0/10 - completely blocks progress

**Fix:** Include the file in distribution

---

### 2. Ambiguous Multi-Column Merge (Line 32)
**Quote:**
```markdown
- Merge on multiple columns (customer_id + order_date)
```

**Problems:**
1. The example datasets don't have a scenario where this makes sense
2. No explanation of WHY you would merge on both columns
3. The example code in line 169 shows a different scenario
4. It's unclear if this is required or just an example

**Student Questions:**
- "Why would I merge on both customer_id AND order_date?"
- "Where in the data do I need to do this?"
- "Is this a required step or just optional practice?"

**Clarity Rating:** 4/10 - confusing and potentially misleading

**Fix:** Either:
- Remove this bullet point (it's not actually needed)
- OR modify data to have duplicate customer_id+date combinations
- OR clarify it's just an example of the syntax, not required

---

### 3. Workflow Example Mismatch (Lines 169)
**Quote:**
```python
# Multi-column merge
multi_merged = pd.merge(df1, df2, on=['customer_id', 'order_date'])
```

**Problem:** Uses generic variable names `df1` and `df2` instead of the actual dataset names used elsewhere

**Student Confusion:** "What are df1 and df2? Are these new datasets I need to create?"

**Clarity Rating:** 5/10 - inconsistent naming

**Fix:** Use consistent dataset names:
```python
# Multi-column merge (example - not required for this assignment)
# If you had duplicate customer_id values, you could merge on both:
multi_merged = pd.merge(orders, customers, on=['customer_id', 'order_date'])
```

---

### 4. Jupyter Execution Assumption (Line 10)
**Quote:**
```bash
jupyter notebook data_generator.ipynb
```

**Problems:**
1. Assumes jupyter is already installed (no prior mention of installation)
2. Opens browser interface - some students may not want this
3. No alternative execution methods mentioned
4. No explanation of what to do after opening

**Student Questions:**
- "I get 'jupyter: command not found', what do I do?"
- "Do I need to run all cells or just some?"
- "Can I run this without opening the browser?"

**Clarity Rating:** 6/10 - assumes too much prior knowledge

**Fix:** Provide multiple execution options:
```markdown
### Option 1: Interactive (recommended for learning)
```bash
jupyter notebook data_generator.ipynb
# Then: Click "Run All" in the Jupyter interface
```

### Option 2: Command line execution
```bash
jupyter nbconvert --to notebook --execute --inplace data_generator.ipynb
```

### Option 3: If jupyter not installed
```bash
pip install jupyter
# Then use Option 1 or 2 above
```
```

---

### 5. Testing Section Incomplete (Lines 235-254)
**Quote:**
```markdown
## Testing Your Work

Before submission, verify:

1. **All output files exist:**
   ```bash
   ls output/q*.csv
   ```
```

**Problems:**
1. Doesn't mention pytest or test_assignment.py
2. No explanation of how to run automated tests
3. Students don't know how to interpret test results
4. Manual verification only - misses the automated testing

**Clarity Rating:** 5/10 - incomplete testing guidance

**Fix:** Add automated testing section:
```markdown
## Testing Your Work

### Manual Verification

1. **All output files exist:**
   ```bash
   ls output/q*.csv
   ```

### Automated Testing (Recommended)

Run the provided test suite:

```bash
pytest test_assignment.py -v
```

Expected output:
```
test_assignment.py::test_q1_output_exists PASSED
test_assignment.py::test_q2_output_exists PASSED
test_assignment.py::test_q3_output_exists PASSED
```

If tests fail:
- Check the error message for details
- Verify your output filenames match exactly
- Ensure you've completed all required steps
```

---

### 6. File Naming Inconsistency (Lines 37, 51, 279)
**Quote from line 37:**
```markdown
**Output:** `output/q1_merged_data.csv`
```

**Quote from line 51:**
```markdown
**Output:** `output/q2_concatenated_data.csv`
```

**But in notebook, students might naturally write:**
```python
combined_vertical.to_csv('output/q2_combined_data.csv', index=False)
```

**Problem:** The word "concatenated" vs "combined" - students might use the wrong one

**Student Impact:** Automated tests might fail

**Clarity Rating:** 6/10 - easy to get wrong

**Fix:** Be very explicit in the notebook template:
```python
# IMPORTANT: Save with this EXACT filename (tests check for it)
combined_vertical.to_csv('output/q2_concatenated_data.csv', index=False)
```

---

### 7. No Success Criteria (Throughout)
**Problem:** README never tells students what "good" looks like

**Missing Information:**
- How many rows should Q1 merge produce? (Answer: 2000)
- Are NaN values expected? (Answer: Yes, ~95 and ~82 respectively)
- What's a reasonable range for category totals in Q3?

**Student Questions:**
- "I got 1905 rows in inner merge, is that right?"
- "My outer join has NaN values, did I do something wrong?"
- "The numbers seem big, are they correct?"

**Clarity Rating:** 5/10 - no validation benchmarks

**Fix:** Add validation checkpoints:
```markdown
## Expected Results

Use these to verify your work is on track:

**Q1: Merging**
- Inner join: ~1,900 rows (some orders don't match customers)
- Left join: 2,000 rows (all orders preserved)
- Outer join with products: 2,000 rows
- Some NaN values are expected (missing data by design)

**Q2: Concatenation**
- Combined 2023+2024: 100 rows (50 from each year)
- Should have a 'year' column with values 2023 and 2024

**Q3: Reshaping**
- Long format: 600 rows (50 products × 12 months)
- Category totals should be in the hundreds of thousands
```

---

### 8. Common Pitfalls Section Quality (Lines 213-230)
**Quote:**
```markdown
### Merge Issues
- **Exploding rows:** Many-to-many joins create Cartesian products. Always check row counts!
```

**Assessment:** This section is actually GOOD! Clear warnings with explanations.

**Clarity Rating:** 9/10 - excellent preventive guidance

**No fix needed** - this is a model for what the rest should look like

---

## Summary of Unclear Instructions

| Issue | Quote Location | Severity | Clarity Score | Fix Priority |
|-------|---------------|----------|---------------|--------------|
| Missing data_generator.ipynb | Lines 8-11 | CRITICAL | 0/10 | P1 |
| No environment setup | Entire README | CRITICAL | 3/10 | P1 |
| Multi-column merge unclear | Line 32 | MODERATE | 4/10 | P2 |
| Workflow example mismatch | Line 169 | MINOR | 5/10 | P3 |
| Jupyter execution assumptions | Line 10 | MODERATE | 6/10 | P2 |
| Testing section incomplete | Lines 235-254 | MODERATE | 5/10 | P2 |
| File naming inconsistency | Multiple | MODERATE | 6/10 | P2 |
| No success criteria | Throughout | MODERATE | 5/10 | P2 |

**Average Clarity Score of Problematic Sections:** 4.3/10

**Overall Assignment Clarity:** 6.5/10 (accounting for clear sections too)

---

## What Students Actually Need

Based on this walkthrough, a student needs:

1. ✅ Clear environment setup (currently missing)
2. ✅ All referenced files included (currently missing data_generator.ipynb)
3. ✅ Template to work from (currently missing assignment.ipynb template)
4. ✅ Consistent naming throughout (minor issues)
5. ✅ Success criteria to validate work (currently missing)
6. ✅ Complete testing instructions (incomplete)
7. ✅ Clear explanation of WHAT to do and WHY (mostly clear, some gaps)

**Bottom Line:** The assignment teaches the right concepts but needs better scaffolding and complete file distribution.
