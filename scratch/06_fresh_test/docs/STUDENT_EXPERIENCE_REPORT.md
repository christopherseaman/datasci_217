# Assignment 6: Student Experience Report

**Student Perspective:** Working through this assignment with only lectures 01-06 knowledge

**Completion Date:** October 21, 2025
**Time to Complete:** ~45 minutes
**Overall Difficulty:** Moderate to Challenging

---

## Setup Experience

### What Went Well
- **Clear Setup Instructions**: The data generator was straightforward and worked perfectly on first run
- **File Verification**: The setup cell that checks for required files is excellent - prevents students from getting stuck
- **Data Preview**: Being able to see sample data immediately helped understand the structure

### Issues Encountered
- **No Issues**: Setup was smooth and well-designed

### Questions/Confusion
- ✅ **Good**: The `display()` function is mentioned in TODOs but I had to use `print()` instead in my script - might confuse students working outside notebooks
- ✅ **Good**: Dataset sizes are reasonable (100 customers, 50 products, 2000 purchases)

---

## Question 1 Experience: Merging Datasets

### Part A: Basic Merge Operations

#### What Was Clear
- **TODO Comments**: Very specific about what to do ("Load data/customers.csv")
- **Progressive Building**: Merge purchases→customers, then add products makes logical sense
- **Example Provided**: The hint for calculating `total_price` with the exact code was helpful

#### Confusion Points
1. **Column Names Not Documented**: I had to use `.head()` to figure out what columns exist
   - Would be helpful to show expected columns: `customer_id`, `name`, `city`, `signup_date`
   - Products have: `product_id`, `product_name`, `category`, `price`
   - Purchases have: `purchase_id`, `customer_id`, `product_id`, `quantity`, `purchase_date`, `store`

2. **Merge Key Inference**: While `on='customer_id'` is obvious, the instructions don't explicitly state which column to merge on
   - **Recommendation**: Add "merge on the `customer_id` column" to be crystal clear

3. **Two-Step Merge Logic**: Why merge purchases→customers first, then add products?
   - This makes sense (purchases is the fact table) but isn't explained
   - Students might wonder why not merge all three at once

#### Methods Used vs. Lectures 01-06
- ✅ `pd.read_csv()` - **Covered in Lecture 02**
- ✅ `pd.merge()` - **Covered in Lecture 06**
- ✅ `.head()` - **Covered in Lecture 01**
- ✅ Column operations (`df['col']`) - **Covered in Lecture 01**
- ✅ `.round()` - **Covered in Lecture 03**
- ⚠️ `how='left'` parameter - **Covered in Lecture 06 but needs practice**

### Part B: Join Type Analysis

#### What Was Clear
- **Good Comparison**: Showing inner vs left join side-by-side helps understand the difference
- **Practical Example**: Finding customers with no purchases is a real-world use case

#### Confusion Points
1. **`.isna()` Method**: TODO says "check for NaN values in purchase columns"
   - The hint is vague - **which** purchase column should we check?
   - I used `purchase_id` but `quantity`, `purchase_date`, or `store` would also work
   - **Recommendation**: Specify "check if `purchase_id` is NaN"

2. **Why Inner vs Left?**: Instructions don't explain WHEN to use each join type
   - Students might mechanically follow instructions without understanding
   - **Recommendation**: Add 1-2 sentences explaining use cases

3. **Result Interpretation**:
   - Inner join: 2000 rows (all purchases have matching customers)
   - Left join: 2001 rows (includes 1 customer with no purchases)
   - This is great data for learning, but the insight isn't highlighted

#### Methods Used vs. Lectures 01-06
- ✅ `.isna()` - **Covered in Lecture 04 (Handling Missing Data)**
- ✅ Boolean indexing - **Covered in Lecture 03**
- ⚠️ Understanding join types - **Covered in Lecture 06 but conceptually challenging**

### Part C: Multi-Column Merge

#### What Was Clear
- **Explicit Hint**: `on=['product_id', 'store']` is provided, making this easier
- **Pre-made DataFrame**: `store_pricing` is created for students, so they can focus on the merge

#### Confusion Points
1. **Why Multi-Column Merge?**: The concept isn't explained well
   - Students might not understand WHY we need to merge on both columns
   - The comment "(Different stores may have different prices for same product)" helps but is subtle
   - **Recommendation**: Emphasize this with an example showing P001 has different discounts at Store A vs Store B

2. **NaN Values Expected**: After merge, most rows have NaN for `discount_pct`
   - This is correct (only P001-P003 have discounts, only at Store A/B)
   - But students might think they did something wrong
   - **Recommendation**: Add a note that NaN values are expected here

#### Methods Used vs. Lectures 01-06
- ✅ `pd.DataFrame()` creation - **Covered in Lecture 01**
- ✅ Multi-column merge syntax - **Covered in Lecture 06**
- ⚠️ Handling NaN from imperfect joins - **Implied in Lecture 04 but not explicitly with merges**

### Part D: Save Results

#### What Was Clear
- **File Saving**: `.to_csv(index=False)` hint is perfect
- **String Formatting**: f-string template for validation report is well-structured

#### Confusion Points
1. **String Writing to File**: The hint says "Use open() with 'w' mode"
   - This is basic Python file I/O, might be new to some students
   - **Recommendation**: Provide example code snippet

2. **Directory Creation**: `os.makedirs('output', exist_ok=True)` is provided
   - Good! But the `exist_ok=True` parameter isn't explained
   - Students might wonder what happens without it

#### Methods Used vs. Lectures 01-06
- ✅ `.to_csv()` - **Covered in Lecture 02**
- ✅ f-strings - **Python basics, should be known**
- ⚠️ File I/O with `open()` - **General Python, may need review**
- ⚠️ `os.makedirs()` - **Not covered, but hint provided**

---

## Question 2 Experience: Concatenating DataFrames

### Part A: Vertical Concatenation

#### What Was Clear
- **Data Splitting Logic**: Filtering by date ranges is clearly shown
- **Quarterly Breakdown**: The date cutoffs (Apr 1, Jul 1, Oct 1) make sense
- **Progressive Teaching**: Shows concat without keys first, then with keys

#### Confusion Points
1. **Date Comparison Strings**: Comparing `purchase_date` column with string '2023-04-01' works
   - But students might not know pandas does automatic type conversion
   - This could fail if dates aren't in the right format
   - **Recommendation**: Add note about date format compatibility

2. **`ignore_index=True` Purpose**: Hint mentions "clean sequential indexing"
   - But doesn't show what happens WITHOUT `ignore_index=True`
   - Students won't see the duplicate indices [0, 1, 2, 0, 1, 2, ...]
   - **Recommendation**: Add a comparison showing both

3. **`keys` Parameter**: The concept of hierarchical indexing is introduced but not explained
   - The output shows `Q1 0`, `Q1 1`, `Q2 0` etc. as multi-level index
   - This is **advanced** for lectures 01-06
   - **Recommendation**: Either skip this or add explanation of what hierarchical index means

#### Methods Used vs. Lectures 01-06
- ✅ Boolean filtering - **Covered in Lecture 03**
- ✅ `pd.concat()` - **Covered in Lecture 06**
- ⚠️ `ignore_index` parameter - **Mentioned in Lecture 06 but needs practice**
- ❌ `keys` parameter and hierarchical indexing - **NOT covered in lectures 01-06**
- ❌ Multi-level index interpretation - **Advanced topic**

### Part B: Horizontal Concatenation

#### What Was Clear
- **Realistic Scenario**: Satisfaction scores and loyalty tiers as separate datasets makes sense
- **Sample with Different Seeds**: Shows overlapping but not identical customer sets
- **Index-Based Concat**: Setting `customer_id` as index before concat is explained

#### Confusion Points
1. **`.set_index()` Method**: Creates indexed DataFrames
   - This changes the DataFrame structure significantly
   - Students might not understand why we need to do this
   - **Recommendation**: Add explanation that horizontal concat aligns on index

2. **`axis=1` Parameter**: Hint says `axis=1` for horizontal
   - But doesn't explain what `axis=0` (vertical) vs `axis=1` (horizontal) means
   - This is confusing terminology for beginners
   - **Recommendation**: Add visual diagram or clearer explanation

3. **`join='outer'` Concept**: Keeps all customers from both datasets
   - This creates NaN values (30 missing satisfaction, 20 missing loyalty)
   - The concept of outer join in horizontal concat is different from merge
   - Students might confuse this with merge's `how='outer'`
   - **Recommendation**: Clarify difference between merge outer join vs concat outer join

4. **Misaligned Indexes**: The assignment asks "how many NaN values?"
   - This is good for checking understanding
   - But doesn't explain WHY the indexes are misaligned (different sample seeds)
   - **Recommendation**: Emphasize this is intentional for teaching purposes

#### Methods Used vs. Lectures 01-06
- ✅ `.sample()` - **Covered in Lecture 05 (Sampling)**
- ✅ `pd.date_range()` - **Covered in Lecture 02/03**
- ✅ `np.random.choice()` - **NumPy basics**
- ⚠️ `.set_index()` - **Mentioned in Lecture 01 but not heavily used**
- ❌ `axis` parameter understanding - **Conceptually challenging, needs more coverage**
- ❌ Horizontal concat with misaligned indexes - **Advanced concept**
- ⚠️ `.to_csv()` with index - **Saving index is different from `index=False` in Q1**

---

## Question 3 Experience: Reshaping and Analysis

### Part A: Pivot Table Analysis

#### What Was Clear
- **Loading Previous Results**: Using Q1's saved CSV creates good workflow continuity
- **Adding Month Column**: `pd.to_datetime().dt.to_period('M')` is provided
- **Clear Goal**: "sales by category and month" is specific

#### Confusion Points
1. **`pd.to_period('M')` Method**: Creates Period objects for months
   - This is **not covered** in lectures 01-06
   - The syntax chain is complex: `pd.to_datetime().dt.to_period('M')`
   - Students might not understand why we need to convert to datetime first
   - **Recommendation**: Add explanation or provide this as pre-written code

2. **`pd.pivot_table()` vs `.pivot()`**: The hint uses `pivot_table` with `aggfunc='sum'`
   - Students learned `.pivot()` in Lecture 06
   - Might not know when to use `pivot_table` instead
   - The reason (handling duplicates) isn't explained in the TODO
   - **Recommendation**: Add note "Use pivot_table instead of pivot because we have multiple purchases per month/category"

3. **Pivot Parameters**: `values`, `index`, `columns`, `aggfunc`
   - This is a lot of parameters to remember
   - The hint provides them, but students might just copy-paste without understanding
   - **Recommendation**: Break down what each parameter does

4. **Period Index in Output**: The saved CSV has Period index which looks like "2023-01"
   - This might cause issues if students try to read it back in
   - **Not explained** in assignment

#### Methods Used vs. Lectures 01-06
- ✅ `pd.read_csv()` - **Covered in Lecture 02**
- ✅ `pd.to_datetime()` - **Covered in Lecture 03**
- ❌ `.dt.to_period('M')` - **NOT covered in lectures 01-06**
- ❌ Period objects - **Advanced datetime topic**
- ✅ `pd.pivot_table()` - **Covered in Lecture 06**
- ⚠️ `aggfunc` parameter - **Mentioned in Lecture 06 but needs practice**

### Part B: Melt and Long Format

#### What Was Clear
- **Reset Index First**: Makes month a regular column before melting
- **Melt Parameters**: Hint provides `id_vars`, `var_name`, `value_name`
- **Practical Analysis**: Groupby after melt shows why long format is useful

#### Confusion Points
1. **Why Melt?**: The assignment doesn't explain when/why to use melt
   - Students might think "we just created the pivot table, why undo it?"
   - The benefit (easier groupby analysis) isn't emphasized
   - **Recommendation**: Add 1-2 sentences on wide vs long format use cases

2. **`.agg()` with Multiple Functions**: `agg(['sum', 'mean'])`
   - This returns a DataFrame with multi-column output
   - Students might have only seen single aggregations before
   - The list syntax `['sum', 'mean']` might be new
   - **Recommendation**: Provide example output

3. **`.sort_values()` Chaining**: `groupby().agg().sort_values()`
   - Chaining three methods together is complex
   - Students might not understand the order matters
   - **Recommendation**: Break into multiple lines with comments

4. **Index Access**: `category_summary.index[0]` and `index[-1]`
   - Assumes students know aggregated result has category as index
   - Might not be obvious after sort_values
   - Works because groupby automatically makes the group column the index

#### Methods Used vs. Lectures 01-06
- ✅ `.reset_index()` - **Covered in Lecture 06**
- ✅ `pd.melt()` - **Covered in Lecture 06**
- ✅ `.groupby()` - **Covered in Lecture 05**
- ⚠️ `.agg()` with multiple functions - **Basic version in Lecture 05, but list syntax might be new**
- ✅ `.sort_values()` - **Covered in Lecture 03**
- ⚠️ Method chaining - **Conceptually challenging for beginners**
- ⚠️ Index access on aggregated result - **Requires understanding groupby index**

---

## Overall Assessment

### Strengths of the Assignment

1. **Progressive Complexity**: Starts simple (basic merges) and builds to complex (pivot/melt)
2. **Real-World Data**: The retail scenario is relatable and realistic
3. **Good Hints**: Most TODOs have helpful hints that guide without giving away the answer
4. **File Generation**: Data generator ensures everyone has identical data
5. **Validation Built-In**: Final checklist cell helps students verify completion
6. **Practical Output**: Creating actual CSV files feels like real data work

### Areas for Improvement

1. **Concept Explanations**: More "why" explanations needed, not just "how"
2. **Method Coverage**: Some methods (`.to_period()`, hierarchical indexing) go beyond lectures 01-06
3. **Error Prevention**: Could warn students about common mistakes
4. **Visual Aids**: Diagrams showing merge types, wide vs long format would help
5. **Expected Output**: Show sample output for complex operations
6. **Difficulty Curve**: Question 3 feels harder than Questions 1-2

### Time Estimate
- **Fast Students**: 30 minutes (if familiar with all concepts)
- **Average Students**: 45-60 minutes (need to experiment and debug)
- **Struggling Students**: 90+ minutes (need to review lectures while working)

### Prerequisites Assessment
**What students MUST know before starting:**
- Basic pandas DataFrame operations (lectures 01-03)
- Filtering and boolean indexing (lecture 03)
- Groupby basics (lecture 05)
- Merge and concat concepts (lecture 06)

**What might need review:**
- File I/O in Python
- Date/datetime handling
- Index manipulation
- Method chaining

---

## Methods Not Covered in Lectures 01-06

### Definitively NOT Covered (need to learn on the fly):
1. **`.dt.to_period('M')`** - Period conversion for dates
2. **`keys` parameter in `pd.concat()`** - Hierarchical indexing
3. **Multi-level index interpretation** - Reading Q1 0, Q1 1 format
4. **`axis` parameter deep understanding** - When to use 0 vs 1
5. **`.set_index()` for concat** - Why indexes matter for horizontal concat

### Probably Covered but Need Practice:
1. **`how` parameter in merge** - Inner vs left vs right vs outer
2. **`aggfunc` in pivot_table** - Understanding aggregation functions
3. **`.agg()` with list** - Multiple aggregations at once
4. **Method chaining** - Combining multiple operations
5. **`ignore_index` in concat** - When and why to use it

### Should Be Fine (assuming lectures were complete):
1. **`pd.read_csv()` / `.to_csv()`** - Basic I/O
2. **`pd.merge()`** - Basic merge syntax
3. **`pd.concat()`** - Basic concatenation
4. **`pd.pivot_table()`** - Basic pivot
5. **`pd.melt()`** - Basic melt
6. **`.groupby()`** - Basic groupby
7. **Boolean filtering** - Filtering with conditions

---

## Recommendations for Improvement

### For Question 1 (Merging):
1. **Add column reference table** at the top showing all dataset columns
2. **Explicitly state merge keys** in each TODO
3. **Show sample output** for each merge step
4. **Add explanation** of why we do 2-step merge (purchases → customers → products)
5. **Clarify** which column to check for NaN in Part B

### For Question 2 (Concatenating):
1. **Add diagram** showing vertical vs horizontal concat
2. **Remove or explain** hierarchical indexing (keys parameter)
3. **Add example** of what happens without `ignore_index=True`
4. **Clarify** axis=0 vs axis=1 with visual
5. **Explain** why we set_index before horizontal concat

### For Question 3 (Reshaping):
1. **Provide `.to_period()` code** instead of making students write it
2. **Add explanation** of when to use pivot_table vs pivot
3. **Show sample output** of pivot table before students run it
4. **Add "why melt?"** explanation with use case
5. **Break down** method chaining into steps with intermediate variables

### General Improvements:
1. **Add "Learning Objectives"** section at the top of each question
2. **Include "Common Mistakes"** warnings
3. **Provide "Check Your Work"** cells with assert statements
4. **Add optional challenge problems** for advanced students
5. **Create troubleshooting guide** for common errors

---

## Student Reflection

### What I Learned:
- How to combine multiple datasets with merge
- The difference between inner and left joins
- How to concatenate DataFrames vertically and horizontally
- How to reshape data between wide and long formats
- Practical workflow: load → merge → analyze → save

### What Was Challenging:
- Understanding when to use merge vs concat
- Figuring out the axis parameter for concat
- Working with Period objects for dates
- Interpreting multi-level indexes from concat with keys
- Remembering all the parameters for pivot_table

### What I'd Want to Know More About:
- Other join types (right, outer)
- When to use pivot vs pivot_table vs melt vs stack/unstack
- How to handle complex date/time operations
- Best practices for merging multiple (3+) datasets
- Performance considerations for large datasets

### Confidence Level After Completing:
- **Merging**: 8/10 - Feel confident with basic merges
- **Concatenating**: 6/10 - Still unsure about axis parameter and when to use vs merge
- **Reshaping**: 5/10 - Understand pivot and melt separately but not when to use which
- **Overall**: 7/10 - Can complete the tasks but would struggle to apply to new scenarios

---

## Conclusion

This is a **well-designed assignment** that covers important data wrangling concepts. With some additional explanations and scaffolding, it would be **excellent** for students who have completed lectures 01-06.

**Main Takeaway**: The assignment works, but students will need to experiment, use `.head()` frequently, and possibly review lecture materials while working. This is good practice but might be frustrating without more guidance.

**Suggested Difficulty Rating**:
- Current: 7/10 (Moderate-Hard)
- With improvements: 5/10 (Moderate)
- Ideal for: Week 6-7 of introductory data science course

**Would Recommend**: Yes, with the improvements noted above.
