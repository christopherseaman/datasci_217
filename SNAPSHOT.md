# DataSci 217 Development Snapshot

**Last Updated:** 2025-10-07

---

## ⚠️ CRITICAL OPEN ITEMS - Assignment 5 (Midterm)

### TIPS.md vs HINTS.md Decision
- **Current State:** `05/assignment/TIPS.md` contains complete solution patterns with full code examples
- **Issue:** This is essentially a solution guide - too much hand-holding for a midterm exam
- **Options:**
  1. Keep TIPS.md as EA/grader reference, remove from student assignment
  2. Create student-facing HINTS.md with lighter scaffolding (no complete implementations)
  3. Provide minimal hints only (function signatures + lecture references)
- **Decision Needed:** Determine what level of scaffolding is appropriate for midterm

### Automated Testing Not Implemented
- **Status:** Assignment 5 grading specification complete (108 pts, ~93 tests)
- **Missing:**
  - No test implementation in `.github/workflows/classroom.yml`
  - No test files in `.github/tests/test_assignment.py`
  - Assignment has not been completed/verified end-to-end
- **Risk:** Cannot verify assignment is solvable or that auto-grading works
- **Next Steps:**
  1. Implement behavioral tests per GRADING_SPEC.md
  2. Complete assignment as student would (verify solvability)
  3. Test auto-grading workflow

---

## Assignment 5 (Midterm) - Current Status

### Completed ✅
- **Scope & Coverage:** Covers Lectures 01-05, excludes NumPy (intentional), no Git/venv testing
- **Point Distribution:** 108 total points
  - Q1: Shell scripting (10 pts)
  - Q2: Python fundamentals (25 pts)
  - Q3: Pandas loading/exploration (10 pts)
  - Q4: Pandas selection/filtering (10 pts)
  - Q5: Missing data handling (15 pts)
  - Q6: Data transformation (23 pts) - includes duplicates, binning, encoding, outliers
  - Q7: Groupby/aggregation (10 pts)
  - Q8: Pipeline automation (5 pts)

- **Documentation:**
  - `README.md` - Student instructions with function signatures
  - `GRADING_SPEC.md` - Complete grading specification with behavioral tests
  - `TIPS.md` - Full solution patterns (⚠️ see decision above)
  - `DATA_SCENARIO.md` - Clinical trial context (needs HIPAA content removed)

- **Data Generation:**
  - `generate_clinical_data.py` - Creates realistic 10k patient dataset
  - Uses hidden variables for realistic correlations (cv_health, treatment_response, site_quality, engagement)
  - Includes realistic data quality issues (missing data, sentinel values, inconsistencies)
  - `data/clinical_trial_raw.csv` - Generated dataset (10,000 rows, 18 columns)
  - `config.txt` - Trial configuration for Q2 parsing exercise

### Topic Coverage Analysis
**Well-Covered:**
- Pandas (60 pts across Q3-Q7) - comprehensive
- Python fundamentals (Q2: 25 pts) - dictionaries, lists, functions, file I/O
- Shell scripting (Q1+Q8: 15 pts) - mkdir, chmod, pipes, exit codes

**Intentional Gaps:**
- NumPy (0 pts) - excluded per design, despite being 70% of Lecture 03
- Git/GitHub - hard to auto-grade
- Virtual environments - hard to auto-grade
- Jupyter interface - not testable

**Added in Final Iteration:**
- Duplicate removal (`.drop_duplicates()`)
- Categorical binning (`pd.cut()`, `pd.qcut()`)
- Dummy encoding (`pd.get_dummies()`)
- Outlier detection (IQR method)

---

## Lecture Status

### Complete ✅
- **Lecture 01:** Python, Command Line, VS Code
- **Lecture 02:** Git, Python Fundamentals (Ch 2-3)
- **Lecture 03:** NumPy & Virtual Environments (Ch 4)
- **Lecture 04:** Pandas on Jupyter (Ch 5-6)
- **Lecture 05:** Data Cleaning (Ch 7)

### Assignments Complete
- **Assignment 1:** ✅ (Lecture 01)
- **Assignment 2:** ✅ (Lecture 02)
- **Assignment 3:** ✅ (Lecture 03)
- **Assignment 4:** ✅ (Lecture 04)
- **Assignment 5:** ⚠️ Designed but not tested (Midterm exam)

---

## Remaining Work - Lectures 6-11

### Lecture 6: Data Wrangling (Ch 8)
- Merge, join, concatenate
- Reshaping (pivot, melt, stack/unstack)
- Advanced groupby
- Assignment 6

### Lecture 7: Plotting and Visualization (Ch 9)
- Matplotlib fundamentals
- Pandas plotting
- Seaborn
- Assignment 7

### Lecture 8: Data Aggregation (Ch 10)
- Advanced GroupBy
- Pivot tables
- Cross-tabulation
- Assignment 8

### Lecture 9: Time Series (Ch 11)
- Date/time operations
- Resampling
- Time series analysis
- Assignment 9

### Lecture 10: Modeling/Advanced Topics (Ch 12+)
- To be determined
- Assignment 10

### Lecture 11: Final Exam/Capstone
- Comprehensive assessment
- Assignment 11 (0-100 scale like Assignment 5)

---

## Key Design Principles (Reference)

### Assignment Design
- Regular assignments (1-4, 6-10): Pass/fail, auto-gradable via pytest
- Exam assignments (5, 11): 0-100+ scale, more granular testing
- Incremental complexity: 3 questions, each building on previous
- Test competence ("can you do it?") not excellence ("did you do it perfectly?")
- All tests <5 pts for grading robustness
- Behavioral testing (what code does) not code inspection (how it's written)

### Content Guidelines
- Core/practical → README.md
- Advanced/theoretical → BONUS.md
- Create demos/assignments ONLY AFTER lecture content is stable
- McKinney is authoritative for Python content
- GUI-based Git/VS Code workflows (not command-line git)
- CLI-first approach (Jupyter delayed until Lecture 4)

---

## Next Session Priorities

1. **Resolve TIPS.md decision** - EA reference vs student scaffolding
2. **Implement Assignment 5 tests** - Create auto-grading workflow
3. **Complete Assignment 5** - Verify it's solvable
4. **Test auto-grading** - Ensure behavioral tests work
5. **Remove HIPAA content** - Clean up DATA_SCENARIO.md
6. **Begin Lecture 6** - Data Wrangling (Ch 8)
