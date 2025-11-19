# Final Exam: Chicago Beach Weather Sensors Analysis

**Total: 100 points (60% code, 40% writeup)**  
**Format: Take-home exam**  
**Time: Recommended 6-8 hours**

## Instructions

This final exam tests your ability to complete a full data science project workflow. You'll follow the same 9-phase workflow demonstrated in Lecture 11, but with the Chicago Beach Weather Sensors dataset.

**This assignment uses a looser scaffold than previous assignments:**
- Write your own code from scratch (no TODO placeholders)
- Follow the 9-phase workflow structure
- Produce required milestone artifacts (see README.md)
- Make your own decisions about approaches
- Reference Lecture 11 notebooks for examples, but implement your own solution

**Assignment Structure:**

This assignment is organized into **9 files**, one per question:

1. **`q1_setup_exploration.md`** - Setup & Exploration (Phase 1-2)
2. **`q2_data_cleaning.md`** - Data Cleaning (Phase 3)
3. **`q3_data_wrangling.md`** - Data Wrangling (Phase 4)
4. **`q4_feature_engineering.md`** - Feature Engineering (Phase 5)
5. **`q5_pattern_analysis.md`** - Pattern Analysis (Phase 6)
6. **`q6_modeling_preparation.md`** - Modeling Preparation (Phase 7)
7. **`q7_modeling.md`** - Modeling (Phase 8)
8. **`q8_results.md`** - Results (Phase 9)
9. **`q9_writeup.md`** - Writeup instructions

**How to Use:**
- Work through the files in order (q1 → q2 → ... → q9)
- Each file can be converted to a notebook using jupytext: `jupytext --to notebook q1_setup_exploration.md`
- Or work directly in the markdown files and convert at the end
- Each file builds on outputs from previous files

---

## Quick Start

1. **Setup environment:**
   ```bash
   cd 11/assignment
   uv venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

2. **Download dataset:**
   ```bash
   chmod +x download_data.sh
   ./download_data.sh
   ```

3. **Start with Q1:**
   ```bash
   jupytext --to notebook q1_setup_exploration.md
   jupyter notebook q1_setup_exploration.ipynb
   ```

4. **Continue through Q2, Q3, Q4, Q5, Q6, Q7, Q8, Q9 in order**

5. **Test your work:**
   ```bash
   pytest -q .github/test/test_assignment.py -v
   ```

---

## Questions Overview

**Phase 1-2: Setup & Exploration (Q1)** - 6 points  
**Phase 3: Data Cleaning (Q2)** - 9 points  
**Phase 4: Data Wrangling (Q3)** - 9 points  
**Phase 5: Feature Engineering (Q4)** - 9 points  
**Phase 6: Pattern Analysis (Q5)** - 6 points  
**Phase 7: Modeling Preparation (Q6)** - 3 points  
**Phase 8: Modeling (Q7)** - 9 points  
**Phase 9: Results (Q8)** - 3 points  
**Code Quality & Execution** - 6 points  
**Writeup (Q9)** - 40 points  

**Total: 100 points**

---

## Required Milestone Artifacts

See `README.md` for detailed artifact specifications. Each phase must produce specific output files for auto-grading.

---

## Submission

Submit your completed notebooks/markdown files and `report.md` through GitHub Classroom.

Good luck!
