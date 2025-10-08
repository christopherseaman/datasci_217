# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

This is **DataSci 217** - an "Introduction" to Python & Data Science Tools course. The repository contains an 11-lecture sequence teaching practical data science skills centered around Python, following McKinney's "Python for Data Analysis" (3rd Ed) as the primary reference, supplemented with command line and development workflow content.

**Core Philosophy:**
- CLI-first approach (Jupyter delayed until Lecture 4/Pandas)
- GUI-based Git/VS Code workflows (not command-line git)
- Practical utility over theoretical depth
- Advanced content reserved for "bonus" or "DLC" material
- Focus on daily data science tools and workflows

**Course Structure:**
- Lectures 1-5: Foundational toolkit (1-unit completion option)
- Lectures 6-11: Advanced mastery and professional skills (2-unit completion)

## Directory Structure

```
datasci_217/
├── 01-11/                    # Lecture directories (numbered 01 through 11)
│   ├── XX/
│   │   ├── README.md         # Notion-formatted lecture content
│   │   ├── BONUS.md          # Advanced/theoretical "DLC" content (optional)
│   │   ├── assignment/       # GitHub Classroom compatible assignment
│   │   │   ├── README.md     # Assignment instructions
│   │   │   ├── main.py       # Student implementation file
│   │   │   ├── test_assignment.py  # Local test file (for student use)
│   │   │   ├── requirements.txt
│   │   │   ├── starter_code.py (optional)
│   │   │   └── .github/
│   │   │       ├── workflows/classroom.yml  # Auto-grading workflow
│   │   │       └── tests/
│   │   │           ├── test_assignment.py   # Remote test (updated independently)
│   │   │           └── requirements.txt     # Remote test dependencies
│   │   ├── demo/             # Hands-on demos (2-3 per lecture)
│   │   │   ├── DEMO_GUIDE.md # Instructor guide for demos
│   │   │   ├── demo1_*.py or .ipynb
│   │   │   └── requirements.txt
│   │   └── media/            # Images, xkcd comics, etc.
├── work/                     # Source materials and planning docs
│   ├── mckinney_content/                # Markdown versions of McKinney chapters
│   │   ├── ch01_preliminaries_extracted.md
│   │   ├── ch02_python_basics_extracted.md
│   │   ├── ch03_builtin_extracted.md
│   │   ├── ch04_numpy_extracted.md
│   │   ├── ch05_pandas_extracted.md
│   │   ├── ch06_data_loading_extracted.md
│   │   ├── ch07_data_cleaning_extracted.md
│   │   ├── ch08_data_wrangling_extracted.md
│   │   ├── ch09_visualization_extracted.md
│   │   ├── ch10_aggregation_extracted.md
│   │   ├── ch11_time_series_extracted.md
│   │   ├── ch12_modeling_extracted.md
│   │   └── ch13_examples_extracted.md
│   ├── mckinney_topics_summary.md      # McKinney content organization
│   ├── missing_semester_topics.md      # CLI/dev workflow content
│   ├── missing-semester-master/        # Missing Semester source materials
│   ├── tlcl_topics.md                  # Linux command line reference
│   ├── implementation_plan.md          # Course design guidelines
│   ├── prerequisites_mapping.md        # Skill dependency tracking
│   ├── lectures_bkp/                   # Previous year's lectures
│   └── example_assignment_with_github_actions/  # Assignment structure template
├── README.md                # Course overview with lecture-to-chapter mapping
├── index.html              # Course website navigation
└── nav-config.js           # Website configuration
```

## Commands

### Running Tests

**Local testing (student workflow):**
```bash
# From assignment directory
python -m pytest test_assignment.py -v
```

**GitHub Actions (auto-grading):**
- Tests run automatically on push to main branch
- Remote tests downloaded from source repository during workflow
- See `.github/workflows/classroom.yml` for configuration

### Virtual Environments

**Using venv (recommended in lectures):**
```bash
# Create environment
python3 -m venv .venv

# Activate (Mac/Linux/WSL)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

**Using uv (modern alternative):**
```bash
uv venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
uv pip install -r requirements.txt
```

### Data Processing (CLI tools emphasized in early lectures)

```bash
# Extract columns from CSV
cut -d',' -f1,3 data.csv

# Count unique values
cut -d',' -f1 data.csv | tail -n +2 | sort | uniq | wc -l

# Filter and count
awk -F',' '$3 > 130' data.csv | wc -l

# Calculate averages
cut -d',' -f5 data.csv | tail -n +2 | awk '{sum+=$1} END {printf "%.2f\n", sum/NR}'
```

## Architecture & Content Guidelines

### Lecture Content Structure

**README.md Format (Notion-compatible):**
- First line: Plain text title (no # prefix for Notion import)
- Headings: Use one less # than standard markdown
  - Section headings: `# Section` (instead of `##`)
  - Subsections: `## Subsection` (instead of `###`)
- Include 2-3 "LIVE DEMO!" callouts to hands-on demos
- Long-form narrative with practical focus
- Include xkcd comics and relevant memes for tone

**Topic Organization:**
1. Brief conceptual description (what it is)
2. Reference section (syntax, parameters, usage)
3. Visual content (1-2 pieces max for visual learners)
4. VERY brief usage example (detailed examples in demos)

**Content Triage (CRITICAL):**
- **Main lecture (README.md):** Essential daily data science tools ONLY - stick to core/practical material
- **BONUS.md (or "DLC"):** Advanced topics, theoretical deep-dives, specialized use cases, material beyond daily tool usage
- When in doubt, move advanced content to BONUS.md - lectures should be lean and practical

### McKinney Content Progression

The course follows McKinney's "Python for Data Analysis" 3rd Ed organization. **See README.md for the definitive lecture-to-chapter mapping.**

Key McKinney chapter resources:
- **Summary:** `work/mckinney_topics_summary.md` - organized by difficulty and practical utility
- **Full chapters:** `work/mckinney_content/chXX_*_extracted.md` - complete markdown versions of chapters 1-13

**General progression (see README.md for exact mapping):**

**Weeks 1-3 (Beginner):**
- Python basics, control flow (Ch 2)
- Built-in data structures (Ch 3)
- Virtual environments and setup (Ch 1)
- NumPy fundamentals begin (Ch 4)

**Weeks 4-8 (Intermediate):**
- NumPy continued (Ch 4)
- Pandas basics (Ch 5)
- File I/O (Ch 6)
- Data cleaning (Ch 7)
- Visualization (Ch 9)

**Weeks 9-11 (Advanced):**
- Data reshaping/merging (Ch 8)
- Group operations (Ch 10)
- Time series (Ch 11)

**Note:** Chapters 12-13 contain capstone/example content that is generally excluded from core lectures but may inform bonus material.

### Assignment Design Principles

**CRITICAL: Create demos and assignments ONLY AFTER lecture content is firmly decided.**

**Structure Requirements:**
- **Regular assignments (Lectures 1-4, 6-10):** Pass/fail auto-gradable via pytest
- **Exam assignments (Lectures 5 and 11):** Larger scope, 0-100 scale, ideally auto-gradable
- Focus on **competence** (can you use the tool?) not **excellence** (did you do it perfectly?)
- Use **incrementally revealed complexity:** Usually 3 questions, each building on previous questions

**GitHub Classroom Integration:**
- `.github/workflows/classroom.yml` auto-grading workflow
- `.github/tests/` directory with remotely-updatable tests
- Remote tests downloaded from source repo during workflow (see example in `work/example_assignment_with_github_actions/`)
- Assignment instructions in `assignment/README.md`
- Student code in `main.py` or similar (NOT in README.md)

**Incremental Complexity Pattern:**
```python
# Question 1: Basic operation (foundation)
# Question 2: Build on Q1 with filtering/transformation
# Question 3: Combine Q1+Q2 concepts with analysis/reporting
```

**Example from Assignment 3:**
1. CLI data tools (foundation: cut, sort, uniq, wc)
2. Virtual environment setup (builds: install packages)
3. NumPy analysis (combines: use installed packages on data)

### Demo Design

**Create demos AFTER lecture content is finalized**
- 2-3 hands-on demos per lecture
- Practical/hands-on review of lecture concepts
- Real-world scenarios with minimal context (lecture provides theory)
- Can be `.py` files (early lectures) or `.ipynb` (Lecture 4+)

**DEMO_GUIDE.md Requirements:**
- Contains ALL steps for ALL demos in the lecture
- Instructive and informative - directly actionable
- **NO instructor talking points** (like "Describe X" or "Explain Y")
- **NO meta-instructions** (like "Now discuss..." or "Tell students about...")
- Instead: Include short explanations directly (bullet lists often work best)
- Integrate explanations with demo steps - show AND explain in one
- Focus: What to do, what it demonstrates, why it matters (concisely)

### Content Sources Integration

**Primary Source (AUTHORITATIVE for Python):**

- McKinney's "Python for Data Analysis" 3rd Ed
- Summary: `work/mckinney_topics_summary.md`
- Full chapters: `work/mckinney_content/chXX_*_extracted.md`
- **McKinney is THE reference** for all Python content organization and progression

**Supplementary Sources:**

- **Command line/shell:** `work/tlcl_topics.md` (The Linux Command Line book)
- **Dev workflows:** `work/missing_semester_topics.md` and `work/missing-semester-master/`
- **Prior patterns:** `work/lectures_bkp/` for established teaching approaches
- **Current best practices:** Web research as needed for tools/workflows

**Integration Strategy:**

- Draw from multiple sources to create cohesive lecture materials
- Integrate CLI and development tools with Python content from McKinney
- See `work/implementation_plan.md` for detailed content standards

**Prerequisite Planning:**

- Follow dependency mapping in `work/prerequisites_mapping.md` (if exists)
- Ensure no skill gaps in lecture sequence
- Each lecture should build on prior knowledge without unexplained jumps

## Key Patterns & Conventions

### Lecture Tone
- Highly knowledgeable with nerdy humor
- Include xkcd comics where relevant
- Occasional appropriate memes
- Approachable but technically rigorous

### Testing Patterns

**Pytest structure for assignments:**
```python
# .github/tests/test_assignment.py (remote, updatable)
import pytest
from pathlib import Path

def test_output_exists():
    assert Path("output/result.txt").exists()

def test_output_content():
    with open("output/result.txt") as f:
        content = f.read()
    assert "expected_value" in content
```

**Local vs Remote Tests:**
- Local `test_assignment.py`: Student reference for development
- `.github/tests/test_assignment.py`: Downloaded fresh each run, can be updated without student repo changes

### Virtual Environment Management
- Use `.venv` as standard name (gitignored)
- Include `requirements.txt` in all assignment and demo directories
- Emphasize activation verification: `which python`
- Teach deactivation for switching projects

### Data File Conventions
- Sample data generation scripts provided where applicable
- CSV format preferred for tabular data
- Include schema documentation in assignment README
- Expected outputs in `expected_outputs/` or `output/` directories

## Development Workflow

### When Creating New Lectures
1. **Plan content** using McKinney progression and prerequisites
2. **Write README.md** with core concepts (Notion format)
3. **Create BONUS.md** for advanced topics
4. **Design assignment** with incremental complexity after lecture is stable
5. **Build demos** that reinforce lecture + assignment concepts
6. **Test auto-grading** workflow

### When Updating Assignments
1. **Modify remote tests** in `.github/tests/` first
2. **Update local tests** in assignment root for student reference
3. **Verify requirements.txt** has all needed dependencies
4. **Test workflow** by pushing to test repository

### When Working with Git (GUI-focused)
- Students use VS Code Source Control panel, not command line git
- GitHub Desktop is acceptable alternative
- Focus on: commit, push, pull, branch creation via GUI
- Command-line git reserved for BONUS content

### Git Commit Messages
- **NO Co-authored-by tags** - Never add "Co-Authored-By: Claude" or similar AI attribution
- **NO cutesy messages** - Keep commits professional and technical
- **Clean commit messages** - Focus on technical changes and impact
- **Standard format** - Use conventional commit format when appropriate (feat:, fix:, docs:, etc.)

## Important Notes & Course-Specific Rules

### Critical Development Rules
1. **No premature content creation:** Don't create demos/assignments before lecture content is firmly decided and stable
2. **Demos and assignments:** Only create AFTER lecture content is finalized
3. **Content focus:** Advanced content → BONUS.md; core/practical material → README.md

### Assignment Rules
4. **Regular assignments (1-4, 6-10):** Pass/fail, pytest auto-gradable, competence-focused
5. **Exam assignments (5, 11):** Larger scope, 0-100 scale, ideally auto-gradable
6. **Question structure:** Usually 3 questions with incrementally revealed complexity (each builds on previous)
7. **Grading philosophy:** Test "can you do it?" NOT "did you do it perfectly?"

### Technology & Workflow Rules
8. **Jupyter delayed:** Not introduced until Lecture 4 (Pandas). Use `.py` files for Lectures 1-3.
9. **CLI emphasis:** Command-line data processing is core skill through Lecture 3
10. **Git workflow:** GUI-based (VS Code/GitHub Desktop), NOT command-line git
11. **VS Code focus:** Development primarily in VS Code, GUI-oriented

### Content Sources
12. **McKinney is authoritative:** For Python content organization and progression
13. **Lecture mapping:** README.md contains definitive lecture-to-chapter mapping
14. **Chapter resources:** Full chapters in `work/mckinney_content/chXX_*_extracted.md`

## External Resources

- Main textbook: [Python for Data Analysis (McKinney)](https://wesmckinney.com/book/)
- Command line: [The Missing Semester](https://missing.csail.mit.edu/)
- Shell reference: [The Linux Command Line](http://linuxcommand.org/tlcl.php)
- Course site: https://not.badmath.org/ds217
- GitHub repo: https://github.com/christopherseaman/datasci_217
