The Complete Data Science Workflow: From Raw Data to Insights

![xkcd 3172: Fifteen Years](media/fifteen_years_2x.png)

*"I used to be a data scientist, but then I automated myself out of a job. Now I'm just a script that runs every morning."*

# Outline

This final lecture walks through a **complete data science project** from start to finish, demonstrating how all the tools and techniques from Lectures 01-10 work together in a real-world workflow.

- **Phase 1-3:** Setup, exploration, and data cleaning
- **Phase 4-5:** Data wrangling, transformation, and feature engineering
- **Phase 6-7:** Pattern analysis and modeling preparation
- **Phase 8-9:** Modeling, evaluation, and results communication

**Demo dataset:** A compact, frozen NYC Yellow Taxi release derived from January-June 2023 trip records. The required workflow predicts next-hour pickup counts; an optional fifth demo uses taxi-zone polygons to help ground the data.

# The Complete Workflow

*Reality check: Real data science projects don't follow a linear path. You'll iterate, backtrack, and discover new questions as you explore. But having a structured workflow helps you stay organized and ensures you don't miss critical steps.*

This lecture demonstrates the **complete data science lifecycle**:

```
Raw Data → Exploration → Cleaning → Wrangling → Feature Engineering
    → Analysis → Modeling → Results → Insights
```

Each phase builds on the previous one, and we'll use techniques from every lecture in this course.

**Key Principle:** This isn't just about individual techniques - it's about **integrating everything you've learned** into a cohesive analytical process.

# Notebook Structure

This lecture is organized into **4 interactive notebooks** that you can follow along with:

1. **Notebook 1: Setup, Exploration & Cleaning**
   - Loading and inspecting data
   - Initial exploration and visualization
   - Systematic data cleaning workflow

2. **Notebook 2: Wrangling & Feature Engineering**
   - Establishing a complete zone-hour panel
   - Time series datetime handling
   - Creating past-only lag and rolling features

3. **Notebook 3: Pattern Analysis & Modeling Prep**
   - Training-only pattern analysis
   - Identifying hourly and weekly structure
   - Temporal train/validation/test splitting
   - Feature-availability checks

4. **Notebook 4: Modeling & Results**
   - A weekly-lag baseline and one scikit-learn pipeline
   - Validation-based model choice
   - One untouched test evaluation and error slices
   - Final visualizations and results communication

## Phase-to-Notebook Mapping

This is really only useful if you get stuck on the final and want to know where to look in today’s lecture.

**Notebook 1:** Phases 1-3 (Setup, Exploration, Cleaning)
- Phase 1: Project Setup & Data Acquisition
- Phase 2: Data Exploration & Understanding
- Phase 3: Data Cleaning & Preprocessing

**Notebook 2:** Phases 4-5 (Wrangling, Feature Engineering)
- Phase 4: Data Wrangling & Transformation
- Phase 5: Feature Engineering & Aggregation

**Notebook 3:** Phases 6-7 (Pattern Analysis, Modeling Prep)
- Phase 6: Pattern Analysis & Advanced Visualization
- Phase 7: Modeling Preparation

**Notebook 4:** Phases 8-9 (Modeling, Results)
- Phase 8: Modeling
- Phase 9: Results & Insights

**How to Use:**
- Each notebook can be run independently (after previous notebooks)
- Follow along during lecture or work through them on your own
- All notebooks use the NYC Taxi Trip Dataset
- Code is executable and well-documented

# Time Series Component

**Note:** This project includes **time series analysis** as a required component. The NYC Taxi dataset has temporal patterns that require time series operations:
- Hourly and daily patterns
- Day-of-week effects
- Seasonal trends
- Time-based feature engineering

You'll see datetime operations, resampling, rolling windows, and temporal modeling - applying the time series skills from Lecture 09 within a complete project workflow.

## Workflow Requirements Checklist

Use this checklist to ensure you complete all required components:

**Phase 1-2 (Setup & Exploration):**
- [ ] Data loaded successfully
- [ ] Initial inspection completed (shape, info, describe)
- [ ] Missing values identified
- [ ] Basic visualizations created (distributions, time series if applicable)
- [ ] Data quality issues documented

**Phase 3 (Cleaning):**
- [ ] Missing data handling strategy chosen and implemented
- [ ] Outliers detected and handled
- [ ] Data types validated and converted
- [ ] Duplicates identified and removed
- [ ] Cleaning decisions documented

**Phase 4 (Wrangling):**
- [ ] Datetime columns parsed correctly
- [ ] Datetime index set
- [ ] Temporal features extracted (hour, day_of_week, month minimum)
- [ ] Multiple datasets merged (if applicable)
- [ ] Data reshaped as needed

**Phase 5 (Feature Engineering):**
- [ ] Derived features created
- [ ] Time-based aggregations performed
- [ ] At least one rolling window calculation
- [ ] Categorical features created (if applicable)
- [ ] Feature list documented

**Phase 6 (Pattern Analysis):**
- [ ] Trends over time identified
- [ ] Seasonal patterns analyzed
- [ ] Correlation analysis completed
- [ ] Advanced visualizations created
- [ ] Key patterns documented

**Phase 7 (Modeling Prep):**
- [ ] Target and one-hour prediction horizon selected
- [ ] Temporal train/validation/test split performed (NOT random split)
- [ ] Features selected and prepared
- [ ] Categorical variables handled (encoding if needed)
- [ ] No data leakage (future data not in training set)

**Phase 8 (Modeling):**
- [ ] Weekly-lag baseline calculated
- [ ] One supported model selected using validation only
- [ ] Final performance evaluated once on the untouched test set
- [ ] Error slices checked by zone and local hour
- [ ] Model performance documented

**Phase 9 (Results):**
- [ ] Final visualizations created
- [ ] Summary tables generated
- [ ] Key findings documented
- [ ] Results communicated clearly

# Connection to Final Assignment

This lecture demonstrates the **same workflow** that you'll use in the final assignment, but with:
- **More detailed explanations** and rationale
- **Live demonstrations** of techniques
- **Best practices** and professional workflows
- **Troubleshooting** common issues

The final assignment follows the same nine-phase structure with a different dataset: a frozen Chicago Beach Weather Sensors release. It focuses on **applying** the workflow rather than repeating the taxi example.

**Key Difference:** This lecture teaches the workflow. The assignment assesses your ability to execute it.

# Learning Objectives

By the end of this lecture, you should be able to:

- Execute a complete data science workflow from raw data to insights
- Integrate techniques from all previous lectures
- Handle real-world data quality issues
- Perform time series analysis on temporal data
- Build and evaluate predictive models
- Communicate results effectively

# Getting Started

1. **Navigate to the demo folder:**
   ```bash
   cd 11/demo
   ```

2. **Set up the environment:**
   ```bash
   uv venv .venv
   source .venv/bin/activate
   uv pip install -r requirements.txt
   ```

3. **Verify the compact NYC Taxi demo release:**
   ```bash
   chmod +x download_data.sh
   ./download_data.sh
   ```

4. **Open the notebooks in order:**
   - `01_setup.ipynb` - Setup, Exploration & Cleaning
   - `02_wrangling.ipynb` - Wrangling & Feature Engineering
   - `03_model_prep.ipynb` - Pattern Analysis & Modeling Prep
   - `04_modeling.ipynb` - Modeling & Results

   Optional: `05_geo_bonus.ipynb` uses supplied geospatial machinery to map zone-level results. It is not assignment material.

5. **Follow along** or work through them independently

6. **Run each cell** to see the results and understand the workflow

---

*"The best way to learn data science is to do data science. This lecture gives you a complete, real-world example of how it all fits together."*

# Holiday fun…

- [Advent of Code](https://adventofcode.com) - Get in the holiday spirit!
- [GameShell](https://github.com/phyver/GameShell) - A game to learn the Unix shell

![xkcd 1513: Code Quality](media/xkcd_1513.png)

*"I honestly didn't think you could even USE emoji in variable names. Or that there were so many different crying ones."*
