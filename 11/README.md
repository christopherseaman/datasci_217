The Complete Data Science Workflow: From Raw Data to Insights

See [BONUS.md](BONUS.md) for advanced topics (if applicable).

*Fun fact: This lecture represents the culmination of everything you've learned - taking raw, messy data and transforming it into actionable insights. It's like going from a pile of ingredients to a gourmet meal, except the ingredients are CSV files and the meal is a predictive model.*

![xkcd 1513: Automation](media/xkcd_1513.png)

*"I used to be a data scientist, but then I automated myself out of a job. Now I'm just a script that runs every morning."*

# Outline

This final lecture walks through a **complete data science project** from start to finish, demonstrating how all the tools and techniques from Lectures 01-10 work together in a real-world workflow.

- **Phase 1-3:** Setup, exploration, and data cleaning
- **Phase 4-5:** Data wrangling, transformation, and feature engineering
- **Phase 6-7:** Pattern analysis and modeling preparation
- **Phase 8-9:** Modeling, evaluation, and results communication

**Dataset:** NYC Taxi Trip Dataset - millions of taxi trips with rich temporal patterns, location data, and fare information.

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
   - Merging datasets
   - Time series datetime handling
   - Reshaping and aggregations
   - Creating derived features

3. **Notebook 3: Pattern Analysis & Modeling Prep**
   - Advanced visualizations
   - Identifying trends and seasonality
   - Temporal train/test splitting
   - Feature preparation

4. **Notebook 4: Modeling & Results**
   - Model training and evaluation
   - Model interpretation
   - Final visualizations
   - Results communication

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

# Connection to Final Exam

This lecture demonstrates the **same workflow** that you'll use in the final exam, but with:
- **More detailed explanations** and rationale
- **Live demonstrations** of techniques
- **Best practices** and professional workflows
- **Troubleshooting** common issues

The final exam will follow the same 9-phase structure but use a different dataset (Chicago Beach Weather Sensors) and focus on **applying** these skills rather than learning them.

**Key Difference:** This lecture teaches the workflow. The exam assesses your ability to execute it.

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

2. **Open the notebooks in order:**
   - `01_setup_exploration_cleaning.ipynb`
   - `02_wrangling_feature_engineering.ipynb`
   - `03_pattern_analysis_modeling_prep.ipynb`
   - `04_modeling_results.ipynb`

3. **Follow along** with the instructor or work through them independently

4. **Run each cell** to see the results and understand the workflow

**Note:** You'll need the NYC Taxi Trip Dataset. See the first notebook for download instructions.

---

*"The best way to learn data science is to do data science. This lecture gives you a complete, real-world example of how it all fits together."*

