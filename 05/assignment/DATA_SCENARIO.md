# Assignment 5 Midterm: Clinical Trial Data Analysis Scenario

## Context

You're a data analyst at a research hospital working with data from a multi-site clinical trial studying cardiovascular health interventions. The trial collected patient data across 5 hospital sites over 2 years, tracking various health metrics and outcomes.

**Your Task:** Build a data processing pipeline to clean, validate, and analyze the clinical trial data for the research team's preliminary analysis.

---

## Dataset: Clinical Trial Patient Data

**File:** `data/clinical_trial_raw.csv`

**~500 patients** with the following variables:

### Patient Demographics
- `patient_id` - Unique identifier (e.g., "P001", "P002")
- `age` - Age in years
- `sex` - Biological sex (M/F)
- `bmi` - Body Mass Index
- `enrollment_date` - Date enrolled in trial (YYYY-MM-DD format)

### Clinical Measurements
- `systolic_bp` - Systolic blood pressure (mmHg)
- `diastolic_bp` - Diastolic blood pressure (mmHg)
- `cholesterol_total` - Total cholesterol (mg/dL)
- `cholesterol_hdl` - HDL cholesterol (mg/dL)
- `cholesterol_ldl` - LDL cholesterol (mg/dL)
- `glucose_fasting` - Fasting glucose (mg/dL)

### Trial Information
- `site` - Hospital site (Site A, Site B, Site C, Site D, Site E)
- `intervention_group` - Treatment group (Control, Treatment A, Treatment B)
- `follow_up_months` - Months in study
- `adverse_events` - Count of adverse events

### Outcome Data
- `outcome_cvd` - Cardiovascular disease event (Yes/No)
- `adherence_pct` - Treatment adherence percentage
- `dropout` - Dropped out of study (Yes/No)

---

## Data Quality Issues (Realistic Clinical Data Problems)

### Missing Data
- Some patients missing BMI measurements (equipment failure)
- Blood pressure readings missing for patients who skipped visits
- Cholesterol tests not performed for some patients (cost constraints)
- Follow-up data incomplete for recent enrollees

### Data Entry Errors
- Some ages recorded as "-999" (sentinel value from data entry system)
- BMI values occasionally entered as weight instead of BMI (>300)
- Blood pressure values swapped (diastolic > systolic)
- Enrollment dates in inconsistent formats

### Inconsistencies
- Site names with inconsistent capitalization ("site a" vs "SITE A")
- Sex coded as M/F in some records, Male/Female in others
- Intervention groups with typos ("Treatmen A", "Contrl")
- Whitespace in text fields

### Data Type Issues
- Dates stored as strings
- Categorical variables stored as strings
- Numeric fields with text entries ("not measured", "N/A")

---

## Analysis Questions (What You'll Answer)

### Data Quality Assessment
1. How complete is the dataset? (missing data by variable and site)
2. Are there outliers or impossible values that need investigation?
3. What percentage of patients have complete cardiovascular measurements?

### Descriptive Statistics
1. What are the baseline characteristics by intervention group?
2. Which site enrolled the most patients?
3. What's the average adherence rate by intervention group?

### Clinical Insights
1. How many patients experienced cardiovascular events by group?
2. What's the relationship between adherence and outcomes?
3. Are there site-specific differences in baseline characteristics?

---

## Configuration File (`config.txt`)

Trial-specific parameters:

```
study_name=CardioHealth Trial 2023
primary_investigator=Dr. Sarah Chen
enrollment_start=2022-01-01
enrollment_end=2023-12-31
min_age=18
max_age=85
target_enrollment=500
sites=5
intervention_groups=3
```

---

## Expected Deliverables

### Q1: Project Setup
- Organize data files into proper directory structure
- Set up analysis workspace

### Q2: Metadata Processing
- Parse and validate trial configuration
- Check enrollment targets and parameters
- Generate trial summary report

### Q3: Data Loading & Exploration
- Load clinical trial dataset
- Generate summary statistics
- Identify data quality issues

### Q4: Data Selection & Filtering
- Extract subsets of patients (e.g., complete cases, specific sites)
- Filter by inclusion criteria

### Q5: Missing Data Handling
- Identify missing patterns by site and variable
- Apply appropriate imputation strategies
- Document missing data decisions

### Q6: Data Transformation
- Convert dates to proper format
- Standardize categorical variables
- Create derived variables (e.g., LDL/HDL ratio, CVD risk categories)
- Clean text fields

### Q7: Group Analysis
- Calculate statistics by intervention group
- Identify top-enrolling sites
- Analyze adherence patterns

### Q8: Automated Pipeline
- Run complete analysis pipeline
- Generate final clean dataset
- Produce data quality report

---

## Clinical Relevance

This assignment mirrors real-world health data science workflows:

1. **Regulatory Compliance** - Clean, documented data is required for clinical trial submissions
2. **Data Quality** - Poor quality data can invalidate study results and waste research funding
3. **Reproducibility** - Automated pipelines ensure analysis can be verified and reproduced
4. **Collaboration** - Multiple sites require standardized data processing
5. **Clinical Impact** - Proper analysis informs treatment decisions affecting patient care

**You're not just learning data science - you're learning how to handle data that impacts human health!**

---

## Ethical Considerations

- Data is **synthetic** - no real patient information
- In real clinical trials:
  - Patient privacy (HIPAA compliance)
  - Data security and access controls
  - Audit trails for all data modifications
  - IRB approval for analysis plans
  - Proper handling of sensitive health information

---

## Success Criteria

Your pipeline should produce:
- ✅ Clean dataset ready for statistical analysis
- ✅ Data quality report documenting all issues found and resolved
- ✅ Reproducible workflow (scripts can be re-run)
- ✅ Summary statistics by intervention group
- ✅ Documented decisions (which imputation methods, why?)

**Remember:** In clinical research, every data decision matters. Document everything!
