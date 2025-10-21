#!/usr/bin/env python3
"""
Execute Q5 Missing Data Analysis
Runs all analysis steps from the notebook in sequence.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import os

# Import utilities from Q3
from q3_data_utils import load_data, detect_missing, fill_missing, clean_data

# Ensure output directory exists
os.makedirs('output', exist_ok=True)

print("="*70)
print("Q5: MISSING DATA ANALYSIS")
print("="*70)

# Load the data
df = load_data('data/clinical_trial_raw.csv')
print(f"\nLoaded {len(df)} patients with {len(df.columns)} columns")

# ============================================================================
# PART 1: DETECT MISSING DATA
# ============================================================================
print("\n" + "="*70)
print("PART 1: DETECT MISSING DATA")
print("="*70)

missing_counts = detect_missing(df)

print("\nMissing Value Counts:")
missing_display = missing_counts[missing_counts > 0]
if len(missing_display) > 0:
    print(missing_display)
else:
    print("No missing values found!")

# Calculate percentages
missing_pct = (missing_counts / len(df) * 100).round(2)
print("\nMissing Value Percentages:")
print(missing_pct[missing_pct > 0])

# Identify columns with missing data
cols_with_missing = missing_counts[missing_counts > 0].index.tolist()
print(f"\nColumns with missing data ({len(cols_with_missing)}):")
for col in cols_with_missing:
    print(f"  - {col}: {missing_counts[col]} ({missing_pct[col]}%)")

# Visualize and save
if len(missing_display) > 0:
    plt.figure(figsize=(10, 6))
    missing_display.plot(kind='bar')
    plt.title('Missing Values by Column')
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Number of Missing Values')
    plt.tight_layout()
    plt.savefig('output/q5_missing_values_plot.png', dpi=300, bbox_inches='tight')
    print("\nVisualization saved to output/q5_missing_values_plot.png")
    plt.close()

# ============================================================================
# PART 2: COMPARE IMPUTATION STRATEGIES
# ============================================================================
print("\n" + "="*70)
print("PART 2: COMPARE IMPUTATION STRATEGIES FOR BMI")
print("="*70)

# Check if BMI has missing values
if 'bmi' in df.columns and df['bmi'].isnull().sum() > 0:
    # Original statistics
    original_mean = df['bmi'].mean()
    original_median = df['bmi'].median()
    original_missing = df['bmi'].isnull().sum()

    print(f"\nOriginal BMI Statistics:")
    print(f"  Missing values: {original_missing}")
    print(f"  Mean (excluding missing): {original_mean:.2f}")
    print(f"  Median (excluding missing): {original_median:.2f}")

    # Strategy 1: Mean imputation
    df_mean = fill_missing(df, 'bmi', strategy='mean')
    mean_after_mean = df_mean['bmi'].mean()
    median_after_mean = df_mean['bmi'].median()

    # Strategy 2: Median imputation
    df_median = fill_missing(df, 'bmi', strategy='median')
    mean_after_median = df_median['bmi'].mean()
    median_after_median = df_median['bmi'].median()

    # Strategy 3: Forward fill
    df_ffill = fill_missing(df, 'bmi', strategy='ffill')
    mean_after_ffill = df_ffill['bmi'].mean()
    median_after_ffill = df_ffill['bmi'].median()
    ffill_remaining = df_ffill['bmi'].isnull().sum()

    # Create comparison table
    comparison = pd.DataFrame({
        'Strategy': ['Original', 'Mean Imputation', 'Median Imputation', 'Forward Fill'],
        'Missing Count': [original_missing, 0, 0, ffill_remaining],
        'Mean': [original_mean, mean_after_mean, mean_after_median, mean_after_ffill],
        'Median': [original_median, median_after_mean, median_after_median, median_after_ffill]
    })

    print("\nImputation Strategy Comparison:")
    print(comparison.to_string(index=False))

    print("\nObservations:")
    print(f"  - Mean imputation preserves the original mean ({mean_after_mean:.2f} ≈ {original_mean:.2f})")
    print(f"  - Median imputation slightly shifts mean to {mean_after_median:.2f}")
    print(f"  - Forward fill creates mean of {mean_after_ffill:.2f} (depends on data order)")
    print(f"  - Mean/Median filled all {original_missing} values; ffill left {ffill_remaining} missing")
else:
    print("\nNo missing values in BMI column to impute")

# ============================================================================
# PART 3: DROPPING MISSING DATA
# ============================================================================
print("\n" + "="*70)
print("PART 3: COMPARE DROPPING STRATEGIES")
print("="*70)

original_rows = len(df)
print(f"\nOriginal dataset: {original_rows} rows")

# Strategy 1: Drop rows with ANY missing values
df_dropany = df.dropna()
dropany_rows = len(df_dropany)
dropany_loss = original_rows - dropany_rows
dropany_loss_pct = (dropany_loss / original_rows * 100)

print(f"\nStrategy 1 - Drop ANY missing:")
print(f"  Rows remaining: {dropany_rows}")
print(f"  Rows lost: {dropany_loss} ({dropany_loss_pct:.1f}%)")

# Strategy 2: Drop rows with missing in critical columns only
critical_cols = ['patient_id', 'age', 'sex', 'site', 'intervention_group']
# Only use columns that exist
critical_cols = [col for col in critical_cols if col in df.columns]
df_dropcritical = df.dropna(subset=critical_cols)
dropcritical_rows = len(df_dropcritical)
dropcritical_loss = original_rows - dropcritical_rows
dropcritical_loss_pct = (dropcritical_loss / original_rows * 100)

print(f"\nStrategy 2 - Drop if missing critical columns {critical_cols}:")
print(f"  Rows remaining: {dropcritical_rows}")
print(f"  Rows lost: {dropcritical_loss} ({dropcritical_loss_pct:.1f}%)")

# Strategy 3: Drop rows with missing in specific columns (bmi only)
if 'bmi' in df.columns:
    df_dropbmi = df.dropna(subset=['bmi'])
    dropbmi_rows = len(df_dropbmi)
    dropbmi_loss = original_rows - dropbmi_rows
    dropbmi_loss_pct = (dropbmi_loss / original_rows * 100)

    print(f"\nStrategy 3 - Drop if missing BMI only:")
    print(f"  Rows remaining: {dropbmi_rows}")
    print(f"  Rows lost: {dropbmi_loss} ({dropbmi_loss_pct:.1f}%)")

    # Create comparison
    drop_comparison = pd.DataFrame({
        'Strategy': ['Drop ANY missing', 'Drop critical missing', 'Drop BMI missing'],
        'Rows Remaining': [dropany_rows, dropcritical_rows, dropbmi_rows],
        'Rows Lost': [dropany_loss, dropcritical_loss, dropbmi_loss],
        'Percentage Lost': [f"{dropany_loss_pct:.1f}%", f"{dropcritical_loss_pct:.1f}%", f"{dropbmi_loss_pct:.1f}%"]
    })

    print("\nComparison Table:")
    print(drop_comparison.to_string(index=False))

    print("\nConclusion:")
    print(f"  Dropping rows with ANY missing value loses {dropany_loss_pct:.1f}% of data.")
    print(f"  Dropping rows only when critical columns are missing loses {dropcritical_loss_pct:.1f}% of data.")
    if dropcritical_rows > dropany_rows:
        print(f"  The selective approach preserves {dropcritical_rows - dropany_rows} more rows.")

# ============================================================================
# PART 4: CREATE CLEAN DATASET
# ============================================================================
print("\n" + "="*70)
print("PART 4: CREATE CLEAN DATASET")
print("="*70)

# Start with a copy
df_clean = df.copy()

# Step 0: Replace sentinel values with NaN
print(f"\nStep 0: Replace sentinel values (-999, -1) with NaN")
sentinel_values = [-999, -1]
sentinel_count = 0
for sentinel in sentinel_values:
    count = (df_clean == sentinel).sum().sum()
    sentinel_count += count
    df_clean = df_clean.replace(sentinel, pd.NA)
print(f"  Replaced {sentinel_count} sentinel values with NaN")

# Step 1: Drop rows with missing critical demographic/identifier columns
critical_cols_clean = ['patient_id', 'age', 'sex']
critical_cols_clean = [col for col in critical_cols_clean if col in df_clean.columns]
before_drop = len(df_clean)
df_clean = df_clean.dropna(subset=critical_cols_clean)
after_drop = len(df_clean)
print(f"\nStep 1: Dropped {before_drop - after_drop} rows with missing critical columns: {critical_cols_clean}")

# Step 2: Impute numeric clinical measurements with median
numeric_cols_to_impute = ['bmi', 'systolic_bp', 'diastolic_bp', 'cholesterol_total',
                          'cholesterol_hdl', 'cholesterol_ldl', 'glucose_fasting',
                          'adherence_pct']

imputed_counts = {}
for col in numeric_cols_to_impute:
    if col in df_clean.columns:
        missing_before = df_clean[col].isnull().sum()
        if missing_before > 0:
            df_clean = fill_missing(df_clean, col, strategy='median')
            imputed_counts[col] = missing_before

print(f"\nStep 2: Imputed missing values in {len(imputed_counts)} numeric columns:")
for col, count in imputed_counts.items():
    print(f"  - {col}: {count} values")

# Verify cleaning
final_missing = detect_missing(df_clean)
remaining_missing = final_missing[final_missing > 0]

print(f"\nFinal missing value counts:")
if len(remaining_missing) > 0:
    print(remaining_missing)
else:
    print("  No missing values remain!")

print(f"\nDataset summary:")
print(f"  Original rows: {len(df)}")
print(f"  Clean rows: {len(df_clean)}")
print(f"  Rows removed: {len(df) - len(df_clean)} ({(len(df) - len(df_clean))/len(df)*100:.1f}%)")

# Save clean dataset
df_clean.to_csv('output/q5_cleaned_data.csv', index=False)
print("\n✓ Clean dataset saved to output/q5_cleaned_data.csv")

# ============================================================================
# CREATE AND SAVE MISSING DATA REPORT
# ============================================================================
report_lines = [
    "MISSING DATA ANALYSIS REPORT",
    "="*60,
    "",
    "Original Dataset:",
    f"  Total rows: {len(df)}",
    f"  Total columns: {len(df.columns)}",
    "",
    "Missing Data Summary:",
]

original_missing = detect_missing(df)
missing_cols = original_missing[original_missing > 0]
if len(missing_cols) > 0:
    for col in missing_cols.index:
        count = original_missing[col]
        pct = (count / len(df) * 100)
        report_lines.append(f"  {col}: {count} ({pct:.1f}%)")
else:
    report_lines.append("  No missing values found in original dataset")

report_lines.extend([
    "",
    "Cleaning Strategy Applied:",
    f"  1. Dropped rows with missing critical columns: {critical_cols_clean}",
    f"     Rows removed: {before_drop - after_drop}",
    "",
    "  2. Imputed numeric columns with MEDIAN (robust to outliers):"
])

if imputed_counts:
    for col, count in imputed_counts.items():
        report_lines.append(f"     - {col}: {count} values imputed")
else:
    report_lines.append("     - No imputation needed")

report_lines.extend([
    "",
    "Final Clean Dataset:",
    f"  Total rows: {len(df_clean)}",
    f"  Rows retained: {len(df_clean)/len(df)*100:.1f}%",
    f"  Remaining missing values: {final_missing.sum()}",
    "",
    "Rationale:",
    "  - Used median imputation (robust to outliers) for clinical measurements",
    "  - Preserved maximum data while ensuring critical fields are complete",
    "  - Median preferred over mean for medical data with potential outliers",
    f"  - Acceptable data retention rate ({len(df_clean)/len(df)*100:.1f}%) for analysis",
    "",
    "Recommendation:",
    "  For this clinical trial dataset, a hybrid strategy is optimal:",
    "  1. DROP rows with missing critical demographic/identifier fields",
    "  2. IMPUTE clinical measurements using MEDIAN (robust to outliers)",
    "  3. AVOID forward fill (inappropriate for cross-sectional data)",
    "",
    "  This balances data retention with data quality, ensuring critical",
    "  fields are complete while using statistically sound imputation for",
    "  secondary measurements. For production, consider adding missing-",
    "  indicator variables to flag imputed values for sensitivity analyses."
])

report_text = "\n".join(report_lines)
with open('output/q5_missing_report.txt', 'w') as f:
    f.write(report_text)

print("\n✓ Missing data report saved to output/q5_missing_report.txt")

print("\n" + "="*70)
print("Q5 ANALYSIS COMPLETE")
print("="*70)
print("\nOutputs generated:")
print("  1. output/q5_cleaned_data.csv - Clean dataset")
print("  2. output/q5_missing_report.txt - Analysis report")
print("  3. output/q5_missing_values_plot.png - Visualization")
print("="*70)
