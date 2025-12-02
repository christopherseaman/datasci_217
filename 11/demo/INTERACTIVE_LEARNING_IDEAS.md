# Interactive Learning Prompts for Future Consideration

These interactive exercises were removed from the main notebooks to keep them focused, but could be valuable additions in the future.

---

## 01_setup.md - IQR Multiplier Experimentation

**Location:** After outlier detection using IQR method (around line 700)

**Concept:** Let students experiment with different IQR multipliers to understand the trade-off between sensitivity and conservatism in outlier detection.

```python
# Compare standard vs. conservative outlier detection
outliers_15, _, _ = detect_outliers_iqr(df, 'trip_distance', iqr_multiplier=1.5)
outliers_30, _, _ = detect_outliers_iqr(df, 'trip_distance', iqr_multiplier=3.0)

display(Markdown(f"""
### Comparing IQR Multipliers

| IQR Multiplier | Outliers Detected | % of Data |
|----------------|-------------------|-----------|
| **1.5** (standard) | {len(outliers_15):,} | {len(outliers_15)/len(df)*100:.2f}% |
| **3.0** (conservative) | {len(outliers_30):,} | {len(outliers_30)/len(df)*100:.2f}% |

Lower multiplier = more sensitive = finds more outliers

**Reflection question:** Which threshold is more appropriate for this dataset?
Consider whether these are true outliers or just NYC's natural variability (airport trips, traffic, etc.).
"""))
```

**Learning objectives:**
- Understand the impact of threshold selection on outlier detection
- Learn to balance sensitivity vs. specificity
- Apply domain knowledge to statistical decisions
- Recognize that "outliers" can be valid extreme values in real-world data

---

## 04_modeling.md - Hyperparameter Experimentation

**Location:** After defining model hyperparameters (around line 145)

**Concept:** Guide students through systematic hyperparameter tuning by changing one parameter at a time.

```markdown
### Hyperparameter Experimentation Guide

The constants above make it easy to experiment! Try modifying these values to see their impact:

| Parameter | Current | Try This | Expected Effect |
|-----------|---------|----------|-----------------|
| `RF_N_ESTIMATORS` | 100 | 50, 200 | More trees = better performance but slower training |
| `RF_MAX_DEPTH` | 10 | 5, 15 | Shallower = less overfitting; deeper = more complex patterns |
| `XGB_LEARNING_RATE` | 0.1 | 0.01, 0.3 | Lower = more conservative (may need more estimators) |
| `XGB_MAX_DEPTH` | 6 | 3, 9 | Shallower trees = less overfitting |

**Exercise:** Change ONE parameter at a time, re-run the models, and observe:
1. How does Test R² change?
2. Does overfitting increase or decrease?
3. How does training time change?

**Pro Tip:** Use the `assess_overfitting()` function to see nuanced feedback on model generalization!
```

**Learning objectives:**
- Understand hyperparameter impact on model performance
- Practice systematic experimentation (one variable at a time)
- Learn the bias-variance tradeoff through hands-on experimentation
- Develop intuition for model tuning

---

## Additional Interactive Exercise Ideas

### 1. Feature Engineering Experimentation (02_wrangling.md)

**Concept:** Let students modify time-of-day boundaries and see how it affects patterns

```python
# Try different time-of-day boundaries
MORNING_START = 6    # What if morning starts at 6 instead of 5?
EVENING_START = 18   # What if evening starts at 6pm instead of 5pm?

# Re-run the analysis and compare:
# - Trip counts by time of day
# - Average fares by time of day
# - Do the patterns still make sense?
```

### 2. Train/Test Split Experimentation (03_model_prep.md)

**Concept:** Show impact of different train/test split ratios

```python
# Compare different split ratios
for ratio in [0.60, 0.70, 0.80, 0.90]:
    split_date = df_model['pickup_datetime'].quantile(ratio)
    train_size = len(df_model[df_model['pickup_datetime'] < split_date])
    test_size = len(df_model[df_model['pickup_datetime'] >= split_date])

    print(f"Split {ratio:.0%}: Train={train_size:,}, Test={test_size:,}")

# Question: How does split ratio affect model evaluation reliability?
```

### 3. Distance Category Experimentation (02_wrangling.md)

**Concept:** Let students redefine distance categories and see impact on analysis

```python
# Original categories: <1, 1-3, 3-10, 10+
# Try finer-grained: <0.5, 0.5-1, 1-2, 2-5, 5-10, 10+
# Try coarser: <2, 2-5, 5+

# How does this change:
# - Distribution of trips across categories?
# - Average fare per category?
# - Insights about NYC taxi usage patterns?
```

---

## Implementation Considerations

### When to add these back:

1. **After core concepts are mastered** - Don't overwhelm students on first pass
2. **In lab/workshop sessions** - Better suited for hands-on practice time
3. **As optional "going further" sections** - For advanced students
4. **In separate "experimentation notebook"** - Keep main notebooks focused

### Design principles:

- **One variable at a time** - Avoid confounding factors
- **Clear expected outcomes** - Set expectations before experiment
- **Reflection questions** - Force students to think about what they observe
- **Real-world context** - Tie experiments to business/domain decisions

### Technical requirements:

- Ensure experiments run quickly (< 30 seconds each)
- Provide clear before/after comparison
- Use visualizations where helpful
- Include "reset to defaults" code block
