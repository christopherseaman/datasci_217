# Hints & Troubleshooting: Final Exam

Common issues, solutions, and hints for the Chicago Beach Weather Sensors analysis.

---

## Q6 Feature Selection Hints

When preparing features for modeling, you'll need to exclude non-predictive columns. For this dataset, consider excluding:

- **Identifiers:** `Measurement ID`, `Measurement Timestamp Label`
- **Categorical text:** `Station Name`, `Precipitation Type` (unless you one-hot encode them)
- **The target variable** and any features derived from it

Use `df.select_dtypes(include=[np.number])` to select only numeric columns, then manually exclude the target.

---

## Data Loading Issues

### Problem: Dataset won't load
**Symptoms:** `FileNotFoundError` or `pd.read_csv()` fails

**Solutions:**
1. Check that `download_data.sh` ran successfully
2. Verify file exists: `ls -lh data/beach_sensors.csv`
3. Check file permissions: `chmod 644 data/beach_sensors.csv`
4. If file is corrupted, re-download: `./download_data.sh`

### Problem: Memory error when loading
**Symptoms:** `MemoryError` when reading CSV

**Solutions:**
1. Load in chunks: `pd.read_csv('data/beach_sensors.csv', chunksize=10000)`
2. Specify data types to reduce memory: `pd.read_csv(..., dtype={'col1': 'int32', 'col2': 'float32'})`
3. Load only needed columns: `pd.read_csv(..., usecols=['col1', 'col2'])`

---

## Datetime Parsing Issues

### Problem: `pd.to_datetime()` fails
**Symptoms:** `ParserError` or incorrect dates

**Solutions:**
1. Check datetime format: `df['datetime'].head()` to see format
2. Specify format explicitly: `pd.to_datetime(df['datetime'], format='%Y-%m-%d %H:%M:%S')`
3. Handle errors: `pd.to_datetime(..., errors='coerce')` to convert invalid dates to NaT
4. Check for timezone issues: `pd.to_datetime(..., utc=True)` if needed

### Problem: Datetime index not working
**Symptoms:** Can't use datetime index for resampling or time-based operations

**Solutions:**
1. Verify index is datetime: `df.index.dtype` should show `datetime64[ns]`
2. Sort index: `df = df.sort_index()` after setting datetime index
3. Check for duplicate index values: `df.index.duplicated().sum()` should be 0

---

## Artifact Verification

### Problem: Artifact file not found by tests
**Symptoms:** Test fails with "file not found" error

**Solutions:**
1. Check file exists: `ls -lh output/qX_filename.ext`
2. Verify exact filename matches requirement (case-sensitive, no extra spaces)
3. Check file is in `output/` directory, not root directory
4. Verify file has content: `wc -l output/qX_filename.ext` (should be > 0 for text files)

### Problem: CSV format issues
**Symptoms:** Test fails on CSV validation

**Solutions:**
1. Check required columns exist: `pd.read_csv('output/qX_file.csv').columns`
2. Verify column names match exactly (case-sensitive, no extra spaces)
3. Check no index column saved: Use `index=False` when saving: `df.to_csv(..., index=False)`
4. Verify data types: `pd.read_csv('output/qX_file.csv').dtypes`
5. Check for extra columns: Compare `df.columns` to required columns list

### Problem: Datetime format in saved files
**Symptoms:** Datetime column not recognized when loading

**Solutions:**
1. When saving with datetime index: Use `df.reset_index().to_csv(..., index=False)` to ensure datetime becomes a column
2. Verify datetime column is saved: `pd.read_csv('output/qX_file.csv', parse_dates=['Measurement Timestamp'])`
3. Check datetime format is preserved: Datetime should be readable string format in CSV

### Problem: Text file format issues
**Symptoms:** Test fails on text file validation

**Solutions:**
1. Check file encoding: Should be UTF-8
2. Verify no extra whitespace: `cat output/qX_file.txt | head -5`
3. For single-number files (e.g., `q2_rows_cleaned.txt`): Ensure only the number, no labels or whitespace
4. Check line endings: Use standard line endings (LF on Unix, CRLF on Windows)

## Missing Data Issues

### Problem: Too much missing data after cleaning
**Symptoms:** Dataset becomes very small after dropping missing values

**Solutions:**
1. Consider imputation instead of dropping: `df.fillna(df.mean())` or `df.fillna(method='ffill')`
2. Drop only rows with missing values in critical columns
3. Use forward-fill for time series: `df.fillna(method='ffill')`
4. Document your strategy in `q2_cleaning_report.txt`

### Problem: Missing data in temporal features
**Symptoms:** Can't extract hour/day_of_week because datetime is missing

**Solutions:**
1. Handle missing datetime first: drop or impute datetime column before extracting features
2. Check datetime parsing worked: `df['datetime'].isnull().sum()`
3. Use `errors='coerce'` in `pd.to_datetime()` to handle invalid dates

---

## Feature Engineering Issues

### Problem: Infinity values in features
**Symptoms:** Model training fails with "Input X contains infinity" error

**Solutions:**
1. Check for infinity: `np.isinf(df.select_dtypes(include=[np.number])).sum()`
2. Replace infinity: `df = df.replace([np.inf, -np.inf], np.nan)`
3. Handle NaN after replacement: Fill or drop as appropriate
4. Check division operations: Ratios can produce infinity if denominator is zero
5. Add small constant to denominators: `df['ratio'] = df['A'] / (df['B'] + 0.1)`

### Problem: Rolling windows causing data leakage
**Symptoms:** Suspiciously perfect model performance (R² > 0.99)

**Solutions:**
1. Check which variables you're creating rolling windows for
2. **Never create rolling windows of your target variable** - this causes data leakage
3. Only create rolling windows of predictor variables (e.g., Wind Speed, Humidity, not Air Temperature if that's your target)
4. See Q6 and Q7 for detailed warnings about data leakage
5. Check feature-target correlations: If any feature has correlation > 0.99 with target, investigate

### Problem: Rolling window calculation fails
**Symptoms:** `df.rolling()` gives errors or NaN values

**Solutions:**
1. Ensure datetime index is set: `df = df.set_index('datetime')`
2. Sort index first: `df = df.sort_index()`
3. Check for sufficient data: rolling windows need enough data points
4. Handle NaN in results: `df.rolling(7).mean().fillna(method='bfill')`

### Problem: Temporal features extraction fails
**Symptoms:** Can't extract hour, day_of_week, month

**Solutions:**
1. Verify datetime index: `df.index.dtype` should be datetime
2. Use `.dt` accessor: `df['hour'] = df.index.hour` (if index) or `df['datetime'].dt.hour` (if column)
3. Check datetime is parsed: `df['datetime'].dtype` should be `datetime64[ns]`

---

## Train/Test Split Issues

### Problem: Temporal split not working
**Symptoms:** Can't split by date or getting wrong split

**Solutions:**
1. Ensure datetime index is set and sorted: `df = df.set_index('datetime').sort_index()`
2. Use date-based splitting: `split_date = df.index.max() - pd.Timedelta(days=30)`
3. Verify split: `train = df[df.index < split_date]` and `test = df[df.index >= split_date]`
4. Check no data leakage: `train.index.max() < test.index.min()` should be True

### Problem: Features and target have different shapes
**Symptoms:** Can't train model, shape mismatch

**Solutions:**
1. Ensure same index: `X_train.index.equals(y_train.index)`
2. Drop index before saving: `X_train.reset_index(drop=True).to_csv(...)`
3. Reload with same index: `X_train = pd.read_csv(..., index_col=0)`

---

## Modeling Issues

### Problem: Model training fails
**Symptoms:** `ValueError` or `TypeError` when fitting model

**Solutions:**
1. Check for NaN in features: `X_train.isnull().sum()` - handle missing values
2. Check data types: `X_train.dtypes` - convert object types to numeric
3. Remove infinite values: `X_train = X_train.replace([np.inf, -np.inf], np.nan)`
4. Ensure features are numeric: `X_train = X_train.select_dtypes(include=[np.number])`

### Problem: Poor model performance
**Symptoms:** Very low R² or high RMSE

**Solutions:**
1. Check for data leakage: ensure no future data in training set
2. Feature scaling: `from sklearn.preprocessing import StandardScaler`
3. Try different models: start simple (Linear Regression), then try Random Forest
4. Check target variable distribution: might need transformation

### Problem: Overfitting
**Symptoms:** Train R² much higher than test R²

**Solutions:**
1. This is expected for complex models - document it
2. Compare train vs test metrics in your report
3. Consider simpler models if overfitting is severe
4. Note this in your limitations section

---

## Visualization Issues

### Problem: Plots not saving
**Symptoms:** `plt.savefig()` creates empty file

**Solutions:**
1. Save before showing: `plt.savefig('output/plot.png')` then `plt.show()`
2. Clear figure after saving: `plt.clf()` or `plt.close()`
3. Check file path exists: `os.makedirs('output', exist_ok=True)`
4. Use absolute path if needed: `plt.savefig(os.path.join('output', 'plot.png'))`

### Problem: Time series plot not showing correctly
**Symptoms:** X-axis dates are wrong or unreadable

**Solutions:**
1. Use datetime index for x-axis: `df.plot(x=df.index, y='column')`
2. Format dates: `import matplotlib.dates as mdates` then `ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))`
3. Rotate labels: `plt.xticks(rotation=45)`

---

## Output File Issues

### Problem: Output files not in correct format
**Symptoms:** Auto-grading fails on file format

**Solutions:**
1. Check column names match requirements exactly
2. Verify data types: `df.dtypes` before saving
3. Save without index if required: `df.to_csv(..., index=False)`
4. Check file encoding: `df.to_csv(..., encoding='utf-8')`

### Problem: Text files have wrong format
**Symptoms:** `q2_rows_cleaned.txt` or similar files fail validation

**Solutions:**
1. Save single number: `with open('output/q2_rows_cleaned.txt', 'w') as f: f.write(str(len(df)))`
2. No extra whitespace: `f.write(str(value).strip())`
3. Check file contents: `cat output/q2_rows_cleaned.txt` to verify

---

## General Debugging Tips

### Check Your Progress
1. After each phase, verify required artifacts exist: `ls -lh output/`
2. Test loading your saved files: `pd.read_csv('output/q2_cleaned_data.csv')`
3. Verify data shapes make sense: `df.shape` should be reasonable
4. Check for obvious errors: `df.describe()` to see if values are reasonable

### Common Mistakes to Avoid
1. **Random train/test split** - Use temporal split for time series!
2. **Data leakage** - Don't use future data to predict past, and don't create rolling windows of your target variable
3. **Forgetting to save artifacts** - Check all required files exist
4. **Not documenting decisions** - Explain your choices in cleaning report
5. **Skipping checkpoints** - Use the checklists to verify completion
6. **Infinity values** - Check for infinity after feature engineering (especially ratios) and handle appropriately

### Getting Help
1. Review Lecture 11 notebooks for similar operations
2. Consult the textbook or online documentation
3. Review error messages carefully - they often tell you what's wrong
4. Contact instructor and EAs for assistance (clarification and nudges)

---

## Phase-Specific Tips

### Q1-Q2 (Setup, Exploration, Cleaning)
- Start simple: load data, look at it, understand structure
- Document everything you find - you'll need it for the writeup
- Don't over-clean: keep some data for analysis

### Q3-Q4 (Wrangling, Feature Engineering)
- Datetime parsing is critical - get this right before moving on
- Extract temporal features early - you'll need them throughout
- Rolling windows need sorted datetime index

### Q5-Q6 (Pattern Analysis, Modeling Prep)
- Temporal split is CRITICAL - don't use random split
- Visualize before modeling - understand your data
- Choose target variable that makes sense

### Q7-Q8 (Modeling, Results)
- Start with simple models - Linear Regression is fine
- Compare train vs test performance
- Document everything for the writeup

### Q9 (Writeup)
- Reference your artifacts - they contain your findings
- Explain your decisions - why did you choose that approach?
- Be honest about limitations - this shows critical thinking

---

Good luck! Remember: the lecture notebooks are your best reference.

