---
notion:
  title_line: "# DLC: Advanced Data Aggregation Topics"
  role: bonus
  status: mapped
  page_id: "3d2d9fdd-1a1a-81b9-9e57-e4de7f2b74da"
  url: "https://app.notion.com/p/3d2d9fdd1a1a81b99e57e4de7f2b74da"
---

# DLC: Advanced Data Aggregation Topics

# Advanced GroupBy Operations

Custom aggregations answer questions beyond built-in means and counts, such as how spread out a clinic's waits are. The function receives one group's selected column as a Series and must return one value; `apply` is the choice when the answer needs several rows.

| Waits at one clinic (minutes) | Calculation | One returned value |
| --- | --- | --- |
| 10, 12, 18 | range: maximum minus minimum | 8 |
| 10, 12, 18 | interquartile range: 75th minus 25th percentile | 4 |

## Custom Aggregation Functions

### Code Snippet: Your Own Function in Named Aggregation

```python
import pandas as pd

visits = pd.DataFrame({
    'clinic': ['Mission', 'Mission', 'Mission', 'Sunset', 'Sunset', 'Sunset', 'Sunset'],
    'wait_min': [10, 12, 18, 5, 9, 14, 40],
})

# A lambda in a (column, function) pair returns one value per group, like 'mean'.
visits.groupby('clinic').agg(
    wait_range=('wait_min', lambda s: s.max() - s.min()),
    wait_iqr=('wait_min', lambda s: s.quantile(0.75) - s.quantile(0.25)),
)
```

```text
         wait_range  wait_iqr
clinic                       
Mission           8       4.0
Sunset           35      12.5
```

## Lambda Functions in GroupBy

A dictionary maps each column to its own function. A list of lambdas gives every result column the name `<lambda_0>`, `<lambda_1>`, and so on; named aggregation, as above, gives readable names instead.

### Code Snippet: Lambda Aggregations

```python
import pandas as pd

visits = pd.DataFrame({
    'clinic': ['Mission'] * 4 + ['Sunset'] * 4,
    'wait_min': [10, 12, 18, 35, 5, 9, 14, 40],
    'provider_id': ['D01', 'D02', 'D01', 'D03', 'D07', 'D07', 'D08', 'D07'],
    'sbp': [128, 142, 151, 119, 135, 160, 122, 148],
})

# One function per column: 95th-percentile wait, providers seen, visits with SBP >= 140
display(visits.groupby('clinic').agg({
    'wait_min': lambda x: x.quantile(0.95),
    'provider_id': lambda x: x.nunique(),
    'sbp': lambda x: (x >= 140).sum(),
}))

# Several lambdas on one column
display(visits.groupby('clinic').agg({
    'wait_min': [
        lambda x: x.mean(),
        lambda x: x.std(),
        lambda x: x.quantile(0.25),
        lambda x: x.quantile(0.75),
    ]
}).round(1))
```

```text
         wait_min  provider_id  sbp
clinic                             
Mission     32.45            3    2
Sunset      36.10            2    2
          wait_min                                 
        <lambda_0> <lambda_1> <lambda_2> <lambda_3>
clinic                                             
Mission       18.8       11.4       11.5       22.2
Sunset        17.0       15.8        8.0       20.5
```

## GroupBy with Time Windows

`pd.Grouper` builds a grouping key from dates: `ME` collects rows into calendar months and labels each group by its month-end date. The timestamps must first be datetime values (`pd.to_datetime`); setting the date index tells the grouper which timestamps to use. A seven-day window (`7D`) is a fixed duration, unlike a calendar month, so it suits weekly surveillance counts.

### Code Snippet: Grouping by Time Windows

Daily emergency department (ED) visits for influenza-like illness (ILI), January to June, falling as flu season ends:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
days = pd.date_range('2025-01-01', periods=181, freq='D')
ed = pd.DataFrame({
    'date': days.strftime('%Y-%m-%d'),  # text dates, as read from a CSV
    'ili_visits': rng.poisson(np.linspace(60, 15, 181)),
})

ed['date'] = pd.to_datetime(ed['date'])
ed = ed.set_index('date')

monthly = ed.groupby(pd.Grouper(freq='ME'))[['ili_visits']].sum()   # month-end groups
display(ed.groupby(pd.Grouper(freq='QE'))[['ili_visits']].mean().round(1))  # quarter-end groups
peak = ed.groupby(pd.Grouper(freq='YE'))[['ili_visits']].max()      # year-end groups
display(ed.groupby(pd.Grouper(freq='7D')).agg({'ili_visits': ['sum', 'mean', 'count']}).head(3))
```

```text
            ili_visits
date                  
2025-03-31        48.5
2025-06-30        26.1
           ili_visits                 
                  sum       mean count
date                                  
2025-01-01        426  60.857143     7
2025-01-08        410  58.571429     7
2025-01-15        380  54.285714     7
```

# Advanced Pivot Table Operations

Multiple row or column keys give each cell a more specific meaning. With region and department down the rows and quarter and year across the columns, one cell describes exactly one combination of region, department, quarter, and year. Select a level before comparing values with different units, such as total charges and mean wait.

## Multi-Level Pivot Tables

### Code Snippet: Multi-Level Pivot Tables

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
visits = pd.DataFrame({
    'region': rng.choice(['North', 'South'], size=16),
    'department': rng.choice(['Cardiology', 'Pediatrics'], size=16),
    'quarter': rng.choice(['Q1', 'Q2'], size=16),
    'year': rng.choice([2024, 2025], size=16),
    'charges': rng.integers(150, 900, size=16),
    'wait_min': rng.normal(20, 6, size=16).round(1),
})

# No fill_value: wait_min is a mean, and an absent cell is not a wait of 0.
pivot = pd.pivot_table(visits,
                       values=['charges', 'wait_min'],
                       index=['region', 'department'],
                       columns=['quarter', 'year'],
                       aggfunc={'charges': 'sum', 'wait_min': 'mean'},
                       margins=True)

# Flatten the column levels; str() handles integer years, and rstrip('_')
# tidies the margin columns, whose lower levels are empty.
pivot.columns = ['_'.join(map(str, col)).rstrip('_') for col in pivot.columns]
print(pivot.shape)
print(list(pivot.columns[:5]))
print(pivot['wait_min_Q1_2024'].isna().sum(), 'absent cells in wait_min_Q1_2024')
```

```text
(5, 10)
['charges_Q1_2024', 'charges_Q1_2025', 'charges_Q2_2024', 'charges_Q2_2025', 'charges_All']
2 absent cells in wait_min_Q1_2024
```

## Pivot Table with Custom Functions

The **weighted mean** is `sum(value × weight) / sum(weight)`: a value with twice the weight contributes twice as much. Combining monthly mean waits needs it: months with 100 and 300 visits and mean waits of 20 and 30 minutes average 27.5 minutes per visit, not 25. This example assumes complete, finite values and weights; missing measurements need an explicit policy before either sum is calculated.

`pivot_table(values='mean_wait')` passes its aggregator only the `mean_wait` column, never the matching `visits`, so no `aggfunc` can weight it. Compute the weighted parts first, sum them by group, then reshape.

### Code Snippet: Weighted-Mean Pivot Workaround

One row per clinic, department, and month, as a scheduling system exports it:

```python
import pandas as pd

monthly = pd.DataFrame({
    'department': ['Cardiology'] * 4 + ['Pediatrics'] * 4,
    'clinic': ['Mission', 'Mission', 'Sunset', 'Sunset'] * 2,
    'mean_wait': [20.0, 30.0, 18.0, 24.0, 12.0, 15.0, 10.0, 16.0],
    'visits': [100, 300, 50, 50, 200, 200, 80, 20],
})

if (monthly['visits'] < 0).any():
    raise ValueError('visit counts must be nonnegative')

monthly = monthly.assign(wait_x_visits=monthly['mean_wait'] * monthly['visits'])
sums = monthly.groupby(['department', 'clinic'])[['wait_x_visits', 'visits']].sum()
visit_totals = sums['visits'].unstack('clinic')
if visit_totals.eq(0).any().any():
    raise ValueError('each observed group must have at least one visit')

sums['wait_x_visits'].unstack('clinic').div(visit_totals)
```

```text
clinic      Mission  Sunset
department                 
Cardiology     27.5    21.0
Pediatrics     13.5    11.2
```

The plain mean for Mission Cardiology would be 25.0; the 300-visit month pulls the weighted mean to 27.5.

## Pivot Table with Missing Data Handling

### Reference Card: Missing Cells in Pivot Tables

- **Missing mean:** `NaN` can mean no rows had that combination, or that all its measurements were missing. Leave it `NaN`; neither means a measured zero.
- **Row count:** `pd.crosstab(df['department'], df['clinic'])` counts rows, while a pivot with `aggfunc='count'` counts non-missing measurements. For one visit with a missing wait, these counts are 1 and 0, and the mean is `NaN`.
- **Zero fill:** use `fill_value=0` only when zero is the correct answer, such as a count of visits for an absent combination. Compare row and measurement counts beside a mean to distinguish no visits from visits with no recorded measurements.

In pandas 3, categorical groupers default to `observed=True`. Use `observed=False` only when a table must include every defined category or category combination, such as every department on a service-line report. In a mean table, an unused category is an all-`NaN` row, which the default `dropna=True` removes, so also pass `dropna=False`. That also keeps rows with a missing key as a `NaN` row.

### Code Snippet: An Unused Department in Mean and Count Tables

```python
import pandas as pd

visits = pd.DataFrame({
    'department': pd.Categorical(['Cardiology', 'Cardiology', 'Pediatrics', 'Pediatrics'],
                                 categories=['Cardiology', 'Pediatrics', 'Dermatology']),
    'clinic': ['Mission', 'Sunset', 'Mission', 'Mission'],
    'wait_min': [10.0, 12.0, 8.0, 6.0],
})

# Mean: absent cells stay NaN; Dermatology has no rows, so its whole row is NaN
means = pd.pivot_table(visits, values='wait_min', index='department', columns='clinic',
                       aggfunc='mean', observed=False, dropna=False)

# Count: an absent combination really is 0 visits
counts = pd.pivot_table(visits, values='wait_min', index='department', columns='clinic',
                        aggfunc='count', fill_value=0, observed=False)
display(means)
display(counts)
```

```text
clinic       Mission  Sunset
department                  
Cardiology      10.0    12.0
Pediatrics       7.0     NaN
Dermatology      NaN     NaN
clinic       Mission  Sunset
department                  
Cardiology         1       1
Pediatrics         2       0
Dermatology        0       0
```

Forward-filling or interpolating fabricates values the same way: `means.ffill()` copies Cardiology's Sunset mean (12.0) into Pediatrics and Dermatology, which have no Sunset visits. To keep only complete rows, use `means.dropna()` (here, only Cardiology).

### Group Coverage Edge Cases

- `pd.pivot_table(..., dropna=True)`, the default, drops rows with a missing key and also any row or column whose cells all came out `NaN`.
- Counting an unused category gives `0`, so its row stays; averaging one gives `NaN` in every cell, so the row disappears unless you pass `dropna=False`.
- The kept `NaN` row is not always last: with a categorical key, `observed=False`, and a `columns=` key, `pivot_table` puts it first, so find it by its label.
- For plain text keys, `observed` does nothing.
- `pd.Categorical(values, categories=[...])` turns a value missing from `categories=` into `NaN`, with a warning that a future pandas will raise an error instead, so list every value that occurs.

# Hierarchical Grouping and MultiIndex

A **MultiIndex** stores several labels for each row, such as clinic and department. Grouping by one level combines rows sharing that label while ignoring the others: Mission Cardiology 412 and Mission Pediatrics 655 become Mission 1,067. `swaplevel` changes label order; `sort_index` changes row order; neither aggregates values.

## MultiIndex Operations

### Code Snippet: MultiIndex Operations

```python
import pandas as pd

counts = pd.DataFrame({
    'clinic': ['Mission', 'Mission', 'Sunset', 'Sunset'],
    'department': ['Cardiology', 'Pediatrics', 'Cardiology', 'Pediatrics'],
    'visits': [412, 655, 198, 730],
}).set_index(['clinic', 'department'])

display(counts.groupby(level=0)[['visits']].sum())              # first level, by position
by_dept = counts.groupby(level='department')[['visits']].mean()  # a level, by name
both = counts.groupby(level=[0, 1])[['visits']].max()            # both levels

display(counts.swaplevel(0, 1).sort_index())  # department first, then sorted
print(counts.loc[('Mission', 'Cardiology'), 'visits'])  # one combination
display(counts.xs('Cardiology', level='department'))    # cross-section of one department
```

```text
         visits
clinic         
Mission    1067
Sunset      928
                    visits
department clinic         
Cardiology Mission     412
           Sunset      198
Pediatrics Mission     655
           Sunset      730
412
         visits
clinic         
Mission     412
Sunset      198
```

## Advanced MultiIndex Grouping

A reusable function can summarize three levels at once, applying a different function to each kind of column: numbers get mean, spread, and count; a category gets its most common value; dates get their first and last.

### Code Snippet: Advanced MultiIndex Grouping

```python
import pandas as pd

def hierarchical_summary(visits):
    """Summarize visits by region, clinic, and department."""
    result = visits.groupby(['region', 'clinic', 'department']).agg({
        'wait_min': ['mean', 'std', 'count'],
        'visit_type': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
        'visit_date': ['min', 'max'],
    })
    # Flatten ('wait_min', 'mean') to 'wait_min_mean'
    result.columns = ['_'.join(col) for col in result.columns]
    return result

visits = pd.DataFrame({
    'region': ['West'] * 5 + ['South'] * 3,
    'clinic': ['Sunset', 'Sunset', 'Sunset', 'Richmond', 'Richmond', 'Bayview', 'Bayview', 'Bayview'],
    'department': ['Cardiology'] * 3 + ['Pediatrics'] * 2 + ['Cardiology'] * 3,
    'wait_min': [22, 31, 18, 12, 9, 40, 35, 28],
    'visit_type': ['follow-up', 'follow-up', 'new', 'well-child', 'sick', 'new', 'follow-up', 'new'],
    'visit_date': pd.to_datetime(['2025-01-06', '2025-02-11', '2025-03-03', '2025-01-15',
                                  '2025-01-29', '2025-02-04', '2025-02-18', '2025-03-24']),
})
hierarchical_summary(visits).round({'wait_min_mean': 1, 'wait_min_std': 1})
```

```text
                            wait_min_mean  wait_min_std  wait_min_count visit_type_<lambda> visit_date_min visit_date_max
region clinic   department                                                                                               
South  Bayview  Cardiology           34.3           6.0               3                 new     2025-02-04     2025-03-24
West   Richmond Pediatrics           10.5           2.1               2                sick     2025-01-15     2025-01-29
       Sunset   Cardiology           23.7           6.7               3           follow-up     2025-01-06     2025-03-03
```

For Richmond Pediatrics, `well-child` and `sick` tie; `mode()` returns them sorted, so `iloc[0]` picks `sick`.

## Summary Statistics by Level

`groupby(level=...)` takes a level name as well as a position, and columns with levels can be grouped by transposing first.

### Reference Card: Summary Statistics by Level

- `df.groupby(level='level_name').sum()`: Aggregate by level name instead of position.
- `df.T.groupby(level='col_level').sum().T`: Aggregate by a column level; transpose so it becomes a row level, then transpose back.

### Code Snippet: Aggregate by Level

```python
import pandas as pd

quarterly = pd.DataFrame(
    [[420, 31, 18], [455, 27, 22], [610, 52, 35], [640, 44, 41]],
    index=[['West', 'West', 'East', 'East'], ['Q1', 'Q2', 'Q1', 'Q2']],
    columns=['visits', 'no_shows', 'referrals'],
)
quarterly.index.names = ['region', 'quarter']

display(quarterly.groupby(level='region').sum())
display(quarterly.groupby(level='quarter').mean())
```

```text
        visits  no_shows  referrals
region                             
East      1250        96         76
West       875        58         40
         visits  no_shows  referrals
quarter                             
Q1        515.0      41.5       26.5
Q2        547.5      35.5       31.5
```

Columns can have levels too, such as orders by service and status. Transpose so the column level becomes a row level while grouping:

```python
import pandas as pd

orders = pd.DataFrame(
    [[120, 112, 40, 31], [95, 90, 22, 22], [210, 188, 65, 50]],
    index=['Mission', 'Sunset', 'Bayview'],
    columns=[['labs', 'labs', 'imaging', 'imaging'],
             ['ordered', 'resulted', 'ordered', 'resulted']],
)
orders.columns.names = ['service', 'status']
orders.T.groupby(level='status').sum().T
```

```text
status   ordered  resulted
Mission      160       143
Sunset       117       112
Bayview      275       238
```

# Reshaping Grouped Results with stack() and unstack()

- The lecture's two-key results have a `MultiIndex`; `unstack()` moves the inner row level into columns and `stack()` moves columns back into rows.
- In pandas 3, `stack()` keeps missing combinations as `NaN` rows; dropping or zero-filling them is a separate choice.

## Preserving Missing Combinations

### Reference Card: `stack()` and `unstack()` Missing-Value Behavior

- `df.stack()`: Move columns into an inner row level, preserving missing combinations; the former `dropna=` argument is no longer accepted.
- `df.stack().dropna()`: Keep only observed values after reshaping.
- `series.unstack()`: Rebuild the table; combinations absent from the Series index come back as `NaN`.
- `series.unstack(fill_value=0)`: Fill those absent combinations with 0 instead; correct only when 0 is the true value.

### Code Snippet: Preserve vs. Drop Missed Questionnaires

PHQ-9 depression scores (0 to 27) at weeks 0, 4, and 8; `NaN` is a missed questionnaire:

```python
import numpy as np
import pandas as pd

phq9 = pd.DataFrame({
    'week_0': [14, 18, np.nan, 9],
    'week_4': [11, np.nan, 16, 7],
    'week_8': [np.nan, 10, 12, np.nan],
}, index=['P001', 'P002', 'P003', 'P004'])

stacked_all = phq9.stack()
print(len(stacked_all))          # every patient-week
print(stacked_all.isna().sum())  # missed questionnaires

# Drop missed questionnaires only when the question calls for observed scores.
stacked_observed = stacked_all.dropna()
print(len(stacked_observed))

# A PHQ-9 of 0 means no symptoms, so fill_value=0 would invent four symptom-free visits.
display(stacked_observed.unstack())
```

```text
12
4
8
      week_0  week_4  week_8
P001    14.0    11.0     NaN
P002    18.0     NaN    10.0
P003     NaN    16.0    12.0
P004     9.0     7.0     NaN
```

Keeping the full result lets a later analysis distinguish a recorded missing value from a combination removed from the table.

# Advanced Statistical Aggregations

Grouped statistics keep populations separate. A percentile describes the distribution inside one group; a rolling statistic instead depends on the sequence of rows inside it.

## Rolling Statistics

A **rolling window** uses the latest fixed number of rows; an **expanding window** uses every row so far. For values 10, 20, 30, a two-row rolling mean is `NaN, 15, 25`, while the expanding mean is `10, 15, 20`. Sort by the relevant time column before using either on dated observations.

### Code Snippet: Rolling and Expanding Statistics

Daily home weights for two heart-failure patients; a gain of more than 2 kg over a few days can signal fluid buildup:

```python
import pandas as pd

weights = pd.DataFrame({
    'patient_id': ['P01'] * 5 + ['P02'] * 5,
    'day': [1, 2, 3, 4, 5] * 2,
    'weight_kg': [82.0, 82.4, 83.1, 84.0, 84.6, 70.2, 70.0, 70.3, 69.9, 70.1],
}).sort_values(['patient_id', 'day'])

by_patient = weights.groupby('patient_id')['weight_kg']
weights['avg_3d'] = by_patient.transform(lambda s: s.rolling(window=3).mean())
weights['sd_3d'] = by_patient.transform(lambda s: s.rolling(window=3).std())
weights['lowest_so_far'] = by_patient.transform(lambda s: s.expanding().min())
weights['gain_kg'] = weights['weight_kg'] - weights['lowest_so_far']
weights.round(2)
```

```text
  patient_id  day  weight_kg  avg_3d  sd_3d  lowest_so_far  gain_kg
0        P01    1       82.0     NaN    NaN           82.0      0.0
1        P01    2       82.4     NaN    NaN           82.0      0.4
2        P01    3       83.1   82.50   0.56           82.0      1.1
3        P01    4       84.0   83.17   0.80           82.0      2.0
4        P01    5       84.6   83.90   0.75           82.0      2.6
5        P02    1       70.2     NaN    NaN           70.2      0.0
6        P02    2       70.0     NaN    NaN           70.0      0.0
7        P02    3       70.3   70.17   0.15           70.0      0.3
8        P02    4       69.9   70.07   0.21           69.9      0.0
9        P02    5       70.1   70.10   0.20           69.9      0.2
```

P01 has gained 2.6 kg since day 1; P02's weight stays within 0.4 kg.

## Time-Based Windows

Row-count windows assume evenly spaced rows. Glucose readings arrive at irregular times, so a window of `'7D'` (the last seven days, however many readings) matches the clinical question better. `groupby(...).rolling('7D', on='taken_at')` needs each group sorted by `taken_at`.

### Code Snippet: Seven-Day Mean Glucose per Patient

```python
import pandas as pd

glucose = pd.DataFrame({
    'patient_id': ['P01', 'P01', 'P01', 'P01', 'P02', 'P02', 'P02'],
    'taken_at': pd.to_datetime(['2025-03-01', '2025-03-02', '2025-03-08', '2025-03-15',
                                '2025-03-01', '2025-03-04', '2025-03-05']),
    'glucose': [182, 170, 150, 141, 96, 104, 99],
}).sort_values(['patient_id', 'taken_at'])

glucose.groupby('patient_id').rolling('7D', on='taken_at')['glucose'].mean().round(1)
```

```text
patient_id  taken_at  
P01         2025-03-01    182.0
            2025-03-02    176.0
            2025-03-08    160.0
            2025-03-15    141.0
P02         2025-03-01     96.0
            2025-03-04    100.0
            2025-03-05     99.7
Name: glucose, dtype: float64
```

On 2025-03-08, P01's window covers 03-02 and 03-08 (seven days back, excluding 03-01), so the mean is 160.0.

## Percentile Aggregations

A function that returns a Series gives several values per group; `apply` stacks them into a long Series, and `unstack()` turns that into one row per clinic.

### Code Snippet: Percentile Aggregations

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
visits = pd.DataFrame({
    'clinic': np.repeat(['Mission', 'Sunset', 'Bayview'], 300),
    'wait_min': np.concatenate([rng.gamma(4, 5, 300), rng.gamma(4, 4, 300),
                                rng.gamma(2, 12, 300)]).round(),
})

def wait_percentiles(series):
    """Wait-time percentiles a scheduling report would show."""
    return pd.Series({
        'p25': series.quantile(0.25),
        'p50': series.quantile(0.50),
        'p75': series.quantile(0.75),
        'p90': series.quantile(0.90),
        'p95': series.quantile(0.95),
        'p99': series.quantile(0.99),
    })

visits.groupby('clinic')['wait_min'].apply(wait_percentiles).unstack().round(1)
```

```text
          p25   p50   p75   p90   p95   p99
clinic                                     
Bayview  12.0  20.0  32.0  48.0  56.1  81.0
Mission  13.0  19.0  25.0  33.1  38.0  51.0
Sunset   10.0  14.0  19.2  26.0  31.0  35.0
```

Bayview's median is close to Mission's, but its 90th and 99th percentiles are far higher: a long tail of very long waits that a mean would hide.

## Statistical Tests in Groups

The normality test asks whether each group's measurements are compatible with a normal distribution, a common check before a t-test. Its p-value is evidence about that assumption, not proof that a group is normal; running many group tests also creates more chances for small p-values. The example reports missing results for groups below the test's minimum sample size.

This optional example requires SciPy, which is not part of Lecture 08's recorded core environment. Add it to the project with `uv add scipy` (in Colab, run `%pip install scipy` in a cell) before running the example.

### Code Snippet: Statistical Tests in Groups

Systolic blood pressure (SBP) by clinic; Ridgecrest, a small rural clinic, has only six readings:

```python
import numpy as np
import pandas as pd
from scipy import stats

rng = np.random.default_rng(42)
bp = pd.DataFrame({
    'clinic': ['Mission'] * 60 + ['Bayview'] * 60 + ['Ridgecrest'] * 6,
    'sbp': np.concatenate([rng.normal(128, 14, 60),
                           rng.gamma(3, 8, 60) + 105,   # right-skewed
                           rng.normal(135, 15, 6)]).round(),
})

def normality_summary(group):
    """Normality test plus shape statistics for one clinic."""
    values = group['sbp'].dropna()

    # scipy.stats.normaltest requires at least eight observations.
    if len(values) < 8:
        stat, p_value = np.nan, np.nan
    else:
        stat, p_value = stats.normaltest(values)

    return pd.Series({
        'n': len(values),
        'normality_stat': stat,
        'normality_p': p_value,
        'mean': values.mean(),
        'std': values.std(),
        'skewness': values.skew(),
        'kurtosis': values.kurt(),
    })

bp.groupby('clinic').apply(normality_summary).round(3)
```

```text
               n  normality_stat  normality_p     mean     std  skewness  kurtosis
clinic                                                                            
Bayview     60.0          13.577        0.001  127.333  13.100     1.119     1.220
Mission     60.0           0.065        0.968  128.933  10.997    -0.060    -0.033
Ridgecrest   6.0             NaN          NaN  145.500  11.292    -0.400    -1.426
```

# Advanced Pivot Table Features

The row key need not already be a column: `pd.cut` can supply intervals directly, such as age bands. `bins=5` would make five equal-width intervals; a list of edges makes the bands a clinical guideline uses. Extracting year and month instead produces calendar groups. These choices change what a row represents, while the aggregation still decides what each cell means.

## Pivot Table with Custom Index

### Code Snippet: Pivot Tables with a Custom Index

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
patients = pd.DataFrame({
    'age': rng.integers(18, 90, size=300),
    'sex': rng.choice(['F', 'M'], size=300),
    'region': rng.choice(['North', 'South'], size=300),
    'clinic': rng.choice(['Clinic 1', 'Clinic 2'], size=300),
})
patients['sbp'] = (105 + 0.45 * patients['age'] + rng.normal(0, 12, size=300)).round()

# Age bands as the row key, made on the fly
display(pd.pivot_table(patients,
                       values='sbp',
                       index=pd.cut(patients['age'], bins=[18, 40, 65, 90], right=False),
                       columns='sex',
                       aggfunc='mean').round(1))

# Two existing columns as a multi-level row key
display(pd.pivot_table(patients,
                       values='sbp',
                       index=['region', 'clinic'],
                       columns='sex',
                       aggfunc='count'))
```

```text
sex           F      M
age                   
[18, 40)  118.6  113.1
[40, 65)  128.8  127.4
[65, 90)  135.5  141.6
sex               F   M
region clinic          
North  Clinic 1  28  33
       Clinic 2  38  36
South  Clinic 1  45  42
       Clinic 2  38  40
```

## Pivot Table with Time Index

### Code Snippet: Pivot Tables with a Time Index

Monthly ED arrivals at two sites; Parkside opened in February 2025:

```python
import pandas as pd

arrivals = pd.DataFrame({
    'month_start': ['2024-11-01', '2024-12-01', '2025-01-01', '2025-02-01', '2025-03-01',
                    '2025-02-01', '2025-03-01'],
    'site': ['Mission'] * 5 + ['Parkside'] * 2,
    'arrivals': [2210, 2480, 2615, 2390, 2105, 310, 655],
})

arrivals['month_start'] = pd.to_datetime(arrivals['month_start'])
arrivals['year'] = arrivals['month_start'].dt.year
arrivals['month'] = arrivals['month_start'].dt.month

# fill_value=0 is right here: Parkside had no arrivals before it opened.
pd.pivot_table(arrivals,
               values='arrivals',
               index=['year', 'month'],
               columns='site',
               aggfunc='sum',
               fill_value=0)
```

```text
site        Mission  Parkside
year month                   
2024 11        2210         0
     12        2480         0
2025 1         2615         0
     2         2390       310
     3         2105       655
```

# Advanced GroupBy Transformations

Lags and rolling windows need each group's rows sorted by time; for prediction, a window may use only past rows.

## Ranking Within Groups

**Rank** is a value's position within its group. For values 10, 10, 30, the default ranks are 1.5, 1.5, 3 because tied positions are averaged; `dense` gives 1, 1, 2, `min` gives 1, 1, 3, and `max` gives 2, 2, 3. `ascending=False` puts the largest value first; `pct=True` expresses the rank relative to the group's size.

### Code Snippet: Ranking Within Groups

ED patients waiting at one moment, ranked by wait within each triage level (ESI 2 is more urgent than ESI 3):

```python
import pandas as pd

queue = pd.DataFrame({
    'patient_id': ['P01', 'P02', 'P03', 'P04', 'P05', 'P06', 'P07'],
    'esi_level': [2, 2, 2, 3, 3, 3, 3],
    'wait_min': [25, 40, 25, 90, 60, 90, 15],
})

by_level = queue.groupby('esi_level')['wait_min']
queue['longest_first'] = by_level.rank(ascending=False)
queue['pct'] = by_level.rank(pct=True).round(2)
queue['dense'] = by_level.rank(method='dense')
queue['min'] = by_level.rank(method='min')
queue['max'] = by_level.rank(method='max')
queue
```

```text
  patient_id  esi_level  wait_min  longest_first   pct  dense  min  max
0        P01          2        25            2.5  0.50    1.0  1.0  2.0
1        P02          2        40            1.0  1.00    2.0  3.0  3.0
2        P03          2        25            2.5  0.50    1.0  1.0  2.0
3        P04          3        90            1.5  0.88    3.0  3.0  4.0
4        P05          3        60            3.0  0.50    2.0  2.0  2.0
5        P06          3        90            1.5  0.88    3.0  3.0  4.0
6        P07          3        15            4.0  0.25    1.0  1.0  1.0
```

## Lag and Lead Operations

A **lag** brings an earlier row's value onto the current row; a **lead** brings a later one. Within values 10, 15, 20, `shift(1)` gives `NaN, 10, 15`, and `diff()` gives `NaN, 5, 5`. Sort each group's observations into time order first. A lead uses future information, so it cannot be a predictor available at the current time.

### Code Snippet: Lag and Lead Within Groups

Hemoglobin A1c (%) at successive diabetes visits:

```python
import pandas as pd

a1c = pd.DataFrame({
    'patient_id': ['P01', 'P01', 'P01', 'P02', 'P02', 'P02'],
    'visit_date': pd.to_datetime(['2025-01-10', '2025-04-12', '2025-07-15',
                                  '2025-02-03', '2025-05-06', '2025-08-08']),
    'a1c': [9.2, 8.1, 7.4, 6.8, 7.3, 7.9],
}).sort_values(['patient_id', 'visit_date'])

by_patient = a1c.groupby('patient_id')['a1c']
a1c['prev_a1c'] = by_patient.shift(1)
a1c['two_back'] = by_patient.shift(2)
a1c['next_a1c'] = by_patient.shift(-1)   # future value: an outcome, never a predictor
a1c['change'] = by_patient.diff().round(1)
a1c['pct_change'] = by_patient.pct_change().round(3)
a1c
```

```text
  patient_id visit_date  a1c  prev_a1c  two_back  next_a1c  change  pct_change
0        P01 2025-01-10  9.2       NaN       NaN       8.1     NaN         NaN
1        P01 2025-04-12  8.1       9.2       NaN       7.4    -1.1      -0.120
2        P01 2025-07-15  7.4       8.1       9.2       NaN    -0.7      -0.086
3        P02 2025-02-03  6.8       NaN       NaN       7.3     NaN         NaN
4        P02 2025-05-06  7.3       6.8       NaN       7.9     0.5       0.074
5        P02 2025-08-08  7.9       7.3       6.8       NaN     0.6       0.082
```

P01 improves at each visit; P02 rises 0.5 to 0.6 points per visit, the pattern a care manager would flag.

# Reusable Report Builders

A small class can keep a table and its grouping together so a monthly report reruns the same steps on new data. Each method delegates to pandas; it adds no new grouping behavior. Direct calls are simpler for a single analysis. Use the class example to trace which object is stored and which pandas method ultimately runs.

## A Clinic Scorecard Class

### Code Snippet: Clinic Scorecard Builder

```python
import pandas as pd

class ClinicScorecard:
    """Group visits once, then build scorecard pieces from the same split."""

    def __init__(self, visits, group_cols):
        self.visits = visits
        self.group_cols = group_cols
        self.grouped = visits.groupby(group_cols)

    def custom_agg(self, col, func):
        """One value per group from your own function."""
        return self.grouped[col].apply(func)

    def multiple_aggs(self, **named):
        """Several named summaries in one pass."""
        return self.grouped.agg(**named)

    def filter_groups(self, condition):
        """Rows of groups that meet a condition, such as enough visits to report."""
        return self.grouped.filter(condition)

    def transform_groups(self, col, func):
        """A group statistic on every row, such as the clinic mean beside each visit."""
        return self.grouped[col].transform(func)

visits = pd.DataFrame({
    'clinic': ['Mission', 'Mission', 'Mission', 'Sunset', 'Sunset', 'Ridgecrest'],
    'wait_min': [12, 30, 18, 9, 14, 45],
    'no_show': [0, 1, 0, 0, 0, 1],
})

card = ClinicScorecard(visits, ['clinic'])
display(card.multiple_aggs(visits=('wait_min', 'size'),
                           p90_wait=('wait_min', lambda s: s.quantile(0.9)),
                           no_show_rate=('no_show', 'mean')).round(2))
print(card.custom_agg('wait_min', lambda s: (s > 15).mean()).round(2).to_dict())
print(card.filter_groups(lambda g: len(g) >= 2)['clinic'].unique().tolist())
print(card.transform_groups('wait_min', 'mean').round(1).tolist())
```

```text
            visits  p90_wait  no_show_rate
clinic                                    
Mission          3      27.6          0.33
Ridgecrest       1      45.0          1.00
Sunset           2      13.5          0.00
{'Mission': 0.67, 'Ridgecrest': 1.0, 'Sunset': 0.0}
['Mission', 'Sunset']
[20.0, 20.0, 20.0, 11.5, 11.5, 45.0]
```

# Measure, Then Optimize

Measure first, with the tools from the lecture's Measure Before Optimizing card, and change only the step where the time or memory goes.

## Timing and Memory Tools

### Code Snippet: Measure Before and After

On `labs`, one million glucose results with a text `clinic` column:

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
labs = pd.DataFrame({'patient_id': rng.integers(0, 10_000, 1_000_000),
                     'clinic': rng.choice(['North', 'South', 'East', 'West'], 1_000_000),
                     'glucose': rng.normal(100, 15, 1_000_000).round(1)})
print(round(labs['clinic'].memory_usage(deep=True) / 1e6, 1))   # 53.5 (MB as text)
labs['clinic'] = labs['clinic'].astype('category')
print(round(labs['clinic'].memory_usage(deep=True) / 1e6, 1))   # 1.0 (MB as category)

by_patient = labs.groupby('patient_id')['glucose']
%timeit by_patient.agg('std')                                   # about 9 ms
%timeit by_patient.agg(lambda s: s.std())                       # about 450 ms: same values, ~50x slower
```

Colab has the `pyarrow` package installed, which stores text more compactly, so there the text column takes 12.5 MB instead of 53.5; the category takes 1.0 MB either way.

# Scaling Past Memory: Chunks and Processes

The lecture's first moves are to measure, compute several summaries in one `.agg()` call, prefer built-in aggregations, and store repeated text keys as `category`. When a file is still too large to load, such as a year of lab results exported from the hospital's laboratory system, or one CPU core is the bottleneck, two further options exist. Both add complexity, so time the complete operation before and after.

## Chunked Processing

`pd.read_csv(path, chunksize=n)` reads a file `n` rows at a time, so only one chunk is in memory at once. Summarize each chunk, then combine the partial summaries. The partial results must combine correctly: chunk sums add up to the total sum and chunk counts to the total count, but averaging chunk means gives the wrong mean unless you keep each chunk's sum and count. Chunking saves memory; it is not automatically faster.

### Code Snippet: Abnormal-Result Rate from a Lab Export in Chunks

```python
import numpy as np
import pandas as pd

# A stand-in export: 50,000 results with a 0/1 abnormal flag
rng = np.random.default_rng(42)
pd.DataFrame({
    'clinic': rng.choice(['Mission', 'Sunset', 'Bayview'], 50_000, p=[0.5, 0.3, 0.2]),
    'abnormal': rng.binomial(1, 0.12, 50_000),
}).to_csv('lab_results_2025.csv', index=False)

def chunked_sum_count(file_path, group_cols, agg_cols, chunk_size=10_000):
    """Sum and count groups from a CSV that is read in chunks."""
    results = []

    for chunk in pd.read_csv(file_path, chunksize=chunk_size):
        if not chunk.empty:
            results.append(chunk.groupby(group_cols)[agg_cols].agg(['sum', 'count']))

    if not results:
        raise ValueError("input file must contain at least one data row")

    levels = list(range(results[0].index.nlevels))
    return pd.concat(results).groupby(level=levels).sum()

totals = chunked_sum_count('lab_results_2025.csv', ['clinic'], ['abnormal'])
totals[('abnormal', 'rate')] = (totals[('abnormal', 'sum')] / totals[('abnormal', 'count')]).round(4)
totals
```

```text
        abnormal               
             sum  count    rate
clinic                         
Bayview     1186   9942  0.1193
Mission     3007  24814  0.1212
Sunset      1851  15244  0.1214
```

Five chunks of 10,000 rows each were read; the rate comes from the combined sum and count.

## Parallel Processing

`multiprocessing.Pool` runs a function on several CPU cores at once, each in a separate Python process. Parallel work adds process startup, copying data between processes, and merge costs, so more processes do not guarantee a faster result.

Each worker process must be able to find the worker function (`payer_totals` below). In a notebook, that works only with the `fork` start method, the default on Linux through Python 3.13 (including Colab). Python 3.14 and later, macOS, and Windows start workers another way (`forkserver` or `spawn`): a worker function defined in a notebook makes the workers fail and the cell can hang. There, put the worker function in a `.py` file and import it, and start the pool from a script under `if __name__ == "__main__":`.

### Code Snippet: Total Claims by Payer on Four Cores

```python
from multiprocessing import Pool

import numpy as np
import pandas as pd

def payer_totals(chunk):
    return chunk.groupby('payer')[['billed_usd']].sum()

def parallel_payer_totals(claims, n_processes=4):
    if n_processes < 1:
        raise ValueError("n_processes must be at least 1")
    if claims.empty:
        raise ValueError("claims must contain at least one row")

    chunk_size = max(1, len(claims) // n_processes)
    chunks = [claims.iloc[i:i + chunk_size]
              for i in range(0, len(claims), chunk_size)]

    with Pool(n_processes) as pool:
        results = pool.map(payer_totals, chunks)

    return pd.concat(results).groupby(level=0)[['billed_usd']].sum()

rng = np.random.default_rng(42)
claims = pd.DataFrame({
    'payer': rng.choice(['Medicare', 'Medicaid', 'Commercial', 'Self-pay'], 400_000,
                        p=[0.35, 0.20, 0.40, 0.05]),
    'billed_usd': rng.integers(80, 2_500, 400_000),
})
result = parallel_payer_totals(claims)
assert result.equals(payer_totals(claims))  # same answer as one process
result
```

```text
            billed_usd
payer                 
Commercial   206428131
Medicaid     103117170
Medicare     179965518
Self-pay      25811905
```

# Advanced Remote Computing

The core lecture introduces SSH, file transfer, and persistent terminal sessions. The tools below extend that workflow to Jupyter on a server and to distributed and cloud-managed data processing.

## Run Jupyter Through an SSH Tunnel

`jupyter lab` (Lecture 04) starts Jupyter as a small web server that your browser talks to. Programs like this listen on a numbered **port**; Jupyter uses 8888 by default. On a shared server, start Jupyter so it accepts connections only from `127.0.0.1`, also called **localhost** ("this same computer"), so nobody else on the network can reach it. An **SSH tunnel**, a private pipe inside your SSH connection, then carries anything sent to port 8888 on your laptop to port 8888 on the server: your browser works as if Jupyter were local, while the kernel, files, memory, and CPU are on the server. For patient data, that also keeps the data on the approved server instead of your laptop.

```text
laptop browser → localhost:8888 ══ SSH tunnel ══> server localhost:8888 → Jupyter
```

### Reference Card: Jupyter Through a Tunnel

- `jupyter lab --ip=127.0.0.1 --port=8888 --no-browser` (on the server, inside tmux): Start Jupyter for localhost only; it prints a URL ending in `?token=...`, a password generated for this session.
- `ssh -N -L 8888:127.0.0.1:8888 user@host` (in a second, local terminal): `-L laptop_port:server_address:server_port` builds the tunnel; `-N` opens no remote shell, so the terminal shows nothing while the tunnel is open.
- Paste the printed `http://127.0.0.1:8888/lab?token=...` URL into your laptop's browser.
- If the server's 8888 is taken (common on a shared server), Jupyter quietly moves to 8889 or higher. Read the port in the URL it prints and use that number as the last field of `-L`, such as `-L 8888:127.0.0.1:8889` for a URL on `:8889`; then open the URL with 8888 in place of the server's port.
- `Ctrl+C` in the tunnel terminal: Close the tunnel; Jupyter keeps running in tmux.
- If port 8888 is already in use on your laptop (for example, by a local Jupyter), use `-L 8889:127.0.0.1:8888` and change 8888 to 8889 in the URL.

### Code Snippet: Start Jupyter and Open the Tunnel

On the server, in a project whose `pyproject.toml` lists JupyterLab (`uv add jupyterlab` adds it), start a persistent session, then launch Jupyter inside it:

```bash
tmux new -s notebooks
source .venv/bin/activate
jupyter lab --ip=127.0.0.1 --port=8888 --no-browser
```

In a second, local terminal:

```bash
ssh -N -L 8888:127.0.0.1:8888 jdoe@analysis.example.org
```

## Distributed Computing

**Distributed computing** divides work into partitions that can run on several workers. Dask records the requested operations first; `.compute()` executes them and collects the summary as a pandas object. The collected result must still fit in memory, even if the input does not: a year of lab results may not fit, but its per-test summary of a few dozen rows does.

Dask is not part of Lecture 08's core environment. Add it to the project with `uv add "dask[dataframe]"` (in Colab, `%pip install "dask[dataframe]"`) before running this example.

### Code Snippet: Distributed Computing with Dask

```python
import dask.dataframe as dd

# A glob reads every monthly export as one partitioned table
labs = dd.read_csv('lab_results_2025_*.csv')

# Nothing runs until .compute()
summary = labs.groupby('test_name').agg({
    'result_value': ['sum', 'mean', 'count']
}).compute()

summary.to_csv('lab_summary_2025.csv')
```

## Cloud Computing

**S3** is cloud object storage: a bucket contains named objects such as CSV files, for example a claims extract a data warehouse drops each night. Reading an `s3://` path changes where bytes come from, not where pandas computes; the DataFrame still occupies the current computer's memory. Writing a result back to S3 sends that result to the named bucket.

pandas reads and writes `s3://` paths directly once the `s3fs` package is installed (`uv add s3fs`; in Colab, `%pip install s3fs`) and your AWS credentials are configured. Without `s3fs`, `pd.read_csv('s3://...')` raises `ImportError: Install s3fs to access S3`.

### Code Snippet: Cloud Computing with S3

```python
import pandas as pd

# Read from S3 (needs s3fs and AWS credentials)
claims = pd.read_csv('s3://clinic-warehouse/claims_2025.csv')

# Process data
by_payer = claims.groupby('payer')[['billed_usd']].sum()

# Save back to S3
by_payer.to_csv('s3://clinic-warehouse/reports/billed_by_payer_2025.csv')
```

# Debugging and Profiling Data Pipelines

The examples below follow one realistic job: a nightly build of per-patient features (visit counts, mean glucose, last A1c) from encounter and lab tables, which slowed from minutes to an hour as the tables grew. Under pandas 3, inferred text uses the `str` dtype, and **Copy-on-Write** makes every selection behave as a copy, so changing a selection never changes the original table. Profile first, treat dtype changes as reviewed data-contract decisions, and apply them through explicit returned objects rather than automatic guesses.

## Memory Profiling and Optimization

The memory examples need the `psutil` and `memory_profiler` packages (`uv add psutil memory_profiler`; in Colab, `%pip install psutil memory_profiler`).

### Code Snippet: Find the Columns That Use the Memory

```python
import numpy as np
import pandas as pd

def analyze_memory_usage(df):
    """Print each column's memory and dtype candidates worth reviewing."""
    usage = df.memory_usage(deep=True).drop('Index')
    total = usage.sum()
    print(f"Total: {total / 1e6:.1f} MB")
    for col, mem in usage.sort_values(ascending=False).items():
        print(f"  {col}: {mem / 1e6:.1f} MB ({mem / total:.0%})")

    print("Candidates to review:")
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_string_dtype(s) and s.nunique() / len(s) < 0.5:
            print(f"  {col}: repeated text ({s.nunique()} values); try category")
        elif s.dtype == 'int64':
            for small in ['int8', 'int16', 'int32']:
                info = np.iinfo(small)
                if info.min <= s.min() and s.max() <= info.max:
                    print(f"  {col}: fits {small}")
                    break
        elif s.dtype == 'float64' and np.allclose(s, s.astype('float32'), equal_nan=True):
            print(f"  {col}: float32 keeps these values")

def apply_reviewed_dtypes(df, dtype_map):
    """
    Return a DataFrame using caller-reviewed dtype conversions.

    Validate integer ranges, float precision, and category semantics before
    constructing dtype_map. This function deliberately does not infer them.
    """
    before = df.memory_usage(deep=True).sum()
    optimized = df.astype(dtype_map)
    after = optimized.memory_usage(deep=True).sum()
    print(f"{before / 1e6:.1f} MB -> {after / 1e6:.1f} MB ({1 - after / before:.0%} smaller)")
    return optimized

rng = np.random.default_rng(42)
n = 500_000
encounters = pd.DataFrame({
    'patient_id': rng.integers(100_000, 360_000, n),
    'age': rng.integers(0, 100, n),
    'clinic': rng.choice(['Mission', 'Sunset', 'Richmond', 'Bayview', 'Excelsior'], n),
    'sbp': rng.normal(128, 16, n).round(),
})
analyze_memory_usage(encounters)
encounters = apply_reviewed_dtypes(encounters, {
    'patient_id': 'int32', 'age': 'int8', 'clinic': 'category', 'sbp': 'float32',
})
```

```text
Total: 19.7 MB
  clinic: 7.7 MB (39%)
  patient_id: 4.0 MB (20%)
  age: 4.0 MB (20%)
  sbp: 4.0 MB (20%)
Candidates to review:
  patient_id: fits int32
  age: fits int8
  clinic: repeated text (5 values); try category
  sbp: float32 keeps these values
19.7 MB -> 5.0 MB (75% smaller)
```

`int8` holds ages up to 127, and SBP readings are whole numbers that `float32` stores exactly; a new feed with ages coded as 999 for unknown would break that contract, which is why a person reviews the map.

### Code Snippet: Line-by-Line Memory with `memory_profiler`

`@profile` prints each line's memory and the increase it caused when the function runs. For this 32 MB table, the `scaled` line added about 90 MiB on a test run: `numeric - numeric.mean()` builds a full-size temporary before the division builds the result, and both stay in memory while that line runs.

```python
import numpy as np
import pandas as pd
from memory_profiler import profile

@profile
def standardized_vitals(vitals):
    numeric = vitals.select_dtypes(include=[np.number])
    correlation = numeric.corr()
    scaled = (numeric - numeric.mean()) / numeric.std()   # the largest increment
    covariance = numeric.cov()
    return scaled.dot(correlation)

rng = np.random.default_rng(42)
vitals = pd.DataFrame(rng.normal(size=(1_000_000, 4)),
                      columns=['sbp', 'dbp', 'heart_rate', 'resp_rate'])
features = standardized_vitals(vitals)
```

## Advanced Debugging Patterns

The decorator below wraps a pipeline step to log its input shape, elapsed time, and memory change. `PipelineDebugger` records checkpoints and evaluates explicit Boolean validation rules. A failed rule raises an error; a saved checkpoint is a CSV snapshot, not an automatic resume mechanism. These examples write copies of data to the current folder, so practice with synthetic data and never point them at real patient data on a laptop.

### Code Snippet: Debug a Lab-Feed Pipeline

```python
import functools
import logging
import os
import time
from datetime import datetime

import pandas as pd
import psutil

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s - %(message)s',
    handlers=[logging.FileHandler('pipeline_debug.log'), logging.StreamHandler()],
)
logger = logging.getLogger('lab_feed')

def debug_pipeline_step(step_name):
    """Log a step's input and output shapes, time, and memory change."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(df, *args, **kwargs):
            logger.info(f"Starting {step_name}: input {getattr(df, 'shape', '')}")
            process = psutil.Process(os.getpid())
            start_mem = process.memory_info().rss / 1e6
            start = time.perf_counter()
            try:
                result = func(df, *args, **kwargs)
            except Exception:
                # The traceback, dtypes, and null counts show what the step received.
                logger.exception(f"{step_name} failed")
                if hasattr(df, 'dtypes'):
                    logger.error(f"Input dtypes: {df.dtypes.astype(str).to_dict()}")
                    logger.error(f"Input nulls: {df.isna().sum().to_dict()}")
                raise
            logger.info(f"{step_name} done in {time.perf_counter() - start:.2f} s, "
                        f"memory {process.memory_info().rss / 1e6 - start_mem:+.1f} MB, "
                        f"output {getattr(result, 'shape', '')}")
            return result
        return wrapper
    return decorator

class PipelineDebugger:
    """Checkpoint snapshots and named validation rules for one pipeline."""

    def __init__(self, pipeline_name):
        self.pipeline_name = pipeline_name
        self.checkpoints = {}

    def checkpoint(self, step_name, data, metadata=None):
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"checkpoint_{self.pipeline_name}_{step_name}_{stamp}.csv"
        data.to_csv(filename, index=False)
        self.checkpoints[step_name] = {'shape': data.shape, 'file': filename,
                                       'metadata': metadata or {}}
        logger.info(f"Checkpoint {step_name}: {data.shape} saved to {filename}")

    def validate_step(self, step_name, data, validations):
        failed = []
        for name, rule in validations.items():
            try:
                passed = bool(rule(data))   # a False result is a failure, too
            except Exception as e:
                passed = False
                logger.error(f"  {name}: error {e}")
            logger.info(f"  {name}: {'PASSED' if passed else 'FAILED'}")
            if not passed:
                failed.append(name)
        if failed:
            raise ValueError(f"{step_name} failed: {', '.join(failed)}")

debugger = PipelineDebugger('lab_feed')

@debug_pipeline_step('clean')
def clean_results(raw):
    clean = raw.dropna(subset=['patient_id', 'glucose'])
    debugger.validate_step('clean', clean, {
        'not_empty': lambda df: len(df) > 0,
        'glucose_plausible': lambda df: df['glucose'].between(10, 1000).all(),
        'one_result_per_draw': lambda df: not df.duplicated(['patient_id', 'collected_at']).any(),
    })
    debugger.checkpoint('clean', clean, {'dropped': len(raw) - len(clean)})
    return clean

raw = pd.DataFrame({
    'patient_id': ['P01', 'P02', 'P02', None, 'P03'],
    'collected_at': ['2025-03-01 07:10', '2025-03-01 07:25', '2025-03-01 07:25',
                     '2025-03-01 07:40', '2025-03-01 08:05'],
    'glucose': [96.0, 182.0, 182.0, 110.0, 88.0],
})
clean_results(raw)
```

The log shows `not_empty: PASSED`, `glucose_plausible: PASSED`, and `one_result_per_draw: FAILED`, then `ValueError: clean failed: one_result_per_draw`, the input dtypes, and the null counts: P02's 07:25 draw arrived twice from the lab interface. Adding `.drop_duplicates(['patient_id', 'collected_at'])` to the cleaning step makes all three rules pass and saves the checkpoint.

## Performance Profiling

**Profiling** records which function calls consume time. `cProfile` measures calls inside one run; the comparison helper repeats entire operations. Compare operations only after confirming that their outputs agree, and compare input sizes to distinguish a fixed startup cost from work that grows with each row.

### Code Snippet: Profile a Slow Per-Patient Feature Build

```python
import cProfile
import pstats
import time
from functools import wraps

import numpy as np
import pandas as pd

def profile_function(sort_by='cumulative', lines_to_show=8):
    """Print the slowest calls each time the function runs."""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            profiler = cProfile.Profile()
            profiler.enable()
            try:
                return func(*args, **kwargs)
            finally:
                profiler.disable()
                print(f"PROFILE: {func.__name__}")
                pstats.Stats(profiler).sort_stats(sort_by).print_stats(lines_to_show)
        return wrapper
    return decorator

def compare_operations(operations, data, iterations=3):
    """Time whole operations on the same data and report each against the fastest."""
    times = {}
    for name, operation in operations.items():
        runs = []
        for _ in range(iterations):
            start = time.perf_counter()
            operation(data)
            runs.append(time.perf_counter() - start)
        times[name] = np.mean(runs)
    fastest = min(times.values())
    for name, t in times.items():
        print(f"  {name}: {t:.3f} s ({t / fastest:.0f}x the fastest)")
    return times

def scaling(operation, sizes, make_data):
    """Time one operation at several input sizes."""
    for size in sizes:
        data = make_data(size)
        start = time.perf_counter()
        operation(data)
        elapsed = time.perf_counter() - start
        print(f"  {size:,} rows: {elapsed:.3f} s ({elapsed / size * 1e6:.2f} µs per row)")

def make_labs(n, seed=42):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({'patient_id': rng.integers(0, n // 20, n),
                         'glucose': rng.normal(105, 25, n).round()})

def features_loop(labs):
    """The slow version: one boolean filter of the whole table per patient."""
    rows = []
    for pid in labs['patient_id'].unique():
        g = labs.loc[labs['patient_id'] == pid, 'glucose']
        rows.append({'patient_id': pid, 'n_tests': len(g), 'mean_glucose': g.mean(),
                     'n_high': (g >= 180).sum()})
    return pd.DataFrame(rows).set_index('patient_id').sort_index()

def features_groupby(labs):
    """One split, built-in aggregations."""
    return (labs.assign(high=labs['glucose'] >= 180)
                .groupby('patient_id')
                .agg(n_tests=('glucose', 'size'), mean_glucose=('glucose', 'mean'),
                     n_high=('high', 'sum')))

labs = make_labs(100_000)
pd.testing.assert_frame_equal(features_loop(labs), features_groupby(labs),
                              check_dtype=False, check_names=False)  # same answer first

profile_function()(features_loop)(labs)
compare_operations({'loop': features_loop, 'groupby': features_groupby}, labs)
scaling(features_loop, [25_000, 50_000, 100_000], make_labs)
```

The profile puts about half of the loop's time in `.loc` and its boolean filter, run once for each of 5,000 patients. On one test machine the loop took 2.4 s and `groupby` 0.007 s, about 340 times faster. The loop's cost per row rose from 19 to 24 µs as the rows grew from 25,000 to 100,000, because each extra patient's filter scans the whole table again.

## Lab-Feed Validation for a Clinical Data Warehouse

A clinical data warehouse receives a lab feed every night. This framework separates a rule from its result: a rule checks the feed, a result records pass or fail and a severity, and the report collects all results. **Severity** describes the importance of a finding: a duplicated result is a warning, a glucose of 9,999 is an error. Here, even a critical result is recorded rather than raised: the loading job inspects the results before deciding whether to load the night's file. The 1,000-row rule is only an example volume check for one site, not a universal minimum.

### Code Snippet: Lab-Feed Validation Framework

```python
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Callable

import numpy as np
import pandas as pd

class Severity(Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

@dataclass
class ValidationResult:
    rule_name: str
    severity: Severity
    passed: bool
    message: str
    details: dict = field(default_factory=dict)
    affected_columns: list = field(default_factory=list)
    affected_rows: int = 0

class FeedValidator:
    """Run named rules on one feed and collect their results."""

    def __init__(self):
        self.rules = {}
        self.results = []

    def add_rule(self, name: str, rule: Callable, severity: Severity = Severity.WARNING):
        self.rules[name] = (rule, severity)

    def validate(self, df: pd.DataFrame) -> list:
        self.results = []
        for name, (rule, severity) in self.rules.items():
            try:
                result = rule(df)
                if not isinstance(result, ValidationResult):   # a plain True/False rule
                    result = ValidationResult(name, severity, bool(result),
                                              'passed' if result else 'failed')
            except Exception as e:
                result = ValidationResult(name, Severity.CRITICAL, False,
                                          f'rule raised {type(e).__name__}: {e}')
            self.results.append(result)
            status = 'PASS' if result.passed else 'FAIL'
            print(f"{status} {name} ({result.severity.value}): {result.message}")
        return self.results

    def get_summary(self) -> dict:
        summary = {'total': len(self.results),
                   'passed': sum(r.passed for r in self.results), 'failed_by_severity': {}}
        for r in self.results:
            if not r.passed:
                key = r.severity.value
                summary['failed_by_severity'][key] = summary['failed_by_severity'].get(key, 0) + 1
        return summary

    def generate_report(self, output_file='lab_feed_validation.md'):
        summary = self.get_summary()
        lines = ["# Lab Feed Validation Report", "",
                 f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
                 f"Rules: {summary['total']}, passed: {summary['passed']}", ""]
        for r in self.results:
            lines.append(f"## {r.rule_name}: {'PASSED' if r.passed else 'FAILED'} ({r.severity.value})")
            lines.append(r.message)
            lines += [f"- {key}: {value}" for key, value in r.details.items()]
            if r.affected_columns:
                lines.append(f"- Affected columns: {', '.join(r.affected_columns)}")
            if r.affected_rows:
                lines.append(f"- Affected rows: {r.affected_rows}")
            lines.append("")
        with open(output_file, 'w') as f:
            f.write("\n".join(lines))

def check_missing(df, threshold=0.05):
    missing = df.isna().mean()
    bad = missing[missing > threshold]
    return ValidationResult('missing_values', Severity.WARNING, bad.empty,
                            f"no column over {threshold:.0%} missing" if bad.empty
                            else f"over {threshold:.0%} missing: " + ', '.join(f"{c} ({p:.1%})" for c, p in bad.items()),
                            {'threshold': threshold}, list(bad.index))

def check_duplicates(df):
    n = int(df.duplicated(['patient_id', 'test_code', 'collected_at']).sum())
    return ValidationResult('duplicate_results', Severity.WARNING, n == 0,
                            f"repeated results: {n} ({n / len(df):.1%})", affected_rows=n)

def check_numeric_text(df, col='result_value'):
    # Lab systems send values such as '<5' or 'HEMOLYZED' in a numeric field.
    as_number = pd.to_numeric(df[col], errors='coerce')
    bad = df.loc[as_number.isna() & df[col].notna(), col]
    return ValidationResult('numeric_result_text', Severity.INFO, bad.empty,
                            f"non-numeric results: {len(bad)}",
                            {'examples': bad.value_counts().head(3).to_dict()}, [col], len(bad))

def check_glucose_range(df, low=10, high=1000):
    value = pd.to_numeric(df['result_value'], errors='coerce')
    bad = (df['test_code'] == 'GLU') & ~value.between(low, high) & value.notna()
    return ValidationResult('glucose_plausible', Severity.ERROR, not bad.any(),
                            f"glucose outside {low} to {high} mg/dL: {int(bad.sum())} results",
                            {'values': sorted(value[bad].unique().tolist())}, ['result_value'], int(bad.sum()))

validator = FeedValidator()
validator.add_rule('missing_values', check_missing)
validator.add_rule('duplicate_results', check_duplicates)
validator.add_rule('numeric_result_text', check_numeric_text, Severity.INFO)
validator.add_rule('glucose_plausible', check_glucose_range, Severity.ERROR)
validator.add_rule('enough_rows', lambda df: len(df) >= 1000, Severity.ERROR)

# One night's feed: 1,200 results with problems planted the way interfaces produce them
rng = np.random.default_rng(42)
feed = pd.DataFrame({
    'patient_id': rng.integers(1000, 1400, 1200).astype(str),
    'test_code': rng.choice(['GLU', 'K', 'NA'], 1200),
    'collected_at': pd.date_range('2025-03-01', periods=1200, freq='min'),
    'result_value': rng.normal(100, 20, 1200).round().astype(str),
})
feed.loc[[5, 17], 'result_value'] = ['<5', 'HEMOLYZED']
feed.loc[feed['test_code'].eq('GLU').idxmax(), 'result_value'] = '9999'
feed.loc[40:119, 'patient_id'] = None
feed = pd.concat([feed, feed.iloc[[200, 201]]], ignore_index=True)

validator.validate(feed)
print(validator.get_summary())
validator.generate_report()
```

```text
FAIL missing_values (warning): over 5% missing: patient_id (6.7%)
FAIL duplicate_results (warning): repeated results: 2 (0.2%)
FAIL numeric_result_text (info): non-numeric results: 2
FAIL glucose_plausible (error): glucose outside 10 to 1000 mg/dL: 1 results
PASS enough_rows (error): passed
{'total': 5, 'passed': 1, 'failed_by_severity': {'warning': 2, 'info': 1, 'error': 1}}
```

The loading job would hold this file: the `glucose_plausible` error is the stop condition, while the warnings go into the report for the interface team.
