# Lecture 07: Data Wrangling + Statistical Visualization

**Duration**: 5 hours  
**Level**: Advanced Professional Track  
**Prerequisites**: L06 Advanced Data Loading + Cleaning

## Professional Context: From Clean Data to Analysis-Ready Insights

With clean, integrated datasets from L06, we now face the next professional challenge: transforming data into analysis-ready formats and creating compelling visualizations that communicate findings effectively.

In professional environments, data wrangling and visualization skills determine whether insights get implemented or ignored:

- **Executive presentations**: Stakeholders need clear, compelling visuals
- **Peer review**: Statistical graphics must meet publication standards  
- **Regulatory submissions**: Visualizations require specific formatting and validation
- **Clinical decision support**: Real-time dashboards must be intuitive and actionable

Today we master the advanced wrangling and visualization techniques that transform data scientists into trusted analytical partners.

## Learning Objectives

By the end of this lecture, you will:

1. **Execute complex data transformations** using advanced pandas operations
2. **Design hierarchical data structures** with multi-level indexing for complex analyses
3. **Create publication-quality statistical visualizations** with proper statistical annotations
4. **Build interactive dashboards** for stakeholder communication
5. **Implement time series and longitudinal data analysis workflows**

---

## Part 1: Advanced Data Transformations (75 minutes)

### Professional Challenge: Longitudinal Clinical Data Analysis

Consider a real-world scenario: analyzing treatment response patterns in a clinical trial with:
- **Repeated measurements**: Lab values collected at baseline, 30, 60, 90 days
- **Multiple treatments**: Drug A vs. Drug B vs. placebo across different sites
- **Complex endpoints**: Primary outcome (efficacy) and safety outcomes
- **Missing data patterns**: Informative missingness due to treatment discontinuation

### Master-Level Pandas Operations

#### 1. Hierarchical Data Restructuring

```python
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class ClinicalDataTransformer:
    """Advanced transformer for clinical trial longitudinal data."""
    
    def __init__(self, study_config: Dict):
        self.config = study_config
        self.visit_schedule = study_config['visit_schedule']
        self.endpoints = study_config['endpoints']
        
    def create_analysis_dataset(self, 
                               demographics: pd.DataFrame,
                               lab_data: pd.DataFrame,
                               outcomes: pd.DataFrame) -> pd.DataFrame:
        """Create analysis-ready dataset with proper hierarchical structure."""
        
        # 1. Reshape lab data from long to wide format with visit structure
        lab_pivoted = self._create_visit_structure(lab_data)
        
        # 2. Calculate derived variables (changes from baseline, percent changes)
        lab_derived = self._calculate_derived_variables(lab_pivoted)
        
        # 3. Merge with demographics maintaining proper index structure  
        analysis_df = self._merge_with_demographics(lab_derived, demographics)
        
        # 4. Add outcome variables with time-to-event calculations
        final_df = self._add_outcomes(analysis_df, outcomes)
        
        return final_df
    
    def _create_visit_structure(self, lab_data: pd.DataFrame) -> pd.DataFrame:
        """Transform long-format lab data to analysis-ready wide format."""
        
        # Create standardized visit labels
        visit_mapping = {day: f"Visit_{i+1}" 
                        for i, day in enumerate(self.visit_schedule)}
        
        lab_data['visit_std'] = lab_data['study_day'].map(
            lambda x: self._map_to_nearest_visit(x, visit_mapping)
        )
        
        # Pivot to create columns for each lab test at each visit
        lab_pivoted = lab_data.pivot_table(
            index=['patient_id', 'treatment_group'],
            columns=['lab_test', 'visit_std'],
            values='lab_value',
            aggfunc='mean'  # Handle multiple values per visit
        )
        
        # Flatten column names for easier access
        lab_pivoted.columns = [f"{lab}_{visit}" for lab, visit in lab_pivoted.columns]
        lab_pivoted = lab_pivoted.reset_index()
        
        return lab_pivoted
    
    def _calculate_derived_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate clinical endpoints: changes from baseline, percent changes."""
        
        derived_df = df.copy()
        
        # For each lab test, calculate derived variables
        for lab_test in self.endpoints['lab_tests']:
            baseline_col = f"{lab_test}_Visit_1"
            
            if baseline_col in df.columns:
                # Absolute change from baseline for each visit
                for visit in ['Visit_2', 'Visit_3', 'Visit_4']:
                    current_col = f"{lab_test}_{visit}"
                    change_col = f"{lab_test}_{visit}_change"
                    
                    if current_col in df.columns:
                        derived_df[change_col] = (
                            df[current_col] - df[baseline_col]
                        )
                
                # Percent change from baseline
                for visit in ['Visit_2', 'Visit_3', 'Visit_4']:
                    current_col = f"{lab_test}_{visit}"
                    pct_change_col = f"{lab_test}_{visit}_pct_change"
                    
                    if current_col in df.columns:
                        derived_df[pct_change_col] = (
                            (df[current_col] - df[baseline_col]) / 
                            df[baseline_col] * 100
                        )
                
                # Maximum change across all visits
                change_cols = [f"{lab_test}_{visit}_change" 
                             for visit in ['Visit_2', 'Visit_3', 'Visit_4']
                             if f"{lab_test}_{visit}_change" in derived_df.columns]
                
                if change_cols:
                    derived_df[f"{lab_test}_max_change"] = (
                        derived_df[change_cols].max(axis=1)
                    )
        
        return derived_df
    
    def _advanced_groupby_operations(self, df: pd.DataFrame) -> pd.DataFrame:
        """Demonstrate advanced groupby operations for clinical analysis."""
        
        # Complex aggregations by treatment group and site
        treatment_summary = (df
                           .groupby(['treatment_group', 'site_id'])
                           .agg({
                               'ldl_cholesterol_Visit_4': ['count', 'mean', 'std', 
                                                         lambda x: x.quantile(0.25),
                                                         lambda x: x.quantile(0.75)],
                               'patient_id': 'count'
                           })
                           .round(2)
        )
        
        # Flatten multi-level column names
        treatment_summary.columns = [
            '_'.join(col).strip() if col[1] else col[0]
            for col in treatment_summary.columns.values
        ]
        
        # Calculate treatment effect sizes by site
        treatment_effects = []
        
        for site in df['site_id'].unique():
            site_data = df[df['site_id'] == site]
            
            for endpoint in ['ldl_cholesterol_Visit_4_change', 'hdl_cholesterol_Visit_4_change']:
                if endpoint in df.columns:
                    # Calculate Cohen's d between treatment groups
                    drug_a = site_data[site_data['treatment_group'] == 'Drug_A'][endpoint].dropna()
                    placebo = site_data[site_data['treatment_group'] == 'Placebo'][endpoint].dropna()
                    
                    if len(drug_a) > 0 and len(placebo) > 0:
                        pooled_std = np.sqrt(((len(drug_a) - 1) * drug_a.var() + 
                                            (len(placebo) - 1) * placebo.var()) / 
                                           (len(drug_a) + len(placebo) - 2))
                        
                        cohens_d = (drug_a.mean() - placebo.mean()) / pooled_std
                        
                        treatment_effects.append({
                            'site_id': site,
                            'endpoint': endpoint,
                            'drug_a_mean': drug_a.mean(),
                            'placebo_mean': placebo.mean(),
                            'mean_difference': drug_a.mean() - placebo.mean(),
                            'cohens_d': cohens_d,
                            'n_drug_a': len(drug_a),
                            'n_placebo': len(placebo)
                        })
        
        return pd.DataFrame(treatment_effects)
```

#### 2. Advanced String Operations for Clinical Coding

```python
class ClinicalCodingTransformer:
    """Advanced text processing for clinical data (ICD codes, medications, etc.)."""
    
    def __init__(self):
        self.icd_patterns = {
            'cardiovascular': r'^I[0-9]',
            'diabetes': r'^E1[0-4]',
            'hypertension': r'^I1[0-5]',
            'hyperlipidemia': r'^E78'
        }
    
    def process_diagnosis_codes(self, df: pd.DataFrame, 
                              diagnosis_col: str = 'primary_diagnosis') -> pd.DataFrame:
        """Advanced processing of ICD-10 diagnosis codes."""
        
        result_df = df.copy()
        
        # Extract and standardize ICD codes
        result_df['icd_clean'] = (
            df[diagnosis_col]
            .str.upper()
            .str.replace(r'[^A-Z0-9.]', '', regex=True)
            .str[:7]  # Standard ICD-10 length
        )
        
        # Create disease category flags using vectorized operations
        for category, pattern in self.icd_patterns.items():
            result_df[f'has_{category}'] = (
                result_df['icd_clean']
                .str.contains(pattern, regex=True, na=False)
            )
        
        # Extract ICD chapter (first character/digit)
        result_df['icd_chapter'] = result_df['icd_clean'].str[0]
        
        # Count comorbidities per patient
        comorbidity_cols = [f'has_{cat}' for cat in self.icd_patterns.keys()]
        result_df['comorbidity_count'] = result_df[comorbidity_cols].sum(axis=1)
        
        return result_df
    
    def process_medication_strings(self, df: pd.DataFrame, 
                                 med_col: str = 'medications') -> pd.DataFrame:
        """Process free-text medication fields."""
        
        result_df = df.copy()
        
        # Common medication patterns
        statin_pattern = r'(atorvastatin|simvastatin|rosuvastatin|lovastatin|pravastatin)'
        ace_pattern = r'(lisinopril|enalapril|captopril|benazepril)'
        beta_blocker_pattern = r'(metoprolol|propranolol|atenolol|carvedilol)'
        
        # Extract medication classes
        result_df['on_statin'] = (
            df[med_col]
            .str.lower()
            .str.contains(statin_pattern, regex=True, na=False)
        )
        
        result_df['on_ace_inhibitor'] = (
            df[med_col]
            .str.lower() 
            .str.contains(ace_pattern, regex=True, na=False)
        )
        
        result_df['on_beta_blocker'] = (
            df[med_col]
            .str.lower()
            .str.contains(beta_blocker_pattern, regex=True, na=False)
        )
        
        # Count total medications (assuming comma-separated)
        result_df['medication_count'] = (
            df[med_col]
            .str.count(',') + 1
            .where(df[med_col].notna(), 0)
        )
        
        return result_df
```

### Hands-On Exercise 1: Complex Clinical Data Transformation

**Scenario**: You have longitudinal lipid panel data from a statin efficacy trial.

**Data Structure**:
- Patient demographics with baseline characteristics
- Lab results collected at 0, 30, 60, 90 days  
- Medication adherence data
- Adverse event reports

**Your Task**: Create an analysis-ready dataset that includes:
- Baseline and change from baseline for all lipid parameters
- Treatment group comparisons
- Time-to-event variables for adverse events
- Derived efficacy endpoints (% achieving LDL <70 mg/dL)

---

## Part 2: Multi-Level Indexing and Hierarchical Data (60 minutes)

### Professional Application: Multi-Site Clinical Trial Analysis

Multi-level indexing becomes essential when analyzing:
- **Hierarchical structures**: Sites → Patients → Visits → Measurements
- **Repeated measures**: Same patient measured multiple times
- **Grouped analyses**: By treatment, site, demographics
- **Complex aggregations**: Cross-tabulations with multiple dimensions

### Advanced Indexing Techniques

#### 1. Creating and Managing Multi-Level Indexes

```python
class HierarchicalDataManager:
    """Manage complex hierarchical clinical trial data."""
    
    def __init__(self):
        self.index_levels = ['site_id', 'patient_id', 'visit_number']
    
    def create_hierarchical_structure(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert flat data to hierarchical multi-index structure."""
        
        # Set hierarchical index
        df_indexed = df.set_index(self.index_levels)
        
        # Sort index for optimal performance
        df_indexed = df_indexed.sort_index()
        
        return df_indexed
    
    def analyze_by_hierarchy(self, df: pd.DataFrame) -> Dict:
        """Perform analyses at different hierarchical levels."""
        
        results = {}
        
        # Site-level analysis
        results['by_site'] = (
            df.groupby(level='site_id')
            .agg({
                'ldl_cholesterol': ['mean', 'std', 'count'],
                'hdl_cholesterol': ['mean', 'std', 'count'],
                'total_cholesterol': ['mean', 'std', 'count']
            })
            .round(2)
        )
        
        # Patient-level analysis (across all visits)
        results['by_patient'] = (
            df.groupby(level=['site_id', 'patient_id'])
            .agg({
                'ldl_cholesterol': ['mean', 'min', 'max', 'count'],
                'systolic_bp': ['mean', 'min', 'max'],
                'weight': ['first', 'last', lambda x: x.last() - x.first()]  # Weight change
            })
            .round(2)
        )
        
        # Visit-level analysis
        results['by_visit'] = (
            df.groupby(level='visit_number')
            .agg({
                'ldl_cholesterol': ['mean', 'std', 'count'],
                'patient_adherence': 'mean'
            })
            .round(2)
        )
        
        return results
    
    def cross_sectional_analysis(self, df: pd.DataFrame, 
                                treatment_col: str = 'treatment_group') -> pd.DataFrame:
        """Create cross-tabulations across hierarchical levels."""
        
        # Reset index to access grouping variables
        df_reset = df.reset_index()
        
        # Create cross-tabulation: Site × Treatment × Visit
        cross_tab = pd.crosstab(
            index=[df_reset['site_id'], df_reset['visit_number']],
            columns=df_reset[treatment_col],
            values=df_reset['ldl_cholesterol'],
            aggfunc=['count', 'mean', 'std']
        )
        
        return cross_tab
    
    def longitudinal_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Analyze longitudinal patterns using advanced indexing."""
        
        # Calculate visit-to-visit changes
        df_changes = (
            df.groupby(level=['site_id', 'patient_id'])
            .pct_change()  # Percent change from previous visit
            .add_suffix('_pct_change')
        )
        
        # Calculate rolling averages (3-visit window)
        df_rolling = (
            df.groupby(level=['site_id', 'patient_id'])
            .rolling(window=3, min_periods=2)
            .mean()
            .add_suffix('_rolling_mean')
        )
        
        # Combine original data with derived features
        combined = pd.concat([df, df_changes, df_rolling], axis=1)
        
        return combined
```

#### 2. Advanced Slicing and Selection Operations

```python
class AdvancedDataSlicer:
    """Advanced data slicing operations for clinical analysis."""
    
    def __init__(self, df: pd.DataFrame):
        self.df = df  # Assumes multi-index DataFrame
    
    def slice_by_treatment_and_visit(self, 
                                   treatment: str, 
                                   visits: List[int]) -> pd.DataFrame:
        """Advanced slicing by treatment group and visit numbers."""
        
        # Reset index to access treatment column, then re-index
        df_reset = self.df.reset_index()
        treatment_data = df_reset[df_reset['treatment_group'] == treatment]
        
        # Re-establish hierarchical index
        treatment_indexed = treatment_data.set_index(['site_id', 'patient_id', 'visit_number'])
        
        # Slice by visits using isin for multiple values
        visit_slice = treatment_indexed.loc[
            treatment_indexed.index.get_level_values('visit_number').isin(visits)
        ]
        
        return visit_slice
    
    def analyze_site_performance(self, endpoint: str = 'ldl_cholesterol') -> pd.DataFrame:
        """Analyze site performance using advanced indexing."""
        
        site_analysis = []
        
        for site in self.df.index.get_level_values('site_id').unique():
            site_data = self.df.xs(site, level='site_id')
            
            # Calculate site-specific metrics
            baseline_mean = site_data.xs(1, level='visit_number')[endpoint].mean()
            final_mean = site_data.xs(4, level='visit_number')[endpoint].mean()
            
            # Calculate percent of patients achieving target
            final_values = site_data.xs(4, level='visit_number')[endpoint].dropna()
            pct_at_target = (final_values < 70).mean() * 100  # LDL < 70 mg/dL
            
            site_analysis.append({
                'site_id': site,
                'n_patients': len(site_data.index.get_level_values('patient_id').unique()),
                'baseline_mean': baseline_mean,
                'final_mean': final_mean,
                'mean_change': final_mean - baseline_mean,
                'pct_at_ldl_target': pct_at_target
            })
        
        return pd.DataFrame(site_analysis)
    
    def identify_treatment_responders(self, 
                                    threshold: float = -30.0,
                                    endpoint: str = 'ldl_cholesterol') -> pd.DataFrame:
        """Identify treatment responders using complex criteria."""
        
        responder_analysis = []
        
        # Group by patient and calculate response
        for (site, patient), patient_data in self.df.groupby(level=['site_id', 'patient_id']):
            
            if 1 in patient_data.index.get_level_values('visit_number') and \
               4 in patient_data.index.get_level_values('visit_number'):
                
                baseline = patient_data.xs(1, level='visit_number')[endpoint].iloc[0]
                final = patient_data.xs(4, level='visit_number')[endpoint].iloc[0]
                
                pct_change = ((final - baseline) / baseline) * 100
                is_responder = pct_change <= threshold  # 30% reduction
                
                responder_analysis.append({
                    'site_id': site,
                    'patient_id': patient,
                    'baseline_value': baseline,
                    'final_value': final,
                    'percent_change': pct_change,
                    'is_responder': is_responder
                })
        
        return pd.DataFrame(responder_analysis)
```

### Hands-On Exercise 2: Hierarchical Data Analysis

**Scenario**: Multi-site diabetes management program with HbA1c monitoring.

**Data Structure**: Site → Patient → Monthly visits with HbA1c, weight, blood pressure

**Your Task**: 
- Create hierarchical index structure
- Analyze site-level performance differences
- Identify patient response patterns
- Calculate time-to-target achievement

---

## Part 3: Publication-Quality Statistical Visualization (90 minutes)

### Professional Standards: Graphics That Get Published

Publication-quality graphics require:
- **Statistical accuracy**: Appropriate chart types for data types
- **Visual clarity**: Clear legends, proper scaling, readable fonts
- **Professional aesthetics**: Journal-ready formatting
- **Statistical annotations**: Confidence intervals, p-values, effect sizes

### Advanced Visualization Framework

#### 1. Statistical Graphics with Seaborn and Matplotlib

```python
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class ClinicalVisualizationMaster:
    """Create publication-quality clinical trial visualizations."""
    
    def __init__(self):
        # Set publication-ready style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("Set2")
        
        # Configure matplotlib for high-DPI displays
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['legend.fontsize'] = 12
        
    def create_primary_endpoint_plot(self, df: pd.DataFrame,
                                   endpoint: str = 'ldl_cholesterol_change',
                                   treatment_col: str = 'treatment_group') -> plt.Figure:
        """Create primary endpoint analysis plot with statistical annotations."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot with individual points
        sns.boxplot(data=df, x=treatment_col, y=endpoint, ax=ax1)
        sns.swarmplot(data=df, x=treatment_col, y=endpoint, ax=ax1, 
                     color='black', alpha=0.5, size=3)
        
        # Add statistical annotations
        treatment_groups = df[treatment_col].unique()
        if len(treatment_groups) == 2:
            group1_data = df[df[treatment_col] == treatment_groups[0]][endpoint].dropna()
            group2_data = df[df[treatment_col] == treatment_groups[1]][endpoint].dropna()
            
            # Perform t-test
            t_stat, p_value = stats.ttest_ind(group1_data, group2_data)
            
            # Add p-value annotation
            ax1.text(0.5, 0.95, f'p = {p_value:.4f}', 
                    transform=ax1.transAxes, ha='center', fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title('Primary Endpoint Analysis')
        ax1.set_ylabel(f'{endpoint.replace("_", " ").title()} (mg/dL)')
        
        # Violin plot showing distribution
        sns.violinplot(data=df, x=treatment_col, y=endpoint, ax=ax2)
        ax2.set_title('Distribution Comparison')
        ax2.set_ylabel('')
        
        plt.tight_layout()
        return fig
    
    def create_longitudinal_plot(self, df: pd.DataFrame) -> plt.Figure:
        """Create sophisticated longitudinal analysis plot."""
        
        # Prepare data for longitudinal plotting
        visit_columns = [col for col in df.columns if 'Visit_' in col and 'ldl_cholesterol' in col and 'change' not in col]
        
        # Reshape to long format
        id_vars = ['patient_id', 'treatment_group']
        df_long = pd.melt(df[id_vars + visit_columns], 
                         id_vars=id_vars,
                         value_vars=visit_columns,
                         var_name='visit',
                         value_name='ldl_cholesterol')
        
        # Extract visit number
        df_long['visit_number'] = df_long['visit'].str.extract(r'Visit_(\d+)').astype(int)
        
        # Calculate time in weeks (assuming visits at 0, 4, 8, 12 weeks)
        visit_to_week = {1: 0, 2: 4, 3: 8, 4: 12}
        df_long['week'] = df_long['visit_number'].map(visit_to_week)
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Individual patient trajectories (sample)
        sample_patients = df_long['patient_id'].unique()[:20]  # Show first 20 patients
        for treatment in df_long['treatment_group'].unique():
            treatment_data = df_long[
                (df_long['treatment_group'] == treatment) & 
                (df_long['patient_id'].isin(sample_patients))
            ]
            
            for patient in treatment_data['patient_id'].unique():
                patient_data = treatment_data[treatment_data['patient_id'] == patient]
                ax1.plot(patient_data['week'], patient_data['ldl_cholesterol'], 
                        alpha=0.3, linewidth=0.5, 
                        color='blue' if treatment == 'Drug_A' else 'red')
        
        ax1.set_xlabel('Week')
        ax1.set_ylabel('LDL Cholesterol (mg/dL)')
        ax1.set_title('Individual Patient Trajectories (Sample)')
        ax1.grid(True, alpha=0.3)
        
        # Mean trajectories with confidence intervals
        sns.lineplot(data=df_long, x='week', y='ldl_cholesterol', 
                    hue='treatment_group', ax=ax2, 
                    marker='o', markersize=8, linewidth=3)
        
        ax2.set_xlabel('Week')
        ax2.set_ylabel('LDL Cholesterol (mg/dL)')
        ax2.set_title('Mean Treatment Response Over Time')
        ax2.legend(title='Treatment Group', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        return fig
    
    def create_forest_plot(self, effect_data: pd.DataFrame) -> plt.Figure:
        """Create forest plot for meta-analysis or subgroup analysis."""
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        y_positions = range(len(effect_data))
        
        # Plot effect sizes with confidence intervals
        ax.errorbar(x=effect_data['effect_size'], 
                   y=y_positions,
                   xerr=[effect_data['effect_size'] - effect_data['ci_lower'],
                         effect_data['ci_upper'] - effect_data['effect_size']],
                   fmt='o', capsize=5, capthick=2, markersize=8,
                   color='darkblue', ecolor='darkblue')
        
        # Add vertical line at null effect
        ax.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        
        # Customize axes
        ax.set_yticks(y_positions)
        ax.set_yticklabels(effect_data['subgroup'])
        ax.set_xlabel('Effect Size (Cohen\'s d)')
        ax.set_title('Treatment Effect by Subgroup')
        
        # Add effect sizes as text
        for i, (idx, row) in enumerate(effect_data.iterrows()):
            ax.text(row['effect_size'] + 0.1, i, 
                   f"{row['effect_size']:.2f} ({row['ci_lower']:.2f}, {row['ci_upper']:.2f})",
                   va='center', fontsize=10)
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        return fig
```

#### 2. Interactive Dashboards with Plotly

```python
class InteractiveClinicalDashboard:
    """Create interactive dashboards for clinical trial monitoring."""
    
    def __init__(self):
        self.colors = {
            'primary': '#1f77b4',
            'secondary': '#ff7f0e', 
            'success': '#2ca02c',
            'warning': '#d62728'
        }
    
    def create_trial_monitoring_dashboard(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive trial monitoring dashboard."""
        
        # Create subplots with different types
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Enrollment Progress',
                'Primary Endpoint Distribution', 
                'Safety Events Over Time',
                'Site Performance Comparison',
                'Treatment Response Rates',
                'Quality Metrics'
            ],
            specs=[
                [{'type': 'scatter'}, {'type': 'histogram'}, {'type': 'scatter'}],
                [{'type': 'bar'}, {'type': 'pie'}, {'type': 'indicator'}]
            ]
        )
        
        # 1. Enrollment progress over time
        enrollment_data = df.groupby('enrollment_date').size().cumsum().reset_index()
        enrollment_data.columns = ['date', 'cumulative_enrollment']
        
        fig.add_trace(
            go.Scatter(
                x=enrollment_data['date'],
                y=enrollment_data['cumulative_enrollment'],
                mode='lines+markers',
                name='Enrollment Progress',
                line=dict(color=self.colors['primary'], width=3)
            ),
            row=1, col=1
        )
        
        # 2. Primary endpoint distribution
        fig.add_trace(
            go.Histogram(
                x=df['ldl_cholesterol_Visit_4_change'].dropna(),
                nbinsx=30,
                name='LDL Change Distribution',
                marker_color=self.colors['secondary']
            ),
            row=1, col=2
        )
        
        # 3. Safety events over time
        if 'adverse_event_date' in df.columns:
            safety_data = df[df['adverse_event'].notna()]
            safety_counts = safety_data.groupby('adverse_event_date').size().reset_index()
            
            fig.add_trace(
                go.Scatter(
                    x=safety_counts['adverse_event_date'],
                    y=safety_counts[0],
                    mode='markers',
                    name='Safety Events',
                    marker=dict(color=self.colors['warning'], size=8)
                ),
                row=1, col=3
            )
        
        # 4. Site performance comparison
        site_performance = (df.groupby('site_id')['ldl_cholesterol_Visit_4_change']
                           .mean().reset_index())
        
        fig.add_trace(
            go.Bar(
                x=site_performance['site_id'],
                y=site_performance['ldl_cholesterol_Visit_4_change'],
                name='Mean LDL Change by Site',
                marker_color=self.colors['success']
            ),
            row=2, col=1
        )
        
        # 5. Treatment response rates
        response_rates = (df.groupby('treatment_group')['is_responder']
                         .mean() * 100).reset_index()
        
        fig.add_trace(
            go.Pie(
                labels=response_rates['treatment_group'],
                values=response_rates['is_responder'],
                name='Response Rates'
            ),
            row=2, col=2
        )
        
        # 6. Overall quality metric
        quality_score = 85  # Calculate from actual data quality metrics
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Data Quality Score"},
                gauge={'axis': {'range': [None, 100]},
                      'bar': {'color': self.colors['success']},
                      'steps': [
                          {'range': [0, 50], 'color': "lightgray"},
                          {'range': [50, 80], 'color': "gray"}
                      ],
                      'threshold': {'line': {'color': "red", 'width': 4},
                                  'thickness': 0.75, 'value': 90}}
            ),
            row=2, col=3
        )
        
        # Update layout
        fig.update_layout(
            height=800,
            showlegend=False,
            title_text="Clinical Trial Monitoring Dashboard",
            title_x=0.5
        )
        
        return fig
```

### Hands-On Exercise 3: Publication-Ready Visualizations

**Scenario**: Create publication-ready figures for a cardiovascular outcomes paper.

**Required Figures**:
1. **Primary endpoint analysis**: Treatment comparison with statistical annotations
2. **Longitudinal response**: Mean trajectories with confidence intervals  
3. **Subgroup analysis**: Forest plot showing treatment effects across demographics
4. **Safety profile**: Comprehensive adverse event visualization

**Quality Standards**: Journal-ready formatting, statistical annotations, professional aesthetics.

---

## Part 4: Time Series and Longitudinal Analysis (75 minutes)

### Professional Applications: Temporal Data Mastery

Time series analysis is crucial for:
- **Clinical monitoring**: Vital signs, lab values over time
- **Epidemiological studies**: Disease surveillance, outbreak detection
- **Healthcare operations**: Patient flow, resource utilization  
- **Pharmacokinetics**: Drug concentration profiles

### Advanced Temporal Analysis

#### 1. Time Series Preprocessing and Feature Engineering

```python
import pandas as pd
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
import numpy as np

class ClinicalTimeSeriesAnalyzer:
    """Advanced time series analysis for clinical data."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        
    def preprocess_time_series(self, df: pd.DataFrame, 
                             timestamp_col: str,
                             value_col: str,
                             patient_col: str) -> pd.DataFrame:
        """Comprehensive time series preprocessing."""
        
        # Ensure datetime format
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Sort by patient and timestamp
        df_sorted = df.sort_values([patient_col, timestamp_col])
        
        # Create time-based features
        df_featured = self._create_temporal_features(df_sorted, timestamp_col)
        
        # Handle missing values with forward fill (clinical context appropriate)
        df_featured[value_col] = (df_featured
                                 .groupby(patient_col)[value_col]
                                 .fillna(method='ffill'))
        
        # Detect and handle outliers
        df_cleaned = self._handle_temporal_outliers(df_featured, value_col, patient_col)
        
        return df_cleaned
    
    def _create_temporal_features(self, df: pd.DataFrame, timestamp_col: str) -> pd.DataFrame:
        """Create comprehensive temporal features."""
        
        df_featured = df.copy()
        
        # Basic temporal components
        df_featured['year'] = df[timestamp_col].dt.year
        df_featured['month'] = df[timestamp_col].dt.month
        df_featured['day_of_week'] = df[timestamp_col].dt.day_name()
        df_featured['hour'] = df[timestamp_col].dt.hour
        df_featured['is_weekend'] = df[timestamp_col].dt.weekend
        
        # Clinical context: shift patterns
        df_featured['shift'] = pd.cut(df_featured['hour'], 
                                    bins=[0, 8, 16, 24], 
                                    labels=['Night', 'Day', 'Evening'],
                                    right=False)
        
        # Time since first measurement (per patient)
        df_featured['days_since_baseline'] = (
            df_featured.groupby('patient_id')[timestamp_col]
            .transform(lambda x: (x - x.min()).dt.days)
        )
        
        return df_featured
    
    def analyze_longitudinal_patterns(self, df: pd.DataFrame,
                                    value_col: str,
                                    patient_col: str) -> Dict:
        """Analyze longitudinal patterns in clinical data."""
        
        patterns = {}
        
        # 1. Individual patient trajectories
        patient_trajectories = []
        
        for patient in df[patient_col].unique():
            patient_data = df[df[patient_col] == patient].copy()
            
            if len(patient_data) >= 3:  # Minimum observations for trend analysis
                # Calculate trend using linear regression
                x = np.arange(len(patient_data))
                y = patient_data[value_col].values
                
                # Remove NaN values
                mask = ~np.isnan(y)
                if mask.sum() >= 2:
                    slope, intercept = np.polyfit(x[mask], y[mask], 1)
                    
                    patient_trajectories.append({
                        'patient_id': patient,
                        'n_observations': len(patient_data),
                        'slope': slope,
                        'baseline_value': y[0] if not np.isnan(y[0]) else np.nan,
                        'final_value': y[-1] if not np.isnan(y[-1]) else np.nan,
                        'trend_direction': 'increasing' if slope > 0 else 'decreasing',
                        'mean_value': np.nanmean(y),
                        'value_range': np.nanmax(y) - np.nanmin(y)
                    })
        
        patterns['individual_trajectories'] = pd.DataFrame(patient_trajectories)
        
        # 2. Population-level trends
        daily_means = (df.groupby(df['timestamp'].dt.date)[value_col]
                      .agg(['mean', 'std', 'count'])
                      .reset_index())
        
        patterns['population_trends'] = daily_means
        
        # 3. Seasonal patterns (if applicable)
        monthly_patterns = (df.groupby(df['timestamp'].dt.month)[value_col]
                           .agg(['mean', 'std'])
                           .reset_index())
        
        patterns['seasonal_patterns'] = monthly_patterns
        
        return patterns
    
    def detect_change_points(self, df: pd.DataFrame, 
                           value_col: str,
                           patient_col: str) -> pd.DataFrame:
        """Detect significant changes in patient trajectories."""
        
        change_points = []
        
        for patient in df[patient_col].unique():
            patient_data = df[df[patient_col] == patient].copy()
            patient_data = patient_data.sort_values('timestamp')
            
            if len(patient_data) >= 5:  # Minimum for change point detection
                values = patient_data[value_col].dropna().values
                
                if len(values) >= 5:
                    # Simple change point detection using moving averages
                    window = max(2, len(values) // 4)
                    
                    # Calculate moving averages
                    early_avg = np.mean(values[:window])
                    late_avg = np.mean(values[-window:])
                    
                    # Check for significant change (>20% change)
                    pct_change = abs((late_avg - early_avg) / early_avg) * 100
                    
                    if pct_change > 20:  # Threshold for clinically significant change
                        change_points.append({
                            'patient_id': patient,
                            'early_average': early_avg,
                            'late_average': late_avg,
                            'percent_change': pct_change,
                            'change_direction': 'increase' if late_avg > early_avg else 'decrease',
                            'n_observations': len(values)
                        })
        
        return pd.DataFrame(change_points)
```

#### 2. Advanced Time Series Visualization

```python
class TimeSeriesVisualizer:
    """Create sophisticated time series visualizations."""
    
    def create_longitudinal_heatmap(self, df: pd.DataFrame,
                                  patient_col: str,
                                  timestamp_col: str,
                                  value_col: str) -> plt.Figure:
        """Create heatmap showing patient trajectories over time."""
        
        # Pivot data for heatmap
        # First, create time bins (e.g., weekly)
        df['time_bin'] = df[timestamp_col].dt.to_period('W')
        
        pivot_data = df.pivot_table(
            index=patient_col,
            columns='time_bin',
            values=value_col,
            aggfunc='mean'
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 10))
        
        sns.heatmap(pivot_data, 
                   cmap='RdYlBu_r', 
                   center=pivot_data.mean().mean(),
                   cbar_kws={'label': f'{value_col.replace("_", " ").title()}'},
                   ax=ax)
        
        ax.set_title('Longitudinal Patient Trajectories Heatmap')
        ax.set_xlabel('Time Period (Weeks)')
        ax.set_ylabel('Patient ID')
        
        plt.tight_layout()
        return fig
    
    def create_trajectory_clusters(self, df: pd.DataFrame,
                                 patient_trajectories: pd.DataFrame) -> plt.Figure:
        """Visualize clusters of similar patient trajectories."""
        
        # Perform clustering on trajectory features
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        
        # Features for clustering: slope, baseline, final value
        features = ['slope', 'baseline_value', 'final_value', 'mean_value']
        trajectory_features = patient_trajectories[features].dropna()
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(trajectory_features)
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(features_scaled)
        
        trajectory_features['cluster'] = clusters
        
        # Create visualization
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Scatter plot: Slope vs Baseline
        scatter = ax1.scatter(trajectory_features['baseline_value'], 
                            trajectory_features['slope'],
                            c=clusters, cmap='viridis', alpha=0.7)
        ax1.set_xlabel('Baseline Value')
        ax1.set_ylabel('Slope (Trend)')
        ax1.set_title('Patient Trajectories: Slope vs Baseline')
        plt.colorbar(scatter, ax=ax1)
        
        # Box plot: Mean values by cluster
        trajectory_features_long = trajectory_features.reset_index()
        sns.boxplot(data=trajectory_features_long, x='cluster', y='mean_value', ax=ax2)
        ax2.set_title('Mean Values by Trajectory Cluster')
        ax2.set_xlabel('Cluster')
        ax2.set_ylabel('Mean Value')
        
        # Histogram: Slope distribution by cluster
        for cluster in sorted(clusters):
            cluster_data = trajectory_features[trajectory_features['cluster'] == cluster]
            ax3.hist(cluster_data['slope'], alpha=0.7, label=f'Cluster {cluster}', bins=20)
        ax3.set_xlabel('Slope')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Slope Distribution by Cluster')
        ax3.legend()
        
        # Summary statistics table
        cluster_summary = trajectory_features.groupby('cluster').agg({
            'slope': ['mean', 'std'],
            'baseline_value': ['mean', 'std'],
            'final_value': ['mean', 'std'],
            'mean_value': ['mean', 'std']
        }).round(2)
        
        # Display as text in subplot
        ax4.axis('off')
        table_data = []
        for cluster in sorted(clusters):
            cluster_data = cluster_summary.loc[cluster]
            table_data.append([
                f'Cluster {cluster}',
                f"{cluster_data[('slope', 'mean')]:.2f} ± {cluster_data[('slope', 'std')]:.2f}",
                f"{cluster_data[('baseline_value', 'mean')]:.1f} ± {cluster_data[('baseline_value', 'std')]:.1f}",
                f"{cluster_data[('final_value', 'mean')]:.1f} ± {cluster_data[('final_value', 'std')]:.1f}"
            ])
        
        table = ax4.table(cellText=table_data,
                         colLabels=['Cluster', 'Slope (Mean ± SD)', 'Baseline (Mean ± SD)', 'Final (Mean ± SD)'],
                         cellLoc='center',
                         loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Cluster Summary Statistics')
        
        plt.tight_layout()
        return fig
```

### Hands-On Exercise 4: Longitudinal Clinical Analysis

**Scenario**: ICU patient monitoring data with continuous vital signs.

**Data**: Blood pressure, heart rate, oxygen saturation collected every 15 minutes for 72 hours

**Your Task**:
- Preprocess and clean the time series data
- Identify patient trajectory patterns
- Detect clinically significant changes
- Create visualization dashboard for ICU monitoring

---

## Assessment and Professional Integration

### Capstone Project: Comprehensive Clinical Trial Analysis

**Project Overview**: Complete analysis of a simulated Phase III cardiovascular trial

**Dataset Components**:
- **Demographics**: 500 patients across 10 sites
- **Laboratory data**: Lipid panels at baseline, 30, 60, 90 days
- **Medication data**: Adherence monitoring
- **Outcomes**: Primary (LDL reduction) and safety endpoints

**Deliverables**:

1. **Data Integration Pipeline** (25 points)
   - Multi-source data loading with validation
   - Hierarchical data structure creation
   - Quality assessment report

2. **Advanced Transformations** (25 points)  
   - Derived variable calculations
   - Time-to-event analysis
   - Missing data handling strategy

3. **Publication-Quality Visualizations** (25 points)
   - Primary endpoint analysis with statistical annotations
   - Longitudinal response trajectories
   - Subgroup analysis forest plots
   - Interactive monitoring dashboard

4. **Professional Documentation** (25 points)
   - Executive summary for clinical team
   - Technical methods documentation
   - Reproducible analysis pipeline
   - Regulatory compliance report

### Professional Skills Mastery Checklist

After this lecture, you should confidently:

- [ ] Execute complex multi-step data transformations
- [ ] Design and manage hierarchical data structures  
- [ ] Create publication-ready statistical visualizations
- [ ] Build interactive dashboards for stakeholder communication
- [ ] Analyze longitudinal and time series clinical data
- [ ] Apply advanced pandas operations for real-world scenarios
- [ ] Integrate clinical domain knowledge into technical solutions

### Looking Ahead to L08

In our next lecture, we'll leverage these data wrangling and visualization foundations to master:
- **Advanced statistical modeling** techniques
- **Machine learning pipelines** for predictive analytics
- **Model interpretation** and clinical decision support
- **Time series forecasting** for healthcare applications

The professional-grade data manipulation and visualization skills you've developed today will be essential as we move toward sophisticated analytical modeling.

---

## Additional Resources

### Technical References
- **Pandas Documentation**: Advanced operations guide
- **Seaborn Gallery**: Statistical visualization examples
- **Plotly Documentation**: Interactive dashboard creation
- **Clinical Data Interchange Standards Consortium (CDISC)**: Data standards

### Statistical Graphics References
- Tufte, E. "The Visual Display of Quantitative Information"
- Cleveland, W. "Visualizing Data"
- Wickham, H. "ggplot2: Elegant Graphics for Data Analysis"

### Clinical Research Integration
- ICH Guidelines for Clinical Data Management
- FDA Guidance on Electronic Source Data
- Clinical Data Acquisition Standards Harmonization (CDASH)

The professional competencies you've developed today position you to create publication-quality analyses that meet both technical and clinical standards.