# Lecture 06: Advanced Data Loading + Data Cleaning Mastery

**Duration**: 4.5 hours  
**Level**: Advanced Professional Track  
**Prerequisites**: L01-L05 foundation concepts

## Professional Context: Why Advanced Data Loading Matters

In professional data science and clinical research environments, you rarely work with clean, single-source datasets. Real-world projects involve:

- **Multi-source integration**: EMR data, lab results, imaging metadata, registry data
- **Large-scale processing**: Millions of patient records, genomic datasets, longitudinal studies
- **Quality assurance**: Regulatory compliance, audit trails, reproducible pipelines
- **Performance optimization**: Memory-efficient processing, parallel operations

Today we master the professional-grade techniques that separate advanced practitioners from beginners.

## Learning Objectives

By the end of this lecture, you will:

1. **Design and implement multi-source data integration pipelines**
2. **Handle large datasets efficiently using chunking and streaming techniques**  
3. **Build comprehensive data quality assessment frameworks**
4. **Create automated cleaning workflows with audit trails**
5. **Optimize memory usage and processing speed for production environments**

---

## Part 1: Multi-Source Data Integration (90 minutes)

### Professional Challenge: Clinical Research Data Pipeline

Imagine you're building a cardiovascular outcomes study combining:
- **EMR data**: Patient demographics, comorbidities, medications
- **Lab data**: Serial biomarkers, lipid panels, HbA1c values  
- **Claims data**: Procedures, diagnoses, healthcare utilization
- **Registry data**: Cardiac catheterization results, outcomes

Each source has different:
- **Formats**: CSV, JSON, Parquet, database tables
- **Schemas**: Different column names, data types, missing value encoding
- **Temporal alignment**: Different time granularities, time zones
- **Quality issues**: Duplicates, outliers, inconsistent coding

### Advanced Loading Techniques

#### 1. Schema-Aware Loading with Validation

```python
import pandas as pd
import numpy as np
from pathlib import Path
import yaml
from typing import Dict, List, Optional
import logging

class DataLoader:
    def __init__(self, config_path: str):
        """Initialize with configuration file defining schemas and rules."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.setup_logging()
    
    def load_with_schema(self, source_name: str, file_path: str) -> pd.DataFrame:
        """Load data with predefined schema validation."""
        schema = self.config['schemas'][source_name]
        
        # Load with expected dtypes
        df = pd.read_csv(
            file_path,
            dtype=schema['dtypes'],
            parse_dates=schema.get('date_columns', []),
            na_values=schema.get('na_values', [])
        )
        
        # Validate required columns
        missing_cols = set(schema['required_columns']) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Log loading summary
        self.logger.info(f"Loaded {source_name}: {df.shape[0]:,} rows, {df.shape[1]} columns")
        
        return df
```

#### 2. Parallel Processing for Multiple Sources

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

class ParallelDataLoader:
    def __init__(self, n_workers: Optional[int] = None):
        self.n_workers = n_workers or mp.cpu_count() - 1
    
    def load_multiple_sources(self, source_configs: List[Dict]) -> Dict[str, pd.DataFrame]:
        """Load multiple data sources in parallel."""
        
        def load_single_source(config):
            source_name = config['name']
            file_path = config['path']
            
            if config.get('file_type') == 'parquet':
                df = pd.read_parquet(file_path)
            elif config.get('file_type') == 'json':
                df = pd.read_json(file_path, lines=config.get('json_lines', False))
            else:
                df = pd.read_csv(file_path, **config.get('read_options', {}))
            
            return source_name, df
        
        results = {}
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            # Submit all loading tasks
            future_to_source = {
                executor.submit(load_single_source, config): config['name']
                for config in source_configs
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_source):
                source_name = future_to_source[future]
                try:
                    name, df = future.result()
                    results[name] = df
                    print(f"✓ Loaded {name}: {df.shape}")
                except Exception as e:
                    print(f"✗ Failed to load {source_name}: {e}")
        
        return results
```

### Hands-On Exercise 1: Multi-Source Integration Pipeline

**Scenario**: You're analyzing factors affecting hospital readmission rates.

**Data Sources**:
1. **Admissions data** (CSV): Patient ID, admission date, discharge date, primary diagnosis
2. **Lab results** (JSON): Patient ID, test date, lab values, reference ranges
3. **Medication data** (Parquet): Patient ID, medication, start date, stop date
4. **Outcomes data** (CSV): Patient ID, readmission date, readmission reason

**Your Task**: Build a pipeline that:
- Loads all sources with appropriate schema validation
- Handles different date formats and missing value encodings
- Creates a unified patient timeline
- Generates a loading report with data quality metrics

---

## Part 2: Large Dataset Handling (75 minutes)

### Professional Challenge: Memory-Efficient Processing

Working with large datasets (>1GB) requires different strategies:
- **Chunking**: Process data in manageable pieces
- **Streaming**: Never load entire dataset into memory
- **Lazy evaluation**: Defer computation until needed
- **Memory profiling**: Monitor and optimize resource usage

### Advanced Techniques

#### 1. Chunked Processing with Progress Tracking

```python
from tqdm import tqdm
import psutil
import gc

class ChunkedProcessor:
    def __init__(self, chunk_size: int = 50000):
        self.chunk_size = chunk_size
        self.memory_threshold = 0.8  # 80% memory usage threshold
    
    def process_large_file(self, file_path: str, processing_func) -> pd.DataFrame:
        """Process large CSV file in chunks with memory monitoring."""
        
        # Get total rows for progress bar
        total_rows = sum(1 for _ in open(file_path)) - 1  # Exclude header
        
        results = []
        memory_warnings = 0
        
        chunk_iter = pd.read_csv(file_path, chunksize=self.chunk_size)
        
        with tqdm(total=total_rows, desc="Processing chunks") as pbar:
            for chunk_num, chunk in enumerate(chunk_iter):
                # Check memory usage
                memory_percent = psutil.virtual_memory().percent / 100
                if memory_percent > self.memory_threshold:
                    memory_warnings += 1
                    gc.collect()  # Force garbage collection
                
                # Process chunk
                processed_chunk = processing_func(chunk)
                results.append(processed_chunk)
                
                pbar.update(len(chunk))
                pbar.set_postfix({
                    'chunk': chunk_num + 1,
                    'memory': f"{memory_percent:.1%}",
                    'warnings': memory_warnings
                })
        
        return pd.concat(results, ignore_index=True)
```

#### 2. Dask for Out-of-Core Processing

```python
import dask.dataframe as dd
import dask

class DaskProcessor:
    def __init__(self, n_workers: int = 4):
        # Configure Dask for optimal performance
        dask.config.set({
            'dataframe.query-planning': True,
            'array.chunk-size': '256MiB'
        })
        self.n_workers = n_workers
    
    def create_large_dataset_pipeline(self, file_pattern: str) -> dd.DataFrame:
        """Create processing pipeline for large datasets using Dask."""
        
        # Read multiple files as single Dask DataFrame
        df = dd.read_csv(
            file_pattern,
            dtype={'patient_id': 'object',  # Avoid int64 memory issues
                   'lab_value': 'float32'},     # Use smaller numeric types
            parse_dates=['test_date']
        )
        
        return df
    
    def complex_aggregation(self, df: dd.DataFrame) -> dd.DataFrame:
        """Perform complex aggregations efficiently."""
        
        result = (df
                  .groupby(['patient_id', df.test_date.dt.date])
                  .agg({
                      'lab_value': ['mean', 'std', 'count'],
                      'abnormal_flag': 'sum'
                  })
                  .reset_index()
                 )
        
        # Flatten column names
        result.columns = ['_'.join(col).strip() if col[1] else col[0] 
                         for col in result.columns.values]
        
        return result
```

### Hands-On Exercise 2: Large Dataset Processing

**Scenario**: You have 10 million lab results (2.5GB CSV) and need to create patient-level summaries.

**Requirements**:
- Calculate rolling 30-day averages for each patient
- Identify abnormal value patterns  
- Generate summary statistics by patient demographics
- Process within 4GB memory limit

**Your Task**: Implement both chunked and Dask solutions, compare performance.

---

## Part 3: Comprehensive Data Quality Assessment (90 minutes)

### Professional Framework: Automated Quality Pipeline

Data quality in professional settings requires:
- **Systematic assessment**: Standardized quality metrics
- **Automated reporting**: Regular quality dashboards
- **Audit trails**: Track all cleaning decisions
- **Threshold-based alerts**: Flag critical quality issues

### Advanced Quality Assessment

#### 1. Comprehensive Quality Metrics

```python
from dataclasses import dataclass
from typing import Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class QualityReport:
    dataset_name: str
    n_rows: int
    n_cols: int
    missing_data: Dict[str, float]
    duplicates: int
    outliers: Dict[str, int]
    data_types: Dict[str, str]
    value_ranges: Dict[str, Dict[str, Any]]
    categorical_issues: Dict[str, List[str]]
    temporal_issues: Dict[str, str]
    quality_score: float

class DataQualityAssessor:
    def __init__(self, config: Dict):
        self.config = config
        self.quality_thresholds = config.get('quality_thresholds', {
            'missing_threshold': 0.3,  # 30% missing is concerning
            'outlier_threshold': 0.05,  # 5% outliers is concerning
            'duplicate_threshold': 0.01  # 1% duplicates is concerning
        })
    
    def comprehensive_assessment(self, df: pd.DataFrame, dataset_name: str) -> QualityReport:
        """Generate comprehensive data quality report."""
        
        # Basic metrics
        n_rows, n_cols = df.shape
        
        # Missing data analysis
        missing_data = self._analyze_missing_data(df)
        
        # Duplicate analysis
        duplicates = self._analyze_duplicates(df)
        
        # Outlier analysis
        outliers = self._analyze_outliers(df)
        
        # Data type consistency
        data_types = self._analyze_data_types(df)
        
        # Value range analysis
        value_ranges = self._analyze_value_ranges(df)
        
        # Categorical data issues
        categorical_issues = self._analyze_categorical_issues(df)
        
        # Temporal consistency
        temporal_issues = self._analyze_temporal_issues(df)
        
        # Overall quality score
        quality_score = self._calculate_quality_score(
            missing_data, duplicates, outliers, n_rows
        )
        
        return QualityReport(
            dataset_name=dataset_name,
            n_rows=n_rows,
            n_cols=n_cols,
            missing_data=missing_data,
            duplicates=duplicates,
            outliers=outliers,
            data_types=data_types,
            value_ranges=value_ranges,
            categorical_issues=categorical_issues,
            temporal_issues=temporal_issues,
            quality_score=quality_score
        )
    
    def _analyze_missing_data(self, df: pd.DataFrame) -> Dict[str, float]:
        """Analyze missing data patterns."""
        missing_pct = (df.isnull().sum() / len(df) * 100).to_dict()
        
        # Identify missing data patterns
        missing_patterns = {}
        for col, pct in missing_pct.items():
            if pct > 0:
                missing_patterns[col] = {
                    'percent': pct,
                    'count': df[col].isnull().sum(),
                    'pattern': self._identify_missing_pattern(df[col])
                }
        
        return missing_patterns
    
    def _analyze_outliers(self, df: pd.DataFrame) -> Dict[str, int]:
        """Identify outliers in numeric columns."""
        outliers = {}
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].nunique() > 10:  # Skip categorical-like columns
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_count = ((df[col] < lower_bound) | 
                               (df[col] > upper_bound)).sum()
                
                if outlier_count > 0:
                    outliers[col] = {
                        'count': outlier_count,
                        'percent': outlier_count / len(df) * 100,
                        'bounds': (lower_bound, upper_bound)
                    }
        
        return outliers
```

#### 2. Automated Quality Dashboard

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

class QualityDashboard:
    def __init__(self):
        self.colors = {
            'excellent': '#2E8B57',    # Sea Green
            'good': '#32CD32',         # Lime Green  
            'warning': '#FF8C00',      # Dark Orange
            'critical': '#DC143C'      # Crimson
        }
    
    def create_quality_dashboard(self, quality_report: QualityReport) -> go.Figure:
        """Create interactive quality assessment dashboard."""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=[
                'Quality Score Overview',
                'Missing Data by Column', 
                'Outlier Distribution',
                'Data Type Distribution',
                'Duplicate Analysis',
                'Temporal Issues'
            ],
            specs=[
                [{'type': 'indicator'}, {'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'pie'}, {'type': 'bar'}, {'type': 'bar'}]
            ]
        )
        
        # Quality score gauge
        score_color = self._get_score_color(quality_report.quality_score)
        
        fig.add_trace(
            go.Indicator(
                mode="gauge+number+delta",
                value=quality_report.quality_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Overall Quality Score"},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': score_color},
                    'steps': [
                        {'range': [0, 50], 'color': self.colors['critical']},
                        {'range': [50, 70], 'color': self.colors['warning']},
                        {'range': [70, 85], 'color': self.colors['good']},
                        {'range': [85, 100], 'color': self.colors['excellent']}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 70
                    }
                }
            ),
            row=1, col=1
        )
        
        # Missing data bar chart
        if quality_report.missing_data:
            missing_cols = list(quality_report.missing_data.keys())
            missing_pcts = [quality_report.missing_data[col]['percent'] 
                           for col in missing_cols]
            
            fig.add_trace(
                go.Bar(
                    x=missing_cols,
                    y=missing_pcts,
                    marker_color=[self._get_missing_color(pct) for pct in missing_pcts],
                    text=[f"{pct:.1f}%" for pct in missing_pcts],
                    textposition='auto'
                ),
                row=1, col=2
            )
        
        return fig
    
    def _get_score_color(self, score: float) -> str:
        """Get color based on quality score."""
        if score >= 85:
            return self.colors['excellent']
        elif score >= 70:
            return self.colors['good']
        elif score >= 50:
            return self.colors['warning']
        else:
            return self.colors['critical']
```

### Hands-On Exercise 3: Automated Quality Pipeline

**Scenario**: You're building a quality assurance system for a multi-site clinical trial.

**Requirements**:
- Process data from 15 sites with different collection practices
- Generate automated quality reports every week
- Flag datasets that don't meet quality thresholds
- Create executive summary dashboards

**Your Task**: Build an end-to-end quality assessment pipeline with automated reporting.

---

## Part 4: Production-Ready Cleaning Workflows (75 minutes)

### Professional Standards: Reproducible Cleaning Pipelines

Production cleaning workflows must be:
- **Auditable**: Track every cleaning decision
- **Reproducible**: Same results every time
- **Configurable**: Easy to modify rules
- **Scalable**: Handle growing datasets
- **Documented**: Clear reasoning for all steps

### Advanced Cleaning Framework

#### 1. Rule-Based Cleaning Engine

```python
from abc import ABC, abstractmethod
from enum import Enum
import json
from datetime import datetime

class CleaningAction(Enum):
    REMOVE = "remove"
    IMPUTE = "impute"  
    TRANSFORM = "transform"
    FLAG = "flag"
    CORRECT = "correct"

class CleaningRule(ABC):
    def __init__(self, rule_id: str, description: str):
        self.rule_id = rule_id
        self.description = description
        self.applied_count = 0
    
    @abstractmethod
    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        """Apply cleaning rule and return modified df and audit log entries."""
        pass

class OutlierRemovalRule(CleaningRule):
    def __init__(self, column: str, method: str = "iqr", multiplier: float = 1.5):
        super().__init__(
            rule_id=f"outlier_{column}_{method}",
            description=f"Remove outliers from {column} using {method} method"
        )
        self.column = column
        self.method = method
        self.multiplier = multiplier
    
    def apply(self, df: pd.DataFrame) -> tuple[pd.DataFrame, list]:
        audit_entries = []
        
        if self.column not in df.columns:
            return df, audit_entries
        
        original_count = len(df)
        
        if self.method == "iqr":
            Q1 = df[self.column].quantile(0.25)
            Q3 = df[self.column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - self.multiplier * IQR
            upper_bound = Q3 + self.multiplier * IQR
            
            # Create mask for outliers
            outlier_mask = (df[self.column] < lower_bound) | (df[self.column] > upper_bound)
            outlier_indices = df[outlier_mask].index.tolist()
            
            # Remove outliers
            df_cleaned = df[~outlier_mask].copy()
            
            removed_count = len(outlier_indices)
            self.applied_count += removed_count
            
            audit_entries.append({
                'rule_id': self.rule_id,
                'action': CleaningAction.REMOVE.value,
                'column': self.column,
                'original_count': original_count,
                'removed_count': removed_count,
                'removed_indices': outlier_indices,
                'bounds': {'lower': lower_bound, 'upper': upper_bound},
                'timestamp': datetime.now().isoformat()
            })
            
            return df_cleaned, audit_entries
        
        return df, audit_entries

class DataCleaningPipeline:
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
        self.rules = []
        self.audit_log = []
        self.execution_history = []
    
    def add_rule(self, rule: CleaningRule):
        """Add cleaning rule to pipeline."""
        self.rules.append(rule)
    
    def execute(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """Execute all cleaning rules and return results with audit trail."""
        
        execution_start = datetime.now()
        df_working = df.copy()
        execution_audit = []
        
        for rule in self.rules:
            rule_start = datetime.now()
            
            df_working, rule_audit = rule.apply(df_working)
            execution_audit.extend(rule_audit)
            
            rule_duration = (datetime.now() - rule_start).total_seconds()
            
            print(f"✓ Applied {rule.rule_id}: "
                  f"{rule.applied_count} changes in {rule_duration:.2f}s")
        
        # Create execution summary
        execution_summary = {
            'pipeline_name': self.pipeline_name,
            'execution_start': execution_start.isoformat(),
            'execution_duration': (datetime.now() - execution_start).total_seconds(),
            'original_shape': df.shape,
            'final_shape': df_working.shape,
            'rules_applied': len(self.rules),
            'total_changes': sum(rule.applied_count for rule in self.rules),
            'audit_entries': execution_audit
        }
        
        self.audit_log.extend(execution_audit)
        self.execution_history.append(execution_summary)
        
        return df_working, execution_summary
```

### Clinical Research Integration: Regulatory Compliance

```python
class ClinicalDataCleaner(DataCleaningPipeline):
    """Specialized cleaning pipeline for clinical research data."""
    
    def __init__(self, study_id: str, protocol_version: str):
        super().__init__(f"clinical_study_{study_id}")
        self.study_id = study_id
        self.protocol_version = protocol_version
        self.setup_clinical_rules()
    
    def setup_clinical_rules(self):
        """Setup standard clinical data cleaning rules."""
        
        # Age validation (must be reasonable for study)
        self.add_rule(RangeValidationRule(
            'age', min_val=18, max_val=120,
            rule_id='age_validation'
        ))
        
        # Date consistency (visit dates must be after enrollment)
        self.add_rule(DateConsistencyRule(
            date_col='visit_date', 
            reference_col='enrollment_date',
            rule_id='visit_date_consistency'
        ))
        
        # Lab value validation (physiologically plausible ranges)
        self.add_rule(LabValueValidationRule(
            lab_ranges_config='configs/lab_reference_ranges.json'
        ))
    
    def generate_regulatory_report(self) -> str:
        """Generate cleaning report meeting regulatory standards."""
        
        report = f"""
# Data Cleaning Report - Clinical Study {self.study_id}

**Protocol Version**: {self.protocol_version}
**Generation Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

- **Total Executions**: {len(self.execution_history)}
- **Total Records Processed**: {sum(exec['original_shape'][0] for exec in self.execution_history)}
- **Total Cleaning Actions**: {sum(exec['total_changes'] for exec in self.execution_history)}

## Cleaning Rules Applied

"""
        
        for rule in self.rules:
            report += f"- **{rule.rule_id}**: {rule.description} (Applied {rule.applied_count} times)\n"
        
        report += "\n## Detailed Audit Trail\n\n"
        
        for entry in self.audit_log[-10:]:  # Last 10 entries
            report += f"- {entry['timestamp']}: {entry['rule_id']} - {entry['action']} - {entry.get('removed_count', 0)} records\n"
        
        return report
```

### Final Exercise: Production Pipeline

**Scenario**: Build a complete data cleaning pipeline for a cardiovascular outcomes registry.

**Requirements**:
- Handle 50,000+ patient records
- Apply clinical validation rules
- Generate regulatory-compliant audit trails  
- Create automated quality reports
- Support configurable cleaning thresholds

---

## Assessment and Next Steps

### Project-Based Assessment

**Mini-Project**: Multi-Source Clinical Data Integration

You'll work with simulated datasets representing:
1. EMR demographics and encounters
2. Laboratory results over time  
3. Medication dispensing records
4. Clinical outcomes

**Deliverables**:
1. **Integration Pipeline**: Load and merge all sources
2. **Quality Assessment**: Comprehensive data quality report
3. **Cleaning Workflow**: Automated cleaning with audit trail
4. **Performance Analysis**: Memory and time optimization
5. **Documentation**: Professional-grade documentation

### Professional Skills Checklist

After this lecture, you should be able to:

- [ ] Design schema-aware data loading systems
- [ ] Implement parallel processing for large datasets  
- [ ] Build comprehensive data quality assessment frameworks
- [ ] Create production-ready cleaning pipelines with audit trails
- [ ] Optimize memory usage for large-scale data processing
- [ ] Generate regulatory-compliant documentation

### Looking Ahead to L07

In our next lecture, we'll build on these data loading and cleaning foundations to master:
- **Complex data transformations** for analysis-ready datasets
- **Hierarchical data structures** and multi-level indexing
- **Publication-quality visualization** with statistical graphics
- **Advanced wrangling techniques** for time series and longitudinal data

The professional competencies you've developed today in data pipeline creation will be essential as we move toward sophisticated analysis workflows.

---

## Additional Resources

- **Clinical Data Standards**: CDISC, HL7 FHIR
- **Data Quality Frameworks**: DAMA-DMBOK, ISO 25012
- **Performance Optimization**: Dask documentation, pandas performance tips
- **Regulatory Compliance**: ICH-GCP guidelines for clinical data