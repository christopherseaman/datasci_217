# Research Applications and Best Practices

Welcome to the final week of DataSci 217! You've built a comprehensive toolkit of data science skills, from command-line fundamentals to advanced visualization. Now it's time to integrate everything into professional research workflows, understand ethical considerations, and plan your continued growth in data science.

By the end of today, you'll understand how to apply your skills in research contexts, create reproducible analysis workflows, and have a clear path for your data science career development.

*[xkcd 435: "Purity" - Shows fields arranged by mathematical purity, from sociology at the bottom through psychology, biology, chemistry, physics, math, with the ultimate pure mathematicians looking down at everyone saying "Oh, that's nice" to physicists.]*

Don't worry - as data scientists, we get to work with everyone and solve real problems!

# Reproducible Research Principles

## The Reproducibility Crisis and Solution

**Why Reproducible Research Matters:**
- **Scientific integrity** - Others can verify and build on your work
- **Career advancement** - Reproducible work demonstrates professional competence
- **Time savings** - Future you will understand past you's decisions
- **Collaboration** - Teams can work together effectively on complex projects
- **Error detection** - Systematic approaches catch mistakes early

### Core Reproducibility Principles

**Reference:**
```python
# File: analysis_template.py
"""
Reproducible Analysis Template for Research Projects

This template follows reproducible research best practices:
1. Clear directory structure
2. Version control integration
3. Environment documentation
4. Parameterized analysis
5. Comprehensive documentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import logging
import json
import argparse

# Configure logging for reproducibility
def setup_logging(log_level='INFO'):
    """
    Setup comprehensive logging for analysis tracking
    """
    log_format = '%(asctime)s - %(levelname)s - %(module)s:%(lineno)d - %(message)s'
    
    # Create logs directory
    Path('logs').mkdir(exist_ok=True)
    
    # Setup file and console logging
    logging.basicConfig(
        level=getattr(logging, log_level),
        format=log_format,
        handlers=[
            logging.FileHandler(f'logs/analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Analysis session started")
    logger.info(f"Working directory: {Path.cwd()}")
    logger.info(f"Python packages: pandas {pd.__version__}, numpy {np.__version__}")
    
    return logger

class ReproducibleAnalysis:
    """
    Framework for reproducible data analysis
    """
    
    def __init__(self, project_name, config_file=None):
        self.project_name = project_name
        self.logger = setup_logging()
        self.results = {}
        self.metadata = {
            'project_name': project_name,
            'start_time': datetime.now().isoformat(),
            'analysis_steps': [],
            'parameters_used': {},
            'data_sources': [],
            'output_files': []
        }
        
        # Load configuration if provided
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                self.config = json.load(f)
            self.logger.info(f"Configuration loaded from {config_file}")
        else:
            self.config = self._default_config()
            self.logger.info("Using default configuration")
    
    def _default_config(self):
        """Default configuration for analysis"""
        return {
            'data': {
                'input_directory': 'data/raw',
                'output_directory': 'data/processed',
                'file_format': 'csv'
            },
            'analysis': {
                'significance_level': 0.05,
                'random_seed': 42,
                'missing_data_threshold': 0.1
            },
            'output': {
                'figures_directory': 'figures',
                'tables_directory': 'tables',
                'reports_directory': 'reports'
            }
        }
    
    def log_step(self, step_name, parameters=None, data_shape=None):
        """
        Log analysis step with metadata
        """
        step_info = {
            'step': step_name,
            'timestamp': datetime.now().isoformat(),
            'parameters': parameters or {},
            'data_shape': data_shape
        }
        
        self.metadata['analysis_steps'].append(step_info)
        if parameters:
            self.metadata['parameters_used'].update(parameters)
        
        self.logger.info(f"Step completed: {step_name}")
        if data_shape:
            self.logger.info(f"Data shape: {data_shape}")
    
    def load_data(self, data_path, validation_rules=None):
        """
        Load and validate data with comprehensive logging
        """
        self.logger.info(f"Loading data from {data_path}")
        
        try:
            if Path(data_path).suffix.lower() == '.csv':
                df = pd.read_csv(data_path)
            elif Path(data_path).suffix.lower() == '.xlsx':
                df = pd.read_excel(data_path)
            else:
                raise ValueError(f"Unsupported file format: {Path(data_path).suffix}")
            
            # Record data source
            self.metadata['data_sources'].append({
                'path': str(data_path),
                'shape': df.shape,
                'columns': list(df.columns),
                'load_time': datetime.now().isoformat()
            })
            
            # Basic validation
            if validation_rules:
                self._validate_data(df, validation_rules)
            
            self.log_step('data_loading', 
                         {'source': str(data_path)}, 
                         df.shape)
            
            return df
            
        except Exception as e:
            self.logger.error(f"Data loading failed: {str(e)}")
            raise
    
    def _validate_data(self, df, rules):
        """
        Validate data against specified rules
        """
        self.logger.info("Validating data quality")
        
        for rule_name, rule_func in rules.items():
            try:
                rule_result = rule_func(df)
                if not rule_result:
                    self.logger.warning(f"Data validation failed: {rule_name}")
                else:
                    self.logger.info(f"Data validation passed: {rule_name}")
            except Exception as e:
                self.logger.error(f"Data validation error in {rule_name}: {str(e)}")
    
    def save_results(self, results, filename, file_type='csv'):
        """
        Save analysis results with metadata
        """
        # Create output directory
        output_dir = Path(self.config['output']['tables_directory'])
        output_dir.mkdir(parents=True, exist_ok=True)
        
        output_path = output_dir / filename
        
        try:
            if file_type == 'csv' and hasattr(results, 'to_csv'):
                results.to_csv(output_path, index=False)
            elif file_type == 'excel' and hasattr(results, 'to_excel'):
                results.to_excel(output_path, index=False)
            elif file_type == 'json':
                with open(output_path, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
            
            self.metadata['output_files'].append({
                'filename': filename,
                'path': str(output_path),
                'type': file_type,
                'created': datetime.now().isoformat()
            })
            
            self.logger.info(f"Results saved: {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save results: {str(e)}")
            raise
    
    def create_analysis_report(self):
        """
        Generate comprehensive analysis report
        """
        # Finalize metadata
        self.metadata['end_time'] = datetime.now().isoformat()
        self.metadata['total_steps'] = len(self.metadata['analysis_steps'])
        
        # Save metadata
        metadata_path = Path(self.config['output']['reports_directory']) / 'analysis_metadata.json'
        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2, default=str)
        
        self.logger.info(f"Analysis metadata saved: {metadata_path}")
        
        # Generate human-readable report
        report_path = Path(self.config['output']['reports_directory']) / 'analysis_report.md'
        
        with open(report_path, 'w') as f:
            f.write(f"# {self.project_name} - Analysis Report\n\n")
            f.write(f"**Generated:** {self.metadata['end_time']}\n")
            f.write(f"**Duration:** {self.metadata['start_time']} to {self.metadata['end_time']}\n\n")
            
            f.write("## Data Sources\n\n")
            for source in self.metadata['data_sources']:
                f.write(f"- **{source['path']}** ({source['shape'][0]} rows × {source['shape'][1]} columns)\n")
            
            f.write("\n## Analysis Steps\n\n")
            for i, step in enumerate(self.metadata['analysis_steps'], 1):
                f.write(f"{i}. **{step['step']}** - {step['timestamp']}\n")
                if step.get('data_shape'):
                    f.write(f"   - Data shape: {step['data_shape']}\n")
                if step.get('parameters'):
                    f.write(f"   - Parameters: {step['parameters']}\n")
            
            f.write("\n## Output Files\n\n")
            for output_file in self.metadata['output_files']:
                f.write(f"- **{output_file['filename']}** ({output_file['type']}) - {output_file['created']}\n")
            
            f.write("\n## Parameters Used\n\n")
            for param, value in self.metadata['parameters_used'].items():
                f.write(f"- **{param}:** {value}\n")
        
        self.logger.info(f"Analysis report generated: {report_path}")
        
        return metadata_path, report_path

# Example usage
def example_reproducible_analysis():
    """
    Example of reproducible analysis workflow
    """
    # Initialize analysis framework
    analysis = ReproducibleAnalysis("Customer_Satisfaction_Study")
    
    # Define data validation rules
    validation_rules = {
        'has_required_columns': lambda df: all(col in df.columns for col in ['customer_id', 'satisfaction_score']),
        'no_empty_dataframe': lambda df: not df.empty,
        'reasonable_satisfaction_range': lambda df: df['satisfaction_score'].between(1, 10).all() if 'satisfaction_score' in df.columns else True
    }
    
    # Load and validate data
    df = analysis.load_data('data/customer_satisfaction.csv', validation_rules)
    
    # Analysis step 1: Basic statistics
    basic_stats = df.describe()
    analysis.log_step('descriptive_statistics', 
                     {'metrics': 'mean, std, min, max, quartiles'})
    analysis.save_results(basic_stats, 'basic_statistics.csv')
    
    # Analysis step 2: Group analysis
    if 'customer_segment' in df.columns:
        segment_analysis = df.groupby('customer_segment')['satisfaction_score'].agg(['mean', 'std', 'count'])
        analysis.log_step('segment_analysis', 
                         {'grouping_variable': 'customer_segment'})
        analysis.save_results(segment_analysis, 'segment_analysis.csv')
    
    # Generate final report
    metadata_file, report_file = analysis.create_analysis_report()
    
    print(f"Analysis complete!")
    print(f"- Metadata: {metadata_file}")
    print(f"- Report: {report_file}")
    
    return analysis

# Uncomment to run example
# example_analysis = example_reproducible_analysis()
```

## Data Management Best Practices

### Research Data Organization

**Reference:**
```python
# File: research_data_manager.py
"""
Research Data Management System

Implements best practices for research data organization:
- Standardized directory structure
- Data versioning and backup
- Metadata management
- Access control and sharing
"""

from pathlib import Path
import shutil
import hashlib
import json
from datetime import datetime
import pandas as pd

class ResearchDataManager:
    """
    Professional research data management system
    """
    
    def __init__(self, project_root):
        self.project_root = Path(project_root)
        self.setup_directory_structure()
        
    def setup_directory_structure(self):
        """
        Create standardized research directory structure
        """
        directories = [
            'data/raw',           # Original, unmodified data
            'data/processed',     # Cleaned and processed data
            'data/interim',       # Intermediate processing steps
            'data/external',      # External reference data
            'notebooks',          # Jupyter notebooks for exploration
            'src',               # Source code modules
            'reports',           # Generated reports and papers
            'figures',           # Publication-ready figures
            'references',        # Literature and documentation
            'logs',              # Analysis logs and metadata
            'config',            # Configuration files
            'tests',             # Unit tests for code validation
            'environments'       # Environment specifications
        ]
        
        for directory in directories:
            (self.project_root / directory).mkdir(parents=True, exist_ok=True)
        
        # Create README files for key directories
        self._create_readme_files()
    
    def _create_readme_files(self):
        """
        Create README files explaining directory purposes
        """
        readme_content = {
            'data/raw': """# Raw Data Directory

This directory contains original, unmodified data files.
- Never edit files in this directory
- Document data sources in metadata files
- Use descriptive filenames with dates
- Keep original data separate from processed versions

File naming convention: YYYY-MM-DD_descriptive-name_version.ext
""",
            'data/processed': """# Processed Data Directory

This directory contains cleaned and processed datasets ready for analysis.
- All processing steps should be documented
- Include processing scripts and parameters used
- Version processed datasets appropriately
- Maintain processing logs for reproducibility

File naming convention: YYYY-MM-DD_dataset-name_processed_v##.ext
""",
            'src': """# Source Code Directory

This directory contains reusable analysis modules and functions.
- Write modular, well-documented code
- Include unit tests for all functions
- Follow consistent coding standards
- Use version control for all changes

Organization:
- data/: Data loading and preprocessing modules
- analysis/: Statistical analysis functions
- visualization/: Plotting and figure generation
- utils/: Utility functions and helpers
""",
            'notebooks': """# Notebooks Directory

This directory contains Jupyter notebooks for data exploration and analysis.
- Use descriptive names indicating purpose and date
- Clear markdown documentation throughout
- Execute notebooks from start to finish before saving
- Export important results to appropriate directories

Naming convention: ##_YYYY-MM-DD_descriptive-purpose.ipynb
""",
            'reports': """# Reports Directory

This directory contains generated reports, papers, and presentations.
- Include both source files and generated outputs
- Version control manuscript drafts
- Document report generation process
- Include supplementary materials

Organization:
- papers/: Academic papers and manuscripts
- presentations/: Slide decks and talk materials
- technical/: Technical reports and documentation
"""
        }
        
        for directory, content in readme_content.items():
            readme_path = self.project_root / directory / 'README.md'
            with open(readme_path, 'w') as f:
                f.write(content)
    
    def register_dataset(self, file_path, dataset_type='raw', description='', source=''):
        """
        Register a new dataset with metadata tracking
        """
        file_path = Path(file_path)
        
        # Calculate file hash for integrity checking
        file_hash = self._calculate_file_hash(file_path)
        
        # Create metadata record
        metadata = {
            'filename': file_path.name,
            'path': str(file_path.relative_to(self.project_root)),
            'type': dataset_type,
            'description': description,
            'source': source,
            'registered_date': datetime.now().isoformat(),
            'file_size_bytes': file_path.stat().st_size,
            'file_hash': file_hash,
            'last_modified': datetime.fromtimestamp(file_path.stat().st_mtime).isoformat()
        }
        
        # If it's a data file, add basic statistics
        if file_path.suffix.lower() in ['.csv', '.xlsx', '.json']:
            try:
                if file_path.suffix.lower() == '.csv':
                    df = pd.read_csv(file_path)
                elif file_path.suffix.lower() == '.xlsx':
                    df = pd.read_excel(file_path)
                
                metadata['data_stats'] = {
                    'rows': len(df),
                    'columns': len(df.columns),
                    'column_names': list(df.columns),
                    'dtypes': {col: str(dtype) for col, dtype in df.dtypes.items()},
                    'missing_values': df.isnull().sum().to_dict()
                }
            except Exception as e:
                metadata['data_stats'] = {'error': f'Could not read data: {str(e)}'}
        
        # Save metadata
        metadata_path = self.project_root / 'logs' / 'data_registry.json'
        
        # Load existing registry or create new
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                registry = json.load(f)
        else:
            registry = {'datasets': []}
        
        # Add new dataset (or update if exists)
        registry['datasets'] = [d for d in registry['datasets'] if d['path'] != metadata['path']]
        registry['datasets'].append(metadata)
        registry['last_updated'] = datetime.now().isoformat()
        
        # Save updated registry
        with open(metadata_path, 'w') as f:
            json.dump(registry, f, indent=2)
        
        print(f"Dataset registered: {file_path.name}")
        return metadata
    
    def _calculate_file_hash(self, file_path):
        """
        Calculate SHA-256 hash of file for integrity checking
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    
    def backup_data(self, backup_location=None):
        """
        Create backup of all data files
        """
        if backup_location is None:
            backup_location = self.project_root.parent / f"{self.project_root.name}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        backup_path = Path(backup_location)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        # Copy data directories
        data_dirs = ['data', 'config', 'logs']
        
        for data_dir in data_dirs:
            source_dir = self.project_root / data_dir
            if source_dir.exists():
                dest_dir = backup_path / data_dir
                shutil.copytree(source_dir, dest_dir, exist_ok=True)
        
        # Create backup manifest
        manifest = {
            'backup_date': datetime.now().isoformat(),
            'source_project': str(self.project_root),
            'backup_location': str(backup_path),
            'directories_backed_up': data_dirs
        }
        
        with open(backup_path / 'backup_manifest.json', 'w') as f:
            json.dump(manifest, f, indent=2)
        
        print(f"Data backup created: {backup_path}")
        return backup_path
    
    def create_data_dictionary(self, dataset_path, output_path=None):
        """
        Generate comprehensive data dictionary for a dataset
        """
        dataset_path = Path(dataset_path)
        
        if output_path is None:
            output_path = self.project_root / 'references' / f"{dataset_path.stem}_data_dictionary.md"
        
        try:
            # Load data
            if dataset_path.suffix.lower() == '.csv':
                df = pd.read_csv(dataset_path)
            elif dataset_path.suffix.lower() == '.xlsx':
                df = pd.read_excel(dataset_path)
            else:
                raise ValueError(f"Unsupported file format: {dataset_path.suffix}")
            
            # Generate data dictionary
            with open(output_path, 'w') as f:
                f.write(f"# Data Dictionary: {dataset_path.name}\n\n")
                f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"**Source:** {dataset_path}\n")
                f.write(f"**Records:** {len(df):,}\n")
                f.write(f"**Variables:** {len(df.columns)}\n\n")
                
                f.write("## Dataset Overview\n\n")
                f.write("| Attribute | Value |\n")
                f.write("|-----------|-------|\n")
                f.write(f"| File Size | {dataset_path.stat().st_size / 1024**2:.2f} MB |\n")
                f.write(f"| Last Modified | {datetime.fromtimestamp(dataset_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')} |\n")
                f.write(f"| Missing Values | {df.isnull().sum().sum():,} ({df.isnull().sum().sum() / (len(df) * len(df.columns)) * 100:.2f}%) |\n\n")
                
                f.write("## Variable Descriptions\n\n")
                f.write("| Variable | Type | Non-Null | Unique | Description | Example Values |\n")
                f.write("|----------|------|----------|--------|--------------|--------------|\n")
                
                for col in df.columns:
                    dtype = str(df[col].dtype)
                    non_null = df[col].count()
                    unique = df[col].nunique()
                    
                    # Get example values
                    if df[col].dtype == 'object':
                        examples = df[col].dropna().head(3).tolist()
                    else:
                        examples = df[col].dropna().head(3).round(2).tolist()
                    
                    examples_str = ', '.join(str(ex) for ex in examples)
                    if len(examples_str) > 50:
                        examples_str = examples_str[:47] + '...'
                    
                    f.write(f"| `{col}` | {dtype} | {non_null:,} | {unique:,} | *[Add description]* | {examples_str} |\n")
                
                f.write("\n## Statistical Summary\n\n")
                
                # Numeric variables summary
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    f.write("### Numeric Variables\n\n")
                    stats_df = df[numeric_cols].describe()
                    f.write(stats_df.round(2).to_markdown())
                    f.write("\n\n")
                
                # Categorical variables summary
                categorical_cols = df.select_dtypes(include=['object']).columns
                if len(categorical_cols) > 0:
                    f.write("### Categorical Variables\n\n")
                    for col in categorical_cols:
                        f.write(f"**{col}**\n\n")
                        value_counts = df[col].value_counts().head()
                        f.write(value_counts.to_markdown())
                        f.write("\n\n")
            
            print(f"Data dictionary created: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error creating data dictionary: {str(e)}")
            return None

# Example usage
def setup_research_project(project_name):
    """
    Setup a new research project with proper data management
    """
    project_path = Path(f"research_projects/{project_name}")
    
    # Initialize data manager
    data_manager = ResearchDataManager(project_path)
    
    # Create project configuration
    config = {
        'project': {
            'name': project_name,
            'created': datetime.now().isoformat(),
            'description': f'Research project: {project_name}',
            'investigators': ['Add investigator names']
        },
        'analysis': {
            'significance_level': 0.05,
            'random_seed': 42,
            'bootstrap_iterations': 1000
        },
        'data_management': {
            'backup_frequency': 'weekly',
            'retention_period': '7 years',
            'sharing_restrictions': 'IRB approval required'
        }
    }
    
    config_path = project_path / 'config' / 'project_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Research project '{project_name}' initialized at {project_path}")
    return data_manager, config_path

# Example usage
# data_manager, config = setup_research_project("Customer_Satisfaction_Analysis")
```

# Clinical and Research Context

## Research Data Types and Challenges

### Common Research Data Scenarios

**Reference:**
```python
def handle_research_data_challenges():
    """
    Demonstrate handling common research data challenges
    """
    print("=== COMMON RESEARCH DATA CHALLENGES ===")
    
    # Challenge 1: Missing data patterns in longitudinal studies
    print("\n1. LONGITUDINAL MISSING DATA")
    
    # Simulate longitudinal study data
    np.random.seed(42)
    participants = 100
    timepoints = 6
    
    # Create participant data with realistic missing patterns
    data = []
    for participant in range(participants):
        for timepoint in range(timepoints):
            # Simulate dropout - higher probability of missing at later timepoints
            if np.random.random() > (0.95 - timepoint * 0.1):
                continue  # Missing data point
            
            data.append({
                'participant_id': f'P{participant:03d}',
                'timepoint': timepoint,
                'outcome_score': np.random.normal(50 + timepoint * 2, 10),
                'age_at_baseline': np.random.randint(18, 80),
                'treatment_group': np.random.choice(['Control', 'Treatment'])
            })
    
    longitudinal_df = pd.DataFrame(data)
    
    print(f"Longitudinal dataset: {len(longitudinal_df)} observations")
    print(f"Unique participants: {longitudinal_df['participant_id'].nunique()}")
    print("Missing data pattern by timepoint:")
    
    missing_pattern = longitudinal_df.groupby('timepoint').size()
    for tp, count in missing_pattern.items():
        print(f"  Timepoint {tp}: {count} participants ({count/participants*100:.1f}%)")
    
    # Challenge 2: Multi-site data harmonization
    print("\n2. MULTI-SITE DATA HARMONIZATION")
    
    # Simulate data from different sites with slightly different protocols
    sites = ['Site_A', 'Site_B', 'Site_C']
    harmonized_data = []
    
    for site in sites:
        n_participants = np.random.randint(50, 150)
        
        for i in range(n_participants):
            record = {
                'site': site,
                'participant_id': f'{site}_{i:03d}',
                'age': np.random.normal(45, 15),
                'primary_outcome': np.random.normal(100, 20)
            }
            
            # Site-specific variations
            if site == 'Site_A':
                record['measurement_device'] = 'Device_X'
                record['primary_outcome'] += 5  # Systematic bias
            elif site == 'Site_B':
                record['measurement_device'] = 'Device_Y'
                record['secondary_measure'] = np.random.normal(50, 10)  # Extra measure
            else:  # Site_C
                record['measurement_device'] = 'Device_Z'
                record['age_category'] = 'Adult' if record['age'] >= 18 else 'Minor'
        
        harmonized_data.extend([record])
    
    multisite_df = pd.DataFrame(harmonized_data)
    
    print("Multi-site dataset characteristics:")
    print(multisite_df.groupby('site').agg({
        'participant_id': 'count',
        'age': 'mean',
        'primary_outcome': 'mean'
    }).round(2))
    
    # Challenge 3: Sensitive data handling
    print("\n3. SENSITIVE DATA PROTECTION")
    
    def create_deidentified_dataset(df, quasi_identifiers=['age'], direct_identifiers=['participant_id']):
        """
        Demonstrate basic de-identification techniques
        """
        df_deidentified = df.copy()
        
        # Remove direct identifiers
        for col in direct_identifiers:
            if col in df_deidentified.columns:
                df_deidentified[col] = df_deidentified[col].apply(lambda x: f'ID_{hash(str(x)) % 10000:04d}')
        
        # Generalize quasi-identifiers
        for col in quasi_identifiers:
            if col in df_deidentified.columns:
                if col == 'age':
                    # Age generalization to 5-year bins
                    df_deidentified[col] = (df_deidentified[col] // 5) * 5
        
        return df_deidentified
    
    deidentified_df = create_deidentified_dataset(multisite_df)
    print("Sample de-identified data:")
    print(deidentified_df.head())
    
    return longitudinal_df, multisite_df, deidentified_df

# Run the demonstration
longitudinal_data, multisite_data, deidentified_data = handle_research_data_challenges()
```

### Ethical Considerations in Data Analysis

**Reference:**
```python
class EthicalDataAnalysis:
    """
    Framework for ensuring ethical considerations in data analysis
    """
    
    def __init__(self):
        self.ethical_checklist = {
            'consent_and_privacy': [
                'Data collection had appropriate consent',
                'Personal identifiers are properly protected',
                'Data sharing agreements are followed',
                'Participant privacy is maintained'
            ],
            'bias_and_fairness': [
                'Analysis methods don\'t discriminate against protected groups',
                'Sample is representative of target population',
                'Historical biases in data are acknowledged',
                'Results are interpreted fairly across groups'
            ],
            'transparency_and_reproducibility': [
                'Analysis methods are fully documented',
                'Code and data are available for verification',
                'Limitations and assumptions are clearly stated',
                'Conflicts of interest are disclosed'
            ],
            'beneficence_and_non_maleficence': [
                'Research aims to benefit participants/society',
                'Potential harms are minimized',
                'Results won\'t be used to harm individuals',
                'Scientific integrity is maintained'
            ]
        }
    
    def conduct_ethical_review(self, analysis_plan):
        """
        Review analysis plan for ethical considerations
        """
        print("=== ETHICAL ANALYSIS REVIEW ===")
        
        review_results = {}
        
        for category, criteria in self.ethical_checklist.items():
            print(f"\n{category.upper().replace('_', ' ')}:")
            category_results = []
            
            for criterion in criteria:
                # In practice, this would involve actual review
                # Here we demonstrate the process
                print(f"  □ {criterion}")
                print(f"    Status: [Requires reviewer assessment]")
                print(f"    Notes: [Document compliance or concerns]")
                category_results.append({
                    'criterion': criterion,
                    'status': 'pending_review',
                    'notes': ''
                })
            
            review_results[category] = category_results
        
        return review_results
    
    def create_data_protection_protocol(self, data_types):
        """
        Create data protection protocol based on data types
        """
        protocol = {
            'data_classification': {},
            'access_controls': {},
            'storage_requirements': {},
            'sharing_restrictions': {}
        }
        
        for data_type in data_types:
            if 'personal' in data_type.lower() or 'identifiable' in data_type.lower():
                protocol['data_classification'][data_type] = 'Highly Sensitive'
                protocol['access_controls'][data_type] = 'Restricted - Named individuals only'
                protocol['storage_requirements'][data_type] = 'Encrypted storage required'
                protocol['sharing_restrictions'][data_type] = 'IRB approval required'
            
            elif 'health' in data_type.lower() or 'medical' in data_type.lower():
                protocol['data_classification'][data_type] = 'Sensitive'
                protocol['access_controls'][data_type] = 'Role-based access'
                protocol['storage_requirements'][data_type] = 'Secure storage required'
                protocol['sharing_restrictions'][data_type] = 'Data use agreement required'
            
            else:
                protocol['data_classification'][data_type] = 'Standard'
                protocol['access_controls'][data_type] = 'Project team access'
                protocol['storage_requirements'][data_type] = 'Standard security'
                protocol['sharing_restrictions'][data_type] = 'Follow institutional policy'
        
        return protocol

# Example usage
ethical_framework = EthicalDataAnalysis()

# Review an analysis plan
analysis_plan = {
    'study_type': 'observational',
    'data_sources': ['survey responses', 'administrative records'],
    'population': 'adult patients',
    'outcomes': ['treatment effectiveness', 'cost analysis']
}

review_results = ethical_framework.conduct_ethical_review(analysis_plan)

# Create data protection protocol
data_types = ['Personal Health Information', 'Survey Responses', 'Administrative Data']
protection_protocol = ethical_framework.create_data_protection_protocol(data_types)

print("\n=== DATA PROTECTION PROTOCOL ===")
for category, details in protection_protocol.items():
    print(f"\n{category.upper().replace('_', ' ')}:")
    for data_type, requirement in details.items():
        print(f"  {data_type}: {requirement}")
```

## Collaboration Patterns in Research

### Team-Based Research Workflows

**Reference:**
```python
def demonstrate_collaborative_workflows():
    """
    Demonstrate best practices for collaborative research
    """
    print("=== COLLABORATIVE RESEARCH WORKFLOWS ===")
    
    # 1. Role definitions and responsibilities
    team_roles = {
        'Principal Investigator': [
            'Overall project oversight',
            'Scientific direction',
            'Ethical compliance',
            'Manuscript preparation'
        ],
        'Data Analyst': [
            'Statistical analysis plan',
            'Data cleaning and processing',
            'Analysis implementation',
            'Results interpretation'
        ],
        'Data Manager': [
            'Database design and management',
            'Data quality assurance',
            'Documentation maintenance',
            'Backup and security'
        ],
        'Research Coordinator': [
            'Study coordination',
            'Timeline management',
            'Communication facilitation',
            'Administrative support'
        ],
        'Domain Expert': [
            'Scientific expertise',
            'Results interpretation',
            'Clinical relevance assessment',
            'Manuscript review'
        ]
    }
    
    print("RESEARCH TEAM ROLES AND RESPONSIBILITIES:")
    for role, responsibilities in team_roles.items():
        print(f"\n{role}:")
        for responsibility in responsibilities:
            print(f"  • {responsibility}")
    
    # 2. Communication protocols
    communication_framework = {
        'Daily': [
            'Progress updates in shared channel',
            'Issue reporting and resolution',
            'Quick questions and clarifications'
        ],
        'Weekly': [
            'Team status meetings',
            'Analysis results review',
            'Timeline and milestone check'
        ],
        'Monthly': [
            'Comprehensive progress review',
            'Stakeholder updates',
            'Strategic planning discussions'
        ],
        'Milestone-based': [
            'Data collection completion',
            'Analysis plan finalization',
            'Results presentation',
            'Manuscript submission'
        ]
    }
    
    print("\n\nCOMMUNICATION PROTOCOLS:")
    for frequency, activities in communication_framework.items():
        print(f"\n{frequency}:")
        for activity in activities:
            print(f"  • {activity}")
    
    # 3. Version control for research
    version_control_strategy = {
        'Code': {
            'tool': 'Git with GitHub/GitLab',
            'structure': 'Feature branches for each analysis',
            'review_process': 'Pull request with peer review',
            'documentation': 'Comprehensive commit messages'
        },
        'Data': {
            'tool': 'DVC (Data Version Control) or similar',
            'structure': 'Versioned datasets with metadata',
            'review_process': 'Data quality checks before commit',
            'documentation': 'Data dictionaries and provenance'
        },
        'Manuscripts': {
            'tool': 'Git + LaTeX or collaborative platforms',
            'structure': 'Chapter/section based organization',
            'review_process': 'Comment and track changes',
            'documentation': 'Version history and change logs'
        },
        'Analysis Results': {
            'tool': 'Version controlled with automated timestamps',
            'structure': 'Organized by analysis phase',
            'review_process': 'Reproducibility verification',
            'documentation': 'Analysis metadata and parameters'
        }
    }
    
    print("\n\nVERSION CONTROL STRATEGY:")
    for component, strategy in version_control_strategy.items():
        print(f"\n{component}:")
        for aspect, approach in strategy.items():
            print(f"  {aspect}: {approach}")

# Example implementation
def create_collaboration_template(project_name):
    """
    Create template structure for collaborative research project
    """
    project_path = Path(f"collaborative_projects/{project_name}")
    
    # Create directory structure
    directories = [
        'team/roles_responsibilities',
        'team/communication_logs',
        'team/meeting_notes',
        'analysis/individual_work',
        'analysis/shared_code',
        'analysis/review_comments',
        'data/access_logs',
        'data/quality_reports',
        'documentation/protocols',
        'documentation/decisions',
        'outputs/drafts',
        'outputs/reviews'
    ]
    
    for directory in directories:
        (project_path / directory).mkdir(parents=True, exist_ok=True)
    
    # Create collaboration configuration
    collaboration_config = {
        'project_name': project_name,
        'created': datetime.now().isoformat(),
        'team_members': [],
        'communication_channels': {
            'daily_updates': 'Slack channel or equivalent',
            'weekly_meetings': 'Video conferencing platform',
            'document_sharing': 'Shared drive or repository'
        },
        'review_process': {
            'code_review': 'All analysis code requires peer review',
            'data_validation': 'Two-person verification of data processing',
            'results_verification': 'Independent reproduction of key findings'
        },
        'conflict_resolution': {
            'technical_disputes': 'Seek additional expert opinion',
            'authorship_questions': 'Follow institutional guidelines',
            'timeline_conflicts': 'Escalate to project leadership'
        }
    }
    
    with open(project_path / 'collaboration_config.json', 'w') as f:
        json.dump(collaboration_config, f, indent=2)
    
    # Create team charter template
    charter_template = f"""# {project_name} - Team Charter

## Project Overview
**Objective:** [Define primary research question and objectives]
**Timeline:** [Project start and end dates]
**Budget:** [If applicable]

## Team Members
| Name | Role | Responsibilities | Contact |
|------|------|-----------------|---------|
| [Name] | [Role] | [Key responsibilities] | [Email/Contact] |

## Communication Plan
- **Daily:** Progress updates via [platform]
- **Weekly:** Team meetings on [day/time]
- **Monthly:** Stakeholder updates and planning

## Decision-Making Process
- **Technical decisions:** Consensus among technical team members
- **Scientific decisions:** PI approval required
- **Budget/timeline decisions:** PI and coordinator consultation

## Conflict Resolution
1. Attempt direct resolution between involved parties
2. Involve project coordinator as mediator
3. Escalate to PI if necessary
4. Involve institutional resources if needed

## Quality Standards
- All analysis code must be peer-reviewed
- Data processing requires two-person verification
- Key findings must be independently reproducible

## Success Metrics
- [Define measurable project outcomes]
- [Timeline milestones]
- [Quality indicators]

---
*Charter created: {datetime.now().strftime('%Y-%m-%d')}*
*Next review: [Schedule regular charter reviews]*
"""
    
    with open(project_path / 'TEAM_CHARTER.md', 'w') as f:
        f.write(charter_template)
    
    print(f"Collaborative project template created: {project_path}")
    print("Key files created:")
    print(f"  • collaboration_config.json - Project configuration")
    print(f"  • TEAM_CHARTER.md - Team charter template")
    print(f"  • Directory structure for collaborative work")
    
    return project_path

# Run demonstrations
demonstrate_collaborative_workflows()
print("\n" + "="*60)
collaboration_template = create_collaboration_template("Multi_Site_Clinical_Study")
```

# LIVE DEMO!
*Building a complete reproducible research workflow: from project setup to final report with full documentation and version control*

# Career Development and Next Steps

## Skills Assessment and Growth Paths

### DataSci 217 Skills Inventory

**Reference:**
```python
def assess_data_science_skills():
    """
    Comprehensive skills assessment for DataSci 217 graduates
    """
    skills_framework = {
        'Technical Foundation': {
            'Command Line Proficiency': {
                'description': 'Navigate filesystem, manage files, run programs',
                'proficiency_levels': [
                    'Basic navigation and file operations',
                    'Text processing with grep, sed, awk',
                    'Shell scripting and automation',
                    'System administration tasks'
                ]
            },
            'Programming (Python)': {
                'description': 'Write clean, efficient Python code for data tasks',
                'proficiency_levels': [
                    'Basic syntax and data structures',
                    'Functions, modules, and error handling',
                    'Object-oriented programming concepts',
                    'Advanced patterns and optimization'
                ]
            },
            'Version Control (Git)': {
                'description': 'Track changes and collaborate on code',
                'proficiency_levels': [
                    'Basic add, commit, push workflow',
                    'Branching and merging',
                    'Collaborative workflows',
                    'Advanced Git operations and strategies'
                ]
            }
        },
        
        'Data Manipulation': {
            'pandas/NumPy': {
                'description': 'Efficiently work with structured data',
                'proficiency_levels': [
                    'Data loading and basic operations',
                    'Grouping, merging, and reshaping',
                    'Advanced indexing and performance optimization',
                    'Custom functions and memory management'
                ]
            },
            'Data Cleaning': {
                'description': 'Handle real-world messy data',
                'proficiency_levels': [
                    'Identify and handle missing values',
                    'Text processing and standardization',
                    'Outlier detection and treatment',
                    'Complex data quality frameworks'
                ]
            },
            'Data Integration': {
                'description': 'Combine data from multiple sources',
                'proficiency_levels': [
                    'Simple joins and concatenations',
                    'Complex multi-table operations',
                    'API data integration',
                    'Real-time data pipeline management'
                ]
            }
        },
        
        'Analysis and Visualization': {
            'Statistical Analysis': {
                'description': 'Apply appropriate statistical methods',
                'proficiency_levels': [
                    'Descriptive statistics and distributions',
                    'Hypothesis testing and confidence intervals',
                    'Regression and correlation analysis',
                    'Advanced statistical modeling'
                ]
            },
            'Data Visualization': {
                'description': 'Create effective visual communications',
                'proficiency_levels': [
                    'Basic plots with matplotlib',
                    'Professional multi-panel figures',
                    'Interactive and dashboard creation',
                    'Advanced visualization design principles'
                ]
            },
            'Exploratory Data Analysis': {
                'description': 'Systematically explore and understand data',
                'proficiency_levels': [
                    'Basic data exploration workflows',
                    'Pattern identification and hypothesis generation',
                    'Advanced EDA techniques',
                    'Automated EDA and insight generation'
                ]
            }
        },
        
        'Professional Skills': {
            'Project Management': {
                'description': 'Organize and execute data science projects',
                'proficiency_levels': [
                    'Personal project organization',
                    'Team collaboration and communication',
                    'Stakeholder management',
                    'Strategic project leadership'
                ]
            },
            'Documentation': {
                'description': 'Create clear, useful documentation',
                'proficiency_levels': [
                    'Code comments and basic README files',
                    'Comprehensive project documentation',
                    'User guides and tutorials',
                    'Technical writing and publication'
                ]
            },
            'Reproducibility': {
                'description': 'Create reproducible analysis workflows',
                'proficiency_levels': [
                    'Consistent file organization',
                    'Environment management and version control',
                    'Automated analysis pipelines',
                    'Research-grade reproducibility standards'
                ]
            }
        }
    }
    
    print("=== DATASCI 217 SKILLS ASSESSMENT ===")
    print("Rate your current proficiency in each area (1-4 scale):\n")
    
    total_skills = 0
    assessment_results = {}
    
    for category, skills in skills_framework.items():
        print(f"{category.upper()}")
        print("-" * len(category))
        
        category_results = {}
        for skill_name, skill_info in skills.items():
            print(f"\n{skill_name}: {skill_info['description']}")
            print("Proficiency levels:")
            for i, level in enumerate(skill_info['proficiency_levels'], 1):
                print(f"  {i}. {level}")
            
            # In interactive environment, would collect user input
            # Here we demonstrate the framework
            print(f"Current level: [Rate 1-4] ___")
            category_results[skill_name] = {
                'description': skill_info['description'],
                'levels': skill_info['proficiency_levels'],
                'current_level': None  # Would be filled by user
            }
            total_skills += 1
        
        assessment_results[category] = category_results
        print("\n" + "="*50 + "\n")
    
    # Generate development recommendations
    print("DEVELOPMENT RECOMMENDATIONS")
    print("Based on your assessment, focus on:")
    print("• Skills rated 1-2: Priority development areas")
    print("• Skills rated 3: Opportunities for advancement")
    print("• Skills rated 4: Mentor others and stay current")
    
    return assessment_results

def create_learning_path(weak_areas, career_goals):
    """
    Create personalized learning path based on assessment
    """
    learning_resources = {
        'Technical Foundation': {
            'books': [
                'The Linux Command Line by William Shotts',
                'Effective Python by Brett Slatkin',
                'Pro Git by Scott Chacon and Ben Straub'
            ],
            'online': [
                'Command Line Bootcamp (various platforms)',
                'Python.org tutorial and documentation',
                'GitHub Learning Lab'
            ],
            'practice': [
                'Daily command line usage',
                'Contribute to open source projects',
                'Build personal automation scripts'
            ]
        },
        
        'Data Manipulation': {
            'books': [
                'Python for Data Analysis by Wes McKinney',
                'Pandas Cookbook by Ted Petrou',
                'Effective Pandas by Matt Harrison'
            ],
            'online': [
                'Kaggle Learn courses',
                'DataCamp pandas tracks',
                'Real Python tutorials'
            ],
            'practice': [
                'Kaggle competitions and datasets',
                'Personal data projects',
                'Recreate published analyses'
            ]
        },
        
        'Analysis and Visualization': {
            'books': [
                'The Grammar of Graphics by Leland Wilkinson',
                'Storytelling with Data by Cole Nussbaumer Knaflic',
                'Statistics Done Wrong by Alex Reinhart'
            ],
            'online': [
                'Matplotlib and Seaborn documentation',
                'Coursera statistics courses',
                'Observable HQ tutorials'
            ],
            'practice': [
                '#MakeoverMonday challenges',
                'Reproduce published figures',
                'Create analysis blog posts'
            ]
        },
        
        'Professional Skills': {
            'books': [
                'The Pragmatic Programmer by Hunt and Thomas',
                'Clean Code by Robert C. Martin',
                'The Data Science Handbook by Field Cady'
            ],
            'online': [
                'Project management courses',
                'Technical writing workshops',
                'Research reproducibility training'
            ],
            'practice': [
                'Lead team projects',
                'Write documentation for others',
                'Mentor junior colleagues'
            ]
        }
    }
    
    career_paths = {
        'Data Analyst': {
            'focus_areas': ['Data Manipulation', 'Analysis and Visualization'],
            'next_steps': [
                'Master advanced Excel/SQL skills',
                'Learn business intelligence tools',
                'Develop domain expertise',
                'Practice stakeholder communication'
            ]
        },
        'Data Scientist': {
            'focus_areas': ['Analysis and Visualization', 'Professional Skills'],
            'next_steps': [
                'Learn machine learning fundamentals',
                'Master statistical modeling',
                'Develop experiment design skills',
                'Build portfolio of end-to-end projects'
            ]
        },
        'Research Data Specialist': {
            'focus_areas': ['Professional Skills', 'Technical Foundation'],
            'next_steps': [
                'Learn research methodology',
                'Master reproducibility tools',
                'Understand research ethics',
                'Develop grant writing skills'
            ]
        },
        'Data Engineer': {
            'focus_areas': ['Technical Foundation', 'Data Manipulation'],
            'next_steps': [
                'Learn database systems and SQL',
                'Master cloud platforms',
                'Understand distributed systems',
                'Practice system design'
            ]
        }
    }
    
    print("PERSONALIZED LEARNING PATH")
    print("=" * 30)
    
    for area in weak_areas:
        if area in learning_resources:
            print(f"\n{area.upper()} DEVELOPMENT PLAN:")
            resources = learning_resources[area]
            
            print("Recommended books:")
            for book in resources['books'][:2]:  # Top 2 recommendations
                print(f"  • {book}")
            
            print("Online resources:")
            for resource in resources['online'][:2]:
                print(f"  • {resource}")
            
            print("Practice opportunities:")
            for practice in resources['practice'][:2]:
                print(f"  • {practice}")
    
    if career_goals in career_paths:
        path_info = career_paths[career_goals]
        print(f"\n{career_goals.upper()} CAREER PATH:")
        print("Priority focus areas:")
        for area in path_info['focus_areas']:
            print(f"  • {area}")
        
        print("Next steps:")
        for step in path_info['next_steps']:
            print(f"  • {step}")

# Run skills assessment
assessment = assess_data_science_skills()

# Example learning path creation
weak_areas = ['Technical Foundation', 'Professional Skills']
career_goal = 'Data Scientist'
create_learning_path(weak_areas, career_goal)
```

## Resources for Continued Learning

### Curated Learning Resources

**Reference:**
```python
def create_comprehensive_resource_guide():
    """
    Comprehensive guide to data science learning resources
    """
    resource_guide = {
        'Essential Books': {
            'Foundation': [
                'Python for Data Analysis by Wes McKinney - The pandas creator\'s guide',
                'The Art of Statistics by David Spiegelhalter - Statistics for everyone',
                'Weapons of Math Destruction by Cathy O\'Neil - Ethics in data science'
            ],
            'Advanced Technical': [
                'Elements of Statistical Learning by Hastie, Tibshirani, Friedman',
                'Pattern Recognition and Machine Learning by Christopher Bishop',
                'Causal Inference: The Mixtape by Scott Cunningham'
            ],
            'Communication': [
                'Storytelling with Data by Cole Nussbaumer Knaflic',
                'The Truthful Art by Alberto Cairo',
                'Made to Stick by Chip Heath and Dan Heath'
            ]
        },
        
        'Online Learning Platforms': {
            'Free': [
                'Kaggle Learn - Practical micro-courses',
                'Coursera (audit) - University-level courses',
                'edX - MIT, Harvard, and other top universities',
                'Khan Academy - Statistics fundamentals',
                'YouTube - 3Blue1Brown, StatQuest, others'
            ],
            'Paid': [
                'DataCamp - Interactive data science learning',
                'Pluralsight - Technology skills development',
                'Udacity - Nanodegree programs',
                'LinkedIn Learning - Professional development'
            ]
        },
        
        'Communities and Networking': {
            'Online Communities': [
                'r/datascience - Reddit community for discussions',
                'Stack Overflow - Technical Q&A',
                'Cross Validated - Statistics Stack Exchange',
                'Kaggle Forums - Competition and dataset discussions',
                'Twitter #DataScience - Industry news and insights'
            ],
            'Professional Organizations': [
                'American Statistical Association (ASA)',
                'International Association for Statistical Computing (IASC)',
                'Local data science meetups and groups',
                'Industry-specific professional organizations'
            ]
        },
        
        'Practice Opportunities': {
            'Competition Platforms': [
                'Kaggle - Premier data science competitions',
                'DrivenData - Social good competitions',
                'Analytics Vidhya - Learning-focused competitions',
                'Zindi - Africa-focused data science challenges'
            ],
            'Project Ideas': [
                'Analyze publicly available datasets (government, research)',
                'Recreate analyses from published papers',
                'Build dashboards for local organizations',
                'Contribute to open source data science projects'
            ]
        },
        
        'Career Development': {
            'Portfolio Building': [
                'GitHub - Showcase code and projects',
                'Personal website/blog - Document learning journey',
                'LinkedIn - Professional networking and visibility',
                'Kaggle - Competition rankings and notebooks'
            ],
            'Job Search Resources': [
                'Indeed, LinkedIn Jobs - Job postings',
                'AngelList - Startup opportunities',
                'Company websites - Direct applications',
                'Networking events and conferences'
            ]
        },
        
        'Staying Current': {
            'News and Trends': [
                'Towards Data Science (Medium) - Industry articles',
                'KDnuggets - News, tutorials, jobs',
                'Data Science Central - Community and resources',
                'Hacker News - Technology discussions'
            ],
            'Research': [
                'arXiv.org - Latest research preprints',
                'Google Scholar - Academic papers',
                'Papers With Code - Research with implementations',
                'Distill.pub - Clear explanations of complex topics'
            ]
        }
    }
    
    print("=== COMPREHENSIVE DATA SCIENCE RESOURCE GUIDE ===")
    
    for category, subcategories in resource_guide.items():
        print(f"\n{category.upper()}")
        print("=" * len(category))
        
        for subcat, resources in subcategories.items():
            print(f"\n{subcat}:")
            for resource in resources:
                print(f"  • {resource}")
    
    # Create action plan template
    print("\n\nACTION PLAN TEMPLATE")
    print("=" * 20)
    
    action_plan = """
## 90-Day Data Science Development Plan

### Month 1: Skill Building
- [ ] Complete 2 online courses in weak areas
- [ ] Read 1 foundational book
- [ ] Start personal data project
- [ ] Join 2 online communities

### Month 2: Practice and Application
- [ ] Complete Kaggle competition or dataset analysis
- [ ] Write 2 blog posts about learning journey
- [ ] Contribute to open source project
- [ ] Attend virtual conference or meetup

### Month 3: Portfolio and Network
- [ ] Finish and document major project
- [ ] Update LinkedIn and GitHub profiles
- [ ] Conduct 3 informational interviews
- [ ] Apply to 5 relevant positions

### Ongoing Activities
- [ ] Daily: 30 minutes of coding/learning
- [ ] Weekly: Read industry articles and papers
- [ ] Monthly: Assess progress and adjust plan
- [ ] Quarterly: Major skill evaluation and pivot
"""
    
    print(action_plan)
    
    return resource_guide

# Generate the comprehensive guide
resource_guide = create_comprehensive_resource_guide()
```

### Building Your Professional Network

**Reference:**
```python
def develop_networking_strategy():
    """
    Strategic approach to building professional data science network
    """
    networking_framework = {
        'Online Presence': {
            'LinkedIn Profile': [
                'Professional headline highlighting data science skills',
                'Summary showcasing projects and learning journey',
                'Skills section with endorsements',
                'Regular posts about learning and projects'
            ],
            'GitHub Profile': [
                'Clean, well-documented repositories',
                'README files explaining projects',
                'Consistent commit history showing growth',
                'Contributions to open source projects'
            ],
            'Personal Brand': [
                'Technical blog or Medium articles',
                'Consistent username across platforms',
                'Professional photo and bio',
                'Portfolio website showcasing work'
            ]
        },
        
        'Community Engagement': {
            'Online Communities': [
                'Participate in discussions (don\'t just lurk)',
                'Share resources and insights',
                'Ask thoughtful questions',
                'Help others with their problems'
            ],
            'Local Events': [
                'Attend meetups and conferences',
                'Volunteer at data science events',
                'Present your work when ready',
                'Join study groups or book clubs'
            ]
        },
        
        'Professional Relationships': {
            'Mentorship': [
                'Find mentors in your target roles',
                'Offer to mentor others learning basics',
                'Join formal mentorship programs',
                'Maintain regular contact with mentors'
            ],
            'Informational Interviews': [
                'Request 15-30 minute conversations',
                'Prepare thoughtful questions',
                'Follow up with thank you and updates',
                'Offer to help with their projects'
            ]
        }
    }
    
    print("=== NETWORKING STRATEGY FOR DATA SCIENTISTS ===")
    
    for category, strategies in networking_framework.items():
        print(f"\n{category.upper()}")
        print("-" * len(category))
        
        for strategy_type, actions in strategies.items():
            print(f"\n{strategy_type}:")
            for action in actions:
                print(f"  • {action}")
    
    # Networking action plan
    networking_plan = """
## NETWORKING ACTION PLAN

### Week 1-2: Foundation
- [ ] Optimize LinkedIn profile with data science focus
- [ ] Clean up GitHub profile and pin best repositories
- [ ] Identify 5 professionals to follow and engage with
- [ ] Join 3 relevant online communities

### Week 3-4: Engagement
- [ ] Comment meaningfully on 10 posts per week
- [ ] Share 2 valuable resources with community
- [ ] Reach out to 2 people for informational interviews
- [ ] Attend 1 virtual meetup or conference

### Month 2: Content Creation
- [ ] Write and publish 1 blog post about your learning
- [ ] Share a completed project on social media
- [ ] Answer 5 questions in online forums
- [ ] Present at local meetup (if ready)

### Month 3: Relationship Building
- [ ] Follow up with all informational interview contacts
- [ ] Offer to help with 2 community projects
- [ ] Connect with 20 new professionals in your field
- [ ] Schedule quarterly coffee chats with key contacts

### Ongoing Maintenance
- [ ] Weekly: Engage with community content
- [ ] Bi-weekly: Reach out to new connections
- [ ] Monthly: Share your own content or insights
- [ ] Quarterly: Evaluate and refresh networking strategy
"""
    
    print(networking_plan)

# Create networking strategy
develop_networking_strategy()
```

# Key Takeaways - Your DataSci 217 Journey

1. **Reproducible research** is essential for scientific integrity and career success
2. **Data management** best practices prevent problems and ensure research quality
3. **Ethical considerations** must be integrated into every analysis decision
4. **Collaborative workflows** are crucial for modern research and industry work
5. **Continuous learning** is required to stay current in the rapidly evolving field
6. **Professional networking** opens doors and provides ongoing support
7. **Skills assessment** helps focus development efforts effectively
8. **Documentation and communication** skills differentiate great data scientists

## What You've Accomplished

Through DataSci 217, you have built a comprehensive foundation:

**Technical Skills:**
- Command-line proficiency for data workflows
- Python programming for data analysis
- Version control with Git for collaboration
- pandas and NumPy for data manipulation
- Data cleaning and quality assessment
- Statistical analysis and visualization
- Automated analysis pipelines
- Professional reporting and communication

**Professional Skills:**
- Systematic approach to data problems
- Debugging and validation techniques
- Project organization and documentation
- Reproducible research practices
- Ethical data handling
- Team collaboration patterns
- Career development planning

**Research Applications:**
- Research data types and challenges
- Multi-site data harmonization
- Longitudinal data analysis
- Sensitive data protection
- Professional collaboration workflows

You are now prepared to tackle real-world data science challenges and contribute meaningfully to research teams, organizations, and your chosen career path.

## Your Next Steps

1. **Complete your final project** - demonstrate mastery of integrated skills
2. **Build your portfolio** - showcase your best work publicly
3. **Continue learning** - identify growth areas and pursue development
4. **Network professionally** - engage with the data science community
5. **Apply your skills** - seek opportunities to use what you've learned
6. **Give back** - help others starting their data science journey

Congratulations on completing DataSci 217! You now have the foundation to excel in data science research and industry applications. The field needs ethical, skilled practitioners like you.

# Final Practice Challenge

Your journey continues beyond this class:

1. **Portfolio Project:**
   - Choose a research question that interests you
   - Apply the complete DataSci 217 workflow from raw data to publication-ready results
   - Document everything using reproducible research principles
   - Share your work with the community

2. **Continuous Learning:**
   - Identify your top 3 skill development priorities
   - Create a 90-day learning plan with specific goals
   - Join professional communities and engage actively
   - Find a mentor and consider mentoring others

3. **Professional Network:**
   - Connect with classmates and maintain relationships
   - Attend data science events and conferences
   - Contribute to open source projects
   - Build your online professional presence

Remember: Great data scientists are made through consistent practice, continuous learning, and ethical application of skills. You have the foundation - now build upon it!

*Welcome to your data science career!*