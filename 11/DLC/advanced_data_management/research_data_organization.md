Advanced Research Data Organization

This is bonus content for DataSci 217 - Lecture 11. These advanced data management techniques build on the core reproducible research principles covered in the main lecture.

Professional Research Data Management System

**Reference:**
```python
File: research_data_manager.py
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
data_manager, config = setup_research_project("Customer_Satisfaction_Analysis")
```

Key Takeaways

This advanced data management system provides:

1. **Standardized Structure** - Consistent organization across all research projects
2. **Metadata Tracking** - Comprehensive documentation of all datasets
3. **Data Integrity** - File hashing and validation for data verification
4. **Automated Backup** - Systematic backup procedures for data protection
5. **Data Dictionaries** - Automatic generation of documentation
6. **Configuration Management** - Centralized project settings

Use this framework for large-scale research projects requiring professional data management standards.