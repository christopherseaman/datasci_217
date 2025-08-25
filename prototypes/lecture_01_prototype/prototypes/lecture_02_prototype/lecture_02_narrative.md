# Lecture 2: Data Structures and Version Control

## Overview

Data science workflows require two fundamental capabilities: organizing complex information efficiently and managing the evolution of your work over time. This lecture bridges the gap between basic programming concepts and professional development practices, introducing Python's powerful data structures alongside Git version control - tools that form the backbone of collaborative data science.

Understanding data structures is crucial because real-world data rarely fits into simple variables. You'll work with collections of observations, relationships between variables, and hierarchical information that requires sophisticated organization. Python's built-in data structures - lists, dictionaries, sets, and tuples - provide elegant solutions for these challenges, enabling you to write cleaner, more efficient code.

Equally important is version control, which transforms isolated coding exercises into collaborative, reproducible research. Git doesn't just track changes - it enables experimentation, collaboration, and the kind of iterative development that characterizes modern data science. By the end of this lecture, you'll understand how to structure complex data and manage the evolution of your analyses with professional-grade tools.

The integration of these topics reflects real-world practice: data scientists use sophisticated data structures to organize their analyses and version control to manage the collaborative development of those analyses. These skills work together to enable the reproducible, scalable workflows that distinguish professional data science from ad-hoc analysis.

## Learning Objectives

By the end of this lecture, students will be able to:

- Use Python lists, dictionaries, sets, and tuples to organize complex data efficiently
- Choose appropriate data structures based on the requirements of specific analytical tasks
- Initialize and manage Git repositories for tracking changes in data science projects
- Create meaningful commits that document the evolution of analytical work
- Collaborate on data science projects using Git branching and merging workflows
- Integrate version control practices into data analysis workflows for reproducible research

## Prerequisites

This lecture builds on the foundation from Lecture 1, assuming familiarity with:

- Python variables, basic data types, and control structures
- Command line navigation and file operations
- Running Python scripts and understanding basic error messages
- Creating and organizing project directories with appropriate file structures

Students should be comfortable with basic Python syntax and have experience running code examples from the command line. No prior experience with Git is required, but understanding of file systems and directories is essential.

## Core Concepts

### Python Data Structures: Organizing Information for Analysis

Data science involves working with collections of related information - multiple observations, related variables, hierarchical categories, and complex relationships. Python's built-in data structures provide powerful, flexible ways to organize this information that mirror how we think about data analytically.

#### Lists: Sequential Data and Observations

Lists represent ordered collections - perfect for time series data, sequences of observations, or any information where order matters. They're mutable, meaning you can modify them as your analysis evolves.

```python
# Temperature readings over a week - order matters for time series
temperature_readings = [23.1, 25.4, 22.8, 24.7, 26.1, 23.9, 22.3]

# Add new observations as data arrives
temperature_readings.append(24.2)
temperature_readings.extend([25.1, 23.7])  # Add multiple values

print(f"Latest reading: {temperature_readings[-1]}°C")
print(f"Weekly average: {sum(temperature_readings) / len(temperature_readings):.1f}°C")
```

Lists excel at representing sequences where position has meaning. Each index corresponds to a specific time point, measurement order, or sequence position that you need to preserve for analysis.

```python
# Working with list slices - essential for data analysis
recent_readings = temperature_readings[-3:]  # Last 3 readings
first_half = temperature_readings[:len(temperature_readings)//2]  # First half of data

# List comprehensions - powerful for data transformation
celsius_to_fahrenheit = [(temp * 9/5) + 32 for temp in temperature_readings]
high_temp_days = [temp for temp in temperature_readings if temp > 24.0]

print(f"High temperature days: {len(high_temp_days)} out of {len(temperature_readings)}")
```

The ability to slice, filter, and transform lists using comprehensions makes them invaluable for preliminary data exploration and transformation tasks.

#### Dictionaries: Key-Value Relationships and Structured Data

Dictionaries map keys to values, perfect for representing records, lookup tables, and any data where you need to associate related pieces of information. They're the foundation for understanding more complex data structures you'll encounter later.

```python
# Patient record - multiple related attributes
patient_data = {
    'patient_id': 'P001',
    'age': 67,
    'temperature': 38.2,
    'blood_pressure': {'systolic': 140, 'diastolic': 90},
    'medications': ['lisinopril', 'metformin'],
    'last_visit': '2024-08-10'
}

# Accessing and modifying structured data
print(f"Patient {patient_data['patient_id']} is {patient_data['age']} years old")
print(f"Current temperature: {patient_data['temperature']}°C")

# Adding new information as it becomes available
patient_data['lab_results'] = {
    'glucose': 145,
    'cholesterol': 220,
    'hemoglobin': 12.1
}

# Nested access for complex structures
systolic_bp = patient_data['blood_pressure']['systolic']
print(f"Systolic BP: {systolic_bp} mmHg")
```

Dictionaries mirror the structure of real-world records where multiple attributes describe a single entity. This makes them essential for data that arrives as records rather than simple sequences.

```python
# Multiple patient records - list of dictionaries pattern
patients = [
    {'id': 'P001', 'age': 67, 'condition': 'diabetes'},
    {'id': 'P002', 'age': 34, 'condition': 'hypertension'},
    {'id': 'P003', 'age': 45, 'condition': 'diabetes'}
]

# Analysis across records
diabetic_patients = [p for p in patients if p['condition'] == 'diabetes']
average_age = sum(p['age'] for p in patients) / len(patients)

print(f"Diabetic patients: {len(diabetic_patients)}")
print(f"Average age: {average_age:.1f} years")
```

This list-of-dictionaries pattern appears constantly in data science, representing datasets where each dictionary is an observation and keys are variables.

#### Sets: Unique Collections and Membership Testing

Sets store unique elements and excel at membership testing, removing duplicates, and set operations like union and intersection - common needs in data cleaning and analysis.

```python
# Research participants from different studies
study_a_participants = {'P001', 'P002', 'P003', 'P005', 'P007'}
study_b_participants = {'P003', 'P004', 'P005', 'P006', 'P008'}

# Set operations for analysis
overlap = study_a_participants & study_b_participants  # Intersection
all_participants = study_a_participants | study_b_participants  # Union
unique_to_a = study_a_participants - study_b_participants  # Difference

print(f"Participants in both studies: {len(overlap)}")
print(f"Total unique participants: {len(all_participants)}")
print(f"Only in Study A: {unique_to_a}")
```

Sets are particularly valuable for data cleaning tasks where you need to identify unique values, find overlaps between groups, or eliminate duplicates from your analysis.

```python
# Data cleaning example - removing duplicate measurements
raw_measurements = [23.1, 24.2, 23.1, 25.4, 24.2, 26.1, 23.1, 24.8]
unique_measurements = list(set(raw_measurements))

print(f"Raw measurements: {len(raw_measurements)}")
print(f"Unique measurements: {len(unique_measurements)}")
print(f"Duplicates removed: {len(raw_measurements) - len(unique_measurements)}")

# Fast membership testing - O(1) average case
def is_high_risk_patient(patient_id, high_risk_set):
    return patient_id in high_risk_set  # Very fast lookup

high_risk = {'P001', 'P005', 'P012', 'P023', 'P034'}
print(f"Patient P005 high risk: {is_high_risk_patient('P005', high_risk)}")
```

#### Tuples: Immutable Records and Coordinate Data

Tuples represent fixed collections - perfect for coordinates, RGB colors, or any grouping that shouldn't change after creation. Their immutability makes them suitable for dictionary keys and ensures data integrity.

```python
# Geographic coordinates for weather stations
weather_stations = [
    ('Station_A', 40.7128, -74.0060, 'New York'),
    ('Station_B', 34.0522, -118.2437, 'Los Angeles'),
    ('Station_C', 41.8781, -87.6298, 'Chicago')
]

# Unpacking tuples for analysis
for station_name, latitude, longitude, city in weather_stations:
    print(f"{station_name} ({city}): {latitude:.2f}°N, {longitude:.2f}°W")

# Using tuples as dictionary keys
temperature_data = {
    ('Station_A', '2024-08-13'): 26.7,
    ('Station_A', '2024-08-14'): 28.1,
    ('Station_B', '2024-08-13'): 32.2,
    ('Station_B', '2024-08-14'): 31.8
}

# Accessing data with composite keys
today_temp = temperature_data[('Station_A', '2024-08-13')]
print(f"Station A temperature today: {today_temp}°C")
```

Tuples ensure that coordinate data, composite keys, and fixed records remain unchanged, preventing accidental modifications that could corrupt your analysis.

### Choosing the Right Data Structure

Selecting appropriate data structures affects both code clarity and performance. Understanding the strengths of each structure helps you write more efficient, maintainable code.

```python
def analyze_patient_visits(visits_data):
    """
    Demonstrate choosing appropriate data structures for different analytical tasks.
    
    Args:
        visits_data: List of visit records
    """
    # List for ordered visit history
    visit_dates = [visit['date'] for visit in visits_data]
    
    # Dictionary for patient lookup
    patients_by_id = {visit['patient_id']: visit for visit in visits_data}
    
    # Set for unique conditions
    unique_conditions = {visit['condition'] for visit in visits_data}
    
    # Tuples for immutable patient records
    patient_summaries = [(v['patient_id'], v['age'], v['condition']) for v in visits_data]
    
    return {
        'chronological_visits': visit_dates,
        'patient_lookup': patients_by_id,
        'conditions_treated': unique_conditions,
        'summary_records': patient_summaries
    }

# Example usage
sample_visits = [
    {'patient_id': 'P001', 'date': '2024-08-10', 'age': 67, 'condition': 'diabetes'},
    {'patient_id': 'P002', 'date': '2024-08-11', 'age': 34, 'condition': 'hypertension'},
    {'patient_id': 'P003', 'date': '2024-08-12', 'age': 45, 'condition': 'diabetes'}
]

analysis = analyze_patient_visits(sample_visits)
print(f"Conditions treated: {analysis['conditions_treated']}")
print(f"Most recent visit: {analysis['chronological_visits'][-1]}")
```

### Version Control with Git: Managing Analytical Evolution

Data science projects evolve continuously - you explore different approaches, collaborate with team members, and iterate on analyses based on new insights. Version control transforms this natural evolution from a source of confusion into a structured, trackable process.

#### Understanding Git: More Than Backup

Git tracks the evolution of your work through commits - snapshots of your project at specific points in time. Unlike simple backup systems, Git maintains a complete history of changes, enables collaboration, and supports experimental development through branching.

```bash
# Initialize a new data science project with Git
mkdir climate_analysis_project
cd climate_analysis_project

# Initialize Git repository
git init
# This creates .git/ directory - the repository database

# Set up project structure
mkdir data scripts results documentation
touch README.md data/raw_data.csv scripts/analysis.py results/.gitkeep

# Check repository status
git status
# Shows untracked files and changes
```

Understanding Git status helps you maintain awareness of changes in your project. The staging area concept - where you prepare changes before committing - gives you control over which changes get recorded together.

```bash
# Add files to staging area
git add README.md
git add scripts/analysis.py
# Or add all files: git add .

# Check what's staged for commit
git status
git diff --staged  # See exactly what changes will be committed

# Create your first commit
git commit -m "Initial project structure with analysis script"
```

Each commit represents a meaningful milestone in your analysis. Good commits are atomic (focused on one change) and have descriptive messages that explain what was accomplished.

#### Tracking Changes in Data Analysis

Data science projects involve multiple types of changes - new data, revised analyses, updated visualizations, and refined documentation. Git helps you track these changes systematically.

```python
# Example: Evolution of an analysis script
# File: scripts/temperature_analysis.py

import pandas as pd
import matplotlib.pyplot as plt

def load_temperature_data(filename):
    """Load and validate temperature data from CSV."""
    try:
        data = pd.read_csv(filename)
        # Validate required columns
        required_cols = ['date', 'temperature', 'location']
        if not all(col in data.columns for col in required_cols):
            raise ValueError(f"Missing required columns: {required_cols}")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def analyze_temperature_trends(data):
    """Analyze temperature trends by location."""
    results = {}
    
    for location in data['location'].unique():
        location_data = data[data['location'] == location]
        results[location] = {
            'mean_temp': location_data['temperature'].mean(),
            'std_temp': location_data['temperature'].std(),
            'min_temp': location_data['temperature'].min(),
            'max_temp': location_data['temperature'].max()
        }
    
    return results

# This represents Version 1 of your analysis
```

As your analysis evolves, Git tracks each change with context:

```bash
# After modifying the analysis script
git add scripts/temperature_analysis.py
git commit -m "Add statistical summary functions for temperature analysis

- Implement load_temperature_data() with validation
- Add analyze_temperature_trends() for location-based analysis
- Include error handling for missing data files"
```

Good commit messages explain both what changed and why, making it easy to understand your project's evolution.

#### Branching: Safe Experimentation

Branches allow you to experiment with new approaches while keeping your stable analysis intact. This is crucial for data science where you often explore multiple analytical paths.

```bash
# Create a new branch for experimental analysis
git branch experimental-clustering
git checkout experimental-clustering
# Or combine: git checkout -b experimental-clustering

# Now you're working in the experimental branch
# Changes here don't affect your main analysis
```

Branching enables parallel development paths - you can try sophisticated clustering algorithms while maintaining a working basic analysis on the main branch.

```python
# File: scripts/clustering_analysis.py (in experimental branch)
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import numpy as np

def perform_temperature_clustering(data, n_clusters=3):
    """
    Experimental: Cluster locations based on temperature patterns.
    
    This experimental analysis explores whether locations with similar
    temperature patterns can be identified through clustering.
    """
    # Prepare data for clustering
    feature_data = data.pivot_table(
        index='location', 
        columns='date', 
        values='temperature'
    ).fillna(method='forward')
    
    # Standardize features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(feature_data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_features)
    
    # Return results with location names
    results = {}
    for i, location in enumerate(feature_data.index):
        results[location] = {
            'cluster': int(clusters[i]),
            'cluster_center': kmeans.cluster_centers_[clusters[i]].tolist()
        }
    
    return results

# This experimental code lives safely in its own branch
```

```bash
# Commit experimental work
git add scripts/clustering_analysis.py
git commit -m "Add experimental clustering analysis

- Implement K-means clustering for temperature patterns
- Include data preprocessing and standardization
- Return cluster assignments with center coordinates

This is experimental - needs validation before merging to main"
```

#### Merging: Integrating Successful Experiments

When experimental work proves valuable, you merge it back into your main analysis branch:

```bash
# Switch back to main branch
git checkout main

# Merge experimental work
git merge experimental-clustering
# This integrates the clustering analysis into your main project

# Clean up - delete the experimental branch
git branch -d experimental-clustering
```

This workflow lets you maintain a stable main analysis while exploring sophisticated approaches in parallel.

### Collaborative Data Science with Git

Real data science happens in teams. Git enables collaboration by managing concurrent changes and maintaining project history across multiple contributors.

#### Remote Repositories: Centralized Collaboration

Remote repositories (like GitHub) serve as centralized locations where team members can share and coordinate their work:

```bash
# Connect your local repository to a remote
git remote add origin https://github.com/username/climate-analysis.git

# Push your work to the remote
git push -u origin main
# The -u flag sets up tracking between local and remote branches

# Check your remote configuration
git remote -v
```

Remote repositories enable distributed collaboration - team members can work independently and merge their contributions systematically.

```bash
# Typical collaborative workflow
git pull origin main          # Get latest changes from team
# Make your changes
git add modified_files.py
git commit -m "Add precipitation correlation analysis"
git push origin main          # Share your changes with team
```

#### Handling Conflicts: When Changes Collide

When multiple people modify the same parts of a file, Git requires manual resolution:

```python
# Example conflict in analysis script
def analyze_temperature_data(data):
<<<<<<< HEAD
    # Your changes
    summary_stats = data.groupby('location').agg({
        'temperature': ['mean', 'std', 'min', 'max', 'count']
    }).round(2)
=======
    # Team member's changes  
    summary_stats = data.groupby('location')['temperature'].agg([
        'mean', 'median', 'std', 'min', 'max'
    ]).round(1)
>>>>>>> feature-branch
    
    return summary_stats
```

Git marks conflicts clearly, showing both versions. You resolve by choosing the best approach:

```python
def analyze_temperature_data(data):
    # Resolved version combining both approaches
    summary_stats = data.groupby('location').agg({
        'temperature': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).round(2)
    
    return summary_stats
```

```bash
# After resolving conflicts
git add scripts/analysis.py
git commit -m "Resolve merge conflict in temperature analysis

Combined statistical measures from both versions:
- Kept mean, std, min, max from original
- Added median from feature branch  
- Kept count for sample size tracking
- Used 2 decimal precision for consistency"
```

### Integrating Data Structures and Version Control

Professional data science workflows integrate sophisticated data organization with systematic change tracking. Here's how these skills work together:

```python
# File: scripts/integrated_analysis.py
"""
Comprehensive climate data analysis demonstrating integration of
data structures and version control practices.

This script shows how proper data organization and version control
work together in professional data science workflows.
"""

import json
import pandas as pd
from datetime import datetime
from typing import Dict, List, Set, Tuple

class ClimateAnalyzer:
    """
    Climate data analyzer demonstrating professional data structure usage.
    
    This class shows how different data structures serve specific purposes
    in a real analysis workflow, while version control tracks the evolution
    of analytical methods.
    """
    
    def __init__(self):
        # Dictionary for configuration - structured key-value pairs
        self.config = {
            'temperature_unit': 'celsius',
            'analysis_window': 30,  # days
            'outlier_threshold': 3.0,  # standard deviations
            'required_columns': ['date', 'location', 'temperature', 'humidity']
        }
        
        # Set for tracking processed locations - fast membership testing
        self.processed_locations: Set[str] = set()
        
        # List for maintaining analysis history - ordered sequence
        self.analysis_history: List[Dict] = []
        
        # Dictionary for caching results - fast lookup by key
        self.results_cache: Dict[Tuple[str, str], Dict] = {}
    
    def load_and_validate_data(self, filepath: str) -> pd.DataFrame:
        """Load climate data with validation and error handling."""
        try:
            # Load data using pandas (we'll explore this more in Lecture 3)
            data = pd.read_csv(filepath)
            
            # Validate required columns using set operations
            required_cols = set(self.config['required_columns'])
            actual_cols = set(data.columns)
            missing_cols = required_cols - actual_cols
            
            if missing_cols:
                raise ValueError(f"Missing columns: {missing_cols}")
            
            # Track which locations we've processed
            self.processed_locations.update(data['location'].unique())
            
            # Record this analysis step
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'data_loaded',
                'filepath': filepath,
                'records': len(data),
                'locations': len(data['location'].unique())
            })
            
            return data
            
        except Exception as e:
            # Log error in analysis history
            self.analysis_history.append({
                'timestamp': datetime.now().isoformat(),
                'action': 'data_load_failed',
                'filepath': filepath,
                'error': str(e)
            })
            raise
    
    def analyze_location_patterns(self, data: pd.DataFrame) -> Dict[str, Dict]:
        """Analyze temperature patterns for each location."""
        results = {}
        
        for location in data['location'].unique():
            # Create cache key using tuple (immutable, hashable)
            cache_key = (location, 'temperature_analysis')
            
            # Check cache first
            if cache_key in self.results_cache:
                results[location] = self.results_cache[cache_key]
                continue
            
            # Perform analysis for this location
            location_data = data[data['location'] == location]
            
            analysis_result = {
                'mean_temperature': float(location_data['temperature'].mean()),
                'std_temperature': float(location_data['temperature'].std()),
                'min_temperature': float(location_data['temperature'].min()),
                'max_temperature': float(location_data['temperature'].max()),
                'data_points': int(len(location_data)),
                'analysis_date': datetime.now().isoformat()
            }
            
            # Cache the result
            self.results_cache[cache_key] = analysis_result
            results[location] = analysis_result
        
        # Record this analysis step
        self.analysis_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'location_analysis_completed',
            'locations_analyzed': len(results)
        })
        
        return results
    
    def get_analysis_summary(self) -> Dict:
        """Generate summary of all analysis performed."""
        return {
            'total_locations_processed': len(self.processed_locations),
            'locations': list(self.processed_locations),
            'analysis_steps': len(self.analysis_history),
            'cached_results': len(self.results_cache),
            'configuration': self.config.copy(),
            'history': self.analysis_history.copy()
        }

# Usage example showing integration of concepts
if __name__ == "__main__":
    analyzer = ClimateAnalyzer()
    
    # This analysis can be tracked with Git commits at each step
    print("Climate Analysis Pipeline")
    print("=" * 40)
    
    # Each major step gets committed to version control
    try:
        # Step 1: Load data (commit after successful loading)
        # data = analyzer.load_and_validate_data('data/climate_data.csv')
        # git add .; git commit -m "Successfully load and validate climate data"
        
        # Step 2: Perform analysis (commit after analysis completion)
        # results = analyzer.analyze_location_patterns(data)
        # git add .; git commit -m "Complete location-based temperature analysis"
        
        # Step 3: Generate summary (commit final results)
        summary = analyzer.get_analysis_summary()
        print(f"Analysis Summary: {summary['total_locations_processed']} locations processed")
        
        # Save results using appropriate data structure
        with open('results/analysis_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # git add .; git commit -m "Generate and save analysis summary
        # - Complete temperature analysis for all locations
        # - Save structured results to JSON format
        # - Update analysis history and caching"
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        # Even failures get tracked in version control
        # git add .; git commit -m "Handle analysis failure: {error message}"
```

This integration demonstrates how data structures serve different purposes within a version-controlled workflow - dictionaries for configuration, sets for fast lookup, lists for ordered history, and tuples for immutable keys.

## Hands-On Practice

### Exercise 1: Data Structure Selection and Usage

Practice choosing appropriate data structures for different analytical scenarios while building intuition about performance and clarity trade-offs.

```python
# Create file: exercises/data_structure_practice.py
"""
Data Structure Practice Exercise

This exercise builds familiarity with Python's built-in data structures
by working through common data science scenarios.
"""

def exercise_1_patient_records():
    """
    Exercise 1: Managing patient records using appropriate data structures.
    
    Task: Create a system for managing patient information that supports
    efficient lookup, maintains visit order, and tracks unique conditions.
    """
    print("Exercise 1: Patient Records Management")
    print("-" * 40)
    
    # TODO: Create a list of patient dictionaries with the following structure:
    # - patient_id (string)
    # - name (string) 
    # - age (integer)
    # - visits (list of visit dates)
    # - conditions (set of condition names)
    # - contact_info (dictionary with phone, email)
    
    patients = [
        # Add at least 3 patient records here
    ]
    
    # TODO: Implement the following functions:
    
    def find_patient_by_id(patient_id):
        """Find patient record by ID. Return None if not found."""
        pass
    
    def add_visit(patient_id, visit_date):
        """Add a new visit date for a patient."""
        pass
    
    def add_condition(patient_id, condition):
        """Add a condition to patient's condition set."""
        pass
    
    def get_patients_with_condition(condition_name):
        """Return list of patient IDs who have the specified condition."""
        pass
    
    def get_all_unique_conditions():
        """Return set of all conditions across all patients."""
        pass
    
    # TODO: Test your functions with sample data
    
    print("✅ Patient records system implemented")

def exercise_2_temperature_analysis():
    """
    Exercise 2: Temperature data analysis using multiple data structures.
    
    Task: Analyze temperature readings from multiple sensors, tracking
    statistics and identifying patterns using appropriate data structures.
    """
    print("\nExercise 2: Temperature Data Analysis")
    print("-" * 42)
    
    # Sample data - temperature readings from different sensors
    raw_readings = [
        ('sensor_01', '2024-08-10', 23.5),
        ('sensor_02', '2024-08-10', 25.1),
        ('sensor_01', '2024-08-11', 24.2),
        ('sensor_03', '2024-08-10', 22.8),
        ('sensor_02', '2024-08-11', 26.3),
        ('sensor_01', '2024-08-12', 23.8),
        ('sensor_03', '2024-08-11', 23.1),
    ]
    
    # TODO: Organize this data using appropriate data structures:
    
    # 1. Dictionary mapping sensor_id to list of (date, temperature) tuples
    sensor_data = {}
    
    # 2. Dictionary mapping date to list of all readings for that date  
    daily_readings = {}
    
    # 3. Set of all unique sensor IDs
    unique_sensors = set()
    
    # 4. List of all temperatures (for overall statistics)
    all_temperatures = []
    
    # TODO: Process raw_readings to populate these data structures
    
    # TODO: Implement analysis functions:
    
    def get_sensor_statistics(sensor_id):
        """Return dict with mean, min, max temperature for a sensor."""
        pass
    
    def get_daily_average(date):
        """Return average temperature across all sensors for a date."""
        pass
    
    def find_temperature_outliers(threshold_std=2.0):
        """Find readings that are more than threshold_std away from mean."""
        pass
    
    def get_sensor_correlation_pairs():
        """Find pairs of sensors that have readings on the same dates."""
        pass
    
    # TODO: Run analysis and display results
    
    print("✅ Temperature analysis system implemented")

def exercise_3_research_collaboration():
    """
    Exercise 3: Research collaboration tracking using sets and dictionaries.
    
    Task: Track researchers, their publications, and collaboration networks
    using set operations and structured data.
    """
    print("\nExercise 3: Research Collaboration Analysis")
    print("-" * 45)
    
    # Sample research data
    researchers = {
        'Dr. Smith': {'field': 'climate_science', 'publications': {'paper_1', 'paper_3', 'paper_5'}},
        'Dr. Jones': {'field': 'data_science', 'publications': {'paper_2', 'paper_3', 'paper_4'}},
        'Dr. Brown': {'field': 'statistics', 'publications': {'paper_1', 'paper_4', 'paper_6'}},
        'Dr. Davis': {'field': 'climate_science', 'publications': {'paper_2', 'paper_5', 'paper_7'}}
    }
    
    # TODO: Implement analysis functions using set operations:
    
    def find_collaborators(researcher1, researcher2):
        """Return set of papers both researchers worked on."""
        pass
    
    def get_field_publications(field_name):
        """Return all publications by researchers in a specific field."""
        pass
    
    def find_most_collaborative_researcher():
        """Return researcher who has collaborated on most papers."""
        pass
    
    def get_collaboration_network():
        """Return dict showing all collaboration pairs and shared papers."""
        pass
    
    def find_unique_contributors(paper_list):
        """For given papers, return researchers who worked on them."""
        pass
    
    # TODO: Test your functions and display results
    
    print("✅ Research collaboration analysis implemented")

if __name__ == "__main__":
    exercise_1_patient_records()
    exercise_2_temperature_analysis() 
    exercise_3_research_collaboration()
    
    print("\n" + "=" * 50)
    print("All data structure exercises completed!")
    print("Try modifying the functions to explore different approaches.")
```

### Exercise 2: Git Workflow for Data Science Projects

Practice version control workflows that mirror real data science project development patterns.

```bash
# Exercise 2: Git Workflow Practice
# Create a new directory for this exercise

mkdir git_practice_project
cd git_practice_project

# Initialize Git repository
git init

# Create initial project structure
mkdir data analysis results documentation
touch README.md
touch analysis/exploratory_analysis.py
touch analysis/statistical_tests.py
touch data/sample_data.csv
touch results/.gitkeep

# Write initial README
cat > README.md << 'EOF'
# Climate Data Analysis Project

## Overview
This project analyzes temperature trends across multiple weather stations.

## Structure
- `data/` - Raw and processed data files
- `analysis/` - Python analysis scripts
- `results/` - Generated outputs and reports
- `documentation/` - Project documentation

## Usage
1. Place data files in `data/` directory
2. Run analysis scripts from `analysis/` directory
3. Check results in `results/` directory
EOF

# Create initial analysis script
cat > analysis/exploratory_analysis.py << 'EOF'
#!/usr/bin/env python3
"""
Exploratory Data Analysis for Climate Project

Initial exploration of temperature data patterns.
"""

import csv
import statistics

def load_temperature_data(filename):
    """Load temperature data from CSV file."""
    temperatures = []
    with open(filename, 'r') as file:
        reader = csv.DictReader(file)
        for row in reader:
            temperatures.append(float(row['temperature']))
    return temperatures

def basic_statistics(data):
    """Calculate basic statistical measures."""
    return {
        'mean': statistics.mean(data),
        'median': statistics.median(data),
        'stdev': statistics.stdev(data) if len(data) > 1 else 0,
        'min': min(data),
        'max': max(data),
        'count': len(data)
    }

if __name__ == "__main__":
    print("Climate Data Exploratory Analysis")
    print("-" * 35)
    # Analysis implementation will be added in future commits
EOF

# TODO: Practice the following Git workflow:

# 1. Add and commit initial project structure
# git add .
# git commit -m "Initial project structure with README and analysis skeleton"

# 2. Create a branch for data loading functionality
# git checkout -b feature/data-loading

# 3. Enhance the analysis script with better data loading
# (Modify analysis/exploratory_analysis.py to add error handling)

# 4. Commit the improvements
# git add analysis/exploratory_analysis.py
# git commit -m "Add robust data loading with error handling"

# 5. Switch back to main and merge
# git checkout main
# git merge feature/data-loading

# 6. Create another branch for statistical analysis
# git checkout -b feature/statistics

# 7. Add statistical_tests.py with content
# (Create comprehensive statistical analysis functions)

# 8. Commit and merge statistical analysis
# git add analysis/statistical_tests.py
# git commit -m "Add statistical analysis functions"
# git checkout main
# git merge feature/statistics

# 9. Create sample data and document the workflow
# (Add sample CSV data and update README)

# 10. Final commit with documentation
# git add .
# git commit -m "Add sample data and complete initial documentation"
```

### Exercise 3: Integration Challenge - Data Processing Pipeline

Combine data structures and version control in a realistic data processing scenario.

```python
# Create file: exercises/integration_challenge.py
"""
Integration Challenge: Climate Data Processing Pipeline

This exercise combines data structures and version control practices
in a realistic data science workflow.
"""

import json
import csv
import os
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional

class ClimateDataProcessor:
    """
    Climate data processing pipeline demonstrating integration of
    data structures with version-controlled development.
    """
    
    def __init__(self, config_file: str = 'config/processing_config.json'):
        """Initialize processor with configuration."""
        # TODO: Load configuration from JSON file using dictionary
        self.config = {}
        
        # TODO: Initialize data structures for different purposes:
        # - processed_files: Set for tracking processed files (no duplicates)
        # - processing_history: List for ordered processing steps
        # - station_data: Dictionary for station information lookup
        # - quality_flags: Dictionary mapping quality codes to descriptions
        
        self.processed_files: Set[str] = set()
        self.processing_history: List[Dict] = []
        self.station_data: Dict[str, Dict] = {}
        self.quality_flags: Dict[str, str] = {
            'A': 'Excellent quality',
            'B': 'Good quality', 
            'C': 'Fair quality',
            'D': 'Poor quality',
            'E': 'Estimated'
        }
    
    def load_station_metadata(self, metadata_file: str) -> None:
        """Load weather station metadata."""
        # TODO: Read station information from CSV
        # Store in self.station_data dictionary with station_id as key
        # Include: station_id, name, latitude, longitude, elevation
        pass
    
    def process_temperature_file(self, data_file: str) -> Dict:
        """Process a single temperature data file."""
        # TODO: Implement file processing:
        # 1. Check if file already processed (use set lookup)
        # 2. Read temperature data from CSV
        # 3. Validate data quality using quality flags
        # 4. Calculate statistics for each station
        # 5. Record processing step in history (list append)
        # 6. Add file to processed files set
        # 7. Return results dictionary
        
        if data_file in self.processed_files:
            return {'status': 'already_processed', 'file': data_file}
        
        # Processing logic here...
        
        # Record processing step
        self.processing_history.append({
            'timestamp': datetime.now().isoformat(),
            'action': 'file_processed',
            'file': data_file,
            'status': 'success'
        })
        
        self.processed_files.add(data_file)
        
        return {'status': 'processed', 'file': data_file}
    
    def analyze_station_trends(self, station_id: str) -> Optional[Dict]:
        """Analyze temperature trends for a specific station."""
        # TODO: Implement station-specific analysis:
        # 1. Check if station exists in metadata (dictionary lookup)
        # 2. Gather all temperature data for this station
        # 3. Calculate monthly/seasonal trends
        # 4. Identify potential outliers
        # 5. Return analysis results
        
        if station_id not in self.station_data:
            return None
        
        # Analysis logic here...
        
        return {'station_id': station_id, 'analysis': 'completed'}
    
    def generate_quality_report(self) -> Dict:
        """Generate data quality assessment report."""
        # TODO: Create comprehensive quality report:
        # 1. Count files by processing status
        # 2. Summarize quality flag distribution
        # 3. List stations with data quality issues
        # 4. Identify temporal gaps in data
        
        report = {
            'processing_summary': {
                'total_files': len(self.processed_files),
                'processing_steps': len(self.processing_history)
            },
            'quality_summary': {},
            'station_summary': len(self.station_data),
            'generated_at': datetime.now().isoformat()
        }
        
        return report
    
    def export_results(self, output_dir: str) -> None:
        """Export processing results to files."""
        # TODO: Save results using appropriate formats:
        # 1. JSON for structured data (dictionaries, lists)
        # 2. CSV for tabular station data
        # 3. Text for processing logs
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Export processing history
        history_file = os.path.join(output_dir, 'processing_history.json')
        with open(history_file, 'w') as f:
            json.dump(self.processing_history, f, indent=2)
        
        # Export station data
        stations_file = os.path.join(output_dir, 'stations.json')
        with open(stations_file, 'w') as f:
            json.dump(self.station_data, f, indent=2)
        
        # Export processed files list
        files_list = os.path.join(output_dir, 'processed_files.txt')
        with open(files_list, 'w') as f:
            for filename in sorted(self.processed_files):
                f.write(f"{filename}\n")


def main():
    """
    Main processing workflow demonstrating integration of concepts.
    
    This workflow should be managed with Git commits at each major step:
    1. Initial setup and configuration
    2. Metadata loading and validation
    3. Data file processing
    4. Analysis and reporting
    5. Results export and documentation
    """
    print("Climate Data Processing Pipeline")
    print("=" * 40)
    
    # TODO: Implement the main workflow:
    
    # Step 1: Initialize processor
    # git add .; git commit -m "Initialize climate data processor with configuration"
    processor = ClimateDataProcessor()
    
    # Step 2: Load station metadata
    # processor.load_station_metadata('data/stations.csv')
    # git add .; git commit -m "Load and validate weather station metadata"
    
    # Step 3: Process temperature data files
    # data_files = ['data/temp_2023_01.csv', 'data/temp_2023_02.csv']
    # for data_file in data_files:
    #     result = processor.process_temperature_file(data_file)
    #     print(f"Processed {data_file}: {result['status']}")
    # git add .; git commit -m "Process temperature data files with quality validation"
    
    # Step 4: Generate analysis reports
    # quality_report = processor.generate_quality_report()
    # git add .; git commit -m "Generate comprehensive data quality reports"
    
    # Step 5: Export results
    # processor.export_results('results/')
    # git add .; git commit -m "Export processing results and analysis summaries"
    
    print("\n✅ Processing pipeline completed!")
    print("Remember to commit each major step to track your progress.")


if __name__ == "__main__":
    main()
```

## Real-World Applications

These integrated skills form the foundation for professional data science workflows across industries and research domains.

**Research Environments**: Academic researchers use sophisticated data structures to organize complex experimental data while Git enables collaboration with colleagues worldwide. Version control becomes essential when multiple researchers contribute to the same analysis, ensuring that experimental modifications are tracked and reversible.

**Industry Applications**: Technology companies process massive datasets using these same fundamental structures - dictionaries for user profiles, lists for time-series data, sets for unique user identification. Git enables the collaborative development of analysis pipelines that process millions of records daily.

**Healthcare Analytics**: Medical research requires careful organization of patient data using dictionaries and lists while maintaining strict version control of analysis scripts for regulatory compliance. The ability to track exactly which version of an analysis produced specific results is crucial for medical research validation.

**Financial Analysis**: Financial institutions use these data structures to organize market data, portfolio information, and risk assessments. Version control ensures that trading algorithms and risk models can be audited and that changes are systematically tested before deployment.

**Climate Research**: Climate scientists organize multi-dimensional data (location, time, measurements) using nested dictionaries and lists while Git enables collaboration on global research projects. The ability to track analytical methods over multi-year research cycles is essential for reproducible climate science.

## Assessment Integration

### Formative Assessment

Throughout this lecture, check your understanding with these key questions:

1. **Data Structure Selection**: "Given a scenario where you need to track unique patient IDs, maintain visit chronology, and enable fast lookup of patient records, which combination of data structures would you choose and why?"

2. **Git Workflow Understanding**: "If you're collaborating on a data analysis project and your teammate has modified the same function you've been working on, describe the Git workflow you would use to integrate both sets of changes."

3. **Integration Thinking**: "Explain how using appropriate data structures can make your Git commits more meaningful and your project history more understandable."

### Summative Assessment Preview

Your assignment will combine these concepts in a comprehensive project:

- **Data Structure Implementation**: Design and implement a data organization system for a multi-dimensional dataset using appropriate Python structures
- **Git Workflow Demonstration**: Track the development of your analysis using meaningful commits, branches for experimental work, and proper merge practices
- **Integration Project**: Create a data processing pipeline that demonstrates both sophisticated data organization and professional version control practices

This assessment mirrors real professional scenarios where data scientists must organize complex information while maintaining collaborative development practices.

## Further Reading and Resources

### Essential Resources
- [Python Data Structures Documentation](https://docs.python.org/3/tutorial/datastructures.html) - Official guide to lists, dictionaries, sets, and tuples with comprehensive examples
- [Pro Git Book](https://git-scm.com/book/en/v2) - Complete Git reference, chapters 1-3 cover fundamentals essential for collaborative development  
- [Real Python Git Tutorial](https://realpython.com/python-git-github-intro/) - Python-focused Git workflow guide with data science applications

### Advanced Topics
- [Git Workflows for Data Science](https://drivendata.github.io/cookiecutter-data-science/) - Professional project organization and Git practices for data science teams
- [Python Data Model](https://docs.python.org/3/reference/datamodel.html) - Deep understanding of how Python data structures work internally
- [Collaborative Data Science](https://the-turing-way.netlify.app/collaboration/collaboration.html) - Best practices for team-based analytical projects

### Practice Environments  
- [Git Branching Interactive](https://learngitbranching.js.org/) - Visual, interactive Git workflow practice with immediate feedback
- [Python Tutor](http://pythontutor.com/) - Visualize how data structures behave during program execution
- [GitHub Skills](https://skills.github.com/) - Hands-on practice with collaborative Git workflows

## Next Steps

In our next lecture, **NumPy and Pandas Foundations**, you'll see how today's data structure concepts scale to handle real-world datasets. The organizational principles you've learned - choosing appropriate structures, tracking changes systematically, and collaborating effectively - directly enable next week's topics:

- NumPy arrays extend list concepts to multi-dimensional numerical computing
- Pandas DataFrames build on dictionary concepts to create spreadsheet-like data analysis
- Version control becomes essential when processing large datasets that require iterative refinement
- The data organization patterns from today enable efficient data manipulation with professional libraries

The foundation you've built today - understanding when to use different data structures and how to track analytical evolution - makes advanced data manipulation feel natural rather than overwhelming.

Start integrating these practices immediately: organize your analysis work with appropriate data structures and track every significant change with descriptive Git commits. These habits, formed early, distinguish professional data scientists from those who struggle with complex, collaborative projects.

---

*Lecture Format: Notion-Compatible Narrative with Embedded Interactive Code*
*Progressive Complexity: Fundamentals → Integration → Real-World Applications*  
*Version: 2.0 - Generated by automated conversion tool*