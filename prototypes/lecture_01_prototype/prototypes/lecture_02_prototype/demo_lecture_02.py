#!/usr/bin/env python3
"""
Lecture 2: Interactive Python Demonstrations
Data Structures and Version Control Integration

This script provides executable demonstrations of concepts from Lecture 2.
Students can run this script to see data structures and Git workflows in action.

Usage:
    python3 demo_lecture_02.py                          # Run all demonstrations
    python3 demo_lecture_02.py --section structures     # Run data structures only
    python3 demo_lecture_02.py --section git            # Run Git workflow demos
    python3 demo_lecture_02.py --interactive            # Interactive mode
"""

import sys
import os
import json
import argparse
import subprocess
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional, Any


# =============================================================================
# SECTION 1: DATA STRUCTURES DEMONSTRATIONS
# =============================================================================

def demonstrate_list_operations():
    """
    Demonstrate list operations in the context of time-series data analysis.
    
    Lists are perfect for ordered data like time series, experimental
    observations, or any sequence where order matters.
    """
    print("=" * 60)
    print("SECTION 1: LISTS - SEQUENTIAL DATA AND TIME SERIES")
    print("=" * 60)
    
    # Time series temperature data - order is crucial
    print("Temperature Data Collection Over Time:")
    print("-" * 40)
    
    temperature_readings = []  # Start with empty list
    
    # Simulate data collection process
    daily_temps = [23.1, 25.4, 22.8, 24.7, 26.1, 23.9, 22.3, 24.2]
    days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday', 'Monday']
    
    for day, temp in zip(days, daily_temps):
        temperature_readings.append(temp)
        print(f"  {day}: {temp}°C (Total readings: {len(temperature_readings)})")
    
    print(f"\nComplete dataset: {temperature_readings}")
    
    # List operations essential for data analysis
    print(f"\nData Analysis Operations:")
    print(f"  Latest reading: {temperature_readings[-1]}°C")
    print(f"  First 3 readings: {temperature_readings[:3]}")
    print(f"  Last 3 readings: {temperature_readings[-3:]}")
    print(f"  Weekend temps: {temperature_readings[5:7]}")  # Sat-Sun
    
    # Statistical analysis using list operations
    average_temp = sum(temperature_readings) / len(temperature_readings)
    max_temp = max(temperature_readings)
    min_temp = min(temperature_readings)
    
    print(f"\nStatistical Summary:")
    print(f"  Average temperature: {average_temp:.1f}°C")
    print(f"  Temperature range: {min_temp}°C to {max_temp}°C")
    print(f"  Temperature spread: {max_temp - min_temp:.1f}°C")
    
    # List comprehensions for data transformation
    print(f"\nData Transformations:")
    fahrenheit_temps = [(temp * 9/5) + 32 for temp in temperature_readings]
    print(f"  Fahrenheit equivalent: {[f'{t:.1f}' for t in fahrenheit_temps[:3]]}... (showing first 3)")
    
    high_temp_days = [temp for temp in temperature_readings if temp > average_temp]
    print(f"  Above-average days: {len(high_temp_days)} out of {len(temperature_readings)}")
    
    # Advanced list operations
    print(f"\nAdvanced Analysis:")
    
    # Temperature differences between consecutive days
    temp_changes = [temperature_readings[i] - temperature_readings[i-1] 
                   for i in range(1, len(temperature_readings))]
    print(f"  Daily temperature changes: {[f'{change:+.1f}' for change in temp_changes[:4]]}...")
    
    # Identify temperature trends
    warming_days = sum(1 for change in temp_changes if change > 0)
    cooling_days = sum(1 for change in temp_changes if change < 0)
    print(f"  Warming days: {warming_days}, Cooling days: {cooling_days}")


def demonstrate_dictionary_operations():
    """
    Demonstrate dictionary operations for structured data and lookup tables.
    
    Dictionaries excel at representing records, lookup tables, and any data
    where you need to associate keys with values.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: DICTIONARIES - STRUCTURED DATA AND RECORDS")
    print("=" * 60)
    
    # Patient record system - typical use case for dictionaries
    print("Patient Record Management System:")
    print("-" * 35)
    
    # Single patient record with nested structure
    patient_001 = {
        'patient_id': 'P001',
        'name': 'Sarah Johnson',
        'age': 67,
        'admission_date': '2024-08-10',
        'vital_signs': {
            'temperature': 38.2,
            'blood_pressure': {'systolic': 140, 'diastolic': 90},
            'heart_rate': 88,
            'oxygen_saturation': 97
        },
        'medications': ['lisinopril', 'metformin', 'aspirin'],
        'allergies': ['penicillin', 'latex'],
        'insurance': {
            'provider': 'HealthCare Plus',
            'policy_number': 'HCP-789456',
            'group_id': 'GRP-001'
        }
    }
    
    print(f"Patient Record for {patient_001['name']}:")
    print(f"  ID: {patient_001['patient_id']}")
    print(f"  Age: {patient_001['age']} years")
    print(f"  Temperature: {patient_001['vital_signs']['temperature']}°C")
    print(f"  Blood Pressure: {patient_001['vital_signs']['blood_pressure']['systolic']}/{patient_001['vital_signs']['blood_pressure']['diastolic']} mmHg")
    print(f"  Medications: {', '.join(patient_001['medications'])}")
    
    # Multiple patient database - list of dictionaries pattern
    print(f"\nPatient Database Analysis:")
    print("-" * 28)
    
    patients_database = [
        {'id': 'P001', 'age': 67, 'condition': 'diabetes', 'temperature': 38.2, 'length_of_stay': 3},
        {'id': 'P002', 'age': 34, 'condition': 'hypertension', 'temperature': 37.1, 'length_of_stay': 1},
        {'id': 'P003', 'age': 45, 'condition': 'diabetes', 'temperature': 36.8, 'length_of_stay': 2},
        {'id': 'P004', 'age': 78, 'condition': 'pneumonia', 'temperature': 39.1, 'length_of_stay': 5},
        {'id': 'P005', 'age': 56, 'condition': 'hypertension', 'temperature': 36.9, 'length_of_stay': 1}
    ]
    
    # Dictionary operations for data analysis
    print(f"Database contains {len(patients_database)} patients")
    
    # Group patients by condition
    conditions = {}
    for patient in patients_database:
        condition = patient['condition']
        if condition not in conditions:
            conditions[condition] = []
        conditions[condition].append(patient['id'])
    
    print(f"\nPatients by condition:")
    for condition, patient_ids in conditions.items():
        print(f"  {condition}: {len(patient_ids)} patients ({', '.join(patient_ids)})")
    
    # Statistical analysis using dictionary aggregation
    total_age = sum(p['age'] for p in patients_database)
    avg_age = total_age / len(patients_database)
    
    total_temp = sum(p['temperature'] for p in patients_database)
    avg_temp = total_temp / len(patients_database)
    
    print(f"\nDatabase Statistics:")
    print(f"  Average age: {avg_age:.1f} years")
    print(f"  Average temperature: {avg_temp:.1f}°C")
    
    # Advanced dictionary operations
    print(f"\nAdvanced Analysis:")
    
    # Create lookup dictionary for fast patient access
    patient_lookup = {p['id']: p for p in patients_database}
    
    # Demonstrate fast lookup
    lookup_id = 'P003'
    if lookup_id in patient_lookup:
        patient = patient_lookup[lookup_id]
        print(f"  Quick lookup for {lookup_id}: {patient['age']} years old, {patient['condition']}")
    
    # Complex filtering using dictionary comprehension
    high_temp_patients = {p['id']: p['temperature'] for p in patients_database if p['temperature'] > 37.5}
    print(f"  High temperature patients: {high_temp_patients}")
    
    # Nested dictionary operations
    print(f"\nNested Dictionary Operations:")
    
    # Update patient record with new lab results
    patient_001['lab_results'] = {
        'blood_glucose': 145,
        'cholesterol': {'total': 220, 'hdl': 45, 'ldl': 155},
        'hemoglobin_a1c': 7.2
    }
    
    print(f"  Added lab results for {patient_001['name']}")
    print(f"  HbA1c: {patient_001['lab_results']['hemoglobin_a1c']}%")
    print(f"  Total cholesterol: {patient_001['lab_results']['cholesterol']['total']} mg/dL")


def demonstrate_set_operations():
    """
    Demonstrate set operations for unique collections and membership testing.
    
    Sets excel at removing duplicates, fast membership testing, and
    set operations like union, intersection, and difference.
    """
    print("\n" + "=" * 60)
    print("SECTION 3: SETS - UNIQUE COLLECTIONS AND MEMBERSHIP")
    print("=" * 60)
    
    # Research study participant tracking
    print("Clinical Research Study Management:")
    print("-" * 38)
    
    # Participants in different studies
    diabetes_study = {'P001', 'P003', 'P005', 'P007', 'P009', 'P012', 'P015'}
    cardiac_study = {'P003', 'P004', 'P006', 'P008', 'P009', 'P011', 'P013'}
    nutrition_study = {'P001', 'P002', 'P004', 'P007', 'P010', 'P012', 'P014'}
    
    print(f"Study participation:")
    print(f"  Diabetes study: {len(diabetes_study)} participants")
    print(f"  Cardiac study: {len(cardiac_study)} participants") 
    print(f"  Nutrition study: {len(nutrition_study)} participants")
    
    # Set operations for research analysis
    print(f"\nCross-Study Analysis:")
    print("-" * 22)
    
    # Participants in multiple studies
    diabetes_cardiac = diabetes_study & cardiac_study  # Intersection
    diabetes_nutrition = diabetes_study & nutrition_study
    cardiac_nutrition = cardiac_study & nutrition_study
    all_three_studies = diabetes_study & cardiac_study & nutrition_study
    
    print(f"  Diabetes + Cardiac: {len(diabetes_cardiac)} participants {sorted(diabetes_cardiac)}")
    print(f"  Diabetes + Nutrition: {len(diabetes_nutrition)} participants {sorted(diabetes_nutrition)}")
    print(f"  Cardiac + Nutrition: {len(cardiac_nutrition)} participants {sorted(cardiac_nutrition)}")
    print(f"  All three studies: {len(all_three_studies)} participants {sorted(all_three_studies)}")
    
    # Total unique participants across all studies
    all_participants = diabetes_study | cardiac_study | nutrition_study  # Union
    print(f"\n  Total unique participants: {len(all_participants)}")
    
    # Participants unique to each study
    diabetes_only = diabetes_study - cardiac_study - nutrition_study
    cardiac_only = cardiac_study - diabetes_study - nutrition_study
    nutrition_only = nutrition_study - diabetes_study - cardiac_study
    
    print(f"  Diabetes only: {len(diabetes_only)} participants {sorted(diabetes_only)}")
    print(f"  Cardiac only: {len(cardiac_only)} participants {sorted(cardiac_only)}")
    print(f"  Nutrition only: {len(nutrition_only)} participants {sorted(nutrition_only)}")
    
    # Data cleaning example with sets
    print(f"\nData Quality Analysis:")
    print("-" * 24)
    
    # Simulate raw data with duplicates
    raw_measurements = [
        'glucose', 'cholesterol', 'glucose', 'blood_pressure', 
        'heart_rate', 'cholesterol', 'temperature', 'glucose',
        'oxygen_saturation', 'blood_pressure', 'weight'
    ]
    
    unique_measurements = set(raw_measurements)
    
    print(f"  Raw measurement list: {len(raw_measurements)} items")
    print(f"  Unique measurements: {len(unique_measurements)} items")
    print(f"  Duplicates removed: {len(raw_measurements) - len(unique_measurements)}")
    print(f"  Unique measurement types: {sorted(unique_measurements)}")
    
    # Fast membership testing
    print(f"\nMembership Testing (Very Fast with Sets):")
    print("-" * 47)
    
    high_risk_conditions = {'diabetes', 'cardiovascular_disease', 'obesity', 'hypertension'}
    
    test_conditions = ['diabetes', 'common_cold', 'hypertension', 'seasonal_allergies']
    
    for condition in test_conditions:
        risk_status = "HIGH RISK" if condition in high_risk_conditions else "standard risk"
        print(f"  {condition}: {risk_status}")


def demonstrate_tuple_operations():
    """
    Demonstrate tuple operations for immutable records and coordinate data.
    
    Tuples represent fixed collections that shouldn't change after creation,
    perfect for coordinates, RGB colors, or immutable records.
    """
    print("\n" + "=" * 60)
    print("SECTION 4: TUPLES - IMMUTABLE RECORDS AND COORDINATES")
    print("=" * 60)
    
    # Weather station coordinate system
    print("Weather Station Network:")
    print("-" * 26)
    
    # Weather stations with coordinates - perfect for tuples
    weather_stations = [
        ('NYC_Central_Park', 40.7829, -73.9654, 42, 'New York'),
        ('LA_Downtown', 34.0522, -118.2437, 71, 'Los Angeles'),
        ('Chicago_ORD', 41.9742, -87.9073, 205, 'Chicago'),
        ('Miami_Beach', 25.7907, -80.1300, 1, 'Miami'),
        ('Denver_Intl', 39.8561, -104.6737, 1655, 'Denver')
    ]
    
    print(f"Network contains {len(weather_stations)} stations:")
    
    for station_name, latitude, longitude, elevation, city in weather_stations:
        print(f"  {station_name} ({city}):")
        print(f"    Location: {latitude:.2f}°N, {longitude:.2f}°W")
        print(f"    Elevation: {elevation}m above sea level")
    
    # Using tuples as dictionary keys
    print(f"\nTemperature Data Storage (Tuples as Keys):")
    print("-" * 47)
    
    # Temperature readings using (station, date) as composite key
    temperature_data = {
        ('NYC_Central_Park', '2024-08-13'): {'temp': 26.7, 'humidity': 68, 'pressure': 1013.2},
        ('NYC_Central_Park', '2024-08-14'): {'temp': 28.1, 'humidity': 72, 'pressure': 1011.8},
        ('LA_Downtown', '2024-08-13'): {'temp': 32.2, 'humidity': 45, 'pressure': 1015.1},
        ('LA_Downtown', '2024-08-14'): {'temp': 31.8, 'humidity': 48, 'pressure': 1014.7},
        ('Chicago_ORD', '2024-08-13'): {'temp': 24.4, 'humidity': 58, 'pressure': 1018.3}
    }
    
    # Access data using tuple keys
    nyc_today = temperature_data[('NYC_Central_Park', '2024-08-13')]
    print(f"  NYC Central Park (2024-08-13):")
    print(f"    Temperature: {nyc_today['temp']}°C")
    print(f"    Humidity: {nyc_today['humidity']}%")
    print(f"    Pressure: {nyc_today['pressure']} hPa")
    
    # Analyzing coordinate data
    print(f"\nGeospatial Analysis:")
    print("-" * 20)
    
    # Extract coordinates for analysis
    latitudes = [station[1] for station in weather_stations]
    longitudes = [station[2] for station in weather_stations]
    elevations = [station[3] for station in weather_stations]
    
    print(f"  Network coverage:")
    print(f"    Latitude range: {min(latitudes):.2f}°N to {max(latitudes):.2f}°N")
    print(f"    Longitude range: {max(longitudes):.2f}°W to {min(longitudes):.2f}°W")
    print(f"    Elevation range: {min(elevations)}m to {max(elevations)}m")
    
    # Tuple unpacking for data processing
    print(f"\nData Processing with Tuple Unpacking:")
    print("-" * 40)
    
    for i, (name, lat, lon, elev, city) in enumerate(weather_stations, 1):
        # Calculate approximate distance from NYC (simplified calculation)
        nyc_lat, nyc_lon = 40.7829, -73.9654
        lat_diff = abs(lat - nyc_lat)
        lon_diff = abs(lon - nyc_lon)
        approx_distance = ((lat_diff**2 + lon_diff**2)**0.5) * 111  # Rough km conversion
        
        print(f"  Station {i}: {name}")
        print(f"    Approximate distance from NYC: {approx_distance:.0f} km")


def demonstrate_data_structure_selection():
    """
    Demonstrate choosing appropriate data structures for different scenarios.
    
    Shows decision-making process for selecting the right data structure
    based on access patterns, mutability needs, and performance requirements.
    """
    print("\n" + "=" * 60)
    print("SECTION 5: CHOOSING THE RIGHT DATA STRUCTURE")
    print("=" * 60)
    
    print("Scenario-Based Data Structure Selection:")
    print("-" * 42)
    
    # Scenario 1: Time series analysis
    print("1. Time Series Analysis:")
    print("   Need: Ordered sequence, indexing, slicing")
    print("   Choice: List")
    
    time_series_data = [23.1, 24.5, 22.8, 25.1, 26.3]  # List for ordered data
    print(f"   Example: Temperature readings {time_series_data}")
    print(f"   Last 3 readings: {time_series_data[-3:]}")
    
    # Scenario 2: Patient record lookup
    print("\n2. Patient Record Lookup:")
    print("   Need: Fast access by ID, structured data")
    print("   Choice: Dictionary")
    
    patient_records = {  # Dictionary for fast lookup
        'P001': {'name': 'John Doe', 'age': 45},
        'P002': {'name': 'Jane Smith', 'age': 52}
    }
    print(f"   Example: Quick lookup P001 -> {patient_records['P001']['name']}")
    
    # Scenario 3: Study participant tracking
    print("\n3. Study Participant Tracking:")
    print("   Need: Unique IDs, membership testing, set operations")
    print("   Choice: Set")
    
    participants = {'P001', 'P002', 'P003', 'P005'}  # Set for unique collection
    new_participant = 'P004'
    if new_participant not in participants:
        participants.add(new_participant)
    print(f"   Example: Unique participants: {len(participants)}")
    
    # Scenario 4: Geographic coordinates
    print("\n4. Geographic Coordinates:")
    print("   Need: Immutable position, hashable for dict keys")
    print("   Choice: Tuple")
    
    station_location = (40.7829, -73.9654)  # Tuple for immutable coordinates
    location_data = {station_location: 'NYC Central Park'}
    print(f"   Example: Location {station_location} -> {location_data[station_location]}")
    
    # Performance comparison demonstration
    print(f"\nPerformance Characteristics:")
    print("-" * 30)
    
    print("  List: O(1) append, O(n) search, O(1) index access")
    print("  Dictionary: O(1) average access, O(1) insert/delete")
    print("  Set: O(1) average membership test, O(1) add/remove")
    print("  Tuple: O(1) access, immutable (no insert/delete)")


# =============================================================================
# SECTION 2: GIT WORKFLOW DEMONSTRATIONS
# =============================================================================

def demonstrate_git_basics():
    """
    Demonstrate basic Git operations in the context of data science projects.
    
    Shows how to initialize repositories, make commits, and track changes
    in analytical work.
    """
    print("\n" + "=" * 60)
    print("SECTION 6: GIT FUNDAMENTALS FOR DATA SCIENCE")
    print("=" * 60)
    
    print("Git Repository Lifecycle:")
    print("-" * 27)
    
    # Simulate Git workflow steps
    git_workflow_steps = [
        ("git init", "Initialize new repository", "Creates .git/ directory"),
        ("git add file.py", "Stage changes", "Prepares changes for commit"),
        ("git commit -m 'message'", "Save changes", "Creates permanent snapshot"),
        ("git status", "Check repository state", "Shows working tree status"),
        ("git log", "View commit history", "Shows all previous commits"),
        ("git diff", "See changes", "Compare working tree to last commit")
    ]
    
    for command, description, purpose in git_workflow_steps:
        print(f"  {command:25} -> {description:20} -> {purpose}")
    
    print(f"\nData Science Project Structure with Git:")
    print("-" * 43)
    
    # Typical data science project structure
    project_structure = {
        'README.md': 'Project overview and instructions',
        'data/': 'Raw and processed data files',
        'scripts/': 'Analysis and processing scripts',
        'notebooks/': 'Jupyter notebooks for exploration',
        'results/': 'Generated outputs and reports',
        'documentation/': 'Additional project documentation',
        '.gitignore': 'Files to exclude from version control'
    }
    
    for item, description in project_structure.items():
        print(f"  {item:20} -> {description}")
    
    # Sample commit messages for data science workflow
    print(f"\nProfessional Commit Message Examples:")
    print("-" * 39)
    
    commit_examples = [
        "Initial project structure with data pipeline skeleton",
        "Add data loading and validation functions",
        "Implement temperature analysis with statistical tests",
        "Fix outlier detection algorithm for edge cases", 
        "Add comprehensive error handling to data processor",
        "Update documentation with usage examples and API reference"
    ]
    
    for i, message in enumerate(commit_examples, 1):
        print(f"  Commit {i}: {message}")
    
    print(f"\nGit Best Practices for Data Scientists:")
    print("-" * 40)
    
    best_practices = [
        "Commit early and often - small, focused changes",
        "Write descriptive commit messages explaining 'why'",
        "Use branches for experimental analysis approaches",
        "Never commit sensitive data or large datasets",
        "Include .gitignore for data files and outputs",
        "Tag releases and major milestones"
    ]
    
    for practice in best_practices:
        print(f"  • {practice}")


def demonstrate_git_branching():
    """
    Demonstrate Git branching workflow for experimental data analysis.
    
    Shows how branches enable safe experimentation while maintaining
    stable analysis pipelines.
    """
    print("\n" + "=" * 60)
    print("SECTION 7: BRANCHING FOR EXPERIMENTAL ANALYSIS")
    print("=" * 60)
    
    print("Branching Workflow for Data Science:")
    print("-" * 36)
    
    # Branching strategy explanation
    branching_strategy = [
        ("main", "Stable, working analysis pipeline", "Always deployable"),
        ("feature/clustering", "Experimental clustering analysis", "Safe to experiment"),
        ("feature/visualization", "New visualization methods", "Parallel development"),
        ("hotfix/data-loading", "Critical bug fixes", "Quick fixes to main"),
        ("research/deep-learning", "Exploratory research", "Long-term experiments")
    ]
    
    print("Branch Types and Purposes:")
    for branch, purpose, note in branching_strategy:
        print(f"  {branch:20} -> {purpose:30} ({note})")
    
    # Simulated branching workflow
    print(f"\nExperimental Analysis Workflow:")
    print("-" * 33)
    
    workflow_steps = [
        "git checkout main                    # Start from stable version",
        "git checkout -b feature/clustering   # Create experimental branch", 
        "# Implement clustering analysis      # Make experimental changes",
        "git add scripts/clustering.py       # Stage experimental code",
        "git commit -m 'Add K-means clustering'  # Commit experiment",
        "# Test and validate results         # Ensure experiment works",
        "git checkout main                   # Return to stable branch",
        "git merge feature/clustering        # Integrate successful experiment",
        "git branch -d feature/clustering    # Clean up experimental branch"
    ]
    
    for step in workflow_steps:
        if step.startswith('#'):
            print(f"    {step}")  # Comment
        else:
            print(f"  {step}")   # Git command
    
    print(f"\nMerge Conflict Resolution:")
    print("-" * 28)
    
    # Example of merge conflict in data science context
    conflict_example = '''
    def analyze_temperature(data):
    <<<<<<< HEAD
        # Main branch: Basic statistical analysis
        stats = {
            'mean': data.mean(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max()
        }
    =======
        # Feature branch: Extended statistical analysis  
        stats = {
            'mean': data.mean(),
            'median': data.median(),
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'skewness': data.skew()
        }
    >>>>>>> feature/advanced-stats
        return stats
    '''
    
    print("Example conflict in analysis function:")
    print(conflict_example)
    
    # Resolution strategy
    resolved_example = '''
    def analyze_temperature(data):
        # Resolved: Combined both approaches
        stats = {
            'mean': data.mean(),
            'median': data.median(),    # Added from feature branch
            'std': data.std(),
            'min': data.min(),
            'max': data.max(),
            'skewness': data.skew()     # Added from feature branch
        }
        return stats
    '''
    
    print("Resolved version:")
    print(resolved_example)


def demonstrate_git_collaboration():
    """
    Demonstrate Git collaboration workflows for team data science projects.
    
    Shows how teams can work together on analysis projects using
    remote repositories and collaborative workflows.
    """
    print("\n" + "=" * 60)
    print("SECTION 8: COLLABORATIVE DATA SCIENCE WITH GIT")
    print("=" * 60)
    
    print("Team Collaboration Workflow:")
    print("-" * 30)
    
    # Collaborative workflow steps
    collaboration_steps = [
        ("Setup", [
            "git clone https://github.com/team/climate-analysis.git",
            "cd climate-analysis",
            "git remote -v    # Verify remote connection"
        ]),
        ("Daily Workflow", [
            "git pull origin main              # Get latest team changes",
            "git checkout -b feature/my-work   # Create feature branch",
            "# Make changes to analysis scripts",
            "git add scripts/my_analysis.py",
            "git commit -m 'Add precipitation correlation analysis'",
            "git push origin feature/my-work   # Share your work"
        ]),
        ("Integration", [
            "# Create Pull Request on GitHub",
            "# Team reviews your changes", 
            "git checkout main",
            "git pull origin main              # Get merged changes",
            "git branch -d feature/my-work     # Clean up local branch"
        ])
    ]
    
    for phase, commands in collaboration_steps:
        print(f"\n{phase}:")
        for command in commands:
            if command.startswith('#'):
                print(f"    {command}")
            else:
                print(f"  {command}")
    
    # Remote repository concepts
    print(f"\nRemote Repository Management:")
    print("-" * 31)
    
    remote_concepts = [
        "origin: Default name for main remote repository",
        "upstream: Original repository when working with forks",
        "push: Send your commits to remote repository",
        "pull: Download and integrate remote changes",
        "fetch: Download remote changes without integrating",
        "clone: Create local copy of remote repository"
    ]
    
    for concept in remote_concepts:
        term, description = concept.split(': ')
        print(f"  {term:10} -> {description}")
    
    print(f"\nCollaborative Best Practices:")
    print("-" * 31)
    
    collab_practices = [
        "Pull before starting work to avoid conflicts",
        "Use descriptive branch names (feature/clustering-analysis)", 
        "Make atomic commits with clear messages",
        "Test your changes before pushing",
        "Review team members' code constructively",
        "Document analysis decisions in commit messages",
        "Use issues to track bugs and feature requests"
    ]
    
    for practice in collab_practices:
        print(f"  • {practice}")


# =============================================================================
# SECTION 3: INTEGRATION DEMONSTRATIONS
# =============================================================================

def demonstrate_integration():
    """
    Demonstrate integration of data structures and Git in realistic workflow.
    
    Shows how data organization and version control work together in
    professional data science development.
    """
    print("\n" + "=" * 60)
    print("SECTION 9: INTEGRATION - DATA STRUCTURES + VERSION CONTROL")
    print("=" * 60)
    
    print("Integrated Data Science Workflow:")
    print("-" * 35)
    
    # Simulate integrated development process
    class ClimateAnalysisProject:
        """Example project showing integration of concepts."""
        
        def __init__(self):
            # Dictionary for project configuration
            self.config = {
                'data_sources': ['NYC', 'LA', 'Chicago'],
                'analysis_window': 30,  # days
                'output_format': 'json'
            }
            
            # Set for tracking processed files
            self.processed_files = set()
            
            # List for maintaining processing history
            self.processing_history = []
            
            # Dictionary for caching analysis results
            self.results_cache = {}
    
    # Project initialization (Git: Initial commit)
    project = ClimateAnalysisProject()
    print("1. Project Initialization:")
    print("   Data structures: Dictionary config, empty collections initialized")
    print("   Git: git init && git add . && git commit -m 'Initial project structure'")
    
    # Data loading phase (Git: Feature branch)  
    sample_data = [
        ('NYC', '2024-08-13', 26.7),
        ('LA', '2024-08-13', 32.2), 
        ('Chicago', '2024-08-13', 24.4)
    ]
    
    print("\n2. Data Loading (Feature Branch):")
    print("   Git: git checkout -b feature/data-loading")
    
    # Process data using appropriate structures
    for location, date, temperature in sample_data:
        # Tuple as cache key (immutable, hashable)
        cache_key = (location, date)
        
        # Dictionary for structured results
        result = {
            'location': location,
            'date': date,
            'temperature': temperature,
            'processed_at': datetime.now().isoformat()
        }
        
        # Cache results in dictionary
        project.results_cache[cache_key] = result
        
        # Track processing in ordered list
        project.processing_history.append({
            'action': 'data_loaded',
            'location': location,
            'timestamp': datetime.now().isoformat()
        })
        
        # Add to processed files set
        project.processed_files.add(f"{location}_{date}.csv")
    
    print(f"   Processed {len(sample_data)} data points")
    print(f"   Cached {len(project.results_cache)} results")
    print(f"   Git: git add . && git commit -m 'Implement data loading pipeline'")
    
    # Analysis phase (Git: Another feature branch)
    print("\n3. Analysis Implementation (Feature Branch):")
    print("   Git: git checkout -b feature/temperature-analysis")
    
    # Analyze data using data structure operations
    locations = set(item[0] for item in sample_data)  # Set for unique locations
    temperatures = [item[2] for item in sample_data]  # List for calculations
    
    analysis_results = {
        'unique_locations': len(locations),
        'avg_temperature': sum(temperatures) / len(temperatures),
        'temperature_range': (min(temperatures), max(temperatures)),
        'analysis_timestamp': datetime.now().isoformat()
    }
    
    # Store analysis in project cache
    project.results_cache['analysis_summary'] = analysis_results
    
    print(f"   Analyzed {len(locations)} unique locations")
    print(f"   Average temperature: {analysis_results['avg_temperature']:.1f}°C")
    print(f"   Git: git add . && git commit -m 'Add temperature analysis functions'")
    
    # Integration phase (Git: Merge to main)
    print("\n4. Integration and Documentation:")
    print("   Git: git checkout main")
    print("   Git: git merge feature/data-loading")
    print("   Git: git merge feature/temperature-analysis")
    
    # Final results using all data structures appropriately
    final_summary = {
        'project_config': project.config,                    # Dictionary
        'files_processed': list(project.processed_files),    # Set -> List for JSON
        'processing_steps': len(project.processing_history), # List length
        'cached_results': len(project.results_cache),        # Dictionary length
        'analysis_results': analysis_results                 # Dictionary
    }
    
    print(f"   Final summary generated with {final_summary['processing_steps']} steps")
    print(f"   Git: git add . && git commit -m 'Complete climate analysis pipeline'")
    print(f"   Git: git tag v1.0 -m 'First complete analysis release'")
    
    print(f"\nIntegration Benefits Demonstrated:")
    print("-" * 37)
    
    benefits = [
        "Data structures organize complex information efficiently",
        "Version control tracks analytical evolution systematically", 
        "Combined approach enables professional collaborative workflows",
        "Each commit represents meaningful analytical progress",
        "Branching allows safe experimentation with new methods",
        "Proper organization makes code review and validation possible"
    ]
    
    for benefit in benefits:
        print(f"  • {benefit}")


def interactive_data_structure_playground():
    """
    Interactive playground for experimenting with data structures.
    
    Provides hands-on experience with different data structures and
    their operations in a data science context.
    """
    print("\n" + "=" * 60)
    print("INTERACTIVE SECTION: DATA STRUCTURE PLAYGROUND")
    print("=" * 60)
    
    print("Welcome to the Data Structure Playground!")
    print("Experiment with different data structures for data science tasks.")
    print()
    
    # Patient data for experiments
    patients_data = [
        {'id': 'P001', 'name': 'Alice Johnson', 'age': 34, 'condition': 'diabetes'},
        {'id': 'P002', 'name': 'Bob Smith', 'age': 67, 'condition': 'hypertension'},
        {'id': 'P003', 'name': 'Carol Davis', 'age': 45, 'condition': 'diabetes'},
        {'id': 'P004', 'name': 'David Wilson', 'age': 23, 'condition': 'asthma'}
    ]
    
    while True:
        print("\nChoose an experiment:")
        print("1. List operations (time series analysis)")
        print("2. Dictionary operations (patient records)")  
        print("3. Set operations (unique conditions)")
        print("4. Tuple operations (coordinate data)")
        print("5. Performance comparison")
        print("0. Exit playground")
        
        try:
            choice = input("\nEnter your choice (0-5): ").strip()
            
            if choice == '0':
                print("Thanks for experimenting with data structures!")
                break
            
            elif choice == '1':
                print("\n--- List Experiments ---")
                temperatures = [float(x) for x in input("Enter temperatures (comma-separated): ").split(',')]
                print(f"Average: {sum(temperatures)/len(temperatures):.1f}°C")
                print(f"Latest 3: {temperatures[-3:]}")
                print(f"Above 25°C: {[t for t in temperatures if t > 25]}")
            
            elif choice == '2':
                print("\n--- Dictionary Experiments ---")
                patient_id = input("Enter patient ID to lookup: ").strip()
                patient_lookup = {p['id']: p for p in patients_data}
                if patient_id in patient_lookup:
                    p = patient_lookup[patient_id]
                    print(f"Found: {p['name']}, age {p['age']}, condition: {p['condition']}")
                else:
                    print("Patient not found")
            
            elif choice == '3':
                print("\n--- Set Experiments ---")
                print("Available conditions:", {p['condition'] for p in patients_data})
                condition = input("Enter condition to find patients: ").strip()
                matching_patients = {p['id'] for p in patients_data if p['condition'] == condition}
                print(f"Patients with {condition}: {matching_patients}")
            
            elif choice == '4':
                print("\n--- Tuple Experiments ---")
                stations = [('NYC', 40.78, -73.97), ('LA', 34.05, -118.24)]
                for name, lat, lon in stations:
                    print(f"{name}: {lat:.2f}°N, {lon:.2f}°W")
            
            elif choice == '5':
                print("\n--- Performance Comparison ---")
                import time
                
                # List vs Set membership testing
                large_list = list(range(10000))
                large_set = set(range(10000))
                
                # Time list search
                start = time.time()
                result = 9999 in large_list
                list_time = time.time() - start
                
                # Time set search  
                start = time.time()
                result = 9999 in large_set
                set_time = time.time() - start
                
                print(f"Search for item 9999:")
                print(f"  List search: {list_time:.6f} seconds")
                print(f"  Set search: {set_time:.6f} seconds")
                print(f"  Set is {list_time/set_time:.0f}x faster!")
            
            else:
                print("Invalid choice. Please enter 0-5.")
        
        except (ValueError, KeyboardInterrupt):
            print("Invalid input or interrupted. Try again.")


# =============================================================================
# MAIN EXECUTION AND COMMAND LINE INTERFACE
# =============================================================================

def main():
    """
    Main function that orchestrates the demonstrations.
    
    Provides command line interface for running specific sections
    or the complete demonstration suite.
    """
    parser = argparse.ArgumentParser(
        description="Interactive demonstrations for Lecture 2: Data Structures and Version Control",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 demo_lecture_02.py                        # Run all demonstrations
    python3 demo_lecture_02.py --section structures   # Data structures only
    python3 demo_lecture_02.py --section git          # Git workflows only
    python3 demo_lecture_02.py --interactive          # Interactive playground
        """
    )
    
    parser.add_argument('--section', 
                       choices=['structures', 'git', 'integration'],
                       help='Run specific section only')
    parser.add_argument('--interactive', action='store_true',
                       help='Include interactive playground')
    
    args = parser.parse_args()
    
    print("LECTURE 2: DATA STRUCTURES AND VERSION CONTROL")
    print("Interactive Demonstrations")  
    print("=" * 65)
    print(f"Demonstration started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Define demonstration sections
    if not args.section or args.section == 'structures':
        demonstrate_list_operations()
        demonstrate_dictionary_operations()
        demonstrate_set_operations()
        demonstrate_tuple_operations()
        demonstrate_data_structure_selection()
    
    if not args.section or args.section == 'git':
        demonstrate_git_basics()
        demonstrate_git_branching()
        demonstrate_git_collaboration()
    
    if not args.section or args.section == 'integration':
        demonstrate_integration()
    
    # Interactive section
    if args.interactive:
        interactive_data_structure_playground()
    
    print("\n" + "=" * 65)
    print("DEMONSTRATION COMPLETED")
    print("=" * 65)
    print("\nKey Takeaways:")
    print("• Choose data structures based on access patterns and requirements")
    print("• Lists for ordered data, dictionaries for lookups, sets for uniqueness")
    print("• Tuples for immutable records and coordinates")
    print("• Git enables safe experimentation and collaborative development")
    print("• Branching allows parallel development of analysis approaches")
    print("• Integration of both skills enables professional data science workflows")
    
    print(f"\nNext Steps:")
    print("1. Practice with the provided exercises")
    print("2. Set up your own Git repository for data science projects")
    print("3. Experiment with different data structures for your analyses")
    print("4. Prepare for next lecture: NumPy and Pandas Foundations")


if __name__ == "__main__":
    main()