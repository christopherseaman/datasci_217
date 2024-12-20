#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import subprocess
import json
from pathlib import Path
import re
from typing import Dict, List, Tuple
import ast
import nbformat
from nbconvert import PythonExporter

class ExamGrader:
    def __init__(self, submissions_dir: str):
        self.submissions_dir = Path(submissions_dir)
        self.results = []
        
    def grade_all(self):
        """Grade all submissions in the directory."""
        for submission_dir in sorted(self.submissions_dir.glob("09-second-exam-*")):
            if submission_dir.name == ".DS_Store":
                continue
            try:
                username = submission_dir.name.replace("09-second-exam-", "")
                print(f"\nGrading submission for: {username}")
                scores = self.grade_submission(submission_dir)
                self.results.append({
                    "username": username,
                    **scores
                })
            except Exception as e:
                print(f"Error grading {submission_dir}: {str(e)}")
                
    def save_results(self, output_file: str):
        """Save results to TSV file."""
        df = pd.DataFrame(self.results)
        # Ensure columns match required format
        columns = ['username', 'total', 'q1_total', 'q2_total', 'q3_total', 'q4_total', 
                  'bonus_total', '1.1', '1.2', '1.3', '1.4', '2.1', '2.2', '2.3',
                  '3.1', '3.2', '3.3', '4.1', '4.2', '4.3', 'bonus']
        df = df.reindex(columns=columns)
        df.to_csv(output_file, sep='\t', index=False)

    def grade_submission(self, submission_dir: Path) -> Dict:
        """Grade a single submission."""
        scores = {
            'total': 0,
            'q1_total': 0,
            'q2_total': 0,
            'q3_total': 0,
            'q4_total': 0,
            'bonus_total': 0,
            '1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0,
            '2.1': 0, '2.2': 0, '2.3': 0,
            '3.1': 0, '3.2': 0, '3.3': 0,
            '4.1': 0, '4.2': 0, '4.3': 0,
            'bonus': 0
        }
        
        # Check required files exist
        required_files = [
            'prepare.sh',
            'ms_data.csv',
            'insurance.lst',
            'analyze_visits.py',
            'stats_analysis.py',
            'visualize.ipynb',
            'readme.md'
        ]
        
        for file in required_files:
            if not (submission_dir / file).exists():
                print(f"Missing required file: {file}")
                return scores

        # Grade Q1: Data Preparation
        scores.update(self.grade_q1(submission_dir))
        
        # Grade Q2: Data Analysis
        scores.update(self.grade_q2(submission_dir))
        
        # Grade Q3: Statistical Analysis
        scores.update(self.grade_q3(submission_dir))
        
        # Grade Q4: Data Visualization
        scores.update(self.grade_q4(submission_dir))
        
        # Grade Bonus Features
        scores.update(self.grade_bonus(submission_dir))
        
        # Calculate totals
        scores['q1_total'] = sum(scores[k] for k in ['1.1', '1.2', '1.3', '1.4'])
        scores['q2_total'] = sum(scores[k] for k in ['2.1', '2.2', '2.3'])
        scores['q3_total'] = sum(scores[k] for k in ['3.1', '3.2', '3.3'])
        scores['q4_total'] = sum(scores[k] for k in ['4.1', '4.2', '4.3'])
        scores['bonus_total'] = scores['bonus']
        scores['total'] = scores['q1_total'] + scores['q2_total'] + scores['q3_total'] + scores['q4_total']
        
        return scores

    def grade_q1(self, submission_dir: Path) -> Dict:
        """Grade Question 1: Data Preparation."""
        scores = {'1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0}
        
        # 1.1: Check if ms_data_dirty.csv was created (3 pts)
        prepare_sh = submission_dir / 'prepare.sh'
        if prepare_sh.exists():
            with open(prepare_sh) as f:
                content = f.read()
                if 'generate_dirty_data.py' in content:
                    scores['1.1'] = 3
        
        # 1.2: Data cleaning operations (10 pts)
        ms_data = submission_dir / 'ms_data.csv'
        if ms_data.exists():
            try:
                df = pd.read_csv(ms_data)
                points = 0
                
                # Check columns (2 pts)
                expected_cols = ['patient_id', 'visit_date', 'age', 'education_level', 'walking_speed']
                if list(df.columns) == expected_cols:
                    points += 2
                
                # Check for no comments/empty lines (2 pts)
                with open(ms_data) as f:
                    lines = f.readlines()
                    if not any(line.startswith('#') for line in lines):
                        points += 1
                    if not any(line.strip() == '' for line in lines):
                        points += 1
                
                # Check data validation (4 pts)
                if df['walking_speed'].between(2.0, 8.0).all():
                    points += 2
                if not df.isna().any().any():
                    points += 2
                
                # Documentation check (1 pt)
                readme = submission_dir / 'readme.md'
                if readme.exists():
                    with open(readme) as f:
                        content = f.read().lower()
                        if 'clean' in content and 'data' in content:
                            points += 1
                
                scores['1.2'] = points
            except Exception as e:
                print(f"Error checking ms_data.csv: {str(e)}")
        
        # 1.3: Insurance categories (3 pts)
        insurance_lst = submission_dir / 'insurance.lst'
        if insurance_lst.exists():
            try:
                with open(insurance_lst) as f:
                    categories = [line.strip() for line in f if line.strip()]
                    points = 0
                    
                    # At least 3 categories (1 pt)
                    if len(categories) >= 3:
                        points += 1
                    
                    # Proper formatting (1 pt)
                    if all(cat.isalnum() or ' ' in cat for cat in categories):
                        points += 1
                    
                    # Documentation (1 pt)
                    readme = submission_dir / 'readme.md'
                    if readme.exists():
                        with open(readme) as f:
                            content = f.read().lower()
                            if 'insurance' in content and 'categor' in content:
                                points += 1
                    
                    scores['1.3'] = points
            except Exception as e:
                print(f"Error checking insurance.lst: {str(e)}")
        
        # 1.4: Data summary (4 pts)
        analyze_visits = submission_dir / 'analyze_visits.py'
        if analyze_visits.exists():
            try:
                with open(analyze_visits) as f:
                    content = f.read()
                    points = 0
                    
                    # Check for comprehensive statistics (2 pts)
                    if 'describe()' in content or 'agg' in content or 'groupby' in content:
                        points += 2
                    
                    # Check for data quality metrics (2 pts)
                    if 'isna()' in content or 'isnull()' in content or 'duplicated()' in content:
                        points += 2
                    
                    scores['1.4'] = points
            except Exception as e:
                print(f"Error checking analyze_visits.py: {str(e)}")
        
        return scores

    def grade_q2(self, submission_dir: Path) -> Dict:
        """Grade Question 2: Data Analysis."""
        scores = {'2.1': 0, '2.2': 0, '2.3': 0}
        
        analyze_visits = submission_dir / 'analyze_visits.py'
        if analyze_visits.exists():
            try:
                with open(analyze_visits) as f:
                    content = f.read()
                    
                    # 2.1: Data loading and structure (8 pts)
                    points = 0
                    if 'try' in content and 'except' in content:  # Error handling
                        points += 2
                    if 'pd.to_datetime' in content:  # Date conversion
                        points += 2
                    if 'sort_values' in content:  # Sorting
                        points += 2
                    if 'astype' in content or 'dtype' in content:  # Data types
                        points += 2
                    scores['2.1'] = points
                    
                    # 2.2: Insurance information (9 pts)
                    points = 0
                    if 'read' in content and 'insurance' in content:  # Reads insurance types
                        points += 2
                    if 'groupby' in content and 'patient_id' in content:  # Consistent assignment
                        points += 3
                    if 'random' in content and ('cost' in content or 'price' in content):  # Cost generation
                        points += 4
                    scores['2.2'] = points
                    
                    # 2.3: Summary statistics (8 pts)
                    points = 0
                    if 'education_level' in content and 'walking_speed' in content:
                        points += 3
                    if 'insurance' in content and ('cost' in content or 'price' in content):
                        points += 3
                    if 'age' in content and 'walking_speed' in content:
                        points += 2
                    scores['2.3'] = points
            except Exception as e:
                print(f"Error checking analyze_visits.py: {str(e)}")
        
        return scores

    def grade_q3(self, submission_dir: Path) -> Dict:
        """Grade Question 3: Statistical Analysis."""
        scores = {'3.1': 0, '3.2': 0, '3.3': 0}
        
        stats_analysis = submission_dir / 'stats_analysis.py'
        if stats_analysis.exists():
            try:
                with open(stats_analysis) as f:
                    content = f.read()
                    
                    # 3.1: Walking speed analysis (8 pts)
                    points = 0
                    if 'regression' in content or 'lm' in content:  # Multiple regression
                        points += 3
                    if 'repeated' in content or 'mixed' in content:  # Repeated measures
                        points += 3
                    if 'trend' in content or 'time' in content:  # Trend testing
                        points += 2
                    scores['3.1'] = points
                    
                    # 3.2: Cost analysis (8 pts)
                    points = 0
                    if 'anova' in content or 'ttest' in content:  # Insurance effect
                        points += 3
                    if 'normality' in content or 'shapiro' in content:  # Distribution analysis
                        points += 2
                    if 'effect_size' in content or 'cohen' in content:  # Effect size
                        points += 3
                    scores['3.2'] = points
                    
                    # 3.3: Advanced analysis (9 pts)
                    points = 0
                    if 'interaction' in content:  # Interaction analysis
                        points += 3
                    if 'confounder' in content or 'covariate' in content:  # Confounders
                        points += 3
                    if 'summary()' in content or 'report' in content:  # Statistical reporting
                        points += 3
                    scores['3.3'] = points
            except Exception as e:
                print(f"Error checking stats_analysis.py: {str(e)}")
        
        return scores

    def grade_q4(self, submission_dir: Path) -> Dict:
        """Grade Question 4: Data Visualization."""
        scores = {'4.1': 0, '4.2': 0, '4.3': 0}
        
        visualize_ipynb = submission_dir / 'visualize.ipynb'
        if visualize_ipynb.exists():
            try:
                with open(visualize_ipynb) as f:
                    nb = nbformat.read(f, as_version=4)
                    
                    # Convert notebook to Python code for easier analysis
                    exporter = PythonExporter()
                    code, _ = exporter.from_notebook_node(nb)
                    
                    # 4.1: Walking speed visualizations (10 pts)
                    points = 0
                    if 'scatter' in code and 'age' in code and 'walking_speed' in code:
                        points += 3
                    if 'boxplot' in code and 'education' in code:
                        points += 3
                    if 'interaction' in code or 'facet' in code:
                        points += 4
                    scores['4.1'] = points
                    
                    # 4.2: Cost visualizations (10 pts)
                    points = 0
                    if 'bar' in code and 'insurance' in code:
                        points += 3
                    if 'boxplot' in code and 'cost' in code:
                        points += 3
                    if 'errorbar' in code or 'confidence' in code:
                        points += 4
                    scores['4.2'] = points
                    
                    # 4.3: Combined visualizations (10 pts)
                    points = 0
                    if 'pairplot' in code or 'scatter_matrix' in code:
                        points += 3
                    if 'facet' in code or 'subplot' in code:
                        points += 4
                    if 'time' in code or 'trend' in code:
                        points += 3
                    scores['4.3'] = points
            except Exception as e:
                print(f"Error checking visualize.ipynb: {str(e)}")
        
        return scores

    def grade_bonus(self, submission_dir: Path) -> Dict:
        """Grade bonus features."""
        bonus_points = 0
        
        # Check all relevant files for bonus features
        files_to_check = [
            'stats_analysis.py',
            'visualize.ipynb',
            'analyze_visits.py',
            'prepare.sh'
        ]
        
        for file in files_to_check:
            file_path = submission_dir / file
            if file_path.exists():
                try:
                    with open(file_path) as f:
                        content = f.read().lower()
                        
                        # Advanced statistical methods (3 pts)
                        if any(method in content for method in ['machine learning', 'cross_val', 'bootstrap']):
                            bonus_points += 1
                        if 'advanced' in content and 'regression' in content:
                            bonus_points += 1
                        if 'cross_validation' in content:
                            bonus_points += 1
                        
                        # Interactive visualizations (3 pts)
                        if any(viz in content for viz in ['plotly', 'interactive', 'widget']):
                            bonus_points += 2
                        if 'bokeh' in content:
                            bonus_points += 1
                        
                        # Additional pattern analysis (2 pts)
                        if 'pattern' in content or 'cluster' in content:
                            bonus_points += 1
                        if 'novel' in content or 'additional analysis' in content:
                            bonus_points += 1
                        
                        # Command-line argument parsing (2 pts)
                        if 'argparse' in content:
                            bonus_points += 1
                        if '--help' in content or 'add_argument' in content:
                            bonus_points += 1
                except Exception as e:
                    print(f"Error checking {file} for bonus: {str(e)}")
        
        return {'bonus': min(bonus_points, 10)}  # Cap at 10 points

def main():
    if len(sys.argv) != 2:
        print("Usage: python grade_exam2.py <submissions_directory>")
        sys.exit(1)
    
    submissions_dir = sys.argv[1]
    if not os.path.isdir(submissions_dir):
        print(f"Error: {submissions_dir} is not a directory")
        sys.exit(1)
    
    grader = ExamGrader(submissions_dir)
    grader.grade_all()
    grader.save_results('exams/09-second-exam-scores.tsv')

if __name__ == "__main__":
    main()
