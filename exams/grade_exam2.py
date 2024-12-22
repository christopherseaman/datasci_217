#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
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
        
    def find_file(self, submission_dir: Path, filename: str) -> Path:
        """Find a file recursively in the submission directory."""
        print(f"\nSearching for {filename} in {submission_dir}")
        
        # First try exact match
        for path in submission_dir.rglob(filename):
            print(f"Found exact match: {path}")
            return path
            
        # Try common variations and additional locations
        filename_lower = filename.lower()
        if filename_lower == 'prepare.sh':
            variations = [
                'clean_ms_data.sh', 'clean.sh', 'data_prep.sh', 'clean_data.sh', 
                'process.sh', 'data_preparation.sh', 'prep_data.sh'
            ]
        elif filename_lower == 'visualize.ipynb':
            variations = [
                'vizualize.ipynb',
                'visualization.ipynb', 
                'viz.ipynb',
                'plots.ipynb',
                'figures.ipynb',
                'analysis.ipynb',
                'q4.ipynb',
                'question4.ipynb',
                'visualisations.ipynb',  # British spelling
                'visualise.ipynb',
                'plotting.ipynb',
                'data_visualization.ipynb',
                'ms_visualization.ipynb',
                'walking_speed_viz.ipynb',
                'cost_analysis.ipynb',
                # Also look for Python files with visualization code
                'visualize.py',
                'visualization.py',
                'viz.py',
                'plots.py',
                'figures.py',
                'visualizevisualize.ipynb',
                'plotting.py'
            ]
        elif filename_lower == 'readme.md':
            variations = [
                'README.md', 'ReadMe.md', 'README.txt', 'readme.txt', 
                'documentation.md', 'documentation.txt', 'report.md', 'report.txt'
            ]
        elif filename_lower == 'ms_data.csv':
            variations = [
                'msdata.csv', 'ms-data.csv', 'ms_clean_data.csv', 'clean_data.csv',
                'data.csv', 'cleaned_data.csv', 'processed_data.csv', 'ms_clean.csv',
                'clean_ms.csv', 'ms.csv', 'visits.csv', 'patient_data.csv',
                'final_data.csv', 'output.csv'
            ]
        elif filename_lower == 'insurance.lst':
            variations = [
                'insurance.txt', 'insurance_categories.txt', 'insurance_categories.lst',
                'insurance_types.txt', 'insurance_types.lst', 'categories.lst',
                'categories.txt', 'insurance_list.txt', 'insurance_list.lst',
                'insurances.txt', 'insurances.lst', 'insurance_data.txt',
                'insurance_data.lst', 'insurance_info.txt', 'insurance_info.lst'
            ]
        elif filename_lower == 'analyze_visits.py':
            variations = [
                'analyze.py', 'analysis.py', 'visit_analysis.py', 'process_visits.py',
                'data_analysis.py', 'visits.py', 'q2.py', 'question2.py',
                'analyze_data.py', 'process_data.py', 'main.py',
                # British spelling variations
                'analyse_visits.py', 'analyse.py', 'analyse_data.py'
            ]
        elif filename_lower == 'stats_analysis.py':
            variations = [
                'statistics.py', 'statistical_analysis.py', 'stats.py', 
                'statistical_tests.py', 'q3.py', 'question3.py', 'analysis_stats.py',
                'stat_tests.py', 'statistical.py'
            ]
        else:
            variations = []
            
        # Common subdirectories to check
        subdirs = ['', 'src/', 'source/', 'code/', 'scripts/', 'python/', 'notebooks/']
        
        print(f"Checking variations: {variations}")
        for subdir in subdirs:
            for variant in variations:
                full_path = submission_dir / subdir / variant
                if full_path.exists():
                    print(f"Found variant {variant} in {subdir}: {full_path}")
                    return full_path
                    
                # Also try recursive search for the variant
                for path in submission_dir.rglob(variant):
                    print(f"Found variant {variant} recursively: {path}")
                    return path
                
        print(f"No match found for {filename}")
        return None
        
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
        print(f"\nDetailed grading for {submission_dir}")
        
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
        
        # Check required files exist and print what was found
        required_files = [
            'prepare.sh',
            'ms_data.csv',
            'insurance.lst',
            'analyze_visits.py',
            'stats_analysis.py',
            'visualize.ipynb',
            'readme.md'
        ]
        
        print("Found files:")
        files = {}
        for file in required_files:
            found_file = self.find_file(submission_dir, file.lower())
            files[file] = found_file
            print(f"  {file}: {found_file}")
            
        # If no files found, return zeros
        if not any(files.values()):
            print("No required files found!")
            return scores

        # Grade each question and print scores
        print("\nGrading Q1...")
        q1_scores = self.grade_q1(submission_dir, files)
        scores.update(q1_scores)
        print(f"Q1 scores: {q1_scores}")
        
        print("\nGrading Q2...")
        q2_scores = self.grade_q2(submission_dir, files)
        scores.update(q2_scores)
        print(f"Q2 scores: {q2_scores}")
        
        print("\nGrading Q3...")
        q3_scores = self.grade_q3(submission_dir, files)
        scores.update(q3_scores)
        print(f"Q3 scores: {q3_scores}")
        
        print("\nGrading Q4...")
        q4_scores = self.grade_q4(submission_dir, files)
        scores.update(q4_scores)
        print(f"Q4 scores: {q4_scores}")
        
        print("\nChecking for bonus points...")
        bonus_scores = self.grade_bonus(submission_dir, files)
        scores.update(bonus_scores)
        print(f"Bonus scores: {bonus_scores}")
        
        # Calculate totals
        scores['q1_total'] = sum(scores[k] for k in ['1.1', '1.2', '1.3', '1.4'])
        scores['q2_total'] = sum(scores[k] for k in ['2.1', '2.2', '2.3'])
        scores['q3_total'] = sum(scores[k] for k in ['3.1', '3.2', '3.3'])
        scores['q4_total'] = sum(scores[k] for k in ['4.1', '4.2', '4.3'])
        scores['bonus_total'] = scores['bonus']
        scores['total'] = scores['q1_total'] + scores['q2_total'] + scores['q3_total'] + scores['q4_total'] + scores['bonus']
        
        print(f"\nFinal scores: {scores}")
        return scores

    def grade_q1(self, submission_dir: Path, files: Dict[str, Path]) -> Dict:
        """Grade Question 1: Data Preparation."""
        scores = {'1.1': 0, '1.2': 0, '1.3': 0, '1.4': 0}
        
        print("\nGrading Q1:")
        prepare_sh = files['prepare.sh']
        insurance_lst = files['insurance.lst']
        
        # 1.1: Check if ms_data_dirty.csv was used (3 pts)
        if prepare_sh and prepare_sh.exists():
            with open(prepare_sh) as f:
                content = f.read().lower()
                if any(x in content for x in [
                    'ms_data_dirty.csv',
                    'generate_dirty_data.py',
                    'python.*generate',
                    './generate'
                ]):
                    scores['1.1'] = 3
                    print("  1.1: ✓ Found data generation (3/3)")
                else:
                    print("  1.1: ✗ Missing data generation (0/3)")
        
        # 1.2: Data cleaning operations (10 pts)
        if prepare_sh and prepare_sh.exists():
            with open(prepare_sh) as f:
                content = f.read().lower()
                points = 0
                
                # Track which components were found
                found = []
                
                # Remove comments (2 pts)
                if any(x in content for x in ['grep -v "#"', 'grep.*#', 'sed.*#', '/^#/d']):
                    points += 2
                    found.append("comments")
                
                # Remove empty lines (2 pts)
                if any(x in content for x in ['grep -v "^$"', 'sed.*^$', '/^$/d', 'empty']):
                    points += 2
                    found.append("empty lines")
                
                # Handle commas (2 pts)
                if any(x in content for x in ['s/,,*/,/g', 's/,,/,/g', 'extra comma']):
                    points += 2
                    found.append("commas")
                
                # Extract columns (2 pts)
                if any(x in content for x in ['cut -d', 'cut.*-f', 'f1,2,4,5,6']):
                    points += 2
                    found.append("columns")
                
                # Documentation (2 pts)
                if any(x in content for x in ['step', 'clean', '#.*input', 'echo']):
                    points += 2
                    found.append("documentation")
                    
                scores['1.2'] = points
                print(f"  1.2: Found {', '.join(found)} ({points}/10)")
        
        # 1.3: Insurance categories (3 pts)
        if insurance_lst and insurance_lst.exists():
            with open(insurance_lst) as f:
                content = f.read().lower()
                lines = [line.strip() for line in content.split('\n') if line.strip()]
                
                points = 0
                if 'insurance_type' in lines or 'insurance' in lines:  # Header
                    points += 1
                if len(lines) >= 3:  # At least 3 categories
                    points += 1
                if len(set(lines)) == len(lines):  # All unique
                    points += 1
                    
                scores['1.3'] = points
                print(f"  1.3: Found {len(lines)} insurance categories ({points}/3)")
        
        # 1.4: Data summary (4 pts)
        if prepare_sh and prepare_sh.exists():
            with open(prepare_sh) as f:
                content = f.read().lower()
                points = 0
                
                found = []
                if any(x in content for x in ['wc -l', 'count', 'total.*visit']):
                    points += 2
                    found.append("visit count")
                if any(x in content for x in ['head', 'first.*record', 'display.*data']):
                    points += 2
                    found.append("data preview")
                    
                scores['1.4'] = points
                print(f"  1.4: Found {', '.join(found)} ({points}/4)")
        
        return scores

    def grade_q2(self, submission_dir: Path, files: Dict[str, Path]) -> Dict:
        """Grade Question 2: Data Analysis."""
        scores = {'2.1': 0, '2.2': 0, '2.3': 0}
        
        if not (files['analyze_visits.py'] and files['analyze_visits.py'].exists()):
            return scores
            
        with open(files['analyze_visits.py']) as f:
            content = f.read().lower()
            
            # 2.1: Data loading and structure (8 pts)
            points = 0
            found = []
            if 'pd.read_csv' in content or 'pandas' in content:
                points += 2
                found.append("data loading")
            
            # Proper date handling (3 pts)
            if 'to_datetime' in content:
                points += 1
                found.append("datetime")
                if 'sort_values' in content and all(x in content for x in ['patient_id', 'visit_date']):
                    points += 2
                    found.append("proper sorting")
            
            # Data validation (3 pts)
            if any(x in content for x in ['dropna', 'isnull', 'isna', 'astype', 'info()', 'describe()']):
                points += 3
                found.append("validation")
                
            scores['2.1'] = points
            print(f"  2.1: Found {', '.join(found)} ({points}/8)")
            
            # 2.2: Insurance information (9 pts)
            points = 0
            found = []
            # Reading insurance types (3 pts)
            if ('read_csv' in content and 'insurance' in content) or ('open' in content and 'insurance.lst' in content):
                points += 3
                found.append("insurance file")
            
            # Consistent assignment (3 pts)
            if 'random.seed' in content or 'np.random.seed' in content:
                points += 1
                found.append("seed")
            if 'mapping' in content or 'dict' in content:
                points += 2
                found.append("patient mapping")
            
            # Cost generation (3 pts)
            if 'cost' in content and ('dict' in content or 'mapping' in content):
                points += 2
                found.append("base costs")
            if 'random' in content and ('normal' in content or 'uniform' in content):
                points += 1
                found.append("variation")
                
            scores['2.2'] = points
            print(f"  2.2: Found {', '.join(found)} ({points}/9)")
            
            # 2.3: Summary statistics (8 pts)
            points = 0
            found = []
            # Education grouping (3 pts)
            if 'groupby' in content and 'education_level' in content:
                points += 3
                found.append("education stats")
            
            # Insurance grouping (3 pts)
            if 'groupby' in content and 'insurance_type' in content:
                points += 3
                found.append("insurance stats")
            
            # Age effects (2 pts)
            if 'corr' in content or ('age' in content and 'walking_speed' in content):
                points += 2
                found.append("age analysis")
                
            scores['2.3'] = points
            print(f"  2.3: Found {', '.join(found)} ({points}/8)")
            
        return scores

    def grade_q3(self, submission_dir: Path, files: Dict[str, Path]) -> Dict:
        """Grade Question 3: Statistical Analysis."""
        scores = {'3.1': 0, '3.2': 0, '3.3': 0}
        
        if not (files['stats_analysis.py'] and files['stats_analysis.py'].exists()):
            return scores
            
        with open(files['stats_analysis.py']) as f:
            content = f.read().lower()
            
            # 3.1: Walking speed analysis (8 pts)
            points = 0
            found = []
            # Mixed model / repeated measures (5 pts)
            if any(x in content for x in ['mixedlm', 'mixed', 'groups']):
                points += 3
                found.append("mixed model")
                if 'patient_id' in content:
                    points += 2
                    found.append("patient grouping")
            
            # Model diagnostics and trends (3 pts)
            if any(x in content for x in ['summary()', 'fit()', '.fit']):
                points += 2
                found.append("model fit")
                if any(x in content for x in ['conf_int', 'p-value', 'p_value', 'pvalue']):
                    points += 1
                    found.append("significance")
                    
            scores['3.1'] = points
            print(f"  3.1: Found {', '.join(found)} ({points}/8)")
            
            # 3.2: Cost analysis (8 pts)
            points = 0
            found = []
            # Basic statistics and tests (4 pts)
            if any(x in content for x in ['f_oneway', 'anova', 'ttest']):
                points += 2
                found.append("statistical test")
            if any(x in content for x in ['mean', 'std', 'describe']):
                points += 2
                found.append("descriptive stats")
            
            # Effect size (4 pts)
            if any(x in content for x in ['cohen', 'eta', 'effect']):
                points += 4
                found.append("effect size")
                    
            scores['3.2'] = points
            print(f"  3.2: Found {', '.join(found)} ({points}/8)")
            
            # 3.3: Advanced analysis (9 pts)
            points = 0
            found = []
            # Interaction analysis (5 pts)
            if '*' in content and ('education' in content or 'age' in content):
                points += 3
                found.append("interaction")
                if 'summary()' in content:
                    points += 2
                    found.append("interaction results")
            
            # Control variables and reporting (4 pts)
            if any(x in content for x in ['insurance_type', 'visit_cost']):
                points += 2
                found.append("controls")
            if any(x in content for x in ['conf_int', 'p-value', 'p_value', 'pvalue']):
                points += 2
                found.append("significance reporting")
                    
            scores['3.3'] = points
            print(f"  3.3: Found {', '.join(found)} ({points}/9)")
            
        return scores

    def grade_q4(self, submission_dir: Path, files: Dict[str, Path]) -> Dict:
        """Grade Question 4: Visualization."""
        scores = {'4.1': 0, '4.2': 0, '4.3': 0}
        
        try:
            if not (files['visualize.ipynb'] and files['visualize.ipynb'].exists()):
                return scores
                
            with open(files['visualize.ipynb']) as f:
                nb = nbformat.read(f, as_version=4)
                code_content = '\n'.join(
                    cell['source'] 
                    for cell in nb['cells'] 
                    if cell['cell_type'] == 'code'
                ).lower()
                
                # 4.1: Basic relationships (10 pts)
                points = 0
                found = []
                # Scatter plots (4 pts)
                if 'scatter' in code_content or 'lmplot' in code_content:
                    points += 4
                    found.append("scatter")
                
                # Distribution plots (3 pts)
                if any(x in code_content for x in ['hist', 'kde', 'density']):
                    points += 3
                    found.append("distributions")
                    
                # Plot customization (3 pts)
                if any(x in code_content for x in ['title', 'xlabel', 'ylabel', 'figsize']):
                    points += 3
                    found.append("customization")
                    
                scores['4.1'] = points
                print(f"  4.1: Found {', '.join(found)} ({points}/10)")
                
                # 4.2: Complex relationships (10 pts)
                points = 0
                found = []
                # Multiple variables (4 pts)
                if any(x in code_content for x in ['boxplot', 'violinplot']):
                    points += 2
                    found.append("categorical")
                if 'facetgrid' in code_content or 'subplot' in code_content:
                    points += 2
                    found.append("facets")
                    
                # Statistical visualization (3 pts)
                if any(x in code_content for x in ['regplot', 'lmplot', 'residplot']):
                    points += 3
                    found.append("statistical")
                    
                # Legend and formatting (3 pts)
                if 'legend' in code_content and any(x in code_content for x in ['style', 'palette']):
                    points += 3
                    found.append("formatting")
                    
                scores['4.2'] = points
                print(f"  4.2: Found {', '.join(found)} ({points}/10)")
                
                # 4.3: Time trends (10 pts)
                points = 0
                found = []
                # Time series basics (4 pts)
                if 'to_datetime' in code_content:
                    points += 2
                    found.append("datetime")
                if any(x in code_content for x in ['lineplot', 'plot', 'tsplot']):
                    points += 2
                    found.append("time plot")
                    
                # Grouping and aggregation (3 pts)
                if 'groupby' in code_content and any(x in code_content for x in ['month', 'year', 'date']):
                    points += 3
                    found.append("time grouping")
                    
                # Multiple series (3 pts)
                if 'hue' in code_content or len(re.findall(r'plot\(.*\)', code_content)) > 1:
                    points += 3
                    found.append("multiple series")
                    
                scores['4.3'] = points
                print(f"  4.3: Found {', '.join(found)} ({points}/10)")
                
        except Exception as e:
            print(f"Error grading visualization: {str(e)}")
            
        return scores

    def grade_bonus(self, submission_dir: Path, files: Dict[str, Path]) -> Dict:
        """Grade bonus features through comprehensive static analysis."""
        bonus_points = 0
        bonus_details = []
        
        # Advanced statistical methods (3 pts)
        for file in ['stats_analysis.py', 'analyze_visits.py']:
            if files[file] and files[file].exists():
                with open(files[file]) as f:
                    content = f.read().lower()
                    
                    # Advanced statistical libraries (1 pt)
                    advanced_libs = [
                        'statsmodels', 'sklearn', 'keras', 'tensorflow', 'torch', 
                        'xgboost', 'lightgbm', 'scipy.stats'
                    ]
                    lib_matches = [lib for lib in advanced_libs if lib in content]
                    if lib_matches:
                        bonus_points += 1
                        bonus_details.extend(lib_matches)
                    
                    # Advanced statistical techniques (2 pts)
                    advanced_techniques = [
                        'mixedlm', 'mixed effects', 'multilevel', 
                        'cross_val', 'bootstrap', 'regularization', 'ridge', 'lasso', 
                        'bayesian', 'monte carlo', 'permutation', 'bootstrapping',
                        'anova', 'f_oneway', 'interaction', 'regression', 
                        'correlation', 'covariance', 'confidence interval',
                        'hierarchical', 'mixed effects', 'multilevel'
                    ]
                    technique_matches = [tech for tech in advanced_techniques if tech in content]
                    if technique_matches:
                        bonus_points += min(2, len(technique_matches))
                        bonus_details.extend(technique_matches)
                        
        # Interactive visualizations (3 pts)
        if files['visualize.ipynb'] and files['visualize.ipynb'].exists():
            with open(files['visualize.ipynb']) as f:
                nb = nbformat.read(f, as_version=4)
                code_content = '\n'.join(
                    cell['source'] 
                    for cell in nb['cells'] 
                    if cell['cell_type'] == 'code'
                ).lower()
                
                # Advanced visualization techniques (2 pts)
                advanced_viz_techniques = [
                    'facetgrid', 'pairplot', 'subplot', 'multiple subplots', 
                    'multiple series', 'interaction plot', 'time series', 
                    'error bars', 'confidence interval', 'regression line'
                ]
                viz_matches = [tech for tech in advanced_viz_techniques if tech in code_content]
                if viz_matches:
                    bonus_points += min(2, len(viz_matches))
                    bonus_details.extend(viz_matches)
                
                # Interactive plotting libraries (1 pt)
                interactive_libs = ['plotly', 'bokeh', 'altair', 'holoviews', 'hvplot', 'dash', 'streamlit']
                lib_matches = [lib for lib in interactive_libs if lib in code_content]
                if lib_matches:
                    bonus_points += 1
                    bonus_details.extend(lib_matches)
                
                # Widgets and interactivity (1 pt)
                if any(x in code_content for x in ['widget', 'interactive', 'ipywidgets', 'interact']):
                    bonus_points += 1
                    bonus_details.append("Interactive Widgets")
                    
        # Additional pattern analysis (2 pts)
        for file in ['stats_analysis.py', 'analyze_visits.py']:
            if files[file] and files[file].exists():
                with open(files[file]) as f:
                    content = f.read().lower()
                    
                    # Advanced analysis techniques (2 pts)
                    pattern_techniques = [
                        'cluster', 'pca', 'decomposition', 'seasonal', 'cyclical', 
                        'time series', 'fourier', 'wavelet', 'anomaly detection',
                        'correlation', 'cointegration', 'spectral analysis',
                        'random seed', 'consistent assignment', 'variation generation'
                    ]
                    technique_matches = [tech for tech in pattern_techniques if tech in content]
                    if technique_matches:
                        bonus_points += min(2, len(technique_matches))
                        bonus_details.extend(technique_matches)
                        
        # Command-line argument parsing (2 pts)
        for file in ['prepare.sh', 'analyze_visits.py', 'stats_analysis.py']:
            if files[file] and files[file].exists():
                with open(files[file]) as f:
                    content = f.read().lower()
                    
                    # Argument parsing (2 pts)
                    if 'argparse' in content or 'click' in content or 'sys.argv' in content:
                        bonus_points += 1
                        bonus_details.append("Argument Parsing")
                        
                        if '--help' in content or 'add_argument' in content:
                            bonus_points += 1
                            bonus_details.append("Help/Advanced Argument Parsing")
                        break
        
        print(f"Bonus Details: {bonus_details}")
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
