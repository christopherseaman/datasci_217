#!/usr/bin/env python3
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import json
import subprocess
from pathlib import Path

class Exam2Grader:
    def __init__(self, submission_dir):
        self.submission_dir = Path(submission_dir)
        self.total_points = 0
        self.feedback = []
        
    def grade_q1(self, points=20):
        """Grade Question 1: Data Preparation"""
        earned = 0
        
        # Check if prepare.sh exists
        prepare_sh = self.submission_dir / 'prepare.sh'
        if not prepare_sh.exists():
            self.feedback.append("Q1: Missing prepare.sh (-20)")
            return earned
            
        # Check if script is executable
        if not os.access(prepare_sh, os.X_OK):
            self.feedback.append("Q1: prepare.sh is not executable (-5)")
            earned -= 5
            
        # Check if ms_data.csv exists
        ms_data = self.submission_dir / 'ms_data.csv'
        if not ms_data.exists():
            self.feedback.append("Q1: Missing ms_data.csv (-10)")
            return earned
            
        # Check insurance.lst exists
        insurance_lst = self.submission_dir / 'insurance.lst'
        if not insurance_lst.exists():
            self.feedback.append("Q1: Missing insurance.lst (-5)")
            earned -= 5
            
        # Validate ms_data.csv format
        try:
            df = pd.read_csv(ms_data)
            required_cols = ['patient_id', 'visit_date', 'age', 
                           'education_level', 'walking_speed']
            
            # Check columns
            if not all(col in df.columns for col in required_cols):
                self.feedback.append("Q1: ms_data.csv missing required columns (-5)")
                earned -= 5
                
            # Check walking speed range
            speeds = df['walking_speed']
            if not all((speeds >= 2.0) & (speeds <= 8.0)):
                self.feedback.append("Q1: Invalid walking speeds found (-5)")
                earned -= 5
                
            # Check for empty lines/extra commas
            with open(ms_data) as f:
                lines = f.readlines()
                if any(line.strip() == '' for line in lines):
                    self.feedback.append("Q1: Empty lines found in ms_data.csv (-2)")
                    earned -= 2
                if any(line.count(',') > 4 for line in lines):
                    self.feedback.append("Q1: Extra commas found in ms_data.csv (-2)")
                    earned -= 2
                    
        except Exception as e:
            self.feedback.append(f"Q1: Error reading ms_data.csv: {str(e)} (-10)")
            earned -= 10
            
        # Check insurance types
        try:
            with open(insurance_lst) as f:
                insurance_types = [line.strip() for line in f if line.strip()]
                if len(insurance_types) < 3:
                    self.feedback.append("Q1: Fewer than 3 insurance types defined (-5)")
                    earned -= 5
        except:
            self.feedback.append("Q1: Error reading insurance.lst (-5)")
            earned -= 5
            
        earned = max(0, points + earned)  # Don't go negative
        self.total_points += earned
        return earned

    def grade_q2(self, points=25):
        """Grade Question 2: Data Analysis"""
        earned = 0
        
        # Check if analyze_visits.py exists
        analyze_py = self.submission_dir / 'analyze_visits.py'
        if not analyze_py.exists():
            self.feedback.append("Q2: Missing analyze_visits.py (-25)")
            return earned
            
        # Try to run the analysis script
        try:
            result = subprocess.run(['python', str(analyze_py)], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.feedback.append(f"Q2: Error running analyze_visits.py: {result.stderr} (-15)")
                earned -= 15
        except Exception as e:
            self.feedback.append(f"Q2: Error executing analyze_visits.py: {str(e)} (-15)")
            earned -= 15
            
        # Check code content
        try:
            with open(analyze_py) as f:
                code = f.read()
                
                # Check for required components
                checks = {
                    'pandas': ('import pandas' in code, "Missing pandas import (-2)"),
                    'datetime': ('datetime' in code, "No datetime handling (-3)"),
                    'groupby': ('.groupby' in code, "No groupby operations (-5)"),
                    'insurance': ('insurance' in code.lower(), "No insurance analysis (-5)"),
                    'education': ('education' in code.lower(), "No education analysis (-5)")
                }
                
                for check, (passed, msg) in checks.items():
                    if not passed:
                        self.feedback.append(f"Q2: {msg}")
                        earned -= int(msg.split('(')[1].split(')')[0])
                        
        except Exception as e:
            self.feedback.append(f"Q2: Error checking analyze_visits.py: {str(e)} (-10)")
            earned -= 10
            
        earned = max(0, points + earned)  # Don't go negative
        self.total_points += earned
        return earned

    def grade_q3(self, points=25):
        """Grade Question 3: Statistical Analysis"""
        earned = 0
        
        # Check if stats_analysis.py exists
        stats_py = self.submission_dir / 'stats_analysis.py'
        if not stats_py.exists():
            self.feedback.append("Q3: Missing stats_analysis.py (-25)")
            return earned
            
        # Check code content
        try:
            with open(stats_py) as f:
                code = f.read()
                
                # Check for required components
                checks = {
                    'scipy': ('import scipy' in code, "Missing scipy import (-3)"),
                    'statsmodels': ('import statsmodels' in code, "Missing statsmodels import (-3)"),
                    'regression': ('regression' in code.lower(), "No regression analysis (-5)"),
                    'p-value': ('pvalue' in code.lower() or 'p_value' in code.lower(), 
                               "No p-value reporting (-5)"),
                    'confidence': ('confidence' in code.lower(), "No confidence intervals (-4)"),
                    'interaction': ('interaction' in code.lower(), "No interaction effects (-5)")
                }
                
                for check, (passed, msg) in checks.items():
                    if not passed:
                        self.feedback.append(f"Q3: {msg}")
                        earned -= int(msg.split('(')[1].split(')')[0])
                        
        except Exception as e:
            self.feedback.append(f"Q3: Error checking stats_analysis.py: {str(e)} (-10)")
            earned -= 10
            
        earned = max(0, points + earned)  # Don't go negative
        self.total_points += earned
        return earned

    def grade_q4(self, points=30):
        """Grade Question 4: Data Visualization"""
        earned = 0
        
        # Check if visualize.ipynb exists
        notebook = self.submission_dir / 'visualize.ipynb'
        if not notebook.exists():
            self.feedback.append("Q4: Missing visualize.ipynb (-30)")
            return earned
            
        # Try to read notebook content
        try:
            with open(notebook) as f:
                nb = json.load(f)
                
                # Check for required visualizations
                viz_checks = {
                    'scatter': ('scatter' in str(nb).lower(), "Missing scatter plot (-5)"),
                    'boxplot': ('boxplot' in str(nb).lower(), "Missing box plots (-5)"),
                    'barplot': ('barplot' in str(nb).lower(), "Missing bar plot (-5)"),
                    'seaborn': ('import seaborn' in str(nb), "Not using seaborn (-5)"),
                    'titles': ('title' in str(nb).lower(), "Missing plot titles (-5)"),
                    'labels': ('label' in str(nb).lower(), "Missing axis labels (-5)")
                }
                
                for check, (passed, msg) in viz_checks.items():
                    if not passed:
                        self.feedback.append(f"Q4: {msg}")
                        earned -= int(msg.split('(')[1].split(')')[0])
                        
        except Exception as e:
            self.feedback.append(f"Q4: Error checking visualize.ipynb: {str(e)} (-15)")
            earned -= 15
            
        earned = max(0, points + earned)  # Don't go negative
        self.total_points += earned
        return earned

    def grade_bonus(self, points=10):
        """Grade bonus points"""
        earned = 0
        
        # Check for advanced features
        advanced_features = {
            'interactive': (
                any(Path(self.submission_dir).glob('*plotly*')) or 
                'interactive' in str(list(Path(self.submission_dir).glob('*.py'))),
                "Interactive visualizations (+3)"
            ),
            'cli': (
                'argparse' in str(list(Path(self.submission_dir).glob('*.py'))),
                "Command-line argument parsing (+2)"
            ),
            'advanced_stats': (
                any('advanced' in p.read_text().lower() 
                    for p in Path(self.submission_dir).glob('*.py')),
                "Advanced statistical methods (+3)"
            ),
            'extra_patterns': (
                any('pattern' in p.read_text().lower() 
                    for p in Path(self.submission_dir).glob('*.py')),
                "Additional pattern analysis (+2)"
            )
        }
        
        for feature, (present, msg) in advanced_features.items():
            if present:
                points = int(msg.split('(')[1].split(')')[0].replace('+',''))
                earned += points
                self.feedback.append(f"Bonus: {msg}")
                
        earned = min(10, earned)  # Cap at 10 bonus points
        self.total_points += earned
        return earned

    def grade(self):
        """Grade all questions and return results"""
        results = {
            "Q1 (20pts)": self.grade_q1(),
            "Q2 (25pts)": self.grade_q2(),
            "Q3 (25pts)": self.grade_q3(),
            "Q4 (30pts)": self.grade_q4(),
            "Bonus (10pts)": self.grade_bonus()
        }
        
        # Generate report
        report = ["Exam 2 Grading Report", "=" * 50]
        report.append(f"\nTotal Points: {self.total_points}/110\n")
        report.append("Breakdown:")
        for q, points in results.items():
            report.append(f"{q}: {points}")
            
        report.append("\nFeedback:")
        for fb in self.feedback:
            report.append(f"- {fb}")
            
        return "\n".join(report)

def main():
    if len(sys.argv) != 2:
        print("Usage: python grade_exam2.py <submission_directory>")
        sys.exit(1)
        
    submission_dir = sys.argv[1]
    grader = Exam2Grader(submission_dir)
    print(grader.grade())

if __name__ == "__main__":
    main()
