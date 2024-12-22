#!/usr/bin/env python3
import os
import sys
import re
from pathlib import Path
import subprocess
from typing import Dict, List, Tuple

def install_requirements():
    """Install required Python packages."""
    requirements = [
        'pandas',
        'numpy',
        'matplotlib',
        'seaborn',
        'scipy',
        'statsmodels',
        'scikit-learn'
    ]
    
    print("Installing required packages...")
    for package in requirements:
        try:
            subprocess.run(['uv', 'pip', 'install', '--quiet', package], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error installing {package}: {e}")

def find_home_path(submission_dir: Path) -> str:
    """Find the common home path used in a submission's files."""
    patterns = [
        r'["\']?[A-Z]:/Users/[^/"\']*/[^/"\']*["\']?',  # Windows
        r'["\']?/Users/[^/"\']*/[^/"\']*["\']?',        # Mac
        r'["\']?/home/[^/"\']*/[^/"\']*["\']?'          # Linux
    ]
    
    # Files that might contain paths
    files_to_check = ['prepare.sh', 'analyze_visits.py', 'stats_analysis.py', 'visualize.ipynb']
    
    for file in files_to_check:
        file_path = submission_dir / file
        if file_path.exists():
            try:
                with open(file_path) as f:
                    content = f.read()
                    for pattern in patterns:
                        matches = re.findall(pattern, content)
                        if matches:
                            # Clean up the found path
                            path = matches[0].strip('"\'')
                            return path
            except Exception:
                continue
    return None

def fix_paths(submission_dir: Path, home_path: str) -> List[str]:
    """Fix hardcoded paths in submission files. Returns list of fixed files."""
    fixed_files = []
    
    # Files that might contain paths
    files_to_check = ['prepare.sh', 'analyze_visits.py', 'stats_analysis.py', 'visualize.ipynb']
    
    for file in files_to_check:
        file_path = submission_dir / file
        if file_path.exists():
            try:
                with open(file_path) as f:
                    content = f.read()
                
                # Replace absolute paths with relative paths
                new_content = content
                
                # Replace home path
                new_content = new_content.replace(home_path, '.')
                
                # Replace common nested paths
                nested_paths = [
                    'Desktop/Assignment9',
                    'Desktop/assignment9',
                    'Desktop/Exam2',
                    'Desktop/exam2',
                    '09-second-exam-*',  # Wildcard for username
                    './Desktop/Assignment9',
                    './Desktop/assignment9',
                    './Desktop/Exam2',
                    './Desktop/exam2',
                    './09-second-exam-*'  # Wildcard for username with ./
                ]
                for path in nested_paths:
                    new_content = new_content.replace(f'{path}/', '')
                    new_content = new_content.replace(path, '.')
                
                if new_content != content:
                    with open(file_path, 'w') as f:
                        f.write(new_content)
                    fixed_files.append(file)
            except Exception as e:
                print(f"Error fixing paths in {file}: {str(e)}")
                
    return fixed_files

def run_submission(submission_dir: Path) -> Dict[str, bool]:
    """Run a submission's scripts in order. Returns dict of script success status."""
    results = {
        'prepare.sh': False,
        'analyze_visits.py': False,
        'stats_analysis.py': False
    }
    
    # Set up environment variables for Python
    env = os.environ.copy()
    env['PYTHONPATH'] = str(submission_dir)  # Add submission dir to Python path
    
    # Common subprocess options
    common_opts = {
        'capture_output': True,
        'text': True,
        'env': env,
        'cwd': submission_dir  # Run from submission directory
    }
    
    # Run prepare.sh
    prepare_sh = submission_dir / 'prepare.sh'
    if prepare_sh.exists():
        print("Running prepare.sh...")
        try:
            os.chmod(prepare_sh, 0o755)  # Make executable
            result = subprocess.run(['bash', 'prepare.sh'], timeout=30, **common_opts)
            if result.returncode == 0:
                results['prepare.sh'] = True
            else:
                print(f"prepare.sh failed with code {result.returncode}")
                if result.stdout: print("stdout:", result.stdout)
                if result.stderr: print("stderr:", result.stderr)
        except subprocess.TimeoutExpired:
            print("prepare.sh timed out after 30 seconds")
        except Exception as e:
            print(f"Error running prepare.sh: {e}")
    
    # Run analyze_visits.py
    analyze_visits = submission_dir / 'analyze_visits.py'
    if analyze_visits.exists():
        print("Running analyze_visits.py...")
        try:
            result = subprocess.run(['python', 'analyze_visits.py'], timeout=60, **common_opts)  # 60 second timeout
            if result.returncode == 0:
                results['analyze_visits.py'] = True
            else:
                print(f"analyze_visits.py failed with code {result.returncode}")
                if result.stdout: print("stdout:", result.stdout)
                if result.stderr: print("stderr:", result.stderr)
        except subprocess.TimeoutExpired:
            print("analyze_visits.py timed out after 60 seconds")
        except Exception as e:
            print(f"Error running analyze_visits.py: {e}")
    
    # Run stats_analysis.py
    stats_analysis = submission_dir / 'stats_analysis.py'
    if stats_analysis.exists():
        print("Running stats_analysis.py...")
        try:
            result = subprocess.run(['python', 'stats_analysis.py'], timeout=120, **common_opts)
            if result.returncode == 0:
                results['stats_analysis.py'] = True
            else:
                print(f"stats_analysis.py failed with code {result.returncode}")
                if result.stdout: print("stdout:", result.stdout)
                if result.stderr: print("stderr:", result.stderr)
        except subprocess.TimeoutExpired:
            print("stats_analysis.py timed out after 120 seconds")
        except Exception as e:
            print(f"Error running stats_analysis.py: {e}")
    
    return results

def main():
    if len(sys.argv) != 2:
        print("Usage: python run_submissions.py <submissions_directory>")
        sys.exit(1)
    
    submissions_dir = Path(sys.argv[1])
    if not submissions_dir.is_dir():
        print(f"Error: {submissions_dir} is not a directory")
        sys.exit(1)
    
    # Install required packages
    install_requirements()
    
    # Process each submission
    for submission_dir in sorted(submissions_dir.glob("09-second-exam-*")):
        if submission_dir.name == ".DS_Store":
            continue
            
        username = submission_dir.name.replace("09-second-exam-", "")
        print(f"\nProcessing submission for: {username}")
        
        # Find and fix paths
        home_path = find_home_path(submission_dir)
        if home_path:
            print(f"Found home path: {home_path}")
            fixed_files = fix_paths(submission_dir, home_path)
            if fixed_files:
                print(f"Fixed paths in: {', '.join(fixed_files)}")
        
        # Run scripts
        results = run_submission(submission_dir)
        print("\nResults:")
        for script, success in results.items():
            print(f"{script}: {'Success' if success else 'Failed'}")

if __name__ == "__main__":
    main()
