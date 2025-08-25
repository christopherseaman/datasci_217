#!/usr/bin/env python3
"""
Batch Lecture Conversion Script

Converts all lecture directories to the new format in one operation.
Provides progress tracking and error handling for bulk conversion.

Usage:
    python scripts/batch_convert.py [--dry-run] [--lectures 01,03,05]

Author: Data Science 217 Course Materials
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List
import subprocess
import time

def find_lecture_directories(project_root: Path) -> List[str]:
    """Find all numbered lecture directories."""
    lecture_dirs = []
    
    for item in project_root.iterdir():
        if item.is_dir() and item.name.isdigit() and len(item.name) == 2:
            lecture_dirs.append(item.name)
    
    return sorted(lecture_dirs)

def convert_lecture(lecture_num: str, dry_run: bool = False) -> bool:
    """Convert a single lecture using the conversion script."""
    script_path = Path(__file__).parent / "convert_lecture.py"
    
    cmd = ["python", str(script_path), lecture_num]
    if dry_run:
        cmd.append("--dry-run")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        print(f"✓ Successfully converted lecture {lecture_num}")
        if dry_run:
            print(f"  {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to convert lecture {lecture_num}")
        print(f"  Error: {e.stderr.strip()}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Batch convert lectures to new format')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show what would be done without making changes')
    parser.add_argument('--lectures', 
                       help='Comma-separated list of specific lectures to convert (e.g., 01,03,05)')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    # Determine which lectures to convert
    if args.lectures:
        lecture_numbers = [num.strip().zfill(2) for num in args.lectures.split(',')]
        # Validate that these directories exist
        valid_lectures = []
        for num in lecture_numbers:
            if (project_root / num).exists():
                valid_lectures.append(num)
            else:
                print(f"Warning: Lecture directory {num} does not exist, skipping")
        lecture_numbers = valid_lectures
    else:
        lecture_numbers = find_lecture_directories(project_root)
    
    if not lecture_numbers:
        print("No lecture directories found to convert")
        return
    
    print("Batch Lecture Conversion")
    print("=" * 50)
    print(f"Found {len(lecture_numbers)} lectures to convert: {', '.join(lecture_numbers)}")
    
    if args.dry_run:
        print("DRY RUN MODE - No files will be modified")
    
    print()
    
    # Convert each lecture
    successful = 0
    failed = 0
    start_time = time.time()
    
    for i, lecture_num in enumerate(lecture_numbers, 1):
        print(f"[{i}/{len(lecture_numbers)}] Converting lecture {lecture_num}...")
        
        if convert_lecture(lecture_num, args.dry_run):
            successful += 1
        else:
            failed += 1
        
        print()  # Blank line for readability
    
    # Summary
    end_time = time.time()
    duration = end_time - start_time
    
    print("=" * 50)
    print("CONVERSION SUMMARY")
    print("=" * 50)
    print(f"Total lectures: {len(lecture_numbers)}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Duration: {duration:.1f} seconds")
    
    if failed == 0:
        print("\n🎉 All conversions completed successfully!")
        if not args.dry_run:
            new_lectures_dir = project_root / "lectures_new"
            print(f"New lecture materials are available in: {new_lectures_dir}")
    else:
        print(f"\n⚠️  {failed} conversions failed. Check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())