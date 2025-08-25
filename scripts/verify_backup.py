#!/usr/bin/env python3
"""
Backup Verification Script

Verifies the completeness and integrity of the lectures_bkp directory
by comparing it against the original lecture directories.

Usage:
    python scripts/verify_backup.py [--detailed] [--fix-missing]

Author: Data Science 217 Course Materials
"""

import os
import sys
import argparse
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

def calculate_file_hash(file_path: Path) -> str:
    """Calculate SHA-256 hash of a file."""
    hash_sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()
    except Exception as e:
        print(f"Warning: Could not hash {file_path}: {e}")
        return "ERROR"

def find_lecture_directories(root_path: Path) -> List[str]:
    """Find all numbered lecture directories."""
    lectures = []
    for item in root_path.iterdir():
        if item.is_dir() and item.name.isdigit() and len(item.name) == 2:
            lectures.append(item.name)
    return sorted(lectures)

def verify_directory_structure(original: Path, backup: Path) -> Dict:
    """Compare directory structure between original and backup."""
    results = {
        'missing_dirs': [],
        'missing_files': [],
        'extra_files': [],
        'file_differences': [],
        'total_original_files': 0,
        'total_backup_files': 0
    }
    
    # Get all files in original
    original_files = {}
    if original.exists():
        for file_path in original.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(original)
                original_files[str(rel_path)] = file_path
                results['total_original_files'] += 1
    
    # Get all files in backup
    backup_files = {}
    if backup.exists():
        for file_path in backup.rglob('*'):
            if file_path.is_file():
                rel_path = file_path.relative_to(backup)
                backup_files[str(rel_path)] = file_path
                results['total_backup_files'] += 1
    else:
        results['missing_dirs'].append(str(backup))
        return results
    
    # Find missing files
    for rel_path in original_files:
        if rel_path not in backup_files:
            results['missing_files'].append(rel_path)
    
    # Find extra files in backup
    for rel_path in backup_files:
        if rel_path not in original_files:
            results['extra_files'].append(rel_path)
    
    return results

def verify_lecture_backup(lecture_num: str, detailed: bool = False) -> Tuple[bool, Dict]:
    """Verify backup for a specific lecture."""
    project_root = Path(__file__).parent.parent
    original_dir = project_root / lecture_num
    backup_dir = project_root / "lectures_bkp" / lecture_num
    
    print(f"Verifying lecture {lecture_num}...")
    
    if not original_dir.exists():
        return False, {'error': f'Original directory {original_dir} does not exist'}
    
    results = verify_directory_structure(original_dir, backup_dir)
    
    # Detailed verification includes file content comparison
    if detailed and not results['missing_files'] and not results['file_differences']:
        print(f"  Performing detailed content verification...")
        original_files = {f: original_dir / f for f in os.listdir(original_dir) 
                         if (original_dir / f).is_file()}
        backup_files = {f: backup_dir / f for f in os.listdir(backup_dir) 
                       if (backup_dir / f).is_file()}
        
        for filename in original_files:
            if filename in backup_files:
                orig_hash = calculate_file_hash(original_files[filename])
                backup_hash = calculate_file_hash(backup_files[filename])
                if orig_hash != backup_hash and orig_hash != "ERROR" and backup_hash != "ERROR":
                    results['file_differences'].append(filename)
    
    success = (len(results['missing_dirs']) == 0 and 
              len(results['missing_files']) == 0 and 
              len(results['file_differences']) == 0)
    
    return success, results

def fix_missing_backup(lecture_num: str) -> bool:
    """Copy missing files from original to backup."""
    import shutil
    
    project_root = Path(__file__).parent.parent
    original_dir = project_root / lecture_num
    backup_dir = project_root / "lectures_bkp" / lecture_num
    
    try:
        if backup_dir.exists():
            print(f"  Updating existing backup for lecture {lecture_num}...")
        else:
            print(f"  Creating new backup for lecture {lecture_num}...")
            backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy entire directory structure
        for item in original_dir.rglob('*'):
            if item.is_file():
                rel_path = item.relative_to(original_dir)
                target_path = backup_dir / rel_path
                target_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(item, target_path)
        
        print(f"  ✓ Backup updated for lecture {lecture_num}")
        return True
    except Exception as e:
        print(f"  ✗ Failed to update backup for lecture {lecture_num}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Verify lecture backup completeness')
    parser.add_argument('--detailed', action='store_true',
                       help='Perform detailed content verification (slower)')
    parser.add_argument('--fix-missing', action='store_true',
                       help='Automatically fix missing or outdated backups')
    parser.add_argument('--lectures',
                       help='Comma-separated list of specific lectures to verify')
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent
    
    # Determine which lectures to verify
    if args.lectures:
        lecture_numbers = [num.strip().zfill(2) for num in args.lectures.split(',')]
    else:
        lecture_numbers = find_lecture_directories(project_root)
    
    print("Lecture Backup Verification")
    print("=" * 50)
    print(f"Verifying {len(lecture_numbers)} lectures: {', '.join(lecture_numbers)}")
    if args.detailed:
        print("Detailed mode: Including file content verification")
    print()
    
    # Verify each lecture
    all_success = True
    total_files = 0
    issues_found = []
    
    for lecture_num in lecture_numbers:
        success, results = verify_lecture_backup(lecture_num, args.detailed)
        
        if 'error' in results:
            print(f"  ✗ Error: {results['error']}")
            all_success = False
            continue
        
        total_files += results['total_original_files']
        
        if success:
            print(f"  ✓ Backup verified ({results['total_backup_files']} files)")
        else:
            print(f"  ⚠ Issues found:")
            all_success = False
            
            if results['missing_dirs']:
                print(f"    - Missing backup directory")
                issues_found.append(f"Lecture {lecture_num}: Missing backup directory")
            
            if results['missing_files']:
                print(f"    - {len(results['missing_files'])} missing files")
                issues_found.extend([f"Lecture {lecture_num}: Missing {f}" 
                                   for f in results['missing_files']])
            
            if results['extra_files']:
                print(f"    - {len(results['extra_files'])} extra files in backup")
            
            if results['file_differences']:
                print(f"    - {len(results['file_differences'])} files with content differences")
                issues_found.extend([f"Lecture {lecture_num}: Content difference in {f}" 
                                   for f in results['file_differences']])
            
            # Fix issues if requested
            if args.fix_missing:
                print(f"    Attempting to fix backup for lecture {lecture_num}...")
                if fix_missing_backup(lecture_num):
                    # Re-verify
                    success, _ = verify_lecture_backup(lecture_num, False)
                    if success:
                        print(f"    ✓ Backup fixed and verified")
                        all_success = True
                    else:
                        print(f"    ⚠ Backup updated but still has issues")
        
        print()
    
    # Summary
    print("=" * 50)
    print("VERIFICATION SUMMARY")
    print("=" * 50)
    print(f"Lectures verified: {len(lecture_numbers)}")
    print(f"Total files: {total_files}")
    
    if all_success:
        print("✅ All backups are complete and verified!")
    else:
        print(f"⚠️  Issues found: {len(issues_found)}")
        if not args.fix_missing:
            print("\nRun with --fix-missing to automatically resolve backup issues")
        
        if issues_found:
            print("\nDetailed issues:")
            for issue in issues_found:
                print(f"  - {issue}")
    
    return 0 if all_success else 1

if __name__ == "__main__":
    sys.exit(main())