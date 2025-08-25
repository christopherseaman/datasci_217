#!/usr/bin/env python3
"""
Setup Verification Script

This script verifies that the student's environment is properly set up
for completing the homework assignment.
"""

import sys
import os
import importlib
from pathlib import Path


def check_python_version():
    """Check if Python version meets requirements."""
    required_version = (3, 8)
    current_version = sys.version_info[:2]
    
    print(f"🐍 Python version: {'.'.join(map(str, current_version))}")
    
    if current_version >= required_version:
        print("✅ Python version is compatible")
        return True
    else:
        print(f"❌ Python {'.'.join(map(str, required_version))} or higher is required")
        return False


def check_required_files():
    """Check if all required files are present."""
    required_files = [
        'main.py',
        'README.md',
        'requirements.txt'
    ]
    
    print("\n📁 Checking required files...")
    all_present = True
    
    for file in required_files:
        if os.path.exists(file):
            print(f"✅ {file} - found")
        else:
            print(f"❌ {file} - missing")
            all_present = False
    
    return all_present


def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'pytest',
        'typing_extensions'  # Use underscore for import
    ]
    
    optional_packages = [
        'pandas',
        'numpy',
        'matplotlib'
    ]
    
    print("\n📦 Checking required dependencies...")
    all_installed = True
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"❌ {package} - not installed")
            all_installed = False
    
    print("\n📦 Checking optional dependencies...")
    for package in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {package} - installed")
        except ImportError:
            print(f"⚪ {package} - not installed (optional)")
    
    return all_installed


def check_main_module():
    """Check if main.py can be imported without errors."""
    print("\n🔍 Checking main.py module...")
    
    try:
        import main
        print("✅ main.py imports successfully")
        
        # Check for required functions
        required_functions = [
            'calculate_average',
            'find_maximum', 
            'count_occurrences',
            'validate_input'
        ]
        
        missing_functions = []
        for func_name in required_functions:
            if hasattr(main, func_name):
                print(f"✅ {func_name} - function found")
            else:
                print(f"⚪ {func_name} - not implemented yet")
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"💡 Still need to implement: {', '.join(missing_functions)}")
        
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in main.py: {e}")
        return False
    except ImportError as e:
        print(f"❌ Import error in main.py: {e}")
        return False
    except Exception as e:
        print(f"❌ Error importing main.py: {e}")
        return False


def check_test_directory():
    """Check if test directory exists and is accessible."""
    print("\n🧪 Checking test setup...")
    
    test_dir = Path('tests')
    if test_dir.exists() and test_dir.is_dir():
        print("✅ tests/ directory found")
        
        # List test files
        test_files = list(test_dir.glob('test_*.py'))
        if test_files:
            print(f"✅ Found {len(test_files)} test files:")
            for test_file in test_files:
                print(f"   - {test_file.name}")
        else:
            print("⚪ No test files found yet")
        
        return True
    else:
        print("⚪ tests/ directory not found (will be created by instructor)")
        return True  # Not critical for initial setup


def run_basic_tests():
    """Run basic functionality tests if possible."""
    print("\n🔧 Running basic functionality tests...")
    
    try:
        import main
        
        # Test function_template if it exists
        if hasattr(main, 'function_template'):
            try:
                result = main.function_template("test", 42)
                print(f"✅ function_template works: {result}")
            except Exception as e:
                print(f"⚠️  function_template has issues: {e}")
        
        # Test DataProcessor if it exists
        if hasattr(main, 'DataProcessor'):
            try:
                processor = main.DataProcessor([1, 2, 3])
                result = processor.process()
                print(f"✅ DataProcessor works: {result}")
            except Exception as e:
                print(f"⚠️  DataProcessor has issues: {e}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Could not run basic tests: {e}")
        return False


def provide_recommendations():
    """Provide recommendations based on setup check results."""
    print("\n💡 Setup Recommendations:")
    print("1. If Python version is too old, install Python 3.8+ from python.org")
    print("2. If packages are missing, run: pip install -r requirements.txt")
    print("3. If main.py has syntax errors, check for typos and missing colons")
    print("4. Make sure to implement all required functions before submission")
    print("5. Run 'python -m pytest tests/' to check your implementation")


def main():
    """Run all setup verification checks."""
    print("🚀 Homework Assignment Setup Verification")
    print("=" * 50)
    
    checks = []
    
    # Run all checks
    checks.append(("Python Version", check_python_version()))
    checks.append(("Required Files", check_required_files()))
    checks.append(("Dependencies", check_dependencies()))
    checks.append(("Main Module", check_main_module()))
    checks.append(("Test Directory", check_test_directory()))
    checks.append(("Basic Tests", run_basic_tests()))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 VERIFICATION SUMMARY")
    print("=" * 50)
    
    passed = 0
    for check_name, result in checks:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{check_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(checks)} checks")
    
    if passed == len(checks):
        print("\n🎉 Setup verification completed successfully!")
        print("You're ready to start working on the assignment.")
    elif passed >= len(checks) - 1:
        print("\n⚠️  Setup mostly complete with minor issues.")
        print("You can start working, but address the issues above.")
    else:
        print("\n❌ Setup needs attention before you can proceed.")
        print("Please fix the issues above before starting the assignment.")
    
    provide_recommendations()
    
    return 0 if passed >= len(checks) - 1 else 1


if __name__ == "__main__":
    sys.exit(main())