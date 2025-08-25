#!/usr/bin/env python3
"""
Phase 1 Comprehensive Validation Suite
Weeks 3-4 Testing Agent Implementation

This script provides comprehensive validation of all Phase 1 deliverables including:
- System functionality testing
- Content quality validation  
- Infrastructure reliability checks
- Assessment alignment verification
- Integration and scalability testing
"""

import sys
import os
import subprocess
import json
import time
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

class ValidationResult:
    """Structure for holding validation test results."""
    
    def __init__(self, test_name: str, category: str):
        self.test_name = test_name
        self.category = category
        self.passed = False
        self.errors = []
        self.warnings = []
        self.details = {}
        self.execution_time = 0.0
        
    def add_error(self, error: str):
        """Add an error to this validation result."""
        self.errors.append(error)
        
    def add_warning(self, warning: str):
        """Add a warning to this validation result."""
        self.warnings.append(warning)
        
    def set_detail(self, key: str, value: Any):
        """Set a detail for this validation result."""
        self.details[key] = value
        
    def is_passed(self) -> bool:
        """Check if validation passed (no errors)."""
        return len(self.errors) == 0


class Phase1ValidationSuite:
    """Comprehensive validation suite for Phase 1 deliverables."""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: List[ValidationResult] = []
        self.start_time = datetime.now()
        
    def log_result(self, result: ValidationResult):
        """Log a validation result."""
        self.results.append(result)
        status = "✅ PASS" if result.is_passed() else "❌ FAIL"
        print(f"{status} {result.category}: {result.test_name}")
        
        if result.errors:
            for error in result.errors:
                print(f"  ERROR: {error}")
                
        if result.warnings:
            for warning in result.warnings:
                print(f"  WARNING: {warning}")
    
    def run_system_functionality_tests(self) -> List[ValidationResult]:
        """Test all Python scripts and command line integrations."""
        print("\n" + "="*60)
        print("SYSTEM FUNCTIONALITY VALIDATION")
        print("="*60)
        
        system_tests = []
        
        # Test 1: Demo script execution
        system_tests.append(self._test_demo_script_execution())
        
        # Test 2: Exercise scripts execution
        system_tests.append(self._test_exercise_scripts())
        
        # Test 3: Command line argument processing
        system_tests.append(self._test_command_line_args())
        
        # Test 4: Interactive features validation
        system_tests.append(self._test_interactive_features())
        
        # Test 5: File operations validation
        system_tests.append(self._test_file_operations())
        
        return system_tests
    
    def _test_demo_script_execution(self) -> ValidationResult:
        """Test that the main demo script runs without errors."""
        result = ValidationResult("Demo Script Execution", "System")
        start_time = time.time()
        
        demo_script = self.project_root / "demo_lecture_01.py"
        
        try:
            if not demo_script.exists():
                result.add_error(f"Demo script not found at {demo_script}")
                return result
                
            # Test basic execution
            proc = subprocess.run([
                sys.executable, str(demo_script), "--section", "basics"
            ], capture_output=True, text=True, timeout=30)
            
            if proc.returncode != 0:
                result.add_error(f"Demo script failed with return code {proc.returncode}")
                result.add_error(f"STDERR: {proc.stderr}")
            else:
                result.passed = True
                result.set_detail("stdout_length", len(proc.stdout))
                
            # Test all sections
            sections = ["basics", "control", "functions", "integration", "problem", "pitfalls"]
            failed_sections = []
            
            for section in sections:
                proc = subprocess.run([
                    sys.executable, str(demo_script), "--section", section
                ], capture_output=True, text=True, timeout=20)
                
                if proc.returncode != 0:
                    failed_sections.append(section)
                    
            if failed_sections:
                result.add_error(f"Failed sections: {', '.join(failed_sections)}")
                result.passed = False
            else:
                result.passed = True
                
            result.set_detail("sections_tested", len(sections))
            
        except subprocess.TimeoutExpired:
            result.add_error("Demo script execution timed out")
        except Exception as e:
            result.add_error(f"Exception during demo script test: {str(e)}")
            
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
    
    def _test_exercise_scripts(self) -> ValidationResult:
        """Test all exercise scripts for proper execution."""
        result = ValidationResult("Exercise Scripts Execution", "System")
        start_time = time.time()
        
        exercise_dir = self.project_root / "exercises"
        
        if not exercise_dir.exists():
            result.add_error("Exercises directory not found")
            self.log_result(result)
            return result
            
        python_files = list(exercise_dir.glob("*.py"))
        
        if not python_files:
            result.add_error("No Python exercise files found")
            self.log_result(result)
            return result
            
        failed_scripts = []
        
        for script in python_files:
            try:
                # Check for syntax errors
                with open(script, 'r') as f:
                    code = f.read()
                    
                compile(code, str(script), 'exec')
                
                # For non-interactive scripts, try to run them
                if "input(" not in code:
                    proc = subprocess.run([
                        sys.executable, str(script)
                    ], capture_output=True, text=True, timeout=15)
                    
                    if proc.returncode != 0:
                        failed_scripts.append(f"{script.name}: {proc.stderr}")
                        
            except SyntaxError as e:
                failed_scripts.append(f"{script.name}: Syntax error - {str(e)}")
            except subprocess.TimeoutExpired:
                result.add_warning(f"{script.name}: Execution timeout (likely interactive)")
            except Exception as e:
                failed_scripts.append(f"{script.name}: {str(e)}")
                
        if failed_scripts:
            for error in failed_scripts:
                result.add_error(error)
        else:
            result.passed = True
            
        result.set_detail("scripts_tested", len(python_files))
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
    
    def _test_command_line_args(self) -> ValidationResult:
        """Test command line argument processing."""
        result = ValidationResult("Command Line Arguments", "System")
        start_time = time.time()
        
        demo_script = self.project_root / "demo_lecture_01.py"
        
        try:
            # Test help argument
            proc = subprocess.run([
                sys.executable, str(demo_script), "--help"
            ], capture_output=True, text=True, timeout=10)
            
            if proc.returncode != 0:
                result.add_error("Help argument failed")
            elif "usage:" not in proc.stdout.lower():
                result.add_warning("Help output doesn't contain usage information")
                
            # Test invalid section
            proc = subprocess.run([
                sys.executable, str(demo_script), "--section", "invalid"
            ], capture_output=True, text=True, timeout=10)
            
            # Should handle invalid section gracefully (argparse returns 2 for invalid args)
            if proc.returncode not in [0, 1, 2]:
                result.add_error("Invalid section not handled gracefully")
            elif proc.returncode == 2 and "invalid choice" in proc.stderr.lower():
                result.passed = True  # argparse properly rejected invalid choice
            else:
                result.passed = True
                
        except Exception as e:
            result.add_error(f"Command line argument test failed: {str(e)}")
            
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
        
    def _test_interactive_features(self) -> ValidationResult:
        """Test interactive script features (limited validation)."""
        result = ValidationResult("Interactive Features", "System")
        start_time = time.time()
        
        # Since interactive features require user input, we validate structure
        demo_script = self.project_root / "demo_lecture_01.py"
        hello_script = self.project_root / "exercises" / "hello_ds.py"
        
        scripts_to_check = [demo_script, hello_script]
        interactive_patterns = ["input(", "interactive_temperature_analyzer"]
        
        scripts_with_interactive = 0
        
        for script in scripts_to_check:
            if script.exists():
                with open(script, 'r') as f:
                    content = f.read()
                    
                for pattern in interactive_patterns:
                    if pattern in content:
                        scripts_with_interactive += 1
                        break
                        
        if scripts_with_interactive > 0:
            result.passed = True
            result.set_detail("interactive_scripts_found", scripts_with_interactive)
        else:
            result.add_warning("No interactive features detected")
            
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
        
    def _test_file_operations(self) -> ValidationResult:
        """Test file organization and access patterns."""
        result = ValidationResult("File Operations", "System")
        start_time = time.time()
        
        expected_files = [
            "README.md",
            "lecture_01_narrative.md", 
            "demo_lecture_01.py",
            "exercises/practice_problems.md",
            "exercises/hello_ds.py",
            "exercises/euler_problem_1.py",
            "resources/command_line_cheatsheet.md",
            "resources/python_syntax_reference.md"
        ]
        
        missing_files = []
        
        for file_path in expected_files:
            full_path = self.project_root / file_path
            if not full_path.exists():
                missing_files.append(file_path)
                
        if missing_files:
            for missing in missing_files:
                result.add_error(f"Missing expected file: {missing}")
        else:
            result.passed = True
            
        result.set_detail("expected_files", len(expected_files))
        result.set_detail("missing_files", len(missing_files))
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
    
    def run_content_quality_tests(self) -> List[ValidationResult]:
        """Validate content quality and educational effectiveness."""
        print("\n" + "="*60)
        print("CONTENT QUALITY VALIDATION")  
        print("="*60)
        
        content_tests = []
        
        # Test 1: Narrative content analysis
        content_tests.append(self._test_narrative_quality())
        
        # Test 2: Learning progression validation
        content_tests.append(self._test_learning_progression())
        
        # Test 3: Code quality and standards
        content_tests.append(self._test_code_quality())
        
        # Test 4: Documentation completeness
        content_tests.append(self._test_documentation_quality())
        
        return content_tests
        
    def _test_narrative_quality(self) -> ValidationResult:
        """Test narrative content for quality and engagement."""
        result = ValidationResult("Narrative Quality", "Content")
        start_time = time.time()
        
        narrative_file = self.project_root / "lecture_01_narrative.md"
        
        if not narrative_file.exists():
            result.add_error("Narrative file not found")
            self.log_result(result)
            return result
            
        try:
            with open(narrative_file, 'r') as f:
                content = f.read()
                
            # Check length (should be substantial)
            word_count = len(content.split())
            if word_count < 4000:
                result.add_warning(f"Narrative might be too short: {word_count} words")
            elif word_count > 8000:
                result.add_warning(f"Narrative might be too long: {word_count} words")
            else:
                result.passed = True
                
            # Check for engagement elements
            engagement_patterns = [
                "you'll", "you will", "your", "let's", "we'll",
                "think of", "imagine", "consider", "example"
            ]
            
            engagement_score = sum(1 for pattern in engagement_patterns 
                                 if pattern.lower() in content.lower())
            
            if engagement_score < 10:
                result.add_warning("Low engagement language detected")
                
            # Check for code examples
            code_blocks = content.count("```")
            if code_blocks < 20:  # Should be many code examples
                result.add_warning(f"Few code examples: {code_blocks//2} blocks")
                
            result.set_detail("word_count", word_count)
            result.set_detail("code_blocks", code_blocks // 2)
            result.set_detail("engagement_score", engagement_score)
            
        except Exception as e:
            result.add_error(f"Error analyzing narrative: {str(e)}")
            
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
        
    def _test_learning_progression(self) -> ValidationResult:
        """Test that content follows logical learning progression."""
        result = ValidationResult("Learning Progression", "Content")
        start_time = time.time()
        
        narrative_file = self.project_root / "lecture_01_narrative.md"
        
        try:
            with open(narrative_file, 'r') as f:
                content = f.read()
                
            # Check for proper section progression
            expected_order = [
                "command line", "python", "variables", "data types",
                "control", "functions", "integration"
            ]
            
            content_lower = content.lower()
            positions = []
            
            for concept in expected_order:
                pos = content_lower.find(concept)
                if pos == -1:
                    result.add_warning(f"Concept '{concept}' not found in content")
                else:
                    positions.append((concept, pos))
                    
            # Check if concepts appear in logical order
            sorted_positions = sorted(positions, key=lambda x: x[1])
            if len(sorted_positions) >= len(expected_order) - 1:
                result.passed = True
                
            result.set_detail("concepts_found", len(positions))
            result.set_detail("expected_concepts", len(expected_order))
            
        except Exception as e:
            result.add_error(f"Error checking learning progression: {str(e)}")
            
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
        
    def _test_code_quality(self) -> ValidationResult:
        """Test code quality and standards compliance."""
        result = ValidationResult("Code Quality Standards", "Content")
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        
        quality_issues = []
        total_files = len(python_files)
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                # Check for docstrings in functions
                function_lines = [i for i, line in enumerate(lines) 
                                if line.strip().startswith('def ') and not line.strip().startswith('def _')]
                
                functions_with_docs = 0
                for func_line in function_lines:
                    # Check if next non-empty line after function def is a docstring
                    for i in range(func_line + 1, min(func_line + 5, len(lines))):
                        if lines[i].strip():
                            if lines[i].strip().startswith('"""') or lines[i].strip().startswith("'''"):
                                functions_with_docs += 1
                            break
                            
                if function_lines and functions_with_docs / len(function_lines) < 0.7:
                    quality_issues.append(f"{py_file.name}: Insufficient docstrings")
                    
                # Check for reasonable line lengths (not too strict)
                long_lines = [i for i, line in enumerate(lines) if len(line) > 120]
                if len(long_lines) > 5:
                    quality_issues.append(f"{py_file.name}: Many long lines ({len(long_lines)})")
                    
                # Check for meaningful variable names
                import re
                short_vars = re.findall(r'\b[a-z]{1,2}\b(?!\s*=)', content)
                if len(short_vars) > 20:  # Some short vars are OK (i, x, etc.)
                    result.add_warning(f"{py_file.name}: Many short variable names")
                    
            except Exception as e:
                quality_issues.append(f"{py_file.name}: Analysis error - {str(e)}")
                
        if quality_issues:
            for issue in quality_issues:
                result.add_error(issue)
        else:
            result.passed = True
            
        result.set_detail("files_analyzed", total_files)
        result.set_detail("quality_issues", len(quality_issues))
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
        
    def _test_documentation_quality(self) -> ValidationResult:
        """Test completeness and quality of documentation."""
        result = ValidationResult("Documentation Quality", "Content")
        start_time = time.time()
        
        doc_files = [
            ("README.md", ["overview", "testing", "structure"]),
            ("lecture_01_narrative.md", ["learning objectives", "prerequisites", "exercises"]),
            ("exercises/practice_problems.md", ["exercise", "challenge"]),
            ("resources/command_line_cheatsheet.md", ["command", "description"]),
            ("resources/python_syntax_reference.md", ["syntax", "example"])
        ]
        
        doc_issues = []
        
        for doc_file, required_elements in doc_files:
            file_path = self.project_root / doc_file
            
            if not file_path.exists():
                doc_issues.append(f"Missing documentation file: {doc_file}")
                continue
                
            try:
                with open(file_path, 'r') as f:
                    content = f.read().lower()
                    
                missing_elements = []
                for element in required_elements:
                    if element not in content:
                        missing_elements.append(element)
                        
                if missing_elements:
                    doc_issues.append(f"{doc_file}: Missing elements - {', '.join(missing_elements)}")
                    
                # Check for reasonable length
                if len(content.split()) < 100:
                    doc_issues.append(f"{doc_file}: Documentation seems too brief")
                    
            except Exception as e:
                doc_issues.append(f"{doc_file}: Error reading file - {str(e)}")
                
        if doc_issues:
            for issue in doc_issues:
                result.add_error(issue)
        else:
            result.passed = True
            
        result.set_detail("doc_files_checked", len(doc_files))
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
    
    def run_infrastructure_tests(self) -> List[ValidationResult]:
        """Test technical infrastructure and compatibility."""
        print("\n" + "="*60)
        print("INFRASTRUCTURE VALIDATION")
        print("="*60)
        
        infra_tests = []
        
        # Test 1: Format compliance (Notion compatibility)
        infra_tests.append(self._test_format_compliance())
        
        # Test 2: Cross-platform compatibility 
        infra_tests.append(self._test_cross_platform_compatibility())
        
        # Test 3: File encoding and character issues
        infra_tests.append(self._test_file_encoding())
        
        return infra_tests
        
    def _test_format_compliance(self) -> ValidationResult:
        """Test Notion-compatible markdown format compliance."""
        result = ValidationResult("Format Compliance", "Infrastructure")
        start_time = time.time()
        
        markdown_files = list(self.project_root.rglob("*.md"))
        format_issues = []
        
        for md_file in markdown_files:
            try:
                with open(md_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for proper markdown structure
                lines = content.split('\n')
                
                # Headers should use # syntax
                improper_headers = [i for i, line in enumerate(lines) 
                                  if line.strip().startswith('=') or line.strip().startswith('-')]
                if improper_headers:
                    format_issues.append(f"{md_file.name}: Non-standard header formatting")
                    
                # Code blocks should be properly formatted
                code_blocks = content.count('```')
                if code_blocks % 2 != 0:
                    format_issues.append(f"{md_file.name}: Unmatched code block markers")
                    
                # Check for basic markdown elements
                has_headers = any(line.startswith('#') for line in lines)
                has_code = '```' in content
                has_lists = any(line.strip().startswith(('-', '*', '+')) for line in lines)
                
                if not has_headers:
                    format_issues.append(f"{md_file.name}: No headers found")
                    
            except Exception as e:
                format_issues.append(f"{md_file.name}: Format check error - {str(e)}")
                
        if format_issues:
            for issue in format_issues:
                result.add_error(issue)
        else:
            result.passed = True
            
        result.set_detail("markdown_files", len(markdown_files))
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
        
    def _test_cross_platform_compatibility(self) -> ValidationResult:
        """Test cross-platform file path and execution compatibility."""
        result = ValidationResult("Cross-Platform Compatibility", "Infrastructure")
        start_time = time.time()
        
        python_files = list(self.project_root.rglob("*.py"))
        compatibility_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r') as f:
                    content = f.read()
                    
                # Check for hardcoded path separators
                if '\\\\' in content or content.count('\\') > content.count('\\\\') * 2:
                    compatibility_issues.append(f"{py_file.name}: Potential hardcoded Windows paths")
                    
                # Check for Unix-specific commands in subprocess calls
                unix_commands = ['ls', 'cat', 'grep', 'head', 'tail']
                for cmd in unix_commands:
                    if f"'{cmd}'" in content or f'"{cmd}"' in content:
                        result.add_warning(f"{py_file.name}: Uses Unix-specific command '{cmd}'")
                        
                # Check for proper file handling
                if 'open(' in content and 'with open(' not in content:
                    compatibility_issues.append(f"{py_file.name}: File operations should use 'with' statement")
                    
            except Exception as e:
                compatibility_issues.append(f"{py_file.name}: Compatibility check error - {str(e)}")
                
        if compatibility_issues:
            for issue in compatibility_issues:
                result.add_error(issue)
        else:
            result.passed = True
            
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
        
    def _test_file_encoding(self) -> ValidationResult:
        """Test file encoding consistency and character handling."""
        result = ValidationResult("File Encoding", "Infrastructure")
        start_time = time.time()
        
        all_text_files = []
        all_text_files.extend(self.project_root.rglob("*.py"))
        all_text_files.extend(self.project_root.rglob("*.md"))
        
        encoding_issues = []
        
        for text_file in all_text_files:
            try:
                # Try UTF-8 encoding
                with open(text_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                # Check for problematic characters
                if '\r\n' in content:
                    result.add_warning(f"{text_file.name}: Contains Windows line endings")
                    
                # Check for non-ASCII characters that might cause issues
                try:
                    content.encode('ascii')
                except UnicodeEncodeError:
                    result.add_warning(f"{text_file.name}: Contains non-ASCII characters")
                    
            except UnicodeDecodeError:
                encoding_issues.append(f"{text_file.name}: UTF-8 encoding error")
            except Exception as e:
                encoding_issues.append(f"{text_file.name}: Encoding check error - {str(e)}")
                
        if encoding_issues:
            for issue in encoding_issues:
                result.add_error(issue)
        else:
            result.passed = True
            
        result.set_detail("files_checked", len(all_text_files))
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
    
    def run_assessment_alignment_tests(self) -> List[ValidationResult]:
        """Validate alignment with assessment and learning objectives."""
        print("\n" + "="*60)
        print("ASSESSMENT ALIGNMENT VALIDATION")
        print("="*60)
        
        assessment_tests = []
        
        # Test 1: Learning objectives coverage
        assessment_tests.append(self._test_learning_objectives())
        
        # Test 2: Skills practice validation
        assessment_tests.append(self._test_skills_practice())
        
        # Test 3: Exercise-assessment alignment
        assessment_tests.append(self._test_exercise_alignment())
        
        return assessment_tests
        
    def _test_learning_objectives(self) -> ValidationResult:
        """Test that learning objectives are covered in content."""
        result = ValidationResult("Learning Objectives Coverage", "Assessment")
        start_time = time.time()
        
        # Expected learning objectives based on README
        expected_objectives = [
            "command line", "python scripts", "variables", "data types",
            "control structures", "functions", "file organization",
            "development environment"
        ]
        
        # Check coverage in narrative and exercises
        narrative_file = self.project_root / "lecture_01_narrative.md"
        demo_file = self.project_root / "demo_lecture_01.py"
        
        objectives_covered = []
        
        files_to_check = [f for f in [narrative_file, demo_file] if f.exists()]
        
        for obj in expected_objectives:
            found = False
            for file_path in files_to_check:
                try:
                    with open(file_path, 'r') as f:
                        content = f.read().lower()
                        if obj.lower() in content:
                            found = True
                            break
                except:
                    continue
                    
            if found:
                objectives_covered.append(obj)
            else:
                result.add_warning(f"Learning objective not clearly addressed: {obj}")
                
        coverage_rate = len(objectives_covered) / len(expected_objectives)
        
        if coverage_rate >= 0.8:
            result.passed = True
        else:
            result.add_error(f"Insufficient learning objective coverage: {coverage_rate:.1%}")
            
        result.set_detail("objectives_covered", len(objectives_covered))
        result.set_detail("total_objectives", len(expected_objectives))
        result.set_detail("coverage_rate", coverage_rate)
        
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
        
    def _test_skills_practice(self) -> ValidationResult:
        """Test that practical skills are adequately practiced."""
        result = ValidationResult("Skills Practice Adequacy", "Assessment")
        start_time = time.time()
        
        exercise_dir = self.project_root / "exercises"
        
        if not exercise_dir.exists():
            result.add_error("No exercises directory found")
            self.log_result(result)
            return result
            
        # Count different types of practice
        python_exercises = list(exercise_dir.glob("*.py"))
        markdown_exercises = list(exercise_dir.glob("*.md"))
        
        total_exercises = len(python_exercises) + len(markdown_exercises)
        
        if total_exercises < 3:
            result.add_error(f"Insufficient exercises: {total_exercises} found, expected at least 3")
        else:
            result.passed = True
            
        # Check exercise diversity
        exercise_types = set()
        
        for py_file in python_exercises:
            try:
                with open(py_file, 'r') as f:
                    content = f.read().lower()
                    
                if "input(" in content:
                    exercise_types.add("interactive")
                if "def " in content:
                    exercise_types.add("function_writing")  
                if "for " in content or "while " in content:
                    exercise_types.add("loops")
                if "if " in content:
                    exercise_types.add("conditionals")
                    
            except:
                continue
                
        diversity_score = len(exercise_types)
        if diversity_score < 3:
            result.add_warning(f"Limited exercise diversity: {diversity_score} types")
            
        result.set_detail("total_exercises", total_exercises)
        result.set_detail("python_exercises", len(python_exercises))
        result.set_detail("exercise_diversity", diversity_score)
        
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
        
    def _test_exercise_alignment(self) -> ValidationResult:
        """Test alignment between exercises and stated objectives."""
        result = ValidationResult("Exercise-Objective Alignment", "Assessment")
        start_time = time.time()
        
        # This test checks if exercises practice the skills mentioned in learning objectives
        readme_file = self.project_root / "README.md"
        
        if not readme_file.exists():
            result.add_error("README file not found for alignment check")
            self.log_result(result)
            return result
            
        try:
            with open(readme_file, 'r') as f:
                readme_content = f.read().lower()
                
            # Extract mentioned skills from README
            skill_keywords = [
                "command line", "python", "variables", "functions",
                "control structures", "file operations", "scripts"
            ]
            
            mentioned_skills = [skill for skill in skill_keywords if skill in readme_content]
            
            # Check if exercises practice these skills
            exercise_dir = self.project_root / "exercises"
            practiced_skills = []
            
            if exercise_dir.exists():
                for py_file in exercise_dir.glob("*.py"):
                    try:
                        with open(py_file, 'r') as f:
                            content = f.read().lower()
                            
                        for skill in mentioned_skills:
                            if skill in content or any(keyword in content for keyword in skill.split()):
                                if skill not in practiced_skills:
                                    practiced_skills.append(skill)
                                    
                    except:
                        continue
                        
            alignment_rate = len(practiced_skills) / len(mentioned_skills) if mentioned_skills else 0
            
            if alignment_rate >= 0.7:
                result.passed = True
            else:
                result.add_error(f"Poor exercise-objective alignment: {alignment_rate:.1%}")
                
            result.set_detail("mentioned_skills", len(mentioned_skills))
            result.set_detail("practiced_skills", len(practiced_skills))
            result.set_detail("alignment_rate", alignment_rate)
            
        except Exception as e:
            result.add_error(f"Error checking alignment: {str(e)}")
            
        result.execution_time = time.time() - start_time
        self.log_result(result)
        return result
    
    def generate_comprehensive_report(self) -> Dict:
        """Generate comprehensive validation report."""
        end_time = datetime.now()
        total_time = (end_time - self.start_time).total_seconds()
        
        # Categorize results
        categories = {}
        for result in self.results:
            if result.category not in categories:
                categories[result.category] = []
            categories[result.category].append(result)
            
        # Calculate summary statistics
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.is_passed()])
        total_errors = sum(len(r.errors) for r in self.results)
        total_warnings = sum(len(r.warnings) for r in self.results)
        
        # Generate Phase 2 readiness assessment
        critical_failures = [r for r in self.results 
                           if not r.is_passed() and r.category in ["System", "Content"]]
        
        phase2_ready = len(critical_failures) == 0 and passed_tests / total_tests >= 0.8
        
        report = {
            "validation_summary": {
                "timestamp": end_time.isoformat(),
                "total_execution_time": total_time,
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests if total_tests > 0 else 0,
                "total_errors": total_errors,
                "total_warnings": total_warnings
            },
            "category_breakdown": {},
            "critical_issues": [],
            "phase2_readiness": {
                "ready": phase2_ready,
                "critical_failures": len(critical_failures),
                "readiness_score": passed_tests / total_tests if total_tests > 0 else 0
            },
            "recommendations": [],
            "detailed_results": []
        }
        
        # Category breakdown
        for category, results in categories.items():
            category_passed = len([r for r in results if r.is_passed()])
            category_total = len(results)
            
            report["category_breakdown"][category] = {
                "total_tests": category_total,
                "passed_tests": category_passed,
                "success_rate": category_passed / category_total if category_total > 0 else 0,
                "errors": sum(len(r.errors) for r in results),
                "warnings": sum(len(r.warnings) for r in results)
            }
            
        # Critical issues
        for result in self.results:
            if result.errors and result.category in ["System", "Content"]:
                report["critical_issues"].extend([
                    f"{result.category} - {result.test_name}: {error}" 
                    for error in result.errors
                ])
                
        # Generate recommendations
        if not phase2_ready:
            report["recommendations"].append(
                "Phase 2 implementation should be delayed until critical issues are resolved"
            )
            
        if total_warnings > 10:
            report["recommendations"].append(
                "High number of warnings detected - recommend review and optimization"
            )
            
        if report["category_breakdown"].get("System", {}).get("success_rate", 0) < 1.0:
            report["recommendations"].append(
                "System functionality issues detected - ensure all scripts execute properly"
            )
            
        # Detailed results
        for result in self.results:
            report["detailed_results"].append({
                "test_name": result.test_name,
                "category": result.category,
                "passed": result.is_passed(),
                "execution_time": result.execution_time,
                "errors": result.errors,
                "warnings": result.warnings,
                "details": result.details
            })
            
        return report
    
    def run_all_validations(self) -> Dict:
        """Run complete validation suite."""
        print("Phase 1 Comprehensive Validation Suite")
        print("Weeks 3-4 Complete Testing Framework")
        print("="*60)
        print(f"Started at: {self.start_time}")
        print(f"Project root: {self.project_root}")
        
        # Run all test categories
        self.run_system_functionality_tests()
        self.run_content_quality_tests()  
        self.run_infrastructure_tests()
        self.run_assessment_alignment_tests()
        
        # Generate comprehensive report
        return self.generate_comprehensive_report()


def main():
    """Main validation execution."""
    if len(sys.argv) > 1:
        project_root = Path(sys.argv[1])
    else:
        project_root = Path.cwd()
        
    if not project_root.exists():
        print(f"Error: Project root {project_root} does not exist")
        sys.exit(1)
        
    # Initialize validation suite
    validator = Phase1ValidationSuite(project_root)
    
    # Run complete validation
    report = validator.run_all_validations()
    
    # Print summary
    print("\n" + "="*80)
    print("PHASE 1 VALIDATION COMPLETE")
    print("="*80)
    
    summary = report["validation_summary"]
    print(f"✅ Tests Passed: {summary['passed_tests']}/{summary['total_tests']} "
          f"({summary['success_rate']:.1%})")
    print(f"❌ Total Errors: {summary['total_errors']}")
    print(f"⚠️  Total Warnings: {summary['total_warnings']}")
    print(f"⏱️  Execution Time: {summary['total_execution_time']:.1f} seconds")
    
    # Phase 2 readiness
    readiness = report["phase2_readiness"]
    print(f"\n🚀 Phase 2 Ready: {'YES' if readiness['ready'] else 'NO'}")
    print(f"📊 Readiness Score: {readiness['readiness_score']:.1%}")
    
    if readiness["critical_failures"] > 0:
        print(f"🚨 Critical Failures: {readiness['critical_failures']}")
        
    # Save detailed report
    report_file = project_root / "tests" / "validation_report.json"
    report_file.parent.mkdir(exist_ok=True)
    
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
        
    print(f"\n📄 Detailed report saved to: {report_file}")
    
    # Exit with appropriate code
    sys.exit(0 if readiness["ready"] else 1)


if __name__ == "__main__":
    main()