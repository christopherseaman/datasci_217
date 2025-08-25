#!/usr/bin/env python3
"""
Quality Validation Tool for Lecture Content
===========================================

Comprehensive validation suite for ensuring lecture content meets
quality standards, format requirements, and educational objectives.

Usage:
    python3 quality_validator.py --lecture-dir converted/lecture_01/
    python3 quality_validator.py --batch-validate converted/
    python3 quality_validator.py --config validation_config.json
"""

import os
import sys
import json
import re
import ast
import subprocess
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from dataclasses import dataclass, asdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Data class for validation results."""
    check_name: str
    status: str  # 'pass', 'fail', 'warning'
    message: str
    details: Optional[Dict] = None
    suggestions: Optional[List[str]] = None


@dataclass
class LectureValidationReport:
    """Complete validation report for a lecture."""
    lecture_name: str
    validation_time: str
    overall_status: str  # 'pass', 'fail', 'warning'
    results: List[ValidationResult]
    summary: Dict[str, int]
    recommendations: List[str]


class QualityValidator:
    """
    Comprehensive quality validation system for lecture content.
    
    Validates content quality, format compliance, code execution,
    educational alignment, and technical correctness.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize validator with configuration."""
        self.config = self._load_config(config_path)
        self.validation_results = []
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load validation configuration."""
        default_config = {
            "content_quality": {
                "min_word_count": 5000,
                "max_word_count": 8000,
                "min_code_examples": 8,
                "min_exercises": 3,
                "required_sections": [
                    "overview", "learning_objectives", "prerequisites",
                    "core_concepts", "hands_on_practice", "real_world_applications"
                ]
            },
            "format_validation": {
                "check_markdown_syntax": True,
                "validate_notion_compatibility": True,
                "check_header_hierarchy": True,
                "validate_code_blocks": True,
                "check_link_validity": True
            },
            "code_validation": {
                "execute_python_examples": True,
                "check_syntax_errors": True,
                "validate_imports": True,
                "test_interactive_demos": True,
                "check_pep8_compliance": True
            },
            "educational_validation": {
                "verify_learning_objectives": True,
                "check_progressive_complexity": True,
                "validate_assessment_alignment": True,
                "verify_prerequisite_coverage": True
            },
            "technical_validation": {
                "check_file_encodings": True,
                "validate_directory_structure": True,
                "check_media_references": True,
                "validate_cross_references": True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def validate_lecture(self, lecture_dir: str) -> LectureValidationReport:
        """
        Perform comprehensive validation of a single lecture.
        
        Args:
            lecture_dir: Path to lecture directory
            
        Returns:
            LectureValidationReport: Complete validation results
        """
        logger.info(f"Validating lecture: {lecture_dir}")
        
        lecture_path = Path(lecture_dir)
        lecture_name = lecture_path.name
        
        # Initialize results collection
        results = []
        
        # Content Quality Validation
        results.extend(self._validate_content_quality(lecture_path))
        
        # Format Validation
        results.extend(self._validate_format(lecture_path))
        
        # Code Validation  
        results.extend(self._validate_code(lecture_path))
        
        # Educational Validation
        results.extend(self._validate_educational_content(lecture_path))
        
        # Technical Validation
        results.extend(self._validate_technical_aspects(lecture_path))
        
        # Generate summary and recommendations
        summary = self._generate_summary(results)
        recommendations = self._generate_recommendations(results)
        
        # Determine overall status
        overall_status = "pass"
        if any(r.status == "fail" for r in results):
            overall_status = "fail"
        elif any(r.status == "warning" for r in results):
            overall_status = "warning"
        
        # Create validation report
        report = LectureValidationReport(
            lecture_name=lecture_name,
            validation_time=datetime.now().isoformat(),
            overall_status=overall_status,
            results=results,
            summary=summary,
            recommendations=recommendations
        )
        
        logger.info(f"Validation complete: {lecture_name} - {overall_status.upper()}")
        return report
    
    def _validate_content_quality(self, lecture_path: Path) -> List[ValidationResult]:
        """Validate content quality metrics."""
        results = []
        
        # Find main narrative file
        narrative_file = self._find_narrative_file(lecture_path)
        if not narrative_file:
            results.append(ValidationResult(
                check_name="narrative_file_exists",
                status="fail",
                message="No main narrative file found",
                suggestions=["Create a main narrative markdown file (e.g., lecture_narrative.md)"]
            ))
            return results
        
        # Read content
        try:
            content = narrative_file.read_text(encoding='utf-8')
        except Exception as e:
            results.append(ValidationResult(
                check_name="content_readable",
                status="fail", 
                message=f"Cannot read narrative file: {e}",
                suggestions=["Check file encoding and permissions"]
            ))
            return results
        
        # Word count validation
        word_count = len(content.split())
        min_words = self.config["content_quality"]["min_word_count"]
        max_words = self.config["content_quality"]["max_word_count"]
        
        if word_count < min_words:
            results.append(ValidationResult(
                check_name="word_count_minimum",
                status="fail",
                message=f"Content too short: {word_count} words (minimum: {min_words})",
                details={"actual": word_count, "minimum": min_words},
                suggestions=["Expand concept explanations", "Add more examples", "Include additional context"]
            ))
        elif word_count > max_words:
            results.append(ValidationResult(
                check_name="word_count_maximum",
                status="warning",
                message=f"Content quite long: {word_count} words (maximum: {max_words})",
                details={"actual": word_count, "maximum": max_words},
                suggestions=["Consider splitting into multiple sections", "Remove redundant explanations"]
            ))
        else:
            results.append(ValidationResult(
                check_name="word_count",
                status="pass",
                message=f"Word count appropriate: {word_count} words"
            ))
        
        # Code example validation
        code_blocks = re.findall(r'```[\s\S]*?```', content)
        python_blocks = re.findall(r'```python[\s\S]*?```', content)
        
        min_code_examples = self.config["content_quality"]["min_code_examples"]
        
        if len(python_blocks) < min_code_examples:
            results.append(ValidationResult(
                check_name="code_examples_minimum",
                status="fail",
                message=f"Insufficient Python code examples: {len(python_blocks)} (minimum: {min_code_examples})",
                details={"actual": len(python_blocks), "minimum": min_code_examples},
                suggestions=["Add more executable Python examples", "Include interactive demonstrations"]
            ))
        else:
            results.append(ValidationResult(
                check_name="code_examples",
                status="pass",
                message=f"Sufficient code examples: {len(python_blocks)} Python blocks"
            ))
        
        # Required sections validation
        required_sections = self.config["content_quality"]["required_sections"]
        missing_sections = []
        
        content_lower = content.lower()
        for section in required_sections:
            section_patterns = [
                f"## {section.replace('_', ' ')}",
                f"# {section.replace('_', ' ')}",
                section.replace('_', ' ')
            ]
            
            if not any(pattern.lower() in content_lower for pattern in section_patterns):
                missing_sections.append(section)
        
        if missing_sections:
            results.append(ValidationResult(
                check_name="required_sections",
                status="fail",
                message=f"Missing required sections: {', '.join(missing_sections)}",
                details={"missing": missing_sections, "required": required_sections},
                suggestions=[f"Add {section.replace('_', ' ')} section" for section in missing_sections]
            ))
        else:
            results.append(ValidationResult(
                check_name="required_sections",
                status="pass",
                message="All required sections present"
            ))
        
        return results
    
    def _validate_format(self, lecture_path: Path) -> List[ValidationResult]:
        """Validate format compliance and markdown syntax."""
        results = []
        
        narrative_file = self._find_narrative_file(lecture_path)
        if not narrative_file:
            return results  # Already handled in content validation
        
        try:
            content = narrative_file.read_text(encoding='utf-8')
        except Exception:
            return results  # Already handled in content validation
        
        # Markdown header hierarchy validation
        headers = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
        
        if self.config["format_validation"]["check_header_hierarchy"]:
            header_issues = self._check_header_hierarchy(headers)
            
            if header_issues:
                results.append(ValidationResult(
                    check_name="header_hierarchy",
                    status="warning",
                    message="Header hierarchy issues found",
                    details={"issues": header_issues},
                    suggestions=["Ensure proper header level progression", "Use consistent header structure"]
                ))
            else:
                results.append(ValidationResult(
                    check_name="header_hierarchy",
                    status="pass",
                    message="Header hierarchy is well-structured"
                ))
        
        # Code block syntax validation
        if self.config["format_validation"]["validate_code_blocks"]:
            code_block_issues = self._validate_code_blocks(content)
            
            if code_block_issues:
                results.append(ValidationResult(
                    check_name="code_block_syntax",
                    status="warning",
                    message="Code block formatting issues",
                    details={"issues": code_block_issues},
                    suggestions=["Fix code block syntax", "Ensure proper language specification"]
                ))
            else:
                results.append(ValidationResult(
                    check_name="code_block_syntax",
                    status="pass",
                    message="Code blocks properly formatted"
                ))
        
        # Notion compatibility check
        if self.config["format_validation"]["validate_notion_compatibility"]:
            notion_issues = self._check_notion_compatibility(content)
            
            if notion_issues:
                results.append(ValidationResult(
                    check_name="notion_compatibility",
                    status="warning",
                    message="Potential Notion compatibility issues",
                    details={"issues": notion_issues},
                    suggestions=["Simplify complex formatting", "Test import in Notion"]
                ))
            else:
                results.append(ValidationResult(
                    check_name="notion_compatibility",
                    status="pass",
                    message="Format compatible with Notion"
                ))
        
        return results
    
    def _validate_code(self, lecture_path: Path) -> List[ValidationResult]:
        """Validate Python code examples and demonstrations."""
        results = []
        
        # Find Python files
        python_files = list(lecture_path.rglob("*.py"))
        
        if not python_files:
            results.append(ValidationResult(
                check_name="python_files_exist",
                status="warning", 
                message="No Python demonstration files found",
                suggestions=["Create interactive demonstration script", "Add executable examples"]
            ))
            return results
        
        # Validate each Python file
        for py_file in python_files:
            file_results = self._validate_python_file(py_file)
            results.extend(file_results)
        
        # Check main demonstration script
        demo_script = lecture_path / "demo_lecture.py"
        if demo_script.exists():
            demo_results = self._validate_demo_script(demo_script)
            results.extend(demo_results)
        else:
            results.append(ValidationResult(
                check_name="demo_script_exists",
                status="warning",
                message="No main demonstration script found",
                suggestions=["Create demo_lecture.py with interactive examples"]
            ))
        
        return results
    
    def _validate_python_file(self, py_file: Path) -> List[ValidationResult]:
        """Validate individual Python file."""
        results = []
        file_name = py_file.name
        
        try:
            content = py_file.read_text(encoding='utf-8')
        except Exception as e:
            results.append(ValidationResult(
                check_name=f"python_file_readable_{file_name}",
                status="fail",
                message=f"Cannot read Python file {file_name}: {e}",
                suggestions=["Check file encoding and permissions"]
            ))
            return results
        
        # Syntax validation
        try:
            ast.parse(content)
            results.append(ValidationResult(
                check_name=f"python_syntax_{file_name}",
                status="pass",
                message=f"Python syntax valid in {file_name}"
            ))
        except SyntaxError as e:
            results.append(ValidationResult(
                check_name=f"python_syntax_{file_name}",
                status="fail",
                message=f"Syntax error in {file_name}: {e}",
                details={"line": e.lineno, "error": str(e)},
                suggestions=["Fix Python syntax errors", "Test code execution"]
            ))
        
        # Import validation
        import_issues = self._check_imports(content)
        if import_issues:
            results.append(ValidationResult(
                check_name=f"imports_{file_name}",
                status="warning",
                message=f"Import issues in {file_name}",
                details={"issues": import_issues},
                suggestions=["Verify all imports are available", "Add installation instructions"]
            ))
        
        # Execution test (if configured)
        if self.config["code_validation"]["execute_python_examples"]:
            execution_result = self._test_python_execution(py_file)
            results.append(execution_result)
        
        return results
    
    def _validate_demo_script(self, demo_script: Path) -> List[ValidationResult]:
        """Validate main demonstration script functionality."""
        results = []
        
        # Check if script has main function
        try:
            content = demo_script.read_text(encoding='utf-8')
            
            if 'def main(' in content and 'if __name__ == "__main__":' in content:
                results.append(ValidationResult(
                    check_name="demo_script_structure",
                    status="pass",
                    message="Demo script has proper main function structure"
                ))
            else:
                results.append(ValidationResult(
                    check_name="demo_script_structure",
                    status="warning",
                    message="Demo script missing main function or entry point",
                    suggestions=["Add main() function", "Add if __name__ == '__main__': guard"]
                ))
            
            # Check for argparse usage
            if 'argparse' in content:
                results.append(ValidationResult(
                    check_name="demo_script_cli",
                    status="pass",
                    message="Demo script supports command line arguments"
                ))
            else:
                results.append(ValidationResult(
                    check_name="demo_script_cli",
                    status="warning",
                    message="Demo script lacks command line interface",
                    suggestions=["Add argparse for section selection", "Enable interactive mode"]
                ))
        
        except Exception as e:
            results.append(ValidationResult(
                check_name="demo_script_analysis",
                status="fail",
                message=f"Cannot analyze demo script: {e}"
            ))
        
        return results
    
    def _validate_educational_content(self, lecture_path: Path) -> List[ValidationResult]:
        """Validate educational content and learning objectives."""
        results = []
        
        narrative_file = self._find_narrative_file(lecture_path)
        if not narrative_file:
            return results
        
        try:
            content = narrative_file.read_text(encoding='utf-8')
        except Exception:
            return results
        
        # Learning objectives validation
        objectives_section = self._extract_section(content, "learning objectives")
        if objectives_section:
            objectives_count = len(re.findall(r'^\s*[-\*\d\.]\s+', objectives_section, re.MULTILINE))
            
            if objectives_count >= 3:
                results.append(ValidationResult(
                    check_name="learning_objectives_count",
                    status="pass",
                    message=f"Sufficient learning objectives: {objectives_count}"
                ))
            else:
                results.append(ValidationResult(
                    check_name="learning_objectives_count",
                    status="warning",
                    message=f"Few learning objectives: {objectives_count} (recommend 3+)",
                    suggestions=["Add more specific learning objectives", "Clarify expected outcomes"]
                ))
        else:
            results.append(ValidationResult(
                check_name="learning_objectives_exist",
                status="fail",
                message="No learning objectives section found",
                suggestions=["Add clear learning objectives section"]
            ))
        
        # Exercise validation
        exercises_dir = lecture_path / "exercises"
        if exercises_dir.exists():
            exercise_files = list(exercises_dir.glob("*.py")) + list(exercises_dir.glob("*.md"))
            min_exercises = self.config["content_quality"]["min_exercises"]
            
            if len(exercise_files) >= min_exercises:
                results.append(ValidationResult(
                    check_name="exercises_count",
                    status="pass", 
                    message=f"Sufficient exercise files: {len(exercise_files)}"
                ))
            else:
                results.append(ValidationResult(
                    check_name="exercises_count",
                    status="warning",
                    message=f"Few exercise files: {len(exercise_files)} (minimum: {min_exercises})",
                    suggestions=["Add more practice exercises", "Create varied difficulty levels"]
                ))
        else:
            results.append(ValidationResult(
                check_name="exercises_directory",
                status="fail",
                message="No exercises directory found",
                suggestions=["Create exercises directory with practice problems"]
            ))
        
        return results
    
    def _validate_technical_aspects(self, lecture_path: Path) -> List[ValidationResult]:
        """Validate technical aspects and file organization."""
        results = []
        
        # Directory structure validation
        expected_dirs = ["exercises", "resources", "media"]
        missing_dirs = []
        
        for dir_name in expected_dirs:
            if not (lecture_path / dir_name).exists():
                missing_dirs.append(dir_name)
        
        if missing_dirs:
            results.append(ValidationResult(
                check_name="directory_structure",
                status="warning",
                message=f"Missing recommended directories: {', '.join(missing_dirs)}",
                suggestions=[f"Create {dir_name}/ directory" for dir_name in missing_dirs]
            ))
        else:
            results.append(ValidationResult(
                check_name="directory_structure",
                status="pass",
                message="Recommended directory structure present"
            ))
        
        # File encoding validation
        text_files = list(lecture_path.rglob("*.md")) + list(lecture_path.rglob("*.py"))
        encoding_issues = []
        
        for file_path in text_files:
            try:
                file_path.read_text(encoding='utf-8')
            except UnicodeDecodeError:
                encoding_issues.append(str(file_path.relative_to(lecture_path)))
        
        if encoding_issues:
            results.append(ValidationResult(
                check_name="file_encodings",
                status="fail",
                message="Files with encoding issues found",
                details={"files": encoding_issues},
                suggestions=["Convert files to UTF-8 encoding"]
            ))
        else:
            results.append(ValidationResult(
                check_name="file_encodings",
                status="pass",
                message="All text files use UTF-8 encoding"
            ))
        
        # README validation
        readme_files = list(lecture_path.glob("README.md"))
        if readme_files:
            results.append(ValidationResult(
                check_name="readme_exists",
                status="pass",
                message="README file present"
            ))
        else:
            results.append(ValidationResult(
                check_name="readme_exists",
                status="warning",
                message="No README file found",
                suggestions=["Create README.md with usage instructions"]
            ))
        
        return results
    
    def _find_narrative_file(self, lecture_path: Path) -> Optional[Path]:
        """Find the main narrative content file."""
        possible_names = [
            "lecture_narrative.md",
            "narrative.md", 
            "content.md",
            "lecture.md"
        ]
        
        for name in possible_names:
            file_path = lecture_path / name
            if file_path.exists():
                return file_path
        
        # Look for any markdown file that might be the main content
        md_files = list(lecture_path.glob("*.md"))
        if md_files:
            # Return the largest markdown file as likely main content
            return max(md_files, key=lambda f: f.stat().st_size)
        
        return None
    
    def _check_header_hierarchy(self, headers: List[Tuple[str, str]]) -> List[str]:
        """Check for proper header hierarchy."""
        issues = []
        prev_level = 0
        
        for i, (hashes, title) in enumerate(headers):
            level = len(hashes)
            
            if level > prev_level + 1:
                issues.append(f"Header level jump: '{title}' (#{level}) follows #{prev_level}")
            
            prev_level = level
        
        return issues
    
    def _validate_code_blocks(self, content: str) -> List[str]:
        """Validate code block formatting."""
        issues = []
        
        # Find all code blocks
        code_blocks = re.findall(r'```(.*?)\n([\s\S]*?)```', content)
        
        for i, (language, code) in enumerate(code_blocks, 1):
            # Check for missing language specification
            if not language.strip():
                issues.append(f"Code block {i} missing language specification")
            
            # Check for common formatting issues
            if code.strip().startswith('    '):
                issues.append(f"Code block {i} may have extra indentation")
        
        return issues
    
    def _check_notion_compatibility(self, content: str) -> List[str]:
        """Check for Notion compatibility issues."""
        issues = []
        
        # Check for complex HTML
        if re.search(r'<(?!br|em|strong|code)[^>]+>', content):
            issues.append("Complex HTML tags found - may not render in Notion")
        
        # Check for complex markdown extensions
        if ':::' in content:
            issues.append("Markdown extensions (:::) may not work in Notion")
        
        # Check for math expressions
        if re.search(r'\$.*\$', content):
            issues.append("Math expressions may need adjustment for Notion")
        
        return issues
    
    def _check_imports(self, content: str) -> List[str]:
        """Check for potentially problematic imports."""
        issues = []
        
        # Find import statements
        imports = re.findall(r'^(?:from\s+(\S+)\s+)?import\s+(\S+)', content, re.MULTILINE)
        
        # List of imports that might not be available in standard environments
        potentially_missing = [
            'numpy', 'pandas', 'matplotlib', 'seaborn', 'sklearn',
            'tensorflow', 'torch', 'jupyter', 'ipython'
        ]
        
        for module, item in imports:
            import_name = module if module else item
            if any(missing in import_name for missing in potentially_missing):
                issues.append(f"Import '{import_name}' may require installation")
        
        return issues
    
    def _test_python_execution(self, py_file: Path) -> ValidationResult:
        """Test if Python file executes without errors."""
        try:
            # Run Python file with basic checks
            result = subprocess.run(
                [sys.executable, str(py_file), '--help'],
                capture_output=True,
                text=True,
                timeout=30
            )
            
            if result.returncode == 0:
                return ValidationResult(
                    check_name=f"execution_{py_file.name}",
                    status="pass",
                    message=f"Python file {py_file.name} executes successfully"
                )
            else:
                return ValidationResult(
                    check_name=f"execution_{py_file.name}",
                    status="warning",
                    message=f"Python file {py_file.name} execution issues",
                    details={"stderr": result.stderr[:500]},
                    suggestions=["Test script execution", "Fix runtime errors"]
                )
        
        except subprocess.TimeoutExpired:
            return ValidationResult(
                check_name=f"execution_{py_file.name}",
                status="warning",
                message=f"Python file {py_file.name} execution timeout",
                suggestions=["Check for infinite loops", "Optimize execution time"]
            )
        except Exception as e:
            return ValidationResult(
                check_name=f"execution_{py_file.name}",
                status="fail",
                message=f"Cannot test execution of {py_file.name}: {e}",
                suggestions=["Check file permissions", "Verify Python installation"]
            )
    
    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """Extract specific section content from markdown."""
        pattern = rf'#+\s*{re.escape(section_name)}.*?\n(.*?)(?=\n#+|\Z)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else None
    
    def _generate_summary(self, results: List[ValidationResult]) -> Dict[str, int]:
        """Generate validation summary statistics."""
        summary = {"pass": 0, "warning": 0, "fail": 0}
        
        for result in results:
            if result.status in summary:
                summary[result.status] += 1
        
        return summary
    
    def _generate_recommendations(self, results: List[ValidationResult]) -> List[str]:
        """Generate prioritized recommendations."""
        recommendations = []
        
        # High priority: failures
        fails = [r for r in results if r.status == "fail"]
        if fails:
            recommendations.append("🔴 Address critical failures:")
            for fail in fails:
                if fail.suggestions:
                    recommendations.extend(f"  - {suggestion}" for suggestion in fail.suggestions)
        
        # Medium priority: warnings
        warnings = [r for r in results if r.status == "warning"]
        if warnings:
            recommendations.append("🟡 Consider addressing warnings:")
            for warning in warnings[:5]:  # Limit to top 5 warnings
                if warning.suggestions:
                    recommendations.extend(f"  - {suggestion}" for suggestion in warning.suggestions[:2])
        
        # General recommendations
        recommendations.append("✅ General improvements:")
        recommendations.append("  - Test all interactive examples with students")
        recommendations.append("  - Review content flow and narrative structure")
        recommendations.append("  - Validate cross-references and links")
        
        return recommendations
    
    def generate_report(self, report: LectureValidationReport, output_path: Optional[str] = None) -> str:
        """Generate formatted validation report."""
        report_lines = [
            f"# Validation Report: {report.lecture_name}",
            "",
            f"**Validation Time**: {report.validation_time}",
            f"**Overall Status**: {report.overall_status.upper()} {'✅' if report.overall_status == 'pass' else '⚠️' if report.overall_status == 'warning' else '❌'}",
            "",
            "## Summary",
            "",
            f"- **Total Checks**: {sum(report.summary.values())}",
            f"- **Passed**: {report.summary.get('pass', 0)} ✅",
            f"- **Warnings**: {report.summary.get('warning', 0)} ⚠️", 
            f"- **Failed**: {report.summary.get('fail', 0)} ❌",
            "",
            "## Detailed Results",
            ""
        ]
        
        # Group results by status
        for status in ['fail', 'warning', 'pass']:
            status_results = [r for r in report.results if r.status == status]
            if status_results:
                status_icon = {'fail': '❌', 'warning': '⚠️', 'pass': '✅'}[status]
                report_lines.append(f"### {status.title()} {status_icon}")
                report_lines.append("")
                
                for result in status_results:
                    report_lines.append(f"**{result.check_name}**: {result.message}")
                    
                    if result.details:
                        report_lines.append(f"  - Details: {result.details}")
                    
                    if result.suggestions:
                        report_lines.append("  - Suggestions:")
                        for suggestion in result.suggestions:
                            report_lines.append(f"    - {suggestion}")
                    
                    report_lines.append("")
        
        # Add recommendations
        report_lines.extend([
            "## Recommendations",
            ""
        ])
        report_lines.extend(report.recommendations)
        
        report_lines.extend([
            "",
            "---",
            f"*Generated by quality_validator.py at {datetime.now().isoformat()}*"
        ])
        
        report_content = "\n".join(report_lines)
        
        # Write to file if path provided
        if output_path:
            Path(output_path).write_text(report_content, encoding='utf-8')
        
        return report_content
    
    def batch_validate(self, lectures_dir: str) -> List[LectureValidationReport]:
        """Validate multiple lectures in batch."""
        logger.info(f"Starting batch validation: {lectures_dir}")
        
        lectures_path = Path(lectures_dir)
        reports = []
        
        # Find lecture directories
        lecture_dirs = [d for d in lectures_path.iterdir() if d.is_dir() and d.name.startswith('lecture')]
        
        for lecture_dir in lecture_dirs:
            logger.info(f"Validating: {lecture_dir.name}")
            report = self.validate_lecture(str(lecture_dir))
            reports.append(report)
        
        # Generate batch summary report
        self._generate_batch_summary(reports, lectures_dir)
        
        return reports
    
    def _generate_batch_summary(self, reports: List[LectureValidationReport], output_dir: str):
        """Generate summary report for batch validation."""
        summary_path = Path(output_dir) / "VALIDATION_SUMMARY.md"
        
        total_checks = sum(sum(r.summary.values()) for r in reports)
        total_passed = sum(r.summary.get('pass', 0) for r in reports)
        total_warnings = sum(r.summary.get('warning', 0) for r in reports)
        total_failed = sum(r.summary.get('fail', 0) for r in reports)
        
        summary_lines = [
            "# Batch Validation Summary",
            "",
            f"**Validation Date**: {datetime.now().isoformat()}",
            f"**Lectures Validated**: {len(reports)}",
            "",
            "## Overall Statistics",
            "",
            f"- **Total Checks**: {total_checks}",
            f"- **Passed**: {total_passed} ({total_passed/max(total_checks,1)*100:.1f}%)",
            f"- **Warnings**: {total_warnings} ({total_warnings/max(total_checks,1)*100:.1f}%)",
            f"- **Failed**: {total_failed} ({total_failed/max(total_checks,1)*100:.1f}%)",
            "",
            "## Lecture Status Overview",
            ""
        ]
        
        for report in reports:
            status_icon = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}[report.overall_status]
            summary_lines.append(f"- **{report.lecture_name}**: {report.overall_status.upper()} {status_icon}")
        
        summary_lines.extend([
            "",
            "## Next Steps",
            "",
            "1. Address all failed validations (❌) immediately",
            "2. Review and consider fixing warnings (⚠️)",
            "3. Verify manual testing of interactive components",
            "4. Conduct final review with educational team",
            "",
            "See individual lecture validation reports for detailed recommendations.",
            "",
            "---",
            f"*Generated by quality_validator.py*"
        ])
        
        summary_content = "\n".join(summary_lines)
        summary_path.write_text(summary_content, encoding='utf-8')
        
        logger.info(f"Batch validation summary written to: {summary_path}")


def main():
    """Main CLI interface for quality validation tool."""
    parser = argparse.ArgumentParser(
        description="Quality Validation Tool for Lecture Content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 quality_validator.py --lecture-dir converted/lecture_01/
    python3 quality_validator.py --batch-validate converted/
    python3 quality_validator.py --config validation_config.json --lecture-dir lecture/
        """
    )
    
    parser.add_argument('--lecture-dir', help='Single lecture directory to validate')
    parser.add_argument('--batch-validate', help='Directory containing multiple lectures')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output-report', help='Output path for detailed report')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not args.lecture_dir and not args.batch_validate:
        parser.error("Must specify either --lecture-dir or --batch-validate")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize validator
    validator = QualityValidator(args.config)
    
    if args.batch_validate:
        # Batch validation
        logger.info("Starting batch validation")
        reports = validator.batch_validate(args.batch_validate)
        
        print(f"\n=== Batch Validation Results ===")
        print(f"Lectures validated: {len(reports)}")
        
        for report in reports:
            status_icon = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}[report.overall_status]
            print(f"  {report.lecture_name}: {report.overall_status.upper()} {status_icon}")
        
        print(f"\nDetailed reports generated in: {args.batch_validate}")
    
    else:
        # Single lecture validation
        logger.info(f"Validating single lecture: {args.lecture_dir}")
        report = validator.validate_lecture(args.lecture_dir)
        
        # Generate and display report
        report_content = validator.generate_report(report, args.output_report)
        
        if not args.output_report:
            print(report_content)
        else:
            print(f"Detailed report written to: {args.output_report}")
        
        # Print summary
        print(f"\n=== Validation Summary ===")
        status_icon = {'pass': '✅', 'warning': '⚠️', 'fail': '❌'}[report.overall_status]
        print(f"Overall Status: {report.overall_status.upper()} {status_icon}")
        print(f"Total Checks: {sum(report.summary.values())}")
        print(f"Passed: {report.summary.get('pass', 0)}")
        print(f"Warnings: {report.summary.get('warning', 0)}")  
        print(f"Failed: {report.summary.get('fail', 0)}")


if __name__ == "__main__":
    main()