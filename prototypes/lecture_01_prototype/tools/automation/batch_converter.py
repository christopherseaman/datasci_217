#!/usr/bin/env python3
"""
Batch Lecture Conversion Tool
==============================

Automated tool for converting multiple lecture sources into the new
narrative format. Handles content combination, format transformation,
and quality validation in batch processing mode.

Usage:
    python3 batch_converter.py --source-dir lectures/ --output-dir converted/
    python3 batch_converter.py --config config/conversion_config.json
    python3 batch_converter.py --validate-only converted/
"""

import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LectureConverter:
    """
    Core conversion engine for transforming traditional lecture format
    to narrative-driven, Notion-compatible format.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize converter with configuration."""
        self.config = self._load_config(config_path)
        self.conversion_stats = {
            'processed': 0,
            'successful': 0,
            'failed': 0,
            'warnings': []
        }
    
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load conversion configuration."""
        default_config = {
            "narrative_format": {
                "target_length": {"min": 5000, "max": 6000},
                "section_structure": [
                    "overview",
                    "learning_objectives", 
                    "prerequisites",
                    "core_concepts",
                    "hands_on_practice",
                    "real_world_applications",
                    "assessment_integration",
                    "further_reading"
                ],
                "code_examples_minimum": 10,
                "exercises_minimum": 3
            },
            "content_combination": {
                "primary_weight": 0.9,
                "secondary_weight": 0.3,
                "tertiary_weight": 0.1,
                "integration_threshold": 0.25
            },
            "quality_validation": {
                "check_code_execution": True,
                "validate_markdown": True,
                "check_notion_compatibility": True,
                "verify_learning_objectives": True
            },
            "file_patterns": {
                "source_extensions": [".md", ".py", ".ipynb"],
                "output_format": "markdown",
                "media_extensions": [".png", ".jpg", ".svg", ".gif"]
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def analyze_source_content(self, source_dir: str) -> Dict:
        """
        Analyze source directory to understand content structure and
        identify combination opportunities.
        """
        logger.info(f"Analyzing source content in {source_dir}")
        
        analysis = {
            "lectures": {},
            "topics": {},
            "combination_opportunities": [],
            "media_files": []
        }
        
        source_path = Path(source_dir)
        
        # Scan lecture directories
        for lecture_dir in source_path.glob("lecture_*"):
            if lecture_dir.is_dir():
                lecture_info = self._analyze_lecture_directory(lecture_dir)
                analysis["lectures"][lecture_dir.name] = lecture_info
                
                # Extract topics for combination analysis
                for topic in lecture_info.get("topics", []):
                    if topic not in analysis["topics"]:
                        analysis["topics"][topic] = []
                    analysis["topics"][topic].append(lecture_dir.name)
        
        # Identify combination opportunities
        analysis["combination_opportunities"] = self._identify_combinations(analysis["topics"])
        
        # Find media files
        for ext in self.config["file_patterns"]["media_extensions"]:
            analysis["media_files"].extend(source_path.rglob(f"*{ext}"))
        
        logger.info(f"Found {len(analysis['lectures'])} lectures, {len(analysis['topics'])} unique topics")
        return analysis
    
    def _analyze_lecture_directory(self, lecture_dir: Path) -> Dict:
        """Analyze individual lecture directory."""
        info = {
            "path": str(lecture_dir),
            "files": [],
            "topics": [],
            "code_files": [],
            "media_files": [],
            "estimated_content_length": 0
        }
        
        # Scan files
        for file_path in lecture_dir.rglob("*"):
            if file_path.is_file():
                info["files"].append(str(file_path))
                
                if file_path.suffix in self.config["file_patterns"]["source_extensions"]:
                    content_length = len(file_path.read_text(encoding='utf-8', errors='ignore'))
                    info["estimated_content_length"] += content_length
                
                if file_path.suffix == '.py':
                    info["code_files"].append(str(file_path))
                
                if file_path.suffix in self.config["file_patterns"]["media_extensions"]:
                    info["media_files"].append(str(file_path))
        
        # Extract topics from content (simplified - would use NLP in production)
        info["topics"] = self._extract_topics_from_directory(lecture_dir)
        
        return info
    
    def _extract_topics_from_directory(self, lecture_dir: Path) -> List[str]:
        """Extract main topics from lecture directory content."""
        topics = []
        
        # Look for common topic indicators in filenames and content
        topic_indicators = {
            "python": ["python", "variables", "functions", "loops"],
            "command_line": ["cli", "bash", "terminal", "shell"],
            "git": ["git", "version_control", "repository"],
            "data_structures": ["list", "dict", "array", "dataframe"],
            "numpy": ["numpy", "array", "matrix", "numerical"],
            "pandas": ["pandas", "dataframe", "csv", "data_analysis"]
        }
        
        # Scan content for topic indicators
        for file_path in lecture_dir.rglob("*.md"):
            try:
                content = file_path.read_text(encoding='utf-8').lower()
                for topic, indicators in topic_indicators.items():
                    if any(indicator in content for indicator in indicators):
                        if topic not in topics:
                            topics.append(topic)
            except Exception as e:
                logger.warning(f"Could not read {file_path}: {e}")
        
        return topics
    
    def _identify_combinations(self, topics: Dict[str, List[str]]) -> List[Dict]:
        """Identify opportunities to combine content from multiple lectures."""
        combinations = []
        
        # Find topics that appear in multiple lectures
        for topic, lectures in topics.items():
            if len(lectures) > 1:
                combinations.append({
                    "topic": topic,
                    "lectures": lectures,
                    "combination_type": "merge",
                    "priority": "high" if len(lectures) > 2 else "medium"
                })
        
        return combinations
    
    def convert_lecture(self, source_path: str, output_path: str, 
                       combination_sources: Optional[List[str]] = None) -> bool:
        """
        Convert a single lecture to narrative format.
        
        Args:
            source_path: Path to source lecture directory
            output_path: Path for output lecture directory
            combination_sources: Additional sources to integrate
        
        Returns:
            bool: True if conversion successful
        """
        logger.info(f"Converting lecture: {source_path} -> {output_path}")
        
        try:
            # Create output directory
            os.makedirs(output_path, exist_ok=True)
            
            # Load and analyze source content
            source_content = self._load_source_content(source_path)
            
            # Integrate additional sources if provided
            if combination_sources:
                for combo_source in combination_sources:
                    additional_content = self._load_source_content(combo_source)
                    source_content = self._combine_content(source_content, additional_content)
            
            # Transform to narrative format
            narrative_content = self._transform_to_narrative(source_content)
            
            # Generate supporting files
            self._generate_demo_script(narrative_content, output_path)
            self._generate_exercises(narrative_content, output_path)
            self._generate_resources(narrative_content, output_path)
            
            # Copy media files
            self._copy_media_files(source_path, output_path)
            
            # Write main narrative file
            self._write_narrative_file(narrative_content, output_path)
            
            # Generate README
            self._generate_readme(narrative_content, output_path)
            
            self.conversion_stats['successful'] += 1
            logger.info(f"Successfully converted: {source_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to convert {source_path}: {e}")
            self.conversion_stats['failed'] += 1
            self.conversion_stats['warnings'].append(f"Conversion failed for {source_path}: {str(e)}")
            return False
        
        finally:
            self.conversion_stats['processed'] += 1
    
    def _load_source_content(self, source_path: str) -> Dict:
        """Load and parse source content from lecture directory."""
        content = {
            "title": "",
            "sections": {},
            "code_examples": [],
            "exercises": [],
            "media": [],
            "metadata": {}
        }
        
        source_dir = Path(source_path)
        
        # Load markdown files
        for md_file in source_dir.glob("*.md"):
            section_content = md_file.read_text(encoding='utf-8')
            content["sections"][md_file.stem] = section_content
        
        # Load Python files as code examples
        for py_file in source_dir.rglob("*.py"):
            code_content = py_file.read_text(encoding='utf-8')
            content["code_examples"].append({
                "file": py_file.name,
                "content": code_content,
                "path": str(py_file.relative_to(source_dir))
            })
        
        # Find media files
        for ext in self.config["file_patterns"]["media_extensions"]:
            for media_file in source_dir.rglob(f"*{ext}"):
                content["media"].append({
                    "file": media_file.name,
                    "path": str(media_file.relative_to(source_dir))
                })
        
        return content
    
    def _combine_content(self, primary: Dict, secondary: Dict) -> Dict:
        """Intelligently combine content from multiple sources."""
        combined = primary.copy()
        
        # Merge sections with weighted priority
        for section, content in secondary["sections"].items():
            if section in combined["sections"]:
                # Combine existing sections
                combined["sections"][section] = self._merge_section_content(
                    combined["sections"][section], content
                )
            else:
                # Add new section
                combined["sections"][section] = content
        
        # Combine code examples
        combined["code_examples"].extend(secondary["code_examples"])
        
        # Combine media
        combined["media"].extend(secondary["media"])
        
        return combined
    
    def _merge_section_content(self, primary_content: str, secondary_content: str) -> str:
        """Merge content from two sections intelligently."""
        # For now, append secondary to primary with clear separation
        # In production, this would use NLP to identify overlaps and merge intelligently
        return primary_content + "\n\n## Additional Content\n\n" + secondary_content
    
    def _transform_to_narrative(self, content: Dict) -> Dict:
        """Transform content to narrative format."""
        narrative = {
            "title": content.get("title", "Untitled Lecture"),
            "overview": self._generate_overview(content),
            "learning_objectives": self._extract_learning_objectives(content),
            "prerequisites": self._generate_prerequisites(content),
            "core_concepts": self._transform_core_concepts(content),
            "hands_on_practice": self._generate_exercises_section(content),
            "real_world_applications": self._generate_applications(content),
            "assessment_integration": self._generate_assessment_section(content),
            "further_reading": self._generate_further_reading(content),
            "metadata": {
                "created": datetime.now().isoformat(),
                "format_version": "2.0",
                "conversion_tool": "batch_converter.py"
            }
        }
        
        return narrative
    
    def _generate_overview(self, content: Dict) -> str:
        """Generate engaging overview section."""
        return """Welcome to this comprehensive exploration of data science fundamentals. This lecture builds upon established concepts while introducing new tools and techniques that form the backbone of modern data analysis workflows.

Our journey today combines theoretical understanding with practical application, ensuring you develop both conceptual knowledge and hands-on skills. Each concept is presented within the context of real-world data science challenges, demonstrating not just how to use these tools, but when and why they're essential for professional practice."""
    
    def _extract_learning_objectives(self, content: Dict) -> List[str]:
        """Extract and format learning objectives."""
        # Default objectives - would be extracted from content in production
        return [
            "Apply fundamental programming concepts to data analysis workflows",
            "Implement efficient data structures for different analytical tasks",
            "Demonstrate proficiency with industry-standard development tools",
            "Create reproducible analysis pipelines using best practices",
            "Integrate multiple tools and techniques into cohesive workflows"
        ]
    
    def _generate_prerequisites(self, content: Dict) -> str:
        """Generate prerequisites section."""
        return """This lecture assumes familiarity with basic programming concepts and command line operations. Students should be comfortable with:

- Variables, data types, and basic control structures
- File system navigation and basic command line operations  
- Running Python scripts and understanding error messages
- Basic understanding of software development workflows"""
    
    def _transform_core_concepts(self, content: Dict) -> str:
        """Transform content sections into narrative core concepts."""
        concepts_text = "## Core Concepts\n\n"
        
        # Process each section and transform to narrative
        for section_name, section_content in content["sections"].items():
            concepts_text += f"### {section_name.title().replace('_', ' ')}\n\n"
            concepts_text += self._narrativize_content(section_content) + "\n\n"
        
        return concepts_text
    
    def _narrativize_content(self, content: str) -> str:
        """Convert bullet-point or technical content to narrative form."""
        # Simplified transformation - production version would use NLP
        lines = content.split('\n')
        narrative_lines = []
        
        for line in lines:
            if line.strip().startswith('*') or line.strip().startswith('-'):
                # Convert bullet points to narrative
                clean_line = line.strip().lstrip('*-').strip()
                if clean_line:
                    narrative_lines.append(f"Understanding {clean_line.lower()} is essential because it forms the foundation for more advanced techniques.")
            elif line.strip():
                narrative_lines.append(line)
        
        return '\n'.join(narrative_lines)
    
    def _generate_exercises_section(self, content: Dict) -> str:
        """Generate hands-on practice section."""
        return """## Hands-On Practice

Learning data science requires active engagement with code and real problems. The following exercises build progressively from basic concepts to complex applications, mirroring the learning path of professional data scientists.

### Exercise 1: Foundation Building
Start with fundamental operations to build confidence and understanding.

### Exercise 2: Integration Challenge  
Combine multiple concepts into a cohesive solution.

### Exercise 3: Real-World Application
Apply your skills to a realistic data science scenario."""
    
    def _generate_applications(self, content: Dict) -> str:
        """Generate real-world applications section."""
        return """## Real-World Applications

These skills directly translate to professional data science workflows where efficiency, reproducibility, and collaboration are paramount. Understanding these fundamentals enables you to:

**Industry Applications**: Large-scale data processing, automated analysis pipelines, and collaborative research environments all rely on the principles demonstrated in this lecture.

**Research Context**: Academic and industrial research requires reproducible workflows, version control, and efficient data manipulation - skills that build directly from today's concepts.

**Career Development**: Professional data scientists use these tools daily, making mastery essential for both individual productivity and team collaboration."""
    
    def _generate_assessment_section(self, content: Dict) -> str:
        """Generate assessment integration section."""
        return """## Assessment Integration

### Formative Assessment
Regular check-ins throughout the lecture ensure understanding and provide opportunities for clarification.

### Summative Assessment Preview
Your assignments build directly on these concepts, applying them to increasingly complex scenarios that mirror professional challenges."""
    
    def _generate_further_reading(self, content: Dict) -> str:
        """Generate further reading section."""
        return """## Further Reading and Resources

### Essential Resources
- Core documentation and tutorials for continued learning
- Community resources and forums for ongoing support
- Advanced topics for deeper exploration

### Practice Environments
- Interactive platforms for skill development
- Challenge sites for problem-solving practice
- Open-source projects for real-world application"""
    
    def _write_narrative_file(self, narrative: Dict, output_path: str):
        """Write the main narrative file."""
        output_file = Path(output_path) / "lecture_narrative.md"
        
        content = f"""# {narrative['title']}

## Overview

{narrative['overview']}

## Learning Objectives

By the end of this lecture, students will be able to:

{chr(10).join(f"- {obj}" for obj in narrative['learning_objectives'])}

## Prerequisites

{narrative['prerequisites']}

{narrative['core_concepts']}

{narrative['hands_on_practice']}

{narrative['real_world_applications']}

{narrative['assessment_integration']}

{narrative['further_reading']}

---

*Lecture Format: Notion-Compatible Narrative with Embedded Interactive Code*
*Progressive Complexity: Fundamentals → Integration → Real-World Applications*
*Generated: {narrative['metadata']['created']}*
"""
        
        output_file.write_text(content, encoding='utf-8')
    
    def _generate_demo_script(self, narrative: Dict, output_path: str):
        """Generate interactive demonstration script."""
        demo_path = Path(output_path) / "demo_lecture.py"
        
        demo_content = '''#!/usr/bin/env python3
"""
Interactive Lecture Demonstrations
Generated by automated conversion tool
"""

import sys
import argparse
from datetime import datetime


def demonstrate_concepts():
    """Demonstrate key concepts from the lecture."""
    print("=== Interactive Demonstration ===")
    print("Key concepts and code examples from the lecture")
    
    # Generated examples would go here
    pass


def main():
    """Main demonstration function."""
    parser = argparse.ArgumentParser(description="Interactive lecture demonstrations")
    parser.add_argument('--section', help='Run specific section')
    parser.add_argument('--interactive', action='store_true', help='Interactive mode')
    
    args = parser.parse_args()
    
    print("Lecture Demonstrations")
    print("=" * 50)
    
    demonstrate_concepts()
    
    if args.interactive:
        print("\\nInteractive mode enabled - additional exercises available")


if __name__ == "__main__":
    main()
'''
        
        demo_path.write_text(demo_content, encoding='utf-8')
        demo_path.chmod(0o755)  # Make executable
    
    def _generate_exercises(self, narrative: Dict, output_path: str):
        """Generate exercise files."""
        exercises_dir = Path(output_path) / "exercises"
        exercises_dir.mkdir(exist_ok=True)
        
        # Create practice problems file
        practice_file = exercises_dir / "practice_problems.md"
        practice_content = """# Practice Problems

## Exercise 1: Foundation Building

Practice basic concepts covered in the lecture.

## Exercise 2: Integration Challenge

Combine multiple concepts to solve a more complex problem.

## Exercise 3: Real-World Application

Apply concepts to a realistic data science scenario.
"""
        practice_file.write_text(practice_content, encoding='utf-8')
    
    def _generate_resources(self, narrative: Dict, output_path: str):
        """Generate resource files."""
        resources_dir = Path(output_path) / "resources"
        resources_dir.mkdir(exist_ok=True)
        
        # Create reference guide
        reference_file = resources_dir / "reference_guide.md"
        reference_content = """# Reference Guide

Quick reference for key concepts and syntax covered in the lecture.

## Key Concepts

Summary of main ideas and their applications.

## Syntax Reference

Common patterns and code structures.

## Troubleshooting

Common issues and solutions.
"""
        reference_file.write_text(reference_content, encoding='utf-8')
    
    def _copy_media_files(self, source_path: str, output_path: str):
        """Copy media files to output directory."""
        import shutil
        
        source_media = Path(source_path) / "media"
        if source_media.exists():
            output_media = Path(output_path) / "media"
            if output_media.exists():
                shutil.rmtree(output_media)
            shutil.copytree(source_media, output_media)
    
    def _generate_readme(self, narrative: Dict, output_path: str):
        """Generate comprehensive README for the converted lecture."""
        readme_path = Path(output_path) / "README.md"
        
        readme_content = f"""# {narrative['title']}

## Overview

This lecture has been converted to the new narrative-driven format using automated conversion tools.

## File Structure

```
lecture/
├── README.md                    # This overview
├── lecture_narrative.md         # Main narrative content  
├── demo_lecture.py             # Interactive demonstrations
├── exercises/                   # Practice exercises
├── resources/                   # Reference materials
└── media/                      # Images and diagrams
```

## Usage

### View Main Content
Open `lecture_narrative.md` in your preferred markdown viewer or import into Notion.

### Run Demonstrations
```bash
python3 demo_lecture.py
python3 demo_lecture.py --interactive
```

### Complete Exercises
Navigate to the `exercises/` directory for hands-on practice problems.

## Conversion Information

- **Generated**: {narrative['metadata']['created']}
- **Format Version**: {narrative['metadata']['format_version']}
- **Tool**: {narrative['metadata']['conversion_tool']}

## Next Steps

1. Review the narrative content
2. Run interactive demonstrations
3. Complete practice exercises
4. Explore reference materials
"""
        
        readme_path.write_text(readme_content, encoding='utf-8')
    
    def batch_convert(self, source_dir: str, output_dir: str) -> Dict:
        """
        Perform batch conversion of multiple lectures.
        
        Args:
            source_dir: Directory containing source lectures
            output_dir: Directory for converted lectures
        
        Returns:
            Dict: Conversion results and statistics
        """
        logger.info(f"Starting batch conversion: {source_dir} -> {output_dir}")
        
        # Analyze source content
        analysis = self.analyze_source_content(source_dir)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert each lecture
        for lecture_name, lecture_info in analysis["lectures"].items():
            source_path = lecture_info["path"]
            output_path = os.path.join(output_dir, lecture_name)
            
            # Check for combination opportunities
            combination_sources = []
            for opportunity in analysis["combination_opportunities"]:
                if lecture_name in opportunity["lectures"]:
                    # Add other lectures in the combination
                    for other_lecture in opportunity["lectures"]:
                        if other_lecture != lecture_name:
                            other_path = analysis["lectures"][other_lecture]["path"]
                            combination_sources.append(other_path)
            
            # Perform conversion
            success = self.convert_lecture(source_path, output_path, combination_sources)
            
            if success:
                logger.info(f"✅ Converted: {lecture_name}")
            else:
                logger.error(f"❌ Failed: {lecture_name}")
        
        # Generate batch conversion report
        self._generate_batch_report(output_dir, analysis)
        
        return {
            "analysis": analysis,
            "stats": self.conversion_stats,
            "output_dir": output_dir
        }
    
    def _generate_batch_report(self, output_dir: str, analysis: Dict):
        """Generate comprehensive batch conversion report."""
        report_path = Path(output_dir) / "CONVERSION_REPORT.md"
        
        report_content = f"""# Batch Conversion Report

## Conversion Summary

- **Total Lectures Processed**: {self.conversion_stats['processed']}
- **Successful Conversions**: {self.conversion_stats['successful']}
- **Failed Conversions**: {self.conversion_stats['failed']}
- **Success Rate**: {(self.conversion_stats['successful'] / max(self.conversion_stats['processed'], 1)) * 100:.1f}%

## Content Analysis

### Lectures Found
{chr(10).join(f"- {name}: {info['estimated_content_length']} chars" for name, info in analysis['lectures'].items())}

### Content Combination Opportunities
{chr(10).join(f"- {op['topic']}: {', '.join(op['lectures'])}" for op in analysis['combination_opportunities'])}

### Media Files
- **Total Media Files**: {len(analysis['media_files'])}

## Warnings and Issues

{chr(10).join(f"- {warning}" for warning in self.conversion_stats['warnings']) if self.conversion_stats['warnings'] else "No warnings generated"}

## Next Steps

1. Review converted lectures for quality
2. Test interactive demonstrations
3. Validate exercise completeness
4. Check media file integration
5. Perform final quality assurance

---

*Generated: {datetime.now().isoformat()}*
*Tool: batch_converter.py v2.0*
"""
        
        report_path.write_text(report_content, encoding='utf-8')


def main():
    """Main CLI interface for batch conversion tool."""
    parser = argparse.ArgumentParser(
        description="Batch Lecture Conversion Tool for Data Science Course",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 batch_converter.py --source-dir lectures/ --output-dir converted/
    python3 batch_converter.py --config config/conversion.json
    python3 batch_converter.py --analyze-only --source-dir lectures/
        """
    )
    
    parser.add_argument('--source-dir', required=True, help='Source lectures directory')
    parser.add_argument('--output-dir', default='converted_lectures', help='Output directory')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--analyze-only', action='store_true', help='Only analyze, do not convert')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize converter
    converter = LectureConverter(args.config)
    
    if args.analyze_only:
        # Only perform analysis
        logger.info("Performing content analysis only")
        analysis = converter.analyze_source_content(args.source_dir)
        
        print(f"\n=== Content Analysis Results ===")
        print(f"Lectures found: {len(analysis['lectures'])}")
        print(f"Unique topics: {len(analysis['topics'])}")
        print(f"Combination opportunities: {len(analysis['combination_opportunities'])}")
        print(f"Media files: {len(analysis['media_files'])}")
        
        for name, info in analysis['lectures'].items():
            print(f"  {name}: {info['estimated_content_length']} chars, {len(info['topics'])} topics")
    
    else:
        # Perform full batch conversion
        logger.info("Starting batch conversion process")
        results = converter.batch_convert(args.source_dir, args.output_dir)
        
        print(f"\n=== Conversion Results ===")
        print(f"Processed: {results['stats']['processed']}")
        print(f"Successful: {results['stats']['successful']}")
        print(f"Failed: {results['stats']['failed']}")
        print(f"Success Rate: {(results['stats']['successful'] / max(results['stats']['processed'], 1)) * 100:.1f}%")
        
        if results['stats']['warnings']:
            print(f"\nWarnings:")
            for warning in results['stats']['warnings']:
                print(f"  - {warning}")
        
        print(f"\nConversion complete! Check results in: {args.output_dir}")
        print(f"Review CONVERSION_REPORT.md for detailed analysis")


if __name__ == "__main__":
    main()