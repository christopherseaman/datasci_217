#!/usr/bin/env python3
"""
Enhanced Lecture Conversion Pipeline with Content Combination
 
This enhanced converter builds on the original conversion script by adding
capabilities to combine content from multiple lectures into unified narratives,
as demonstrated in the Lecture 1 prototype.

Usage:
    python scripts/enhanced_converter.py <primary_lecture> [--combine <secondary_lectures>] [--dry-run]

Examples:
    python scripts/enhanced_converter.py 01 --combine 02,03 --dry-run
    python scripts/enhanced_converter.py 04 --combine 05
    
Features:
- Content combination from multiple source lectures
- Intelligent narrative weaving based on prototype patterns
- Enhanced code extraction and integration
- Prototype-based template customization
"""

import os
import sys
import argparse
import json
import shutil
import re
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

@dataclass
class ContentSource:
    """Represents a source of content with its integration percentage."""
    lecture_number: str
    percentage: int  # How much of this lecture to integrate (0-100)
    priority: str   # 'primary', 'secondary', 'tertiary'
    focus_areas: List[str]  # Specific topics to extract

class EnhancedLectureConverter:
    """Enhanced converter with content combination capabilities."""
    
    def __init__(self, primary_lecture: str, combination_lectures: List[str] = None, dry_run: bool = False):
        self.primary_lecture = primary_lecture.zfill(2)
        self.combination_lectures = [l.zfill(2) for l in (combination_lectures or [])]
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent
        
        # Enhanced template system
        self.templates_dir = self.project_root / "templates"
        self.prototype_dir = self.project_root / "prototypes" / "lecture_01_prototype"
        
        # Target structure
        self.new_lectures_dir = self.project_root / "lectures_new_enhanced"
        self.target_dir = self.new_lectures_dir / f"lecture_{self.primary_lecture}"
        
        # Content combination patterns from prototype
        self.combination_patterns = {
            "01": {
                "primary": ContentSource("01", 90, "primary", ["python_basics", "variables", "functions"]),
                "secondary": ContentSource("02", 25, "secondary", ["environment_setup", "git_basics"]),
                "tertiary": ContentSource("03", 30, "tertiary", ["command_line", "shell_basics"])
            }
            # Add more patterns as needed
        }
    
    def convert(self):
        """Enhanced conversion with content combination."""
        print(f"Enhanced conversion: Lecture {self.primary_lecture}")
        if self.combination_lectures:
            print(f"Combining with: {', '.join(self.combination_lectures)}")
        
        # Create enhanced structure
        self._create_enhanced_structure()
        
        # Read all content sources
        all_content = self._read_all_content_sources()
        
        # Apply combination pattern or default strategy
        combined_content = self._combine_content(all_content)
        
        # Generate enhanced materials
        self._generate_enhanced_narrative(combined_content)
        self._generate_enhanced_demo(combined_content)
        self._generate_enhanced_exercises(combined_content)
        self._copy_and_organize_media(combined_content)
        
        print(f"✓ Enhanced conversion complete for Lecture {self.primary_lecture}")
        return True
    
    def _create_enhanced_structure(self):
        """Create enhanced directory structure based on prototype."""
        dirs_to_create = [
            self.new_lectures_dir,
            self.target_dir,
            self.target_dir / "media",
            self.target_dir / "exercises", 
            self.target_dir / "resources"
        ]
        
        for dir_path in dirs_to_create:
            if not self.dry_run:
                dir_path.mkdir(parents=True, exist_ok=True)
            print(f"  Creating: {dir_path}")
    
    def _read_all_content_sources(self) -> Dict:
        """Read content from primary and combination sources."""
        all_sources = {}
        
        # Read primary lecture
        all_sources[self.primary_lecture] = self._read_lecture_content(self.primary_lecture)
        
        # Read combination lectures
        for lecture_num in self.combination_lectures:
            all_sources[lecture_num] = self._read_lecture_content(lecture_num)
        
        return all_sources
    
    def _read_lecture_content(self, lecture_num: str) -> Dict:
        """Read content from a specific lecture directory."""
        source_dir = self.project_root / lecture_num
        content = {
            'lecture_number': lecture_num,
            'title': f'Lecture {lecture_num}',
            'index_md': '',
            'demo_content': '',
            'assignment_md': '',
            'media_files': [],
            'extracted_concepts': []
        }
        
        # Read main content
        index_path = source_dir / "index.md"
        if index_path.exists():
            content['index_md'] = index_path.read_text(encoding='utf-8')
            content['extracted_concepts'] = self._extract_concepts(content['index_md'])
            
            # Extract title
            title_match = re.search(r'^#\s+(.+)$', content['index_md'], re.MULTILINE)
            if title_match:
                content['title'] = title_match.group(1)
        
        # Read demo content
        for demo_file in ['demo.md', 'demo.py', 'demo.ipynb']:
            demo_path = source_dir / demo_file
            if demo_path.exists():
                if demo_file.endswith('.ipynb'):
                    content['demo_content'] = self._extract_from_notebook(demo_path)
                else:
                    content['demo_content'] = demo_path.read_text(encoding='utf-8')
                break
        
        # Read assignment
        assignment_path = source_dir / "assignment.md"
        if assignment_path.exists():
            content['assignment_md'] = assignment_path.read_text(encoding='utf-8')
        
        # Collect media
        media_dir = source_dir / "media"
        if media_dir.exists():
            content['media_files'] = list(media_dir.glob("*"))
        
        return content
    
    def _extract_concepts(self, markdown_content: str) -> List[str]:
        """Extract key concepts from markdown content."""
        concepts = []
        
        # Extract headers as concepts
        header_pattern = r'^#+\s+(.+)$'
        headers = re.findall(header_pattern, markdown_content, re.MULTILINE)
        concepts.extend(headers)
        
        # Extract code block languages/topics
        code_pattern = r'```(\w+)'
        code_langs = re.findall(code_pattern, markdown_content)
        concepts.extend([f"code_{lang}" for lang in code_langs])
        
        # Extract command mentions
        command_pattern = r'`([a-zA-Z]+)`'
        commands = re.findall(command_pattern, markdown_content)
        concepts.extend([f"command_{cmd}" for cmd in commands if len(cmd) > 1])
        
        return list(set(concepts))  # Remove duplicates
    
    def _combine_content(self, all_content: Dict) -> Dict:
        """Combine content from multiple sources intelligently."""
        # Check if we have a predefined combination pattern
        if self.primary_lecture in self.combination_patterns:
            return self._apply_combination_pattern(all_content)
        else:
            return self._apply_default_combination(all_content)
    
    def _apply_combination_pattern(self, all_content: Dict) -> Dict:
        """Apply predefined combination pattern (like Lecture 1 prototype)."""
        pattern = self.combination_patterns[self.primary_lecture]
        combined = {
            'title': '',
            'narrative_sections': [],
            'demo_sections': [],
            'exercises': [],
            'media_files': [],
            'resources': []
        }
        
        # Process each source according to its role in the pattern
        for source_key, source_info in pattern.items():
            lecture_num = source_info.lecture_number
            if lecture_num in all_content:
                content = all_content[lecture_num]
                
                # Extract relevant portions based on focus areas and percentage
                extracted = self._extract_focused_content(content, source_info)
                
                if source_info.priority == "primary":
                    combined['title'] = extracted['title']
                    combined['narrative_sections'].extend(extracted['sections'])
                else:
                    # Integrate secondary/tertiary content
                    combined['narrative_sections'].extend(extracted['integrated_sections'])
                
                combined['demo_sections'].extend(extracted['demo_parts'])
                combined['exercises'].extend(extracted['exercises'])
                combined['media_files'].extend(content['media_files'])
        
        return combined
    
    def _extract_focused_content(self, content: Dict, source_info: ContentSource) -> Dict:
        """Extract content focused on specific areas with percentage filtering."""
        extracted = {
            'title': content['title'],
            'sections': [],
            'integrated_sections': [],
            'demo_parts': [],
            'exercises': []
        }
        
        # Simple implementation - in full version, would use NLP/semantic analysis
        full_text = content['index_md']
        
        # Filter based on focus areas
        if source_info.focus_areas:
            relevant_sections = []
            for focus_area in source_info.focus_areas:
                # Extract sections related to focus area
                area_content = self._extract_sections_by_topic(full_text, focus_area)
                relevant_sections.extend(area_content)
            
            # Apply percentage filter (simplified)
            target_length = int(len(relevant_sections) * source_info.percentage / 100)
            extracted['sections'] = relevant_sections[:target_length]
            
            # For secondary sources, create integrated sections
            if source_info.priority != "primary":
                extracted['integrated_sections'] = self._create_integrated_sections(
                    relevant_sections[:target_length], source_info
                )
        
        return extracted
    
    def _extract_sections_by_topic(self, text: str, topic: str) -> List[str]:
        """Extract sections related to a specific topic."""
        # Simplified topic extraction - would be enhanced with better NLP
        sections = []
        lines = text.split('\n')
        current_section = []
        in_relevant_section = False
        
        topic_keywords = {
            'python_basics': ['python', 'variable', 'syntax', 'data type'],
            'command_line': ['command', 'shell', 'terminal', 'bash', 'cli'],
            'environment_setup': ['install', 'setup', 'environment', 'config'],
            'git_basics': ['git', 'version control', 'repository', 'commit']
        }
        
        keywords = topic_keywords.get(topic, [topic])
        
        for line in lines:
            line_lower = line.lower()
            
            # Check if line contains relevant keywords
            if any(keyword in line_lower for keyword in keywords):
                in_relevant_section = True
                current_section.append(line)
            elif line.startswith('#') and current_section:
                # New section, save current if relevant
                if in_relevant_section:
                    sections.append('\n'.join(current_section))
                current_section = [line]
                in_relevant_section = any(keyword in line_lower for keyword in keywords)
            else:
                current_section.append(line)
        
        # Add final section if relevant
        if in_relevant_section and current_section:
            sections.append('\n'.join(current_section))
        
        return sections
    
    def _create_integrated_sections(self, sections: List[str], source_info: ContentSource) -> List[str]:
        """Create narrative sections that integrate secondary content smoothly."""
        integrated = []
        
        for section in sections:
            # Add contextual introduction for secondary content
            intro = f"Building on these fundamentals, let's explore {source_info.priority} concepts that enhance our workflow:"
            integrated_section = f"{intro}\n\n{section}"
            integrated.append(integrated_section)
        
        return integrated
    
    def _generate_enhanced_narrative(self, combined_content: Dict):
        """Generate narrative using prototype-based patterns."""
        if self.prototype_dir.exists():
            # Use prototype as template
            prototype_narrative = self.prototype_dir / "lecture_01_narrative.md"
            if prototype_narrative.exists():
                template_content = prototype_narrative.read_text(encoding='utf-8')
                
                # Customize for current lecture
                customized = self._customize_prototype_template(template_content, combined_content)
                
                target_file = self.target_dir / f"lecture_{self.primary_lecture}_narrative.md"
                if not self.dry_run:
                    target_file.write_text(customized, encoding='utf-8')
                print(f"  Generated narrative: {target_file}")
                return
        
        # Fallback to original template
        self._generate_standard_narrative(combined_content)
    
    def _customize_prototype_template(self, template: str, content: Dict) -> str:
        """Customize prototype template for different lectures."""
        # Replace lecture-specific content while maintaining structure
        customized = template
        
        # Update title and lecture number
        customized = re.sub(
            r'# Lecture \d+: [^\n]+',
            f"# Lecture {self.primary_lecture}: {content.get('title', 'Enhanced Lecture')}",
            customized
        )
        
        # Update learning objectives based on combined content
        if content.get('narrative_sections'):
            objectives = self._generate_learning_objectives(content['narrative_sections'])
            # Replace objectives section (simplified)
            objectives_section = "By the end of this lecture, students will be able to:\n\n" + "\n".join(f"- {obj}" for obj in objectives)
            customized = re.sub(
                r'By the end of this lecture, students will be able to:.*?(?=##)',
                objectives_section + "\n\n",
                customized,
                flags=re.DOTALL
            )
        
        return customized
    
    def _generate_learning_objectives(self, sections: List[str]) -> List[str]:
        """Generate learning objectives from content sections."""
        # Simplified objective generation
        objectives = []
        concepts = set()
        
        for section in sections:
            # Extract action verbs and concepts
            if 'function' in section.lower():
                concepts.add('create and use functions')
            if 'command' in section.lower():
                concepts.add('execute command line operations')
            if 'python' in section.lower():
                concepts.add('write basic Python programs')
            if 'data' in section.lower():
                concepts.add('process data using programming tools')
        
        return list(concepts)[:6]  # Limit to 6 objectives
    
    def _generate_enhanced_demo(self, combined_content: Dict):
        """Generate demo script based on prototype patterns."""
        if self.prototype_dir.exists():
            prototype_demo = self.prototype_dir / "demo_lecture_01.py"
            if prototype_demo.exists():
                template_content = prototype_demo.read_text(encoding='utf-8')
                
                # Customize demo for current content
                customized = self._customize_demo_template(template_content, combined_content)
                
                target_file = self.target_dir / f"demo_lecture_{self.primary_lecture}.py"
                if not self.dry_run:
                    target_file.write_text(customized, encoding='utf-8')
                print(f"  Generated demo: {target_file}")
                return
        
        # Fallback implementation
        print("  Warning: No prototype demo template found")
    
    def _customize_demo_template(self, template: str, content: Dict) -> str:
        """Customize demo template for different lectures."""
        # Update docstring and lecture references
        customized = re.sub(
            r'Lecture \d+: [^\n]+',
            f"Lecture {self.primary_lecture}: {content.get('title', 'Enhanced Lecture')}",
            template
        )
        
        # Update demonstration sections based on content
        # This would be enhanced with actual code extraction and adaptation
        
        return customized
    
    def _generate_enhanced_exercises(self, combined_content: Dict):
        """Generate exercises based on combined content."""
        exercises_content = "# Enhanced Practice Exercises\n\n"
        exercises_content += f"Based on concepts from Lecture {self.primary_lecture}"
        
        if self.combination_lectures:
            exercises_content += f" and related topics from lectures {', '.join(self.combination_lectures)}"
        
        exercises_content += ".\n\n"
        exercises_content += "## Integrated Challenges\n\n"
        exercises_content += "These exercises combine multiple concepts for deeper learning.\n"
        
        target_file = self.target_dir / "exercises" / "practice_problems.md"
        if not self.dry_run:
            target_file.write_text(exercises_content, encoding='utf-8')
        print(f"  Generated exercises: {target_file}")
    
    def _copy_and_organize_media(self, combined_content: Dict):
        """Copy and organize media files from all sources."""
        target_media = self.target_dir / "media"
        
        all_media = combined_content.get('media_files', [])
        
        for media_file in all_media:
            if media_file.is_file():
                target_file = target_media / media_file.name
                if not self.dry_run:
                    shutil.copy2(media_file, target_file)
                print(f"  Copied media: {media_file.name}")
    
    def _apply_default_combination(self, all_content: Dict) -> Dict:
        """Apply default combination strategy when no pattern exists."""
        # Simple default: primary lecture gets 80%, others split the remaining 20%
        combined = {
            'title': all_content[self.primary_lecture]['title'],
            'narrative_sections': [all_content[self.primary_lecture]['index_md']],
            'demo_sections': [all_content[self.primary_lecture]['demo_content']],
            'exercises': [],
            'media_files': all_content[self.primary_lecture]['media_files'],
            'resources': []
        }
        
        # Add combination content as appendices
        for lecture_num in self.combination_lectures:
            if lecture_num in all_content:
                content = all_content[lecture_num]
                combined['narrative_sections'].append(f"## Supplementary: {content['title']}\n\n{content['index_md'][:1000]}...")
                combined['media_files'].extend(content['media_files'])
        
        return combined
    
    def _extract_from_notebook(self, notebook_path: Path) -> str:
        """Extract content from Jupyter notebook."""
        try:
            with open(notebook_path, 'r', encoding='utf-8') as f:
                notebook = json.load(f)
            
            extracted_content = []
            for cell in notebook.get('cells', []):
                if cell['cell_type'] == 'markdown':
                    extracted_content.append(''.join(cell['source']))
                elif cell['cell_type'] == 'code':
                    code = ''.join(cell['source'])
                    extracted_content.append(f"```python\n{code}\n```")
            
            return '\n\n'.join(extracted_content)
        except Exception as e:
            print(f"Warning: Could not parse notebook {notebook_path}: {e}")
            return ""
    
    def _generate_standard_narrative(self, combined_content: Dict):
        """Generate standard narrative when prototype not available."""
        template_path = self.templates_dir / "lecture_template.md"
        if not template_path.exists():
            print(f"Warning: No template found at {template_path}")
            return
        
        template_content = template_path.read_text(encoding='utf-8')
        
        # Basic substitution
        customized = template_content.replace(
            '{LECTURE_NUMBER}', self.primary_lecture
        ).replace(
            '{LECTURE_TITLE}', combined_content.get('title', 'Combined Lecture')
        )
        
        target_file = self.target_dir / f"lecture_{self.primary_lecture}.md"
        if not self.dry_run:
            target_file.write_text(customized, encoding='utf-8')
        print(f"  Generated narrative: {target_file}")

def main():
    parser = argparse.ArgumentParser(
        description='Enhanced lecture converter with content combination',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python enhanced_converter.py 01 --combine 02,03 --dry-run
    python enhanced_converter.py 04 --combine 05
    python enhanced_converter.py 07  # Single lecture conversion
        """
    )
    
    parser.add_argument('primary_lecture', help='Primary lecture number (e.g., 01, 04)')
    parser.add_argument('--combine', help='Comma-separated list of lectures to combine (e.g., 02,03)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    # Parse combination lectures
    combination_lectures = []
    if args.combine:
        combination_lectures = [l.strip() for l in args.combine.split(',')]
    
    converter = EnhancedLectureConverter(args.primary_lecture, combination_lectures, args.dry_run)
    success = converter.convert()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()