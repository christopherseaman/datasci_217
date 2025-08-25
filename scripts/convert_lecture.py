#!/usr/bin/env python3
"""
Lecture Format Conversion Pipeline

This script converts existing lecture materials to the new standardized format.
It handles both markdown files and Jupyter notebooks, creating the new
Python + Markdown combination format.

Usage:
    python scripts/convert_lecture.py <lecture_number> [--dry-run]

Example:
    python scripts/convert_lecture.py 01
    python scripts/convert_lecture.py 03 --dry-run

Author: Data Science 217 Course Materials
"""

import os
import sys
import argparse
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import re

class LectureConverter:
    """Converts lecture materials to new standardized format."""
    
    def __init__(self, lecture_number: str, dry_run: bool = False):
        self.lecture_number = lecture_number.zfill(2)  # Ensure 2-digit format
        self.dry_run = dry_run
        self.project_root = Path(__file__).parent.parent
        self.templates_dir = self.project_root / "templates"
        self.source_dir = self.project_root / self.lecture_number
        self.backup_dir = self.project_root / "lectures_bkp" / self.lecture_number
        
        # New format structure
        self.new_lectures_dir = self.project_root / "lectures_new"
        self.target_dir = self.new_lectures_dir / f"lecture_{self.lecture_number}"
        
    def convert(self):
        """Main conversion process."""
        print(f"Converting Lecture {self.lecture_number} to new format...")
        
        if not self.source_dir.exists():
            print(f"Error: Source directory {self.source_dir} does not exist")
            return False
            
        # Create target directory structure
        self._create_target_structure()
        
        # Read existing content
        existing_content = self._read_existing_content()
        
        # Generate new materials
        self._generate_narrative_markdown(existing_content)
        self._generate_demo_python(existing_content)
        self._copy_media_files()
        
        print(f"✓ Conversion complete for Lecture {self.lecture_number}")
        if self.dry_run:
            print("  (Dry run - no files were actually created)")
        else:
            print(f"  New materials available in: {self.target_dir}")
            
        return True
    
    def _create_target_structure(self):
        """Create the directory structure for the new format."""
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
            print(f"  Creating directory: {dir_path}")
    
    def _read_existing_content(self) -> Dict:
        """Read and parse existing lecture materials."""
        content = {
            'title': f'Lecture {self.lecture_number}',
            'index_md': '',
            'demo_content': '',
            'assignment_md': '',
            'media_files': []
        }
        
        # Read index.md if it exists
        index_path = self.source_dir / "index.md"
        if index_path.exists():
            content['index_md'] = index_path.read_text(encoding='utf-8')
            # Extract title from first heading
            title_match = re.search(r'^#\s+(.+)$', content['index_md'], re.MULTILINE)
            if title_match:
                content['title'] = title_match.group(1)
        
        # Read demo content (could be .md, .py, or .ipynb)
        for demo_file in ['demo.md', 'demo.py', 'demo.ipynb']:
            demo_path = self.source_dir / demo_file
            if demo_path.exists():
                if demo_file.endswith('.ipynb'):
                    content['demo_content'] = self._extract_from_notebook(demo_path)
                else:
                    content['demo_content'] = demo_path.read_text(encoding='utf-8')
                break
        
        # Read assignment if it exists
        assignment_path = self.source_dir / "assignment.md"
        if assignment_path.exists():
            content['assignment_md'] = assignment_path.read_text(encoding='utf-8')
        
        # List media files
        media_dir = self.source_dir / "media"
        if media_dir.exists():
            content['media_files'] = list(media_dir.glob("*"))
        
        return content
    
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
    
    def _generate_narrative_markdown(self, content: Dict):
        """Generate the new narrative markdown file."""
        template_path = self.templates_dir / "lecture_template.md"
        if not template_path.exists():
            print(f"Warning: Template not found at {template_path}")
            return
        
        template_content = template_path.read_text(encoding='utf-8')
        
        # Replace template placeholders
        new_content = template_content.replace(
            '{LECTURE_NUMBER}', self.lecture_number
        ).replace(
            '{LECTURE_TITLE}', content['title'].replace(f'Lecture {self.lecture_number}:', '').strip()
        )
        
        # Try to extract and integrate existing content
        if content['index_md']:
            # Insert existing content in appropriate sections
            new_content = self._merge_existing_content(new_content, content['index_md'])
        
        target_file = self.target_dir / f"lecture_{self.lecture_number}.md"
        if not self.dry_run:
            target_file.write_text(new_content, encoding='utf-8')
        
        print(f"  Generated: {target_file}")
    
    def _generate_demo_python(self, content: Dict):
        """Generate the new Python demo file."""
        template_path = self.templates_dir / "demo_template.py"
        if not template_path.exists():
            print(f"Warning: Demo template not found at {template_path}")
            return
        
        template_content = template_path.read_text(encoding='utf-8')
        
        # Replace template placeholders
        new_content = template_content.replace(
            '{LECTURE_NUMBER}', self.lecture_number
        ).replace(
            '{LECTURE_TITLE}', content['title'].replace(f'Lecture {self.lecture_number}:', '').strip()
        ).replace(
            '{DATE}', datetime.now().strftime('%Y-%m-%d')
        )
        
        # TODO: Add logic to extract and convert existing demo code
        
        target_file = self.target_dir / f"demo_lecture_{self.lecture_number}.py"
        if not self.dry_run:
            target_file.write_text(new_content, encoding='utf-8')
        
        print(f"  Generated: {target_file}")
    
    def _merge_existing_content(self, template: str, existing: str) -> str:
        """Merge existing content into the new template structure."""
        # This is a simplified implementation
        # In a full implementation, you'd parse the existing markdown
        # and intelligently place content in appropriate template sections
        
        # For now, add existing content as a reference section
        reference_section = f"""

## Original Content Reference

The following content was extracted from the original lecture materials:

{existing}

---
"""
        
        return template + reference_section
    
    def _copy_media_files(self):
        """Copy media files to the new structure."""
        source_media = self.source_dir / "media"
        target_media = self.target_dir / "media"
        
        if not source_media.exists():
            return
        
        for media_file in source_media.glob("*"):
            if media_file.is_file():
                target_file = target_media / media_file.name
                if not self.dry_run:
                    shutil.copy2(media_file, target_file)
                print(f"  Copied media: {media_file.name}")

def main():
    parser = argparse.ArgumentParser(description='Convert lecture to new format')
    parser.add_argument('lecture_number', help='Lecture number to convert (e.g., 01, 03)')
    parser.add_argument('--dry-run', action='store_true', 
                       help='Show what would be done without making changes')
    
    args = parser.parse_args()
    
    converter = LectureConverter(args.lecture_number, args.dry_run)
    success = converter.convert()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()