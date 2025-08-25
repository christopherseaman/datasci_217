#!/usr/bin/env python3
"""
Deployment Script for DataSci 217 Course Materials
Migrates content from work/ directory to proper lecture structure
Validates completeness and creates missing components
"""

import os
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

class CourseDeployment:
    def __init__(self, base_path: str = "/home/christopher/projects/datasci_217"):
        self.base_path = Path(base_path)
        self.work_dir = self.base_path / "work"
        self.lectures_dir = self.base_path / "lectures"
        self.existing_dirs = [f"{i:02d}" for i in range(1, 13) if (self.base_path / f"{i:02d}").exists()]
        self.deployment_log = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log deployment activities"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        self.deployment_log.append(log_entry)
        print(log_entry)
    
    def validate_source_content(self) -> Dict[str, bool]:
        """Validate that all required source content exists"""
        self.log("Validating source content in work/ directory")
        
        validation = {
            "narrative_files_exist": True,
            "narrative_count": 0,
            "missing_narratives": []
        }
        
        # Check for all 10 narrative files
        for i in range(1, 11):
            narrative_file = self.work_dir / f"L{i:02d}_narrative_content.md"
            if narrative_file.exists():
                validation["narrative_count"] += 1
                self.log(f"✓ Found narrative content for lecture {i}")
            else:
                validation["narrative_files_exist"] = False
                validation["missing_narratives"].append(i)
                self.log(f"✗ Missing narrative content for lecture {i}", "ERROR")
        
        self.log(f"Narrative content validation: {validation['narrative_count']}/10 files found")
        return validation
    
    def validate_existing_structure(self) -> Dict[str, any]:
        """Analyze existing lecture directory structure"""
        self.log("Analyzing existing lecture directory structure")
        
        structure = {
            "directories": self.existing_dirs,
            "assignments": {},
            "index_files": {},
            "media_dirs": {},
            "demo_files": {}
        }
        
        for dir_name in self.existing_dirs:
            dir_path = self.base_path / dir_name
            
            # Check for assignment files
            assignment_file = dir_path / "assignment.md"
            structure["assignments"][dir_name] = assignment_file.exists()
            
            # Check for index files  
            index_file = dir_path / "index.md"
            structure["index_files"][dir_name] = index_file.exists()
            
            # Check for media directories
            media_dir = dir_path / "media"
            structure["media_dirs"][dir_name] = media_dir.exists()
            
            # Check for demo files
            demo_files = list(dir_path.glob("demo.*"))
            structure["demo_files"][dir_name] = len(demo_files) > 0
            
            self.log(f"Directory {dir_name}: assignment={assignment_file.exists()}, index={index_file.exists()}, media={media_dir.exists()}")
        
        return structure
    
    def create_deployment_structure(self):
        """Create proper lectures/ directory structure"""
        self.log("Creating deployment directory structure")
        
        if self.lectures_dir.exists():
            self.log(f"Lectures directory already exists at {self.lectures_dir}")
            backup_name = f"lectures_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            backup_path = self.base_path / backup_name
            shutil.move(str(self.lectures_dir), str(backup_path))
            self.log(f"Existing lectures directory backed up to {backup_path}")
        
        # Create lecture directories 01-10
        for i in range(1, 11):
            lecture_dir = self.lectures_dir / f"{i:02d}"
            lecture_dir.mkdir(parents=True, exist_ok=True)
            
            # Create subdirectories
            (lecture_dir / "media").mkdir(exist_ok=True)
            (lecture_dir / "code").mkdir(exist_ok=True)
            (lecture_dir / "exercises").mkdir(exist_ok=True)
            
            self.log(f"✓ Created structure for lecture {i:02d}")
    
    def migrate_narrative_content(self):
        """Migrate narrative content from work/ to lectures/"""
        self.log("Migrating narrative content to lecture directories")
        
        for i in range(1, 11):
            source_file = self.work_dir / f"L{i:02d}_narrative_content.md"
            target_dir = self.lectures_dir / f"{i:02d}"
            target_file = target_dir / "lecture.md"
            
            if source_file.exists():
                shutil.copy2(str(source_file), str(target_file))
                self.log(f"✓ Migrated narrative content for lecture {i:02d}")
            else:
                self.log(f"✗ Missing narrative content for lecture {i:02d}", "ERROR")
    
    def migrate_existing_components(self):
        """Migrate existing assignments, media, and support files"""
        self.log("Migrating existing course components")
        
        # Mapping of existing directories to new lecture numbers
        migration_map = {
            "01": "01", "02": "02", "03": "03", "04": "04",
            "05": "05", "06": "06", "07": "07", "08": "08", 
            "09": "09", "11": "10"  # Note: 11 -> 10 mapping
        }
        
        for old_dir, new_dir in migration_map.items():
            old_path = self.base_path / old_dir
            new_path = self.lectures_dir / new_dir
            
            if not old_path.exists():
                continue
                
            # Migrate assignment files
            old_assignment = old_path / "assignment.md"
            if old_assignment.exists():
                shutil.copy2(str(old_assignment), str(new_path / "assignment.md"))
                self.log(f"✓ Migrated assignment from {old_dir} to {new_dir}")
            
            # Migrate index files
            old_index = old_path / "index.md"
            if old_index.exists():
                shutil.copy2(str(old_index), str(new_path / "index.md"))
                self.log(f"✓ Migrated index from {old_dir} to {new_dir}")
            
            # Migrate media directories
            old_media = old_path / "media"
            if old_media.exists():
                new_media = new_path / "media"
                if new_media.exists():
                    shutil.rmtree(str(new_media))
                shutil.copytree(str(old_media), str(new_media))
                self.log(f"✓ Migrated media from {old_dir} to {new_dir}")
            
            # Migrate demo files
            for demo_file in old_path.glob("demo.*"):
                shutil.copy2(str(demo_file), str(new_path / demo_file.name))
                self.log(f"✓ Migrated {demo_file.name} from {old_dir} to {new_dir}")
            
            # Migrate Python files
            for py_file in old_path.glob("*.py"):
                target_code_dir = new_path / "code"
                target_code_dir.mkdir(exist_ok=True)
                shutil.copy2(str(py_file), str(target_code_dir / py_file.name))
                self.log(f"✓ Migrated {py_file.name} to {new_dir}/code/")
    
    def create_missing_assignments(self):
        """Create placeholder assignment files for missing lectures"""
        self.log("Creating missing assignment files")
        
        assignment_template = """# Assignment {lecture_num}: {title}

## Overview
Complete the following exercises to reinforce concepts from Lecture {lecture_num}.

## Prerequisites
- Completion of Lecture {lecture_num}
- Understanding of core concepts covered in class

## Exercises

### Exercise 1: Core Concepts
*TODO: Add specific exercises based on lecture content*

### Exercise 2: Practical Application
*TODO: Add hands-on programming exercises*

### Exercise 3: Critical Thinking
*TODO: Add analysis or design questions*

## Submission Guidelines
- Submit all code as .py files
- Include comments explaining your approach
- Test your code with provided examples
- Submit written responses as markdown (.md) files

## Assessment Criteria
- Correctness of implementation
- Code quality and style
- Understanding demonstrated in explanations
- Completeness of submission

## Due Date
*TODO: Set appropriate due date*

---
*This assignment was auto-generated and needs customization based on lecture content.*
"""
        
        lecture_titles = {
            1: "Command Line Fundamentals + Python Setup",
            2: "Data Structures + Version Control", 
            3: "NumPy/Pandas Foundations",
            4: "Data Analysis/Visualization",
            5: "Applied Projects/Best Practices",
            6: "Scientific Computing",
            7: "Data Manipulation Advanced", 
            8: "Statistical Analysis/Visualization",
            9: "Machine Learning/Advanced Analysis",
            10: "Applied Projects/Clinical Integration"
        }
        
        for i in range(1, 11):
            assignment_file = self.lectures_dir / f"{i:02d}" / "assignment.md"
            if not assignment_file.exists():
                content = assignment_template.format(
                    lecture_num=i,
                    title=lecture_titles.get(i, f"Lecture {i}")
                )
                assignment_file.write_text(content)
                self.log(f"✓ Created assignment template for lecture {i:02d}")
    
    def validate_deployment(self) -> Dict[str, any]:
        """Validate the completed deployment"""
        self.log("Validating deployment completeness")
        
        validation = {
            "lecture_dirs": 0,
            "lecture_files": 0,
            "assignment_files": 0,
            "missing_components": [],
            "success": True
        }
        
        for i in range(1, 11):
            lecture_dir = self.lectures_dir / f"{i:02d}"
            lecture_file = lecture_dir / "lecture.md"
            assignment_file = lecture_dir / "assignment.md"
            
            if lecture_dir.exists():
                validation["lecture_dirs"] += 1
            else:
                validation["missing_components"].append(f"lecture_{i:02d}_directory")
                validation["success"] = False
            
            if lecture_file.exists():
                validation["lecture_files"] += 1
            else:
                validation["missing_components"].append(f"lecture_{i:02d}_content")
                validation["success"] = False
                
            if assignment_file.exists():
                validation["assignment_files"] += 1
            else:
                validation["missing_components"].append(f"assignment_{i:02d}")
                validation["success"] = False
        
        self.log(f"Deployment validation: {validation['lecture_dirs']}/10 directories, {validation['lecture_files']}/10 lectures, {validation['assignment_files']}/10 assignments")
        
        return validation
    
    def generate_deployment_report(self, validation_results: Dict):
        """Generate comprehensive deployment report"""
        report_content = f"""# Course Deployment Report
**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Deployment Script Version**: 1.0

## Deployment Summary
- **Status**: {'SUCCESS' if validation_results['success'] else 'INCOMPLETE'}
- **Lecture Directories**: {validation_results['lecture_dirs']}/10
- **Lecture Content Files**: {validation_results['lecture_files']}/10  
- **Assignment Files**: {validation_results['assignment_files']}/10

## Migration Details
"""
        
        if validation_results['missing_components']:
            report_content += f"""
## Missing Components
{chr(10).join(f'- {component}' for component in validation_results['missing_components'])}
"""
        
        report_content += f"""
## Deployment Log
```
{chr(10).join(self.deployment_log)}
```

## Next Steps
1. Review and customize generated assignment templates
2. Validate all code examples in migrated content
3. Test course progression and dependencies
4. Update any absolute file paths to relative paths
5. Prepare for quality assurance testing
"""
        
        report_file = self.base_path / "scripts" / "deployment_report.md"
        report_file.write_text(report_content)
        self.log(f"✓ Generated deployment report at {report_file}")
    
    def execute_full_deployment(self):
        """Execute complete deployment workflow"""
        self.log("Starting full course deployment")
        
        # Phase 1: Validation
        source_validation = self.validate_source_content()
        existing_structure = self.validate_existing_structure()
        
        if not source_validation["narrative_files_exist"]:
            self.log("CRITICAL: Missing narrative content files - cannot proceed", "ERROR")
            return False
        
        # Phase 2: Structure Creation
        self.create_deployment_structure()
        
        # Phase 3: Content Migration
        self.migrate_narrative_content()
        self.migrate_existing_components()
        
        # Phase 4: Gap Filling
        self.create_missing_assignments()
        
        # Phase 5: Validation
        deployment_validation = self.validate_deployment()
        
        # Phase 6: Reporting
        self.generate_deployment_report(deployment_validation)
        
        self.log(f"Deployment {'COMPLETED SUCCESSFULLY' if deployment_validation['success'] else 'COMPLETED WITH ISSUES'}")
        return deployment_validation['success']

if __name__ == "__main__":
    deployer = CourseDeployment()
    success = deployer.execute_full_deployment()
    exit(0 if success else 1)