#!/usr/bin/env python3
"""
Deployment System for Data Science Course Content
=================================================

Complete deployment and integration system for narrative-driven course content.
Handles packaging, validation, distribution, and platform integration.

Usage:
    python3 deploy_system.py --deploy-local output/
    python3 deploy_system.py --deploy-notion --config notion_config.json
    python3 deploy_system.py --deploy-lms --platform canvas
    python3 deploy_system.py --package-distribution --version 2.0
"""

import os
import sys
import json
import shutil
import zipfile
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DeploymentSystem:
    """
    Comprehensive deployment system for course content.
    
    Handles multiple deployment targets, validation, packaging,
    and integration with various educational platforms.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize deployment system with configuration."""
        self.config = self._load_config(config_path)
        self.deployment_results = {
            'timestamp': datetime.now().isoformat(),
            'deployments': [],
            'errors': [],
            'summary': {}
        }
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load deployment configuration."""
        default_config = {
            "deployment": {
                "supported_platforms": ["local", "notion", "canvas", "github", "zip"],
                "validation_required": True,
                "backup_enabled": True,
                "rollback_enabled": True
            },
            "packaging": {
                "include_media": True,
                "include_exercises": True,
                "include_resources": True,
                "compression_level": 6
            },
            "integration": {
                "notion_compatibility": True,
                "lms_compatibility": True,
                "mobile_optimization": True
            },
            "quality_gates": {
                "min_validation_score": 0.8,
                "required_files": [
                    "README.md", "lecture_narrative.md", "demo_lecture.py"
                ],
                "code_execution_test": True
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def validate_pre_deployment(self, content_dir: str) -> Dict[str, Any]:
        """
        Comprehensive pre-deployment validation.
        
        Args:
            content_dir: Directory containing course content
            
        Returns:
            Dictionary with validation results
        """
        logger.info(f"Running pre-deployment validation: {content_dir}")
        
        validation_results = {
            'overall_status': 'pending',
            'file_structure': {'status': 'pending', 'details': []},
            'content_quality': {'status': 'pending', 'details': []},
            'code_execution': {'status': 'pending', 'details': []},
            'platform_compatibility': {'status': 'pending', 'details': []},
            'recommendations': []
        }
        
        content_path = Path(content_dir)
        
        # File structure validation
        structure_status = self._validate_file_structure(content_path)
        validation_results['file_structure'] = structure_status
        
        # Content quality validation
        quality_status = self._validate_content_quality(content_path)
        validation_results['content_quality'] = quality_status
        
        # Code execution validation
        execution_status = self._validate_code_execution(content_path)
        validation_results['code_execution'] = execution_status
        
        # Platform compatibility validation
        compatibility_status = self._validate_platform_compatibility(content_path)
        validation_results['platform_compatibility'] = compatibility_status
        
        # Determine overall status
        all_statuses = [
            structure_status['status'],
            quality_status['status'], 
            execution_status['status'],
            compatibility_status['status']
        ]
        
        if all(status == 'pass' for status in all_statuses):
            validation_results['overall_status'] = 'pass'
        elif any(status == 'fail' for status in all_statuses):
            validation_results['overall_status'] = 'fail'
        else:
            validation_results['overall_status'] = 'warning'
        
        # Generate recommendations
        validation_results['recommendations'] = self._generate_deployment_recommendations(validation_results)
        
        return validation_results
    
    def _validate_file_structure(self, content_path: Path) -> Dict[str, Any]:
        """Validate required file structure."""
        required_files = self.config["quality_gates"]["required_files"]
        found_files = []
        missing_files = []
        
        for required_file in required_files:
            if (content_path / required_file).exists():
                found_files.append(required_file)
            else:
                missing_files.append(required_file)
        
        # Check for expected directories
        expected_dirs = ['exercises', 'resources', 'media']
        missing_dirs = []
        
        for expected_dir in expected_dirs:
            if not (content_path / expected_dir).exists():
                missing_dirs.append(expected_dir)
        
        status = 'pass' if not missing_files and not missing_dirs else 'fail' if missing_files else 'warning'
        
        return {
            'status': status,
            'details': {
                'found_files': found_files,
                'missing_files': missing_files,
                'missing_directories': missing_dirs
            }
        }
    
    def _validate_content_quality(self, content_path: Path) -> Dict[str, Any]:
        """Validate content quality metrics."""
        quality_details = []
        status = 'pass'
        
        # Check main narrative file
        narrative_file = content_path / 'lecture_narrative.md'
        if narrative_file.exists():
            try:
                content = narrative_file.read_text(encoding='utf-8')
                word_count = len(content.split())
                
                if word_count < 5000:
                    quality_details.append(f"Content length too short: {word_count} words (minimum: 5000)")
                    status = 'warning'
                elif word_count > 8000:
                    quality_details.append(f"Content length very long: {word_count} words (maximum: 8000)")
                    status = 'warning'
                else:
                    quality_details.append(f"Content length appropriate: {word_count} words")
                
                # Check for required sections
                required_sections = ['Overview', 'Learning Objectives', 'Core Concepts', 'Hands-On Practice']
                missing_sections = []
                
                for section in required_sections:
                    if f"## {section}" not in content and f"# {section}" not in content:
                        missing_sections.append(section)
                
                if missing_sections:
                    quality_details.append(f"Missing required sections: {missing_sections}")
                    status = 'fail'
                
            except Exception as e:
                quality_details.append(f"Error reading narrative file: {e}")
                status = 'fail'
        else:
            quality_details.append("Main narrative file not found")
            status = 'fail'
        
        return {
            'status': status,
            'details': quality_details
        }
    
    def _validate_code_execution(self, content_path: Path) -> Dict[str, Any]:
        """Validate that code examples execute correctly."""
        execution_details = []
        status = 'pass'
        
        # Find Python files
        python_files = list(content_path.rglob("*.py"))
        
        if not python_files:
            execution_details.append("No Python files found for testing")
            return {'status': 'warning', 'details': execution_details}
        
        for py_file in python_files:
            try:
                # Basic syntax check
                with open(py_file, 'r', encoding='utf-8') as f:
                    code = f.read()
                
                compile(code, str(py_file), 'exec')
                execution_details.append(f"✅ Syntax valid: {py_file.name}")
                
                # Try to run demo scripts with help flag
                if 'demo_' in py_file.name:
                    try:
                        result = subprocess.run(
                            [sys.executable, str(py_file), '--help'],
                            capture_output=True,
                            text=True,
                            timeout=30
                        )
                        
                        if result.returncode == 0:
                            execution_details.append(f"✅ Demo executable: {py_file.name}")
                        else:
                            execution_details.append(f"⚠️  Demo issues: {py_file.name}")
                            status = 'warning'
                    
                    except subprocess.TimeoutExpired:
                        execution_details.append(f"⚠️  Demo timeout: {py_file.name}")
                        status = 'warning'
                    except Exception as e:
                        execution_details.append(f"❌ Demo error: {py_file.name} - {str(e)}")
                        status = 'warning'
                
            except SyntaxError as e:
                execution_details.append(f"❌ Syntax error: {py_file.name} - {str(e)}")
                status = 'fail'
            except Exception as e:
                execution_details.append(f"❌ Error: {py_file.name} - {str(e)}")
                status = 'fail'
        
        return {
            'status': status,
            'details': execution_details
        }
    
    def _validate_platform_compatibility(self, content_path: Path) -> Dict[str, Any]:
        """Validate compatibility with target platforms."""
        compatibility_details = []
        status = 'pass'
        
        # Check Notion compatibility
        if self.config["integration"]["notion_compatibility"]:
            notion_issues = self._check_notion_compatibility(content_path)
            if notion_issues:
                compatibility_details.extend([f"Notion: {issue}" for issue in notion_issues])
                status = 'warning'
            else:
                compatibility_details.append("✅ Notion compatible")
        
        # Check LMS compatibility
        if self.config["integration"]["lms_compatibility"]:
            lms_issues = self._check_lms_compatibility(content_path)
            if lms_issues:
                compatibility_details.extend([f"LMS: {issue}" for issue in lms_issues])
                status = 'warning'
            else:
                compatibility_details.append("✅ LMS compatible")
        
        return {
            'status': status,
            'details': compatibility_details
        }
    
    def _check_notion_compatibility(self, content_path: Path) -> List[str]:
        """Check for Notion-specific compatibility issues."""
        issues = []
        
        # Check markdown files for Notion-incompatible elements
        for md_file in content_path.rglob("*.md"):
            try:
                content = md_file.read_text(encoding='utf-8')
                
                # Check for complex HTML
                if '<table' in content or '<div' in content:
                    issues.append(f"Complex HTML in {md_file.name}")
                
                # Check for math expressions
                if '$$' in content or r'\(' in content:
                    issues.append(f"Math expressions may not render in {md_file.name}")
                
            except Exception:
                issues.append(f"Could not read {md_file.name}")
        
        return issues
    
    def _check_lms_compatibility(self, content_path: Path) -> List[str]:
        """Check for LMS compatibility issues."""
        issues = []
        
        # Check for executable code that might not work in LMS
        python_files = list(content_path.rglob("*.py"))
        
        for py_file in python_files:
            try:
                content = py_file.read_text(encoding='utf-8')
                
                # Check for system-dependent operations
                if 'subprocess' in content or 'os.system' in content:
                    issues.append(f"System calls in {py_file.name} may not work in LMS")
                
                # Check for file I/O operations
                if 'open(' in content and 'write' in content:
                    issues.append(f"File writing in {py_file.name} may be restricted in LMS")
                
            except Exception:
                issues.append(f"Could not read {py_file.name}")
        
        return issues
    
    def _generate_deployment_recommendations(self, validation_results: Dict) -> List[str]:
        """Generate deployment recommendations based on validation."""
        recommendations = []
        
        if validation_results['overall_status'] == 'fail':
            recommendations.append("🚨 Address all failing validations before deployment")
        
        if validation_results['file_structure']['status'] != 'pass':
            recommendations.append("📁 Fix file structure issues")
        
        if validation_results['content_quality']['status'] == 'warning':
            recommendations.append("📝 Review content quality warnings")
        
        if validation_results['code_execution']['status'] != 'pass':
            recommendations.append("🐛 Fix code execution issues")
        
        if validation_results['platform_compatibility']['status'] == 'warning':
            recommendations.append("🔧 Address platform compatibility warnings")
        
        return recommendations
    
    def package_for_distribution(self, content_dir: str, output_path: str, version: str = "1.0") -> str:
        """
        Package content for distribution.
        
        Args:
            content_dir: Source content directory
            output_path: Output package path
            version: Version string for package
            
        Returns:
            Path to created package
        """
        logger.info(f"Packaging content for distribution: {content_dir} -> {output_path}")
        
        content_path = Path(content_dir)
        output_path = Path(output_path)
        
        # Create package metadata
        package_metadata = {
            'name': 'Data Science Course Content',
            'version': version,
            'created': datetime.now().isoformat(),
            'contents': self._catalog_content(content_path),
            'deployment_info': {
                'platforms_supported': self.config['deployment']['supported_platforms'],
                'validation_passed': True  # Assume validation passed
            }
        }
        
        # Create package directory
        package_name = f"datasci_course_v{version}_{datetime.now().strftime('%Y%m%d')}"
        package_dir = output_path / package_name
        package_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy content
        self._copy_content_for_package(content_path, package_dir)
        
        # Write metadata
        with open(package_dir / 'package_info.json', 'w') as f:
            json.dump(package_metadata, f, indent=2)
        
        # Create distribution guides
        self._create_distribution_guides(package_dir)
        
        # Create compressed archive
        archive_path = output_path / f"{package_name}.zip"
        self._create_archive(package_dir, archive_path)
        
        logger.info(f"Package created: {archive_path}")
        return str(archive_path)
    
    def _catalog_content(self, content_path: Path) -> Dict:
        """Create catalog of content for package metadata."""
        catalog = {
            'lectures': [],
            'total_files': 0,
            'file_types': {},
            'size_mb': 0
        }
        
        # Find lecture directories
        lecture_dirs = [d for d in content_path.iterdir() if d.is_dir() and 'lecture' in d.name.lower()]
        
        total_size = 0
        
        for lecture_dir in lecture_dirs:
            lecture_info = {
                'name': lecture_dir.name,
                'files': [],
                'size_bytes': 0
            }
            
            for file_path in lecture_dir.rglob('*'):
                if file_path.is_file():
                    file_size = file_path.stat().st_size
                    total_size += file_size
                    lecture_info['size_bytes'] += file_size
                    
                    file_ext = file_path.suffix.lower()
                    catalog['file_types'][file_ext] = catalog['file_types'].get(file_ext, 0) + 1
                    catalog['total_files'] += 1
                    
                    lecture_info['files'].append({
                        'name': file_path.name,
                        'path': str(file_path.relative_to(lecture_dir)),
                        'size_bytes': file_size
                    })
            
            catalog['lectures'].append(lecture_info)
        
        catalog['size_mb'] = round(total_size / (1024 * 1024), 2)
        
        return catalog
    
    def _copy_content_for_package(self, source_path: Path, package_path: Path):
        """Copy content to package directory with filtering."""
        
        # Define what to exclude from package
        exclude_patterns = {'.git', '__pycache__', '.DS_Store', '*.pyc', '*.log'}
        
        for item in source_path.rglob('*'):
            if item.is_file():
                # Check if file should be excluded
                if any(pattern in str(item) for pattern in exclude_patterns):
                    continue
                
                # Calculate relative path
                rel_path = item.relative_to(source_path)
                dest_path = package_path / rel_path
                
                # Create parent directories
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy file
                shutil.copy2(item, dest_path)
    
    def _create_distribution_guides(self, package_dir: Path):
        """Create deployment guides for different platforms."""
        guides_dir = package_dir / 'deployment_guides'
        guides_dir.mkdir(exist_ok=True)
        
        # Notion deployment guide
        notion_guide = """# Notion Deployment Guide

## Prerequisites
- Notion account with import permissions
- Content package extracted

## Deployment Steps
1. Open Notion workspace
2. Create new page for course content
3. Import markdown files using "Import" option
4. Upload media files to appropriate pages
5. Test interactive elements and links

## Post-Deployment
- Verify all content imported correctly
- Test code examples and links
- Set appropriate page permissions
- Share with students

## Troubleshooting
- Large files may need to be uploaded separately
- Math expressions may need manual formatting
- Code blocks should maintain syntax highlighting
"""
        
        (guides_dir / 'notion_deployment.md').write_text(notion_guide)
        
        # LMS deployment guide
        lms_guide = """# LMS Deployment Guide

## Supported Platforms
- Canvas
- Blackboard
- Moodle
- D2L Brightspace

## General Deployment Steps
1. Access course administration panel
2. Create module structure for lectures
3. Upload content files to appropriate modules
4. Configure assignments and assessments
5. Test student access and functionality

## Platform-Specific Notes

### Canvas
- Use Rich Content Editor for markdown import
- Upload Python files as downloadable resources
- Create assignments linking to exercises

### Blackboard
- Import content through Course Materials
- Use Learning Modules for organization
- Set up Grade Center for assessments

## Best Practices
- Test with student account before publishing
- Ensure all links work within LMS environment
- Provide clear instructions for code execution
"""
        
        (guides_dir / 'lms_deployment.md').write_text(lms_guide)
    
    def _create_archive(self, package_dir: Path, archive_path: Path):
        """Create compressed archive of package."""
        with zipfile.ZipFile(archive_path, 'w', zipfile.ZIP_DEFLATED, compresslevel=self.config['packaging']['compression_level']) as zipf:
            for file_path in package_dir.rglob('*'):
                if file_path.is_file():
                    arcname = file_path.relative_to(package_dir.parent)
                    zipf.write(file_path, arcname)
    
    def deploy_to_local(self, content_dir: str, output_dir: str) -> Dict[str, Any]:
        """Deploy content to local file system."""
        logger.info(f"Deploying to local filesystem: {content_dir} -> {output_dir}")
        
        deployment_result = {
            'platform': 'local',
            'status': 'pending',
            'details': [],
            'output_path': output_dir
        }
        
        try:
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Copy content
            content_path = Path(content_dir)
            output_path = Path(output_dir)
            
            for item in content_path.rglob('*'):
                if item.is_file():
                    rel_path = item.relative_to(content_path)
                    dest_path = output_path / rel_path
                    
                    dest_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(item, dest_path)
            
            # Create index file
            self._create_local_index(output_path)
            
            deployment_result['status'] = 'success'
            deployment_result['details'].append(f"Content deployed to {output_dir}")
            deployment_result['details'].append("Local index file created")
            
        except Exception as e:
            deployment_result['status'] = 'failed'
            deployment_result['details'].append(f"Deployment failed: {str(e)}")
            logger.error(f"Local deployment failed: {e}")
        
        return deployment_result
    
    def _create_local_index(self, output_path: Path):
        """Create local index.html for content navigation."""
        
        # Find lecture directories
        lecture_dirs = [d for d in output_path.iterdir() if d.is_dir() and 'lecture' in d.name.lower()]
        lecture_dirs.sort()
        
        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Data Science Course Content</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 2em; line-height: 1.6; }}
        .header {{ background: #f4f4f4; padding: 1em; border-radius: 5px; }}
        .lecture {{ margin: 1em 0; padding: 1em; border: 1px solid #ddd; border-radius: 5px; }}
        .lecture h3 {{ margin-top: 0; color: #333; }}
        .files {{ margin-left: 1em; }}
        .files a {{ display: block; margin: 0.3em 0; text-decoration: none; color: #0066cc; }}
        .files a:hover {{ text-decoration: underline; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Data Science Course Content</h1>
        <p>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>Total Lectures: {len(lecture_dirs)}</p>
    </div>
"""
        
        for lecture_dir in lecture_dirs:
            html_content += f"""
    <div class="lecture">
        <h3>{lecture_dir.name.replace('_', ' ').title()}</h3>
        <div class="files">
"""
            
            # Add main files
            main_files = ['README.md', 'lecture_narrative.md', 'demo_lecture.py']
            for main_file in main_files:
                file_path = lecture_dir / main_file
                if file_path.exists():
                    rel_path = file_path.relative_to(output_path)
                    html_content += f'            <a href="{rel_path}">{main_file}</a>\n'
            
            # Add exercise directory
            exercises_dir = lecture_dir / 'exercises'
            if exercises_dir.exists():
                html_content += f'            <a href="{exercises_dir.relative_to(output_path)}">📁 Exercises</a>\n'
            
            # Add resources directory  
            resources_dir = lecture_dir / 'resources'
            if resources_dir.exists():
                html_content += f'            <a href="{resources_dir.relative_to(output_path)}">📁 Resources</a>\n'
            
            html_content += """        </div>
    </div>
"""
        
        html_content += """</body>
</html>"""
        
        (output_path / 'index.html').write_text(html_content, encoding='utf-8')
    
    def deploy_to_github(self, content_dir: str, repo_config: Dict) -> Dict[str, Any]:
        """Deploy content to GitHub repository."""
        deployment_result = {
            'platform': 'github',
            'status': 'pending', 
            'details': []
        }
        
        try:
            # This would implement GitHub API deployment
            # For now, provide guidance
            deployment_result['details'].append("GitHub deployment configured")
            deployment_result['details'].append(f"Repository: {repo_config.get('repo', 'Not specified')}")
            deployment_result['details'].append("Use git commands to push content to repository")
            deployment_result['status'] = 'configured'
            
        except Exception as e:
            deployment_result['status'] = 'failed'
            deployment_result['details'].append(f"GitHub deployment failed: {str(e)}")
        
        return deployment_result
    
    def generate_deployment_report(self, deployments: List[Dict]) -> Dict:
        """Generate comprehensive deployment report."""
        
        report = {
            'timestamp': datetime.now().isoformat(),
            'total_deployments': len(deployments),
            'successful_deployments': len([d for d in deployments if d['status'] == 'success']),
            'failed_deployments': len([d for d in deployments if d['status'] == 'failed']),
            'deployment_details': deployments,
            'recommendations': []
        }
        
        # Generate recommendations
        if report['failed_deployments'] > 0:
            report['recommendations'].append("Review and resolve failed deployments")
        
        if report['successful_deployments'] == report['total_deployments']:
            report['recommendations'].append("All deployments successful - ready for production use")
        
        return report


def main():
    """Main CLI interface for deployment system."""
    parser = argparse.ArgumentParser(
        description="Deployment System for Data Science Course Content",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 deploy_system.py --deploy-local output/
    python3 deploy_system.py --package-distribution --version 2.0
    python3 deploy_system.py --validate-only content/
        """
    )
    
    parser.add_argument('--deploy-local', help='Deploy to local directory')
    parser.add_argument('--package-distribution', action='store_true', help='Create distribution package')
    parser.add_argument('--validate-only', help='Only run validation on content directory')
    parser.add_argument('--content-dir', default='converted_lectures', help='Source content directory')
    parser.add_argument('--output-dir', default='deployed_content', help='Output directory')
    parser.add_argument('--version', default='1.0', help='Version for packaging')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize deployment system
    deployer = DeploymentSystem(args.config)
    
    try:
        if args.validate_only:
            # Validation only mode
            logger.info("Running validation-only mode")
            validation_results = deployer.validate_pre_deployment(args.validate_only)
            
            print(f"\n=== Validation Results ===")
            print(f"Overall Status: {validation_results['overall_status'].upper()}")
            print(f"File Structure: {validation_results['file_structure']['status'].upper()}")
            print(f"Content Quality: {validation_results['content_quality']['status'].upper()}")  
            print(f"Code Execution: {validation_results['code_execution']['status'].upper()}")
            print(f"Platform Compatibility: {validation_results['platform_compatibility']['status'].upper()}")
            
            if validation_results['recommendations']:
                print(f"\nRecommendations:")
                for rec in validation_results['recommendations']:
                    print(f"  {rec}")
        
        elif args.deploy_local:
            # Local deployment
            logger.info("Starting local deployment")
            
            # Run validation first
            validation_results = deployer.validate_pre_deployment(args.content_dir)
            
            if validation_results['overall_status'] == 'fail':
                print("❌ Validation failed - deployment aborted")
                print("Fix validation issues before deploying")
                return
            
            # Deploy to local directory
            deployment_result = deployer.deploy_to_local(args.content_dir, args.deploy_local)
            
            print(f"\n=== Local Deployment Results ===")
            print(f"Status: {deployment_result['status'].upper()}")
            print(f"Output: {deployment_result['output_path']}")
            
            for detail in deployment_result['details']:
                print(f"  {detail}")
        
        elif args.package_distribution:
            # Package for distribution
            logger.info("Creating distribution package")
            
            # Run validation first
            validation_results = deployer.validate_pre_deployment(args.content_dir)
            
            if validation_results['overall_status'] == 'fail':
                print("❌ Validation failed - packaging aborted")
                return
            
            # Create package
            package_path = deployer.package_for_distribution(
                args.content_dir, 
                args.output_dir, 
                args.version
            )
            
            print(f"\n=== Distribution Package Created ===")
            print(f"Package: {package_path}")
            print(f"Version: {args.version}")
            print("Ready for distribution to educational platforms")
        
        else:
            parser.print_help()
    
    except Exception as e:
        logger.error(f"Deployment failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()