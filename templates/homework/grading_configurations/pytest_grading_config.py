"""
Pytest Grading Configuration
============================

Configuration file for automated grading with pytest.
Defines point distributions, test categories, and grading rules.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import json
import os
from pathlib import Path


@dataclass
class GradingConfig:
    """Configuration for automated grading system"""
    
    # Point distribution by category
    points: Dict[str, int] = None
    
    # Test execution timeouts (seconds)
    timeouts: Dict[str, int] = None
    
    # Required files for submission
    required_files: List[str] = None
    
    # Optional files
    optional_files: List[str] = None
    
    # Grade thresholds
    thresholds: Dict[str, float] = None
    
    # Test markers and their weights
    test_markers: Dict[str, Dict[str, Any]] = None
    
    def __post_init__(self):
        """Initialize default values if not provided"""
        if self.points is None:
            self.points = self.default_points()
        
        if self.timeouts is None:
            self.timeouts = self.default_timeouts()
            
        if self.required_files is None:
            self.required_files = self.default_required_files()
            
        if self.optional_files is None:
            self.optional_files = self.default_optional_files()
            
        if self.thresholds is None:
            self.thresholds = self.default_thresholds()
            
        if self.test_markers is None:
            self.test_markers = self.default_test_markers()
    
    @staticmethod
    def default_points() -> Dict[str, int]:
        """Default point distribution"""
        return {
            'function_tests': 40,
            'edge_cases': 20, 
            'error_handling': 15,
            'code_quality': 15,
            'documentation': 10
        }
    
    @staticmethod
    def default_timeouts() -> Dict[str, int]:
        """Default timeout settings (seconds)"""
        return {
            'function_call': 5,
            'file_operation': 10,
            'subprocess': 30,
            'integration_test': 60,
            'performance_test': 120
        }
    
    @staticmethod
    def default_required_files() -> List[str]:
        """Default required files"""
        return ['main.py']
    
    @staticmethod
    def default_optional_files() -> List[str]:
        """Default optional files"""
        return ['README.md', 'requirements.txt', 'config.json']
    
    @staticmethod
    def default_thresholds() -> Dict[str, float]:
        """Default grade thresholds"""
        return {
            'excellent': 0.90,
            'good': 0.80,
            'satisfactory': 0.70,
            'needs_improvement': 0.60,
            'failing': 0.00
        }
    
    @staticmethod
    def default_test_markers() -> Dict[str, Dict[str, Any]]:
        """Default test markers configuration"""
        return {
            'function_test': {
                'weight': 0.40,
                'description': 'Basic function implementation tests',
                'timeout': 5
            },
            'edge_case': {
                'weight': 0.20,
                'description': 'Edge cases and boundary conditions',
                'timeout': 5
            },
            'error_handling': {
                'weight': 0.15,
                'description': 'Exception handling and validation',
                'timeout': 5
            },
            'integration': {
                'weight': 0.15,
                'description': 'Integration between components',
                'timeout': 30
            },
            'performance': {
                'weight': 0.10,
                'description': 'Performance and efficiency tests',
                'timeout': 60
            }
        }
    
    def get_total_points(self) -> int:
        """Get total possible points"""
        return sum(self.points.values())
    
    def get_grade_letter(self, score: float) -> str:
        """Convert numeric score to letter grade"""
        percentage = score / self.get_total_points()
        
        if percentage >= self.thresholds['excellent']:
            return 'A'
        elif percentage >= self.thresholds['good']:
            return 'B'
        elif percentage >= self.thresholds['satisfactory']:
            return 'C'
        elif percentage >= self.thresholds['needs_improvement']:
            return 'D'
        else:
            return 'F'
    
    def get_grade_description(self, score: float) -> str:
        """Get descriptive text for grade"""
        percentage = score / self.get_total_points()
        
        if percentage >= self.thresholds['excellent']:
            return "🏆 Outstanding work! Exceeds expectations."
        elif percentage >= self.thresholds['good']:
            return "🌟 Excellent work! Meets all requirements."
        elif percentage >= self.thresholds['satisfactory']:
            return "✅ Good work! Meets most requirements."
        elif percentage >= self.thresholds['needs_improvement']:
            return "⚠️ Needs improvement. Review feedback carefully."
        else:
            return "❌ Significant issues found. Please review and resubmit."
    
    @classmethod
    def load_from_file(cls, config_path: str) -> 'GradingConfig':
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            return cls()  # Return default config
        
        try:
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            return cls(
                points=config_data.get('points'),
                timeouts=config_data.get('timeouts'),
                required_files=config_data.get('required_files'),
                optional_files=config_data.get('optional_files'),
                thresholds=config_data.get('thresholds'),
                test_markers=config_data.get('test_markers')
            )
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: Could not load config from {config_path}: {e}")
            return cls()  # Return default config
    
    def save_to_file(self, config_path: str) -> None:
        """Save configuration to JSON file"""
        config_data = {
            'points': self.points,
            'timeouts': self.timeouts,
            'required_files': self.required_files,
            'optional_files': self.optional_files,
            'thresholds': self.thresholds,
            'test_markers': self.test_markers
        }
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)


# Assignment-specific configurations
class AssignmentConfigs:
    """Pre-defined configurations for common assignment types"""
    
    @staticmethod
    def function_based() -> GradingConfig:
        """Configuration for function-based assignments"""
        return GradingConfig(
            points={
                'function_tests': 50,
                'edge_cases': 20,
                'error_handling': 20,
                'code_quality': 10
            },
            test_markers={
                'function_test': {'weight': 0.50, 'timeout': 5},
                'edge_case': {'weight': 0.20, 'timeout': 5},
                'error_handling': {'weight': 0.20, 'timeout': 5},
                'integration': {'weight': 0.10, 'timeout': 10}
            }
        )
    
    @staticmethod
    def data_processing() -> GradingConfig:
        """Configuration for data processing assignments"""
        return GradingConfig(
            points={
                'data_loading': 15,
                'data_cleaning': 25,
                'data_analysis': 25,
                'output_format': 15,
                'performance': 10,
                'documentation': 10
            },
            timeouts={
                'function_call': 10,
                'file_operation': 30,
                'data_processing': 120
            },
            required_files=['main.py', 'data_processor.py'],
            test_markers={
                'function_test': {'weight': 0.40, 'timeout': 10},
                'data_test': {'weight': 0.30, 'timeout': 30},
                'integration': {'weight': 0.20, 'timeout': 60},
                'performance': {'weight': 0.10, 'timeout': 120}
            }
        )
    
    @staticmethod
    def cli_application() -> GradingConfig:
        """Configuration for command-line interface assignments"""
        return GradingConfig(
            points={
                'argument_parsing': 20,
                'core_functionality': 30,
                'output_format': 15,
                'error_handling': 20,
                'help_system': 10,
                'documentation': 5
            },
            timeouts={
                'cli_execution': 30,
                'help_command': 5,
                'file_processing': 60
            },
            test_markers={
                'cli_test': {'weight': 0.40, 'timeout': 30},
                'function_test': {'weight': 0.30, 'timeout': 10},
                'error_handling': {'weight': 0.20, 'timeout': 10},
                'integration': {'weight': 0.10, 'timeout': 60}
            }
        )
    
    @staticmethod
    def file_io() -> GradingConfig:
        """Configuration for file I/O assignments"""
        return GradingConfig(
            points={
                'file_reading': 20,
                'file_writing': 20,
                'data_validation': 15,
                'format_handling': 15,
                'error_handling': 20,
                'performance': 10
            },
            timeouts={
                'file_operation': 30,
                'large_file_test': 120
            },
            test_markers={
                'file_io_test': {'weight': 0.50, 'timeout': 30},
                'validation_test': {'weight': 0.20, 'timeout': 10},
                'error_handling': {'weight': 0.20, 'timeout': 10},
                'performance': {'weight': 0.10, 'timeout': 120}
            }
        )


# Global configuration instance
DEFAULT_CONFIG = GradingConfig()

# Configuration loader
def get_config(assignment_type: Optional[str] = None, 
               config_file: Optional[str] = None) -> GradingConfig:
    """
    Get appropriate configuration for grading.
    
    Args:
        assignment_type: Type of assignment ('function', 'data', 'cli', 'file_io')
        config_file: Path to custom configuration file
    
    Returns:
        GradingConfig instance
    """
    if config_file and os.path.exists(config_file):
        return GradingConfig.load_from_file(config_file)
    
    if assignment_type == 'function':
        return AssignmentConfigs.function_based()
    elif assignment_type == 'data':
        return AssignmentConfigs.data_processing()
    elif assignment_type == 'cli':
        return AssignmentConfigs.cli_application()
    elif assignment_type == 'file_io':
        return AssignmentConfigs.file_io()
    else:
        # Look for config file in current directory
        local_config = Path('grading_config.json')
        if local_config.exists():
            return GradingConfig.load_from_file(str(local_config))
        
        return DEFAULT_CONFIG


# Example usage for instructors
if __name__ == "__main__":
    # Create example configuration files
    configs = {
        'function_assignment_config.json': AssignmentConfigs.function_based(),
        'data_processing_config.json': AssignmentConfigs.data_processing(),
        'cli_application_config.json': AssignmentConfigs.cli_application(),
        'file_io_config.json': AssignmentConfigs.file_io()
    }
    
    for filename, config in configs.items():
        config.save_to_file(filename)
        print(f"Created {filename}")
    
    # Demo configuration loading
    print("\nDemo: Loading and using configuration")
    config = get_config('function')
    print(f"Total points: {config.get_total_points()}")
    print(f"Grade for 85/100: {config.get_grade_letter(85)} - {config.get_grade_description(85)}")