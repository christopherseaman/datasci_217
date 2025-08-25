#!/usr/bin/env python3
"""
Assessment Alignment Automation Tool
====================================

Automated system for aligning lecture content with learning objectives and
assessment requirements. Ensures educational coherence and measurable outcomes.

Usage:
    python3 assessment_aligner.py --lecture-dir lectures/lecture_01/
    python3 assessment_aligner.py --batch-align lectures/
    python3 assessment_aligner.py --generate-rubrics lectures/
"""

import os
import sys
import json
import re
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, asdict
from collections import defaultdict

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LearningObjective:
    """Data class for learning objectives."""
    id: str
    description: str
    bloom_level: str  # remember, understand, apply, analyze, evaluate, create
    measurable_verb: str
    content_area: str
    assessment_method: str


@dataclass
class AssessmentAlignment:
    """Data class for assessment alignment results."""
    objective_id: str
    content_coverage: float  # 0-1 score
    exercise_alignment: List[str]
    formative_checkpoints: List[str]
    summative_connections: List[str]
    alignment_strength: str  # weak, moderate, strong
    recommendations: List[str]


@dataclass
class AlignmentReport:
    """Complete alignment assessment report."""
    lecture_name: str
    generated_at: str
    overall_alignment_score: float
    learning_objectives: List[LearningObjective]
    alignment_results: List[AssessmentAlignment]
    gaps_identified: List[str]
    recommendations: List[str]
    rubric_suggestions: Dict[str, Any]


class AssessmentAligner:
    """
    Comprehensive assessment alignment system for educational content.
    
    Analyzes content against learning objectives, identifies alignment gaps,
    and generates assessment recommendations and rubrics.
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize aligner with configuration."""
        self.config = self._load_config(config_path)
        self.bloom_taxonomy = self._initialize_bloom_taxonomy()
        self.measurable_verbs = self._initialize_measurable_verbs()
        
    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load assessment alignment configuration."""
        default_config = {
            "learning_objectives": {
                "minimum_per_lecture": 4,
                "maximum_per_lecture": 8,
                "bloom_level_distribution": {
                    "remember": 0.1,
                    "understand": 0.2,
                    "apply": 0.3,
                    "analyze": 0.2,
                    "evaluate": 0.1,
                    "create": 0.1
                }
            },
            "content_analysis": {
                "key_concept_indicators": [
                    "fundamental", "essential", "important", "crucial",
                    "students will", "learning", "understanding"
                ],
                "exercise_indicators": [
                    "practice", "exercise", "challenge", "problem",
                    "hands-on", "activity", "assignment"
                ],
                "assessment_indicators": [
                    "quiz", "test", "exam", "assignment", "project",
                    "evaluation", "assessment", "rubric"
                ]
            },
            "alignment_criteria": {
                "strong_threshold": 0.8,
                "moderate_threshold": 0.6,
                "weak_threshold": 0.4,
                "content_coverage_weight": 0.4,
                "exercise_alignment_weight": 0.3,
                "assessment_integration_weight": 0.3
            },
            "rubric_generation": {
                "performance_levels": ["Exemplary", "Proficient", "Developing", "Beginning"],
                "criteria_categories": [
                    "Technical Accuracy", "Problem Solving", "Communication",
                    "Integration", "Professional Practice"
                ]
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path, 'r') as f:
                user_config = json.load(f)
                default_config.update(user_config)
        
        return default_config
    
    def _initialize_bloom_taxonomy(self) -> Dict[str, Dict]:
        """Initialize Bloom's Taxonomy framework."""
        return {
            "remember": {
                "description": "Recall facts and basic concepts",
                "verbs": ["define", "list", "recall", "identify", "name", "state"],
                "assessment_methods": ["multiple choice", "true/false", "matching", "fill-in-blank"]
            },
            "understand": {
                "description": "Explain ideas or concepts",
                "verbs": ["explain", "describe", "summarize", "interpret", "classify"],
                "assessment_methods": ["short answer", "explanation", "categorization"]
            },
            "apply": {
                "description": "Use information in new situations",
                "verbs": ["use", "execute", "implement", "demonstrate", "solve"],
                "assessment_methods": ["problem solving", "case studies", "simulations"]
            },
            "analyze": {
                "description": "Draw connections among ideas",
                "verbs": ["analyze", "compare", "contrast", "examine", "categorize"],
                "assessment_methods": ["analysis papers", "case studies", "critiques"]
            },
            "evaluate": {
                "description": "Justify a decision or course of action",
                "verbs": ["evaluate", "critique", "judge", "defend", "support"],
                "assessment_methods": ["essays", "debates", "peer reviews"]
            },
            "create": {
                "description": "Produce new or original work",
                "verbs": ["create", "design", "develop", "construct", "produce"],
                "assessment_methods": ["projects", "portfolios", "presentations"]
            }
        }
    
    def _initialize_measurable_verbs(self) -> Set[str]:
        """Initialize set of measurable action verbs."""
        all_verbs = set()
        for level_info in self.bloom_taxonomy.values():
            all_verbs.update(level_info["verbs"])
        return all_verbs
    
    def extract_learning_objectives(self, content: str) -> List[LearningObjective]:
        """
        Extract and analyze learning objectives from lecture content.
        
        Args:
            content: Raw lecture content text
            
        Returns:
            List of parsed learning objectives
        """
        objectives = []
        
        # Find learning objectives section
        obj_section = self._extract_section(content, "learning objectives")
        if not obj_section:
            logger.warning("No learning objectives section found")
            return objectives
        
        # Extract individual objectives
        objective_lines = re.findall(r'[-•]\s+(.+)', obj_section, re.MULTILINE)
        
        for i, obj_text in enumerate(objective_lines):
            # Parse objective components
            measurable_verb = self._extract_measurable_verb(obj_text)
            bloom_level = self._classify_bloom_level(measurable_verb)
            content_area = self._extract_content_area(obj_text)
            assessment_method = self._suggest_assessment_method(bloom_level)
            
            objective = LearningObjective(
                id=f"LO_{i+1:02d}",
                description=obj_text.strip(),
                bloom_level=bloom_level,
                measurable_verb=measurable_verb,
                content_area=content_area,
                assessment_method=assessment_method
            )
            
            objectives.append(objective)
        
        logger.info(f"Extracted {len(objectives)} learning objectives")
        return objectives
    
    def _extract_measurable_verb(self, objective_text: str) -> str:
        """Extract measurable action verb from objective."""
        words = objective_text.lower().split()
        
        for word in words:
            # Clean word of punctuation
            clean_word = re.sub(r'[^\w]', '', word)
            if clean_word in self.measurable_verbs:
                return clean_word
        
        return "understand"  # Default fallback
    
    def _classify_bloom_level(self, verb: str) -> str:
        """Classify Bloom's taxonomy level based on verb."""
        for level, info in self.bloom_taxonomy.items():
            if verb in info["verbs"]:
                return level
        
        return "understand"  # Default fallback
    
    def _extract_content_area(self, objective_text: str) -> str:
        """Extract main content area from objective description."""
        # Look for key content area indicators
        content_indicators = {
            "python": ["python", "programming", "code", "script"],
            "data_structures": ["list", "dictionary", "array", "dataframe"],
            "statistics": ["statistical", "analysis", "mean", "correlation"],
            "visualization": ["plot", "chart", "graph", "visual"],
            "machine_learning": ["model", "prediction", "classification"],
            "git": ["git", "version", "repository", "commit"],
            "numpy": ["numpy", "array", "vectorized"],
            "pandas": ["pandas", "dataframe", "csv"]
        }
        
        text_lower = objective_text.lower()
        
        for area, keywords in content_indicators.items():
            if any(keyword in text_lower for keyword in keywords):
                return area
        
        return "general"
    
    def _suggest_assessment_method(self, bloom_level: str) -> str:
        """Suggest appropriate assessment method for Bloom level."""
        methods = self.bloom_taxonomy[bloom_level]["assessment_methods"]
        return methods[0] if methods else "demonstration"
    
    def analyze_content_coverage(self, content: str, objectives: List[LearningObjective]) -> Dict[str, float]:
        """
        Analyze how well content covers each learning objective.
        
        Args:
            content: Full lecture content
            objectives: List of learning objectives
            
        Returns:
            Dictionary mapping objective IDs to coverage scores
        """
        coverage_scores = {}
        
        for objective in objectives:
            score = self._calculate_coverage_score(content, objective)
            coverage_scores[objective.id] = score
        
        return coverage_scores
    
    def _calculate_coverage_score(self, content: str, objective: LearningObjective) -> float:
        """Calculate coverage score for a single objective."""
        content_lower = content.lower()
        
        # Extract key terms from objective
        obj_terms = self._extract_key_terms(objective.description)
        
        # Count matches in content
        matches = 0
        for term in obj_terms:
            if term.lower() in content_lower:
                matches += 1
        
        # Base score from term coverage
        base_score = matches / len(obj_terms) if obj_terms else 0
        
        # Boost score based on content area coverage
        area_boost = self._calculate_area_coverage(content, objective.content_area)
        
        # Bloom level complexity adjustment
        bloom_weight = self._get_bloom_weight(objective.bloom_level)
        
        # Combined score
        final_score = min(1.0, (base_score * 0.6) + (area_boost * 0.3) + (bloom_weight * 0.1))
        
        return final_score
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """Extract key terms from objective description."""
        # Remove common stop words and extract meaningful terms
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over'
        }
        
        words = re.findall(r'\w+', text.lower())
        key_terms = [word for word in words if word not in stop_words and len(word) > 2]
        
        return key_terms
    
    def _calculate_area_coverage(self, content: str, content_area: str) -> float:
        """Calculate how well content covers a specific area."""
        area_keywords = {
            "python": ["python", "variable", "function", "loop", "condition"],
            "data_structures": ["list", "dictionary", "array", "dataframe", "series"],
            "statistics": ["mean", "median", "standard", "correlation", "analysis"],
            "git": ["git", "commit", "branch", "merge", "repository"],
            "numpy": ["numpy", "array", "vectorized", "mathematical"],
            "pandas": ["pandas", "dataframe", "csv", "data", "filtering"]
        }
        
        keywords = area_keywords.get(content_area, [])
        if not keywords:
            return 0.5  # Neutral score for unknown areas
        
        content_lower = content.lower()
        matches = sum(1 for keyword in keywords if keyword in content_lower)
        
        return min(1.0, matches / len(keywords))
    
    def _get_bloom_weight(self, bloom_level: str) -> float:
        """Get complexity weight for Bloom level."""
        weights = {
            "remember": 0.2,
            "understand": 0.3,
            "apply": 0.5,
            "analyze": 0.7,
            "evaluate": 0.8,
            "create": 1.0
        }
        
        return weights.get(bloom_level, 0.5)
    
    def analyze_exercise_alignment(self, content: str, objectives: List[LearningObjective]) -> Dict[str, List[str]]:
        """
        Analyze alignment between exercises and learning objectives.
        
        Args:
            content: Full lecture content
            objectives: List of learning objectives
            
        Returns:
            Dictionary mapping objective IDs to aligned exercises
        """
        # Extract exercise sections from content
        exercises = self._extract_exercises(content)
        
        alignment_map = {}
        
        for objective in objectives:
            aligned_exercises = []
            
            for exercise_name, exercise_content in exercises.items():
                alignment_score = self._calculate_exercise_alignment(exercise_content, objective)
                
                if alignment_score > self.config["alignment_criteria"]["moderate_threshold"]:
                    aligned_exercises.append(exercise_name)
            
            alignment_map[objective.id] = aligned_exercises
        
        return alignment_map
    
    def _extract_exercises(self, content: str) -> Dict[str, str]:
        """Extract exercise sections from content."""
        exercises = {}
        
        # Find exercise sections
        exercise_pattern = r'###?\s*(Exercise\s+\d+[^#]*?)(?=###|\Z)'
        matches = re.findall(exercise_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for i, exercise_content in enumerate(matches, 1):
            exercise_name = f"Exercise_{i}"
            exercises[exercise_name] = exercise_content
        
        # Also look for hands-on practice sections
        practice_pattern = r'###?\s*(.*?practice.*?)(?=###|\Z)'
        practice_matches = re.findall(practice_pattern, content, re.DOTALL | re.IGNORECASE)
        
        for i, practice_content in enumerate(practice_matches):
            practice_name = f"Practice_{i+1}"
            exercises[practice_name] = practice_content
        
        return exercises
    
    def _calculate_exercise_alignment(self, exercise_content: str, objective: LearningObjective) -> float:
        """Calculate alignment score between exercise and objective."""
        # Check for verb alignment
        verb_match = objective.measurable_verb.lower() in exercise_content.lower()
        verb_score = 0.3 if verb_match else 0
        
        # Check for content area alignment
        area_score = self._calculate_area_coverage(exercise_content, objective.content_area)
        
        # Check for key term overlap
        obj_terms = self._extract_key_terms(objective.description)
        exercise_terms = self._extract_key_terms(exercise_content)
        
        term_overlap = len(set(obj_terms) & set(exercise_terms))
        term_score = min(0.4, term_overlap / len(obj_terms) if obj_terms else 0)
        
        return verb_score + (area_score * 0.4) + term_score
    
    def generate_alignment_report(self, lecture_dir: str) -> AlignmentReport:
        """
        Generate comprehensive alignment report for a lecture.
        
        Args:
            lecture_dir: Path to lecture directory
            
        Returns:
            Complete alignment assessment report
        """
        logger.info(f"Generating alignment report for: {lecture_dir}")
        
        lecture_path = Path(lecture_dir)
        lecture_name = lecture_path.name
        
        # Find main content file
        content_file = self._find_content_file(lecture_path)
        if not content_file:
            raise FileNotFoundError("No main content file found")
        
        # Load content
        content = content_file.read_text(encoding='utf-8')
        
        # Extract learning objectives
        objectives = self.extract_learning_objectives(content)
        
        # Analyze content coverage
        coverage_scores = self.analyze_content_coverage(content, objectives)
        
        # Analyze exercise alignment
        exercise_alignment = self.analyze_exercise_alignment(content, objectives)
        
        # Generate alignment results
        alignment_results = []
        for objective in objectives:
            coverage = coverage_scores.get(objective.id, 0)
            exercises = exercise_alignment.get(objective.id, [])
            
            # Calculate alignment strength
            alignment_strength = self._classify_alignment_strength(coverage, len(exercises))
            
            # Generate recommendations
            recommendations = self._generate_objective_recommendations(objective, coverage, exercises)
            
            alignment_result = AssessmentAlignment(
                objective_id=objective.id,
                content_coverage=coverage,
                exercise_alignment=exercises,
                formative_checkpoints=[],  # TODO: Extract formative assessments
                summative_connections=[],   # TODO: Extract summative connections
                alignment_strength=alignment_strength,
                recommendations=recommendations
            )
            
            alignment_results.append(alignment_result)
        
        # Calculate overall alignment score
        overall_score = sum(coverage_scores.values()) / len(coverage_scores) if coverage_scores else 0
        
        # Identify gaps
        gaps = self._identify_alignment_gaps(objectives, alignment_results)
        
        # Generate overall recommendations
        overall_recommendations = self._generate_overall_recommendations(alignment_results, gaps)
        
        # Generate rubric suggestions
        rubric_suggestions = self._generate_rubric_suggestions(objectives)
        
        # Create report
        report = AlignmentReport(
            lecture_name=lecture_name,
            generated_at=datetime.now().isoformat(),
            overall_alignment_score=overall_score,
            learning_objectives=objectives,
            alignment_results=alignment_results,
            gaps_identified=gaps,
            recommendations=overall_recommendations,
            rubric_suggestions=rubric_suggestions
        )
        
        logger.info(f"Alignment report generated: overall score {overall_score:.2f}")
        return report
    
    def _find_content_file(self, lecture_path: Path) -> Optional[Path]:
        """Find main content file in lecture directory."""
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
        
        # Look for any markdown file
        md_files = list(lecture_path.glob("*.md"))
        if md_files:
            return max(md_files, key=lambda f: f.stat().st_size)
        
        return None
    
    def _classify_alignment_strength(self, coverage: float, exercise_count: int) -> str:
        """Classify alignment strength based on coverage and exercises."""
        strong_threshold = self.config["alignment_criteria"]["strong_threshold"]
        moderate_threshold = self.config["alignment_criteria"]["moderate_threshold"]
        
        if coverage >= strong_threshold and exercise_count >= 1:
            return "strong"
        elif coverage >= moderate_threshold or exercise_count >= 1:
            return "moderate"
        else:
            return "weak"
    
    def _generate_objective_recommendations(self, objective: LearningObjective, 
                                          coverage: float, exercises: List[str]) -> List[str]:
        """Generate specific recommendations for an objective."""
        recommendations = []
        
        if coverage < 0.6:
            recommendations.append(f"Increase content coverage for '{objective.content_area}' concepts")
        
        if not exercises:
            recommendations.append(f"Add practice exercises for '{objective.measurable_verb}' activities")
        
        if objective.bloom_level in ["apply", "analyze", "evaluate", "create"] and len(exercises) < 2:
            recommendations.append(f"Add more complex exercises for {objective.bloom_level}-level learning")
        
        return recommendations
    
    def _identify_alignment_gaps(self, objectives: List[LearningObjective], 
                                results: List[AssessmentAlignment]) -> List[str]:
        """Identify major alignment gaps."""
        gaps = []
        
        # Check Bloom level distribution
        bloom_counts = defaultdict(int)
        for obj in objectives:
            bloom_counts[obj.bloom_level] += 1
        
        total_objectives = len(objectives)
        target_distribution = self.config["learning_objectives"]["bloom_level_distribution"]
        
        for level, target_ratio in target_distribution.items():
            actual_ratio = bloom_counts[level] / total_objectives
            if actual_ratio < target_ratio * 0.5:  # Less than half the target
                gaps.append(f"Under-representation of {level}-level objectives")
        
        # Check for objectives with weak alignment
        weak_alignments = [r for r in results if r.alignment_strength == "weak"]
        if len(weak_alignments) > len(results) * 0.3:  # More than 30% weak
            gaps.append("High proportion of weakly aligned objectives")
        
        # Check for missing assessment methods
        assessment_methods = set(obj.assessment_method for obj in objectives)
        if len(assessment_methods) < 3:
            gaps.append("Limited variety in suggested assessment methods")
        
        return gaps
    
    def _generate_overall_recommendations(self, results: List[AssessmentAlignment], 
                                        gaps: List[str]) -> List[str]:
        """Generate overall improvement recommendations."""
        recommendations = []
        
        # Address identified gaps
        for gap in gaps:
            if "Under-representation" in gap:
                recommendations.append(f"Balance learning objectives: {gap}")
            elif "weakly aligned" in gap:
                recommendations.append("Strengthen content-objective alignment through targeted examples")
            elif "assessment methods" in gap:
                recommendations.append("Diversify assessment approaches to match Bloom levels")
        
        # General improvements
        weak_results = [r for r in results if r.alignment_strength == "weak"]
        if weak_results:
            recommendations.append("Focus improvement efforts on weakly aligned objectives")
        
        strong_results = [r for r in results if r.alignment_strength == "strong"]
        if len(strong_results) > len(results) * 0.7:  # More than 70% strong
            recommendations.append("Consider raising cognitive complexity for advanced learners")
        
        return recommendations
    
    def _generate_rubric_suggestions(self, objectives: List[LearningObjective]) -> Dict[str, Any]:
        """Generate assessment rubric suggestions."""
        rubric = {
            "rubric_type": "analytic",
            "performance_levels": self.config["rubric_generation"]["performance_levels"],
            "criteria": {}
        }
        
        # Group objectives by content area
        area_objectives = defaultdict(list)
        for obj in objectives:
            area_objectives[obj.content_area].append(obj)
        
        # Generate criteria for each area
        for area, objs in area_objectives.items():
            criteria_name = area.replace("_", " ").title()
            
            # Determine primary Bloom level for this area
            bloom_levels = [obj.bloom_level for obj in objs]
            primary_bloom = max(set(bloom_levels), key=bloom_levels.count)
            
            rubric["criteria"][criteria_name] = {
                "description": f"Demonstrates {primary_bloom}-level understanding of {criteria_name}",
                "bloom_level": primary_bloom,
                "objectives_covered": [obj.id for obj in objs],
                "assessment_method": objs[0].assessment_method if objs else "demonstration"
            }
        
        return rubric
    
    def _extract_section(self, content: str, section_name: str) -> Optional[str]:
        """Extract specific section content from markdown."""
        pattern = rf'#+\s*{re.escape(section_name)}.*?\n(.*?)(?=\n#+|\Z)'
        match = re.search(pattern, content, re.IGNORECASE | re.DOTALL)
        return match.group(1) if match else None
    
    def save_report(self, report: AlignmentReport, output_path: str) -> None:
        """Save alignment report to file."""
        with open(output_path, 'w') as f:
            json.dump(asdict(report), f, indent=2, default=str)
        
        logger.info(f"Alignment report saved to: {output_path}")
    
    def batch_align_lectures(self, lectures_dir: str) -> List[AlignmentReport]:
        """Perform alignment analysis on multiple lectures."""
        logger.info(f"Starting batch alignment analysis: {lectures_dir}")
        
        lectures_path = Path(lectures_dir)
        reports = []
        
        # Find lecture directories
        lecture_dirs = [d for d in lectures_path.iterdir() 
                       if d.is_dir() and d.name.startswith('lecture')]
        
        for lecture_dir in lecture_dirs:
            try:
                logger.info(f"Analyzing alignment: {lecture_dir.name}")
                report = self.generate_alignment_report(str(lecture_dir))
                reports.append(report)
                
                # Save individual report
                report_path = lecture_dir / "alignment_report.json"
                self.save_report(report, str(report_path))
                
            except Exception as e:
                logger.error(f"Failed to analyze {lecture_dir.name}: {e}")
        
        # Generate batch summary
        self._generate_batch_alignment_summary(reports, lectures_dir)
        
        return reports
    
    def _generate_batch_alignment_summary(self, reports: List[AlignmentReport], 
                                        output_dir: str) -> None:
        """Generate summary report for batch alignment analysis."""
        summary_path = Path(output_dir) / "ALIGNMENT_SUMMARY.md"
        
        # Calculate overall statistics
        total_objectives = sum(len(r.learning_objectives) for r in reports)
        avg_alignment_score = sum(r.overall_alignment_score for r in reports) / len(reports)
        
        # Analyze Bloom level distribution across all lectures
        all_bloom_levels = []
        for report in reports:
            all_bloom_levels.extend([obj.bloom_level for obj in report.learning_objectives])
        
        bloom_distribution = defaultdict(int)
        for level in all_bloom_levels:
            bloom_distribution[level] += 1
        
        # Generate summary content
        summary_lines = [
            "# Batch Alignment Analysis Summary",
            "",
            f"**Analysis Date**: {datetime.now().isoformat()}",
            f"**Lectures Analyzed**: {len(reports)}",
            f"**Total Learning Objectives**: {total_objectives}",
            f"**Average Alignment Score**: {avg_alignment_score:.2f}",
            "",
            "## Individual Lecture Results",
            ""
        ]
        
        for report in reports:
            score_icon = "🟢" if report.overall_alignment_score >= 0.8 else "🟡" if report.overall_alignment_score >= 0.6 else "🔴"
            summary_lines.append(
                f"- **{report.lecture_name}**: {report.overall_alignment_score:.2f} {score_icon} "
                f"({len(report.learning_objectives)} objectives)"
            )
        
        summary_lines.extend([
            "",
            "## Bloom Taxonomy Distribution",
            ""
        ])
        
        for level, count in sorted(bloom_distribution.items()):
            percentage = (count / total_objectives) * 100
            summary_lines.append(f"- **{level.title()}**: {count} objectives ({percentage:.1f}%)")
        
        # Add common gaps and recommendations
        all_gaps = []
        all_recommendations = []
        
        for report in reports:
            all_gaps.extend(report.gaps_identified)
            all_recommendations.extend(report.recommendations)
        
        if all_gaps:
            summary_lines.extend([
                "",
                "## Common Alignment Gaps",
                ""
            ])
            
            gap_counts = defaultdict(int)
            for gap in all_gaps:
                gap_counts[gap] += 1
            
            for gap, count in sorted(gap_counts.items(), key=lambda x: x[1], reverse=True):
                summary_lines.append(f"- {gap} (appears in {count} lectures)")
        
        summary_lines.extend([
            "",
            "## Next Steps",
            "",
            "1. Address common alignment gaps identified across lectures",
            "2. Balance Bloom taxonomy distribution where needed",
            "3. Strengthen weakly aligned objectives with targeted content",
            "4. Develop comprehensive assessment rubrics",
            "5. Validate alignment through instructor review and student feedback",
            "",
            "---",
            f"*Generated by assessment_aligner.py*"
        ])
        
        summary_content = "\n".join(summary_lines)
        summary_path.write_text(summary_content, encoding='utf-8')
        
        logger.info(f"Batch alignment summary written to: {summary_path}")


def main():
    """Main CLI interface for assessment alignment tool."""
    parser = argparse.ArgumentParser(
        description="Assessment Alignment Automation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python3 assessment_aligner.py --lecture-dir lectures/lecture_01/
    python3 assessment_aligner.py --batch-align lectures/
    python3 assessment_aligner.py --generate-rubrics lectures/lecture_01/ --output rubric.json
        """
    )
    
    parser.add_argument('--lecture-dir', help='Single lecture directory to analyze')
    parser.add_argument('--batch-align', help='Directory containing multiple lectures')
    parser.add_argument('--generate-rubrics', help='Generate assessment rubrics for lecture')
    parser.add_argument('--config', help='Configuration file path')
    parser.add_argument('--output', help='Output file path for results')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if not any([args.lecture_dir, args.batch_align, args.generate_rubrics]):
        parser.error("Must specify --lecture-dir, --batch-align, or --generate-rubrics")
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize aligner
    aligner = AssessmentAligner(args.config)
    
    try:
        if args.batch_align:
            # Batch alignment analysis
            logger.info("Starting batch alignment analysis")
            reports = aligner.batch_align_lectures(args.batch_align)
            
            print(f"\n=== Batch Alignment Results ===")
            print(f"Lectures analyzed: {len(reports)}")
            
            for report in reports:
                score_icon = "✅" if report.overall_alignment_score >= 0.8 else "⚠️" if report.overall_alignment_score >= 0.6 else "❌"
                print(f"  {report.lecture_name}: {report.overall_alignment_score:.2f} {score_icon}")
            
            avg_score = sum(r.overall_alignment_score for r in reports) / len(reports)
            print(f"\nAverage alignment score: {avg_score:.2f}")
            
        elif args.lecture_dir:
            # Single lecture analysis
            logger.info(f"Analyzing single lecture: {args.lecture_dir}")
            report = aligner.generate_alignment_report(args.lecture_dir)
            
            # Save report if output specified
            if args.output:
                aligner.save_report(report, args.output)
            
            # Print summary
            print(f"\n=== Alignment Analysis Results ===")
            print(f"Lecture: {report.lecture_name}")
            print(f"Overall Score: {report.overall_alignment_score:.2f}")
            print(f"Learning Objectives: {len(report.learning_objectives)}")
            
            print(f"\nAlignment Strength Distribution:")
            strength_counts = defaultdict(int)
            for result in report.alignment_results:
                strength_counts[result.alignment_strength] += 1
            
            for strength, count in strength_counts.items():
                print(f"  {strength.title()}: {count}")
            
            if report.gaps_identified:
                print(f"\nGaps Identified:")
                for gap in report.gaps_identified:
                    print(f"  - {gap}")
        
        elif args.generate_rubrics:
            # Generate assessment rubrics
            logger.info(f"Generating rubrics for: {args.generate_rubrics}")
            report = aligner.generate_alignment_report(args.generate_rubrics)
            
            output_file = args.output or f"{args.generate_rubrics}/assessment_rubric.json"
            
            with open(output_file, 'w') as f:
                json.dump(report.rubric_suggestions, f, indent=2)
            
            print(f"\n=== Rubric Generation Results ===")
            print(f"Rubric saved to: {output_file}")
            print(f"Criteria generated: {len(report.rubric_suggestions.get('criteria', {}))}")
    
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()