#!/usr/bin/env python3
"""
Example: Command Line Interface Assignment - Text File Analyzer

This example demonstrates a complete CLI homework assignment with argument parsing,
file processing, and different output formats.

Student: Example Student
Course: Data Science 217
Assignment: Command Line Tools
"""

import argparse
import sys
import os
import json
from pathlib import Path
from typing import Dict, List, Optional
import re
from collections import Counter


def analyze_text_file(file_path: str) -> Dict:
    """
    Analyze a text file and return statistics.
    
    Args:
        file_path: Path to the text file to analyze
    
    Returns:
        Dictionary containing analysis results
    
    Raises:
        FileNotFoundError: If file doesn't exist
        PermissionError: If file can't be read
        UnicodeDecodeError: If file encoding is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except PermissionError:
        raise PermissionError(f"Permission denied reading file: {file_path}")
    except UnicodeDecodeError as e:
        raise UnicodeDecodeError(f"File encoding error in {file_path}: {e}")
    
    # Basic statistics
    lines = content.split('\n')
    words = content.split()
    characters = len(content)
    characters_no_spaces = len(content.replace(' ', '').replace('\n', '').replace('\t', ''))
    
    # Word frequency analysis
    # Clean words (remove punctuation, convert to lowercase)
    clean_words = []
    for word in words:
        clean_word = re.sub(r'[^a-zA-Z0-9]', '', word.lower())
        if clean_word:  # Only add non-empty words
            clean_words.append(clean_word)
    
    word_freq = Counter(clean_words)
    
    # Sentence analysis (basic - split on periods, exclamations, questions)
    sentences = re.split(r'[.!?]+', content)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Paragraph analysis (split on double newlines)
    paragraphs = content.split('\n\n')
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Reading time estimation (average 200 words per minute)
    reading_time_minutes = len(words) / 200 if words else 0
    
    return {
        'file_info': {
            'file_path': file_path,
            'file_size_bytes': os.path.getsize(file_path),
            'encoding': 'utf-8'
        },
        'basic_stats': {
            'lines': len(lines),
            'words': len(words),
            'characters': characters,
            'characters_no_whitespace': characters_no_spaces,
            'sentences': len(sentences),
            'paragraphs': len(paragraphs)
        },
        'word_analysis': {
            'unique_words': len(word_freq),
            'most_common_words': word_freq.most_common(10),
            'average_word_length': sum(len(word) for word in clean_words) / len(clean_words) if clean_words else 0
        },
        'readability': {
            'average_words_per_sentence': len(words) / len(sentences) if sentences else 0,
            'average_sentences_per_paragraph': len(sentences) / len(paragraphs) if paragraphs else 0,
            'estimated_reading_time_minutes': reading_time_minutes
        }
    }


def format_output(analysis: Dict, format_type: str) -> str:
    """
    Format analysis results for different output types.
    
    Args:
        analysis: Analysis results dictionary
        format_type: Output format ('text', 'json', 'csv')
    
    Returns:
        Formatted string output
    """
    if format_type == 'json':
        return json.dumps(analysis, indent=2)
    
    elif format_type == 'csv':
        # Simple CSV format for basic stats
        csv_lines = ['metric,value']
        basic_stats = analysis['basic_stats']
        for key, value in basic_stats.items():
            csv_lines.append(f'{key},{value}')
        return '\n'.join(csv_lines)
    
    else:  # text format (default)
        output_lines = []
        
        # File information
        file_info = analysis['file_info']
        output_lines.extend([
            "📄 FILE ANALYSIS REPORT",
            "=" * 50,
            f"File: {file_info['file_path']}",
            f"Size: {file_info['file_size_bytes']:,} bytes",
            ""
        ])
        
        # Basic statistics
        basic_stats = analysis['basic_stats']
        output_lines.extend([
            "📊 BASIC STATISTICS",
            "-" * 20,
            f"Lines: {basic_stats['lines']:,}",
            f"Words: {basic_stats['words']:,}",
            f"Characters: {basic_stats['characters']:,}",
            f"Characters (no whitespace): {basic_stats['characters_no_whitespace']:,}",
            f"Sentences: {basic_stats['sentences']:,}",
            f"Paragraphs: {basic_stats['paragraphs']:,}",
            ""
        ])
        
        # Word analysis
        word_analysis = analysis['word_analysis']
        output_lines.extend([
            "🔤 WORD ANALYSIS",
            "-" * 15,
            f"Unique words: {word_analysis['unique_words']:,}",
            f"Average word length: {word_analysis['average_word_length']:.1f} characters",
            ""
        ])
        
        # Most common words
        if word_analysis['most_common_words']:
            output_lines.extend([
                "📈 MOST COMMON WORDS",
                "-" * 20
            ])
            for i, (word, count) in enumerate(word_analysis['most_common_words'][:5], 1):
                output_lines.append(f"{i:2d}. {word:<15} ({count:,} times)")
            output_lines.append("")
        
        # Readability
        readability = analysis['readability']
        output_lines.extend([
            "📖 READABILITY METRICS",
            "-" * 22,
            f"Average words per sentence: {readability['average_words_per_sentence']:.1f}",
            f"Average sentences per paragraph: {readability['average_sentences_per_paragraph']:.1f}",
            f"Estimated reading time: {readability['estimated_reading_time_minutes']:.1f} minutes",
            ""
        ])
        
        return '\n'.join(output_lines)


def save_output(content: str, output_file: str) -> None:
    """
    Save output to a file.
    
    Args:
        content: Content to save
        output_file: Path where to save the content
    
    Raises:
        PermissionError: If file can't be written
        OSError: If there's an OS-level error
    """
    try:
        # Ensure directory exists
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
    except PermissionError:
        raise PermissionError(f"Permission denied writing to: {output_file}")
    except Exception as e:
        raise OSError(f"Error writing to {output_file}: {e}")


def create_sample_file(file_path: str = 'sample.txt') -> None:
    """Create a sample text file for testing."""
    sample_content = """The Art of Programming

Programming is both a science and an art. It requires logical thinking,
creativity, and attention to detail. A good programmer must understand
not only the syntax of a programming language, but also the principles
of software design and problem-solving.

The History of Computing

The field of computer science has evolved rapidly over the past century.
From the early mechanical calculators to modern quantum computers,
the journey has been remarkable. Each generation of computers has
brought new possibilities and challenges.

Learning to Code

Learning programming can be challenging, but it is also very rewarding.
Start with the basics: understand variables, loops, and functions.
Practice regularly and don't be afraid to make mistakes. Every error
is a learning opportunity.

The most important skill for a programmer is problem-solving.
Before writing code, think about the problem you're trying to solve.
Break it down into smaller, manageable pieces. Then implement
each piece step by step.

Remember: good code is not just code that works, but code that
is readable, maintainable, and efficient. Always write code as
if the person maintaining it is a violent psychopath who knows
where you live!
"""
    
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(sample_content)
    
    print(f"Created sample file: {file_path}")


def main():
    """Main function with command line argument parsing."""
    parser = argparse.ArgumentParser(
        description='Analyze text files and provide detailed statistics',
        epilog='Example: python main.py input.txt -o report.txt -f text'
    )
    
    # Positional argument
    parser.add_argument(
        'input_file',
        help='Path to the text file to analyze'
    )
    
    # Optional arguments
    parser.add_argument(
        '-o', '--output',
        help='Output file path (default: print to stdout)',
        default=None
    )
    
    parser.add_argument(
        '-f', '--format',
        choices=['text', 'json', 'csv'],
        default='text',
        help='Output format (default: text)'
    )
    
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Enable verbose output'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Suppress progress messages'
    )
    
    parser.add_argument(
        '--create-sample',
        action='store_true',
        help='Create a sample text file for testing'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version='Text Analyzer 1.0.0'
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    try:
        # Handle special commands
        if args.create_sample:
            create_sample_file()
            if not args.quiet:
                print("Sample file created successfully!")
            return 0
        
        # Validate input file
        if not os.path.exists(args.input_file):
            print(f"Error: File not found: {args.input_file}", file=sys.stderr)
            return 1
        
        # Progress message
        if args.verbose and not args.quiet:
            print(f"Analyzing file: {args.input_file}")
        
        # Analyze the file
        analysis = analyze_text_file(args.input_file)
        
        if args.verbose and not args.quiet:
            print(f"Analysis complete. Found {analysis['basic_stats']['words']} words.")
        
        # Format output
        formatted_output = format_output(analysis, args.format)
        
        # Output results
        if args.output:
            save_output(formatted_output, args.output)
            if not args.quiet:
                print(f"Results saved to: {args.output}")
        else:
            print(formatted_output)
        
        return 0
        
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except PermissionError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except UnicodeDecodeError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())