# Lecture Format Guidelines

## Overview

Standard format for DataSci 217 lecture materials using narrative-driven learning with embedded practical demonstrations.

## Format Philosophy

- **Story-driven**: Each lecture tells a coherent story about why concepts matter
- **Context-first**: Always explain the "why" before diving into the "how"
- **Progressive complexity**: Build from simple concepts to advanced applications
- **Real-world connections**: Connect every concept to practical data science work

## File Structure

```
lectures_new/
├── lecture_01/
│   ├── lecture_01.md              # Main narrative content
│   ├── demo_lecture_01.py         # Executable demonstration code
│   ├── media/                     # Images, diagrams, etc.
│   ├── exercises/                 # Practice problems
│   └── resources/                 # Additional materials
├── lecture_02/
│   └── [same structure]
└── ...
```

## Content Standards

### Narrative Markdown (.md) Files

#### Required Sections
1. **Overview** - Brief, engaging introduction
2. **Learning Objectives** - Specific, measurable outcomes
3. **Prerequisites** - Links to previous concepts
4. **Core Concepts** - Main content with narrative flow
5. **Hands-On Practice** - Guided exercises
6. **Common Pitfalls** - Troubleshooting guidance
7. **Real-World Applications** - Industry relevance
8. **Assessment Integration** - Connection to graded work
9. **Further Reading** - Extension materials
10. **Next Steps** - Preview of upcoming content

#### Writing Style Guidelines
- **Conversational but professional tone**
- **Use "you" to address students directly**
- **Include rhetorical questions to engage thinking**
- **Provide multiple examples for abstract concepts**
- **Use analogies to make difficult concepts accessible**
- **Avoid bullet points in favor of flowing paragraphs**

#### Code Integration
```markdown
When introducing code concepts, embed them naturally:

The pandas library provides powerful data manipulation tools. Consider this example of loading a dataset:

```python
import pandas as pd

# Load data with explicit encoding to handle special characters
df = pd.read_csv('data.csv', encoding='utf-8')
print(f"Loaded {len(df)} rows of data")
```

This code demonstrates three important principles: explicit imports, defensive programming with encoding specification, and immediate feedback through informative print statements. These practices become crucial when working with real-world datasets that often contain...
```

### Python Demo Files (.py)

#### Structure Template
```python
#!/usr/bin/env python3
"""
Comprehensive docstring explaining the demo's purpose
and how it relates to the lecture content.
"""

# Clear section headers
# ============================================================================
# SECTION 1: [MAJOR_TOPIC]
# ============================================================================

def demonstrate_concept():
    """
    Detailed docstring explaining what this function demonstrates
    and why it matters for student learning.
    """
    # Step-by-step implementation with educational comments
    pass

if __name__ == "__main__":
    # Interactive execution that students can follow along with
    pass
```

#### Code Quality Standards
- **Educational comments**: Explain why, not just what
- **Error handling**: Show proper exception handling
- **Type hints**: Use where helpful for clarity
- **Docstrings**: Comprehensive documentation for all functions
- **Modular design**: Break complex operations into understandable functions
- **Interactive elements**: Allow students to modify and experiment

## Conversion Process

### Using the Conversion Script

```bash
# Convert a single lecture
python scripts/convert_lecture.py 01

# Dry run to see what would be changed
python scripts/convert_lecture.py 03 --dry-run

# Convert all lectures
for i in {01..12}; do
    if [ -d "$i" ]; then
        python scripts/convert_lecture.py "$i"
    fi
done
```

### Manual Conversion Steps

1. **Analyze existing content**
   - Read through all current materials (index.md, demo files, assignments)
   - Identify key learning objectives
   - Note practical examples and exercises

2. **Create narrative structure**
   - Use template as starting point
   - Write engaging introduction that motivates the topic
   - Develop core concepts section with flowing narrative
   - Integrate existing examples into the story

3. **Develop Python demonstration**
   - Extract code examples from existing demos
   - Structure as educational functions with clear progression
   - Add interactive elements for student exploration
   - Include common pitfalls and debugging examples

4. **Quality review**
   - Ensure narrative flows logically
   - Verify all code examples run correctly
   - Check that learning objectives align with content
   - Test that exercises match difficulty level

## Assessment Integration

### Formative Assessment
- **Quick comprehension checks** embedded in narrative
- **Interactive coding exercises** in Python demos
- **Reflection questions** that connect to real-world applications

### Summative Assessment
- **Clear alignment** between lecture content and assignment requirements
- **Scaffolded complexity** that builds on demonstrated concepts
- **Practical applications** that mirror industry workflows

## Quality Assurance

### Content Review Checklist
- [ ] Narrative is engaging and flows logically
- [ ] Code examples are tested and functional
- [ ] Learning objectives are clear and measurable
- [ ] Prerequisites are accurately identified
- [ ] Real-world applications are relevant and current
- [ ] Common pitfalls section addresses actual student challenges
- [ ] Assessment alignment is explicit
- [ ] Next steps create anticipation for following lecture

### Technical Review Checklist
- [ ] All code runs without errors
- [ ] File structure follows standard format
- [ ] Media files are properly referenced
- [ ] Markdown formatting is Notion-compatible
- [ ] Links between materials work correctly
- [ ] Interactive elements function as intended

## Implementation Timeline

### Phase 1: Infrastructure (Complete)
- ✓ Template creation
- ✓ Conversion script development
- ✓ Directory structure establishment
- ✓ Guidelines documentation

### Phase 2: Pilot Conversion
- [ ] Convert 2-3 representative lectures
- [ ] Test with students/instructors
- [ ] Refine templates based on feedback
- [ ] Update conversion process

### Phase 3: Full Conversion
- [ ] Convert all remaining lectures
- [ ] Cross-lecture coherence review
- [ ] Final quality assurance pass
- [ ] Deployment to course platform

### Phase 4: Maintenance
- [ ] Regular content updates
- [ ] Student feedback integration
- [ ] Continuous improvement process

## Support and Troubleshooting

### Common Issues
- **Code examples not running**: Check Python version compatibility and library versions
- **Markdown formatting problems**: Verify Notion compatibility using test imports
- **Missing media files**: Ensure all references use relative paths and files exist

### Getting Help
- Check the conversion script logs for specific error messages
- Review the template files for proper formatting examples
- Test converted materials in Notion before finalizing

---

*Guidelines Version: 1.0*
*Last Updated: 2025-01-13*