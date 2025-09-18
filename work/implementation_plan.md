# DataSci 217 Implementation Plan - UPDATED REQUIREMENTS

## Project Objective (REVISED)
Reorganize DataSci 217 from existing content into a coherent 11-lecture sequence:
- **Lectures 1-5**: Complete foundational toolkit (1-unit completion option)  
- **Lectures 6-11**: Advanced mastery and professional skills (2-unit completion)
- **Target**: 11 lectures total (01-11, no lecture 12) in folders 01/ -> 11/ in the repository root, each lecture with hands-on demos and an assignment (details below)
- **Method**: Evidence-based analysis focusing on practical utility and daily data science tools

---

## 📋 **IMPLEMENTATION ROADMAP**

### **Phase 1: Content Assessment** ✅ COMPLETED ???

## 🗂️ **REVISED TARGET DELIVERABLES**

### **Required Directory Structure**:
```
01-11/                             # Final lecture directories (root level)
├── 01/
│   ├── README.md                   # Notion-formatted narrative content
│   ├── assignment/                # GitHub Classroom compatible
│   │   ├── README.md              # Assignment instructions
│   │   ├── test_*.py              # Pytest testing framework  
│   │   ├── starter_code.py        # Template for students
│   │   └── requirements.txt       # Dependencies
│   ├── demo1.md                   # Hands-on demo 1 (instructor guide)
│   ├── demo2.ipynb               # Hands-on demo 2 (if applicable)
│   └── [media files]             # Supporting materials
├── 02/ ... ├── 11/               # All lectures follow same structure
```

### **Content Standards**:

1. **Lecture Format**:
   - **Title**: First line = plain text title (no # prefix)
   - **Notion-compatible headings**: Use one less # than standard markdown
     * Section headings: `# Section Name` (instead of `##`)
     * Subsections: `## Subsection Name` (instead of `###`)
     * Sub-subsections: `### Details` (instead of `####`)
   - 2-3 embedded hands-on demo callouts
   - Long-form narrative format
   - Practical utility focus

2. **Lecture Content Organization**:
   - **Topic Structure**:
     * Brief conceptual description (what the method/tool is)
     * Reference section (function/method structure, inputs/outputs, params)
     * Visual content (at most 1-2 pieces for visual learners)
     * VERY brief usage example (context covered in demos)
   - **Content Triage**:
     * **Main lecture**: Essential daily data science tools only
     * **Bonus content**: Advanced topics, theoretical deep-dives, 
   - **Tone**: Highly knowledgeable with nerdy humor, relevant xkcd comics (suggestions for similar also welcome), occasional memes

3. **Interactive Components per Lecture**:
   - **Demos**: Exactly 2-3 hands-on demos per lecture
     * Step-by-step instructor guidance
     * Practical/hands-on review of lecture concepts
     * Real-world scenarios, minimal context (lecture provides theory)
   - **Assignment**: Exactly 1 assignment per lecture
     * Gradable via pytest for GitHub Classroom submission
     * Progressive difficulty within assignment (multiple questions building on earlier ones)
     * Focus on practical competence demonstration with methods/tools, not excellence
     * Dependency management across lectures

4. **Assignment Structure**:
   - Separate assignment/ subdirectory per lecture
   - GitHub Classroom compatibility required
   - Pytest-based automated testing framework following `work/example_assignment_with_github_actions/` structure:
     * `.github/workflows/classroom.yml` for automated grading
     * `.github/tests/` directory with test files that can be updated remotely
     * Assignment content in dedicated files (not README.md - reserve for student use)
     * Requirements.txt for dependencies
   - Progressive difficulty within each assignment (questions build on previous questions)
   - Practical competence-focused grading (can you use the tool?) rather than excellence-focused
   - Clear grading criteria and test cases

5. **Content Source Integration**:
   - **Python Content**: McKinney (`work/mckinney_topics_summary.md`) is THE authoritative reference for all Python content organization and progression, exclude capstone project work
   - **Non-Python Content**: Draw from multiple sources:
     * Command line/shell: `work/tlcl_topics.md` (The Linux Command Line book)
     * Development workflows: `work/missing_semester_topics.md`
     * Existing lecture content: `lectures_bkp/` for established patterns
     * Internet resources: As needed for current best practices
   - Integrate content from these sources into cohesive lecture materials
   - **Prerequisite Planning**: Follow dependency mapping in `work/prerequisites_mapping.md` to ensure no skill gaps in lecture sequence

6. **Bonus Content Strategy ("DLC")**:
   - Advanced topics that exceed daily tool usage
   - Theoretical foundations beyond practical application
   - Optional enrichment for motivated students

---

## 📊 **CONTENT ANALYSIS METHODOLOGY**

### **For Existing Lectures**:
1. **Topic Extraction**: List every concept, tool, and skill covered
2. **Prerequisite Analysis**: What knowledge each topic assumes
3. **Content Volume**: Estimate teaching time and complexity
4. **Practical Components**: Document hands-on exercises and assignments
5. **Transfer Value**: Assess broad applicability vs. specialized use

### **For McKinney Content**:
1. **Chapter Structure**: How McKinney organizes Python/data science concepts
2. **Concept Progression**: McKinney's approach to skill building
3. **Coverage Gaps**: Topics in McKinney not covered in existing lectures
4. **Pedagogical Approach**: How McKinney teaches concepts effectively
5. **Integration Opportunities**: Where McKinney enhances existing content

### **For Combined Analysis**:
1. **Content Overlap**: Where existing lectures and McKinney cover same topics
2. **Complementary Content**: Where sources enhance each other
3. **Gap Identification**: Missing concepts not covered by either source
4. **Reorganization Strategy**: How to optimally combine both sources
5. **Quality vs. Quantity**: What to cut for 10-lecture constraint
