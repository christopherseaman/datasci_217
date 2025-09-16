Here's a comprehensive prompt you can use to instruct an LLM to review each lecture for adherence to the restructure plan:

```
# Lecture Review Prompt for DataSci 217

You are reviewing lecture content for DataSci 217, a data science course. Please analyze each lecture file against the specifications in `work/reorganized_lectures_proposed_restructure.md` and provide detailed feedback on three key areas:

## 1. Content Adherence

**Review against the proposed restructure plan:**
- **Word count targets**: Check if content meets the target word counts (see table in lines 340-354)
- **McKinney alignment**: Verify content follows McKinney chapter progression as specified
- **Content completeness**: Check if all required content areas are covered:
  - For Lecture 01: Extended CLI Examples, Enhanced Python Fundamentals, Debugging Foundation
  - For other lectures: Check against their specific content requirements
- **Missing components**: Identify if demos, assignments, or bonus content are missing (see status indicators)
- **Content quality**: Ensure practical utility focus and professional context

## 2. Lecture Style Adherence

**Check for proper lecture format:**
- **Title format**: First line should be plain text title (no # prefix)
- **Narrative style**: Long-form, practical utility focus with professional context
- **Tone**: Highly knowledgeable with nerdy humor, relevant xkcd comics, occasional memes
- **Content organization**: Brief conceptual description → Reference section → Visual content → Brief usage example
- **Demo integration**: 2-3 hands-on demo callouts embedded in content
- **Progressive difficulty**: Content builds appropriately within and across lectures

## 3. Notion-Style Formatting

**Verify proper heading hierarchy:**
- **Main title**: Plain text (no #) - should be the first line
- **Major sections**: Use `# Section Name` (not ##)
- **Subsections**: Use `## Subsection Name` (not ###)
- **Sub-subsections**: Use `### Details` (not ####)
- **Content structure**: Each major section should have multiple subsections
- **Consistency**: All lectures should follow the same heading pattern

## 4. Specific Requirements by Lecture

**For each lecture, check:**
- **Lecture 01**: Should have ~11 major sections (# headings) covering CLI, Python basics, and debugging
- **Lecture 02**: Should include Git error interpretation and environment troubleshooting
- **Lecture 03**: Should include data structure error patterns and file I/O troubleshooting
- **Lecture 04**: Should include function debugging and CLI error interpretation
- **Lecture 05**: Should include NumPy error messages and array troubleshooting
- **Lectures 06-11**: Check for appropriate debugging integration and content balance

## 5. Output Format

For each lecture file, provide:

```
## Lecture [XX]: [Title]

### Content Adherence
- **Word count**: [Current] / [Target] words
- **McKinney alignment**: [Assessment]
- **Missing content**: [List any gaps]
- **Content quality**: [Assessment]

### Lecture Style
- **Title format**: [Correct/Incorrect]
- **Narrative style**: [Assessment]
- **Demo integration**: [Count and quality]
- **Tone**: [Assessment]

### Notion-Style Formatting
- **Main title**: [Correct/Incorrect]
- **Major sections (#)**: [Count and list]
- **Subsections (##)**: [Assessment]
- **Heading hierarchy**: [Correct/Incorrect]

### Recommendations
- [Specific actionable improvements]
- [Priority level: High/Medium/Low]
```

## 6. Priority Focus Areas

**High Priority:**
- Missing demos and assignments (Lectures 04, 06-07, 09-11)
- Content expansion needed (Lecture 01: 1,000 → 3,000 words)
- Content reduction needed (Lecture 11: 6,014 → 3,200 words)
- Notion-style heading conversion

**Medium Priority:**
- Debugging content integration
- Content balance for 90-minute sessions
- McKinney alignment verification

**Low Priority:**
- Tone and style refinements
- Minor content adjustments

## 7. Success Criteria

A lecture meets standards when:
- ✅ Word count within 10% of target
- ✅ All required content areas covered
- ✅ Proper Notion-style heading hierarchy
- ✅ Appropriate number of major sections (# headings)
- ✅ Demo callouts present (where required)
- ✅ Professional tone with practical focus
- ✅ McKinney alignment maintained

Please review each lecture file systematically and provide detailed feedback for improvement.
```

This prompt will help ensure each lecture is reviewed comprehensively against the restructure plan, with specific attention to the Notion-style formatting and content requirements you've established.