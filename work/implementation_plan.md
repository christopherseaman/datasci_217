# DataSci 217 Implementation Plan - UPDATED REQUIREMENTS

## Project Objective (REVISED)
Reorganize DataSci 217 from existing content into a coherent 11-lecture sequence:
- **Lectures 1-5**: Complete foundational toolkit (1-unit completion option)  
- **Lectures 6-11**: Advanced mastery and professional skills (2-unit completion)
- **Target**: 11 lectures total (01-11, no lecture 12)
- **Method**: Evidence-based analysis focusing on practical utility and daily data science tools

---

## Current Status: REQUIRES REVISION

Previous work completed 10-lecture sequence, but requirements specify 11 lectures with enhanced structure.

---

## 🔍 **UPDATED SWARM ASSESSMENT FINDINGS**

### **Current State Analysis**:
- **lectures_bkp/**: Original slide-format content (7,474 total lines) - needs load reduction
- **lectures_inter/**: Strong narrative content L01-L09, L10 removed per instructor preference  
- **Content Load Analysis**: L07 (1,227 lines), L09 (917 lines), L11 (1,006 lines) are heaviest

### **Load Reduction Strategy**:
- **Git Simplification**: Focus on VS Code/GitHub GUI, move command line git to bonus
- **Content Streamlining**: Heavy topics to bonus sections within lectures
- **Practical Focus**: Emphasize daily data science tools, reduce theoretical content

### **Gap Analysis Against Requirements**:

#### ❌ **Priority Updates Needed**:
1. **Load Reduction**: Student overwhelm concern - need strategic content reduction
2. **Git Approach**: Shift from command line to VS Code/GitHub GUI focus
3. **Lecture Structure**: 11 lectures (01-11) with lectures_inter/10 excluded
4. **Bonus Content**: Move heavy/advanced topics to lecture-associated bonus sections
5. **Book References**: Update with McKinney, Shotts, MIT Missing Semester attribution

#### ✅ **Strong Foundation Available**:
- Quality narrative content in lectures_inter/ (L01-L09)
- Clinical research content (lectures_bkp/L11) for final lecture
- Good practical focus already established  
- Existing assignment frameworks to build upon

---

## 📋 **REVISED IMPLEMENTATION ROADMAP**

### **Phase 1: Content Assessment** ✅ COMPLETED

#### **Step 1.1: Requirements Assessment**
- [x] Analyzed lectures_bkp/ and lectures_inter/ content structure
- [x] Identified gap: need 11 lectures (01-11) not 10 lectures
- [x] Assessed demo requirements: need 2-3 hands-on demos per lecture
- [x] Evaluated assignment structure for GitHub Classroom compatibility
- [x] Documented content that maintains practical utility focus

#### **Step 1.2: Content Quality Assessment**
- [x] Validated narrative transformation quality in lectures_inter/
- [x] Confirmed professional development focus maintained
- [x] Identified L11 content source: lectures_bkp/11 clinical research focus
- [x] Verified assignment structure needs standardization

### **Phase 2: Restructuring Design** ⏳ IN PROGRESS

#### **Step 2.1: Content Load Reduction Strategy**
- [ ] Identify heavy content for bonus sections (L07 viz theory, L09 advanced automation)
- [ ] Simplify git to VS Code/GitHub GUI approach only
- [ ] Streamline lectures_inter/ L01-L09 to essential daily tools
- [ ] Create bonus/ subdirectories within lecture folders for advanced topics
- [ ] Ensure no student overwhelm while maintaining utility

#### **Step 2.2: 11-Lecture Structure Design**
- [ ] Use lectures_inter/ L01-L09 as foundation (excluding L10)  
- [ ] Create L10 from lighter content (file operations, workflows)
- [ ] Integrate lectures_bkp/L11 clinical research as L11
- [ ] Maintain L01-L05 foundational, L06-L11 advanced structure
- [ ] Focus on practical daily data science tools throughout

#### **Step 2.3: Git Approach Redesign**
- [ ] Remove command line git requirements from core lectures
- [ ] Focus on VS Code git integration and GitHub web interface
- [ ] Cover git concepts but emphasize GUI tools
- [ ] Move advanced git topics to bonus sections
- [ ] Ensure students can collaborate without command line complexity

#### **Step 2.4: Resource Integration** 
- [ ] Update course materials with McKinney Python book integration
- [ ] Reference Shotts Linux Command Line for bonus command line content
- [ ] Credit MIT Missing Semester influence on course design
- [ ] Maintain practical focus while acknowledging theoretical foundations

### **Phase 3: Implementation** 📋 PLANNED

#### **Step 3.1: Lecture Content Deployment**
- [ ] Deploy 11-lecture sequence to directories 01-11
- [ ] Integrate clinical research content as L11
- [ ] Apply Notion formatting to all content
- [ ] Ensure long-form narrative format maintained

#### **Step 3.2: Demo Integration**  
- [ ] Add 2-3 hands-on demos per lecture
- [ ] Create step-by-step instructor guides
- [ ] Embed demo callouts within lecture content
- [ ] Test demo scenarios for timing and complexity

#### **Step 3.3: Assignment Standardization**
- [ ] Create assignment/ subdirectory for each lecture  
- [ ] Implement pytest-compatible testing frameworks
- [ ] Add automated grading capabilities
- [ ] Ensure GitHub Classroom compatibility

#### **Step 3.4: Quality Assurance**
- [ ] Validate content progression and difficulty
- [ ] Test assignment dependency chains
- [ ] Verify practical utility focus maintained  
- [ ] Confirm non-overwhelming student experience

---

## 🗂️ **REVISED TARGET DELIVERABLES**

### **Required Directory Structure**:
```
01-11/                             # Final lecture directories (root level)
├── 01/
│   ├── index.md                   # Notion-formatted narrative content
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
1. **Lecture Structure**:
   - First line = title (no # prefix)
   - # headings for major sections  
   - 2-3 embedded hands-on demo callouts
   - Long-form narrative format
   - Practical utility focus

2. **Assignment Structure**:
   - Separate assignment/ subdirectory
   - GitHub Classroom compatibility
   - Pytest-based automated testing
   - Progressive difficulty ensuring dependency management
   - Focus on competence demonstration

3. **Demo Requirements**:
   - 2-3 hands-on demos per lecture
   - Step-by-step instructor guidance
   - Complement lecture progression
   - Real-world practical scenarios

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

---

## 🎯 **REVISED SUCCESS CRITERIA**

### **Structural Requirements**:
- [ ] 11 lectures total (L01-L11, no L12) 
- [ ] L01-L05: Complete foundational toolkit (1-unit completion)
- [ ] L06-L11: Advanced professional skills (2-unit completion)  
- [ ] No prerequisite gaps in L01-L11 sequence

### **Content Quality Standards**:  
- [ ] Not overwhelming for students - practical utility focus
- [ ] Daily data science tools emphasis (minus R)
- [ ] Long-form narrative markdown documents
- [ ] Notion-compatible formatting structure

### **Interactive Components**:
- [ ] 2-3 hands-on demos per lecture (instructor-led)  
- [ ] Step-by-step demonstration scenarios
- [ ] Embedded practical exercises within lectures

### **Assignment Framework**:
- [ ] GitHub Classroom compatible structure
- [ ] Automated pytest-based grading
- [ ] Progressive difficulty with dependency management
- [ ] Competence demonstration for each lecture
- [ ] Separate assignment/ subdirectories

---

## 📋 **IMMEDIATE NEXT STEPS**

### **PHASE 1: PLAN REVIEW** ⏳ UPDATED FOR LOAD REDUCTION
The implementation plan has been updated to address student overwhelm concerns:

1. **Load Reduction**: Strategic content reduction and bonus section approach
2. **Git Simplification**: VS Code/GitHub GUI focus, command line moved to bonus
3. **Practical Focus**: Daily data science tools emphasis, theory to bonus  
4. **Resource Integration**: McKinney, Shotts, MIT Missing Semester attribution
5. **11-Lecture Structure**: Using lectures_inter/ L01-L09 + streamlined L10-L11

### **KEY CONTENT REDUCTION STRATEGIES IDENTIFIED**

**High-Priority for Bonus Sections**:
- **Advanced Git**: Command line workflows → bonus content
- **Visualization Theory**: L07 design principles → bonus content  
- **Advanced Automation**: L09 complex scripting → bonus content
- **Theoretical Foundations**: Move to bonus while keeping practical applications

**Maintained in Core**:
- Essential daily workflows
- GUI-based version control
- Practical data analysis patterns
- Real-world project applications

---

## 🎯 **IMPLEMENTATION READINESS**

### **Assets Available**:
✅ High-quality narrative content (lectures_inter/ L01-L09, excluding L10)  
✅ Clinical research content (lectures_bkp/L11) for final lecture  
✅ Load reduction strategies identified and documented
✅ Git simplification approach defined (VS Code/GitHub GUI focus)
✅ Resource attribution updated (McKinney, Shotts, MIT Missing Semester)
✅ README.md updated with new course philosophy and structure

### **Load Reduction Plan Ready**:  
✅ Bonus content strategy defined for heavy topics
✅ GUI-first approach specified to reduce command line complexity
✅ Content prioritization completed (essential vs. advanced)
✅ Student overwhelm mitigation strategies documented
✅ Practical utility focus maintained throughout

### **Specific Content Recommendations**:
- **L02 Git**: Focus on VS Code integration, GitHub web interface
- **L07 Visualization**: Core plotting skills, theory moved to bonus/
- **L09 Automation**: Basic workflows, advanced scripting to bonus/
- **L10**: Create from file operations + workflow integration (lighter content)
- **L11**: Adapt lectures_bkp/L11 clinical research to practical applications

**STATUS**: Updated plan ready for review with load reduction focus