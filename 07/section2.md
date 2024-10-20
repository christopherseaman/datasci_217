---

marp: true
theme: sqrl
paginate: true
class: invert
---

# Lecture 07: Data Visualization for Health Data Science

## Section 2: Design Principles for Effective Visualization

---

# Introduction to Design Principles

- **Importance of Good Design**
  - Enhances comprehension and retention
  - Communicates data accurately and ethically
  - *Because nobody wants their data presentation to be the next plot twist nobody asked for!*

- **Based on Works of Edward Tufte and Claus O. Wilke**
  - Focus on clarity, precision, and efficiency in data presentation
  - *Think of them as the Jedi Masters of data visualization!*

---

# Key Concepts

## Simplify and Focus

- **Eliminate Non-Essential Elements**
  - Remove unnecessary gridlines, backgrounds, and decorations
  - *Less is more—channel your inner minimalist!*

- **Highlight Key Data**
  - Use visual emphasis (bolding, color) to draw attention to important information
  - *Make your data the Beyoncé of the visualization—let it steal the spotlight!*

---

## Definitions

### Data-Ink Ratio

- **Definition**: The proportion of ink used to present actual data compared to the total ink used in the graphic

- **Goal**: Maximize data-ink ratio by reducing non-essential elements

- *In other words, don't let your chart wear a heavy coat in summer—keep it light!*

#### Example:

- **High Data-Ink Ratio**:

  ![High Data-Ink Ratio](media/high_data_ink_ratio.png)

  #FIXME-{{Insert an example of a clean, simple chart with a high data-ink ratio}}

- **Low Data-Ink Ratio**:

  ![Low Data-Ink Ratio](media/low_data_ink_ratio.png)

  #FIXME-{{Insert an example of a cluttered chart with a low data-ink ratio}}

---

### Chartjunk

- **Definition**: Unnecessary or distracting decorations in data visualizations that do not improve the viewer's understanding

- **Includes**:
  - Excessive colors or patterns
  - 3D effects that distort data
  - Decorative images or clip art

- *Chartjunk is like adding pineapple to pizza—it might seem like a good idea to some, but it often just complicates things!*

#### Example:

- **Avoid Chartjunk**:

  ![Avoid Chartjunk](media/avoid_chartjunk.png)

  #FIXME-{{Provide side-by-side examples of a chart with chartjunk and a cleaned-up version}}

---

# Good vs. Bad Practices

## Side-by-Side Comparison

### Good Example

- **Features**:
  - Clear labels and titles
  - Minimalist design
  - Accurate representation of data

  ![Good Visualization](media/good_visualization.png)

  #FIXME-{{Insert a well-designed chart here}}

---

### Bad Example

- **Issues**:
  - Distracting colors
  - Misleading scales
  - Unnecessary 3D effects

  ![Bad Visualization](media/bad_visualization.png)

  #FIXME-{{Insert a poorly designed chart here}}

---

# Ethical Representation of Data

- **Avoid Misleading Visuals**
  - Start axes at zero when appropriate to prevent exaggeration
    - *Unless you're trying to create the next big conspiracy theory, keep it real!*
  - Use consistent scales across related visuals
    - *Inconsistency is only cool in plot twists, not in plots!*

- **Accurate Data Representation**
  - Do not manipulate visuals to mislead or bias the audience
    - *No one likes a data manipulator—trust is key!*
  - Clearly indicate any data exclusions or manipulations
    - *Transparency isn't just for windows!*

---

# Use of Color

- **Functional Use**
  - Differentiate data categories meaningfully
    - *Think Power Rangers—each color represents a different hero!*
  - Use color to highlight important data points

- **Accessibility**
  - Use colorblind-friendly palettes (e.g., Viridis, Cividis)
    - *Because everyone should be able to appreciate your data masterpiece!*
  - Ensure sufficient contrast between colors

---

# Specific Considerations in Health Data Visualization

## Privacy and Confidentiality

- **Aggregate Data**
  - Use aggregated or de-identified data to protect patient privacy
    - *Doctor's code: First, do no harm—even in data!*

- **Geographic Detail**
  - Be cautious with maps; avoid pinpointing exact locations if not necessary
    - *We're not playing 'Where's Waldo' with patient data!*

## Ethical Representation

- **Sensitive Topics**
  - Present data on sensitive health issues with care and respect
    - *Handle with care—like it's hot coffee!*

- **Cultural Sensitivity**
  - Use symbols and icons that are culturally appropriate
    - *Avoid unintentional faux pas—it's not just about the data, but about respect!*

---

# Interactive Activity

## Critiquing a Visualization

- **Exercise**:
  - Examine the following chart and identify areas for improvement

  ![Sample Visualization to Critique](media/sample_visualization_to_critique.png)

  #FIXME-{{Provide a sample chart with obvious design flaws for students to critique}}

- **Consider**:
  - Clarity of labels and titles
  - Use of color and chart elements
  - Ethical representation of data

- *Channel your inner Simon Cowell—be constructively critical!*

---

# Applying Principles in Practice

- **Critical Evaluation**
  - Assess visualizations for simplicity and clarity
    - *Is your chart more complicated than the plot of 'Inception'? Simplify!*

- **Iterative Refinement**
  - Make incremental improvements based on feedback
    - *Remember, even Tony Stark upgraded his suit—improve iteratively!*

- **Audience Consideration**
  - Tailor visuals to the needs and expectations of the audience
    - *Know your audience—don't serve sushi at a pizza party!*

---

# Tell a Story

- **Narrative Flow**
  - Guide the viewer through the data logically
    - *Be the J.K. Rowling of data—craft an engaging narrative!*

- **Annotations and Highlights**
  - Use text and markers to emphasize key points
    - *Because sometimes even data needs a little spotlight!*

- **Provide Context**
  - Background information aids interpretation
    - *Set the scene before the action begins!*

---

# Takeaways

- **Simplicity is Key**
  - Strive for clarity and avoid unnecessary complexity
    - *Don't turn your chart into a 'Where's Waldo' page!*

- **Ethics Matter**
  - Represent data honestly and responsibly
    - *With great data comes great responsibility!*

- **Design with Purpose**
  - Every element in your visualization should serve a function
    - *No more, no less—just like a perfectly balanced equation!*

---

# Additional Resources

- **"The Visual Display of Quantitative Information"** by Edward Tufte
- **"Fundamentals of Data Visualization"** by Claus O. Wilke
- **Color Brewer 2**: [colorbrewer2.org](http://colorbrewer2.org/) for choosing colorblind-friendly palettes
- *Because lifelong learning is the real infinite game!*

---
