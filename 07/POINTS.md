# Data Visualization: From Exploration to Communication

Instructor cues for the lecture. The canonical explanations and executable
examples are in [README.md](README.md); optional advanced material is in
[BONUS.md](BONUS.md).

## Visualization contract

- Ask for the question, audience, intended descriptive claim, unit, plotting-table grain, variable types, and encoding choices before choosing a chart.
- Distinguish exploratory displays (inspect patterns and data quality) from explanatory displays (communicate one supported finding).
- Position on a common scale is usually easier to compare than area or decoration. Use color with a redundant cue when category identity matters.

## Tufte and honest communication

- Remove chartjunk, keep useful data-ink, label directly where practical, and preserve context needed to interpret the comparison.
- Bar/length encodings generally need a zero baseline because length represents magnitude. Line axes depend on the question and range; zero is not universal. Clearly label any truncated range or break.
- A connecting line only implies continuity for an ordered variable. A trend claim needs an appropriate fitted model or careful descriptive wording.
- Accessibility cues: readable labels and contrast, color-safe palettes, redundant encodings, and a concise text alternative.

## Visualization ecosystem

- matplotlib: figure/axes control and publication customization.
- pandas: quick DataFrame exploration; seaborn: statistical graphics and distributions.
- Altair/plotnine: declarative grammar; Bokeh/Plotly: browser interactivity and dashboards.
- Tool choice follows the visualization contract; interactivity supplements, rather than hides, essential values and context.

## matplotlib

- `plt.subplots()` returns a Figure and Axes; use Axes methods for titles, labels, limits, grids, legends, and marks. `fig.savefig()` and `plt.show()` complete a display workflow.
- Demonstrate a small multi-panel figure, then customize title, labels, limits, color, marker, line style, and restrained gridlines.
- Name visual variables meaningfully; never rely on color alone. Mention that `figsize`, `tight_layout`, and explicit labels improve presentation output.

## LIVE DEMO!

## pandas quick exploration

- `df.plot()` defaults to a line view; use `kind='bar'`, `'hist'`, `'scatter'`, or `'box'` when the data and question support those marks.
- Show `subplots`, `figsize`, titles/labels, legend, and grid options on a small DataFrame. Check index meaning before treating it as x.

## seaborn statistical graphics

- Seaborn accepts long-form data and returns familiar axes for further matplotlib customization. Demonstrate scatter, box, histogram, and a grouping variable.
- Optional distribution cues: pairplot, jointplot, violin/strip plots, and KDE. Explain what each mark summarizes and its assumptions; do not imply causation from a visual pattern.

## LIVE DEMO!

## Optional survey: modern libraries

- Altair, plotnine, Bokeh, and Plotly offer alternate grammars or delivery media. Keep labels, visible context, honest encodings, and text alternatives.
- Refer to [README.md](README.md) for the ecosystem comparison and [BONUS.md](BONUS.md) for extended API examples; this survey is not assessed.

## LIVE DEMO!
