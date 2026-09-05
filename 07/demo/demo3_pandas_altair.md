# Demo 3: Altair Declarative Charts

## Learning Objectives

- Build a chart with `Chart(data).mark_*().encode(...)`
- State the data type for each encoded field
- Add useful tooltips and basic zoom/pan interaction
- Compose two readable charts without hiding their essential comparison

## Setup

```python
import altair as alt
import pandas as pd

sessions = pd.DataFrame({
    'activities_completed': [1, 2, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7],
    'reflection_score': [52, 57, 61, 65, 69, 72, 55, 61, 66, 71, 75, 79],
    'pathway': ['Independent'] * 6 + ['Guided'] * 6,
})
```

Each row is one prepared learning session. `activities_completed` and
`reflection_score` are quantitative; `pathway` is nominal. The visible pattern
is descriptive of these prepared rows only.

## Chart → mark → typed encodings

Start with the table, choose a point mark, then make every mapping explicit.
`Q` means quantitative and `N` means nominal.

```python
base_scatter = alt.Chart(sessions).mark_point(filled=True, size=90).encode(
    x=alt.X('activities_completed:Q', title='Activities completed (count)'),
    y=alt.Y('reflection_score:Q', title='Reflection score (points)'),
    color=alt.Color('pathway:N', title='Pathway'),
    shape=alt.Shape('pathway:N', title='Pathway'),
).properties(
    width=360,
    height=260,
    title='Prepared sessions: reflection score and activity count',
)

base_scatter
```

The code states a claim about the display, not a causal claim: the points let
us compare the two prepared pathways at their observed activity counts.

## Tooltips and interaction

Tooltips can reveal the value behind a visible mark. Interaction supports
inspection, but the title, axes, legend, and main comparison remain visible
without hover.

```python
interactive_scatter = base_scatter.encode(
    tooltip=[
        alt.Tooltip('activities_completed:Q', title='Activities completed'),
        alt.Tooltip('reflection_score:Q', title='Reflection score'),
        alt.Tooltip('pathway:N', title='Pathway'),
    ]
).interactive()

interactive_scatter
```

## Composition

A quantitative measure can also be summarized by a nominal category. Here,
`N` declares the pathway categories and `Q` declares the quantitative mean.
This is a new summarized view, so say what each bar represents before
comparing it with the session-level scatterplot.

```python
mean_score = alt.Chart(sessions).mark_bar().encode(
    x=alt.X('pathway:N', title='Pathway'),
    y=alt.Y('mean(reflection_score):Q', title='Mean reflection score (points)'),
    color=alt.Color('pathway:N', legend=None),
    tooltip=[
        alt.Tooltip('pathway:N', title='Pathway'),
        alt.Tooltip('mean(reflection_score):Q', title='Mean reflection score', format='.1f'),
    ],
).properties(
    width=260,
    height=260,
    title='Mean score in prepared sessions',
)

alt.hconcat(interactive_scatter, mean_score)
```

The left view has one point per session; the right view has one bar per
pathway. The bar chart uses a zero baseline because bar length encodes
magnitude. A shared browser chart is still incomplete without a text
alternative that names both views, their units, the comparison, and the
prepared-data limitation.

## Optional extensions

Faceting, linked selections, transforms, dashboards, and other browser tools
are outside this core demo. Use them only after the chart → mark → typed
encodings path and the visible-context requirements are secure.

## Key Takeaways

1. Altair makes the data, mark, and encodings explicit.
2. Type shorthands (`Q`, `N`, `O`, and `T`) document how a field is used.
3. Tooltips and interaction supplement rather than replace visible context.
4. Composition compares distinct views only when each view's unit and grain
   are clear.
