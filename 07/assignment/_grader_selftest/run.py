# /// script
# requires-python = "==3.12.13"
# dependencies = [
#   "numpy==2.0.2",
#   "pandas==3.0.5",
#   "matplotlib==3.11.1",
#   "seaborn==0.13.2",
#   "nbclient==0.10.2",
#   "nbformat==5.10.4",
#   "ipykernel==6.29.5",
#   "Pillow==12.3.0",
# ]
# ///

"""Adversarial release harness for the Assignment 07 grader contract."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

from classroom50_grader import (
    OUTPUT_NAMES,
    REQUIRED_CONTEXT_ENV,
    TEST_SPECS,
    _execute_notebook,
    grade_submission,
)


ASSIGNMENT_DIR = Path(__file__).resolve().parent.parent
UTC_DATETIME = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")

SOLUTION_CELLS = {
    "a07-load": r'''FORMAT_DTYPES = {'format': 'string', 'stage': 'string', 'completion_percent': 'int64'}
SESSION_DTYPES = {'session_id': 'string', 'pathway': 'string', 'activities_completed': 'int64', 'reflection_score': 'int64'}
CHECKPOINT_DTYPES = {'pathway': 'string', 'checkpoint_number': 'int64', 'completion_percent': 'int64'}

format_completion = pd.read_csv(DATA_DIR / 'format_completion.csv', dtype=FORMAT_DTYPES)
session_observations = pd.read_csv(DATA_DIR / 'session_observations.csv', dtype=SESSION_DTYPES)
pathway_checkpoints = pd.read_csv(DATA_DIR / 'pathway_checkpoints.csv', dtype=CHECKPOINT_DTYPES)
assert format_completion.shape == (4, 3) and format_completion.columns.tolist() == list(FORMAT_DTYPES)
assert session_observations.shape == (12, 4) and session_observations.columns.tolist() == list(SESSION_DTYPES)
assert pathway_checkpoints.shape == (8, 3) and pathway_checkpoints.columns.tolist() == list(CHECKPOINT_DTYPES)
format_completion_original = format_completion.copy(deep=True)
session_observations_original = session_observations.copy(deep=True)
pathway_checkpoints_original = pathway_checkpoints.copy(deep=True)
''',
    "a07-task1-contract": """## Task 1 contract

I am asking how reflection score varies with activities completed within the twelve prepared sessions and how the visible pattern differs by pathway. One row and point is one synthetic learning session. These complete, synthetic, nonrandom rows can describe only themselves: they cannot establish cause, uncertainty, prediction, a population effect, or a general relationship.
""",
    "a07-task1-evidence": r'''exploration_question = 'Within the twelve prepared sessions, how does reflection score vary with activities completed for each pathway?'
exploration_observation = 'Within these twelve rows, reflection scores generally appear higher at larger activity counts in both pathways, with Facilitated points often above Independent points at comparable counts.'
exploration_limitation = 'This small prepared descriptive fixture is not a random population sample, so the visible pattern cannot establish cause or be generalized beyond these twelve rows.'
exploration_grain = 'one row per synthetic learning session'
exploration_roles = {
    'session_id': 'identifier',
    'pathway': 'categorical',
    'activities_completed': 'quantitative',
    'reflection_score': 'quantitative',
}
assert all(isinstance(value, str) and value.strip() for value in [exploration_question, exploration_observation, exploration_limitation])
''',
    "a07-explore-function": r'''def build_exploratory_chart(session_table, pathway_order):
    """Return one exploratory Figure and Axes without saving a file."""
    required = ['session_id', 'pathway', 'activities_completed', 'reflection_score']
    missing = [column for column in required if column not in session_table.columns]
    if missing:
        raise ValueError(f'missing session columns: {missing}')
    labels = list(pathway_order)
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError('pathway_order must contain two distinct labels')
    prepared = session_table.copy(deep=True)
    if prepared[required].isna().any().any():
        raise ValueError('session rows must be complete')
    if set(prepared['pathway'].tolist()) != set(labels):
        raise ValueError('session pathways must exactly match pathway_order')
    if len(set(prepared['session_id'].tolist())) != len(prepared):
        raise ValueError('session_id must be unique')
    if not pd.api.types.is_integer_dtype(prepared['activities_completed']) or not pd.api.types.is_integer_dtype(prepared['reflection_score']):
        raise ValueError('session quantitative fields must be integers')
    figure, axes = plt.subplots(figsize=(7.2, 4.6))
    sns.scatterplot(
        data=prepared,
        x='activities_completed',
        y='reflection_score',
        hue='pathway',
        style='pathway',
        hue_order=labels,
        style_order=labels,
        palette={labels[0]: BLUE, labels[1]: ORANGE},
        markers={labels[0]: 'o', labels[1]: 's'},
        s=75,
        ax=axes,
    )
    axes.set_xlabel('Activities completed (count)')
    axes.set_ylabel('Reflection score (points)')
    axes.set_title('Exploratory view of activities completed and reflection score')
    axes.legend(title='Pathway')
    return figure, axes
''',
    "a07-explore-run": r'''exploratory_order = ['Independent', 'Facilitated']
exploratory_snapshot = session_observations.copy(deep=True)
exploratory_figure, exploratory_axes = build_exploratory_chart(session_observations, exploratory_order)
assert exploratory_figure.axes == [exploratory_axes]
assert len(exploratory_axes.collections[0].get_offsets()) == 12
exploratory_colors = {tuple(np.round(color, 6)) for color in exploratory_axes.collections[0].get_facecolors()}
assert exploratory_colors == {tuple(np.round(matplotlib.colors.to_rgba(BLUE), 6)), tuple(np.round(matplotlib.colors.to_rgba(ORANGE), 6))}
exploratory_marker_geometry = set()
for marker_path in exploratory_axes.collections[0].get_paths():
    exploratory_marker_geometry.add(tuple(tuple(np.round(vertex, 6)) for vertex in marker_path.vertices))
assert len(exploratory_marker_geometry) == 2
assert exploratory_axes.get_xlabel() == 'Activities completed (count)'
assert exploratory_axes.get_ylabel() == 'Reflection score (points)'
assert exploratory_axes.get_title() == 'Exploratory view of activities completed and reflection score'
assert exploratory_axes.get_legend().get_title().get_text() == 'Pathway'
assert session_observations.equals(exploratory_snapshot)
display_figure(exploratory_figure)
''',
    "a07-task1-reflection": """### Task 1 reflection

The one-session grain means every point is one prepared row, not an average or population estimate. Activities and reflection use position because they are quantitative; pathway uses color plus marker shape because it is categorical, while session ID only identifies rows. The pattern I describe is therefore restricted to visible marks in these twelve rows and is exploratory, not evidence that activities or pathway caused a score.
""",
    "a07-task2-critique": """### Critique

The title states a cause that four prepared descriptive values cannot support. Starting the percentage axis at 76 exaggerates bar-length differences. Omitting the percentage unit makes magnitude ambiguous. Color alone can make format identity unavailable to readers who cannot distinguish the hues. The yellow canvas and heavy two-axis grid compete with the bars. The redesign must retain the four values while using a descriptive title, zero baseline, explicit unit, hatch redundancy, and restrained decoration.
""",
    "a07-critique-evidence": r'''critique_entries = [
    {'category': 'unsupported claim', 'problem': 'The causal title exceeds what four prepared descriptive rows can show.', 'repair': 'Use a bounded descriptive title that names only the comparison.'},
    {'category': 'truncated baseline', 'problem': 'A 76 percent baseline exaggerates small differences in bar lengths.', 'repair': 'Start the quantitative bar-length scale at zero.'},
    {'category': 'missing unit', 'problem': 'The unlabeled y-axis leaves the prepared percentage magnitude ambiguous.', 'repair': 'Label the axis as prepared completion percent.'},
    {'category': 'color-only encoding', 'problem': 'Hue alone may not preserve format identity for every reader or rendering.', 'repair': 'Add distinct hatches as a redundant format cue.'},
    {'category': 'distracting decoration', 'problem': 'The yellow canvas and heavy grids compete with the data marks.', 'repair': 'Use a neutral background, restrained grid treatment, and fewer spines.'},
]
assert len(critique_entries) == 5
''',
    "a07-redesign-function": r'''def build_critique_redesign(summary_table, format_order, stage_order):
    """Return a repaired four-bar Figure and Axes for any valid prepared 2-by-2 table."""
    required = ['format', 'stage', 'completion_percent']
    missing = [column for column in required if column not in summary_table.columns]
    if missing:
        raise ValueError(f'missing summary columns: {missing}')
    format_labels = list(format_order)
    stage_labels = list(stage_order)
    if len(format_labels) != 2 or len(set(format_labels)) != 2:
        raise ValueError('format_order must contain two distinct labels')
    if len(stage_labels) != 2 or len(set(stage_labels)) != 2:
        raise ValueError('stage_order must contain two distinct labels')
    prepared = summary_table.copy(deep=True)
    if prepared[required].isna().any().any() or not pd.api.types.is_integer_dtype(prepared['completion_percent']):
        raise ValueError('summary rows must be complete with integer percentages')
    if set(prepared['format'].tolist()) != set(format_labels) or set(prepared['stage'].tolist()) != set(stage_labels):
        raise ValueError('summary labels must exactly match caller orders')
    pairs = list(zip(prepared['format'].tolist(), prepared['stage'].tolist()))
    expected_pairs = {(format_label, stage_label) for format_label in format_labels for stage_label in stage_labels}
    if len(pairs) != 4 or set(pairs) != expected_pairs:
        raise ValueError('summary table must have one row per requested format and stage')
    figure, axes = plt.subplots(figsize=(7.4, 4.4))
    positions = np.arange(len(stage_labels))
    width = 0.36
    hatches = ['//', '\\\\']
    for format_index, format_label in enumerate(format_labels):
        values = []
        for stage_label in stage_labels:
            matched = prepared.loc[
                (prepared['format'] == format_label) & (prepared['stage'] == stage_label),
                'completion_percent',
            ]
            values.append(int(matched.iloc[0]))
        container = axes.bar(
            positions + (format_index - 0.5) * width,
            values,
            width,
            label=format_label,
            color=COURSE_COLORS[format_index],
            hatch=hatches[format_index],
        )
        axes.bar_label(container, fmt='%d%%', padding=3)
    axes.set_xticks(positions, stage_labels)
    axes.set_xlabel('Stage')
    axes.set_ylabel('Prepared completion (%)')
    axes.set_title('Prepared completion by delivery format and stage')
    axes.set_ylim(bottom=0)
    axes.grid(False)
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    axes.legend(title='Delivery format', loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False)
    return figure, axes
''',
    "a07-redesign-run": r'''redesign_format_order = ['Recorded', 'Live']
redesign_stage_order = ['Start', 'Finish']
redesign_snapshot = format_completion.copy(deep=True)
redesign_figure, redesign_axes = build_critique_redesign(format_completion, redesign_format_order, redesign_stage_order)
assert [int(patch.get_height()) for patch in redesign_axes.patches] == [81, 77, 82, 80]
assert redesign_axes.get_ylim()[0] == 0 and len(redesign_axes.patches) == 4
assert redesign_axes.get_xlabel() == 'Stage' and redesign_axes.get_ylabel() == 'Prepared completion (%)'
assert redesign_axes.get_title() == 'Prepared completion by delivery format and stage'
assert {patch.get_hatch() for patch in redesign_axes.patches} == {'//', '\\\\'}
assert {text.get_text() for text in redesign_axes.texts} == {'81%', '77%', '82%', '80%'}
redesign_legend = redesign_axes.get_legend()
assert redesign_legend.get_title().get_text() == 'Delivery format' and not redesign_legend.get_frame_on()
assert format_completion.equals(redesign_snapshot)
redesign_figure.set_size_inches(7.4, 4.4)
redesign_figure.savefig(CRITIQUE_IMAGE_PATH, dpi=150, bbox_inches='tight')
display_figure(redesign_figure)
''',
    "a07-task3-contract": """## Task 3 contract

My question asks how prepared completion changes across four ordered checkpoints for Independent and Facilitated pathways. The audience is a learning-support coordinator who will use the descriptive comparison to decide which checkpoint experiences need follow-up. The intended claim is bounded to the prepared rows: Facilitated begins one percentage point lower and finishes nine points higher. The unit is prepared completion percent, and one row/position is one pathway at one checkpoint. Pathway is categorical, checkpoint is ordered, and completion is quantitative. Lines connect a shared ordered sequence; they do not show that pathway caused the differences.
""",
    "a07-final-contract-values": r'''question = 'How does prepared completion compare across four checkpoints for the Independent and Facilitated pathways?'
audience = 'A learning-support coordinator who will use the comparison to identify checkpoint experiences for follow-up.'
intended_claim = 'In the prepared summary, Facilitated begins one percentage point lower than Independent and finishes nine points higher.'
displayed_unit = 'prepared completion percent'
plotting_grain = 'one row per pathway and checkpoint'
variable_roles = {
    'pathway': 'categorical',
    'checkpoint_number': 'ordered',
    'completion_percent': 'quantitative',
}
pathway_order = ['Independent', 'Facilitated']
assert all(isinstance(value, str) and value.strip() for value in [question, audience, intended_claim])
''',
    "a07-supporting-data": r'''supporting_columns = ['pathway', 'checkpoint_number', 'completion_percent']
explanatory_supporting_data = pathway_checkpoints.loc[:, supporting_columns].copy(deep=True)
assert explanatory_supporting_data.shape == (8, 3) and explanatory_supporting_data.columns.tolist() == supporting_columns
supporting_pairs = list(zip(explanatory_supporting_data['pathway'].tolist(), explanatory_supporting_data['checkpoint_number'].tolist()))
assert len(set(supporting_pairs)) == 8
assert list(dict.fromkeys(explanatory_supporting_data['pathway'].tolist())) == pathway_order
for pathway_label in pathway_order:
    assert explanatory_supporting_data.loc[explanatory_supporting_data['pathway'] == pathway_label, 'checkpoint_number'].tolist() == [1, 2, 3, 4]
assert pathway_checkpoints.equals(pathway_checkpoints_original)
explanatory_supporting_data.to_csv(SUPPORTING_DATA_PATH, index=False, encoding='utf-8', lineterminator='\n')
supporting_round_trip = pd.read_csv(SUPPORTING_DATA_PATH, dtype=CHECKPOINT_DTYPES)
pd.testing.assert_frame_equal(supporting_round_trip, explanatory_supporting_data)
assert supporting_round_trip.dtypes.equals(explanatory_supporting_data.dtypes)
assert SUPPORTING_DATA_PATH.read_bytes() == fixture_bytes['pathway_checkpoints.csv']
assert sha256_bytes(SUPPORTING_DATA_PATH.read_bytes()) == EXPECTED_FILES['pathway_checkpoints.csv']['sha256']
''',
    "a07-explanatory-function": r'''def build_explanatory_chart(checkpoint_table, pathway_order):
    """Return a dynamic two-pathway Figure, Axes, and final-gap annotation."""
    required = ['pathway', 'checkpoint_number', 'completion_percent']
    missing = [column for column in required if column not in checkpoint_table.columns]
    if missing:
        raise ValueError(f'missing checkpoint columns: {missing}')
    labels = list(pathway_order)
    if len(labels) != 2 or len(set(labels)) != 2:
        raise ValueError('pathway_order must contain two distinct labels')
    prepared = checkpoint_table.copy(deep=True)
    if prepared[required].isna().any().any():
        raise ValueError('checkpoint rows must be complete')
    if not pd.api.types.is_integer_dtype(prepared['checkpoint_number']) or not pd.api.types.is_integer_dtype(prepared['completion_percent']):
        raise ValueError('checkpoint and completion values must be integers')
    if set(prepared['pathway'].tolist()) != set(labels):
        raise ValueError('checkpoint pathways must exactly match pathway_order')
    pairs = list(zip(prepared['pathway'].tolist(), prepared['checkpoint_number'].tolist()))
    if len(set(pairs)) != len(prepared):
        raise ValueError('pathway/checkpoint grain must be unique')
    paths = []
    checkpoint_sets = []
    for label in labels:
        path = prepared.loc[prepared['pathway'] == label, required].sort_values('checkpoint_number')
        paths.append(path)
        checkpoint_sets.append(path['checkpoint_number'].tolist())
    if len(checkpoint_sets[0]) < 2 or checkpoint_sets[0] != checkpoint_sets[1]:
        raise ValueError('both pathways need the same set of at least two checkpoints')
    figure, axes = plt.subplots(figsize=(8, 4.8))
    markers = ['o', 's']
    line_styles = ['-', '--']
    for index, label in enumerate(labels):
        axes.plot(
            paths[index]['checkpoint_number'],
            paths[index]['completion_percent'],
            label=label,
            color=COURSE_COLORS[index],
            marker=markers[index],
            linestyle=line_styles[index],
            linewidth=2.2,
            markersize=7,
        )
    final_checkpoint = checkpoint_sets[0][-1]
    final_values = [int(path['completion_percent'].iloc[-1]) for path in paths]
    final_gap = abs(final_values[0] - final_values[1])
    higher_index = 1 if final_values[1] >= final_values[0] else 0
    if final_values[0] == final_values[1]:
        title = f'Both pathways finish equally in the prepared {len(checkpoint_sets[0])}-checkpoint summary'
    else:
        title = f'{labels[higher_index]} finishes higher in the prepared {len(checkpoint_sets[0])}-checkpoint summary'
    axes.set_xlabel('Checkpoint')
    axes.set_ylabel('Prepared completion (%)')
    axes.set_title(title)
    axes.set_xticks(checkpoint_sets[0])
    axes.legend(title='Pathway')
    axes.spines['top'].set_visible(False)
    axes.spines['right'].set_visible(False)
    annotation = axes.annotate(
        f'Checkpoint {final_checkpoint} observed gap: {final_gap} percentage points',
        xy=(final_checkpoint, final_values[higher_index]),
        xytext=(-18, -44),
        textcoords='offset points',
        ha='right',
        va='top',
        color=COURSE_COLORS[higher_index],
        bbox={'boxstyle': 'round,pad=0.25', 'facecolor': 'white', 'edgecolor': '#777777'},
        arrowprops={'arrowstyle': '->', 'color': COURSE_COLORS[higher_index]},
    )
    return figure, axes, annotation
''',
    "a07-explanatory-run": r'''explanatory_snapshot = explanatory_supporting_data.copy(deep=True)
explanatory_figure, explanatory_axes, final_gap_annotation = build_explanatory_chart(explanatory_supporting_data, pathway_order)
assert explanatory_figure.axes == [explanatory_axes]
assert [line.get_xdata().tolist() for line in explanatory_axes.lines] == [[1, 2, 3, 4], [1, 2, 3, 4]]
assert [line.get_ydata().tolist() for line in explanatory_axes.lines] == [[58, 63, 67, 70], [57, 65, 72, 79]]
assert explanatory_axes.get_title() == 'Facilitated finishes higher in the prepared 4-checkpoint summary'
assert explanatory_axes.get_xlabel() == 'Checkpoint' and explanatory_axes.get_ylabel() == 'Prepared completion (%)'
assert final_gap_annotation.get_text() == 'Checkpoint 4 observed gap: 9 percentage points'
assert tuple(final_gap_annotation.xy) == (4, 79)
assert explanatory_axes.get_legend().get_title().get_text() == 'Pathway'
assert explanatory_supporting_data.equals(explanatory_snapshot)
explanatory_figure.savefig(EXPLANATORY_IMAGE_PATH, dpi=150, bbox_inches='tight')
display_figure(explanatory_figure)
''',
    "a07-evidence-export": r'''explanatory_text_alternative = ('The line chart plots checkpoint number on the x-axis and prepared completion percent on the y-axis for Independent and Facilitated pathways. '
    'Independent rises from 58 percent at checkpoint 1 to 70 percent at checkpoint 4, while Facilitated rises from 57 percent to 79 percent, leaving an observed nine-point final gap. '
    'These prepared descriptive rows support comparison only and do not establish that pathway caused the difference.')
visualization_evidence = {
    'schema': 'datasci217/a07-visualization-evidence/v1',
    'question': question,
    'audience': audience,
    'intended_claim': intended_claim,
    'displayed_unit': displayed_unit,
    'grain': plotting_grain,
    'variable_roles': variable_roles,
    'exploration': {
        'question': exploration_question,
        'grain': exploration_grain,
        'variable_roles': exploration_roles,
        'observation': exploration_observation,
        'limitation': exploration_limitation,
    },
    'critique': critique_entries,
    'text_alternative': explanatory_text_alternative,
}
with EVIDENCE_JSON_PATH.open('w', encoding='utf-8', newline='\n') as evidence_stream:
    json.dump(visualization_evidence, evidence_stream, ensure_ascii=False, indent=2)
    evidence_stream.write('\n')
with TEXT_ALTERNATIVE_PATH.open('w', encoding='utf-8', newline='\n') as text_stream:
    text_stream.write(explanatory_text_alternative + '\n')
assert json.loads(EVIDENCE_JSON_PATH.read_text(encoding='utf-8')) == visualization_evidence
assert TEXT_ALTERNATIVE_PATH.read_bytes() == (explanatory_text_alternative + '\n').encode('utf-8')
''',
    "a07-visual-review": """## Final human visual review

1. The title and annotation answer the coordinator's checkpoint comparison without causal language.
2. Two lines connect the same four ordered checkpoint positions, making direction visible.
3. The exact disclosed CSV values, labeled percentage axis, and descriptive title preserve context and scale.
4. Color is reinforced by circle/solid and square/dashed encodings, with direct units and a legend.
5. The arrow identifies the higher final mark and labels the observed nine-point gap without claiming cause.
6. The text alternative names the chart, axes, both pathways, endpoints, gap, and causal limitation.
7. I inspected the saved PNG for unclipped title, labels, legend, annotation, readable type, and nonoverlapping marks.
8. The small prepared fixture still cannot support population inference, prediction, or a causal pathway effect.
""",
}


def _copy_template(destination: Path) -> None:
    def ignore(_directory: str, names: list[str]) -> set[str]:
        return {"_grader_selftest", ".venv", "__pycache__", ".ipynb_checkpoints", ".pytest_cache", "result.json"}.intersection(names)
    shutil.copytree(ASSIGNMENT_DIR, destination, ignore=ignore)


def _notebook(root: Path) -> dict:
    return json.loads((root / "assignment.ipynb").read_text(encoding="utf-8"))


def _write_notebook(root: Path, notebook: dict) -> None:
    (root / "assignment.ipynb").write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n", encoding="utf-8", newline="\n")


def _replace_cell(root: Path, cell_id: str, source: str) -> None:
    notebook = _notebook(root)
    cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
    cell["source"] = source
    cell["execution_count"] = None if cell["cell_type"] == "code" else cell.get("execution_count")
    if cell["cell_type"] == "code":
        cell["outputs"] = []
    _write_notebook(root, notebook)


def _complete_sources(root: Path) -> None:
    notebook = _notebook(root)
    by_id = {cell["id"]: cell for cell in notebook["cells"]}
    for cell_id, source in SOLUTION_CELLS.items():
        by_id[cell_id]["source"] = source
        if by_id[cell_id]["cell_type"] == "code":
            by_id[cell_id]["execution_count"] = None
            by_id[cell_id]["outputs"] = []
    _write_notebook(root, notebook)


def materialize_correct(root: Path) -> None:
    _copy_template(root)
    _complete_sources(root)
    _execute_notebook(root, root)


@contextmanager
def grader_context():
    old = os.environ.copy()
    values = {
        "classroom": "datasci-217-2026",
        "assignment": "assignment-07",
        "submission": "submit/2026-07-19T12-00-00Z-a07c0de",
        "commit": "https://git.example.edu/commit/a07-correct",
        "release": "https://git.example.edu/release/a07-v1",
        "review": "https://review.example.edu/a07/student-007",
    }
    for key, environment_name in REQUIRED_CONTEXT_ENV.items():
        os.environ[environment_name] = values[key]
    os.environ["REVIEW_URL"] = values["review"]
    try:
        yield values
    finally:
        os.environ.clear()
        os.environ.update(old)


def _assert_result(result: dict, expected_context: dict[str, str], success: bool) -> None:
    assert list(result) == [
        "schema", "classroom", "assignment", "submission", "commit", "release",
        "review", "datetime", "score", "max-score", "tests",
    ]
    assert result["schema"] == "classroom50/result/v1"
    assert {key: result[key] for key in expected_context} == expected_context
    assert UTC_DATETIME.fullmatch(result["datetime"])
    assert result["max-score"] == 80
    assert [test["max-score"] for test in result["tests"]] == [10, 15, 25, 25, 5]
    assert [test["test-name"] for test in result["tests"]] == [name for name, _ in TEST_SPECS]
    assert all(list(test) == ["test-name", "passed", "score", "max-score"] for test in result["tests"])
    assert all(isinstance(test["passed"], bool) for test in result["tests"])
    assert result["score"] == sum(test["score"] for test in result["tests"])
    if success:
        assert result["score"] == 80 and all(test["passed"] for test in result["tests"])
    else:
        assert result["score"] < 80 and not all(test["passed"] for test in result["tests"])


def _grade_case(root: Path, expected_context: dict[str, str], success: bool) -> dict:
    result = grade_submission(root)
    _assert_result(result, expected_context, success)
    return result


def _copy_case(source: Path, destination: Path) -> None:
    shutil.copytree(source, destination)


def _static_mutation_cases(correct: Path, cases: Path):
    mutations = []
    for cell_id in ("a07-header", "a07-setup", "a07-terms-data", "a07-task2-context", "a07-supplied-flawed", "a07-final-verify"):
        def mutate_cell(root: Path, target=cell_id):
            notebook = _notebook(root)
            next(cell for cell in notebook["cells"] if cell["id"] == target)["source"] += "\nchanged"
            _write_notebook(root, notebook)
        mutations.append((f"protected-cell-{cell_id}", mutate_cell))
    for relative in (
        ".python-version", "requirements.txt", ".gitignore", "README.md",
        "PLATFORM_CHECK.md", "check_assignment.py", "data/fixture.json",
        "data/format_completion.csv", "data/pathway_checkpoints.csv",
        "data/session_observations.csv",
    ):
        def mutate_file(root: Path, target=relative):
            with (root / target).open("ab") as stream:
                stream.write(b"changed\n")
        mutations.append(("protected-file-" + relative.replace("/", "-"), mutate_file))
    for name in OUTPUT_NAMES:
        def missing_output(root: Path, target=name):
            (root / "output" / target).unlink()
        mutations.append((f"missing-{name}", missing_output))
        def stale_output(root: Path, target=name):
            (root / "output" / target).write_bytes(b"stale\n")
        mutations.append((f"stale-{name}", stale_output))
    def duplicate_id(root: Path):
        notebook = _notebook(root); notebook["cells"][1]["id"] = notebook["cells"][0]["id"]; _write_notebook(root, notebook)
    def reordered(root: Path):
        notebook = _notebook(root); notebook["cells"][3], notebook["cells"][4] = notebook["cells"][4], notebook["cells"][3]; _write_notebook(root, notebook)
    def wrong_type(root: Path):
        notebook = _notebook(root); notebook["cells"][3]["cell_type"] = "markdown"; notebook["cells"][3].pop("outputs", None); notebook["cells"][3].pop("execution_count", None); _write_notebook(root, notebook)
    def malformed(root: Path):
        (root / "assignment.ipynb").write_text("{bad", encoding="utf-8")
    def fixture_missing(root: Path):
        (root / "data" / "format_completion.csv").unlink()
    def fixture_extra(root: Path):
        (root / "data" / "extra.csv").write_text("x\n1\n", encoding="utf-8")
    def fixture_renamed(root: Path):
        (root / "data" / "session_observations.csv").rename(root / "data" / "sessions.csv")
    def fixture_crlf(root: Path):
        path = root / "data" / "pathway_checkpoints.csv"; path.write_bytes(path.read_bytes().replace(b"\n", b"\r\n"))
    def extra_output(root: Path):
        (root / "output" / "exploratory.png").write_bytes((root / "output" / "pathway_explanatory.png").read_bytes())
    def legacy_output(root: Path):
        (root / "output" / "q1_chart.png").write_bytes((root / "output" / "pathway_explanatory.png").read_bytes())
    def root_sentinel(root: Path):
        (root / "student-extra.txt").write_text("extra\n", encoding="utf-8")
    def stored_success_broken_source(root: Path):
        _replace_cell(root, "a07-explore-function", "def build_exploratory_chart(session_table, pathway_order):\n    return missing_name\n")
    mutations.extend([
        ("duplicate-cell-id", duplicate_id), ("reordered-cells", reordered),
        ("changed-cell-type", wrong_type), ("malformed-notebook", malformed),
        ("fixture-missing", fixture_missing), ("fixture-extra", fixture_extra),
        ("fixture-renamed", fixture_renamed), ("fixture-crlf", fixture_crlf),
        ("extra-exploratory-png", extra_output), ("legacy-png", legacy_output),
        ("unrelated-submission-sentinel", root_sentinel),
        ("stored-output-broken-source", stored_success_broken_source),
    ])
    for index, (name, mutate) in enumerate(mutations):
        destination = cases / f"static-{index:03d}-{name}"
        _copy_case(correct, destination)
        mutate(destination)
        yield name, destination


def _behavior_mutation_cases(correct: Path, cases: Path):
    replacements = [
        ("exploration-dropped-points", "data=prepared,", "data=prepared.iloc[:-1],"),
        ("exploration-one-encoding", "style='pathway',", "style=None,"),
        ("redesign-truncated", "axes.set_ylim(bottom=0)", "axes.set_ylim(50, 90)"),
        ("redesign-wrong-legend", "loc='upper left', bbox_to_anchor=(1.01, 1), frameon=False", "loc='best', frameon=True"),
        ("explanatory-hardcoded-leader", "title = f'{labels[higher_index]} finishes higher", "title = f'Facilitated finishes higher"),
        ("explanatory-wrong-tie-target", "higher_index = 1 if final_values[1] >= final_values[0] else 0", "higher_index = 0 if final_values[0] >= final_values[1] else 1"),
        ("explanatory-input-mutation", "prepared = checkpoint_table.copy(deep=True)", "prepared = checkpoint_table\n    prepared.sort_values('checkpoint_number', inplace=True)"),
    ]
    cell_for = {
        "exploration": "a07-explore-function",
        "redesign": "a07-redesign-function",
        "explanatory": "a07-explanatory-function",
    }
    for index, (name, old, new) in enumerate(replacements):
        destination = cases / f"behavior-{index:03d}-{name}"
        _copy_case(correct, destination)
        family = name.split("-")[0]
        cell_id = cell_for[family]
        notebook = _notebook(destination)
        cell = next(cell for cell in notebook["cells"] if cell["id"] == cell_id)
        assert old in cell["source"], name
        cell["source"] = cell["source"].replace(old, new, 1)
        _write_notebook(destination, notebook)
        yield name, destination


def _scope_mutation_cases(correct: Path, cases: Path):
    fragments = [
        "session_observations.groupby('pathway')",
        "session_observations['reflection_score'].mean()",
        "pd.concat([session_observations, session_observations])",
        "pd.to_datetime('2026-01-01')",
        "sns.regplot(data=session_observations, x='activities_completed', y='reflection_score')",
        "sns.load_dataset('tips')",
        "np.random.default_rng(7)",
        "requests.get('https://example.org')",
        "drive.mount('/content/drive')",
    ]
    for index, fragment in enumerate(fragments):
        destination = cases / f"scope-{index:03d}"
        _copy_case(correct, destination)
        _replace_cell(destination, "a07-explore-run", SOLUTION_CELLS["a07-explore-run"] + "\n" + fragment + "\n")
        yield fragment, destination


def _run_cli_shape(
    submission: Path,
    success: bool,
    environment: dict[str, str] | None = None,
) -> dict:
    environment = os.environ.copy() if environment is None else environment
    grader = Path(__file__).resolve().parent / "autograder.py"
    with tempfile.TemporaryDirectory(prefix="a07-cli-") as cli_name:
        cli_root = Path(cli_name)
        completed = subprocess.run(
            [sys.executable, str(grader), str(submission)],
            cwd=cli_root,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        assert completed.returncode == 0, completed.stderr
        result_path = cli_root / "result.json"
        assert result_path.read_bytes().endswith(b"\n")
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["schema"] == "classroom50/result/v1" and result["max-score"] == 80
        assert (result["score"] == 80) is success
        assert UTC_DATETIME.fullmatch(result["datetime"])
        return result


def _assert_delivery_inventory(
    correct: Path,
    temporary: Path,
    expected_context: dict[str, str],
) -> None:
    accepted = temporary / "accepted-delivery"
    _copy_case(correct, accepted)
    (accepted / ".classroom50.yaml").write_text("version: 1\n", encoding="utf-8")
    workflow = accepted / ".github/workflows/autograde.yaml"
    workflow.parent.mkdir(parents=True, exist_ok=True)
    workflow.write_text("name: autograde\n", encoding="utf-8")
    git_config = accepted / ".git/config"
    git_config.parent.mkdir(parents=True)
    git_config.write_text("[core]\n", encoding="utf-8")
    public = subprocess.run(
        [sys.executable, str(accepted / "check_assignment.py")],
        cwd=accepted,
        text=True,
        capture_output=True,
        check=False,
    )
    assert public.returncode == 0, public.stdout + public.stderr
    _assert_result(grade_submission(accepted), expected_context, True)
    production = subprocess.run(
        [sys.executable, str(Path(__file__).resolve().parent / "autograder.py")],
        cwd=accepted,
        env=os.environ.copy(),
        text=True,
        capture_output=True,
        check=False,
    )
    assert production.returncode == 0, production.stdout + production.stderr
    assert json.loads((accepted / "result.json").read_text())["score"] == 80
    (accepted / "result.json").unlink()

    for label, relative in (
        ("extra-root", "notes.txt"),
        ("extra-workflow", ".github/workflows/extra.yaml"),
        ("grader-tree", "_grader_selftest/copied.py"),
        ("nested-git", "ordinary/.git/nested.txt"),
    ):
        rejected = temporary / f"inventory-{label}"
        _copy_case(accepted, rejected)
        path = rejected / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("unexpected\n", encoding="utf-8")
        public = subprocess.run(
            [sys.executable, str(rejected / "check_assignment.py")],
            cwd=rejected,
            text=True,
            capture_output=True,
            check=False,
        )
        assert public.returncode == 1
        _assert_result(grade_submission(rejected), expected_context, False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-behavior-mutants", action="store_true")
    parser.add_argument("--only-behavior-mutants", action="store_true")
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="a07-release-harness-") as temporary_name:
        temporary = Path(temporary_name)
        correct = temporary / "correct submission"
        cases = temporary / "cases"
        cases.mkdir()
        materialize_correct(correct)
        with grader_context() as expected_context:
            correct_result = _grade_case(correct, expected_context, True)
            print(f"[HARNESS] correct submission: {correct_result['score']}/80")
            _assert_delivery_inventory(correct, temporary, expected_context)

            if not args.only_behavior_mutants:
                starter = cases / "starter"
                _copy_template(starter)
                starter_result = grade_submission(starter)
                _assert_result(starter_result, expected_context, False)

                corrected = cases / "corrected-resubmission"
                materialize_correct(corrected)
                _grade_case(corrected, expected_context, True)

                for name, case in _static_mutation_cases(correct, cases):
                    _grade_case(case, expected_context, False)
                    print(f"[HARNESS] rejected static mutant: {name}")
                for fragment, case in _scope_mutation_cases(correct, cases):
                    _grade_case(case, expected_context, False)
                    print(f"[HARNESS] rejected scope mutant: {fragment}")
            if not args.skip_behavior_mutants:
                for name, case in _behavior_mutation_cases(correct, cases):
                    _grade_case(case, expected_context, False)
                    print(f"[HARNESS] rejected behavior mutant: {name}")

            if not args.only_behavior_mutants:
                public = subprocess.run(
                    [sys.executable, str(correct / "check_assignment.py")],
                    cwd=correct,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                assert public.returncode == 0, public.stdout + public.stderr
                _run_cli_shape(correct, True)
                _run_cli_shape(starter, False)

                for label, review_value in (("missing", None), ("empty", "   ")):
                    fallback_env = os.environ.copy()
                    if review_value is None:
                        fallback_env.pop("REVIEW_URL", None)
                    else:
                        fallback_env["REVIEW_URL"] = review_value
                    fallback_result = _run_cli_shape(correct, True, fallback_env)
                    assert fallback_result["review"] == fallback_result["commit"]

        if args.only_behavior_mutants:
            print("Assignment 07 behavior-mutant harness passed.")
            return 0
        grader = Path(__file__).resolve().parent / "autograder.py"
        base_context = os.environ.copy()
        with grader_context():
            base_context = os.environ.copy()
        for environment_name in REQUIRED_CONTEXT_ENV.values():
            for label, replacement in (("missing", None), ("empty", "   ")):
                failure_root = temporary / f"context-{environment_name.lower()}-{label}"
                failure_root.mkdir()
                broken_context = dict(base_context)
                if replacement is None:
                    broken_context.pop(environment_name, None)
                else:
                    broken_context[environment_name] = replacement
                failed = subprocess.run(
                    [sys.executable, str(grader), str(correct)],
                    cwd=failure_root,
                    env=broken_context,
                    text=True,
                    capture_output=True,
                    check=False,
                )
                assert failed.returncode != 0
                assert not (failure_root / "result.json").exists()
    print("Assignment 07 adversarial release harness passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
