#!/usr/bin/env python3
"""Fast, dependency-free structural checks for the course repository."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LECTURES = [ROOT / f"{number:02d}" for number in range(1, 12)]

L4_DEMO_NOTEBOOKS = {
    "demo1_jupyter_basics.ipynb",
    "demo2_pandas_basics.ipynb",
    "demo3_data_io.ipynb",
}
L4_PORTABLE_KERNELSPEC = {
    "display_name": "Python 3",
    "language": "python",
    "name": "python3",
}
L4_ANSCOMBE_SHA256 = (
    "a0c1f636aa0347101de76271e7efe4c86a22ef28cda62886eaff23a1bf1924b1"
)
L4_SCOPE_PATTERNS = {
    "Matplotlib import": re.compile(r"\b(?:import|from)\s+matplotlib\b"),
    "Seaborn import": re.compile(r"\b(?:import|from)\s+seaborn\b"),
    "plotting call": re.compile(r"\.plot\s*\("),
    "cleaning/grouping/advanced pandas call": re.compile(
        r"\.(?:"
        r"fillna|dropna|isna|notna|duplicated|drop_duplicates|"
        r"groupby|agg|aggregate|transform|pivot|pivot_table|"
        r"query|isin|select_dtypes|value_counts|unique|nunique|rank|"
        r"rename|drop|astype|to_datetime|read_excel|read_json"
        r")\s*\("
    ),
    "notebook magic": re.compile(r"(?m)^\s*%"),
    "Google Drive or manual upload": re.compile(
        r"drive\.mount|files\.upload|/content/drive"
    ),
}

L5_DEMO_NOTEBOOKS = {
    "demo1_audit_decisions.ipynb",
    "demo2_targeted_transformations.ipynb",
    "demo3_validated_pipeline.ipynb",
}
L5_PEOPLE_SHA256 = (
    "7b3223154756aa59f2f00027ddbadaa2"
    "25eeee51ad75d0df91de1fd8d14abe2d"
)
L5_SCOPE_PATTERNS = {
    "plotting library import": re.compile(
        r"\b(?:import|from)\s+(?:matplotlib|seaborn|altair|plotly)\b"
    ),
    "plotting call": re.compile(r"\.(?:plot|hist|boxplot)\s*\("),
    "join/group/reshape call": re.compile(
        r"\.(?:"
        r"merge|join|concat|melt|pivot|pivot_table|"
        r"groupby|agg|aggregate|transform"
        r")\s*\("
    ),
    "encoding or binning call": re.compile(
        r"\.(?:get_dummies|cut|qcut|factorize)\s*\("
    ),
    "chained inplace mutation": re.compile(r"\binplace\s*=\s*True\b"),
    "notebook magic": re.compile(r"(?m)^\s*%"),
    "Google Drive or manual upload": re.compile(
        r"drive\.mount|files\.upload|/content/drive"
    ),
}

L6_DEMO_NOTEBOOKS = {
    "demo1_validated_merge.ipynb",
    "demo2_concat_alignment.ipynb",
    "demo3_structural_reshape.ipynb",
}
L6_FIXTURE_SHA256 = {
    "scores_wide.csv": (
        "5098b5e9a0165f9f7f6e22bc761f01d"
        "5ddb7205af9b88d41876f116cea2d7c38"
    ),
    "sites_history.csv": (
        "42e3b766ca41024b33883463d49c9be5"
        "6d3998536e7f00e0ba483dd799fdd935"
    ),
    "visits.csv": (
        "ccff0b9eaab1b6aae702734628db50b5"
        "223b04efc0071e1e9b4b9d6796e0c930"
    ),
}
L6_SCOPE_PATTERNS = {
    "plotting library import": re.compile(
        r"\b(?:import|from)\s+(?:matplotlib|seaborn|altair|plotly)\b"
    ),
    "plotting call": re.compile(r"\.(?:plot|hist|boxplot)\s*\("),
    "grouping or aggregating pivot call": re.compile(
        r"\.(?:groupby|agg|aggregate|transform|pivot_table)\s*\("
    ),
    "time-series call": re.compile(
        r"\.(?:to_period|resample|rolling|expanding|ewm|shift|asfreq)\s*\("
    ),
    "datetime conversion": re.compile(r"\bpd\.to_datetime\s*\("),
    "notebook magic": re.compile(r"(?m)^\s*%"),
    "Google Drive or manual upload": re.compile(
        r"drive\.mount|files\.upload|/content/drive"
    ),
}

L7_DEMO_NOTEBOOKS = {
    "demo1_critique_redesign.ipynb",
    "demo2_figure_axes.ipynb",
    "demo3_explore_explain.ipynb",
}
L7_FIXTURE_SHA256 = {
    "followup_summary.csv": (
        "928e929c0779800eb9f5b4cfbfadcafe"
        "0a0fc160d315265ce36182047c32c6f9"
    ),
    "participant_scores.csv": (
        "8eecd1393f3dbd4599269ba41724b283"
        "25ea2035f926ffeca674c2150abfc165"
    ),
    "program_progress.csv": (
        "c48d53634f711d4f60b32f230633a47"
        "c77e56d8b1eac5f8c84fbad3858f85b36"
    ),
}
L7_NOTEBOOK_FIXTURES = {
    "demo1_critique_redesign.ipynb": {"followup_summary.csv"},
    "demo2_figure_axes.ipynb": {
        "participant_scores.csv",
        "program_progress.csv",
    },
    "demo3_explore_explain.ipynb": {
        "participant_scores.csv",
        "program_progress.csv",
    },
}
L7_SCOPE_PATTERNS = {
    "advanced or interactive plotting library": re.compile(
        r"\b(?:import|from)\s+(?:altair|bokeh|plotly|plotnine|holoviews)\b"
    ),
    "network dataset": re.compile(r"\b(?:sns\.)?load_dataset\s*\("),
    "random data": re.compile(r"\b(?:np\.)?random\."),
    "cleaning/join/reshape/grouping call": re.compile(
        r"\.(?:"
        r"fillna|dropna|drop_duplicates|merge|concat|melt|"
        r"pivot|pivot_table|groupby|agg|aggregate|transform"
        r")\s*\("
    ),
    "time-series call": re.compile(
        r"\.(?:to_period|resample|rolling|expanding|ewm|shift|asfreq)\s*\("
    ),
    "correlation/regression/model call": re.compile(
        r"(?:\.corr\s*\(|\b(?:regplot|lmplot)\s*\(|"
        r"\b(?:sklearn|statsmodels|scipy\.stats)\b)"
    ),
    "out-of-scope chart family": re.compile(
        r"\.(?:pie|violinplot|kdeplot|pairplot|jointplot|heatmap)\s*\("
    ),
    "notebook magic": re.compile(r"(?m)^\s*%"),
    "Google Drive or manual upload": re.compile(
        r"drive\.mount|files\.upload|/content/drive"
    ),
}

L7_ASSIGNMENT_CELL_IDS = [
    "a07-header",
    "a07-setup",
    "a07-terms-data",
    "a07-load",
    "a07-task1-contract",
    "a07-task1-evidence",
    "a07-explore-function",
    "a07-explore-run",
    "a07-task1-reflection",
    "a07-task2-context",
    "a07-supplied-flawed",
    "a07-task2-critique",
    "a07-critique-evidence",
    "a07-redesign-function",
    "a07-redesign-run",
    "a07-task3-contract",
    "a07-final-contract-values",
    "a07-supporting-data",
    "a07-explanatory-function",
    "a07-explanatory-run",
    "a07-evidence-export",
    "a07-visual-review",
    "a07-final-verify",
]
L7_ASSIGNMENT_MARKDOWN_IDS = {
    "a07-header",
    "a07-terms-data",
    "a07-task1-contract",
    "a07-task1-reflection",
    "a07-task2-context",
    "a07-task2-critique",
    "a07-task3-contract",
    "a07-visual-review",
}
L7_ASSIGNMENT_FIXTURE_SHA256 = {
    "fixture.json": (
        "1c3397cb2d98ae239f6a7cd254bb3aa9"
        "980d94cd23af4546c834a9262de0a28c"
    ),
    "format_completion.csv": (
        "20ad900633154f5f3a2c09cfbc2f890"
        "f8423da0897d6345841745332110be66a"
    ),
    "pathway_checkpoints.csv": (
        "ec9a336b7fb97418a6f058704f2509c"
        "8cee6b13d744efb7a6e3e99224ef8c258"
    ),
    "session_observations.csv": (
        "fc4d69ab836288a2fe9c505c65c0841"
        "3e137e51ab1914cd0e350f6e6636da096"
    ),
}
L7_ASSIGNMENT_PROTECTED_CELL_SHA256 = {
    "a07-header": "011748bb7034d207811f6f934d6fa80c6f4a6c4cedeab5b9c368dbf56152aeef",
    "a07-setup": "0bb09a1b55bd5488b844b6c99042f3f8bb6934e66eb831f98cd3dfd4a21baaa1",
    "a07-terms-data": "f762645fcbbe9d28dbd3d77ae4d124baa9030fdc41a1295be3a5b4d9634dcad4",
    "a07-task2-context": "7ddee594b77784ea7b2684f82f1fb1215bbdf79224ffd153a2269bfec3278fa2",
    "a07-supplied-flawed": "96bfd9dd6114ad84f305dc8567e757ac8ce33cfed55c546447f763bb73bf867b",
    "a07-final-verify": "25bc659c20a016d53b2fc1eb1c27a34d9829d94964ed5d6efc4658b74f0af375",
}

L8_ASSIGNMENT_CELL_IDS = [
    "a08-header",
    "a08-setup",
    "a08-terms-data",
    "a08-load",
    "a08-task1-contract",
    "a08-task1-values",
    "a08-count-function",
    "a08-task1-run",
    "a08-task1-save",
    "a08-task1-explain",
    "a08-task2-prompt",
    "a08-center-summary-function",
    "a08-context-function",
    "a08-two-key-function",
    "a08-task2-run",
    "a08-task2-save",
    "a08-task2-explain",
    "a08-task3-prompt",
    "a08-pivot-values",
    "a08-pivot-function",
    "a08-task3-run",
    "a08-task3-save",
    "a08-task3-explain",
    "a08-synthesis",
    "a08-final-verify",
]
L8_ASSIGNMENT_MARKDOWN_IDS = {
    "a08-header",
    "a08-terms-data",
    "a08-task1-contract",
    "a08-task1-explain",
    "a08-task2-prompt",
    "a08-task2-explain",
    "a08-task3-prompt",
    "a08-task3-explain",
    "a08-synthesis",
}
L8_ASSIGNMENT_FIXTURE_SHA256 = {
    "fixture.json": (
        "b2fee1c48fb678b81318d2f085c42e2"
        "f9b480bd6c4eed6f07ef118b9bfd70860"
    ),
    "support_requests.csv": (
        "a9136161332c5da9f8f1251d869bbd01"
        "4ed762751675fb757f81a79cff5352d6"
    ),
}
L8_ASSIGNMENT_PROTECTED_FILE_SHA256 = {
    "README.md": "c7382a76e6cce665176d8a3d65dfb2c103d65a70132b1b41ba68c7cc79079f32",
    "PLATFORM_CHECK.md": "d60455f2ea443990929cea97260c509399454e8bb839acc7043e60bbc3120b41",
    "check_assignment.py": "64256b16b0bae2a29192b2397ca52c80bac5c5e21a0edf32fefb4d53038d6144",
}
L8_ASSIGNMENT_PROTECTED_CELL_SHA256 = {
    "a08-header": "c49240238f3ebd296cf0c211170c56764f657972b9976a2a6c821d745b59b700",
    "a08-setup": "7d64bb798e93b090281c6f427dd10d113caffe31debf836eeefcb18b8b162778",
    "a08-terms-data": "0888ac5c6da29f2fc882d06cc707956518908ad339ce2a3319625100cc9e2d0c",
    "a08-task2-prompt": "c121bd5943bff1925995836ece9508768e394ff2da0d9d906ee7d319ccb972e6",
    "a08-task3-prompt": "45e6d964a75c3a478c63d4623416fec1f814c576ffa344f7b1e416618604ead0",
    "a08-final-verify": "70e228d9b9a14fb1a6f111a42acb08f78103f125f099a449b53e148278713b76",
}

L9_ASSIGNMENT_CELL_IDS = [
    "a09-header",
    "a09-setup",
    "a09-terms-data",
    "a09-load",
    "a09-task1-contract",
    "a09-task1-values",
    "a09-prepare-function",
    "a09-task1-run",
    "a09-task1-save",
    "a09-task1-explain",
    "a09-task2-prompt",
    "a09-hourly-function",
    "a09-summary-function",
    "a09-task2-run",
    "a09-task2-save",
    "a09-task2-explain",
    "a09-task3-prompt",
    "a09-features-function",
    "a09-task3-features-run",
    "a09-availability-values",
    "a09-blocks-function",
    "a09-task3-run",
    "a09-task3-save",
    "a09-task3-explain",
    "a09-synthesis",
    "a09-final-verify",
]
L9_ASSIGNMENT_MARKDOWN_IDS = {
    "a09-header",
    "a09-terms-data",
    "a09-task1-contract",
    "a09-task1-explain",
    "a09-task2-prompt",
    "a09-task2-explain",
    "a09-task3-prompt",
    "a09-task3-explain",
    "a09-synthesis",
}
L9_ASSIGNMENT_FIXTURE_SHA256 = {
    "fixture.json": "27558bc4da7738775879501a6f11a0a9d874f3948823e54bb5e82ab91a02d703",
    "zone_co2_readings.csv": "c21c8571b4fe9a1e84a5224c7bffce972bb6f9517df172d92b3661a2bf9452f4",
}
L9_ASSIGNMENT_PROTECTED_FILE_SHA256 = {
    "README.md": "03026ba8f1d57e57b4a030c2ec1cd3cf0358a24df8829b735a099b47881654ff",
    "PLATFORM_CHECK.md": "019a5c52b6c7adca37c0c95300a633d15c594258928a9080dab24d6b6026952c",
    "check_assignment.py": "511fd29f063829b2bb799398be411ae2e7c77aafc37aac34db8fb6af3a2e9824",
}
L9_ASSIGNMENT_PROTECTED_CELL_SHA256 = {
    "a09-header": "131ea3ecea5c880816109cc7c1b03980dcd3c0c2e4cd9d17b34d68a5af3e9163",
    "a09-setup": "2a2426aaadd2dfa4fb8fd231d47f0f5918f0ce86dce6e485f7e9b92ee950f754",
    "a09-terms-data": "f8ad0129811ecd03ed7c0dc60b860b68a1be545003e4961cdaea6b43489dfa59",
    "a09-task2-prompt": "d1d9d124b7fd9904ee9a47256745def60937945d1bfe40194239314199bedc1b",
    "a09-task3-prompt": "90f1797b0b65472c3567451467e85e144a2add9110a3d11a1714d3d31fe75966",
    "a09-final-verify": "015c87bbc3742a61961efbf806ed513f67363191766ad8bcb7d1bf7200fb996a",
}

L8_DEMO_NOTEBOOKS = {
    "demo1_grouping_grain_counts.ipynb",
    "demo2_named_aggregation_transform.ipynb",
    "demo3_aggregating_pivot.ipynb",
}
L8_ENCOUNTERS_SHA256 = (
    "24a31904c1371553ff3af627dc21146ed"
    "743c8c0c47452ade3628c2fc199c5dc"
)
L8_NOTEBOOK_OUTPUTS = {
    "demo1_grouping_grain_counts.ipynb": {
        "count_comparison.csv",
    },
    "demo2_named_aggregation_transform.ipynb": {
        "facility_summary.csv",
        "encounters_with_context.csv",
        "facility_service_summary.csv",
    },
    "demo3_aggregating_pivot.ipynb": {
        "mean_charge_pivot.csv",
    },
}
L8_SCOPE_PATTERNS = {
    "plotting library import": re.compile(
        r"\b(?:import|from)\s+(?:matplotlib|seaborn|altair|bokeh|plotly)\b"
    ),
    "plotting call": re.compile(
        r"\.(?:plot|hist|boxplot|bar|scatter|pie|imshow)\s*\("
    ),
    "cleaning or imputation call": re.compile(
        r"\.(?:fillna|ffill|bfill|interpolate|drop_duplicates|replace)\s*\("
    ),
    "join or structural reshape call": re.compile(
        r"\.(?:merge|join|melt|pivot)\s*\("
    ),
    "advanced GroupBy or crosstab call": re.compile(
        r"\.(?:apply|filter)\s*\(|\bpd\.crosstab\s*\(|\bMultiIndex\b"
    ),
    "time-series call": re.compile(
        r"\.(?:to_period|resample|rolling|expanding|ewm|shift|asfreq)\s*\("
    ),
    "statistics or modeling library": re.compile(
        r"\b(?:sklearn|statsmodels|scipy\.stats|xgboost|tensorflow|torch)\b"
    ),
    "remote or performance tooling": re.compile(
        r"\b(?:ssh|tmux|dask|multiprocessing|joblib|numba)\b"
    ),
    "network access": re.compile(
        r"\b(?:requests|urlopen|urlretrieve|wget|curl)\b|https?://"
    ),
    "random or mutable date": re.compile(
        r"\b(?:np\.)?random\.|\b(?:datetime|Timestamp)\.now\s*\("
        r"|\bdate\.today\s*\("
    ),
    "notebook magic": re.compile(r"(?m)^\s*%"),
    "Google Drive or manual upload": re.compile(
        r"drive\.mount|files\.upload|/content/drive"
    ),
}

L9_DEMO_NOTEBOOKS = {
    "demo1_temporal_structure.ipynb",
    "demo2_frequency_measurement.ipynb",
    "demo3_past_only_features.ipynb",
}
L9_STATIONS_SHA256 = (
    "57dcdb82372805cf1dda83a7c227b463f"
    "e997cf1437275d64d01b9719ff26b54"
)
L9_NOTEBOOK_OUTPUTS = {
    "demo1_temporal_structure.ipynb": {
        "prepared_panel.csv": (
            431,
            "a9e2b75c2f4e9f9a3778b53cd87e68d4a559511c368cad80b7153b1109a987ba",
        ),
    },
    "demo2_frequency_measurement.ipynb": {
        "hourly_grid.csv": (
            788,
            "7054dfb410b36f35ef53ff4e02cc77fb633ff413e1dcd9f66d1807674053b40e",
        ),
        "two_hour_summary.csv": (
            361,
            "0558659b66336e71c3c67769097aadf4e2616a2d4f913425bb498463528a9d6f",
        ),
    },
    "demo3_past_only_features.ipynb": {
        "temporal_features.csv": (
            633,
            "5a6524e8dbb37da3cc056cc648e5ff444c12cb485214bef1a95c3a67a22af3ab",
        ),
        "availability_decisions.csv": (
            326,
            "d4125def8dcf8e23b9f33574f1dd9e14a5ed3f92889f88b455509110ad87e505",
        ),
        "chronological_blocks.csv": (
            535,
            "7ea9752756ef882dbe19318bfbb1614c33ff6bbca45a5bbc5effe4bcad065a67",
        ),
    },
}
L9_SCOPE_PATTERNS = {
    "plotting library import": re.compile(
        r"\b(?:import|from)\s+(?:matplotlib|seaborn|altair|bokeh|plotly)\b"
    ),
    "plotting or image call": re.compile(
        r"\.(?:plot|hist|boxplot|bar|scatter|imshow|savefig)\s*\("
    ),
    "fill, interpolation, or value replacement": re.compile(
        r"\.(?:fillna|ffill|bfill|interpolate|replace)\s*\("
    ),
    "advanced or future window computation": re.compile(
        r"\.(?:ewm|expanding|apply)\s*\(|\bcenter\s*=\s*True\b"
        r"|\.shift\s*\(\s*-"
    ),
    "statistics, forecasting, or modeling library": re.compile(
        r"\b(?:sklearn|statsmodels|scipy\.stats|xgboost|tensorflow|torch|prophet)\b"
    ),
    "runtime data network access": re.compile(
        r"\b(?:requests|urlopen|urlretrieve|wget|curl)\b"
    ),
    "random or mutable date": re.compile(
        r"\b(?:np\.)?random\.|\b(?:datetime|Timestamp)\.now\s*\("
        r"|\bdate\.today\s*\("
    ),
    "obsolete pandas time alias": re.compile(
        r"(?:freq\s*=|resample\s*\(|rolling\s*\()\s*"
        r"[\"'](?:H|[0-9]+H|Q|A|T|M)[\"']"
    ),
    "notebook magic": re.compile(r"(?m)^\s*%"),
    "Google Drive or manual upload": re.compile(
        r"drive\.mount|files\.upload|/content/drive"
    ),
}


def audit_lecture04_demos(errors: list[str]) -> None:
    """Enforce the narrow source and scope policy for the three Lecture 04 demos."""

    demo_dir = ROOT / "04" / "demo"
    actual_notebooks = {path.name for path in demo_dir.glob("*.ipynb")}
    if actual_notebooks != L4_DEMO_NOTEBOOKS:
        missing = sorted(L4_DEMO_NOTEBOOKS - actual_notebooks)
        unexpected = sorted(actual_notebooks - L4_DEMO_NOTEBOOKS)
        if missing:
            errors.append(f"Lecture 04 demo notebooks missing: {', '.join(missing)}")
        if unexpected:
            errors.append(
                f"unexpected Lecture 04 demo notebooks: {', '.join(unexpected)}"
            )

    paired_markdown = sorted(
        path.name
        for path in demo_dir.glob("demo*.md")
        if path.with_suffix(".ipynb").name in L4_DEMO_NOTEBOOKS
    )
    if paired_markdown:
        errors.append(
            "Lecture 04 paired Markdown must not be committed: "
            + ", ".join(paired_markdown)
        )

    fixture = demo_dir / "data" / "anscombe.csv"
    if not fixture.is_file():
        errors.append("missing Lecture 04 fixture: 04/demo/data/anscombe.csv")
    else:
        actual_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if actual_sha256 != L4_ANSCOMBE_SHA256:
            errors.append(
                "unexpected Lecture 04 anscombe.csv checksum: "
                f"{actual_sha256}"
            )

    for name in sorted(L4_DEMO_NOTEBOOKS & actual_notebooks):
        path = demo_dir / name
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # The repository-wide notebook parser reports the detailed error.
            continue

        kernelspec = notebook.get("metadata", {}).get("kernelspec")
        if kernelspec != L4_PORTABLE_KERNELSPEC:
            errors.append(
                f"non-portable Lecture 04 kernelspec: {path.relative_to(ROOT)}"
            )

        cells = notebook.get("cells", [])
        cell_ids = [cell.get("id") for cell in cells]
        if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(
            set(cell_ids)
        ):
            errors.append(
                f"missing or duplicate Lecture 04 cell id: {path.relative_to(ROOT)}"
            )

        code_source: list[str] = []
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(
                    "stored execution state in Lecture 04 demo: "
                    f"{path.relative_to(ROOT)}"
                )
                break
            source = cell.get("source", [])
            code_source.append("".join(source) if isinstance(source, list) else source)

        joined_source = "\n".join(code_source)
        for label, pattern in L4_SCOPE_PATTERNS.items():
            if pattern.search(joined_source):
                errors.append(
                    f"Lecture 04 scope violation ({label}): {path.relative_to(ROOT)}"
                )


def audit_lecture05_demos(errors: list[str]) -> None:
    """Enforce the source, fixture, state, and scope policy for Lecture 05 demos."""

    demo_dir = ROOT / "05" / "demo"
    actual_notebooks = {path.name for path in demo_dir.glob("*.ipynb")}
    if actual_notebooks != L5_DEMO_NOTEBOOKS:
        missing = sorted(L5_DEMO_NOTEBOOKS - actual_notebooks)
        unexpected = sorted(actual_notebooks - L5_DEMO_NOTEBOOKS)
        if missing:
            errors.append(f"Lecture 05 demo notebooks missing: {', '.join(missing)}")
        if unexpected:
            errors.append(
                f"unexpected Lecture 05 demo notebooks: {', '.join(unexpected)}"
            )

    paired_markdown = sorted(
        path.name
        for path in demo_dir.glob("demo*.md")
        if path.with_suffix(".ipynb").name in L5_DEMO_NOTEBOOKS
    )
    if paired_markdown:
        errors.append(
            "Lecture 05 paired Markdown must not be committed: "
            + ", ".join(paired_markdown)
        )

    fixture = demo_dir / "data" / "supplied_people_raw.csv"
    if not fixture.is_file():
        errors.append(
            "missing Lecture 05 fixture: 05/demo/data/supplied_people_raw.csv"
        )
    else:
        actual_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if actual_sha256 != L5_PEOPLE_SHA256:
            errors.append(
                "unexpected Lecture 05 supplied_people_raw.csv checksum: "
                f"{actual_sha256}"
            )

    requirements = demo_dir / "requirements.txt"
    if requirements.is_file():
        expected_requirements = "numpy==2.0.2\npandas==3.0.3\n"
        if requirements.read_text(encoding="utf-8") != expected_requirements:
            errors.append("unexpected Lecture 05 demo requirements")
    else:
        errors.append("missing Lecture 05 demo requirements.txt")

    for name in sorted(L5_DEMO_NOTEBOOKS & actual_notebooks):
        path = demo_dir / name
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # The repository-wide notebook parser reports the detailed error.
            continue

        kernelspec = notebook.get("metadata", {}).get("kernelspec")
        if kernelspec != L4_PORTABLE_KERNELSPEC:
            errors.append(
                f"non-portable Lecture 05 kernelspec: {path.relative_to(ROOT)}"
            )

        cells = notebook.get("cells", [])
        cell_ids = [cell.get("id") for cell in cells]
        if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(
            set(cell_ids)
        ):
            errors.append(
                f"missing or duplicate Lecture 05 cell id: {path.relative_to(ROOT)}"
            )

        code_source: list[str] = []
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(
                    "stored execution state in Lecture 05 demo: "
                    f"{path.relative_to(ROOT)}"
                )
                break
            source = cell.get("source", [])
            code_source.append("".join(source) if isinstance(source, list) else source)

        joined_source = "\n".join(code_source)
        if 'PANDAS_CANDIDATE = "3.0.3"' not in joined_source:
            errors.append(
                f"missing Lecture 05 pandas candidate setup: {path.relative_to(ROOT)}"
            )
        if L5_PEOPLE_SHA256 not in joined_source:
            errors.append(
                f"missing Lecture 05 fixture checksum: {path.relative_to(ROOT)}"
            )
        for label, pattern in L5_SCOPE_PATTERNS.items():
            if pattern.search(joined_source):
                errors.append(
                    f"Lecture 05 scope violation ({label}): {path.relative_to(ROOT)}"
                )

        forward_fill_count = len(re.findall(r"\.ffill\s*\(", joined_source))
        backward_fill_count = len(re.findall(r"\.bfill\s*\(", joined_source))
        if name == "demo2_targeted_transformations.ipynb":
            if forward_fill_count != 1 or backward_fill_count:
                errors.append(
                    "Lecture 05 Demo 2 must contain exactly one rejected "
                    "forward-fill preview and no backward fill"
                )
        elif forward_fill_count or backward_fill_count:
            errors.append(
                f"unexpected Lecture 05 adjacent-row fill: {path.relative_to(ROOT)}"
            )


def audit_lecture06_demos(errors: list[str]) -> None:
    """Enforce the source, fixture, state, and scope policy for Lecture 06 demos."""

    demo_dir = ROOT / "06" / "demo"
    actual_notebooks = {path.name for path in demo_dir.glob("*.ipynb")}
    if actual_notebooks != L6_DEMO_NOTEBOOKS:
        missing = sorted(L6_DEMO_NOTEBOOKS - actual_notebooks)
        unexpected = sorted(actual_notebooks - L6_DEMO_NOTEBOOKS)
        if missing:
            errors.append(f"Lecture 06 demo notebooks missing: {', '.join(missing)}")
        if unexpected:
            errors.append(
                f"unexpected Lecture 06 demo notebooks: {', '.join(unexpected)}"
            )

    paired_markdown = sorted(
        path.name
        for path in demo_dir.glob("demo*.md")
        if path.with_suffix(".ipynb").name in L6_DEMO_NOTEBOOKS
    )
    if paired_markdown:
        errors.append(
            "Lecture 06 paired Markdown must not be committed: "
            + ", ".join(paired_markdown)
        )

    for name, expected_sha256 in sorted(L6_FIXTURE_SHA256.items()):
        fixture = demo_dir / "data" / name
        if not fixture.is_file():
            errors.append(f"missing Lecture 06 fixture: 06/demo/data/{name}")
            continue
        actual_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            errors.append(
                f"unexpected Lecture 06 {name} checksum: {actual_sha256}"
            )

    requirements = demo_dir / "requirements.txt"
    if requirements.is_file():
        expected_requirements = "numpy==2.0.2\npandas==3.0.3\n"
        if requirements.read_text(encoding="utf-8") != expected_requirements:
            errors.append("unexpected Lecture 06 demo requirements")
    else:
        errors.append("missing Lecture 06 demo requirements.txt")

    for name in sorted(L6_DEMO_NOTEBOOKS & actual_notebooks):
        path = demo_dir / name
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # The repository-wide notebook parser reports the detailed error.
            continue

        kernelspec = notebook.get("metadata", {}).get("kernelspec")
        if kernelspec != L4_PORTABLE_KERNELSPEC:
            errors.append(
                f"non-portable Lecture 06 kernelspec: {path.relative_to(ROOT)}"
            )

        cells = notebook.get("cells", [])
        cell_ids = [cell.get("id") for cell in cells]
        if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(
            set(cell_ids)
        ):
            errors.append(
                f"missing or duplicate Lecture 06 cell id: {path.relative_to(ROOT)}"
            )

        code_source: list[str] = []
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(
                    "stored execution state in Lecture 06 demo: "
                    f"{path.relative_to(ROOT)}"
                )
                break
            source = cell.get("source", [])
            code_source.append("".join(source) if isinstance(source, list) else source)

        joined_source = "\n".join(code_source)
        if 'PANDAS_CANDIDATE = "3.0.3"' not in joined_source:
            errors.append(
                f"missing Lecture 06 pandas candidate setup: {path.relative_to(ROOT)}"
            )
        for label, pattern in L6_SCOPE_PATTERNS.items():
            if pattern.search(joined_source):
                errors.append(
                    f"Lecture 06 scope violation ({label}): {path.relative_to(ROOT)}"
                )


def audit_lecture07_demos(errors: list[str]) -> None:
    """Enforce the source, fixture, state, and scope policy for Lecture 07 demos."""

    demo_dir = ROOT / "07" / "demo"
    actual_notebooks = {path.name for path in demo_dir.glob("*.ipynb")}
    if actual_notebooks != L7_DEMO_NOTEBOOKS:
        missing = sorted(L7_DEMO_NOTEBOOKS - actual_notebooks)
        unexpected = sorted(actual_notebooks - L7_DEMO_NOTEBOOKS)
        if missing:
            errors.append(f"Lecture 07 demo notebooks missing: {', '.join(missing)}")
        if unexpected:
            errors.append(
                f"unexpected Lecture 07 demo notebooks: {', '.join(unexpected)}"
            )

    paired_markdown = sorted(
        path.name
        for path in demo_dir.glob("demo*.md")
        if path.with_suffix(".ipynb").name in L7_DEMO_NOTEBOOKS
    )
    if paired_markdown:
        errors.append(
            "Lecture 07 paired Markdown must not be committed: "
            + ", ".join(paired_markdown)
        )

    for name, expected_sha256 in sorted(L7_FIXTURE_SHA256.items()):
        fixture = demo_dir / "data" / name
        if not fixture.is_file():
            errors.append(f"missing Lecture 07 fixture: 07/demo/data/{name}")
            continue
        actual_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if actual_sha256 != expected_sha256:
            errors.append(
                f"unexpected Lecture 07 {name} checksum: {actual_sha256}"
            )

    expected_small_files = {
        ".python-version": "3.12.13\n",
        "requirements.txt": (
            "numpy==2.0.2\n"
            "pandas==3.0.3\n"
            "matplotlib==3.10.8\n"
            "seaborn==0.13.2\n"
        ),
        ".gitignore": (
            ".ipynb_checkpoints/\n"
            "output/\n"
            "__pycache__/\n"
            "*.py[cod]\n"
        ),
    }
    for filename, expected_text in expected_small_files.items():
        path = demo_dir / filename
        if not path.is_file():
            errors.append(f"missing Lecture 07 demo {filename}")
        elif path.read_text(encoding="utf-8") != expected_text:
            errors.append(f"unexpected Lecture 07 demo {filename}")

    tracked_output_check = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", "07/demo/output"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked_output_check.returncode != 0:
        errors.append("unable to verify tracked Lecture 07 demo output")
    elif tracked_output_check.stdout.strip():
        errors.append("generated Lecture 07 demo output must not be committed")

    global_cell_ids: set[str] = set()
    for name in sorted(L7_DEMO_NOTEBOOKS & actual_notebooks):
        path = demo_dir / name
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # The repository-wide notebook parser reports the detailed error.
            continue

        kernelspec = notebook.get("metadata", {}).get("kernelspec")
        if kernelspec != L4_PORTABLE_KERNELSPEC:
            errors.append(
                f"non-portable Lecture 07 kernelspec: {path.relative_to(ROOT)}"
            )

        cells = notebook.get("cells", [])
        cell_ids = [cell.get("id") for cell in cells]
        if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(
            set(cell_ids)
        ):
            errors.append(
                f"missing or duplicate Lecture 07 cell id: {path.relative_to(ROOT)}"
            )
        duplicate_global_ids = sorted(set(cell_ids) & global_cell_ids)
        if duplicate_global_ids:
            errors.append(
                "Lecture 07 cell IDs must be globally unique: "
                + ", ".join(duplicate_global_ids)
            )
        global_cell_ids.update(cell_id for cell_id in cell_ids if cell_id)

        code_source: list[str] = []
        for cell in cells:
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(
                    "stored execution state in Lecture 07 demo: "
                    f"{path.relative_to(ROOT)}"
                )
                break
            source = cell.get("source", [])
            code_source.append("".join(source) if isinstance(source, list) else source)

        joined_source = "\n".join(code_source)
        for version in ("2.0.2", "3.0.3", "3.10.8", "0.13.2"):
            if version not in joined_source:
                errors.append(
                    f"missing Lecture 07 candidate {version}: {path.relative_to(ROOT)}"
                )
        for fixture_name in L7_NOTEBOOK_FIXTURES[name]:
            expected_checksum = L7_FIXTURE_SHA256[fixture_name]
            if expected_checksum not in joined_source:
                errors.append(
                    "missing Lecture 07 fixture checksum "
                    f"{fixture_name}: {path.relative_to(ROOT)}"
                )
        for label, pattern in L7_SCOPE_PATTERNS.items():
            if pattern.search(joined_source):
                errors.append(
                    f"Lecture 07 scope violation ({label}): {path.relative_to(ROOT)}"
                )

        if name == "demo1_critique_redesign.ipynb":
            required_fragments = {
                'OUTPUT_DIRECTORY / "followup_redesign.png"',
                'ylim=(0, 80)',
                'hatch="//"',
                'hatch="\\\\"',
                "do not establish",
            }
        elif name == "demo2_figure_axes.ipynb":
            required_fragments = {
                'OUTPUT_DIRECTORY / "core_line_chart.png"',
                '"mean_score": [65.6, 73.2]',
                "bin_edges = [60, 65, 70, 75, 80, 85]",
                "boxplot(",
            }
        else:
            required_fragments = {
                'OUTPUT_DIRECTORY / "program_progress_explanatory.png"',
                'OUTPUT_DIRECTORY / "explanatory_supporting_data.csv"',
                'OUTPUT_DIRECTORY / "explanatory_text_alternative.txt"',
                "sns.scatterplot(",
                "Round 5 observed separation: 7 points",
            }
        missing_fragments = sorted(
            fragment for fragment in required_fragments if fragment not in joined_source
        )
        if missing_fragments:
            errors.append(
                f"missing Lecture 07 contract in {path.relative_to(ROOT)}: "
                + ", ".join(missing_fragments)
            )


def audit_assignment07(errors: list[str]) -> None:
    """Enforce the accepted Assignment 07 starter and instructor-grader surface."""

    assignment_dir = ROOT / "07" / "assignment"
    student_files = {
        ".gitignore",
        ".python-version",
        "PLATFORM_CHECK.md",
        "README.md",
        "assignment.ipynb",
        "check_assignment.py",
        "requirements.txt",
        "data/fixture.json",
        "data/format_completion.csv",
        "data/pathway_checkpoints.csv",
        "data/session_observations.csv",
        "output/.gitkeep",
    }
    actual_student_files = {
        path.relative_to(assignment_dir).as_posix()
        for path in assignment_dir.rglob("*")
        if path.is_file()
        and "_grader_selftest" not in path.relative_to(assignment_dir).parts
        and "__pycache__" not in path.relative_to(assignment_dir).parts
    }
    if actual_student_files != student_files:
        missing = sorted(student_files - actual_student_files)
        unexpected = sorted(actual_student_files - student_files)
        if missing:
            errors.append(f"Assignment 07 student files missing: {', '.join(missing)}")
        if unexpected:
            errors.append(
                "unexpected Assignment 07 student files: " + ", ".join(unexpected)
            )

    expected_small_files = {
        ".python-version": "3.12.13\n",
        "requirements.txt": (
            "numpy==2.0.2\n"
            "pandas==3.0.3\n"
            "matplotlib==3.10.8\n"
            "seaborn==0.13.2\n"
        ),
        ".gitignore": (
            ".ipynb_checkpoints/\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            ".pytest_cache/\n"
            ".venv/\n"
            "venv/\n"
        ),
        "_grader_selftest/requirements.txt": (
            "numpy==2.0.2\n"
            "pandas==3.0.3\n"
            "matplotlib==3.10.8\n"
            "seaborn==0.13.2\n"
            "nbclient==0.10.2\n"
            "nbformat==5.10.4\n"
            "ipykernel==6.29.5\n"
            "Pillow==12.3.0\n"
        ),
    }
    for relative, expected in expected_small_files.items():
        path = assignment_dir / relative
        if not path.is_file():
            errors.append(f"missing Assignment 07 {relative}")
        elif path.read_text(encoding="utf-8") != expected:
            errors.append(f"unexpected Assignment 07 {relative}")

    for name, expected_sha256 in L7_ASSIGNMENT_FIXTURE_SHA256.items():
        path = assignment_dir / "data" / name
        if not path.is_file():
            errors.append(f"missing Assignment 07 fixture: {name}")
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            errors.append(f"unexpected Assignment 07 fixture checksum: {name}")
        if not raw.endswith(b"\n") or b"\r" in raw:
            errors.append(f"Assignment 07 fixture must use LF/final newline: {name}")

    notebook_path = assignment_dir / "assignment.ipynb"
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5:
        errors.append("Assignment 07 notebook must use format 4.5")
    cells = notebook.get("cells", [])
    ids = [cell.get("id") for cell in cells if isinstance(cell, dict)]
    if ids != L7_ASSIGNMENT_CELL_IDS or len(ids) != len(set(ids)):
        errors.append("Assignment 07 notebook cell IDs/order changed")
        return
    if notebook.get("metadata", {}).get("kernelspec") != L4_PORTABLE_KERNELSPEC:
        errors.append("Assignment 07 notebook kernelspec is not portable")
    for cell in cells:
        expected_type = (
            "markdown" if cell["id"] in L7_ASSIGNMENT_MARKDOWN_IDS else "code"
        )
        if cell.get("cell_type") != expected_type:
            errors.append(f"Assignment 07 cell type changed: {cell['id']}")
        if expected_type == "code" and (
            cell.get("execution_count") is not None or cell.get("outputs")
        ):
            errors.append(f"stored execution state in Assignment 07: {cell['id']}")
        if cell["id"] in L7_ASSIGNMENT_PROTECTED_CELL_SHA256:
            source = cell.get("source", "")
            joined = "".join(source) if isinstance(source, list) else source
            observed = hashlib.sha256(joined.encode()).hexdigest()
            if observed != L7_ASSIGNMENT_PROTECTED_CELL_SHA256[cell["id"]]:
                errors.append(f"protected Assignment 07 cell changed: {cell['id']}")

    combined_guidance = "\n".join(
        (assignment_dir / name).read_text(encoding="utf-8")
        for name in ("README.md", "PLATFORM_CHECK.md")
        if (assignment_dir / name).is_file()
    )
    if "colab.research.google.com" in combined_guidance.lower():
        errors.append("Assignment 07 must not publish a Colab badge")
    if re.search(r"\b\d+\s*(?:minutes?|hours?)\b", combined_guidance, re.I):
        errors.append("Assignment 07 must not make a timing claim")
    for required in (
        "Classroom50",
        "VS Code Source Control",
        "GitHub Desktop",
        "python check_assignment.py",
        "critique_redesign.png",
        "pathway_explanatory.png",
    ):
        if required not in combined_guidance:
            errors.append(f"missing Assignment 07 workflow contract: {required}")

    for relative in (
        "_grader_selftest/autograder.py",
        "_grader_selftest/classroom50_grader.py",
        "_grader_selftest/run.py",
        "_grader_selftest/README.md",
    ):
        if not (assignment_dir / relative).is_file():
            errors.append(f"missing Assignment 07 instructor asset: {relative}")


def audit_assignment08(errors: list[str]) -> None:
    """Enforce the accepted Assignment 08 starter and instructor-grader surface."""

    assignment_dir = ROOT / "08" / "assignment"
    student_files = {
        ".gitignore",
        ".python-version",
        "PLATFORM_CHECK.md",
        "README.md",
        "assignment.ipynb",
        "check_assignment.py",
        "requirements.txt",
        "data/fixture.json",
        "data/support_requests.csv",
        "output/.gitkeep",
    }
    actual_student_files = {
        path.relative_to(assignment_dir).as_posix()
        for path in assignment_dir.rglob("*")
        if path.is_file()
        and "_grader_selftest" not in path.relative_to(assignment_dir).parts
        and "__pycache__" not in path.relative_to(assignment_dir).parts
    }
    if actual_student_files != student_files:
        missing = sorted(student_files - actual_student_files)
        unexpected = sorted(actual_student_files - student_files)
        if missing:
            errors.append(f"Assignment 08 student files missing: {', '.join(missing)}")
        if unexpected:
            errors.append(
                "unexpected Assignment 08 student files: " + ", ".join(unexpected)
            )

    expected_small_files = {
        ".python-version": "3.12.13\n",
        "requirements.txt": "numpy==2.0.2\npandas==3.0.3\n",
        ".gitignore": (
            ".ipynb_checkpoints/\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            ".pytest_cache/\n"
            ".venv/\n"
            "venv/\n"
        ),
        "_grader_selftest/requirements.txt": (
            "ipykernel==6.29.5\n"
            "nbclient==0.10.2\n"
            "nbformat==5.10.4\n"
            "numpy==2.0.2\n"
            "pandas==3.0.3\n"
        ),
    }
    for relative, expected in expected_small_files.items():
        path = assignment_dir / relative
        if not path.is_file():
            errors.append(f"missing Assignment 08 {relative}")
        elif path.read_text(encoding="utf-8") != expected:
            errors.append(f"unexpected Assignment 08 {relative}")

    for relative, expected_sha256 in L8_ASSIGNMENT_PROTECTED_FILE_SHA256.items():
        path = assignment_dir / relative
        if not path.is_file():
            errors.append(f"missing Assignment 08 protected file: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            errors.append(f"protected Assignment 08 file changed: {relative}")

    for name, expected_sha256 in L8_ASSIGNMENT_FIXTURE_SHA256.items():
        path = assignment_dir / "data" / name
        if not path.is_file():
            errors.append(f"missing Assignment 08 fixture: {name}")
            continue
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            errors.append(f"unexpected Assignment 08 fixture checksum: {name}")
        if not raw.endswith(b"\n") or b"\r" in raw:
            errors.append(f"Assignment 08 fixture must use LF/final newline: {name}")

    keep_path = assignment_dir / "output" / ".gitkeep"
    if keep_path.is_file() and keep_path.read_bytes():
        errors.append("Assignment 08 output/.gitkeep must be empty")

    notebook_path = assignment_dir / "assignment.ipynb"
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5:
        errors.append("Assignment 08 notebook must use format 4.5")
    cells = notebook.get("cells", [])
    ids = [cell.get("id") for cell in cells if isinstance(cell, dict)]
    if ids != L8_ASSIGNMENT_CELL_IDS or len(ids) != len(set(ids)):
        errors.append("Assignment 08 notebook cell IDs/order changed")
        return
    if notebook.get("metadata", {}).get("kernelspec") != L4_PORTABLE_KERNELSPEC:
        errors.append("Assignment 08 notebook kernelspec is not portable")
    for cell in cells:
        expected_type = (
            "markdown" if cell["id"] in L8_ASSIGNMENT_MARKDOWN_IDS else "code"
        )
        if cell.get("cell_type") != expected_type:
            errors.append(f"Assignment 08 cell type changed: {cell['id']}")
        if expected_type == "code" and (
            cell.get("execution_count") is not None or cell.get("outputs")
        ):
            errors.append(f"stored execution state in Assignment 08: {cell['id']}")
        if cell["id"] in L8_ASSIGNMENT_PROTECTED_CELL_SHA256:
            source = cell.get("source", "")
            joined = "".join(source) if isinstance(source, list) else source
            observed = hashlib.sha256(joined.encode()).hexdigest()
            if observed != L8_ASSIGNMENT_PROTECTED_CELL_SHA256[cell["id"]]:
                errors.append(f"protected Assignment 08 cell changed: {cell['id']}")

    combined_guidance = "\n".join(
        (assignment_dir / name).read_text(encoding="utf-8")
        for name in ("README.md", "PLATFORM_CHECK.md")
        if (assignment_dir / name).is_file()
    )
    if "colab.research.google.com" in combined_guidance.lower():
        errors.append("Assignment 08 must not publish a Colab badge")
    if re.search(r"\b\d+\s*(?:minutes?|hours?)\b", combined_guidance, re.I):
        errors.append("Assignment 08 must not make a timing claim")
    for required in (
        "Assignment Colab is not supported",
        "Classroom50",
        "VS Code Source Control",
        "GitHub Desktop",
        "python check_assignment.py",
        "center_count_summary.csv",
        "center_summary.csv",
        "requests_with_context.csv",
        "center_channel_summary.csv",
        "mean_resolution_pivot.csv",
    ):
        if required not in combined_guidance:
            errors.append(f"missing Assignment 08 workflow contract: {required}")

    instructor_assets = (
        "_grader_selftest/autograder.py",
        "_grader_selftest/classroom50_grader.py",
        "_grader_selftest/run.py",
        "_grader_selftest/README.md",
    )
    for relative in instructor_assets:
        if not (assignment_dir / relative).is_file():
            errors.append(f"missing Assignment 08 instructor asset: {relative}")
    grader_path = assignment_dir / "_grader_selftest" / "classroom50_grader.py"
    if grader_path.is_file():
        grader_source = grader_path.read_text(encoding="utf-8")
        if re.search(r"(?:from|import)\s+check_assignment", grader_source):
            errors.append("Assignment 08 central grader must not import the public checker")
        for required in (
            '"classroom50/result/v1"',
            'Path("result.json")',
            '"REVIEW_URL"',
            "return 2",
        ):
            if required not in grader_source:
                errors.append(f"missing Assignment 08 grader contract: {required}")


def audit_assignment09(errors: list[str]) -> None:
    """Enforce the accepted Assignment 09 starter and central-grader surface."""

    assignment_dir = ROOT / "09" / "assignment"
    student_files = {
        ".gitignore",
        ".python-version",
        "PLATFORM_CHECK.md",
        "README.md",
        "assignment.ipynb",
        "check_assignment.py",
        "requirements.txt",
        "data/fixture.json",
        "data/zone_co2_readings.csv",
        "output/.gitkeep",
    }
    actual_student_files = {
        path.relative_to(assignment_dir).as_posix()
        for path in assignment_dir.rglob("*")
        if path.is_file()
        and "_grader_selftest" not in path.relative_to(assignment_dir).parts
        and "__pycache__" not in path.relative_to(assignment_dir).parts
    }
    if actual_student_files != student_files:
        missing = sorted(student_files - actual_student_files)
        unexpected = sorted(actual_student_files - student_files)
        if missing:
            errors.append(f"Assignment 09 student files missing: {', '.join(missing)}")
        if unexpected:
            errors.append(
                "unexpected Assignment 09 student files: " + ", ".join(unexpected)
            )

    expected_small_files = {
        ".python-version": "3.12.13\n",
        "requirements.txt": "numpy==2.0.2\npandas==3.0.3\n",
        ".gitignore": (
            ".ipynb_checkpoints/\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            ".pytest_cache/\n"
            ".venv/\n"
            "venv/\n"
        ),
        "_grader_selftest/requirements.txt": (
            "ipykernel==6.29.5\n"
            "nbclient==0.10.2\n"
            "nbformat==5.10.4\n"
            "numpy==2.0.2\n"
            "pandas==3.0.3\n"
        ),
    }
    for relative, expected in expected_small_files.items():
        path = assignment_dir / relative
        if not path.is_file():
            errors.append(f"missing Assignment 09 {relative}")
        elif path.read_text(encoding="utf-8") != expected:
            errors.append(f"unexpected Assignment 09 {relative}")

    for relative, expected_sha256 in L9_ASSIGNMENT_PROTECTED_FILE_SHA256.items():
        path = assignment_dir / relative
        if not path.is_file():
            errors.append(f"missing Assignment 09 protected file: {relative}")
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
            errors.append(f"protected Assignment 09 file changed: {relative}")

    expected_sizes = {"fixture.json": 473, "zone_co2_readings.csv": 380}
    for name, expected_sha256 in L9_ASSIGNMENT_FIXTURE_SHA256.items():
        path = assignment_dir / "data" / name
        if not path.is_file():
            errors.append(f"missing Assignment 09 fixture: {name}")
            continue
        raw = path.read_bytes()
        if len(raw) != expected_sizes[name]:
            errors.append(f"unexpected Assignment 09 fixture size: {name}")
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            errors.append(f"unexpected Assignment 09 fixture checksum: {name}")
        if not raw.endswith(b"\n") or b"\r" in raw:
            errors.append(f"Assignment 09 fixture must use LF/final newline: {name}")

    output_dir = assignment_dir / "output"
    output_files = {
        path.name for path in output_dir.iterdir() if path.is_file()
    } if output_dir.is_dir() else set()
    if output_files != {".gitkeep"}:
        errors.append("Assignment 09 starter output must contain only .gitkeep")
    elif (output_dir / ".gitkeep").read_bytes():
        errors.append("Assignment 09 output/.gitkeep must be empty")

    notebook_path = assignment_dir / "assignment.ipynb"
    try:
        notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return
    if notebook.get("nbformat") != 4 or notebook.get("nbformat_minor") != 5:
        errors.append("Assignment 09 notebook must use format 4.5")
    cells = notebook.get("cells", [])
    ids = [cell.get("id") for cell in cells if isinstance(cell, dict)]
    if ids != L9_ASSIGNMENT_CELL_IDS or len(ids) != len(set(ids)):
        errors.append("Assignment 09 notebook cell IDs/order changed")
        return
    if notebook.get("metadata", {}).get("kernelspec") != L4_PORTABLE_KERNELSPEC:
        errors.append("Assignment 09 notebook kernelspec is not portable")
    for cell in cells:
        expected_type = (
            "markdown" if cell["id"] in L9_ASSIGNMENT_MARKDOWN_IDS else "code"
        )
        if cell.get("cell_type") != expected_type:
            errors.append(f"Assignment 09 cell type changed: {cell['id']}")
        if expected_type == "code" and (
            cell.get("execution_count") is not None or cell.get("outputs")
        ):
            errors.append(f"stored execution state in Assignment 09: {cell['id']}")
        if cell["id"] in L9_ASSIGNMENT_PROTECTED_CELL_SHA256:
            source = cell.get("source", "")
            joined = "".join(source) if isinstance(source, list) else source
            observed = hashlib.sha256(joined.encode()).hexdigest()
            if observed != L9_ASSIGNMENT_PROTECTED_CELL_SHA256[cell["id"]]:
                errors.append(f"protected Assignment 09 cell changed: {cell['id']}")

    combined_guidance = "\n".join(
        (assignment_dir / name).read_text(encoding="utf-8")
        for name in ("README.md", "PLATFORM_CHECK.md")
        if (assignment_dir / name).is_file()
    )
    if "colab.research.google.com" in combined_guidance.lower():
        errors.append("Assignment 09 must not publish a Colab badge")
    if re.search(r"\b\d+\s*(?:minutes?|hours?)\b", combined_guidance, re.I):
        errors.append("Assignment 09 must not make a timing claim")
    for required in (
        "Assignment Colab is not supported",
        "Classroom50",
        "VS Code Source Control",
        "GitHub Desktop",
        "python check_assignment.py",
        "prepared_panel.csv",
        "hourly_grid.csv",
        "two_hour_summary.csv",
        "temporal_features.csv",
        "availability_decisions.csv",
        "chronological_blocks.csv",
    ):
        if required not in combined_guidance:
            errors.append(f"missing Assignment 09 workflow contract: {required}")

    instructor_assets = (
        "_grader_selftest/autograder.py",
        "_grader_selftest/classroom50_grader.py",
        "_grader_selftest/run.py",
        "_grader_selftest/README.md",
    )
    for relative in instructor_assets:
        if not (assignment_dir / relative).is_file():
            errors.append(f"missing Assignment 09 instructor asset: {relative}")
    grader_path = assignment_dir / "_grader_selftest" / "classroom50_grader.py"
    if grader_path.is_file():
        grader_source = grader_path.read_text(encoding="utf-8")
        if re.search(r"(?:from|import)\s+check_assignment", grader_source):
            errors.append("Assignment 09 central grader must not import the public checker")
        for required in (
            '"classroom50/result/v1"',
            'Path("result.json")',
            '"CLASSROOM"',
            '"ASSIGNMENT"',
            '"SUBMISSION_TAG"',
            '"COMMIT_URL"',
            '"RELEASE_URL"',
            'os.environ.get("REVIEW_URL", "").strip() or result["commit"]',
            "return 2",
        ):
            if required not in grader_source:
                errors.append(f"missing Assignment 09 grader contract: {required}")

    expected_pep_dependencies = {
        '"ipykernel==6.29.5"',
        '"nbclient==0.10.2"',
        '"nbformat==5.10.4"',
        '"numpy==2.0.2"',
        '"pandas==3.0.3"',
    }
    for relative in (
        "_grader_selftest/classroom50_grader.py",
        "_grader_selftest/run.py",
    ):
        path = assignment_dir / relative
        if path.is_file():
            source = path.read_text(encoding="utf-8")
            if '# requires-python = "==3.12.13"' not in source:
                errors.append(f"missing Assignment 09 PEP 723 Python pin: {relative}")
            missing = sorted(
                dependency
                for dependency in expected_pep_dependencies
                if dependency not in source
            )
            if missing:
                errors.append(f"missing Assignment 09 PEP 723 dependencies: {relative}")
    public_source = (assignment_dir / "check_assignment.py").read_text(encoding="utf-8")
    if (
        '# requires-python = "==3.12.13"' not in public_source
        or '"numpy==2.0.2"' not in public_source
        or '"pandas==3.0.3"' not in public_source
    ):
        errors.append("Assignment 09 public checker PEP 723 metadata differs")


def audit_classroom50_runner_contract(errors: list[str]) -> None:
    """Reject local-default or invented runner context in implemented graders."""

    graders = {
        "Assignment 04": ROOT / "04/assignment/_grader_selftest/autograder.py",
        "Assignment 05": ROOT / "05/assignment/_grader_selftest/classroom50_grader.py",
        "Assignment 06": ROOT / "06/assignment/_grader_selftest/classroom50_grader.py",
        "Assignment 07": ROOT / "07/assignment/_grader_selftest/classroom50_grader.py",
        "Assignment 08": ROOT / "08/assignment/_grader_selftest/classroom50_grader.py",
        "Assignment 09": ROOT / "09/assignment/_grader_selftest/classroom50_grader.py",
    }
    required = (
        '"CLASSROOM"',
        '"ASSIGNMENT"',
        '"SUBMISSION_TAG"',
        '"COMMIT_URL"',
        '"RELEASE_URL"',
        '"REVIEW_URL"',
        "datetime.timezone.utc",
        '"classroom50/result/v1"',
    )
    forbidden = ("CLASSROOM50_", "example.invalid")
    for label, path in graders.items():
        if not path.is_file():
            errors.append(f"missing {label} central grader: {path.relative_to(ROOT)}")
            continue
        source = path.read_text(encoding="utf-8")
        for fragment in required:
            if fragment not in source:
                errors.append(f"missing {label} runner contract: {fragment}")
        for fragment in forbidden:
            if fragment in source:
                errors.append(f"invented/local-default {label} runner contract: {fragment}")

    assignment07 = graders["Assignment 07"].read_text(encoding="utf-8")
    for fragment in ("CLASSROOM50_REVIEW_DIR", "_write_review_bundle", "review-bundle/v1"):
        if fragment in assignment07:
            errors.append(f"invented Assignment 07 review storage remains: {fragment}")

    expected_bundle_files = {
        "04": {
            "README.md", "alternate_fixture/fixture.json",
            "alternate_fixture/purchases.csv", "autograder.py", "grader_core.py",
            "protected_files.json", "requirements.txt", "run.py",
        },
        **{
            assignment: {
                "README.md", "autograder.py", "classroom50_grader.py",
                "requirements.txt", "run.py",
            }
            for assignment in ("05", "06", "07", "08", "09")
        },
    }
    inventory_fragments = {
        '".classroom50.yaml"',
        '".github/workflows/autograde.yaml"',
        'parts[0] != ".git"',
        '"_grader_selftest/copied.py"',
        '".github/workflows/extra.yaml"',
        '"notes.txt"',
        '"ordinary/.git/nested.txt"',
    }
    for assignment, expected_files in expected_bundle_files.items():
        bundle = ROOT / assignment / "assignment/_grader_selftest"
        actual_files = {
            path.relative_to(bundle).as_posix()
            for path in bundle.rglob("*")
            if path.is_file() and "__pycache__" not in path.relative_to(bundle).parts
        }
        if actual_files != expected_files:
            errors.append(f"Assignment {assignment} production bundle inventory differs")

        bootstrap = bundle / "autograder.py"
        if bootstrap.is_file():
            source = bootstrap.read_text(encoding="utf-8")
            for fragment in (
                '# requires-python = "==3.12.13"',
                "# dependencies = []",
                "sys.executable",
                '"-m"',
                '"pip"',
                '"requirements.txt"',
                '"result.json"',
            ):
                if fragment not in source:
                    errors.append(
                        f"missing Assignment {assignment} bootstrap contract: {fragment}"
                    )

        public_source = (ROOT / assignment / "assignment/check_assignment.py").read_text(encoding="utf-8")
        grader_name = "grader_core.py" if assignment == "04" else "classroom50_grader.py"
        grader_source = (bundle / grader_name).read_text(encoding="utf-8")
        harness_source = (bundle / "run.py").read_text(encoding="utf-8")
        combined_inventory_source = public_source + grader_source + harness_source
        for fragment in inventory_fragments:
            if fragment not in combined_inventory_source:
                errors.append(
                    f"missing Assignment {assignment} repository-inventory probe: {fragment}"
                )
        if "autograder.py" not in harness_source:
            errors.append(
                f"Assignment {assignment} harness does not invoke production autograder.py"
            )


def audit_lecture08_demos(errors: list[str]) -> None:
    """Enforce the accepted grain, aggregation, and pivot demo contract."""

    demo_dir = ROOT / "08" / "demo"
    expected_files = {
        ".gitignore",
        ".python-version",
        "DEMO_GUIDE.md",
        "requirements.txt",
        "data/encounters.csv",
        *L8_DEMO_NOTEBOOKS,
    }
    actual_files = {
        path.relative_to(demo_dir).as_posix()
        for path in demo_dir.rglob("*")
        if path.is_file() and "output" not in path.relative_to(demo_dir).parts
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        if missing:
            errors.append(f"Lecture 08 demo files missing: {', '.join(missing)}")
        if unexpected:
            errors.append(
                f"unexpected Lecture 08 demo files: {', '.join(unexpected)}"
            )

    fixture = demo_dir / "data" / "encounters.csv"
    if not fixture.is_file():
        errors.append("missing Lecture 08 fixture: 08/demo/data/encounters.csv")
    else:
        actual_sha256 = hashlib.sha256(fixture.read_bytes()).hexdigest()
        if actual_sha256 != L8_ENCOUNTERS_SHA256:
            errors.append(
                "unexpected Lecture 08 encounters.csv checksum: "
                f"{actual_sha256}"
            )

    expected_small_files = {
        ".python-version": "3.12.13\n",
        "requirements.txt": "numpy==2.0.2\npandas==3.0.3\n",
        ".gitignore": (
            ".ipynb_checkpoints/\n"
            "output/\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            ".venv/\n"
            "venv/\n"
            "env/\n"
        ),
    }
    for filename, expected_text in expected_small_files.items():
        path = demo_dir / filename
        if not path.is_file():
            errors.append(f"missing Lecture 08 demo {filename}")
        elif path.read_text(encoding="utf-8") != expected_text:
            errors.append(f"unexpected Lecture 08 demo {filename}")

    tracked_output_check = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", "08/demo/output"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked_output_check.returncode != 0:
        errors.append("unable to verify tracked Lecture 08 demo output")
    elif tracked_output_check.stdout.strip():
        errors.append("generated Lecture 08 demo output must not be committed")

    guide_path = demo_dir / "DEMO_GUIDE.md"
    if guide_path.is_file():
        guide_text = guide_path.read_text(encoding="utf-8")
        required_guide_fragments = {
            *L8_DEMO_NOTEBOOKS,
            L8_ENCOUNTERS_SHA256,
            "Learner prediction checkpoints",
            "Expected outcome",
            "not automatically saved back",
            "Independent local candidate",
            "Fresh Colab runtime | pending",
            "Immutable release-tag badge | pending",
            "GroupBy `size()`",
            "GroupBy **transform**",
            "structural `pivot`",
            "aggregating `pivot_table`",
        }
        missing_guide_fragments = sorted(
            fragment
            for fragment in required_guide_fragments
            if fragment not in guide_text
        )
        if missing_guide_fragments:
            errors.append(
                "missing Lecture 08 guide contract: "
                + ", ".join(missing_guide_fragments)
            )
        if re.search(r"\binstructor\b|\bnow discuss\b", guide_text, re.I):
            errors.append(
                "Lecture 08 guide must use actionable learner checkpoints, "
                "not instructor talking points"
            )

    global_cell_ids: set[str] = set()
    actual_notebooks = {path.name for path in demo_dir.glob("*.ipynb")}
    for name in sorted(L8_DEMO_NOTEBOOKS & actual_notebooks):
        path = demo_dir / name
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            # The repository-wide notebook parser reports the detailed error.
            continue

        kernelspec = notebook.get("metadata", {}).get("kernelspec")
        if kernelspec != L4_PORTABLE_KERNELSPEC:
            errors.append(
                f"non-portable Lecture 08 kernelspec: {path.relative_to(ROOT)}"
            )

        cells = notebook.get("cells", [])
        cell_ids = [cell.get("id") for cell in cells]
        if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(
            set(cell_ids)
        ):
            errors.append(
                f"missing or duplicate Lecture 08 cell id: {path.relative_to(ROOT)}"
            )
        duplicate_global_ids = sorted(set(cell_ids) & global_cell_ids)
        if duplicate_global_ids:
            errors.append(
                "Lecture 08 cell IDs must be globally unique: "
                + ", ".join(duplicate_global_ids)
            )
        global_cell_ids.update(cell_id for cell_id in cell_ids if cell_id)

        first_source = ""
        if cells:
            source = cells[0].get("source", [])
            first_source = "".join(source) if isinstance(source, list) else source
        landing_fragments = {
            "Learning question",
            "one recorded encounter per row",
            "Colab-first",
            "ephemeral",
            "not automatically saved back",
            "stored output is not execution evidence",
            "Assignment Colab support remains conditional",
        }
        if cells and cells[0].get("cell_type") != "markdown":
            errors.append(
                f"Lecture 08 landing cell must be Markdown: {path.relative_to(ROOT)}"
            )
        missing_landing = sorted(
            fragment for fragment in landing_fragments if fragment not in first_source
        )
        if missing_landing:
            errors.append(
                f"missing Lecture 08 landing contract in {path.relative_to(ROOT)}: "
                + ", ".join(missing_landing)
            )

        code_source: list[str] = []
        markdown_source: list[str] = []
        for cell in cells:
            source = cell.get("source", [])
            source_text = "".join(source) if isinstance(source, list) else source
            if cell.get("cell_type") == "markdown":
                markdown_source.append(source_text)
                continue
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(
                    "stored execution state in Lecture 08 demo: "
                    f"{path.relative_to(ROOT)}"
                )
                break
            code_source.append(source_text)

        joined_source = "\n".join(code_source)
        joined_markdown = "\n".join(markdown_source)
        for version in ("3.12.13", "2.0.2", "3.0.3"):
            if version not in joined_source:
                errors.append(
                    f"missing Lecture 08 candidate {version}: {path.relative_to(ROOT)}"
                )
        for fragment in (
            L8_ENCOUNTERS_SHA256,
            'lineterminator="\\n"',
            'encoding="utf-8"',
            "observed=True",
            "sort=True",
            "dropna=True",
            'DEMO_DIRECTORY / "output"',
        ):
            if fragment not in joined_source:
                errors.append(
                    f"missing Lecture 08 execution contract {fragment}: "
                    f"{path.relative_to(ROOT)}"
                )
        for output_name in L8_NOTEBOOK_OUTPUTS[name]:
            if output_name not in joined_source:
                errors.append(
                    f"missing Lecture 08 output {output_name}: "
                    f"{path.relative_to(ROOT)}"
                )
        for label, pattern in L8_SCOPE_PATTERNS.items():
            if pattern.search(joined_source):
                errors.append(
                    f"Lecture 08 scope violation ({label}): {path.relative_to(ROOT)}"
                )

        groupby_count = len(re.findall(r"\.groupby\s*\(", joined_source))
        pivot_table_count = len(
            re.findall(r"\bpd\.pivot_table\s*\(", joined_source)
        )
        concat_count = len(re.findall(r"\bpd\.concat\s*\(", joined_source))
        if name == "demo1_grouping_grain_counts.ipynb":
            if groupby_count != 1 or pivot_table_count or concat_count != 1:
                errors.append("unexpected Lecture 08 Demo 1 operation count")
            term_fragments = {
                "grouping unit",
                "Output row grain",
                "GroupBy object",
            }
        elif name == "demo2_named_aggregation_transform.ipynb":
            if groupby_count != 3 or pivot_table_count or concat_count:
                errors.append("unexpected Lecture 08 Demo 2 operation count")
            term_fragments = {
                "Named aggregation",
                "GroupBy **transform**",
                "two-key group",
            }
        else:
            if groupby_count != 1 or pivot_table_count != 1 or concat_count:
                errors.append("unexpected Lecture 08 Demo 3 operation count")
            term_fragments = {
                "structural `pivot`",
                "aggregating **`pivot_table`**",
                "five required choices",
            }
        missing_terms = sorted(
            fragment for fragment in term_fragments if fragment not in joined_markdown
        )
        if missing_terms:
            errors.append(
                f"missing Lecture 08 term definition in {path.relative_to(ROOT)}: "
                + ", ".join(missing_terms)
            )


def audit_lecture09_demos(errors: list[str]) -> None:
    """Enforce the accepted entity-aware temporal demo contract."""

    demo_dir = ROOT / "09" / "demo"
    expected_files = {
        ".gitignore",
        ".python-version",
        "DEMO_GUIDE.md",
        "requirements.txt",
        "data/station_observations.csv",
        *L9_DEMO_NOTEBOOKS,
    }
    actual_files = {
        path.relative_to(demo_dir).as_posix()
        for path in demo_dir.rglob("*")
        if path.is_file() and "output" not in path.relative_to(demo_dir).parts
    }
    if actual_files != expected_files:
        missing = sorted(expected_files - actual_files)
        unexpected = sorted(actual_files - expected_files)
        if missing:
            errors.append(f"Lecture 09 demo files missing: {', '.join(missing)}")
        if unexpected:
            errors.append(
                f"unexpected Lecture 09 demo files: {', '.join(unexpected)}"
            )

    fixture = demo_dir / "data" / "station_observations.csv"
    if not fixture.is_file():
        errors.append(
            "missing Lecture 09 fixture: 09/demo/data/station_observations.csv"
        )
    else:
        fixture_bytes = fixture.read_bytes()
        actual_sha256 = hashlib.sha256(fixture_bytes).hexdigest()
        if len(fixture_bytes) != 310:
            errors.append(
                f"unexpected Lecture 09 station fixture size: {len(fixture_bytes)}"
            )
        if actual_sha256 != L9_STATIONS_SHA256:
            errors.append(
                "unexpected Lecture 09 station_observations.csv checksum: "
                f"{actual_sha256}"
            )

    expected_small_files = {
        ".python-version": "3.12.13\n",
        "requirements.txt": "numpy==2.0.2\npandas==3.0.3\n",
        ".gitignore": (
            ".ipynb_checkpoints/\n"
            "output/\n"
            "__pycache__/\n"
            "*.py[cod]\n"
            ".venv/\n"
            "venv/\n"
            "env/\n"
        ),
    }
    for filename, expected_text in expected_small_files.items():
        path = demo_dir / filename
        if not path.is_file():
            errors.append(f"missing Lecture 09 demo {filename}")
        elif path.read_text(encoding="utf-8") != expected_text:
            errors.append(f"unexpected Lecture 09 demo {filename}")

    tracked_output_check = subprocess.run(
        ["git", "-C", str(ROOT), "ls-files", "--", "09/demo/output"],
        capture_output=True,
        text=True,
        check=False,
    )
    if tracked_output_check.returncode != 0:
        errors.append("unable to verify tracked Lecture 09 demo output")
    elif tracked_output_check.stdout.strip():
        errors.append("generated Lecture 09 demo output must not be committed")

    guide_path = demo_dir / "DEMO_GUIDE.md"
    if guide_path.is_file():
        guide_text = guide_path.read_text(encoding="utf-8")
        required_guide_fragments = {
            *L9_DEMO_NOTEBOOKS,
            L9_STATIONS_SHA256,
            "Learner prediction checkpoints",
            "Expected outcome",
            "not automatically saved back",
            "Independent local candidate | 2026-07-18",
            "nbclient 0.10.2",
            "PASS; see independent evidence",
            "Fresh Colab runtime | pending",
            "Immutable release-tag badge | pending",
            "timestamp",
            "period",
            "single series",
            "panel",
            "Localization",
            "conversion",
            "`asfreq()`",
            "`resample()`",
            "Source-value missingness",
            "grid-created row",
            "observation-count window",
            "elapsed-time",
            "Information availability",
            "future leakage",
            "seven `earlier` rows and three `later_holdout` rows",
        }
        for outputs in L9_NOTEBOOK_OUTPUTS.values():
            for output_name, (expected_size, expected_sha256) in outputs.items():
                required_guide_fragments.update(
                    {output_name, str(expected_size), expected_sha256}
                )
        missing_guide_fragments = sorted(
            fragment
            for fragment in required_guide_fragments
            if fragment not in guide_text
        )
        if missing_guide_fragments:
            errors.append(
                "missing Lecture 09 guide contract: "
                + ", ".join(missing_guide_fragments)
            )
        if re.search(
            r"\binstructor\b|\bnow discuss\b|\bduration\b|\bminutes?\b",
            guide_text,
            re.I,
        ):
            errors.append(
                "Lecture 09 guide must use direct learner actions without "
                "instructor, meta, or lesson-duration language"
            )

    other_cell_ids: dict[str, Path] = {}
    for other_path in ROOT.rglob("*.ipynb"):
        if other_path.parent == demo_dir:
            continue
        try:
            other_notebook = json.loads(other_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        for cell in other_notebook.get("cells", []):
            cell_id = cell.get("id")
            if cell_id:
                other_cell_ids.setdefault(cell_id, other_path)

    lecture_cell_ids: set[str] = set()
    actual_notebooks = {path.name for path in demo_dir.glob("*.ipynb")}
    for name in sorted(L9_DEMO_NOTEBOOKS & actual_notebooks):
        path = demo_dir / name
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue

        kernelspec = notebook.get("metadata", {}).get("kernelspec")
        if kernelspec != L4_PORTABLE_KERNELSPEC:
            errors.append(
                f"non-portable Lecture 09 kernelspec: {path.relative_to(ROOT)}"
            )

        cells = notebook.get("cells", [])
        cell_ids = [cell.get("id") for cell in cells]
        if any(not cell_id for cell_id in cell_ids) or len(cell_ids) != len(
            set(cell_ids)
        ):
            errors.append(
                f"missing or duplicate Lecture 09 cell id: {path.relative_to(ROOT)}"
            )
        duplicate_lecture_ids = sorted(set(cell_ids) & lecture_cell_ids)
        if duplicate_lecture_ids:
            errors.append(
                "Lecture 09 cell IDs must be unique across its notebooks: "
                + ", ".join(duplicate_lecture_ids)
            )
        duplicate_repository_ids = sorted(set(cell_ids) & set(other_cell_ids))
        if duplicate_repository_ids:
            collisions = ", ".join(
                f"{cell_id} ({other_cell_ids[cell_id].relative_to(ROOT)})"
                for cell_id in duplicate_repository_ids
            )
            errors.append(
                "Lecture 09 cell IDs must be globally unique: " + collisions
            )
        lecture_cell_ids.update(cell_id for cell_id in cell_ids if cell_id)

        first_source = ""
        if cells:
            source = cells[0].get("source", [])
            first_source = "".join(source) if isinstance(source, list) else source
        landing_fragments = {
            "Learning question",
            "grain",
            "Colab-first",
            "ephemeral",
            "not automatically saved back",
            "stored output is not execution evidence",
            "Assignment Colab support remains conditional",
            "synthetic, non-identifying",
        }
        if cells and cells[0].get("cell_type") != "markdown":
            errors.append(
                f"Lecture 09 landing cell must be Markdown: {path.relative_to(ROOT)}"
            )
        missing_landing = sorted(
            fragment for fragment in landing_fragments if fragment not in first_source
        )
        if missing_landing:
            errors.append(
                f"missing Lecture 09 landing contract in {path.relative_to(ROOT)}: "
                + ", ".join(missing_landing)
            )

        code_source: list[str] = []
        markdown_source: list[str] = []
        for cell in cells:
            source = cell.get("source", [])
            source_text = "".join(source) if isinstance(source, list) else source
            if cell.get("cell_type") == "markdown":
                markdown_source.append(source_text)
                continue
            if cell.get("cell_type") != "code":
                continue
            if cell.get("execution_count") is not None or cell.get("outputs"):
                errors.append(
                    "stored execution state in Lecture 09 demo: "
                    f"{path.relative_to(ROOT)}"
                )
                break
            code_source.append(source_text)

        joined_source = "\n".join(code_source)
        joined_markdown = "\n".join(markdown_source)
        for version in ("3.12.13", "2.0.2", "3.0.3"):
            if version not in joined_source:
                errors.append(
                    f"missing Lecture 09 candidate {version}: {path.relative_to(ROOT)}"
                )
        for fragment in (
            L9_STATIONS_SHA256,
            'lineterminator="\\n"',
            'encoding="utf-8"',
            "observed=True",
            "sort=True",
            "dropna=True",
            'DEMO_DIRECTORY / "output"',
            'tz_localize("America/Los_Angeles")',
            'tz_convert("UTC")',
            'kind="stable"',
        ):
            if fragment not in joined_source:
                errors.append(
                    f"missing Lecture 09 execution contract {fragment}: "
                    f"{path.relative_to(ROOT)}"
                )
        if joined_source.count(".tz_localize(") != 1:
            errors.append(
                f"Lecture 09 must localize exactly once: {path.relative_to(ROOT)}"
            )
        if joined_source.count(".tz_convert(") != 1:
            errors.append(
                f"Lecture 09 must convert timezone exactly once: {path.relative_to(ROOT)}"
            )
        for output_name, (expected_size, expected_sha256) in L9_NOTEBOOK_OUTPUTS[
            name
        ].items():
            for fragment in (output_name, str(expected_size), expected_sha256):
                if fragment not in joined_source:
                    errors.append(
                        f"missing Lecture 09 output contract {fragment}: "
                        f"{path.relative_to(ROOT)}"
                    )
        for label, pattern in L9_SCOPE_PATTERNS.items():
            if pattern.search(joined_source):
                errors.append(
                    f"Lecture 09 scope violation ({label}): {path.relative_to(ROOT)}"
                )

        resample_count = len(re.findall(r"\.resample\s*\(", joined_source))
        asfreq_count = len(re.findall(r"\.asfreq\s*\(", joined_source))
        rolling_count = len(re.findall(r"\.rolling\s*\(", joined_source))
        shift_count = len(re.findall(r"\.shift\s*\(", joined_source))
        diff_count = len(re.findall(r"\.diff\s*\(", joined_source))
        if name == "demo1_temporal_structure.ipynb":
            if resample_count or asfreq_count or rolling_count or shift_count:
                errors.append("unexpected Lecture 09 Demo 1 temporal operation")
            required_terms = {
                "**timestamp**",
                "**period**",
                "**entity key**",
                "**single series**",
                "**panel**",
                "**regular**",
                "**irregular**",
                "**Localization**",
                "**conversion**",
                "**DatetimeIndex**",
            }
            required_code = {
                'pd.Period("2026-01-15", freq="D")',
                'freq="h"',
                'prepared.set_index("observed_at")',
                '"south": [2.0, 1.0, 2.0, 1.0]',
            }
        elif name == "demo2_frequency_measurement.ipynb":
            if (
                resample_count != 2
                or asfreq_count != 1
                or rolling_count
                or shift_count
            ):
                errors.append("unexpected Lecture 09 Demo 2 temporal operation count")
            required_terms = {
                "**Upsampling**",
                "**downsampling**",
                "**Source-value missingness**",
                "**grid-created row**",
                "**Measurement meaning**",
                "**left-closed, left-labeled bin**",
            }
            required_code = {
                '.resample("h")',
                ".asfreq()",
                '.resample("2h", closed="left", label="left")',
                'hourly_grid["source_row"].isna()',
                'hourly_grid["source_row"].eq(1)',
                'reading_count=("source_row", "sum")',
            }
        else:
            if (
                resample_count
                or asfreq_count
                or rolling_count != 2
                or shift_count != 2
                or diff_count != 1
            ):
                errors.append("unexpected Lecture 09 Demo 3 temporal operation count")
            required_terms = {
                "**lag**",
                "**lead**",
                "**difference**",
                "**observation-count window**",
                "**elapsed-time window**",
                "**prediction timestamp**",
                "**Information availability**",
                "**future leakage**",
                "**chronological holdout**",
            }
            required_code = {
                'by_station.shift(1)',
                'by_station.diff()',
                '.rolling(window=2, min_periods=1)',
                '.rolling("2h", closed="left", min_periods=1)',
                'validate="one_to_one"',
                '"keep", "keep", "reject", "reject"',
                '"earlier"',
                '"later_holdout"',
            }
        missing_terms = sorted(
            fragment for fragment in required_terms if fragment not in joined_markdown
        )
        if missing_terms:
            errors.append(
                f"missing Lecture 09 term definition in {path.relative_to(ROOT)}: "
                + ", ".join(missing_terms)
            )
        missing_code = sorted(
            fragment for fragment in required_code if fragment not in joined_source
        )
        if missing_code:
            errors.append(
                f"missing Lecture 09 implementation contract in {path.relative_to(ROOT)}: "
                + ", ".join(missing_code)
            )


def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    notebook_count = 0

    for lecture in LECTURES:
        if not (lecture / "README.md").is_file():
            errors.append(f"missing lecture README: {lecture.relative_to(ROOT)}")
        if not (lecture / "demo").is_dir():
            errors.append(f"missing demo directory: {lecture.relative_to(ROOT)}")
        if lecture.name != "01" and not (lecture / "assignment").is_dir():
            errors.append(f"missing assignment directory: {lecture.relative_to(ROOT)}")

        for notebook in lecture.rglob("*.ipynb"):
            notebook_count += 1
            try:
                data = json.loads(notebook.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, json.JSONDecodeError) as exc:
                errors.append(f"invalid notebook {notebook.relative_to(ROOT)}: {exc}")
                continue
            if data.get("nbformat") != 4:
                warnings.append(f"unexpected notebook format: {notebook.relative_to(ROOT)}")
            if not data.get("metadata", {}).get("kernelspec"):
                warnings.append(f"missing kernelspec: {notebook.relative_to(ROOT)}")

    audit_lecture04_demos(errors)
    audit_lecture05_demos(errors)
    audit_lecture06_demos(errors)
    audit_lecture07_demos(errors)
    audit_assignment07(errors)
    audit_lecture08_demos(errors)
    audit_assignment08(errors)
    audit_classroom50_runner_contract(errors)
    audit_lecture09_demos(errors)
    audit_assignment09(errors)

    text_suffixes = {".md", ".html", ".yml", ".yaml", ".js"}
    platform_pattern = re.compile(r"classroom\.github\.com|GitHub Classroom|ds217_25f", re.I)
    attachment_pattern = re.compile(r"(?:src=[\"']|\]\()attachment:", re.I)

    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in text_suffixes:
            continue
        if any(part in {".git", "node_modules", "_site", "dist", "work"} for part in path.parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeError:
            continue
        if platform_pattern.search(text):
            warnings.append(f"legacy Classroom reference: {path.relative_to(ROOT)}")
        if attachment_pattern.search(text):
            warnings.append(f"unresolved attachment URL: {path.relative_to(ROOT)}")

    print(f"Lectures checked: {len(LECTURES)}")
    print(f"Notebooks parsed: {notebook_count}")
    print(f"Errors: {len(errors)}")
    for item in errors:
        print(f"ERROR: {item}")
    print(f"Warnings: {len(warnings)}")
    for item in sorted(set(warnings)):
        print(f"WARN: {item}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
