"""Run the Lecture 03 demos the way a student does and check the guide's expected output against real runs.

setup_demo.sh runs through `curl | sh` with HOME pointed at a scratch folder. The script the students run
downloads from GitHub, so the test runs a scratch copy whose `base_url` names this repository's 03/demo
instead: it downloads the working-tree files and never touches the real home folder.
Demo 1's environment commands then run with uv exactly as the guide gives them, and every demo script
runs from the resulting ~/03-demo with the Python of the environment Demo 1 built. uv reuses the real
uv cache and installed Pythons, and `uv add` needs the package index, as it does for students. Outside
the environment, `python` is the uv-managed Python 3.13 that Lecture 01 installs, with no NumPy.

    python3 scripts/test_lecture03_demos.py
"""

from pathlib import Path
import os
import re
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "03" / "demo"
GUIDE_TEXT = (DEMOS / "DEMO_GUIDE.md").read_text(encoding="utf-8")
SETUP_COMMAND = (
    "curl -fsSL https://raw.githubusercontent.com/christopherseaman/datasci_217/main/03/demo/setup_demo.sh | sh"
)
PYTHON_SCRIPTS = (
    "demo2_types_and_lists.py",
    "demo2_numpy_performance.py",
    "demo2_numpy_arrays.py",
    "demo3_bp_analysis.py",
    "demo3_csv_summary.py",
)
# Each script, and the span of DEMO_GUIDE.md whose ```text blocks must account for every line it
# prints. Demo 1's span ends before the guide shows the summary and the log, because those are
# files it writes rather than lines it prints.
SPANS = {
    "setup_demo.sh": ("## 1.1 Download the Demo Files", "```bash\ncd ~/03-demo"),
    "demo1_cli_pipeline.sh": ("## 1.6 Run a Pipeline Script", "Open the saved summary and log:"),
    "count_clinics.sh": ("## 1.7 Save a Pipeline as a Script", "# Demo 2:"),
    "demo2_types_and_lists.py": ("## 2.1 Check Types", "## 2.2 Compare a List Loop"),
    "demo2_numpy_performance.py": ("## 2.2 Compare a List Loop", "## 2.3 Data Types"),
    "demo2_numpy_arrays.py": ("## 2.3 Data Types", "# Demo 3:"),
    "demo3_bp_analysis.py": ("# Demo 3:", "## 3.4 Summarize the Bundled CSV by Clinic"),
    "demo3_csv_summary.py": ("## 3.4 Summarize the Bundled CSV by Clinic", "### Check the counts against the shell"),
}
# What Demo 1's environment commands print, as the guide's comments state it; `x` stands for any digit.
ENVIRONMENT_OUTPUT = (
    "Pinned `.python-version` to `3.13`",
    "Initialized project `03-demo`",
    "Creating virtual environment with seed packages at: .venv",
    "Using CPython 3.13.x",
    "+ numpy==2.3.3",
    "Python 3.13.x",
)
# The command Demo 1.3 runs with the environment off, to show the lecture's ModuleNotFoundError.
EXPECTED_ERROR_COMMAND = 'python -c "import numpy as np"\n'


TIMESTAMP = re.compile(r"\d{8}_\d{6}")


def normalize(text):
    """Blank out what legitimately differs between runs: the run timestamp and the timings.

    This makes the guide's blocks comparable on any machine, and it is deliberately blind: every
    timestamp becomes one token and every measurement becomes one word. What those values have to
    be true of is checked separately, by captured_timestamp() and by the Demo 2 timing checks.
    """
    text = TIMESTAMP.sub("TIMESTAMP", text)
    text = re.sub(r"\d+\.\d+ ms", "ELAPSED ms", text)
    return re.sub(r"\d+\.\d+x faster", "SPEEDUPx faster", text)


def captured_timestamp(path, summary, log_lines, stdout):
    """The one timestamp a Demo 1 run captured, checked in every place the guide shows it.

    demo1_cli_pipeline.sh reads the clock once into `timestamp` and reuses it, which is the point
    of the demo: the guide says "One captured timestamp names the result file and labels both log
    lines", and shows the same value three times in the log block. normalize() cannot tell one
    captured timestamp from three separate clock reads, so compare the raw values here.
    """
    stamp = path.stem.removeprefix("summary_")
    assert TIMESTAMP.fullmatch(stamp), path.name
    assert TIMESTAMP.findall(summary) == [stamp], summary
    assert TIMESTAMP.findall("\n".join(log_lines)) == [stamp, stamp, stamp], log_lines
    assert TIMESTAMP.findall(stdout) == [stamp], stdout
    return stamp


def span(first_heading, last_heading):
    """The guide between two of its headings."""
    start = GUIDE_TEXT.index(first_heading)
    return GUIDE_TEXT[start:GUIDE_TEXT.index(last_heading, start)]


def fenced(text, language):
    """The contents of every ```language block in text."""
    return re.findall(rf"^```{language}\n(.*?)^```$", text, re.M | re.S)


def guide_blocks(first_heading=None, last_heading=None):
    """The guide's ```text blocks, normalized, or only those between two of its headings."""
    text = GUIDE_TEXT if first_heading is None else span(first_heading, last_heading)
    return fenced(normalize(text), "text")


def block_start(lines, block, position):
    """Where the block sits in lines as an exact consecutive run at or after position, or -1."""
    wanted = block.splitlines()
    for start in range(position, len(lines) - len(wanted) + 1):
        if lines[start:start + len(wanted)] == wanted:
            return start
    return -1


def quotes(output, block):
    """True when the block is an exact consecutive run of the output's lines."""
    return block_start(output.splitlines(), block, 0) >= 0


def accounts_for(output, blocks):
    """True when the blocks are exact consecutive runs, in guide order, covering every line.

    Only blank lines may fall between one block and the next, so an output line the guide never
    shows, or one it shows differently, fails the check.
    """
    lines = output.splitlines()
    position = 0
    for block in blocks:
        start = block_start(lines, block, position)
        if start < 0 or any(line.strip() for line in lines[position:start]):
            return False
        position = start + len(block.splitlines())
    return not any(line.strip() for line in lines[position:])


def uv_location(*command):
    """A directory uv reports for the real user, such as its cache, read before HOME moves."""
    return subprocess.run(["uv", *command], check=True, capture_output=True, text=True).stdout.strip()


def student_environment(home):
    """The variables a student's shell would have, with HOME and uv's configuration in scratch.

    Settings that would pick another interpreter or environment than a student's shell does are
    dropped; the rest of uv's settings, such as a package index mirror, are kept.
    """
    redirecting = {"UV_PYTHON", "UV_PROJECT_ENVIRONMENT", "UV_SYSTEM_PYTHON", "UV_CONFIG_FILE", "UV_NO_CONFIG"}
    env = {name: value for name, value in os.environ.items()
           if name not in redirecting and not name.startswith(("VIRTUAL_ENV", "CONDA", "PYTHON"))}
    # Lecture 01's `uv python install 3.13 --default` links `python` in ~/.local/bin, which its shell
    # setup puts first on PATH; link the same uv-managed Python there, never a virtual environment's.
    bin_dir = home / ".local" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(uv_location("python", "find", "--system", "--managed-python", "3.13"))
    env.update(
        PATH=f"{bin_dir}{os.pathsep}{env.get('PATH', os.defpath)}",
        HOME=str(home),
        XDG_CONFIG_HOME=str(home / ".config"),
        XDG_DATA_HOME=str(home / ".local" / "share"),
        XDG_CACHE_HOME=str(home / ".cache"),
        UV_CACHE_DIR=uv_location("cache", "dir"),
        UV_PYTHON_INSTALL_DIR=uv_location("python", "dir"),
        PYTHONDONTWRITEBYTECODE="1",
    )
    return env


def run():
    # The setup script downloads from the address the guide's command names, file by file.
    setup = (DEMOS / "setup_demo.sh").read_text(encoding="utf-8")
    assert SETUP_COMMAND in GUIDE_TEXT, "the guide no longer gives the one-line setup command"
    base_url = SETUP_COMMAND.split()[2].rsplit("/", 1)[0]
    base_url_line = f'base_url="{base_url}"'
    assert setup.count(base_url_line) == 1, "setup_demo.sh downloads from elsewhere"
    downloads = re.findall(r'^curl -fsSL "\$base_url/([^"]+)" -o (\S+)$', setup, re.M)
    assert all(source == target for source, target in downloads), downloads
    shipped = {path.name for path in DEMOS.iterdir() if path.is_file() and path.name != "DEMO_GUIDE.md"}
    assert {source for source, _ in downloads} == shipped, (sorted(downloads), sorted(shipped))
    assert setup in fenced(span("## 1.5 Read a Shell Script", "## 1.6"), "bash"), "1.5 must show setup_demo.sh"

    (ROOT / "scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch", prefix="lecture03-") as temporary:
        home = Path(temporary) / "home"
        home.mkdir()
        env = student_environment(home)
        demo = home / "03-demo"
        # The same script with only its download address changed, so it reads this repository's files.
        local_setup = Path(temporary) / "setup_demo.sh"
        local_setup.write_text(setup.replace(base_url_line, f'base_url="{DEMOS.as_uri()}"'), encoding="utf-8")
        setup_command = f"curl -fsSL {local_setup.as_uri()} | sh"

        def shell(command, cwd=demo, check=True):
            """Run a Bash command as the student's terminal would, with pipefail so a pipe cannot hide a failure."""
            return subprocess.run(
                ["bash", "-c", f"set -o pipefail\n{command}"], cwd=cwd, env=env, check=check,
                capture_output=True, text=True,
            )

        # Demo 1.1: one command makes ~/03-demo and fills it with byte-identical copies of the demo files.
        installed = shell(setup_command, cwd=home).stdout
        assert sorted(path.name for path in demo.iterdir()) == sorted(shipped), sorted(demo.iterdir())
        for name in shipped:
            assert (demo / name).read_bytes() == (DEMOS / name).read_bytes(), name

        # A second run stops at mkdir and leaves every file as it was.
        before = {path: path.read_bytes() for path in demo.iterdir()}
        again = shell(setup_command, cwd=home, check=False)
        assert again.returncode != 0 and "File exists" in again.stderr and not again.stdout, again
        assert {path: path.read_bytes() for path in demo.iterdir()} == before

        # Demo 1.2 to 1.4: the guide's environment commands, run as written in one terminal session,
        # except the one 1.3 runs to show an expected error, which runs on its own at the same point.
        blocks = fenced(span("## 1.2 Create the Environment", "## 1.5 Read a Shell Script"), "bash")
        failing = blocks.index(EXPECTED_ERROR_COMMAND)
        assert EXPECTED_ERROR_COMMAND in fenced(span("## 1.3 Recreate It", "## 1.4"), "bash")
        before_error = shell("set -e\n" + "\n".join(blocks[:failing]))
        # A new shell is the state `deactivate` leaves: the student's PATH, with no environment active.
        error = shell(EXPECTED_ERROR_COMMAND, check=False)
        assert error.returncode != 0 and not error.stdout, error
        [shown_error] = fenced(span(EXPECTED_ERROR_COMMAND, "## 1.4"), "text")
        assert error.stderr == shown_error, (error.stderr, shown_error)
        after_error = shell("set -e\n" + "\n".join(blocks[failing + 1:]))
        # With the environment off, sys.executable is the Python outside it, with no .venv in its path.
        assert before_error.stdout.endswith(f"\n{home / '.local' / 'bin' / 'python'}\n"), before_error.stdout
        stdout = before_error.stdout + after_error.stdout
        printed = stdout + before_error.stderr + after_error.stderr
        for expected in ENVIRONMENT_OUTPUT:
            assert expected in GUIDE_TEXT, expected
            pattern = re.escape(expected).replace("x", r"\d+")
            assert re.search(pattern, printed), (expected, printed)
        assert printed.count("+ numpy==2.3.3") == 3, printed  # uv add, uv sync, uv pip install -r
        assert stdout.count("\n2.3.3\n") == 3, stdout
        venv_python = demo / ".venv" / "bin" / "python"
        assert f"\n{venv_python}\n" in stdout, stdout
        assert (demo / ".python-version").read_text(encoding="utf-8") == "3.13\n"
        assert fenced(span("## 1.2 Create the Environment", "## 1.3"), "toml") == [
            (demo / "pyproject.toml").read_text(encoding="utf-8")]
        assert quotes(stdout, (demo / "pyproject.toml").read_text(encoding="utf-8"))
        assert (demo / "uv.lock").is_file() and (demo / "recreation-check" / "uv.lock").is_file()
        requirements = (demo / "requirements.txt").read_text(encoding="utf-8")
        assert fenced(span("## 1.4 Share It", "## 1.5"), "text")[0] == requirements, requirements
        # --seed put pip in the demo environment, and neither uv add nor uv sync removed it.
        for folder in (demo, demo / "recreation-check", demo / "pip-check"):
            assert (folder / ".venv" / "bin" / "pip").is_file(), folder

        def python(*arguments):
            """Run the demo environment's Python from ~/03-demo, as `python` does once it is active."""
            return subprocess.run(
                [str(venv_python), *arguments], cwd=demo, env=env, check=True,
                capture_output=True, text=True,
            ).stdout

        # Every script is import-safe: importing runs no analysis and prints nothing.
        modules = ", ".join(name.removesuffix(".py") for name in PYTHON_SCRIPTS)
        assert python("-c", f"import {modules}") == ""

        # Demo 1.6 runs from ~/03-demo and writes only its data/, logs/, and results/ there.
        before = {path.relative_to(demo) for path in demo.rglob("*")}
        first = shell("bash demo1_cli_pipeline.sh").stdout
        results = sorted((demo / "results").glob("summary_*.txt"))
        log = (demo / "logs" / "processing.log").read_text(encoding="utf-8")
        assert len(results) == 1, results
        summary = results[0].read_text(encoding="utf-8")
        assert normalize(summary) == (
            "run timestamp: TIMESTAMP\nencounters: 6\nclinic counts:\n"
            "      3 Cardiology\n      2 Nephrology\n      1 Primary Care\n"
        )
        assert normalize(log).splitlines() == [
            "TIMESTAMP pipeline started",
            "TIMESTAMP wrote results/summary_TIMESTAMP.txt",
        ]
        assert (demo / "data" / "raw" / "encounters.csv").read_text(encoding="utf-8").count("\n") == 7
        stamp = captured_timestamp(results[0], summary, log.splitlines(), first)

        # A second run keeps the first result and appends its own two log lines under its own
        # timestamp, which is how a timestamped run keeps every result instead of overwriting it.
        time.sleep(1.1)  # the run timestamp has one-second resolution
        second = shell("bash demo1_cli_pipeline.sh").stdout
        assert normalize(second) == normalize(first)
        both = sorted((demo / "results").glob("summary_*.txt"))
        lines = (demo / "logs" / "processing.log").read_text(encoding="utf-8").splitlines()
        assert len(both) == 2 and results[0] in both, both
        assert len(lines) == 4 and lines[:2] == log.splitlines(), lines
        [newer] = [path for path in both if path != results[0]]
        assert captured_timestamp(
            newer, newer.read_text(encoding="utf-8"), lines[2:], second) != stamp
        created = {path.relative_to(demo) for path in demo.rglob("*")} - before
        assert {path.parts[0] for path in created} == {"data", "logs", "results"}, sorted(created)

        # Demo 1.7: the script the guide has students paste, saved as count_clinics.sh and run.
        [pasted] = [block for block in fenced(span("## 1.7 Save a Pipeline", "# Demo 2:"), "bash")
                    if block.startswith("#!/bin/bash")]
        (demo / "count_clinics.sh").write_text(pasted, encoding="utf-8")
        counted = shell("bash count_clinics.sh\ncat results/clinic_counts_*.txt").stdout
        assert len(list((demo / "results").glob("clinic_counts_*.txt"))) == 1

        runs = {name: python(name) for name in PYTHON_SCRIPTS}
        # Apart from its timings, every Demo 2 and Demo 3 script repeats itself exactly.
        for name, text in runs.items():
            assert normalize(python(name)) == normalize(text), name

        # normalize() blanks the four numbers in the performance comparison, so check them here:
        # they are positive, the reported saving and speedup follow from the two measurements
        # (within the rounding the printed digits allow), and calibrating the same million values
        # two ways gives the same sample, which is the claim the guide makes for this demo.
        measured = runs["demo2_numpy_performance.py"]
        timings = re.findall(r"^Time: (\d+\.\d+) ms$", measured, re.M)
        saved = re.findall(r"^Time saved: (\d+\.\d+) ms$", measured, re.M)
        speedup = re.findall(r"^Speedup: (\d+\.\d+)x faster!$", measured, re.M)
        assert len(timings) == 2 and len(saved) == len(speedup) == 1, measured
        listed, arrayed = float(timings[0]), float(timings[1])
        saving, ratio = float(saved[0]), float(speedup[0])
        assert listed > 0 and arrayed > 0 and saving > 0 and ratio > 0, measured
        # Every printed number rounds an exact one, so the saving and the speedup have to land
        # inside the interval the two printed measurements allow, not hit one exact value.
        list_low, list_high = listed - 0.005, listed + 0.005
        array_low, array_high = arrayed - 0.005, arrayed + 0.005
        assert list_low - array_high - 0.005 <= saving <= list_high - array_low + 0.005, measured
        assert list_low / array_high - 0.05 <= ratio <= list_high / array_low + 0.05, measured
        samples = re.findall(r"^Result sample: \[(.+)\]$", measured, re.M)
        assert len(samples) == 2 and samples[0].split(", ") == samples[1].split(), samples

        # The guide accounts for every line these scripts print, block by block and in order.
        printed = {**runs, "setup_demo.sh": installed, "demo1_cli_pipeline.sh": first,
                   "count_clinics.sh": counted}
        for name, (first_heading, last_heading) in SPANS.items():
            blocks = guide_blocks(first_heading, last_heading)
            assert blocks, name
            assert accounts_for(normalize(printed[name]), blocks), name

        # The shell checks the guide asks students to compare with the CSV summary.
        previews = shell("head -n 3 encounters.csv").stdout
        counts = shell("cut -d',' -f4 encounters.csv | tail -n +2 | sort | uniq -c").stdout
        assert counts == counted.split("\n", 1)[1], (counts, counted)

        # No block anywhere in the guide goes unverified, including Demo 1's and the shell's.
        outputs = [normalize(text) for text in
                   (installed, requirements, error.stderr, first, summary, log, counted, *runs.values(),
                    previews, counts)]
        for block in guide_blocks():
            assert any(quotes(output, block) for output in outputs), block

    print("Lecture 03: setup, environment commands, demo outputs, repeat runs, and every guide "
          "expectation matched real runs.")


if __name__ == "__main__":
    run()
