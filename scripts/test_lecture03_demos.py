"""Run Lecture 03 demos in isolation and check the guide's expected output against real runs."""

from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "03" / "demo"
SCRIPTS = (
    "demo1_cli_pipeline.sh",
    "demo2_python_collections.py",
    "demo2_numpy_performance.py",
    "demo2_numpy_arrays.py",
    "demo3_bp_analysis.py",
    "demo3_csv_summary.py",
)
# Each script, and the span of DEMO_GUIDE.md whose ```text blocks must account for every line it
# prints. Demo 1's span ends before the guide shows the summary and the log, because those are
# files it writes rather than lines it prints.
SPANS = {
    "demo1_cli_pipeline.sh": ("# 1. Shell Pipeline", "Open the saved summary and log:"),
    "demo2_python_collections.py": ("## 2.1 Inspect Values", "## 2.2 Compare a List Loop"),
    "demo2_numpy_performance.py": ("## 2.2 Compare a List Loop", "## 2.3 Create Arrays"),
    "demo2_numpy_arrays.py": ("## 2.3 Create Arrays", "# 3. NumPy Blood-Pressure Analysis"),
    "demo3_bp_analysis.py": ("# 3. NumPy Blood-Pressure Analysis", "## 3.4 Optional"),
    "demo3_csv_summary.py": ("## 3.4 Optional", "### Check the counts against the shell"),
}


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


def guide_blocks(first_heading=None, last_heading=None):
    """The guide's ```text blocks, or only those between two of its headings."""
    guide = normalize((DEMOS / "DEMO_GUIDE.md").read_text(encoding="utf-8"))
    if first_heading is not None:
        start = guide.index(first_heading)
        guide = guide[start:guide.index(last_heading, start)]
    return re.findall(r"^```text\n(.*?)^```$", guide, re.M | re.S)


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
    shows - or one it shows differently - fails the check.
    """
    lines = output.splitlines()
    position = 0
    for block in blocks:
        start = block_start(lines, block, position)
        if start < 0 or any(line.strip() for line in lines[position:start]):
            return False
        position = start + len(block.splitlines())
    return not any(line.strip() for line in lines[position:])


def run():
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch", prefix="lecture03-") as temporary:
        demo = Path(temporary)
        for name in (*SCRIPTS, "encounters.csv"):
            shutil.copy2(DEMOS / name, demo / name)
        # 03/demo cannot change during this run: these copies are what execute, and every command
        # below names a working directory inside `demo`.

        def python(*arguments, cwd=demo):
            return subprocess.run(
                [sys.executable, *arguments], cwd=cwd, check=True,
                capture_output=True, text=True,
            ).stdout

        def shell(command, cwd):
            return subprocess.run(
                command, cwd=cwd, check=True, shell=True,
                capture_output=True, text=True,
            ).stdout

        # Every script is import-safe: importing runs no analysis and prints nothing.
        assert python("-c", "import demo2_python_collections, demo2_numpy_performance, "
                            "demo2_numpy_arrays, demo3_bp_analysis, demo3_csv_summary") == ""

        # Demo 1 runs from a disposable directory and writes only below it.
        pipeline = demo / "cli-run"
        pipeline.mkdir()
        before = {path.relative_to(demo) for path in demo.rglob("*")}
        first = shell(f"bash {demo / 'demo1_cli_pipeline.sh'}", pipeline)
        results = sorted((pipeline / "results").glob("summary_*.txt"))
        log = (pipeline / "logs" / "processing.log").read_text(encoding="utf-8")
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
        stamp = captured_timestamp(results[0], summary, log.splitlines(), first)

        # A second run keeps the first result and appends its own two log lines under its own
        # timestamp, which is how a timestamped run keeps every result instead of overwriting it.
        time.sleep(1.1)  # the run timestamp has one-second resolution
        second = shell(f"bash {demo / 'demo1_cli_pipeline.sh'}", pipeline)
        assert normalize(second) == normalize(first)
        both = sorted((pipeline / "results").glob("summary_*.txt"))
        lines = (pipeline / "logs" / "processing.log").read_text(encoding="utf-8").splitlines()
        assert len(both) == 2 and results[0] in both, both
        assert len(lines) == 4 and lines[:2] == log.splitlines(), lines
        [newer] = [path for path in both if path != results[0]]
        assert captured_timestamp(
            newer, newer.read_text(encoding="utf-8"), lines[2:], second) != stamp

        # The guide warns that Demo 1 writes data/, logs/, and results/ below the current
        # directory, so nothing it wrote may escape the disposable directory it was run from.
        created = {path.relative_to(demo) for path in demo.rglob("*")} - before
        assert all(path.parts[0] == "cli-run" for path in created), sorted(created)

        runs = {
            "demo2_python_collections.py": python("demo2_python_collections.py"),
            "demo2_numpy_arrays.py": python("demo2_numpy_arrays.py"),
            "demo3_bp_analysis.py": python("demo3_bp_analysis.py"),
            "demo3_csv_summary.py": python("demo3_csv_summary.py"),
            "demo2_numpy_performance.py": python("demo2_numpy_performance.py"),
        }
        # Apart from its timings, every Demo 2 and Demo 3 script repeats itself exactly.
        for name, text in runs.items():
            assert normalize(python(name)) == normalize(text), name

        # normalize() blanks the four numbers in the performance comparison, so check them here:
        # they are positive, the reported saving and speedup follow from the two measurements
        # (within the rounding the printed digits allow), and doubling the same million values
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
        printed = {**runs, "demo1_cli_pipeline.sh": first}
        for name, (first_heading, last_heading) in SPANS.items():
            blocks = guide_blocks(first_heading, last_heading)
            assert blocks, name
            assert accounts_for(normalize(printed[name]), blocks), name

        # The shell checks the guide asks students to compare with the CSV summary.
        previews = shell("head -n 3 encounters.csv", demo)
        counts = shell("cut -d',' -f4 encounters.csv | tail -n +2 | sort | uniq -c", demo)

        # No block anywhere in the guide goes unverified, including Demo 1's and the shell's.
        outputs = [normalize(text) for text in
                   (first, summary, log, *runs.values(), previews, counts)]
        for block in guide_blocks():
            assert any(quotes(output, block) for output in outputs), block

    print("Lecture 03: demo outputs, repeat runs, and every guide expectation matched real runs.")


if __name__ == "__main__":
    run()
