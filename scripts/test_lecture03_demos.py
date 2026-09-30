"""Run the Lecture 03 demos the way a student does and check the guide's expected output against real runs.

setup_demo.sh runs through `curl | sh` with HOME pointed at a scratch folder. The script the students run
downloads from GitHub, so the test runs a scratch copy whose `base_url` names this repository's 03/demo
instead: it downloads the working-tree files and never touches the real home folder.
Demo 1's environment commands then run with uv exactly as the guide gives them, and every demo script
runs from the resulting ~/03-demo with the Python of the environment Demo 1 built. uv reuses the real
uv cache and installed Pythons, and `uv add` needs the package index, as it does for students. Outside
the environment, `python` is the uv-managed Python 3.13 that Lecture 01 installs, with no NumPy.

A student pastes the guide into Bash or into Zsh, the macOS default, where `#` starts no comment at an
interactive prompt. So no command line in the guide, the lecture, or the bonus page carries a `#`, and
the whole guide is then pasted, block by block, into an interactive Bash and, when it is installed,
an interactive Zsh, each in a fresh home; every expectation the guide states has to hold in both.

    python3 scripts/test_lecture03_demos.py
"""

from pathlib import Path
import os
import pty
import re
import select
import shutil
import subprocess
import sys
import tempfile
import time


ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "03" / "demo"
GUIDE_TEXT = (DEMOS / "DEMO_GUIDE.md").read_text(encoding="utf-8")
# Every page a student copies shell commands from.
PAGES = {name: (ROOT / "03" / name).read_text(encoding="utf-8") for name in ("README.md", "BONUS.md")}
PAGES["demo/DEMO_GUIDE.md"] = GUIDE_TEXT
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
    "count_clinics.sh": ("## 1.7 Save a Pipeline as a Script", "## 1.8 Search"),
    "search": ("## 1.8 Search", "# Demo 2:"),
    "demo2_types_and_lists.py": ("## 2.1 Check Types", "## 2.2 Compare a List Loop"),
    "demo2_numpy_performance.py": ("## 2.2 Compare a List Loop", "## 2.3 Data Types"),
    "demo2_numpy_arrays.py": ("## 2.3 Data Types", "# Demo 3:"),
    "demo3_bp_analysis.py": ("# Demo 3:", "## 3.4 Summarize the Bundled CSV by Clinic"),
    "demo3_csv_summary.py": ("## 3.4 Summarize the Bundled CSV by Clinic", "### Check the counts against the shell"),
}
# The span of the guide whose text blocks name, in order, the lines to look for among what uv prints.
ENVIRONMENT_SPAN = ("## 1.2 Create the Environment", "## 1.5 Read a Shell Script")
# The command Demo 1.3 runs with the environment off, to show the lecture's ModuleNotFoundError.
EXPECTED_ERROR_COMMAND = 'python3 -c "import numpy as np"\n'
# An interactive shell with no startup files, as a student's terminal pastes into it.
SHELLS = {
    "bash": ["bash", "--norc", "--noprofile", "--noediting", "-i", "-s"],
    "zsh": ["zsh", "-f", "+o", "promptsp", "-i", "-s"],
}
MARK = "--- next command ---"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]")
# The `(name) ` an active environment puts before the prompt, which a shell prints before each command.
ENVIRONMENT_PROMPT = re.compile(r"(?:\([\w.-]+\) )+")
# The same prompt printed once the command's output has ended.
PROMPT_AFTER = re.compile(r"(?<=\n)(?:\([\w.-]+\) )+\Z")
# What a shell or uv prints when it cannot run a pasted line, such as `uv python pin 3.13 # note` in Zsh.
SHELL_ERROR = re.compile(r"^(?:(?:bash|zsh)(?::\d+)?: |error: )|command not found|no matches found", re.M)


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


def pasted_lines(block):
    """The lines of a bash block a student types or pastes at the prompt.

    A block that opens with `#!` is a file, shown to read or to paste into `cat > FILE`, and a
    here-document's body is file contents too: their `#` lines are comments in a script.
    """
    if block.startswith("#!"):
        return []
    lines, closing = [], None
    for line in block.splitlines():
        if closing is not None:
            closing = None if line.strip() == closing else closing
            continue
        lines.append(line)
        heredoc = re.search(r"<<-?\s*['\"]?(\w+)", line)
        if heredoc:
            closing = heredoc.group(1)
    return lines


def check_no_prompt_comments():
    """No command line a student copies carries a `#`: Zsh passes it on as an argument, not a comment."""
    for name, text in PAGES.items():
        for block in fenced(text, "bash"):
            for line in pasted_lines(block):
                assert not re.search(r"(^|\s)#", line), (
                    f"03/{name}: Zsh passes # to the command, not a comment: {line!r}")


def in_order(output, block):
    """True when each line of the block is a line of the output, in the same order, with any lines between.

    `3.13.x` stands for any 3.13 release, and leading and trailing spaces do not count.
    """
    lines = [line.strip() for line in output.splitlines()]
    position = 0
    for wanted in block.splitlines():
        pattern = re.compile(re.escape(wanted.strip()).replace(r"3\.13\.x", r"3\.13\.\d+"))
        found = next((index for index in range(position, len(lines)) if pattern.fullmatch(lines[index])), None)
        if found is None:
            return False
        position = found + 1
    return True


def guide_commands(setup_command):
    """Every command the guide has a student paste at the prompt, in guide order.

    Each line of a bash block is one command. A block that opens with `#!` is a file, which the
    prose before it says to paste into `cat > FILE`, so it is written with a here-document, as that
    paste does. A script shown only to read belongs in a ```text block, like every other output, so
    that pasting it by mistake cannot run it. The setup command downloads this repository's files.
    """
    commands, position = [], 0
    for match in re.finditer(r"^```bash\n(.*?)^```$", GUIDE_TEXT, re.M | re.S):
        block, before, position = match.group(1), GUIDE_TEXT[position:match.start()], match.end()
        if block.startswith("#!"):
            target = re.findall(r"`cat > (\S+)`", before)
            assert target, f"a script to read goes in a ```text block, not a ```bash one: {block.splitlines()[0]!r}"
            commands.append(f"cat > {target[-1]} <<'PASTED'\n{block}PASTED")
            continue
        commands += [setup_command if line == SETUP_COMMAND else line for line in block.splitlines() if line.strip()]
    return commands


def terminal(argv, cwd, env, script):
    """Run an interactive shell on a pty with the script piped in; return everything it printed."""
    master, slave = pty.openpty()
    process = subprocess.Popen(argv, cwd=cwd, env=env, stdin=subprocess.PIPE, stdout=slave, stderr=slave)
    os.close(slave)
    process.stdin.write(script.encode())      # small enough for the pipe buffer
    process.stdin.close()
    chunks = []
    while select.select([master], [], [], 300)[0]:
        try:
            data = os.read(master, 4096)
        except OSError:                       # the shell closed the pty and exited
            break
        if not data:
            break
        chunks.append(data)
    os.close(master)
    process.wait()
    return b"".join(chunks).decode().replace("\r\n", "\n")


def paste_guide(shell, home, env, setup_command):
    """Paste the whole guide into one interactive shell, as a student in one terminal would.

    Returns (command, environment prompt shown before it, output) for each command.
    """
    commands = guide_commands(setup_command)
    script = "".join(f"echo '{MARK}'\n{command}\n" for command in [*commands, ""])
    env = {**env, "TERM": "dumb", "PS1": "", "PS2": ""}
    transcript = ANSI.sub("", terminal(SHELLS[shell], home, env, script))
    outputs = transcript.split(MARK + "\n")[1:]
    assert len(outputs) == len(commands) + 1, transcript    # the last is the shell leaving
    steps = []
    for command, output in zip(commands, outputs):
        prompt = ENVIRONMENT_PROMPT.match(output)
        prompt = prompt.group() if prompt else ""
        shown = PROMPT_AFTER.sub("", output[len(prompt):])
        steps.append((command, prompt, shown))
    return steps


def output_of(steps, start):
    """The output of the one pasted command that starts with start."""
    found = [output for command, _, output in steps if command.startswith(start)]
    assert len(found) == 1, f"{start!r} ran {len(found)} times"
    return found[0]


def check_pasted(shell, steps, look_for, shown_error):
    """What a student sees after pasting the guide into this shell is what the guide says they see."""
    for command, prompt, output in steps:
        if command == EXPECTED_ERROR_COMMAND.strip():
            assert output == shown_error and not prompt, (shell, prompt, output)
        else:
            assert not SHELL_ERROR.search(output), (shell, command, output)
        # Demos 2 and 3 run in the (03-demo) environment, which the guide says the prompt shows.
        if command.startswith("python3 demo"):
            assert prompt == "(03-demo) ", (shell, command, prompt)
    transcript = "".join(output for _, _, output in steps)
    assert in_order(transcript, "".join(look_for)), (shell, transcript)
    search = pasted_lines(fenced(span(*SPANS["search"]), "bash")[0])
    printed = {
        "setup_demo.sh": output_of(steps, "curl "),
        "demo1_cli_pipeline.sh": output_of(steps, "bash demo1_cli_pipeline.sh"),
        "count_clinics.sh": output_of(steps, "bash count_clinics.sh") + output_of(steps, "cat results/clinic_counts_"),
        "search": "".join(output for command, _, output in steps if command in search),
        **{name: output_of(steps, f"python3 {name}") for name in PYTHON_SCRIPTS},
    }
    for name, (first_heading, last_heading) in SPANS.items():
        assert accounts_for(normalize(printed[name]), guide_blocks(first_heading, last_heading)), (shell, name)
    for block in guide_blocks():
        assert block in look_for or quotes(normalize(transcript), block), (shell, block)


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
    managed = uv_location("python", "find", "--system", "--managed-python", "3.13")
    # `uv python install 3.13 --default` (Lecture 01) puts both `python` and `python3` here.
    (bin_dir / "python").symlink_to(managed)
    (bin_dir / "python3").symlink_to(managed)
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
    assert setup in fenced(span("## 1.5 Read a Shell Script", "## 1.6"), "text"), "1.5 must show setup_demo.sh"
    check_no_prompt_comments()

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

        def shell(command, cwd=demo, check=True, merged=False):
            """Run a Bash command as the student's terminal would, with pipefail so a pipe cannot hide a failure.

            `merged` sends error output into the standard output, in the order a terminal shows both.
            """
            return subprocess.run(
                ["bash", "-c", f"set -o pipefail\n{command}"], cwd=cwd, env=env, check=check,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT if merged else subprocess.PIPE, text=True,
            )

        # Demo 1.1: one command makes ~/03-demo and fills it with byte-identical copies of the demo files.
        installed = shell(setup_command, cwd=home).stdout
        assert sorted(path.name for path in demo.iterdir()) == sorted(shipped), sorted(demo.iterdir())
        for name in shipped:
            assert (demo / name).read_bytes() == (DEMOS / name).read_bytes(), name

        # 1.5 shows what `cat setup_demo.sh` prints.
        setup_listing = shell("cat setup_demo.sh").stdout

        # A second run stops at mkdir and leaves every file as it was.
        before = {path: path.read_bytes() for path in demo.iterdir()}
        again = shell(setup_command, cwd=home, check=False)
        assert again.returncode != 0 and "File exists" in again.stderr and not again.stdout, again
        assert {path: path.read_bytes() for path in demo.iterdir()} == before

        # Demo 1.2 to 1.4: the guide's environment commands, run as written in one terminal session,
        # except the one 1.3 runs to show an expected error, which runs on its own at the same point.
        blocks = fenced(span(*ENVIRONMENT_SPAN), "bash")
        failing = blocks.index(EXPECTED_ERROR_COMMAND)
        assert EXPECTED_ERROR_COMMAND in fenced(span("## 1.3 Recreate It", "## 1.4"), "bash")
        before_error = shell("set -e\n" + "\n".join(blocks[:failing]), merged=True)
        # A new shell is the state `deactivate` leaves: the student's PATH, with no environment active.
        error = shell(EXPECTED_ERROR_COMMAND, check=False)
        assert error.returncode != 0 and not error.stdout, error
        shown_error = fenced(span(EXPECTED_ERROR_COMMAND, "## 1.4"), "text")[0]
        assert error.stderr == shown_error, (error.stderr, shown_error)
        after_error = shell("set -e\n" + "\n".join(blocks[failing + 1:]), merged=True)
        # With the environment off, sys.executable is the Python outside it, with no .venv in its path.
        assert before_error.stdout.endswith(f"\n{home / '.local' / 'bin' / 'python3'}\n"), before_error.stdout
        stdout = printed = before_error.stdout + after_error.stdout
        # The guide's other text blocks here are the error and requirements.txt; the rest name, in
        # order, the lines to look for among everything uv and Python print.
        requirements = (demo / "requirements.txt").read_text(encoding="utf-8")
        look_for = [block for block in fenced(span(*ENVIRONMENT_SPAN), "text")
                    if block not in (shown_error, requirements)]
        assert len(look_for) == 3 and in_order(printed, "".join(look_for)), (look_for, printed)
        assert printed.count("+ numpy==2.3.3") == 3, printed  # uv add, uv sync, uv pip install -r
        assert stdout.count("\n2.3.3\n") == 3, stdout
        venv_python = demo / ".venv" / "bin" / "python3"
        assert f"\n{venv_python}\n" in stdout, stdout
        assert (demo / ".python-version").read_text(encoding="utf-8") == "3.13\n"
        assert fenced(span("## 1.2 Create the Environment", "## 1.3"), "toml") == [
            (demo / "pyproject.toml").read_text(encoding="utf-8")]
        assert quotes(stdout, (demo / "pyproject.toml").read_text(encoding="utf-8"))
        assert (demo / "uv.lock").is_file() and (demo / "recreation-check" / "uv.lock").is_file()
        assert fenced(span("## 1.4 Share It", "## 1.5"), "text")[0] == requirements, requirements
        # --seed put pip in the demo environment, and neither uv add nor uv sync removed it.
        for folder in (demo, demo / "recreation-check", demo / "pip-check"):
            assert (folder / ".venv" / "bin" / "pip").is_file(), folder

        # A student who picks Demo 1 up again at 1.3 or 1.5 in a new terminal, in any folder, starts with the
        # same two re-entry lines as Demos 2 and 3, so 1.3's `deactivate` always has an environment to leave.
        reentry = "cd ~/03-demo\nsource .venv/bin/activate\n"
        for first_heading, last_heading in (("## 1.3 Recreate It", "## 1.4"), ("## 1.5 Read", "## 1.6"),
                                            ("# Demo 2:", "## 2.1"), ("# Demo 3:", "## 3.1")):
            assert fenced(span(first_heading, last_heading), "bash")[0] == reentry, first_heading
        resumed = shell("set -e\n" + "".join(fenced(span("## 1.3 Recreate It", "## 1.4"), "bash")[:2]),
                        cwd=home, merged=True)
        assert resumed.stdout == f"{home / '.local' / 'bin' / 'python3'}\n", resumed.stdout

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

        # Demo 1.8: a wildcard and grep, whose counts match the two pipelines' Cardiology counts.
        [search_block] = fenced(span(*SPANS["search"]), "bash")
        searched = shell(search_block).stdout
        cardiology = [line.split()[0] for line in (first + counted).splitlines() if line.endswith(" Cardiology")]
        assert searched.splitlines()[1:3] == cardiology == ["3", "260"], (searched, cardiology)

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
                   "count_clinics.sh": counted, "search": searched}
        for name, (first_heading, last_heading) in SPANS.items():
            blocks = guide_blocks(first_heading, last_heading)
            assert blocks, name
            assert accounts_for(normalize(printed[name]), blocks), name

        # The shell checks the guide asks students to compare with the CSV summary.
        previews = shell("head -n 3 encounters.csv").stdout
        counts = shell("cut -d',' -f4 encounters.csv | tail -n +2 | sort | uniq -c").stdout
        assert counts == counted.split("\n", 1)[1], (counts, counted)

        # No block anywhere in the guide goes unverified, including Demo 1's and the shell's; the
        # environment's lines to look for were matched in order above.
        outputs = [normalize(text) for text in
                   (installed, setup_listing, requirements, error.stderr, first, summary, log, counted, searched,
                    *runs.values(), previews, counts)]
        for block in guide_blocks():
            assert block in look_for or any(quotes(output, block) for output in outputs), block

        # The whole guide pasted into an interactive shell, in a fresh home for each shell.
        pasted_into = []
        for name in SHELLS:
            if not shutil.which(name):
                print(f"{name} is not installed: the guide was not pasted into {name}.")
                continue
            student = Path(temporary) / f"paste-{name}"
            student.mkdir()
            steps = paste_guide(name, student, student_environment(student), setup_command)
            check_pasted(name, steps, look_for, shown_error)
            pasted_into.append(name)

    print("Lecture 03: no pasted command carries a # comment; setup, environment commands, demo outputs, "
          "repeat runs, and every guide expectation matched real runs, and the whole guide pasted into "
          f"{' and '.join(pasted_into)} matched too.")


if __name__ == "__main__":
    run()
