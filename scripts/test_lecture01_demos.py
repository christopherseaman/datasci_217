"""Check Lecture 01's demos, guide, and lecture transcripts against real Python 3.13 and Bash runs.

The Python demos run as scripts and are typed into the real interactive prompt the way the
guide tells students to type them. The lecture's `>>>` transcripts are replayed keystroke by
keystroke, and the shell script is pasted into `cat` in an interactive Bash.
"""

from pathlib import Path
import fcntl
import os
import pty
import re
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import termios
import time


ROOT = Path(__file__).resolve().parents[1]
DEMOS = ROOT / "01" / "demo"
GUIDE = DEMOS / "DEMO_GUIDE.md"
SETUP = DEMOS / "01_github_vscode_setup_guide.md"
LECTURE = ROOT / "01" / "README.md"
BONUS = ROOT / "01" / "BONUS.md"
PYTHON_DEMOS = (
    "03a_values.py", "03b_strings.py", "03c_calculations.py", "04a_decisions.py",
    "04b_for_loops.py", "04c_loop_control.py", "04d_debugging.py", "04e_measurement_workflow.py",
)
SHELL_DEMO = "02_cli_navigation_demo.sh"
ANSI = re.compile(r"\x1b\[[0-9;?]*[A-Za-z]|\x1b[=>]|\x1b\][^\x07]*\x07")
BACKSPACE = "\x7f"
CTRL_C = "\x03"


def fences(text, language):
    """Every fenced block of one language, in document order."""
    return re.findall(r"^```%s\n(.*?)^```$" % language, text, re.M | re.S)


def after(text, anchor, language):
    """The first fenced block of a language that follows an anchor string."""
    return fences(text[text.index(anchor):], language)[0]


class Terminal:
    """A program on a pseudo-terminal, typed at the way a student types at it."""

    def __init__(self, argv, cwd, env):
        pid, self.fd = pty.fork()
        if pid == 0:                                  # the child: the pty is its terminal
            try:
                os.chdir(cwd)
                os.execvpe(argv[0], argv, env)
            finally:
                os._exit(127)
        self.pid = pid
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", 50, 200, 0, 0))
        self.raw = b""
        self.settle(3.0)

    def settle(self, first_wait=1.0):
        """Read until the program has been quiet for a moment."""
        wait = first_wait
        while select.select([self.fd], [], [], wait)[0]:
            try:
                data = os.read(self.fd, 65536)
            except OSError:                           # the program closed the pty and exited
                return
            if not data:
                return
            self.raw += data
            wait = 0.25

    def type(self, keys):
        os.write(self.fd, keys.encode())
        self.settle()

    def close(self, keys):
        os.write(self.fd, keys.encode())
        self.settle()
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            pid, _ = os.waitpid(self.pid, os.WNOHANG)
            if pid:
                break
            time.sleep(0.05)
        else:
            os.kill(self.pid, 9)
            os.waitpid(self.pid, 0)
            raise AssertionError("the terminal program did not exit")
        os.close(self.fd)
        return self.raw.decode(errors="replace")


def repl_environment(home):
    return {"PATH": os.environ["PATH"], "HOME": str(home), "TERM": "xterm-256color",
            "LC_ALL": "C.UTF-8", "PYTHONDONTWRITEBYTECODE": "1"}


def repl_screen(raw):
    """The final text of each screen line: the prompt's redraws collapse to what was typed."""
    lines = []
    for line in ANSI.sub("", raw).replace("\r", "").split("\n"):
        prompt = max(line.rfind(">>> "), line.rfind("... "))
        if line.startswith((">>>", "...")) and prompt >= 0:
            line = line[prompt:]
        lines.append(line.rstrip())
    while lines and not lines[-1]:
        lines.pop()
    return lines


def guided_keys(lines):
    """Keystrokes for code lines typed the guide's way at the 3.13 prompt.

    Leave out the indentation, press Backspace once per level to move back out, and press
    Enter on an empty line to finish a block. A blank line outside a block is skipped.
    """
    keys, auto = [], 0
    for line in lines:
        content = line.strip()
        if not content:
            if auto:
                keys.append("\r")                     # Enter on the empty ... line runs the block
                auto = 0
            continue
        indent = len(line) - len(line.lstrip())
        assert indent <= auto and (auto - indent) % 4 == 0, f"cannot type {line!r} after indent {auto}"
        keys.append(BACKSPACE * ((auto - indent) // 4) + content + "\r")
        auto = indent + 4 if content.endswith(":") else indent
    if auto:
        keys.append("\r")
    return keys


def type_into_repl(keys, cwd, home, leave=True):
    repl = Terminal([sys.executable, "-q"], cwd, repl_environment(home))
    for key in keys:
        repl.type(key)
    return repl_screen(repl.close("exit()\r" if leave else ""))


def printed(screen):
    """What the REPL printed, without the typed lines and the closing exit()."""
    return [line for line in screen if not line.startswith((">>>", "..."))]


def check_text_rules():
    for page in (LECTURE, BONUS, GUIDE, SETUP):
        text = page.read_text(encoding="utf-8")
        assert "Ctrl+D" not in text, f"{page.name} still finishes cat with Ctrl+D"
        assert "\u2014" not in text, f"{page.name} has an em dash"
    # Demo 1 in the guide is the setup guide, word for word.
    guide = GUIDE.read_text(encoding="utf-8")
    start = guide.index("\n\n", guide.index("[Setup source on GitHub]")) + 2
    demo_one = guide[start:guide.index("\n# Demo 2")]
    assert demo_one == SETUP.read_text(encoding="utf-8").split("\n\n", 1)[1], "Demo 1 drifted from the setup guide"


def check_guide_quotes_files(guide):
    """The block under each file's GitHub link is that file, character for character."""
    names = re.findall(r"^\[([\w.]+) on GitHub\]\(", guide, re.M)
    assert sorted(names) == sorted((SHELL_DEMO, *PYTHON_DEMOS)), names
    for name in names:
        anchor = f"[{name} on GitHub]"
        block = after(guide, anchor, "python" if name.endswith(".py") else "bash")
        assert block == (DEMOS / name).read_text(encoding="utf-8"), f"the guide misquotes {name}"


def check_script_output(guide, work, python):
    """Each `python3 NAME` block is followed by exactly what the script prints."""
    outputs = {}
    for name in PYTHON_DEMOS:
        command = f"```bash\npython3 {name}\n```"
        assert guide.count(command) == 1, f"the guide runs {name} {guide.count(command)} times"
        expected = after(guide, command, "text")
        actual = python(name, cwd=work).stdout
        assert actual == expected, f"{name} printed:\n{actual}\nthe guide says:\n{expected}"
        outputs[name] = actual
    return outputs


def check_repl(outputs, work, home):
    """Typed the guide's way, every demo prints at the prompt what it prints as a script."""
    for name in PYTHON_DEMOS:
        source = (DEMOS / name).read_text(encoding="utf-8").splitlines()
        screen = type_into_repl(guided_keys(source), work, home)
        assert printed(screen) == outputs[name].splitlines(), f"{name} at the prompt:\n" + "\n".join(screen)

    # Typing the shown spaces as well: the second line of a block is an unexpected indent.
    loop = (DEMOS / "04b_for_loops.py").read_text(encoding="utf-8").splitlines()[:6]
    screen = type_into_repl([line + "\r" for line in loop] + ["\r"], work, home)
    assert "IndentationError: unexpected indent" in screen, "\n".join(screen)


def check_lecture_transcripts(lecture, work, home):
    """The lecture's `>>>` transcripts are what the 3.13 prompt shows for those keystrokes."""
    for anchor in ("### The `>>>` Prompt Indents for You", "#### Interactive Mode Example"):
        transcript = after(lecture, anchor, "console").splitlines()
        shown = [line for line in transcript if not line.startswith("$ ")]
        typed = [line[4:] for line in shown if line.startswith((">>>", "..."))]
        if typed[-1] == "exit()":                     # the transcript leaves the prompt itself
            screen = type_into_repl(guided_keys(typed[:-1]) + ["exit()\r"], work, home, leave=False)
            expected = shown
        else:
            screen = type_into_repl(guided_keys(typed), work, home)
            expected = shown + [">>> exit()"]
        assert screen == expected, f"{anchor} at the prompt:\n" + "\n".join(screen)

    # Its claims about typing the spaces as well, on its own if-block.
    correct = after(lecture, "### Indentation Matters!", "python").splitlines()
    statements = [line for line in correct if line.strip() and not line.startswith("#")]
    screen = type_into_repl([line + "\r" for line in statements[:3]] + ["\r"], work, home)
    assert printed(screen) == ["Positive"], "\n".join(screen)
    screen = type_into_repl([line + "\r" for line in statements] + ["\r"], work, home)
    assert "IndentationError: unexpected indent" in screen, "\n".join(screen)


def check_debugging_steps(guide, work, python):
    """Each broken line in 04d raises the error the guide's table names, as described."""
    source = (DEMOS / "04d_debugging.py").read_text(encoding="utf-8")
    lines = source.splitlines(keepends=True)
    table = re.findall(r"^\| `(#[^`]+)` \| `([^`]+)` \|$", guide, re.M)
    assert [error.split(":")[0] for _, error in table] == ["NameError", "TypeError", "ValueError"], table
    broken = work / "broken"
    broken.mkdir()
    runs = {}
    for commented, error in table:
        assert lines.count(commented + "\n") == 1, f"04d has no line {commented!r}"
        (broken / "04d_debugging.py").write_text(source.replace(commented, commented[1:], 1), encoding="utf-8")
        run = python("04d_debugging.py", cwd=broken, check=False)
        assert run.returncode == 1 and run.stderr.splitlines()[-1] == error, run.stderr
        runs[error.split(":")[0]] = run
    starts = re.search(r"the `TypeError` run starts with `([^`]+)`", guide).group(1)
    assert runs["TypeError"].stdout == starts + "\n", runs["TypeError"].stdout

    # A # removed by retyping, with the space after it left behind.
    first = table[0][0]
    (broken / "04d_debugging.py").write_text(source.replace(first, " " + first[1:], 1), encoding="utf-8")
    run = python("04d_debugging.py", cwd=broken, check=False)
    assert run.stderr.splitlines()[-1] == "IndentationError: unexpected indent", run.stderr

    # The IndentationError step: the named line loses its four spaces; nothing runs.
    code, number = re.search(r"delete the four spaces before `([^`]+)` on line (\d+)", guide).groups()
    number = int(number)
    assert lines[number - 1] == "    " + code + "\n", f"line {number} is not {code}"
    lines[number - 1] = code + "\n"
    practice = work / "practice"
    practice.mkdir()
    (practice / "04d_debugging.py").write_text("".join(lines), encoding="utf-8")
    run = python("04d_debugging.py", cwd=practice, check=False)
    expected = after(guide, f"on line {number}, save, and run:", "text")
    assert run.returncode == 1 and run.stdout == "", run.stdout
    assert run.stderr.replace(str(practice), "/home/alice/practice") == expected, run.stderr


def check_tried_values(guide, work, python):
    """The values the guide invites students to try give the results it names."""
    decisions = (DEMOS / "04a_decisions.py").read_text(encoding="utf-8")
    assert "Set `systolic` to 145, then 118, and rerun (stage 2 hypertension, then normal)." in guide
    for value, category in (("145", "stage 2 hypertension"), ("118", "normal")):
        (work / "try.py").write_text(decisions.replace("systolic = 135", "systolic = " + value, 1), encoding="utf-8")
        assert python("try.py", cwd=work).stdout.splitlines()[0] == "systolic category: " + category
    workflow = (DEMOS / "04e_measurement_workflow.py").read_text(encoding="utf-8")
    assert "Lower `review_above` to 80 and rerun: 88 is now flagged too, and `readings to review` becomes 3." in guide
    (work / "try.py").write_text(workflow.replace("review_above = 100", "review_above = 80", 1), encoding="utf-8")
    lines = python("try.py", cwd=work).stdout.splitlines()
    assert "Visit 3 heart rate: 88 bpm REVIEW" in lines and lines[-1] == "readings to review: 3", lines


def check_shell_demo(guide, work, home):
    """Demo 2: the typed commands, then the script pasted into cat and run with bash."""
    practice = work / "shell-practice"
    practice.mkdir()
    env = {"PATH": os.environ["PATH"], "HOME": str(home), "LC_ALL": "C", "TERM": "dumb", "PS1": "$ "}
    manual = guide[guide.index("## 2.1"):guide.index("## 2.2")]
    typed = "".join(fences(manual, "bash"))
    run = subprocess.run(["bash", "--norc", "--noprofile", "-e"], input=typed, cwd=practice, env=env,
                         capture_output=True, text=True, check=True)
    pwds = [line for line in run.stdout.splitlines() if line.startswith("/")]
    assert pwds == [str(practice), str(practice / "clinic_manual"), str(practice)], run.stdout
    manual_project = practice / "clinic_manual"
    assert sorted(p.name for p in manual_project.iterdir()) == ["README.txt", "data", "results", "scripts"]
    assert [p.name for p in (manual_project / "results").iterdir()] == ["visits_raw.csv"]
    # The cat, head, and tail output the guide shows, in order, and the file it describes.
    shown = fences(manual, "text")
    assert len(shown) == 2, shown
    assert run.stdout.index(shown[0]) < run.stdout.index(shown[1]), run.stdout
    visits = shown[0]
    for name in ("data/visits.csv", "results/visits_raw.csv"):
        assert (manual_project / name).read_text(encoding="utf-8") == visits, name

    script = (DEMOS / SHELL_DEMO).read_text(encoding="utf-8")
    command = after(guide, "## 2.2", "bash")
    assert command == f"cat > {SHELL_DEMO}\n", command
    pasted = script.rstrip("\n")                       # a copied code block has no final newline

    def paste(press_enter):
        bash = Terminal(["bash", "--norc", "--noprofile", "-i"], practice, env)
        bash.type(command.replace("\n", "\r"))
        bash.type(pasted)
        if press_enter:
            bash.type("\r")
        bash.type(CTRL_C)
        bash.close("exit\r")
        return (practice / SHELL_DEMO).read_text(encoding="utf-8")

    # Without Enter, Ctrl+C drops the unfinished last line, as the guide warns.
    assert paste(press_enter=False) == script[:script.rstrip("\n").rindex("\n") + 1]
    # Enter, then Ctrl+C: the file is the script.
    assert paste(press_enter=True) == script

    run = subprocess.run(["bash", SHELL_DEMO], cwd=practice, env=env, capture_output=True, text=True)
    assert run.returncode == 0 and run.stderr == "", run.stderr
    made = practice / "clinic_practice"
    for folder in ("data", "scripts", "results"):
        assert (made / folder).is_dir(), folder
    assert (made / "README.txt").read_text() == ""
    for name in ("data/visits.csv", "results/visits_raw.csv"):
        assert (made / name).read_text(encoding="utf-8") == visits, name
    assert sorted(p.name for p in (made / "results").iterdir()) == ["visits_raw.csv"]
    # 2.3 says the script's cat, head, and tail output matches 2.1.
    assert "The `cat`, `head`, and `tail` output matches 2.1." in guide
    assert run.stdout.index(shown[0]) < run.stdout.index(shown[1]), run.stdout


def check_lecture_errors(lecture, work, python):
    """The lecture's IndentationError and NameError examples print the tracebacks it shows."""
    project = work / "datasci217"
    project.mkdir()
    for heading in ("## IndentationError: Check the Block Under the Colon",
                    "## NameError: Check the Name and Its Definition"):
        section = lecture[lecture.index(heading):]
        (project / "analysis.py").write_text(fences(section, "python")[0], encoding="utf-8")
        run = python("analysis.py", cwd=project, check=False)
        assert run.returncode == 1 and run.stdout == "", run.stdout
        shown = run.stderr.replace(str(project), "/home/alice/datasci217")
        assert shown == fences(section, "text")[0], f"{heading}:\n{shown}"
    # The running-total snippet still prints what the lecture says.
    snippet = after(lecture, "### Code Snippet: Practical Data Science Example", "python")
    (project / "average.py").write_text(snippet, encoding="utf-8")
    expected = after(lecture, "### Code Snippet: Practical Data Science Example", "text")
    assert python("average.py", cwd=project).stdout == expected


def run():
    assert sys.version_info[:2] == (3, 13), f"run with Python 3.13, not {sys.version.split()[0]}"
    guide = GUIDE.read_text(encoding="utf-8")
    lecture = LECTURE.read_text(encoding="utf-8")
    check_text_rules()
    check_guide_quotes_files(guide)

    (ROOT / "scratch").mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(dir=ROOT / "scratch", prefix="lecture01-") as temporary:
        work = Path(temporary)
        home = work / "home"
        home.mkdir()
        for name in PYTHON_DEMOS:
            shutil.copy2(DEMOS / name, work / name)

        def python(*arguments, cwd, check=True):
            return subprocess.run([sys.executable, "-B", *arguments], cwd=cwd, check=check,
                                  capture_output=True, text=True, env=repl_environment(home))

        outputs = check_script_output(guide, work, python)
        check_repl(outputs, work, home)
        check_lecture_transcripts(lecture, work, home)
        check_debugging_steps(guide, work, python)
        check_tried_values(guide, work, python)
        check_shell_demo(guide, work, home)
        check_lecture_errors(lecture, work, python)
    print("Lecture 01: guide quotes, script output, typed prompt sessions, lecture transcripts, "
          "debugging steps, tried values, cat paste, and tracebacks match real runs.")


if __name__ == "__main__":
    run()
