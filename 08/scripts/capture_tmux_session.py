"""Screenshot a real tmux session running analysis.py into ../media/tmux_session.png.

An inner tmux (session "analysis") runs the job. An outer tmux with its status bar
off attaches to it, so the outer pane's capture holds the inner screen and its status
bar exactly as a terminal would show them. The capture's ANSI colors are drawn with PIL.
The inner status bar drops tmux's default clock, so reruns differ only if tmux does.

Needs tmux and the DejaVu Sans Mono font. Run from the repository root:
    uv run --project 08/demo python 08/scripts/capture_tmux_session.py
"""
import re
import subprocess
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUT = Path(__file__).resolve().parents[1] / "media" / "tmux_session.png"
COLS, ROWS, STEPS = 64, 13, 10

ANALYSIS = '''import time

# One progress line per second for two minutes
for step in range(1, 121):
    print("step", step, "of 120")
    time.sleep(1)
print("done")
'''


def tmux(sock, *args, check=True):
    return subprocess.run(["tmux", "-L", sock, "-f", "/dev/null", *args], check=check,
                          capture_output=True, text=True).stdout


def capture(home):
    project = home / "project"
    project.mkdir()
    (project / "analysis.py").write_text(ANALYSIS)
    shell = (f"cd {project} && env -i HOME={home} PATH=/usr/bin:/bin TERM=tmux-256color "
             r"PS1='jdoe@analysis:\w\$ ' bash --norc --noprofile")
    try:
        tmux("ds217inner", "new-session", "-d", "-s", "analysis", "-x", str(COLS), "-y", str(ROWS), shell)
        # The default status-right quotes the pane title (the machine's hostname) and adds a clock;
        # show the server's name only, so the image does not change with the time of the run
        tmux("ds217inner", "select-pane", "-t", "analysis", "-T", "analysis")
        tmux("ds217inner", "set", "-g", "status-right", '"#{=21:pane_title}"')
        tmux("ds217outer", "new-session", "-d", "-s", "shot", "-x", str(COLS), "-y", str(ROWS),
             "env -u TMUX tmux -L ds217inner attach -t analysis")
        tmux("ds217outer", "set", "-g", "status", "off")
        tmux("ds217outer", "resize-window", "-t", "shot", "-x", str(COLS), "-y", str(ROWS))
        time.sleep(1)
        tmux("ds217inner", "send-keys", "-t", "analysis", "clear", "Enter")
        time.sleep(0.5)
        tmux("ds217inner", "send-keys", "-t", "analysis", "time python3 analysis.py", "Enter")
        # Poll until step STEPS is on screen; the job prints one step per second
        target, deadline = f"step {STEPS} of 120", time.time() + STEPS + 10
        while time.time() < deadline:
            raw = tmux("ds217outer", "capture-pane", "-e", "-p", "-t", "shot")
            if target in raw:
                return raw
            time.sleep(0.05)
        raise SystemExit(f"{target!r} never appeared in the tmux pane")
    finally:
        tmux("ds217outer", "kill-server", check=False)
        tmux("ds217inner", "kill-server", check=False)


# VS Code Dark+ terminal palette
PALETTE = ["#000000", "#cd3131", "#0dbc79", "#e5e510", "#2472c8", "#bc3fbc", "#11a8cd", "#e5e5e5",
           "#666666", "#f14c4c", "#23d18b", "#f5f543", "#3b8eea", "#d670d6", "#29b8db", "#e5e5e5"]
FG, BG = "#cccccc", "#1e1e1e"


def color256(n):
    if n < 16:
        return PALETTE[n]
    if n < 232:
        n -= 16
        steps = [0, 95, 135, 175, 215, 255]
        return "#%02x%02x%02x" % (steps[n // 36], steps[(n // 6) % 6], steps[n % 6])
    level = 8 + (n - 232) * 10
    return "#%02x%02x%02x" % (level, level, level)


def parse(text):
    """Turn captured lines into rows of (char, fg, bg, bold) cells."""
    grid = []
    for line in text.split("\n")[:ROWS]:
        fg, bg, bold, rev = None, None, False, False
        cells = []
        for token in re.split(r"(\x1b\[[0-9;]*m)", line):
            if token.startswith("\x1b["):
                codes = [int(c) if c else 0 for c in token[2:-1].split(";")]
                i = 0
                while i < len(codes):
                    c = codes[i]
                    if c == 0:
                        fg, bg, bold, rev = None, None, False, False
                    elif c == 1:
                        bold = True
                    elif c == 22:
                        bold = False
                    elif c == 7:
                        rev = True
                    elif c == 27:
                        rev = False
                    elif 30 <= c <= 37:
                        fg = PALETTE[c - 30]
                    elif 90 <= c <= 97:
                        fg = PALETTE[c - 82]
                    elif c == 39:
                        fg = None
                    elif 40 <= c <= 47:
                        bg = PALETTE[c - 40]
                    elif 100 <= c <= 107:
                        bg = PALETTE[c - 92]
                    elif c == 49:
                        bg = None
                    elif c in (38, 48) and codes[i + 1] == 5:
                        if c == 38:
                            fg = color256(codes[i + 2])
                        else:
                            bg = color256(codes[i + 2])
                        i += 2
                    elif c in (38, 48) and codes[i + 1] == 2:
                        rgb = "#%02x%02x%02x" % tuple(codes[i + 2:i + 5])
                        if c == 38:
                            fg = rgb
                        else:
                            bg = rgb
                        i += 4
                    i += 1
            else:
                for ch in token:
                    f, b = fg or FG, bg or BG
                    if rev:
                        f, b = b, f
                    cells.append((ch, f, b, bold))
        grid.append(cells)
    return grid


def render(grid, path):
    size = 26
    regular = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", size)
    boldf = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", size)
    cw, ch = int(round(regular.getlength("M"))), int(size * 1.35)
    pad = 20
    img = Image.new("RGB", (COLS * cw + 2 * pad, ROWS * ch + 2 * pad), BG)
    draw = ImageDraw.Draw(img)
    for r, cells in enumerate(grid):
        last_bg = cells[-1][2] if cells else BG
        # tmux pads the status line with spaces; extend its background to the full width
        for c in range(COLS):
            char, fg, bg, bold = cells[c] if c < len(cells) else (" ", FG, last_bg if r == ROWS - 1 else BG, False)
            x, y = pad + c * cw, pad + r * ch
            if bg != BG:
                draw.rectangle([x, y, x + cw, y + ch], fill=bg)
            if char != " ":
                draw.text((x, y + (ch - size) / 2), char, font=boldf if bold else regular, fill=fg)
    img.save(path)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp:
        raw = capture(Path(tmp))
    print(re.sub(r"\x1b\[[0-9;]*m", "", raw))
    render(parse(raw), OUT)
    print("Generated:", OUT.name)
