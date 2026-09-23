"""Check lecture Markdown against the styling rules in AGENTS.md.

Only the mechanical rules live here; content organization still needs a reader.
Run with no arguments to check every lecture and bonus page.
"""
import re
import sys
from pathlib import Path

MEDIA = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
DEMO = "# LIVE DEMO!"


def body(path):
    """Lines outside code fences, as (number, text), plus the frontmatter values."""
    text = path.read_text(encoding="utf-8")
    meta = {}
    if text.startswith("---\n"):
        end = text.find("\n---", 4)
        for key, value in re.findall(r"^\s*(title_line|role|status):\s*([^\n]+)", text[4:end], re.M):
            meta[key] = value.strip().strip('"').strip("'")
        offset = text[: end + 4].count("\n") + 1
        text = text[end + 4 :].lstrip("\n")
    else:
        offset = 0
    lines, fence = [], None
    for number, line in enumerate(text.splitlines(), start=offset + 1):
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker and fence is None:
            fence = marker.group(1)[0]
            continue
        if marker and fence and marker.group(1)[0] == fence:
            fence = None
            continue
        if fence is None:
            lines.append((number, line))
    return lines, meta


def check(path):
    lines, meta = body(path)
    problems = []
    levels = [(n, len(m.group(1)), m.group(2)) for n, l in lines
              if (m := re.match(r"^(#{1,6}) (.*)", l)) and l.strip() != DEMO]

    previous = 0
    for number, level, title in levels:
        if level > previous + 1 and previous:
            problems.append(f"{path}:{number}: heading jumps from h{previous} to h{level} ({title!r})")
        if level > 4:
            problems.append(f"{path}:{number}: h{level} heading; Notion supports at most h4")
        previous = level

    demos = [n for n, l in lines if l.strip() == DEMO]
    for number in demos:
        # A demo marker may sit inside a Notion synced block, so an XML tag may follow it.
        following = [l for n, l in lines if n > number and l.strip() and not l.lstrip().startswith("<")]
        if following and not following[0].startswith("# "):
            problems.append(f"{path}:{number}: demo marker is followed by {following[0][:40]!r}, not a '#' topic")
    if demos:
        trailing = [(n, l) for n, l in lines if n > demos[-1] and l.strip() and not l.lstrip().startswith("<")]
        if trailing:
            problems.append(f"{path}:{trailing[0][0]}: content after the final demo marker")

    # The opening lines run to the second h1; the first is the page's own title.
    topics = [n for n, level, _ in levels if level == 1]
    first_topic = topics[1] if len(topics) > 1 else None
    inside_xml = 0
    for number, line in lines:
        stripped = line.strip()
        if re.match(r"</[a-z_]+>", stripped):
            inside_xml = max(0, inside_xml - 1)
            continue
        if re.match(r"<[a-z_]+\b[^>]*>$", stripped) and not stripped.endswith("/>"):
            inside_xml += 1
            continue
        if stripped.startswith("#") and "LIVE DEMO" in stripped.upper() and stripped != DEMO:
            problems.append(f"{path}:{number}: demo marker is {stripped!r}, not {DEMO!r}")
        if re.fullmatch(r"(\*\s*){3,}|(-\s*){3,}|(_\s*){3,}", stripped):
            problems.append(f"{path}:{number}: horizontal rule; Notion pages do not use them")
        # A bold line ending in a colon is an inline label, not a section heading, and the
        # opening lines before the first topic are prescribed separately.
        if (not inside_xml and first_topic and number > first_topic
                and (m := re.fullmatch(r"\*\*([^*]+)\*\*", stripped)) and not m.group(1).endswith(":")):
            problems.append(f"{path}:{number}: bold pseudo-heading {stripped[:40]!r}; use a real heading")
        if (m := re.fullmatch(r"!\[([^]]*)\]\(([^)]+)\)", stripped)):
            if not m.group(1).strip() and Path(m.group(2)).suffix.lower() in MEDIA:
                problems.append(f"{path}:{number}: image has no caption in its link text")

    for index, (number, line) in enumerate(lines[:-1]):
        if re.fullmatch(r"!\[[^]]*\]\([^)]+\)", line.strip()):
            rest = [l for _, l in lines[index + 1 : index + 3] if l.strip()]
            if rest and re.fullmatch(r"[*_].+[*_]", rest[0].strip()):
                problems.append(f"{path}:{number}: italic paragraph under an image; captions belong in the link text")

    if meta.get("role") == "lecture" and meta.get("title_line"):
        heading = next((l for _, l in lines if l.strip()), "")
        if heading.strip() != meta["title_line"]:
            problems.append(f"{path}: first line is {heading[:40]!r}, not the front matter title_line")

    # The Notion publisher rejects a table whose rows have different cell counts; an unescaped
    # pipe inside a cell, even in code, adds a cell. Write a literal pipe as \\|.
    table, previous_number = [], None
    for number, line in lines + [(None, "")]:
        if line.startswith("|") and (previous_number is None or number == previous_number + 1):
            table.append((number, line))
        else:
            widths = {len(re.split(r"(?<!\\)\|", row.strip().strip("|"))) for _, row in table}
            if len(widths) > 1:
                problems.append(f"{path}:{table[0][0]}: table rows have different cell counts {sorted(widths)}; escape a literal pipe as \\|")
            table = [(number, line)] if line.startswith("|") else []
        previous_number = number if line.startswith("|") else None
    return problems


def main():
    targets = [Path(a) for a in sys.argv[1:]] or sorted(
        p for n in range(1, 12) for p in Path(f"{n:02d}").glob("*.md") if p.name in {"README.md", "BONUS.md"}
    )
    problems = [problem for path in targets for problem in check(path)]
    for problem in problems:
        print(problem)
    print(f"{len(targets)} files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
