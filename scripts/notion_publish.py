#!/usr/bin/env python3
"""Prepare mapped Markdown for a Notion page; never performs network writes.

Images are uploaded to Notion rather than linked to GitHub: run scripts/notion_media.py
first and pass its manifest with --media-manifest, and every local image becomes a
`file-upload://` source. Without a manifest the image falls back to its raw GitHub URL,
which keeps the payload usable but leaves the media hosted outside Notion.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

REPO = "https://github.com/christopherseaman/datasci_217"
RAW = "https://raw.githubusercontent.com/christopherseaman/datasci_217/main"
MEDIA: dict[str, str] = {}
CHILD = re.compile(r"<(?:page|database|file)\b[^>]*>.*?</(?:page|database|file)>|<(?:page|database|file)\b[^>]*/>", re.I | re.S)
LINK = re.compile(r"(!?)\[([^]]*)\]\(([^)]+)\)")


def frontmatter(text: str) -> tuple[dict[str, str], str]:
    if not text.startswith("---\n"):
        return {}, text
    end = text.find("\n---", 4)
    if end < 0:
        return {}, text
    block, body = text[4:end], text[end + 4 :]
    values = {}
    for key, value in re.findall(r"^\s*(page_id|url|title|title_line):\s*([^\n]+)", block, re.M):
        values[key] = json.loads(value) if value.startswith('"') else value.strip("'")
    return values, body.lstrip("\n")


def _child_blocks(text: str) -> tuple[list[str], str]:
    blocks = CHILD.findall(text)
    return blocks, CHILD.sub("", text)


def _child_urls(blocks: list[str]) -> set[str]:
    urls: set[str] = set()
    for block in blocks:
        urls.update(re.findall(r"(?:url|href)\s*=\s*[\"']([^\"']+)", block, re.I))
    return urls


def _mapped_url(path: Path, target: str) -> str | None:
    local = (path.parent / target.split("#", 1)[0]).resolve()
    if local.suffix.lower() not in {".md", ".markdown"} or not local.is_file():
        return None
    meta, _ = frontmatter(local.read_text(encoding="utf-8"))
    return meta.get("url")


def _link_url(path: Path, target: str, image: bool) -> str:
    if target.startswith(("http://", "https://", "mailto:", "#", "file:", "data:")):
        return target
    if target.startswith("/"):
        return f"{REPO}/tree/main{target}"
    target, hash_sep, fragment = target.partition("#")
    mapped = _mapped_url(path, target)
    if mapped:
        return mapped + (hash_sep + fragment if hash_sep else "")
    resolved = (path.parent / target).resolve()
    try:
        repo_path = resolved.relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        repo_path = resolved.relative_to(path.parent).as_posix()
    is_media = image or Path(target).suffix.lower() in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
    if is_media and MEDIA:
        uploaded = MEDIA.get(repo_path)
        if uploaded is None:
            raise ValueError(f"no Notion upload recorded for {repo_path}; rerun scripts/notion_media.py")
        return uploaded
    base = RAW if is_media else REPO + "/blob/main"
    return f"{base}/{repo_path}" + (hash_sep + fragment if hash_sep else "")


def _replace_links(line: str, path: Path) -> str:
    # Match the whole link first, even when its label contains inline code.
    tokens = re.compile(r"(`+)(?!`)(.*?)\1(?!`)|!?\[[^]]*\]\([^)]+\)")
    def replace(match):
        link = LINK.fullmatch(match[0])
        if link is None:
            return match[0]
        return link[1] + f"[{link[2]}]({_link_url(path, link[3], bool(link[1]))})"
    return tokens.sub(replace, line)


def _is_nav_line(line: str, path: Path, child_urls: set[str]) -> bool:
    links = list(LINK.finditer(line.strip()))
    remainder = LINK.sub("", line.strip())
    boilerplate = re.fullmatch(r"See\s+for (?:the )?optional (?:extensions|extension notes)\.", remainder)
    if not links or (not boilerplate and re.sub(r"[\s·|,;:/-]", "", remainder)):
        return False
    return all(not m.group(1) and _link_url(path, m.group(3), False) in child_urls for m in links)


def table(block: list[str]) -> str:
    """The Notion API requires table blocks rather than pipe-table syntax."""
    rows = [re.split(r"(?<!\\)\|", line.strip().strip("|")) for line in block]
    if len(rows) < 2 or not all(re.fullmatch(r"\s*:?-{3,}:?\s*", cell) for cell in rows[1]):
        return "".join(block)
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("Unequal table row widths; escape literal pipes as \\|")
    return '<table header-row="true">\n' + "\n".join(
        # Inside <td> a pipe is plain text, and Notion would keep the backslash.
        "<tr>" + "".join("<td>" + cell.strip().replace("\\|", "|") + "</td>" for cell in row) + "</tr>"
        for row in [rows[0], *rows[2:]]
    ) + "\n</table>\n"


def prepare(source: Path, current: Path) -> str:
    metadata, body = frontmatter(source.read_text(encoding="utf-8"))
    current_text = current.read_text(encoding="utf-8")
    _, current_body = frontmatter(current_text)
    blocks, _ = _child_blocks(current_body)
    child_urls = _child_urls(blocks)
    child_by_url = {url: block for block in blocks for url in _child_urls([block])}
    placed_children: set[str] = set()

    lines: list[str] = []
    fence: tuple[str, int] | None = None
    local_title: str | None = metadata.get("title_line")
    table_lines: list[str] = []
    for line in body.splitlines(keepends=True):
        if fence is None and line.startswith("|"):
            table_lines.append(_replace_links(line, source))
            continue
        if table_lines:
            lines.append(table(table_lines))
            table_lines = []
        marker = re.match(r"^\s*(`{3,}|~{3,})", line)
        if marker and fence is None:
            fence = (marker.group(1)[0], len(marker.group(1)))
            lines.append(line)
            continue
        if marker and fence and marker.group(1)[0] == fence[0] and len(marker.group(1)) >= fence[1] and not line[marker.end():].strip():
            fence = None
            lines.append(line)
            continue
        if fence is None:
            link = LINK.fullmatch(line.strip())
            if link and not link[1]:
                child = child_by_url.get(_link_url(source, link[3], False))
                if child:
                    if child not in placed_children:
                        lines.append(child + "\n")
                        placed_children.add(child)
                    continue
            if _is_nav_line(line, source, child_urls):
                continue
        if fence is None and line.strip() == local_title:
            local_title = None
            continue
        if fence is None:
            line = _replace_links(line, source)
        lines.append(line)

    if table_lines:
        lines.append(table(table_lines))
    if fence:
        raise ValueError("Unclosed code fence")
    content = "".join(lines)
    prefix = "\n".join(block for block in blocks if block not in placed_children)
    if prefix:
        prefix += "\n\n"
    return prefix + content.lstrip("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("current", type=Path, help="fetched Notion page content")
    parser.add_argument("--media-manifest", type=Path, help="manifest written by scripts/notion_media.py")
    args = parser.parse_args()
    if args.media_manifest:
        manifest = json.loads(args.media_manifest.read_text(encoding="utf-8"))
        MEDIA.update({key: entry["markdown_source"] for key, entry in manifest.items()})
    print(prepare(args.source, args.current), end="")


if __name__ == "__main__":
    main()
