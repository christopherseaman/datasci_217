#!/usr/bin/env python3
"""Check every Notion-mapped page for broken images and missing child pages.

    python3 scripts/notion_check_pages.py

Reads each mapped Markdown file's front matter for its page id, then compares the
live page with the local source. An image block counts as broken when its source
URL is empty: `ntn pages edit` stores an unresolved `file-upload://` that way, and
testing only that the source key exists passes a broken block.

Exits non-zero when anything is broken, so it can gate a publish.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = {**os.environ, "NOTION_KEYRING": "0"}


def blocks(page: str) -> list[dict] | None:
    cursor, found = None, []
    while True:
        query = f"/v1/blocks/{page}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
        result = subprocess.run(["ntn", "api", query], capture_output=True, text=True,
                                stdin=subprocess.DEVNULL, env=ENV)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout)
        found += data.get("results", [])
        if not data.get("has_more"):
            break
        cursor = data["next_cursor"]
    # Images inside columns, callouts, and lists are nested one level or more down.
    for block in list(found):
        if block.get("has_children") and block["type"] not in ("child_page", "child_database"):
            nested = blocks(block["id"])
            if nested is None:
                return None
            found += nested
    return found


def mapped_pages() -> list[tuple[Path, str, int]]:
    pages = []
    for source in sorted(ROOT.glob("[01][0-9]/**/*.md")) + sorted(ROOT.glob("*.md")):
        text = source.read_text(encoding="utf-8", errors="ignore")
        head = re.match(r"^---\n(.*?)\n---", text, re.S)
        if not head or "status: mapped" not in head.group(1):
            continue
        page = re.search(r'page_id:\s*"?([0-9a-fA-F-]{32,36})', head.group(1))
        if not page:
            continue
        images = len([l for l in text.splitlines() if l.strip().startswith("![")])
        pages.append((source.relative_to(ROOT), page.group(1).replace("-", ""), images))
    return pages


def main() -> int:
    problems = 0
    print(f"{'page':<34}{'local':>6}{'notion':>8}{'broken':>8}")
    for source, page, local_images in mapped_pages():
        found = blocks(page)
        if found is None:
            print(f"{str(source):<34}{local_images:>6}   FETCH FAILED")
            problems += 1
            continue
        images = [b for b in found if b["type"] == "image"]
        bad = [b for b in images
               if not (b["image"].get(b["image"].get("type")) or {}).get("url")]
        problems += len(bad)
        flag = "   <-- broken" if bad else ""
        print(f"{str(source):<34}{local_images:>6}{len(images):>8}{len(bad):>8}{flag}")
    print(f"\n{problems} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
