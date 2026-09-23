#!/usr/bin/env python3
"""Publish one mapped Markdown file to its Notion page, safely, in one step.

    python3 scripts/notion_push.py 02/README.md [--dry-run]

Publishing by hand takes five steps in a fixed order, and getting one wrong is
expensive, so this does all of them and refuses to continue when a check fails:

  1. Page every child block to the END and collect the child-page tags. A `<page>`
     tag missing from the payload is not ignored: `ntn pages edit` hangs on it
     until it is killed, which reads as a stall rather than a mistake.
  2. Upload the page's local images, which expire about an hour after upload.
  3. Build the payload, then REFUSE to publish unless every child page found in
     step 1 is present in it.
  4. Write the page.
  5. Attach real images, because `ntn pages edit` stores every `file-upload://`
     source as an external block with an empty URL, which renders broken.

Then it re-reads the page and confirms the children survived and no image is
broken. An image is broken when its source URL is empty; checking only that the
source key exists passes a broken block.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV = {**os.environ, "NOTION_KEYRING": "0"}
EDIT_TIMEOUT = 180


def fail(message: str) -> None:
    print(f"error: {message}", file=sys.stderr)
    raise SystemExit(1)


def api(path: str, *arguments: str) -> dict:
    result = subprocess.run(["ntn", "api", path, *arguments], capture_output=True,
                            text=True, stdin=subprocess.DEVNULL, env=ENV)
    if result.returncode != 0:
        fail(f"{path}: {result.stderr.strip()[:200]}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def blocks(page: str) -> list[dict]:
    cursor, found = None, []
    while True:
        query = f"/v1/blocks/{page}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
        data = api(query)
        found += data.get("results", [])
        if not data.get("has_more"):
            return found
        cursor = data["next_cursor"]


def page_id_of(source: Path) -> str:
    text = source.read_text(encoding="utf-8")
    match = re.search(r'page_id:\s*"?([0-9a-fA-F-]{32,36})', text)
    if not match:
        fail(f"{source} has no page_id in its front matter; it is not a mapped page")
    if "status: mapped" not in text:
        fail(f"{source} is not marked `status: mapped`")
    return match.group(1).replace("-", "")


def broken(block: dict) -> bool:
    image = block.get("image", {})
    return not (image.get(image.get("type")) or {}).get("url")


def run(script: str, *arguments: str) -> None:
    result = subprocess.run([sys.executable, str(ROOT / "scripts" / script), *arguments],
                            capture_output=True, text=True)
    if result.returncode != 0:
        fail(f"{script}: {(result.stderr or result.stdout).strip()[:300]}")
    for line in result.stdout.splitlines():
        print(f"    {line}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    source = args.source
    page = page_id_of(source)

    children = [b for b in blocks(page) if b["type"] in ("child_page", "child_database")]
    tags = [f'<{"page" if b["type"] == "child_page" else "database"} '
            f'url="https://app.notion.com/p/{b["id"].replace("-", "")}">'
            f'{b[b["type"]].get("title", "")}</{"page" if b["type"] == "child_page" else "database"}>'
            for b in children]
    print(f"  {source}: page {page}, {len(children)} child page(s)")

    work = Path(tempfile.mkdtemp())
    current = work / "current"
    current.write_text("\n".join(tags) + ("\n" if tags else ""), encoding="utf-8")

    manifest = work / "media.json"
    print("  uploading media")
    run("notion_media.py", str(manifest), str(source))

    payload = work / "payload"
    built = subprocess.run([sys.executable, str(ROOT / "scripts" / "notion_publish.py"),
                            str(source), str(current), "--media-manifest", str(manifest)],
                           capture_output=True, text=True)
    if built.returncode != 0:
        fail(f"notion_publish.py: {built.stderr.strip()[:300]}")
    payload.write_text(built.stdout, encoding="utf-8")

    # The guard: a child page left out of the payload wedges the edit.
    missing = [b[b["type"]].get("title", "") for b in children
               if b["id"].replace("-", "") not in built.stdout]
    if missing:
        fail(f"the payload does not mention these child pages, which would hang the edit: {missing}")
    print(f"  payload {len(built.stdout)} bytes, all {len(children)} child page(s) preserved")

    if args.dry_run:
        print("  dry run: nothing published")
        return 0

    result = subprocess.run(["ntn", "pages", "edit", page, "--content", built.stdout, "--json"],
                            capture_output=True, text=True, stdin=subprocess.DEVNULL,
                            env=ENV, timeout=EDIT_TIMEOUT)
    if result.returncode != 0 or '"id"' not in result.stdout[:80]:
        fail(f"publish failed: {(result.stderr or result.stdout).strip()[:300]}")
    print("  published")

    print("  attaching images")
    run("notion_attach_media.py", page, str(source))

    after = blocks(page)
    kept = len([b for b in after if b["type"] in ("child_page", "child_database")])
    images = [b for b in after if b["type"] == "image"]
    bad = [b for b in images if broken(b)]
    if kept != len(children):
        fail(f"child pages went from {len(children)} to {kept}")
    if bad:
        fail(f"{len(bad)} image block(s) still have no source")
    print(f"  verified: {kept} child page(s), {len(images)} image(s), 0 broken")
    return 0


if __name__ == "__main__":
    sys.exit(main())
