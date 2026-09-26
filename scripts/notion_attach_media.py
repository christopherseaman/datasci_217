#!/usr/bin/env python3
"""Replace a published page's placeholder image blocks with real uploaded images.

`ntn pages edit` does not resolve the `file-upload://` sources that
scripts/notion_publish.py emits: it stores each one as an external image with an
empty URL, which renders as a broken image. This walks a published page, matches
every such block to its local file by caption, uploads the file, inserts a proper
image block in its place, and deletes the placeholder.

    python3 scripts/notion_attach_media.py <page-id> <source.md> [--dry-run]

Run it immediately after publishing that page. An upload that is never attached
expires about an hour after it is created.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from notion_media import media_targets, upload  # noqa: E402

LINK = re.compile(r"!\[([^]]*)\]\(([^)]+)\)")
ENV = {**os.environ, "NOTION_KEYRING": "0"}


def api(path: str, *arguments: str, body: dict | None = None) -> dict:
    command = ["ntn", "api", path, *arguments]
    if body is not None:
        command += ["-d", json.dumps(body)]
    result = subprocess.run(command, capture_output=True, text=True,
                            stdin=subprocess.DEVNULL, env=ENV)
    if result.returncode != 0:
        raise RuntimeError(f"{path}: {result.stderr.strip()[:300]}")
    return json.loads(result.stdout) if result.stdout.strip() else {}


def page_blocks(page: str) -> list[dict]:
    """Every block on the page, including those inside columns, callouts, and lists."""
    cursor, found = None, []
    while True:
        query = f"/v1/blocks/{page}/children?page_size=100" + (f"&start_cursor={cursor}" if cursor else "")
        data = api(query)
        found += data.get("results", [])
        if not data.get("has_more"):
            break
        cursor = data["next_cursor"]
    for block in list(found):
        if block.get("has_children") and block["type"] not in ("child_page", "child_database"):
            found += page_blocks(block["id"])
    return found


def captioned_images(source: Path) -> list[tuple[str, Path]]:
    """Every local image the document references, as (caption, path), in order."""
    pairs, fenced = [], False
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        for caption, target in LINK.findall(line):
            if target.startswith(("http://", "https://", "data:", "file-upload://")):
                continue
            local = (source.parent / target.split("#", 1)[0]).resolve()
            if local.is_file():
                pairs.append((caption.strip(), local))
    return pairs


def caption_key(caption: str) -> str:
    """A caption as Notion stores it: plain text, so Markdown code and emphasis marks are gone."""
    return re.sub(r"[`*_]", "", caption).strip()


def placeholder(block: dict) -> bool:
    """True when the image block has no usable source, however it is typed."""
    image = block.get("image", {})
    return not (image.get(image.get("type")) or {}).get("url")


def caption_of(block: dict) -> str:
    return "".join(run.get("plain_text", "") for run in block["image"].get("caption", [])).strip()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page")
    parser.add_argument("source", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    wanted = {caption_key(caption): path for caption, path in captioned_images(args.source)}
    broken = [b for b in page_blocks(args.page) if b["type"] == "image" and placeholder(b)]
    if not broken:
        print("no placeholder images on this page")
        return 0

    repaired, skipped = 0, []
    for block in broken:
        caption = caption_of(block)
        local = wanted.get(caption_key(caption))
        if local is None:
            skipped.append(caption[:70] or "(no caption)")
            continue
        if args.dry_run:
            print(f"would attach {local.name} for {caption[:60]!r}")
            repaired += 1
            continue
        upload_id = upload(local)
        # A nested image (inside a column, say) is replaced within its own parent block.
        parent = block.get("parent", {}).get("block_id") or args.page
        api(f"/v1/blocks/{parent}/children", "-X", "PATCH", body={
            "position": {"type": "after_block", "after_block": {"id": block["id"]}},
            "children": [{
                "object": "block", "type": "image",
                "image": {
                    "type": "file_upload",
                    "file_upload": {"id": upload_id},
                    "caption": block["image"].get("caption", []),
                },
            }],
        })
        api(f"/v1/blocks/{block['id']}", "-X", "DELETE")
        print(f"attached {local.name}")
        repaired += 1

    print(f"{repaired} repaired, {len(skipped)} unmatched")
    for caption in skipped:
        print(f"  no local image captioned {caption!r}")
    return 1 if skipped else 0


if __name__ == "__main__":
    sys.exit(main())
