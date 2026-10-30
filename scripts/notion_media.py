#!/usr/bin/env python3
"""Upload a mapped page's local media to Notion and record the uploads in a manifest.

Run this immediately before publishing: an upload that is never attached to a page
expires about an hour after it is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import mimetypes
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}
LINK = re.compile(r"!\[([^]]*)\]\(([^)]+)\)")
# Notion-native video and PDF embeds, written in the page as `<video src="media/x.mp4"></video>`.
MEDIA_TAG = re.compile(r'<(?P<kind>video|pdf)\s+src="(?P<src>[^"]+)"[^>]*>(?P<caption>.*?)</(?P=kind)>')
# A Notion bookmark, written on its own line as `<bookmark url="https://..."></bookmark>`.
BOOKMARK = re.compile(r'<bookmark\s+url="(?P<url>[^"]+)"[^>]*></bookmark>')


def bookmark_urls(source: Path) -> list[str]:
    """URLs of the bookmarks `source` places on their own lines outside code fences, in order."""
    found, fenced = [], False
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        elif not fenced and (match := BOOKMARK.fullmatch(line.strip())):
            found.append(match["url"])
    return found


def media_targets(source: Path) -> list[Path]:
    """Local images, videos, and PDFs that `source` embeds, in document order."""
    found: list[Path] = []
    fenced = False
    for line in source.read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
            continue
        if fenced:
            continue
        images = [target for _, target in LINK.findall(line)
                  if Path(target.split("#", 1)[0]).suffix.lower() in IMAGE_SUFFIXES]
        for target in images + [target for _, target, _ in MEDIA_TAG.findall(line)]:
            if target.startswith(("http://", "https://", "data:", "file-upload://")):
                continue
            local = (source.parent / target.split("#", 1)[0]).resolve()
            if local.is_file() and local not in found:
                found.append(local)
    return found


RETRY_DELAYS = (0, 5, 15, 30, 60, 120)


def run_ntn(command: list[str], stdin_path: Path | None = None, **kwargs) -> subprocess.CompletedProcess:
    """Run an ntn command, waiting and retrying while Notion rate-limits it (HTTP 429)."""
    for delay in RETRY_DELAYS:
        time.sleep(delay)
        # A file on stdin is reopened each attempt, since a failed attempt has already read it.
        with (stdin_path.open("rb") if stdin_path else open(os.devnull, "rb")) as handle:
            result = subprocess.run(command, stdin=handle, capture_output=True,
                                    env={**os.environ, "NOTION_KEYRING": "0"}, **kwargs)
        stdout, stderr = (result.stdout.decode(errors="replace"), result.stderr.decode(errors="replace"))
        result = subprocess.CompletedProcess(command, result.returncode, stdout, stderr)
        if result.returncode == 0 or "429" not in stderr + stdout:
            return result
    return result


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def upload(path: Path) -> str:
    """Upload one file through the Notion CLI and return its file upload id."""
    content_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    result = run_ntn(["ntn", "files", "create", "--filename", path.name,
                      "--content-type", content_type, "--json"], stdin_path=path)
    if result.returncode != 0:
        raise RuntimeError(f"{path}: upload failed: {result.stderr.strip()[:200]}")
    payload = json.loads(result.stdout)
    if payload.get("status") != "uploaded":
        raise RuntimeError(f"{path}: upload status {payload.get('status')!r}")
    return payload["id"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, help="JSON file to write; reused entries are refreshed")
    parser.add_argument("sources", type=Path, nargs="+", help="mapped Markdown files to collect media from")
    parser.add_argument("--dry-run", action="store_true", help="list what would be uploaded")
    args = parser.parse_args()

    manifest = json.loads(args.manifest.read_text(encoding="utf-8")) if args.manifest.is_file() else {}
    root = Path.cwd().resolve()
    for source in args.sources:
        for local in media_targets(source):
            key = local.relative_to(root).as_posix() if local.is_relative_to(root) else str(local)
            if args.dry_run:
                print(f"would upload {key}")
                continue
            file_id = upload(local)
            manifest[key] = {
                "id": file_id,
                "markdown_source": f"file-upload://{file_id}",
                "sha256": digest(local),
                "uploaded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            }
            print(f"uploaded {key} -> {file_id}")
    if not args.dry_run:
        args.manifest.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    sys.exit(main())
