#!/usr/bin/env python3
"""Download official LIBERO Spatial HDF5 files without torch or huggingface_hub."""
from __future__ import annotations

import argparse
import json
from pathlib import Path, PurePosixPath
import re
import sys
from urllib.parse import quote
from urllib.request import Request, urlopen

REPO_ID = "yifengzhu-hf/LIBERO-datasets"
SUITE = "libero_spatial"
BASE_URL = "https://huggingface.co"


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("datasets/libero_spatial"),
                        help="destination suite directory (default: datasets/libero_spatial)")
    parser.add_argument("--task", default="all", help="unique filename substring or all (default: all)")
    parser.add_argument("--list", action="store_true", help="list remote tasks and sizes without downloading")
    return parser


def remote_tasks():
    url = f"{BASE_URL}/api/datasets/{REPO_ID}/tree/main/{SUITE}?limit=100"
    files = []
    while url:
        with urlopen(Request(url, headers={"User-Agent": "nano-vla-libero-downloader"}), timeout=60) as response:
            entries = json.load(response)
            link = response.headers.get("Link", "")
        for entry in entries:
            path = PurePosixPath(entry["path"])
            if (entry["type"] == "file" and path.parent == PurePosixPath(SUITE)
                    and path.suffix == ".hdf5" and ".." not in path.parts):
                files.append(entry)
        match = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = match.group(1) if match else None
    if not files:
        raise ValueError(f"No HDF5 files found in {REPO_ID}/{SUITE}")
    return sorted(files, key=lambda item: item["path"])


def select_tasks(files, task):
    if task == "all":
        return files
    matches = [entry for entry in files if task.lower() in PurePosixPath(entry["path"]).name.lower()]
    if len(matches) != 1:
        raise ValueError(f"Task {task!r} matched {len(matches)} files; use --list or a more specific name")
    return matches


def download(entry, data_dir):
    data_dir.mkdir(parents=True, exist_ok=True)
    target = data_dir / PurePosixPath(entry["path"]).name
    size = entry["size"]
    if target.exists():
        if target.stat().st_size == size:
            print(f"Already downloaded: {target}")
            return
        raise ValueError(f"{target} exists but has the wrong size; move/remove it before retrying")
    partial = target.with_suffix(target.suffix + ".part")
    offset = partial.stat().st_size if partial.exists() else 0
    if offset > size:
        raise ValueError(f"{partial} is larger than expected; remove it before retrying")
    if offset < size:
        url = f"{BASE_URL}/datasets/{REPO_ID}/resolve/main/{quote(entry['path'], safe='/')}"
        headers = {"User-Agent": "nano-vla-libero-downloader"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urlopen(Request(url, headers=headers), timeout=120) as response:
            if offset and response.status == 206:
                content_range = response.headers.get("Content-Range", "")
                if not content_range.startswith(f"bytes {offset}-"):
                    raise ValueError(f"Unexpected resume response: {content_range}")
            elif response.status == 200:
                offset = 0  # Server ignored Range: restart, never append a complete response.
            else:
                raise ValueError(f"Unexpected HTTP status {response.status}")
            print(f"Downloading {target.name}: {offset:,}/{size:,} bytes", flush=True)
            with partial.open("ab" if offset else "wb") as output:
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
    actual = partial.stat().st_size
    if actual != size:
        raise ValueError(f"Incomplete download: {actual:,}/{size:,} bytes; rerun to resume {partial}")
    partial.replace(target)
    print(f"Saved {target} ({size:,} bytes)")


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        files = select_tasks(remote_tasks(), args.task)
        if args.list:
            for entry in files:
                print(f"{PurePosixPath(entry['path']).name}\t{entry['size']:,} bytes")
            print(f"Total: {sum(entry['size'] for entry in files):,} bytes")
        else:
            for entry in files:
                download(entry, args.data_dir)
        return 0
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(1, f"error: {exc}\n")
    except KeyboardInterrupt:
        print("\nInterrupted; rerun to resume .part files.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
