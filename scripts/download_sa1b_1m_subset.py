#!/usr/bin/env python3
"""Stream the first N SA-1B tar shards into flat image/annotation pairs.

Specify --num-tars (1..100). Shards start at sa_000000.tar and are consumed in
numeric order, following the authors' training subset (GitHub issue #23).
Archives are streamed, not stored. Increasing --num-tars extends an existing
ordered download; an interrupted shard is streamed again with saved files skipped.
"""

import argparse
import csv
from http.client import HTTPException
import json
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path, PurePosixPath
from urllib.error import URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
MEMBER = re.compile(r"(sa_[0-9]+)\.(jpg|json)$")
STATE_NAME = ".sa1b-download-state.json"


class DownloadProgress:
    """Count network bytes as tarfile reads, without buffering the archive."""

    def __init__(self, response, name):
        self.response = response
        self.name = name
        length = response.headers.get("Content-Length", "")
        self.total = int(length) if length.isdigit() and int(length) > 0 else None
        self.downloaded = 0
        self.started = time.monotonic()
        self.last_update = self.started
        self.terminal = sys.stderr.isatty()
        self.width = 0

    def __enter__(self):
        self.render()
        return self

    def __exit__(self, *exc):
        self.render()
        if self.terminal:
            print(file=sys.stderr, flush=True)

    def read(self, size=-1):
        data = self.response.read(size)
        self.downloaded += len(data)
        now = time.monotonic()
        if now - self.last_update >= (0.2 if self.terminal else 10):
            self.render()
        return data

    def render(self):
        now = time.monotonic()
        speed = self.downloaded / max(now - self.started, 0.001)
        transferred = f"{self.downloaded / 2**20:,.1f} MiB"
        if self.total:
            fraction = min(self.downloaded / self.total, 1)
            filled = int(24 * fraction)
            bar = "#" * filled + "-" * (24 - filled)
            eta = f"{max(self.total - self.downloaded, 0) / speed:.0f}s" if speed else "--"
            status = f"[{bar}] {fraction:6.1%} {transferred}/{self.total / 2**20:,.1f} MiB | ETA {eta}"
        else:
            status = f"[total unknown] {transferred}"
        line = f"  {self.name} {status} | {speed / 2**20:.1f} MiB/s"
        if self.terminal:
            print("\r" + line.ljust(self.width), end="", file=sys.stderr, flush=True)
            self.width = len(line)
        else:
            print(line, file=sys.stderr, flush=True)
        self.last_update = now

    def message(self, text):
        if self.terminal:
            print("\r" + " " * self.width + "\r", end="", file=sys.stderr)
        print(text, file=sys.stderr, flush=True)
        if self.terminal:
            self.render()


def read_links(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        if not {"file_name", "cdn_link"}.issubset(reader.fieldnames or []):
            raise ValueError("Links file must have file_name and cdn_link TSV columns")
        links = []
        seen = set()
        for row in reader:
            name, url = row["file_name"].strip(), row["cdn_link"].strip()
            if not name.endswith(".tar"):
                continue  # The supplied list also includes sa_images_ids.txt.
            if not url.startswith(("https://", "http://")):
                raise ValueError(f"Invalid archive entry: {name!r}")
            if name not in seen:
                links.append((name, url))
                seen.add(name)
    if not links:
        raise ValueError("Links file contains no archives")
    return links


def inventory(output):
    """Count only nonempty files, including interrupted pairs for slot accounting."""
    files = {}
    for path in output.iterdir():
        match = MEMBER.fullmatch(path.name)
        if match and path.is_file() and path.stat().st_size:
            files.setdefault(match[1], set()).add(match[2])
    return files


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def stream_shard(url, output, files, timeout, name="archive"):
    """Consume an entire shard, saving every image/annotation pair."""
    complete = sum(len(parts) == 2 for parts in files.values())
    request = Request(url, headers={"User-Agent": "LoftUp-SA1B-subset/1.0"})
    with urlopen(request, timeout=timeout) as response, DownloadProgress(response, name) as progress:
        with tarfile.open(fileobj=progress, mode="r|*") as archive:
            for member in archive:
                # Never extract paths, links, devices, or unrelated archive contents.
                match = MEMBER.fullmatch(PurePosixPath(member.name).name)
                if not member.isfile() or not match:
                    continue
                stem, extension = match.groups()
                if stem not in files:
                    files[stem] = set()
                if extension in files[stem]:
                    continue
                if member.size <= 0:
                    raise ValueError(f"Empty archive member: {member.name}")
                destination = output / f"{stem}.{extension}"
                temporary = destination.with_suffix(destination.suffix + ".part")
                source = archive.extractfile(member)
                try:
                    with source, temporary.open("wb") as handle:
                        shutil.copyfileobj(source, handle, length=1024 * 1024)
                    if temporary.stat().st_size != member.size:
                        raise OSError(f"Incomplete archive member: {member.name}")
                    temporary.replace(destination)
                finally:
                    temporary.unlink(missing_ok=True)
                files[stem].add(extension)
                if len(files[stem]) == 2:
                    complete += 1
                    if complete % 1000 == 0:
                        progress.message(f"  Saved {complete:,} image/annotation pairs")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--links-file", type=Path, default=ROOT / "datasets/sa-1b-links.txt")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "datasets/sa1b")
    parser.add_argument("--num-tars", type=int, required=True,
                        help="Download the first N complete shards, starting at sa_000000.tar (1..100)")
    parser.add_argument("--retries", type=int, default=3, help="Retries per shard after the initial attempt")
    parser.add_argument("--timeout", type=float, default=120, help="Socket timeout in seconds")
    args = parser.parse_args()
    if not 1 <= args.num_tars <= 100:
        parser.error("--num-tars must be between 1 and 100")
    if args.retries < 0 or args.timeout <= 0:
        parser.error("--retries must be nonnegative and --timeout must be positive")

    available = dict(read_links(args.links_file))
    names = [f"sa_{index:06d}.tar" for index in range(args.num_tars)]
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Links file is missing requested shards: {', '.join(missing)}")
    links = [(name, available[name]) for name in names]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / STATE_NAME
    files = inventory(output)
    state = {"version": 2, "selection": "first-tars", "completed": [], "in_progress": None}
    if state_path.exists():
        saved = json.loads(state_path.read_text(encoding="utf-8"))
        if saved.get("version") != 2 or saved.get("selection") != "first-tars":
            raise ValueError("Output uses the old shuffled/image-count download. Use a new --output-dir; existing data was not changed.")
        state = saved
        selected_before = set(state["completed"])
        if state.get("in_progress"):
            selected_before.add(state["in_progress"])
        if not selected_before.issubset(names):
            raise ValueError("Output contains shards beyond --num-tars. Increase the count or use a new --output-dir.")
    elif files:
        raise ValueError("Output contains images without ordered download state. Use a new --output-dir to avoid mixing subsets.")
    else:
        save_state(state_path, state)
    complete = sum(len(parts) == 2 for parts in files.values())
    print(f"Found {complete:,} complete pairs; requested {len(names)} shards ({names[0]} through {names[-1]})", flush=True)
    completed = set(state["completed"])
    for index, (name, url) in enumerate(links, 1):
        if name in completed:
            continue
        state["in_progress"] = name
        save_state(state_path, state)
        print(f"Streaming shard {index}/{len(links)}: {name}", flush=True)
        for attempt in range(args.retries + 1):
            try:
                stream_shard(url, output, files, args.timeout, name)
                break
            except (OSError, URLError, HTTPException, tarfile.TarError, EOFError) as error:
                # Do not print signed URLs from exception messages.
                if attempt == args.retries:
                    raise RuntimeError(
                        f"Failed to read {name} ({type(error).__name__}). Check connectivity, disk space, "
                        "and whether the signed links need refreshing, then rerun to resume."
                    ) from None
                delay = min(2 ** attempt, 30)
                print(f"  Read failed ({type(error).__name__}); retrying in {delay}s", flush=True)
                time.sleep(delay)
        incomplete = [stem for stem, parts in files.items() if len(parts) != 2]
        if incomplete:
            raise RuntimeError(
                f"{len(incomplete)} selected images lack a JPEG or JSON after {name}; "
                "the shard was not marked complete. Check the archive and rerun."
            )
        state["completed"].append(name)
        state["in_progress"] = None
        save_state(state_path, state)
    complete = sum(len(parts) == 2 for parts in files.values())
    print(f"Done: {len(names)} complete shards, {complete:,} image/annotation pairs in {output}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except (OSError, ValueError, RuntimeError) as error:
        print(f"Error: {error}", file=sys.stderr)
        sys.exit(1)
