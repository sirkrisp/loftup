#!/usr/bin/env python3
"""Download SA-1B, resize image/mask pairs, and upload WebDataset tars to B2.

Credentials use B2_APPLICATION_ID/B2_APPLICATION_KEY, or the standard AWS variables.
Downloads, pair conversion/shard writing, and uploads overlap via bounded queues.
Source members stay in memory; only output shards are written to disk.
A Rich dashboard is shown in interactive terminals. Only finalized shards are uploaded. Rerun the same command to resume.
"""

import argparse
from collections import deque
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor
import fcntl
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
from queue import Empty, Full, Queue
from threading import Event, Thread
import re
import sys
import tarfile
import time
from urllib.parse import urlparse

from PIL import Image

try:
    from . import download_sa1b_1m_subset as download
    from .sa1b_dashboard import Dashboard
except ImportError:
    import download_sa1b_1m_subset as download
    from sa1b_dashboard import Dashboard


def b2_credentials():
    """Prefer an explicit B2 pair; otherwise use boto3's normal credential lookup."""
    key_id = os.environ.get("B2_APPLICATION_ID")
    key = os.environ.get("B2_APPLICATION_KEY")
    if key_id or key:
        if not key_id or not key:
            raise ValueError("Set both B2_APPLICATION_ID and B2_APPLICATION_KEY")
        return {"aws_access_key_id": key_id, "aws_secret_access_key": key}
    return {}


def client_error_details(error):
    """Report selected service error fields, never the request or full response."""
    response = getattr(error, "response", {})
    details = response.get("Error", {})
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode", "unknown")
    message = str(details.get("Message", ""))
    for name in (
        "B2_APPLICATION_ID",
        "B2_APPLICATION_KEY",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
    ):
        value = os.environ.get(name)
        if value:
            message = message.replace(value, "[redacted]")
    message = re.sub(r"https?://\S+", "[URL redacted]", message)
    message = " ".join(message.split())[:500]
    operation = getattr(error, "operation_name", "S3 request")
    return f"{operation}: HTTP {status}, {details.get('Code', 'unknown')}: {message}"


def encode_pair(key, image_bytes, annotation_bytes, size=896, quality=95):
    """Resize the shorter side only downwards; keep masks at their original resolution.

    Masks are stored unresized (original RLE/area/bbox passed through as-is):
    training (target_transform in train_loftup_stage*.py) resizes every mask with
    NEAREST to its actual crop size on every epoch regardless of stored resolution,
    so resizing masks here at prep time was pure duplicate work with no effect on
    the final training input.

    Returns a fourth `timing` dict ({image_seconds, mask_seconds, num_annotations})
    so callers can profile where encoding time goes.
    """
    source = json.loads(annotation_bytes)
    image_start = time.monotonic()
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB")
        width, height = image.size
        scale = min(1.0, size / min(width, height))
        target = (max(1, round(width * scale)), max(1, round(height * scale)))
        if target != image.size:
            image = image.resize(target, Image.Resampling.BILINEAR)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=quality)
    image_seconds = time.monotonic() - image_start

    mask_start = time.monotonic()
    raw_annotations = source["annotations"]
    annotations = []
    for annotation in raw_annotations:
        rle = annotation["segmentation"]
        if rle["size"] != [height, width]:
            raise ValueError(f"Mask/image dimensions differ for {key}.jpg")
        annotations.append(
            {
                "id": annotation["id"],
                "segmentation": rle,
                "area": annotation["area"],
                "bbox": annotation["bbox"],
            }
        )
    mask_seconds = time.monotonic() - mask_start

    metadata = {
        "image": {
            **source.get("image", {}),
            "file_name": f"{key}.jpg",
            "width": target[0],
            "height": target[1],
        },
        "original_size": [height, width],
        "annotations": annotations,
    }
    timing = {
        "image_seconds": image_seconds,
        "mask_seconds": mask_seconds,
        "num_annotations": len(raw_annotations),
    }
    return (
        key,
        output.getvalue(),
        json.dumps(metadata, separators=(",", ":")).encode(),
        timing,
    )


def _encode_pair_task(pair, size, quality):
    """Run in a worker process so JPEG/mask encoding uses its own CPU core."""
    key, jpg, metadata, timing = encode_pair(*pair, size, quality)
    return (key, jpg, metadata), timing


def _submit(pool, fn, args):
    """Submit to pool, or run inline (as an already-resolved future) if pool is None."""
    if pool is None:
        future = Future()
        try:
            future.set_result(fn(*args))
        except BaseException as error:
            future.set_exception(error)
        return future
    return pool.submit(fn, *args)


def write_sample(archive, sample):
    key, jpg, metadata = sample
    for extension, payload in (("jpg", jpg), ("json", metadata)):
        member = tarfile.TarInfo(f"{key}.{extension}")
        member.size = len(payload)
        member.mode = 0o644
        archive.addfile(member, io.BytesIO(payload))


class SourceProgress(download.DownloadProgress):
    def __init__(self, response, name, dashboard):
        super().__init__(response, name)
        self.dashboard = dashboard
        if dashboard and dashboard.enabled:
            self.terminal = False

    def render(self):
        if self.dashboard and self.dashboard.enabled:
            self.dashboard.download(
                self.name, self.downloaded, self.total, self.started
            )
            self.last_update = time.monotonic()
        else:
            super().render()

    def read(self, size=-1):
        data = super().read(size)
        if self.dashboard and self.dashboard.enabled:
            self.dashboard.download(
                self.name, self.downloaded, self.total, self.started
            )
        return data


def stream_pairs(args, url, name, stopped, skip=0, excluded=()):
    """Yield raw pairs in member completion order, without extracting files.

    Replayed pairs are discarded as soon as their second member arrives.
    Nonadjacent members are buffered up to --max-buffer-mb; never spill to disk.
    """
    pending = {}
    seen = {}
    completed = 0
    buffered_bytes = 0
    limit = int(getattr(args, "max_buffer_mb", 256) * 1e6)
    request = download.Request(url, headers={"User-Agent": "LoftUp-SA1B-subset/1.0"})
    with download.urlopen(request, timeout=args.timeout) as response:
        with SourceProgress(
            response, name, getattr(args, "dashboard", None)
        ) as progress:
            with tarfile.open(fileobj=progress, mode="r|*") as archive:
                for member in archive:
                    if stopped.is_set():
                        raise DownloadStopped()
                    match = download.MEMBER.fullmatch(PurePosixPath(member.name).name)
                    if not member.isfile() or not match:
                        continue
                    key, extension = match.groups()
                    parts = seen.setdefault(key, set())
                    if extension in parts:
                        continue
                    if member.size <= 0:
                        raise ValueError(
                            f"Empty archive member in {name}: {key}.{extension}"
                        )
                    parts.add(extension)
                    # While replaying, some first halves may belong to later pairs.
                    # Keep those halves until their second member establishes order.
                    keep = key not in excluded
                    if keep:
                        if buffered_bytes + member.size > limit:
                            raise ValueError(
                                f"Pair buffer exceeded --max-buffer-mb in {name}; increase the limit for nonadjacent members"
                            )
                        with archive.extractfile(member) as source:
                            payload = source.read()
                        if len(payload) != member.size:
                            raise OSError(
                                f"Incomplete member in {name}: {key}.{extension}"
                            )
                        pending.setdefault(key, {})[extension] = payload
                        buffered_bytes += len(payload)
                    if len(parts) == 2:
                        completed += 1
                        pair = pending.pop(key, {})
                        buffered_bytes -= sum(map(len, pair.values()))
                        if completed > skip and keep:
                            yield (key, pair["jpg"], pair["json"]), completed
                if not seen or any(len(parts) != 2 for parts in seen.values()):
                    raise ValueError(f"Incomplete or empty image/JSON pairs in {name}")
                if completed < skip:
                    raise ValueError(
                        f"Source {name} has fewer pairs than the resume cursor"
                    )


class DownloadStopped(Exception):
    """Internal signal used to stop a blocked download producer."""


def iter_samples(args, links, cursor, excluded=(), pool=None):
    """Download while submitting each pair for conversion, in a bounded queue of futures.

    Submission happens on the download thread as soon as a pair arrives, so pool
    workers can encode earlier samples while later ones are still downloading.
    The consumer waits on each future in submission order, so output order and
    the resume cursor are unaffected by which sample finishes encoding first.
    """
    first_source, first_offset = cursor
    for source_index in range(first_source, len(links)):
        name, url = links[source_index]
        ready = Queue(maxsize=getattr(args, "max_pending_pairs", 32))
        stopped = Event()
        done = Event()
        errors = []
        offset = first_offset if source_index == first_source else 0

        def produce():
            delivered = offset
            try:
                for attempt in range(args.retries + 1):
                    try:
                        for pair, position in stream_pairs(
                            args,
                            url,
                            name,
                            stopped,
                            delivered,
                            excluded if source_index == first_source else (),
                        ):
                            future = _submit(
                                pool,
                                _encode_pair_task,
                                (pair, args.size, args.jpeg_quality),
                            )
                            while not stopped.is_set():
                                try:
                                    ready.put((future, position), timeout=0.1)
                                    delivered = position
                                    break
                                except Full:
                                    pass
                            else:
                                raise DownloadStopped()
                        return
                    except (
                        OSError,
                        EOFError,
                        tarfile.TarError,
                        download.HTTPException,
                    ):
                        if attempt == args.retries:
                            raise RuntimeError(
                                f"Download failed for {name}; refresh links or rerun to resume"
                            ) from None
                        if stopped.wait(min(2**attempt, 30)):
                            return
            except DownloadStopped:
                pass
            except BaseException as error:
                errors.append(error)
            finally:
                done.set()

        worker = Thread(target=produce, name="sa1b-download")
        worker.start()
        try:
            while True:
                try:
                    future, position = ready.get(timeout=0.1)
                except Empty:
                    if done.is_set() and ready.empty():
                        if errors:
                            raise errors[0]
                        break
                    continue
                yield future.result(), [source_index, position]
        finally:
            stopped.set()
            worker.join()


def iter_resume_samples(args, links, state, pool):
    cursor = state["cursor"]
    if cursor[0] == state.get("legacy_source", -1):
        if "legacy_skip" not in state:
            # Old checkpoints count numerically sorted IDs. Scan names once to
            # identify committed pairs, then switch to archive completion order.
            name, url = links[cursor[0]]
            keys = set()
            for attempt in range(args.retries + 1):
                try:
                    request = download.Request(
                        url, headers={"User-Agent": "LoftUp-SA1B-subset/1.0"}
                    )
                    with download.urlopen(request, timeout=args.timeout) as response:
                        with SourceProgress(
                            response, name, getattr(args, "dashboard", None)
                        ) as progress:
                            with tarfile.open(fileobj=progress, mode="r|*") as archive:
                                for member in archive:
                                    match = download.MEMBER.fullmatch(
                                        PurePosixPath(member.name).name
                                    )
                                    if member.isfile() and match and match[2] == "jpg":
                                        keys.add(match[1])
                    break
                except (OSError, EOFError, tarfile.TarError, download.HTTPException):
                    if attempt == args.retries:
                        raise RuntimeError(
                            f"Download failed while migrating {name}; rerun to resume"
                        ) from None
                    keys.clear()
                    time.sleep(min(2**attempt, 30))
            if len(keys) < cursor[1]:
                raise ValueError(
                    f"Source {name} has fewer pairs than the legacy cursor"
                )
            state["legacy_skip"] = sorted(keys, key=lambda key: int(key[3:]))[
                : cursor[1]
            ]
            state["cursor"] = cursor = [cursor[0], 0]
            download.save_state(args.work_dir / "state.json", state)
        yield from iter_samples(
            args, links, cursor, excluded=set(state["legacy_skip"]), pool=pool
        )
    else:
        yield from iter_samples(args, links, cursor, pool=pool)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class BackgroundUploads:
    """Bounded queue: upload overlaps preprocessing; errors propagate to producer."""

    def __init__(self, args, client):
        self.args = args
        self.client = client
        self.dashboard = getattr(args, "dashboard", None)
        self.pool = ThreadPoolExecutor(max_workers=args.upload_workers)
        self.pending = deque()

    def submit(self, path):
        if len(self.pending) >= self.args.max_pending_uploads:
            self.pending.popleft().result()
        # Surface failures promptly even when an earlier upload is still running.
        for future in self.pending:
            if future.done():
                future.result()
        if self.dashboard:
            self.dashboard.queued(path.name, path.stat().st_size)
        self.pending.append(self.pool.submit(self.upload, path))

    def upload(self, path):
        try:
            if self.dashboard:
                self.dashboard.upload_status(path.name, "Hashing")
            self._upload(path)
        except BaseException:
            if self.dashboard:
                self.dashboard.upload_status(path.name, "Failed")
            raise

    def _upload(self, path):
        from botocore.exceptions import ClientError
        from boto3.s3.transfer import TransferConfig

        key = "/".join(
            part for part in (self.args.prefix.strip("/"), path.name) if part
        )
        digest = sha256(path)
        length = path.stat().st_size
        for attempt in range(self.args.retries + 1):
            try:
                if self.dashboard:
                    self.dashboard.upload_status(
                        path.name, "Checking remote", reset=True
                    )
                try:
                    remote = self.client.head_object(Bucket=self.args.bucket, Key=key)
                except ClientError as error:
                    if error.response["Error"]["Code"] not in (
                        "404",
                        "NoSuchKey",
                        "NotFound",
                    ):
                        raise
                    remote = None
                if remote is not None:
                    if (
                        remote["ContentLength"] != length
                        or remote.get("Metadata", {}).get("sha256") != digest
                    ):
                        raise FileExistsError(
                            f"Different object already exists: {key}; use a fresh prefix"
                        )
                else:
                    if self.dashboard:
                        self.dashboard.upload_status(path.name, "Uploading")
                    self.client.upload_file(
                        str(path),
                        self.args.bucket,
                        key,
                        ExtraArgs={"Metadata": {"sha256": digest}},
                        Config=TransferConfig(max_concurrency=2),
                        Callback=lambda amount: (
                            self.dashboard.advance(path.name, amount)
                            if self.dashboard
                            else None
                        ),
                    )
                    if self.dashboard:
                        self.dashboard.upload_status(path.name, "Verifying")
                    remote = self.client.head_object(Bucket=self.args.bucket, Key=key)
                    if (
                        remote["ContentLength"] != length
                        or remote.get("Metadata", {}).get("sha256") != digest
                    ):
                        raise RuntimeError(f"Upload verification failed: {key}")
                download.save_state(
                    path.with_suffix(".uploaded.json"),
                    {"key": key, "sha256": digest, "bytes": length},
                )
                if not self.args.keep_local:
                    path.unlink()
                if self.dashboard:
                    self.dashboard.completed(path.name)
                print(f"Uploaded {key} ({length / 1e6:.1f} MB)", flush=True)
                return
            except FileExistsError:
                raise
            except Exception as error:
                if attempt == self.args.retries:
                    raise RuntimeError(
                        f"Upload failed for {path.name} ({type(error).__name__}); local file retained"
                    ) from None
                if self.dashboard:
                    self.dashboard.upload_status(path.name, "Retrying")
                time.sleep(min(2**attempt, 30))

    def close(self):
        try:
            for future in self.pending:
                future.result()
        finally:
            self.pool.shutdown(wait=True, cancel_futures=True)


def run(args, links, client):
    destination = (
        f"s3://{args.bucket}/{args.prefix.strip('/')}"
        if client is not None
        else "Local only"
    )
    with Dashboard(
        destination,
        enabled=sys.stderr.isatty() and not getattr(args, "no_dashboard", False),
    ) as dashboard:
        args.dashboard = dashboard
        return prepare(args, links, client)


def prepare(args, links, client):
    dashboard = args.dashboard
    state_path = args.work_dir / "state.json"
    configuration = {
        key: getattr(args, key)
        for key in (
            "size",
            "jpeg_quality",
            "samples_per_shard",
            "bucket",
            "endpoint",
            "prefix",
            "final_shard",
        )
    }
    configuration["sources"] = [name for name, _ in links]
    state = {
        "version": 2,
        "configuration": configuration,
        "cursor": [0, 0],
        "samples_per_shard": args.samples_per_shard,
        "shards": [],
        "finished": False,
    }
    if state_path.exists():
        state = json.loads(state_path.read_text())
        previous = dict(state["configuration"])
        old_sources = previous.pop("sources")
        previous.pop("target_mb", None)
        previous.pop("calibration_samples", None)
        previous["samples_per_shard"] = (
            state["samples_per_shard"] or args.samples_per_shard
        )
        current = dict(configuration)
        new_sources = current.pop("sources")
        if (
            state["version"] not in (1, 2)
            or previous != current
            or new_sources[: len(old_sources)] != old_sources
        ):
            raise ValueError(
                "Resume settings differ; use the original settings or a new work directory/prefix"
            )
        state["samples_per_shard"] = args.samples_per_shard
        if state["version"] == 1:
            # Only this source uses the old cursor/order; later sources stream.
            state["legacy_source"] = state["cursor"][0]
            state["version"] = 2
            download.save_state(state_path, state)
        if len(new_sources) > len(old_sources):
            state["configuration"] = configuration
            state["finished"] = False
            download.save_state(state_path, state)
    else:
        download.save_state(state_path, state)
    output = args.work_dir / "shards"
    output.mkdir(exist_ok=True)
    uploads = BackgroundUploads(args, client) if client is not None else None
    pool = ProcessPoolExecutor(max_workers=max(1, getattr(args, "encode_workers", 1)))
    samples = None
    try:
        for entry in state["shards"]:
            path = output / entry["name"]
            if path.with_suffix(".uploaded.json").exists():
                dashboard.completed(path.name)
            elif uploads:
                if not path.exists():
                    raise FileNotFoundError(f"Missing unuploaded shard: {path}")
                uploads.submit(path)
        if state["finished"]:
            return
        samples = iter_resume_samples(args, links, state, pool)
        count = state["samples_per_shard"]
        print(f"Using {count:,} images per full shard", flush=True)
        exhausted = False
        while not exhausted:
            name = f"sa1b-{len(state['shards']):06d}.tar"
            path = output / name
            temporary = path.with_suffix(".tar.part")
            written = 0
            cursor = state["cursor"]
            with tarfile.open(temporary, "w", format=tarfile.USTAR_FORMAT) as archive:
                for _ in range(count):
                    item = next(samples, None)
                    if item is None:
                        exhausted = True
                        break
                    (sample, timing), cursor = item
                    write_sample(archive, sample)
                    written += 1
                    dashboard.fill(name, written, count)
                    dashboard.timing(timing)
            if not written:
                temporary.unlink()
                break
            if written < count and args.final_shard == "retain":
                temporary.replace(output / "remainder.tar.pending")
                dashboard.set_status("Remainder retained locally")
                print(
                    f"Retained {written} leftover samples in remainder.tar.pending (not uploaded)",
                    flush=True,
                )
                break
            temporary.replace(path)
            state["cursor"] = cursor
            state["shards"].append(
                {"name": name, "samples": written, "bytes": path.stat().st_size}
            )
            download.save_state(state_path, state)
            print(
                f"Created {name}: {written:,} images, {path.stat().st_size / 1e6:.1f} MB",
                flush=True,
            )
            if uploads:
                uploads.submit(path)
        state["finished"] = True
        if not (written and written < count and args.final_shard == "retain"):
            (output / "remainder.tar.pending").unlink(missing_ok=True)
        download.save_state(state_path, state)
    finally:
        try:
            if samples is not None:
                samples.close()
        finally:
            try:
                if uploads:
                    if sys.exc_info()[0] is None:
                        dashboard.set_status("Waiting for uploads")
                    uploads.close()
            finally:
                pool.shutdown(wait=True, cancel_futures=True)
    total = sum(entry["samples"] for entry in state["shards"])
    average = (
        sum(entry["bytes"] for entry in state["shards"])
        / max(1, len(state["shards"]))
        / 1e6
    )
    print(
        f"Done: {len(state['shards'])} shards, {total:,} images, average {average:.1f} MB",
        flush=True,
    )
    if dashboard.timing_count:
        avg_image_ms = dashboard.image_seconds_total / dashboard.timing_count * 1000
        avg_mask_ms = dashboard.mask_seconds_total / dashboard.timing_count * 1000
        avg_annotations = dashboard.annotations_total / dashboard.timing_count
        print(
            f"Avg encode time: {avg_image_ms:.1f} ms image, {avg_mask_ms:.1f} ms masks/image "
            f"({avg_annotations:.1f} masks/image)",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--links-file", type=Path, default=download.ROOT / "datasets/sa-1b-links.txt"
    )
    parser.add_argument(
        "--num-tars",
        type=int,
        required=True,
        help="Number of source tars, starting at zero",
    )
    parser.add_argument(
        "--work-dir", type=Path, default=download.ROOT / "datasets/sa1b-webdataset"
    )
    parser.add_argument(
        "--size",
        type=int,
        default=896,
        help="Maximum shorter side; never upscale or crop",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument(
        "--samples-per-shard",
        type=int,
        default=1000,
        help="Images per full shard (default: 1000)",
    )
    parser.add_argument(
        "--final-shard",
        choices=("keep", "retain"),
        default="retain",
        help="keep: upload smaller last shard; retain: leave leftovers locally",
    )
    parser.add_argument("--bucket", default=os.environ.get("B2_BUCKET"))
    parser.add_argument("--endpoint", default=os.environ.get("B2_ENDPOINT_URL"))
    parser.add_argument("--prefix", default="sa1b-896")
    parser.add_argument(
        "--max-pending-pairs",
        type=int,
        default=10000,
        help="Maximum downloaded pairs queued ahead of conversion",
    )
    parser.add_argument(
        "--encode-workers",
        type=int,
        default=os.cpu_count() or 1,
        help="Parallel worker processes for JPEG/mask encoding (default: all CPU cores)",
    )
    parser.add_argument(
        "--max-buffer-mb",
        type=float,
        default=10000,
        help="Memory limit in decimal MB for raw pairs awaiting matching members",
    )
    parser.add_argument(
        "--no-dashboard", action="store_true", help="Use plain progress logs"
    )
    parser.add_argument("--upload-workers", type=int, default=2)
    parser.add_argument("--max-pending-uploads", type=int, default=4)
    parser.add_argument(
        "--keep-local", action="store_true", help="Keep uploaded tars on disk"
    )
    parser.add_argument(
        "--local-only", action="store_true", help="Build tars without uploading"
    )
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=120)
    args = parser.parse_args()
    for key in (
        "num_tars",
        "size",
        "samples_per_shard",
        "upload_workers",
        "max_pending_uploads",
        "max_pending_pairs",
        "max_buffer_mb",
        "encode_workers",
        "timeout",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            parser.error(f"--{key.replace('_', '-')} must be positive and finite")
    if args.retries < 0 or not 1 <= args.jpeg_quality <= 100:
        parser.error("Invalid retries, JPEG quality, or samples per shard")
    client = None
    if not args.local_only:
        if not args.bucket or not args.endpoint:
            parser.error(
                "Provide --bucket and --endpoint (or B2_BUCKET/B2_ENDPOINT_URL), or --local-only"
            )
        parsed = urlparse(args.endpoint)
        if parsed.scheme != "https" or not (parsed.hostname or "").endswith(
            ".backblazeb2.com"
        ):
            parser.error("--endpoint must be an HTTPS Backblaze S3 endpoint")
        import boto3
        from botocore.config import Config

        client = boto3.client(
            "s3",
            endpoint_url=args.endpoint,
            **b2_credentials(),
            region_name=parsed.hostname.split(".")[1],
            config=Config(
                signature_version="s3v4",
                retries={"max_attempts": 3},
                connect_timeout=30,
                read_timeout=args.timeout,
            ),
        )
        from botocore.exceptions import ClientError

        try:
            client.head_bucket(Bucket=args.bucket)
        except ClientError as error:
            raise RuntimeError(
                f"Cannot access bucket {args.bucket!r}. {client_error_details(error)}. "
                "Check that the application key allows this bucket and that the endpoint matches its region."
            ) from None
    available = dict(download.read_links(args.links_file))
    names = [f"sa_{index:06d}.tar" for index in range(args.num_tars)]
    missing = [name for name in names if name not in available]
    if missing:
        raise ValueError(f"Links file is missing {missing[0]}")
    args.work_dir.mkdir(parents=True, exist_ok=True)
    with (args.work_dir / ".lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another process is using this work directory") from None
        run(args, [(name, available[name]) for name in names], client)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted; rerun the same command to resume.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        # SDK/network exception text may include credentials or signed URLs.
        if isinstance(
            error, (ValueError, RuntimeError, FileExistsError, FileNotFoundError)
        ):
            print(f"Error: {error}", file=sys.stderr)
        else:
            print(
                f"Error: {type(error).__name__}; check configuration/connectivity and rerun.",
                file=sys.stderr,
            )
        sys.exit(1)
