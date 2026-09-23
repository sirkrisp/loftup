"""Streaming reader for JPEG/COCO-RLE WebDataset shards, including private B2.

No shard cache or shell commands. Default training selects a fixed global image
pool without repeats; legacy hash-split cycling remains available explicitly.
"""

from contextlib import contextmanager
import glob
import hashlib
import io
import json
import os
from pathlib import Path
import random
import tarfile
from urllib.parse import urlparse
from urllib.request import urlopen

import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils
import torch
from torch.utils.data import DataLoader, IterableDataset, get_worker_info
import torchvision.transforms.functional as TF


def s3_client(endpoint=None):
    import boto3
    from botocore.config import Config
    endpoint = endpoint or os.environ.get("B2_ENDPOINT_URL")
    kwargs = {}
    key_id, key = os.environ.get("B2_APPLICATION_ID"), os.environ.get("B2_APPLICATION_KEY")
    if key_id or key:
        if not key_id or not key:
            raise ValueError("Set both B2_APPLICATION_ID and B2_APPLICATION_KEY")
        kwargs.update(aws_access_key_id=key_id, aws_secret_access_key=key)
    if endpoint:
        parsed = urlparse(endpoint)
        if parsed.scheme != "https" or not parsed.hostname:
            raise ValueError("S3 endpoint must be an HTTPS URL")
        kwargs["endpoint_url"] = endpoint
        if parsed.hostname.endswith(".backblazeb2.com"):
            kwargs["region_name"] = parsed.hostname.split(".")[1]
    return boto3.client("s3", **kwargs, config=Config(
        signature_version="s3v4", connect_timeout=30, read_timeout=120,
        retries={"max_attempts": 3},
    ))


def resolve_shards(source, endpoint=None):
    """Snapshot a local directory/glob, explicit URL list, or s3://bucket/prefix/."""
    if isinstance(source, (list, tuple)) or (not isinstance(source, (str, Path)) and hasattr(source, "__iter__")):
        shards = [str(item) for item in source]
    else:
        source = str(source)
        if source.startswith("s3://"):
            parsed = urlparse(source)
            if parsed.path.endswith(".tar"):
                shards = [source]
            else:
                prefix = parsed.path.lstrip("/").rstrip("/")
                prefix = prefix + "/" if prefix else ""
                pages = s3_client(endpoint).get_paginator("list_objects_v2").paginate(Bucket=parsed.netloc, Prefix=prefix)
                shards = [f"s3://{parsed.netloc}/{item['Key']}" for page in pages
                          for item in page.get("Contents", []) if item["Key"].endswith(".tar")]
        elif source.startswith(("https://", "http://")):
            shards = [source]
        else:
            path = Path(source).expanduser()
            shards = [str(item.resolve()) for item in path.glob("*.tar")] if path.is_dir() else glob.glob(str(path))
    shards = sorted(set(shards))
    if not shards:
        raise ValueError("No completed .tar shards found; check the source path or B2 prefix")
    if any(not urlparse(shard).path.endswith(".tar") for shard in shards):
        raise ValueError("Specify complete .tar shards (not .part or .pending files)")
    return shards


@contextmanager
def open_shard(url, endpoint=None):
    parsed = urlparse(str(url))
    if parsed.scheme == "s3":
        stream = s3_client(endpoint).get_object(Bucket=parsed.netloc, Key=parsed.path.lstrip("/"))["Body"]
    elif parsed.scheme in ("http", "https"):
        stream = urlopen(str(url), timeout=120)
    elif not parsed.scheme:
        stream = open(url, "rb")
    else:
        raise ValueError(f"Unsupported shard scheme: {parsed.scheme}")
    try:
        yield stream
    finally:
        stream.close()


def iter_encoded_samples(shard, endpoint=None):
    """Read adjacent JPG/JSON pairs; malformed samples fail rather than disappear."""
    with open_shard(shard, endpoint) as stream, tarfile.open(fileobj=stream, mode="r|*") as archive:
        key, sample = None, {}
        for member in archive:
            if not member.isfile():
                continue
            stem, separator, extension = member.name.rpartition(".")
            if not separator or extension not in {"jpg", "json"}:
                continue
            if key is not None and stem != key:
                if set(sample) != {"jpg", "json"}:
                    raise ValueError(f"Incomplete WebDataset sample: {key}")
                yield {"__key__": key, "__url__": str(shard), **sample}
                sample = {}
            key = stem
            if extension in sample:
                raise ValueError(f"Duplicate member for sample: {key}")
            with archive.extractfile(member) as handle:
                sample[extension] = handle.read()
        if key is not None:
            if set(sample) != {"jpg", "json"}:
                raise ValueError(f"Incomplete WebDataset sample: {key}")
            yield {"__key__": key, "__url__": str(shard), **sample}


def decode_sample(sample):
    """Public exploration API: return PIL RGB image and original JSON metadata."""
    with Image.open(io.BytesIO(sample["jpg"])) as image:
        image = image.convert("RGB")
    metadata = json.loads(sample["json"])
    return image, metadata


def decode_masks(metadata, size):
    """Yield one independent mask at a time, preserving overlapping instances.

    pycocotools returns Fortran-ordered arrays; PIL silently does a slow implicit
    conversion when handed one, so convert to C-order once here instead.
    """
    width, height = size
    for annotation in metadata["annotations"]:
        rle = annotation["segmentation"]
        if rle["size"] != [height, width]:
            raise ValueError("Mask dimensions do not match the stored image")
        if isinstance(rle["counts"], list):
            rle = mask_utils.frPyObjects(rle, height, width)
        yield np.ascontiguousarray(mask_utils.decode(rle))


def sample_split(key, val_fraction=0.05, seed=42):
    value = int.from_bytes(hashlib.sha256(f"{seed}:{key}".encode()).digest()[:8], "big")
    return "val" if value / 2**64 < val_fraction else "train"


def training_sample(sample, transform, target_transform, max_masks):
    image, metadata = decode_sample(sample)
    img = transform(image) if transform else TF.to_tensor(image)
    height, width = img.shape[-2:]
    labels = torch.full((max_masks, height, width), -1.0)
    # Masks are stored at their original (pre-resize) resolution; decode against
    # that, not the (possibly downscaled) stored image size.
    mask_height, mask_width = metadata["original_size"]
    # Current training transforms are deterministic resize + center crop.
    for index, mask in enumerate(decode_masks(metadata, (mask_width, mask_height))):
        if index >= max_masks:
            break
        label = target_transform(Image.fromarray(mask)) if target_transform else torch.from_numpy(mask.copy()).unsqueeze(0)
        if label.shape[-2:] != (height, width):
            raise ValueError("Image and mask transforms must produce matching dimensions")
        labels[index] = label.squeeze(0)
    return {"img": img, "label": labels, "__key__": sample["__key__"],
            "img_path": f"{sample['__url__']}::{sample['__key__']}.jpg",
            "label_path": f"{sample['__url__']}::{sample['__key__']}.json"}


def consumer_shards(shards, consumer, consumers):
    """Disjoint shards when possible; otherwise stride samples in shared shards."""
    if len(shards) >= consumers:
        return [(url, 0, 1) for url in shards[consumer::consumers]]
    index = consumer % len(shards)
    owners = list(range(index, consumers, len(shards)))
    return [(shards[index], owners.index(consumer), len(owners))]


def buffered_shuffle(samples, size, rng):
    buffer = []
    for sample in samples:
        if len(buffer) < size:
            buffer.append(sample)
        else:
            index = rng.randrange(len(buffer))
            yield buffer[index]
            buffer[index] = sample
    rng.shuffle(buffer)
    yield from buffer


class SA1BWebDataset(IterableDataset):
    def __init__(self, shards, split="train", transform=None, target_transform=None,
                 batch_size=2, batches_per_epoch=1000, max_masks=150,
                 val_fraction=0.05, split_seed=42, shuffle_buffer=32, endpoint=None):
        super().__init__()
        if split not in {"train", "val"} or not 0 < val_fraction < 1:
            raise ValueError("Use split=train/val and 0 < val_fraction < 1")
        if min(batch_size, batches_per_epoch, max_masks) < 1 or shuffle_buffer < 0:
            raise ValueError("Batch sizes, epoch length, and max_masks must be positive")
        self.shards = resolve_shards(shards, endpoint)
        self.split, self.transform, self.target_transform = split, transform, target_transform
        self.batch_size, self.batches_per_epoch = batch_size, batches_per_epoch
        self.max_masks, self.val_fraction, self.split_seed = max_masks, val_fraction, split_seed
        self.shuffle_buffer, self.endpoint = shuffle_buffer, endpoint
        self.distributed_context = None

    def __len__(self):
        return self.batch_size * self.batches_per_epoch

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        # Lightning starts workers after distributed initialization. Environment
        # fallback also works with spawned workers that have no process group.
        if self.distributed_context is not None:
            rank, world = self.distributed_context
        elif torch.distributed.is_available() and torch.distributed.is_initialized():
            rank, world = torch.distributed.get_rank(), torch.distributed.get_world_size()
        else:
            rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
        assigned = consumer_shards(self.shards, rank * workers + worker_id, world * workers)
        quota = len(range(worker_id, self.batches_per_epoch, workers)) * self.batch_size
        if not quota:
            return
        rng = random.Random(torch.initial_seed() if self.split == "train" else self.split_seed)
        produced = 0
        while produced < quota:
            if self.split == "train":
                rng.shuffle(assigned)
            def selected():
                for shard, offset, stride in assigned:
                    for index, sample in enumerate(iter_encoded_samples(shard, self.endpoint)):
                        if index % stride == offset and sample_split(sample["__key__"], self.val_fraction, self.split_seed) == self.split:
                            yield sample
            stream = selected()
            if self.split == "train" and self.shuffle_buffer:
                stream = buffered_shuffle(stream, self.shuffle_buffer, rng)
            before = produced
            try:
                for sample in stream:
                    yield training_sample(sample, self.transform, self.target_transform, self.max_masks)
                    produced += 1
                    if produced == quota:
                        return
            finally:
                stream.close()
            if produced == before:
                raise ValueError(f"No {self.split} samples assigned to rank {rank}, worker {worker_id}; use more shards or fewer workers")


class FiniteSA1BWebDataset(SA1BWebDataset):
    """A fixed sample interval, partitioned into full batches without cycling.

    Prepared shards contain a fixed number of samples. Membership uses sorted
    shard order and member position, so both stages see exactly the same pool.
    """

    def __init__(self, *args, sample_start, sample_count, samples_per_shard, world_size, **kwargs):
        super().__init__(*args, **kwargs)
        if sample_count <= 0 or samples_per_shard <= 0 or world_size <= 0 or sample_start < 0:
            raise ValueError("Sample counts and world size must be positive")
        if sample_count % (self.batch_size * world_size):
            raise ValueError("Sample count must be divisible by batch_size * num_gpus for exact DDP epochs")
        if sample_start + sample_count > len(self.shards) * samples_per_shard:
            raise ValueError("Not enough prepared shards for the requested train/validation images")
        self.sample_start, self.sample_count = sample_start, sample_count
        self.samples_per_shard, self.world_size = samples_per_shard, world_size
        self.batches_per_epoch = sample_count // (self.batch_size * world_size)

    def __iter__(self):
        worker = get_worker_info()
        worker_id, workers = (worker.id, worker.num_workers) if worker else (0, 1)
        rank, world = self.distributed_context or (
            int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)))
        if world != self.world_size:
            raise ValueError(f"Configured {self.world_size} GPUs but loader has {world} ranks")
        # Contiguous, batch-aligned slices keep network reads largely disjoint.
        start_batch = self.batches_per_epoch * worker_id // workers
        end_batch = self.batches_per_epoch * (worker_id + 1) // workers
        start = self.sample_start + (rank * self.batches_per_epoch + start_batch) * self.batch_size
        end = self.sample_start + (rank * self.batches_per_epoch + end_batch) * self.batch_size
        if start == end:
            return
        ranges = []
        for index in range(start // self.samples_per_shard, (end + self.samples_per_shard - 1) // self.samples_per_shard):
            base = index * self.samples_per_shard
            ranges.append((self.shards[index], max(0, start - base), min(self.samples_per_shard, end - base)))
        rng = random.Random(torch.initial_seed())
        if self.split == "train":
            rng.shuffle(ranges)

        def selected():
            for shard, first, stop in ranges:
                seen = 0
                stream = iter_encoded_samples(shard, self.endpoint)
                try:
                    for index, sample in enumerate(stream):
                        seen = index + 1
                        if index >= first:
                            yield sample
                        if seen == stop:
                            break
                finally:
                    stream.close()
                if seen < stop:
                    raise ValueError(f"Shard {shard} has fewer than {stop} samples; check samples_per_shard")

        stream = selected()
        if self.split == "train" and self.shuffle_buffer:
            stream = buffered_shuffle(stream, self.shuffle_buffer, rng)
        try:
            for sample in stream:
                yield training_sample(sample, self.transform, self.target_transform, self.max_masks)
        finally:
            stream.close()


class StreamingDataLoader(DataLoader):
    def __iter__(self):
        # Capture rank in the parent, before fork/spawn creates worker copies.
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            self.dataset.distributed_context = (torch.distributed.get_rank(), torch.distributed.get_world_size())
        else:
            self.dataset.distributed_context = (int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)))
        return super().__iter__()


def make_webdataset_loaders(cfg, transform, target_transform):
    options = cfg.webdataset
    if not options.shards:
        raise ValueError("Set webdataset.shards to a local shard directory/glob or s3://bucket/prefix/")
    shards = resolve_shards(options.shards, options.endpoint)
    common = dict(shards=shards, transform=transform, target_transform=target_transform,
                  max_masks=options.max_masks, val_fraction=cfg.sa1b_val_fraction,
                  split_seed=cfg.sa1b_split_seed, endpoint=options.endpoint,
                  shuffle_buffer=options.shuffle_buffer)
    if options.get("train_samples") is not None:
        finite = dict(samples_per_shard=options.samples_per_shard, world_size=cfg.num_gpus)
        train = FiniteSA1BWebDataset(split="train", batch_size=cfg.batch_size,
            sample_start=0, sample_count=options.train_samples, **finite, **common)
        val = FiniteSA1BWebDataset(split="val", batch_size=1,
            sample_start=options.train_samples, sample_count=options.val_samples, **finite, **common)
        print(f"WebDataset: {len(shards)} shards; {options.train_samples:,} unique training images, "
              f"{options.val_samples:,} validation images; {len(train) // cfg.batch_size:,} batches/rank; "
              f"{cfg.num_workers} streaming workers/rank")
    else:
        train = SA1BWebDataset(split="train", batch_size=cfg.batch_size,
                              batches_per_epoch=options.train_batches, **common)
        val = SA1BWebDataset(split="val", batch_size=1,
                            batches_per_epoch=options.val_batches, **common)
    kwargs = dict(num_workers=cfg.num_workers, pin_memory=True)
    if cfg.num_workers:
        kwargs["prefetch_factor"] = options.get("prefetch_factor", 1)
    return (StreamingDataLoader(train, batch_size=cfg.batch_size, **kwargs),
            StreamingDataLoader(val, batch_size=1, **kwargs))
