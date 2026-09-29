"""Read every pair and decode every image in selected shards; fail with its key.

Usage: python -m scripts.check_sa1b_shards s3://bucket/prefix/shard.tar [...]
Add --check-masks to also decode all original-resolution COCO masks.
"""

import argparse
import time

from datasets.sa1b_webdataset import decode_masks, decode_sample, iter_encoded_samples


def check_shard(shard, endpoint=None, expected_samples=1000, check_masks=False):
    print(f"Checking {shard}", flush=True)
    started = time.monotonic()
    count = 0
    keys = set()
    samples = iter_encoded_samples(shard, endpoint)
    try:
        for sample in samples:
            try:
                if sample['__key__'] in keys:
                    raise ValueError('Duplicate sample key')
                keys.add(sample['__key__'])
                image, metadata = decode_sample(sample)
                if check_masks:
                    height, width = metadata['original_size']
                    for mask in decode_masks(metadata, (width, height)):
                        pass
                image.close()
            except Exception as error:
                error.add_note(f"Corrupt sample: {shard}::{sample['__key__']}")
                raise
            count += 1
            if count % 25 == 0:
                print(f"  {count} decoded; last key={sample['__key__']}", flush=True)
    finally:
        samples.close()
    if count != expected_samples:
        raise ValueError(f'{shard}: expected {expected_samples} samples, found {count}')
    print(f"PASS {shard}: {count} samples in {time.monotonic() - started:.1f}s", flush=True)
    return count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shards', nargs='+')
    parser.add_argument('--endpoint', default=None, help='Defaults to B2_ENDPOINT_URL')
    parser.add_argument('--expected-samples', type=int, default=1000)
    parser.add_argument('--check-masks', action='store_true')
    args = parser.parse_args()
    for shard in args.shards:
        check_shard(shard, args.endpoint, args.expected_samples, args.check_masks)


if __name__ == '__main__':
    main()
