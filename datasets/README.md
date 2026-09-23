# Preparing Datasets for Evaluation

Here is a list on how to prepare the datasets for evaluation.

## COCOStuff (coarse labels)

You can download COCO dataset, COCOStuff label maps and coarse labels following the instrcutions in [IIC](https://github.com/xu-ji/IIC/blob/master/datasets/README.txt).


## Cityscapes
You can follow instructions here in [Mask2Former](https://github.com/facebookresearch/Mask2Former/blob/main/datasets/README.md)

## DAVIS 2017

You can download the DAVIS 2017 dataset by running:

```bash
wget https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-480p.zip
```

## SA-1B training subset

The authors used **the first 100 shards**, `sa_000000.tar` through `sa_000099.tar`
([issue #23](https://github.com/andrehuang/loftup/issues/23#issuecomment-3908435195)).
Choose how many of those complete shards to download; the count is required:

```bash
# Start with one shard (sa_000000.tar).
python3 scripts/download_sa1b_1m_subset.py --num-tars 1

# Extend the same folder to the first ten shards (sa_000000.tar .. sa_000009.tar).
python3 scripts/download_sa1b_1m_subset.py --num-tars 10

# Download all 100 shards of the authors' subset only when requested.
python3 scripts/download_sa1b_1m_subset.py --num-tars 100
```

The standard-library-only script reads signed URLs from `datasets/sa-1b-links.txt`,
streams each selected tar completely, and writes flat JPEG/JSON pairs into
`datasets/sa1b`. It does not retain tar archives. `--num-tars` accepts 1 through
100, always starting at shard zero. Counts are total desired shards, not additional
shards. Image counts depend on the archives; downloads no longer stop mid-shard
at an image-count limit. The old `--num-images` and shuffle `--seed` options have
been removed.

Rerun with the same count to resume, or a larger count to extend. Completed shards
are skipped; an interrupted shard is streamed again while saved files are skipped.
Keep `.sa1b-download-state.json` with the data and run only one downloader per
output directory. Signed URLs can be refreshed without affecting resume.

**Existing downloads from the old shuffled downloader are a different subset.**
The script leaves them untouched and rejects mixing them into an ordered download.
Use a separate directory, for example:

```bash
python3 scripts/download_sa1b_1m_subset.py --num-tars 1 --output-dir datasets/official/sa1b
python train_loftup_stage1.py gpu=1x3090 model_type=dinov3splus pytorch_data_dir=datasets/official
```

The training loader caps its combined pool at `sa1b_sample_size=1000000` complete
pairs and reserves `sa1b_val_fraction=0.05` for validation using
`sa1b_split_seed=42`. A smaller download uses the available pairs. Set the sample
size to `null` to use all downloaded pairs. The holdout is a local addition.
Finish downloading before training and keep files and split settings fixed across
stages; extending the download changes split membership.

## SA-1B WebDataset shards with background Backblaze uploads

`prepare_sa1b_webdataset.py` downloads the first `--num-tars` source archives,
resizes image/mask pairs, and writes standard WebDataset tar shards. Install the
optional upload/dashboard dependencies and configure a Backblaze B2 application key:

```bash
uv sync --extra data-prep
export B2_APPLICATION_ID='your-b2-application-key-id'
export B2_APPLICATION_KEY='your-b2-application-key'
export B2_BUCKET='your-bucket'
export B2_ENDPOINT_URL='https://s3.us-west-004.backblazeb2.com'

uv run --extra data-prep python scripts/prepare_sa1b_webdataset.py \
  --num-tars 10 \
  --work-dir datasets/sa1b-webdataset \
  --prefix sa1b-896
```

Use your bucket's actual regional endpoint. The application key needs bucket
access plus permission to read/check and write objects. Uploads use Backblaze's
[S3-compatible boto3 interface](https://www.backblaze.com/docs/en/cloud-storage-use-the-aws-sdk-for-python-with-backblaze-b2).
No WebDataset package is needed to write the
[standard tar format](https://github.com/webdataset/webdataset/blob/main/README.md).

`B2_APPLICATION_ID` is the application's **keyID**, not the bucket ID. Set both
B2 credential variables together; they take precedence over AWS credentials.
If neither is set, the script uses boto3's normal credential lookup, including
`AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`.

- **Resize:** if the shorter side exceeds 896, shrink it to 896 while preserving
  aspect ratio. Smaller images are unchanged in dimensions. There is no crop;
  the longer side can exceed 896. Images use bilinear interpolation and JPEG
  quality 95. `--size` and `--jpeg-quality` override these defaults. Images are
  re-encoded even when their dimensions do not change. Masks are **not**
  resized: training (`target_transform` in `train_loftup_stage*.py`) already
  resizes every mask with nearest-neighbor to its actual crop size on every
  epoch regardless of stored resolution, so resizing masks here at prep time
  would be pure duplicate work with no effect on the final training input.
- **Sample format:** consecutive `sa_<id>.jpg` and `sa_<id>.json` members. JSON
  contains `image`, `original_size` (`[height, width]`), and `annotations`.
  Each annotation has its original `id`, `segmentation` RLE, `area`, and
  `[x, y, width, height]` `bbox`, passed through unchanged from the source at
  original resolution (matching `original_size`, not the resized image).
  Overlapping masks remain independent. Original annotation quality scores,
  prompts, and crop metadata are omitted.
- **Shard size:** each full shard contains **1,000 images**. Override this with
  `--samples-per-shard`. Shard byte sizes vary with the encoded image/mask data;
  the measured final average is printed.
- **Equal counts:** by default the final incomplete group is saved locally as
  `shards/remainder.tar.pending` and is **not uploaded**. Extending `--num-tars`
  includes those samples in the next full shard (the source may be downloaded
  again). To include every image immediately, select `--final-shard keep` from
  the start; the final uploaded shard may then contain fewer images.
- **Streaming:** read JPEG/JSON members directly from each incoming tar into
  memory, resize complete pairs, and write them into output shards. Downloading,
  conversion/writing, and uploads run concurrently. No source images or
  annotations are extracted to disk.
- **Dashboard:** interactive terminals show a Rich dashboard with the destination
  bucket/prefix, source download progress, current shard fill percentage and image
  count, active upload names and percentages, queued shard count, and the five
  most recent completed uploads. Use `--no-dashboard` for plain logs; redirected
  output also uses plain logs automatically.
- **Background uploads:** two workers upload finalized shards while preprocessing
  continues. A bounded queue of four uploads applies backpressure when uploads
  fall behind. Configure with `--upload-workers` and `--max-pending-uploads`.
  Uploads retry, then verify remote length and SHA-256 metadata before deleting
  the local tar. This verifies stored metadata, not a remote re-download/hash.
  Use `--keep-local` to retain uploaded tars. The process waits for outstanding
  uploads before exiting and fails if any upload fails.
- **Resume:** rerun the same command. `state.json` checkpoints finalized shards;
  upload receipts track completed uploads. Unfinished shards are regenerated,
  and failed uploads retain their local tar. Resuming replays the current source
  stream and skips committed pairs; no raw-file cache is needed. Older numeric-order
  checkpoints require one initial scan of member names to identify committed IDs.
  Increasing `--num-tars` extends the
  subset with the same image count per shard. Other generation/destination
  settings must stay the same. Signed links can be refreshed. Use a fresh B2
  prefix for a different dataset; existing objects with different checksums
  are rejected. Keep the work directory and run only one process against it.
- **Disk/memory usage:** disk holds the current output shard and the bounded queue
  of finalized shards (plus any retained remainder or `--keep-local` files).
  `--max-pending-pairs` limits the queue of raw pairs awaiting conversion (default
  32). `--max-buffer-mb` limits the additional raw-member pairing buffer (default
  256 decimal MB). If matching members are too far apart to fit, the run stops
  with a message to increase that limit, rather than extracting files to disk.
  Decoding/resizing also requires memory for the current image and masks.

For a local conversion without Backblaze, add `--local-only`. A later run without
that flag uploads existing shards, provided the destination settings match.
Use the streaming loader below to train directly from these shards.

## Train from WebDataset shards

Both stages default to `s3://sa1b-webdataset/sa1b-896/` at
`https://s3.eu-central-003.backblazeb2.com`. Run `uv sync`, export
`B2_APPLICATION_ID`/`B2_APPLICATION_KEY` with **list/read** access, and authenticate
HF via `hf auth login` or `HF_TOKEN` with a write-capable token.

```bash
uv run python train_loftup_stage1.py gpu=1x3090 model_type=dinov3splus
uv run python train_loftup_stage2.py model_type=dinov3splus num_gpus=1 pretrained_upsampler=/path/to/stage1.ckpt
# Local alternative:
uv run python train_loftup_stage1.py webdataset.shards=/path/to/completed/shards
```

`webdataset.shards` accepts a directory/glob, URL list, single HTTPS/S3 tar, or
S3 prefix. `SA1B_WEBDATASET_SHARDS` overrides the source and `B2_ENDPOINT_URL`
overrides the endpoint. Keep the shard snapshot fixed across stages; only
completed `.tar` files are selected.

Sorted shard order and member position select the first **1,000,000 training
images**, then **5,000 validation images**, independently of GPU/worker count.
`webdataset.samples_per_shard=1000` matches the preparation state (1,174 output
shards from 105 source tars); change it if repacking. Insufficient shards and
truncated selections fail. Each training image is used once per epoch, without
cycling. `webdataset.train_samples` must divide evenly by `batch_size * num_gpus`;
`webdataset.val_samples` must divide evenly by `num_gpus`. Gradient accumulation
does not change the image count. Four-GPU/batch-two training runs 125,000 batches
per rank; one-GPU/batch-two training runs 500,000.

Eight workers per GPU stream/download and decode concurrently. Contiguous,
batch-aligned worker/rank ranges prevent duplicate samples; boundary shards may
be read by multiple consumers. Training shuffles shard ranges and uses a
compressed 32-sample shuffle buffer. Tune `num_workers` for bandwidth/CPU/RAM;
`webdataset.prefetch_factor=1` bounds decoded batch buffering, especially for
896-pixel masks. There is no disk shard cache or mid-shard network resume;
network errors stop training. Checkpoint resume does not restore the stream cursor.

Matching deterministic image/mask transforms produce `img: [B,3,H,W]` and
`label: [B,150,H,W]`. Unused mask slots contain `-1`, overlaps are preserved, and
`webdataset.max_masks` controls the cap.

Periodic, epoch and final checkpoints upload automatically to private
`Krispin/loftup` under stage/run-specific directories. Uploads are synchronous,
retry three times, and retain the local checkpoint on failure. Repository
access/privacy is checked before fitting. Set `hf.enabled=false` to disable.

For older experiments, `webdataset.train_samples=null` restores the cycling
stream with per-GPU `train_batches`/`val_batches` budgets and seeded hash split.
Default finite mode ignores those budgets and `sa1b_sample_size`,
`sa1b_val_fraction`, and `sa1b_split_seed` for membership.

### Explore images and masks

Open [`notebooks/explore_sa1b_webdataset.ipynb`](../notebooks/explore_sa1b_webdataset.ipynb):

```bash
uv sync --extra data-prep --extra notebooks
uv run --extra data-prep --extra notebooks jupyter lab notebooks/explore_sa1b_webdataset.ipynb
```

The notebook defaults to the `sa1b-webdataset` Backblaze bucket. Set `SOURCE` to a
local directory if preferred. It offers image selection, colored mask overlays,
an individual mask selector, annotation counts, and a preview of training tensors.
Only a small preview is held in memory. Launch Jupyter from a shell with the B2
credentials exported; do not put keys in notebook cells.
