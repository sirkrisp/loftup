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
