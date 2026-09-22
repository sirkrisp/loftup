# [LoftUp: A Coordinate-Based Feature Upsampler for Vision Foundation Models](https://arxiv.org/abs/2504.14032)

ICCV2025 (oral)

[[Arxiv]](https://arxiv.org/abs/2504.14032) [[Project Page]](https://andrehuang.github.io/loftup-site/)

[Haiwen Huang](https://andrehuang.github.io/), Anpei Chen, Volodymyr Havrylov, Andreas Geiger, Dan Zhang

![Teaser](figures/loftup-teaser.png)

**TL;DR:** LoftUp achieves the strongest feature upsampling performance at a comparable speed to bilinear upsampling.

![bike-packing](examples/bike-packing.gif)
![camel](examples/clip-camel.gif)
![horsejump](examples/siglip2-horsejump.gif)


## Contents
- [Install](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#install)
- [Inference with pretrained upsamplers](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#inference-with-pretrained-upsamplers)
- [Evaluation on downstream tasks](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#inference-with-pretrained-upsamplers)
- [Training LoftUp upsamplers](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#inference-with-pretrained-upsamplers)
- [Citation](https://github.com/andrehuang/loftup/tree/main?tab=readme-ov-file#inference-with-pretrained-upsamplers)

## Install

Install the Python dependencies with [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv sync
```

This creates a local `.venv` using Python 3.11 and installs the dependencies for
inference, training, and evaluation, including the bundled DAVIS evaluator.
The PyTorch stack is pinned to 2.5.1 with CUDA 12.1 wheels on Linux and Windows,
following [uv's PyTorch integration](https://docs.astral.sh/uv/guides/integration/pytorch/).
An NVIDIA driver is required for GPU execution. On macOS, PyTorch comes from PyPI.
The `uv.lock` file records exact dependency versions for reproducible installs.

Run scripts from the repository root using `uv run`, for example:

```bash
uv run python example_usage.py
```

Alternatively, activate the environment with `source .venv/bin/activate` before
running the Python commands below. Model weights and datasets are downloaded
separately. The legacy YAML files also list the system `ffmpeg` executable;
install it separately if needed for external video workflows (the Python scripts
here use OpenCV for video output).

The original Conda environments remain available:

```bash
conda env create -f environment_cuda11.yaml
```

or 

```bash
conda env create -f environment.yaml
```


## Inference with pretrained upsamplers

All pre-trained upsamplers are available on 🤗 here: https://huggingface.co/models?search=loftup.

We provide example code for using LoftUp in [example_usage.py](example_usage.py). Currently we provide:


|Backbone Name          | Featurizer Class              | HF hub                                  | Torch Hub Repo | Torch Hub Name |
|-------------------| ---|------------------------------------------------|------|-----|
| DINOv2 S/14     | [dinov2](featurizers/DINOv2.py)     | [haiwen/loftup-dinov2s](https://huggingface.co/haiwen/loftup-dinov2s)   | andrehuang/loftup | loftup_dinov2s|
| DINOv2 S/14 + Reg | [dinov2s_reg](featurizers/DINOv2.py)     | [haiwen/loftup-dinov2s_reg](https://huggingface.co/haiwen/loftup-dinov2s_reg)| andrehuang/loftup | loftup_dinov2s_reg|
| DINOv3 S+/16 | [dinov3splus](featurizers/DINOv3.py) | - | - | - |
| DINOv2 B/14 | [dinov2b](featurizers/DINOv2.py) | [haiwen/loftup-dinov2b](https://huggingface.co/haiwen/loftup-dinov2b) | andrehuang/loftup | loftup_dinov2b|
| DINOv2 B/14 + Reg | [dinov2b_reg](featurizers/DINOv2.py)     | [haiwen/loftup-dinov2b_reg](https://huggingface.co/haiwen/loftup-dinov2b_reg)|andrehuang/loftup | loftup_dinov2b_reg|
| CLIP ViT B/16 | [clip](featurizers/CLIP.py) |[haiwen/loftup-clip](https://huggingface.co/haiwen/loftup-clip) | andrehuang/loftup | loftup_clip|
|SigLIP ViT B/16 | [siglip](featurizers/SigLIP.py) | [haiwen/loftup-siglip](https://huggingface.co/haiwen/loftup-siglip)| andrehuang/loftup | loftup_siglip|
|SigLIP2 ViT B/16 | [siglip2](featurizers/SigLIP.py) | [haiwen/loftup-siglip2](https://huggingface.co/haiwen/loftup-siglip2)| andrehuang/loftup | loftup_siglip2|

To use torch hub checkpoints, simply run 
```python
upsampler = torch.hub.load('andrehuang/loftup', model_torch_hub_name, pretrained=True)
```
For example, ```upsampler = torch.hub.load('andrehuang/loftup', loftup_dinov2s, pretrained=True)```.

The upsampler class is defined at [UpsamplerwithChannelNorm](https://github.com/andrehuang/loftup/blob/7ce8a97e720465819a2a6b24a7c24c192da394b6/upsamplers/upsamplers.py#L109).

## Evaluation on Downstream Tasks

### Dataset Preparation

See [Preparing Datasets for Evaluation](datasets/README.md).

### Semantic Segmentation
For semantic segmentation, our implementation is adapted from [FeatUp](https://github.com/mhamilton723/FeatUp). You can use [eval_seg.py](eval_seg.py) by running:

```bash
python eval_seg.py  ++upsampler_path=/path/to/your/upsampler
```

You can also configure other hyper-parameters such as output_dir and dataset directory. The config file is [configs/eval_seg.yaml](configs/eval_seg.yaml). 

### Video Object Segmentation
For video object segmentation on DAVIS, our code is modified from the implementation in [LiFT](https://github.com/saksham-s/lift). Specifically, we first extract segmentaiton results by  running:

```bash
    python eval_davis.py --dataroot your_davis_data_dir --model_type "dinov2" --output_dir your_output_dir --imsize 224 --upsampler_path=your_upsampler_path
```

Then run the following to get evaluation results:

```bash
python davis2017-evaluation/evaluation_method.py --davis_path /your_davis_data_dir --task semi-supervised --results_path your_output_dir/davis_vidseg_224 --imsize 224
```

### Others
For interactive segmentation, please check out [iSegProbe](https://github.com/havrylovv/iSegProbe).

For open-vocabulary segmentation, please check out [ProxyCLIP](https://github.com/mc-lan/ProxyCLIP).

For depth and normal estimation, please check out [Probe3D](https://github.com/mbanani/probe3d).


## Training LoftUp Upsamplers

For DINOv3, the featurizer loads weights from Meta's gated Hugging Face
repositories using the [Transformers DINOv3 loader](https://huggingface.co/docs/transformers/model_doc/dinov3).
Log in once with an account that has model access:

```bash
uv run hf auth login
```

An existing Hugging Face login or `HF_TOKEN` is used automatically. Hugging Face
access does not authenticate the default `dl.fbaipublicfiles.com` Torch Hub URL.

This repository contains training scripts for training LoftUp upsamplers. The training is done in two stages:

SA-1B images are split into disjoint training (95%) and validation (5%) sets
using seed 42. Both training stages use the same split settings. Validation runs
at the end of each epoch over all held-out images and logs
`val/reconstruction_mse`: the MSE between backbone features and upsampled features
resized back to the backbone grid. This is a reconstruction diagnostic, not a
downstream segmentation score. Stage 1 also visualizes the first validation image.

Configure the split with `sa1b_val_fraction=0.05`, `sa1b_split_seed=42`, and
`sa1b_sample_size=1000000`. The sample limit applies to the combined pool before
splitting; use `sa1b_sample_size=null` to include all complete image/JSON pairs.
Membership is reproducible for an unchanged dataset and identical settings;
finish downloading before training and keep these settings fixed across stages.

### Stage 1: Basic Feature Upsampling

Stage 1 training (`train_loftup_stage1.py`) trains upsamplers to convert low-resolution features to high-resolution features using reconstruction loss.

Activate the uv environment (`source .venv/bin/activate`), then choose a GPU
preset and backbone. `uv run python` also works without activation:

```bash
python train_loftup_stage1.py gpu="1x3090" model_type="dinov3splus"
python train_loftup_stage1.py gpu="1xh100" model_type="dinov3base"
python train_loftup_stage1.py gpu="4x3090" model_type="dinov3splus"
python train_loftup_stage1.py gpu="4xh100" model_type="dinov3base"
python train_loftup_stage1.py gpu="4x4090" model_type="dinov3splus"
```

| GPU preset | GPUs | Batch per GPU | Accumulation steps | Effective global batch |
|---|---:|---:|---:|---:|
| `1x3090` | 1 | 1 | 8 | 8 |
| `1xh100` | 1 | 2 | 4 | 8 |
| `4x3090` | 4 | 1 | 2 | 8 |
| `4xh100` | 4 | 2 | 1 | 8 |
| `4x4090` | 4 | 1 | 2 | 8 |

These are conservative execution presets, not measured optimal batch sizes.
They select the number of visible GPUs; use `CUDA_VISIBLE_DEVICES` to choose
specific devices. Four-GPU runs use DDP. All presets default to float32.
The final incomplete accumulation window is normalized by its actual number
of microbatches; incomplete per-device batches are dropped. Accumulation does
not reproduce full-batch BatchNorm statistics.

Learning rates follow the [author's clarification in issue #25](https://github.com/andrehuang/loftup/issues/25#issuecomment-5775180056):
**Stage 1 `1e-4`, Stage 2 `1e-3`**, as in the released configs.
Other Stage 1 defaults are AdamW, effective batch 8, one epoch,
two cross-attention blocks, and mask refinement `sam_mask_alpha=0.8`.
The authors identify the [first 100 SA-1B tar shards](https://github.com/andrehuang/loftup/issues/23#issuecomment-3908435195)
as the training subset. The downloader accepts `--num-tars N` to start with fewer
shards in that same order. The dataset pool is capped at 1M image/JSON pairs; the local 5% holdout leaves
950k training images when the full pool is available. Smaller local datasets
use the available pairs. This holdout and the DINOv3 backbones are local
extensions, not the paper's exact experimental protocol.

Settings not specified in that appendix retain the released training recipe:
224-pixel input/output, four jitters, random projection dimension 64,
`tv_weight=0.001` from the authors' example command, and reconstruction clamping.
AdamW weight decay defaults to `0.01`; the paper does not specify it.
The remaining released training implementation is not claimed to reproduce all
paper details. Stage 2's optimizer/training defaults are unchanged by these
Stage 1 presets; only its dataset pool limit is aligned to preserve split membership.

You can still override individual settings without `++`, for example
`batch_size=1 accumulation_steps=8`, or inspect the resolved configuration with
`--cfg job --resolve`. Keep `batch_size * num_gpus * accumulation_steps = 8`
to retain the target batch size.

### Stage 2: High-Resolution Supervision

Stage 2 training (`train_loftup_stage2.py`) fine-tunes the Stage 1 upsampler with high-resolution supervision for improved quality.

**Example training command:**
```bash
python train_loftup_stage2.py ++dataset="sa1b" ++epochs=1 ++hr_res=896 ++batch_size=2 ++consistency_method="bilinear" ++model_type="dinov3splus" ++num_gpus=4 ++affinity_loss=True ++pytorch_data_dir='datasets' ++pretrained_upsampler="path/to/stage1_checkpoint.ckpt" ++upsampler_type="loftup" ++sam_mask_hr_alpha=0.5 ++sam_mask_reg=0.0 ++lr=1e-3 ++use_featup=False ++aug_size ++n_jitters=2
```

### W&B logging and feature visualization

Both stages support W&B alongside the existing TensorBoard logger. Authenticate
once, then enable it for a run:

```bash
uv run wandb login
uv run python train_loftup_stage1.py gpu=1x3090 model_type=dinov3splus wandb.enabled=true
uv run python train_loftup_stage2.py model_type=dinov3splus num_gpus=1 pretrained_upsampler=/path/to/stage1.ckpt wandb.enabled=true
```

Set `wandb.project`, `wandb.entity`, `wandb.name`, or `wandb.group` to customize the
run. `wandb.mode=offline` records locally without uploading. W&B records existing
training losses, validation reconstruction MSE, and the resolved configuration;
model checkpoint uploads are disabled.

The shared callback lives in `vis/callbacks.py`. It shows **input RGB, low-resolution
feature PCA, bilinear feature PCA, and learned LoftUp feature PCA**. All three
feature panels use a PCA basis and color scale fitted on the same low-resolution
features, so their colors are comparable. These are feature visualizations, not
segmentation predictions.

By default it caches up to four validation examples. Optionally put fixed images
in `vis/images/` or set `vis.image_dir=/path/to/images`. Images are resized and
center-cropped to the display size. Previews run every 500 optimizer steps and
at validation end, only on global rank zero; sanity validation is not logged.
PNG copies are saved under the TensorBoard run directory in `features/stage1/`
or `features/stage2/`. They appear under `stage1/features` or `stage2/features`
in W&B and TensorBoard.

Use `vis.every_n_steps=100`, `vis.max_images=2`, or `vis.output_size=448` to change
frequency, image count, or output resolution. Larger outputs require more GPU
memory. `vis.every_n_steps=0` limits previews to validation end;
`vis.enabled=false` disables them. The callback replaces Stage 1's old embedded
TensorBoard visualization, and supports Stage 2 with the same display.

### Configuration

Both training scripts use Hydra for configuration management. Configuration files are located in `configs/`:
- `configs/train_loftup_stage1.yaml` - Stage 1 configuration
- `configs/train_loftup_stage2.yaml` - Stage 2 configuration

**Key configuration parameters:**
- `model_type`: Feature extractor type (e.g., "dinov2", "dinov3splus", "clip")
- `upsampler_type`: Type of upsampler to train (e.g., "loftup")
- `batch_size`: Training batch size
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `load_size`: Input image size for feature extraction
- `upsample_size`: Target size for upsampled features
- `n_jitters`: Number of jittering augmentations per training step
- `tv_weight`: Weight for total variation loss
- `sam_mask_alpha`: Weight for SAM mask adjustment (Stage 1)
- `sam_mask_hr_alpha`: Weight for SAM mask adjustment (Stage 2)


For more details, see the configuration files in `configs/` and the training scripts themselves.

## Citation
If you find our work helpful, please cite:

```
@misc{huang2025loftuplearningcoordinatebasedfeature,
      title={LoftUp: Learning a Coordinate-Based Feature Upsampler for Vision Foundation Models}, 
      author={Haiwen Huang and Anpei Chen and Volodymyr Havrylov and Andreas Geiger and Dan Zhang},
      year={2025},
      eprint={2504.14032},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.14032}, 
}
```
