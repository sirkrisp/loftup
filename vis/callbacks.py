"""Shared experiment logging and feature previews for both training stages."""
from pathlib import Path
from itertools import islice

import numpy as np
from PIL import Image, ImageDraw, ImageOps
import torch
import torch.nn.functional as F
from torchvision import transforms as T
from pytorch_lightning import Callback
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from pytorch_lightning.utilities.seed import isolate_rng
from omegaconf import OmegaConf

from upsamplers import norm, unnorm
from .progress import ResumeProgressBar


def feature_panel(image, low, high):
    """RGB / native features / bilinear baseline / learned features, in one PCA basis."""
    low = low.detach().float().cpu()
    high = high.detach().float().cpu()
    samples = low[0].flatten(1).T
    mean = samples.mean(0)
    _, _, axes = torch.linalg.svd(samples - mean, full_matrices=False)
    basis = axes[:3].T
    if basis.shape[1] < 3:
        basis = F.pad(basis, (0, 3 - basis.shape[1]))
    projected = (samples - mean) @ basis
    minimum = projected.amin(0)
    scale = (projected.amax(0) - minimum).clamp_min(1e-6)

    def color(features):
        _, channels, height, width = features.shape
        rgb = ((features[0].reshape(channels, -1).T - mean) @ basis - minimum) / scale
        return rgb.clamp(0, 1).T.reshape(1, 3, height, width)

    height, width = image.shape[-2:]
    low_rgb = color(low)
    columns = [
        image.detach().float().cpu().clamp(0, 1),
        F.interpolate(low_rgb, (height, width), mode="nearest"),
        F.interpolate(low_rgb, (height, width), mode="bilinear", align_corners=False),
        F.interpolate(color(high), (height, width), mode="bilinear", align_corners=False),
    ]
    panel = Image.new("RGB", (width * 4, height + 24), "white")
    draw = ImageDraw.Draw(panel)
    for index, (label, tensor) in enumerate(zip(
        ("Input", "Low-res PCA", "Bilinear PCA", "LoftUp PCA"), columns
    )):
        array = (tensor[0].permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
        panel.paste(Image.fromarray(array), (index * width, 24))
        draw.text((index * width + 4, 5), label, fill="black")
    return panel


class FeatureVisualization(Callback):
    """Log fixed images (or cached validation images) only on global rank zero."""
    def __init__(self, image_dir="vis/images", max_images=4, every_n_steps=500,
                 input_size=224, output_size=224, stage="stage1"):
        if min(max_images, input_size, output_size) < 1 or every_n_steps < 0:
            raise ValueError("Visualization sizes/count must be positive; interval must be nonnegative")
        self.image_dir = Path(image_dir)
        self.max_images = max_images
        self.every_n_steps = every_n_steps
        self.input_size = input_size
        self.output_size = output_size
        self.stage = stage
        self.examples = []
        self._last_step = None
        self._fixed_images = False

    def setup(self, trainer, pl_module, stage):
        if not trainer.is_global_zero or self.examples:
            return
        paths = sorted(p for p in self.image_dir.glob("**/*")
                       if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})
        transform = T.Compose([T.Resize(self.output_size), T.CenterCrop(self.output_size), T.ToTensor(), norm])
        for path in paths[:self.max_images]:
            with Image.open(path) as image:
                self.examples.append((path.stem, transform(ImageOps.exif_transpose(image).convert("RGB")).unsqueeze(0)))
        self._fixed_images = bool(self.examples)

    def on_train_start(self, trainer, pl_module):
        # Lightning skips sanity validation when resuming. The preview cache is
        # process-local, so older checkpoints also need fresh validation images.
        if not trainer.is_global_zero or self.examples:
            return
        loaders = trainer.val_dataloaders
        if loaders is None:
            return
        if not isinstance(loaders, (list, tuple)):
            loaders = [loaders]
        # Match sanity validation's RNG isolation: drawing preview batches must
        # not change training's random augmentations or dropout sequence.
        with isolate_rng():
            for loader in loaders:
                for batch_idx, batch in enumerate(islice(loader, self.max_images)):
                    self.on_validation_batch_end(trainer, pl_module, None, batch, batch_idx)
                    if len(self.examples) >= self.max_images:
                        return

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if not trainer.is_global_zero or self._fixed_images or len(self.examples) >= self.max_images:
            return
        images = batch["img"] if isinstance(batch, dict) else batch[0]
        paths = batch.get("img_path", []) if isinstance(batch, dict) else []
        for index, image in enumerate(images):
            name = Path(paths[index]).stem if index < len(paths) else f"validation_{batch_idx}_{index}"
            if name not in {entry[0] for entry in self.examples}:
                self.examples.append((name, image.detach().cpu().unsqueeze(0)))
            if len(self.examples) >= self.max_images:
                break

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = trainer.global_step
        if self.every_n_steps and step > 0 and step % self.every_n_steps == 0:
            self._log(trainer, pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        self._log(trainer, pl_module)

    @torch.no_grad()
    def _log(self, trainer, pl_module):
        if not trainer.is_global_zero or trainer.sanity_checking or not self.examples:
            return
        if self._last_step == trainer.global_step:
            return
        # Preserve all nested train/eval flags, including the frozen backbone.
        modules = list(pl_module.model.modules()) + list(pl_module.upsampler.modules())
        modes = [(module, module.training) for module in modules]
        panels, captions = [], []
        try:
            pl_module.model.eval()
            pl_module.upsampler.eval()
            for name, image in self.examples:
                image = image.to(pl_module.device)
                guidance = F.interpolate(image, (self.output_size, self.output_size), mode="bilinear", align_corners=False)
                backbone_input = F.interpolate(image, (self.input_size, self.input_size), mode="bilinear", align_corners=False)
                low = pl_module.model(backbone_input)
                high = pl_module.upsampler(low, guidance)
                panels.append(feature_panel(unnorm(guidance), low, high))
                captions.append(name)
        finally:
            for module, training in modes:
                module.training = training
        # TensorBoard is always the primary logger, giving each run a unique directory.
        # Trainer.log_dir broadcasts under DDP, so don't call it in a rank-zero hook.
        tensorboard = next(logger for logger in trainer.loggers if isinstance(logger, TensorBoardLogger))
        destination = Path(tensorboard.log_dir) / "features" / self.stage
        destination.mkdir(parents=True, exist_ok=True)
        for index, panel in enumerate(panels):
            panel.save(destination / f"step_{trainer.global_step:08d}_{index}.png")
        for logger in trainer.loggers:
            if isinstance(logger, WandbLogger):
                logger.log_image(key=f"{self.stage}/features", images=panels,
                                 caption=captions, step=trainer.global_step)
            elif isinstance(logger, TensorBoardLogger):
                for index, panel in enumerate(panels):
                    logger.experiment.add_image(f"{self.stage}/features/{index}", np.array(panel),
                                                trainer.global_step, dataformats="HWC")
                logger.experiment.flush()
        self._last_step = trainer.global_step


def create_logging(cfg, log_dir, name, stage):
    """Keep TensorBoard, optionally add W&B, and share callback settings."""
    loggers = [TensorBoardLogger(log_dir, default_hp_metric=False)]
    if cfg.wandb.enabled:
        if cfg.wandb.mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be online, offline, or disabled")
        loggers.append(WandbLogger(
            project=cfg.wandb.project, entity=cfg.wandb.entity,
            name=cfg.wandb.name or name, group=cfg.wandb.group,
            save_dir=log_dir, mode=cfg.wandb.mode, log_model=False,
            config=OmegaConf.to_container(cfg, resolve=True), tags=[stage, cfg.model_type],
        ))
    callbacks = [ResumeProgressBar()]
    if cfg.vis.enabled:
        callbacks.append(FeatureVisualization(
            image_dir=cfg.vis.image_dir, max_images=cfg.vis.max_images,
            every_n_steps=cfg.vis.every_n_steps, input_size=cfg.vis.input_size,
            output_size=cfg.vis.output_size, stage=stage,
        ))
    return loggers, callbacks
