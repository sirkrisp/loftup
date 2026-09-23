"""Training loader selection shared by both LoftUp stages."""

from torch.utils.data import DataLoader
from .util import get_dataset


def create_training_loaders(cfg, transform, target_transform):
    if cfg.dataset == "sa1b_webdataset":
        from .sa1b_webdataset import make_webdataset_loaders
        return make_webdataset_loaders(cfg, transform, target_transform)
    split_kwargs = dict(sample_size=cfg.sa1b_sample_size,
                        val_fraction=cfg.sa1b_val_fraction,
                        split_seed=cfg.sa1b_split_seed) if cfg.dataset == "sa1b" else {}
    common = dict(transform=transform, target_transform=target_transform,
                  include_labels=False, **split_kwargs)
    train = get_dataset(cfg.pytorch_data_dir, cfg.dataset, "train", **common)
    val = get_dataset(cfg.pytorch_data_dir, cfg.dataset, "val", **common)
    return (DataLoader(train, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers),
            DataLoader(val, 1, shuffle=False, num_workers=cfg.num_workers))
