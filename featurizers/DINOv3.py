import torch.nn as nn
from transformers import AutoModel


class DINOv3Featurizer(nn.Module):
    def __init__(self, arch, patch_size, feat_type):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type

        if arch not in {
            "dinov3_vits16", "dinov3_vits16plus", "dinov3_vitb16",
            "dinov3_vitl16", "dinov3_vith16plus", "dinov3_vit7b16",
        }:
            raise NotImplementedError(f"Unknown architecture {arch}")
        # Use saved HF credentials (or HF_TOKEN) for the gated model download.
        repo_id = f"facebook/{arch.replace('_', '-')}-pretrain-lvd1689m"
        self.model = AutoModel.from_pretrained(repo_id)
        self.dim = self.model.config.hidden_size
        if patch_size != self.model.config.patch_size:
            raise ValueError(f"Expected patch size {self.model.config.patch_size}, got {patch_size}")

    def get_cls_token(self, img):
        return self.model(pixel_values=img).last_hidden_state[:, 0]

    def forward(self, img, n=1, include_cls=False):
        h = img.shape[2] // self.patch_size
        w = img.shape[3] // self.patch_size
        tokens = self.model(pixel_values=img).last_hidden_state
        # HF returns CLS, register tokens, then normalized spatial patch tokens.
        patch_tokens = tokens[:, 1 + self.model.config.num_register_tokens:]
        return patch_tokens.reshape(-1, h, w, self.dim).permute(0, 3, 1, 2)
