import torch
import torch.nn as nn


class DINOv3Featurizer(nn.Module):
    def __init__(self, arch, patch_size, feat_type):
        super().__init__()
        self.arch = arch
        self.patch_size = patch_size
        self.feat_type = feat_type

        self.model = torch.hub.load("facebookresearch/dinov3", arch)
        if "vits" in arch:
            self.dim = 384
        elif "vitb" in arch:
            self.dim = 768
        elif "vitl" in arch:
            self.dim = 1024
        elif "vith" in arch:
            self.dim = 1280
        elif "vit7b" in arch:
            self.dim = 4096
        else:
            raise NotImplementedError(f"Unknown architecture {arch}")

    def get_cls_token(self, img):
        return self.model.forward(img)

    def forward(self, img, n=1, include_cls=False):
        h = img.shape[2] // self.patch_size
        w = img.shape[3] // self.patch_size
        patch_tokens = self.model.forward_features(img)["x_norm_patchtokens"]
        return patch_tokens.reshape(-1, h, w, self.dim).permute(0, 3, 1, 2)
