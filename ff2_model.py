# -*- coding: utf-8 -*-
"""FF2 模型定义(自包含,只依赖 torch / timm)

特征 = concat(CLS token, GeM(patch tokens))  → 2048 维
这个 2048 维向量就是对外提供的 embedding。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

BACKBONES = {
    "dinov3l": ("vit_large_patch16_dinov3.lvd1689m", 512),
    "dinov2l": ("vit_large_patch14_dinov2.lvd142m", 518),
}


def build_backbone(backbone="dinov3l", n_classes=51, img_size=592, pretrained=False):
    """返回 (model, mean, std, size)

    pretrained=False 时不会联网下载骨干 —— 从 .pth 加载完整权重时应该用 False,
    否则每次运行都会去 HuggingFace 拉 1.2GB。
    """
    import timm
    timm_id, default_size = BACKBONES[backbone]
    size = img_size or default_size
    try:
        model = timm.create_model(timm_id, pretrained=pretrained, num_classes=n_classes,
                                  dynamic_img_size=True)
    except TypeError:
        model = timm.create_model(timm_id, pretrained=pretrained, num_classes=n_classes)
    cfg = timm.data.resolve_data_config({}, model=model, pretrained=pretrained)
    return model, cfg["mean"], cfg["std"], size


class DualHeadNet(nn.Module):
    """CLS 与 GeM(patch) 拼接后接一个线性头"""

    def __init__(self, base, n_classes, pool="gem"):
        super().__init__()
        self.base = base
        self.pool = pool
        self.num_prefix = int(getattr(base, "num_prefix_tokens", 1) or 1)
        D = int(base.num_features)
        base.head = nn.Identity()
        self._tok = None
        base.norm.register_forward_hook(self._hook)
        self.cls_head = nn.Linear(2 * D, n_classes)
        if pool == "gem":
            self.gem_p = nn.Parameter(torch.tensor(1.0))

    def _hook(self, m, inp, out):
        self._tok = out

    def forward(self, x):
        _ = self.base(x)
        tok = self._tok
        cls = tok[:, 0]
        pat = tok[:, self.num_prefix:]
        if self.pool == "gem":
            p = F.softplus(self.gem_p) + 1e-3
            patch = pat.clamp_min(1e-6).pow(p).mean(dim=1).pow(1.0 / p)
        else:
            patch = pat.mean(dim=1)
        feat = torch.cat([cls, patch], dim=1)
        self.last_feat = feat            # 供取 embedding
        return self.cls_head(feat)
