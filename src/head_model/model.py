from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)


@dataclass(frozen=True)
class ShapeBasis:
    mean: torch.Tensor
    components: torch.Tensor
    coeff_std: torch.Tensor

    @property
    def vertex_count(self) -> int:
        return int(self.mean.shape[0])

    @property
    def component_count(self) -> int:
        return int(self.components.shape[0])


def load_shape_basis(path: str | Path) -> ShapeBasis:
    data = np.load(path)
    mean = torch.from_numpy(np.asarray(data["mean"], dtype=np.float32))
    components = torch.from_numpy(np.asarray(data["components"], dtype=np.float32))
    coeff_std = torch.from_numpy(np.asarray(data["coeff_std"], dtype=np.float32))
    if mean.ndim != 2 or mean.shape[-1] != 3:
        raise ValueError(f"shape basis mean must be [V,3], got {mean.shape}")
    if components.ndim != 3 or components.shape[1:] != mean.shape:
        raise ValueError(
            f"shape basis components must be [K,V,3] matching mean, got {components.shape}"
        )
    if coeff_std.shape != (components.shape[0],):
        raise ValueError("coeff_std must contain one scale per PCA component")
    return ShapeBasis(mean=mean, components=components, coeff_std=coeff_std)


class DenseEvidenceHead(nn.Module):
    """Cheap dense geometry heads over the final MobileNet feature map."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = 128
        self.shared = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, groups=hidden, bias=False),
            nn.BatchNorm2d(hidden),
            nn.Hardswish(inplace=True),
        )
        self.normal = nn.Conv2d(hidden, 3, 1)
        self.depth = nn.Conv2d(hidden, 1, 1)
        self.mask = nn.Conv2d(hidden, 1, 1)
        self.confidence = nn.Conv2d(hidden, 1, 1)

    def forward(self, x: torch.Tensor, output_size: tuple[int, int]) -> dict[str, torch.Tensor]:
        x = self.shared(x)
        normal = F.interpolate(
            self.normal(x), size=output_size, mode="bilinear", align_corners=False
        )
        normal = F.normalize(normal, dim=1, eps=1e-6)
        depth = F.interpolate(
            self.depth(x), size=output_size, mode="bilinear", align_corners=False
        )
        depth = F.softplus(depth)
        mask_logits = F.interpolate(
            self.mask(x), size=output_size, mode="bilinear", align_corners=False
        )
        confidence_logits = F.interpolate(
            self.confidence(x), size=output_size, mode="bilinear", align_corners=False
        )
        return {
            "normal": normal,
            "depth": depth,
            "mask_logits": mask_logits,
            "confidence_logits": confidence_logits,
        }


class HeadScanLite(nn.Module):
    """Variable-view head reconstructor with a registered PCA mesh prior."""

    def __init__(
        self,
        basis: ShapeBasis,
        *,
        token_dim: int = 256,
        transformer_layers: int = 3,
        transformer_heads: int = 4,
        pretrained_backbone: bool = False,
    ) -> None:
        super().__init__()
        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained_backbone else None
        mobile = mobilenet_v3_small(weights=weights)
        self.backbone = mobile.features
        feature_channels = 576

        self.dense = DenseEvidenceHead(feature_channels)
        self.token_projection = nn.Sequential(
            nn.Linear(feature_channels, token_dim),
            nn.LayerNorm(token_dim),
            nn.GELU(),
        )
        self.angle_projection = nn.Sequential(
            nn.Linear(4, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, token_dim),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim,
            nhead=transformer_heads,
            dim_feedforward=token_dim * 3,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.fusion = nn.TransformerEncoder(
            layer,
            num_layers=transformer_layers,
            enable_nested_tensor=False,
        )
        self.fused_norm = nn.LayerNorm(token_dim)
        self.coeff_head = nn.Sequential(
            nn.Linear(token_dim, token_dim),
            nn.GELU(),
            nn.Linear(token_dim, basis.component_count),
        )
        self.view_quality = nn.Sequential(
            nn.Linear(token_dim, token_dim // 2),
            nn.GELU(),
            nn.Linear(token_dim // 2, 1),
        )

        self.register_buffer("shape_mean", basis.mean.clone(), persistent=True)
        self.register_buffer("shape_components", basis.components.clone(), persistent=True)
        self.register_buffer("shape_coeff_std", basis.coeff_std.clone(), persistent=True)

    @property
    def component_count(self) -> int:
        return int(self.shape_components.shape[0])

    @property
    def vertex_count(self) -> int:
        return int(self.shape_mean.shape[0])

    def reconstruct_vertices(self, coeff_norm: torch.Tensor) -> torch.Tensor:
        coeff = coeff_norm * self.shape_coeff_std.unsqueeze(0)
        delta = torch.einsum("bk,kvj->bvj", coeff, self.shape_components)
        return self.shape_mean.unsqueeze(0) + delta

    @staticmethod
    def _angle_features(view_angles: torch.Tensor) -> torch.Tensor:
        radians = torch.deg2rad(view_angles)
        yaw = radians[..., 0]
        pitch = radians[..., 1]
        return torch.stack(
            [torch.sin(yaw), torch.cos(yaw), torch.sin(pitch), torch.cos(pitch)],
            dim=-1,
        )

    def _tokens_from_features(
        self,
        features: torch.Tensor,
        *,
        batch: int,
        views: int,
        view_angles: torch.Tensor,
    ) -> torch.Tensor:
        pooled = F.adaptive_avg_pool2d(features, 1).flatten(1)
        tokens = self.token_projection(pooled).reshape(batch, views, -1)
        return tokens + self.angle_projection(self._angle_features(view_angles))

    def _fuse_tokens(
        self,
        tokens: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key_padding = ~view_valid.bool()
        fused_tokens = self.fusion(tokens, src_key_padding_mask=key_padding)
        quality_logits = self.view_quality(fused_tokens).squeeze(-1)
        quality_logits = quality_logits.masked_fill(key_padding, -1e4)
        quality = torch.softmax(quality_logits, dim=1)
        fused = torch.sum(fused_tokens * quality.unsqueeze(-1), dim=1)
        return self.fused_norm(fused), quality

    def encode_views(
        self,
        images: torch.Tensor,
        view_angles: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        b, v, c, h, w = images.shape
        flat = images.reshape(b * v, c, h, w)
        features = self.backbone(flat)
        tokens = self._tokens_from_features(
            features,
            batch=b,
            views=v,
            view_angles=view_angles,
        )
        dense_flat = self.dense(features, (h, w))
        dense = {
            key: value.reshape(b, v, value.shape[1], h, w)
            for key, value in dense_flat.items()
        }
        return tokens, dense

    def forward_geometry(
        self,
        images: torch.Tensor,
        view_angles: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Geometry-only mobile/export path; skips dense evidence heads."""
        if images.ndim != 5:
            raise ValueError("images must be [B,V,3,H,W]")
        if view_angles.shape[:2] != images.shape[:2] or view_angles.shape[-1] != 2:
            raise ValueError("view_angles must be [B,V,2]")
        if view_valid.shape != images.shape[:2]:
            raise ValueError("view_valid must be [B,V]")

        b, v, c, h, w = images.shape
        flat = images.reshape(b * v, c, h, w)
        features = self.backbone(flat)
        tokens = self._tokens_from_features(
            features,
            batch=b,
            views=v,
            view_angles=view_angles,
        )
        fused, quality = self._fuse_tokens(tokens, view_valid)
        coeff_norm = self.coeff_head(fused)
        vertices = self.reconstruct_vertices(coeff_norm)
        return vertices, coeff_norm, quality

    def forward(
        self,
        images: torch.Tensor,
        view_angles: torch.Tensor,
        view_valid: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if images.ndim != 5:
            raise ValueError("images must be [B,V,3,H,W]")
        if view_angles.shape[:2] != images.shape[:2] or view_angles.shape[-1] != 2:
            raise ValueError("view_angles must be [B,V,2]")
        if view_valid.shape != images.shape[:2]:
            raise ValueError("view_valid must be [B,V]")

        tokens, dense = self.encode_views(images, view_angles)
        fused, quality = self._fuse_tokens(tokens, view_valid)
        coeff_norm = self.coeff_head(fused)
        vertices = self.reconstruct_vertices(coeff_norm)
        return {
            "vertices": vertices,
            "coeff_norm": coeff_norm,
            "view_quality": quality,
            **dense,
        }
