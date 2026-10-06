from __future__ import annotations

import json
import math
from pathlib import Path
import random
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import ColorJitter, InterpolationMode

from .model import ShapeBasis, load_shape_basis


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def read_manifest(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "id" not in row or "vertices" not in row or not row.get("views"):
                raise ValueError(f"{path}:{line_number}: expected id, vertices and non-empty views")
            records.append(row)
    if not records:
        raise ValueError(f"manifest is empty: {path}")
    return records


def resolve_path(dataset_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else dataset_root / path


def build_shape_basis(
    manifest_path: str | Path,
    output_path: str | Path,
    *,
    components: int = 128,
) -> Path:
    manifest_path = Path(manifest_path)
    root = manifest_path.parent
    records = read_manifest(manifest_path)
    meshes = []
    vertex_count = None
    for row in records:
        vertices = np.load(resolve_path(root, row["vertices"])).astype(np.float32)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError(f"{row['vertices']}: vertices must be [V,3]")
        if vertex_count is None:
            vertex_count = vertices.shape[0]
        elif vertices.shape[0] != vertex_count:
            raise ValueError("all registered meshes must have identical vertex count/topology")
        meshes.append(vertices)

    x = np.stack(meshes, axis=0)
    mean = x.mean(axis=0, keepdims=True)
    flat = (x - mean).reshape(len(x), -1).astype(np.float32, copy=False)
    max_components = max(1, min(int(components), len(x) - 1, flat.shape[1]))

    # Full SVD becomes unnecessarily expensive for thousands of synthetic
    # identities. On Colab use PyTorch's truncated randomized PCA and the GPU
    # when available. Tiny datasets retain exact NumPy SVD for test stability.
    if len(x) >= 256 and max_components < min(flat.shape):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        q = min(min(flat.shape), max_components + min(16, max_components))
        print(
            f"[basis] randomized PCA | samples={len(x)} | features={flat.shape[1]} | "
            f"components={max_components} | device={device}",
            flush=True,
        )
        torch.manual_seed(42)
        matrix = torch.from_numpy(flat).to(device)
        with torch.no_grad():
            _, singular_t, vectors = torch.pca_lowrank(
                matrix,
                q=q,
                center=False,
                niter=3,
            )
        comp_flat = (
            vectors[:, :max_components]
            .T.contiguous()
            .cpu().numpy()
            .astype(np.float32)
        )
        singular = singular_t[:max_components].cpu().numpy().astype(np.float32)
        coeff = (matrix @ vectors[:, :max_components]).cpu().numpy()
        del matrix, vectors, singular_t
        if device.type == "cuda":
            torch.cuda.empty_cache()
    else:
        _, singular, vh = np.linalg.svd(flat, full_matrices=False)
        comp_flat = vh[:max_components].astype(np.float32)
        singular = singular[:max_components].astype(np.float32)
        coeff = flat @ comp_flat.T

    coeff_std = coeff.std(axis=0).astype(np.float32)
    coeff_std = np.maximum(coeff_std, 1e-4)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        mean=mean[0].astype(np.float32),
        components=comp_flat.reshape(max_components, vertex_count, 3),
        coeff_std=coeff_std,
        explained_singular_values=singular[:max_components].astype(np.float32),
    )
    return output_path


def project_vertices(vertices: torch.Tensor, basis: ShapeBasis) -> torch.Tensor:
    centered = vertices - basis.mean
    coeff = torch.einsum("vj,kvj->k", centered, basis.components)
    return coeff / basis.coeff_std.clamp_min(1e-6)


def _load_mask(path: Path, image_size: int) -> torch.Tensor:
    mask = Image.open(path).convert("L")
    mask = TF.resize(mask, [image_size, image_size], interpolation=InterpolationMode.NEAREST)
    return (TF.pil_to_tensor(mask).float() / 255.0).clamp(0, 1)


def _load_float_map(path: Path, image_size: int, channels: int) -> torch.Tensor:
    array = np.load(path).astype(np.float32)
    if channels == 1:
        if array.ndim == 2:
            array = array[None]
        elif array.ndim == 3 and array.shape[-1] == 1:
            array = np.moveaxis(array, -1, 0)
    else:
        if array.ndim == 3 and array.shape[-1] == channels:
            array = np.moveaxis(array, -1, 0)
    if array.shape[0] != channels:
        raise ValueError(f"{path}: expected {channels} channels, got {array.shape}")
    tensor = torch.from_numpy(array)
    tensor = TF.resize(tensor, [image_size, image_size], interpolation=InterpolationMode.BILINEAR)
    if channels == 3:
        tensor = torch.nn.functional.normalize(tensor, dim=0, eps=1e-6)
    return tensor


class RegisteredHeadDataset(Dataset):
    def __init__(
        self,
        manifest_path: str | Path,
        basis_path: str | Path,
        *,
        image_size: int = 256,
        max_views: int = 8,
        min_views: int = 1,
        random_views: bool = True,
        augment: bool = False,
        depth_scale_mm: float = 250.0,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.root = self.manifest_path.parent
        self.records = read_manifest(self.manifest_path)
        self.basis = load_shape_basis(basis_path)
        self.image_size = int(image_size)
        self.max_views = int(max_views)
        self.min_views = int(min_views)
        self.random_views = bool(random_views)
        self.augment = bool(augment)
        self.depth_scale_mm = float(depth_scale_mm)
        self.color_jitter = ColorJitter(
            brightness=0.15,
            contrast=0.15,
            saturation=0.10,
            hue=0.02,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _choose_views(self, views: list[dict[str, Any]]) -> list[dict[str, Any]]:
        available = min(len(views), self.max_views)
        if self.random_views:
            count = random.randint(min(self.min_views, available), available)
            return random.sample(views, count)
        return views[:available]

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        row = self.records[index]
        vertices = torch.from_numpy(
            np.load(resolve_path(self.root, row["vertices"])).astype(np.float32)
        )
        if tuple(vertices.shape) != tuple(self.basis.mean.shape):
            raise ValueError(
                f"{row['id']}: mesh shape {tuple(vertices.shape)} does not match basis "
                f"{tuple(self.basis.mean.shape)}"
            )
        coeff_norm = project_vertices(vertices, self.basis)
        chosen = self._choose_views(row["views"])

        s = self.image_size
        images = torch.zeros(self.max_views, 3, s, s)
        angles = torch.zeros(self.max_views, 2)
        view_valid = torch.zeros(self.max_views, dtype=torch.bool)
        normals = torch.zeros(self.max_views, 3, s, s)
        normal_valid = torch.zeros(self.max_views, dtype=torch.bool)
        depths = torch.zeros(self.max_views, 1, s, s)
        depth_valid = torch.zeros(self.max_views, dtype=torch.bool)
        masks = torch.zeros(self.max_views, 1, s, s)
        mask_valid = torch.zeros(self.max_views, dtype=torch.bool)
        confidence = torch.zeros(self.max_views, 1, s, s)
        confidence_valid = torch.zeros(self.max_views, dtype=torch.bool)

        for slot, view in enumerate(chosen):
            image_path = resolve_path(self.root, view["image"])
            image = Image.open(image_path).convert("RGB")
            image = TF.resize(image, [s, s], interpolation=InterpolationMode.BILINEAR)
            if self.augment:
                image = self.color_jitter(image)
            images[slot] = TF.normalize(
                TF.to_tensor(image),
                IMAGENET_MEAN,
                IMAGENET_STD,
            )
            angles[slot, 0] = float(view.get("yaw", 0.0))
            angles[slot, 1] = float(view.get("pitch", 0.0))
            view_valid[slot] = True

            if view.get("normal"):
                normals[slot] = _load_float_map(
                    resolve_path(self.root, view["normal"]),
                    s,
                    3,
                )
                normal_valid[slot] = True
            if view.get("depth"):
                depths[slot] = (
                    _load_float_map(resolve_path(self.root, view["depth"]), s, 1)
                    / self.depth_scale_mm
                )
                depth_valid[slot] = True
            if view.get("mask"):
                masks[slot] = _load_mask(resolve_path(self.root, view["mask"]), s)
                mask_valid[slot] = True
            if view.get("confidence"):
                confidence[slot] = _load_mask(
                    resolve_path(self.root, view["confidence"]),
                    s,
                )
                confidence_valid[slot] = True
            elif view.get("mask"):
                confidence[slot] = masks[slot]
                confidence_valid[slot] = True

        return {
            "id": row["id"],
            "images": images,
            "view_angles": angles,
            "view_valid": view_valid,
            "vertices": vertices,
            "coeff_norm": coeff_norm,
            "normal": normals,
            "normal_valid": normal_valid,
            "depth": depths,
            "depth_valid": depth_valid,
            "mask": masks,
            "mask_valid": mask_valid,
            "confidence": confidence,
            "confidence_valid": confidence_valid,
        }


def _sphere_vertices(rings: int = 16, sectors: int = 32) -> np.ndarray:
    vertices = []
    for i in range(rings):
        phi = math.pi * (i + 0.5) / rings
        for j in range(sectors):
            theta = 2.0 * math.pi * j / sectors
            vertices.append(
                [
                    math.sin(phi) * math.cos(theta),
                    math.cos(phi),
                    math.sin(phi) * math.sin(theta),
                ]
            )
    return np.asarray(vertices, dtype=np.float32)


def create_synthetic_smoke_dataset(
    root: str | Path,
    *,
    train_subjects: int = 16,
    val_subjects: int = 4,
    views_per_subject: int = 8,
    image_size: int = 160,
    seed: int = 42,
) -> Path:
    """Create a tiny dataset that only validates the complete pipeline."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    template = _sphere_vertices()
    yaws = np.linspace(-157.5, 157.5, views_per_subject, dtype=np.float32)

    def make_split(name: str, count: int) -> None:
        records = []
        for idx in range(count):
            sid = f"{name}_{idx:04d}"
            subject = root / sid
            subject.mkdir(parents=True, exist_ok=True)
            width = float(rng.uniform(72.0, 88.0))
            height = float(rng.uniform(105.0, 125.0))
            depth_axis = float(rng.uniform(85.0, 105.0))
            nose = float(rng.uniform(3.0, 10.0))
            vertices = template.copy()
            vertices[:, 0] *= width
            vertices[:, 1] *= height
            vertices[:, 2] *= depth_axis
            frontness = np.clip(vertices[:, 2] / max(depth_axis, 1e-6), 0.0, 1.0)
            center_y = np.exp(-((vertices[:, 1] / 35.0) ** 2))
            center_x = np.exp(-((vertices[:, 0] / 28.0) ** 2))
            vertices[:, 2] += nose * frontness**6 * center_x * center_y
            mesh_rel = f"{sid}/vertices.npy"
            np.save(root / mesh_rel, vertices.astype(np.float32))

            views = []
            for vi, yaw in enumerate(yaws):
                stem = subject / f"view_{vi:02d}"
                angle = math.radians(float(yaw))
                projected_width = (
                    abs(math.cos(angle)) * width
                    + abs(math.sin(angle)) * depth_axis
                )
                rx = max(18, int(projected_width / 105.0 * image_size * 0.34))
                ry = max(26, int(height / 125.0 * image_size * 0.40))
                cx, cy = image_size // 2, image_size // 2

                image = Image.new("RGB", (image_size, image_size), (30, 32, 36))
                draw = ImageDraw.Draw(image)
                base = int(130 + 45 * math.cos(angle))
                draw.ellipse(
                    (cx-rx, cy-ry, cx+rx, cy+ry),
                    fill=(base+35, base+12, base),
                )
                nose_shift = int(math.sin(angle) * nose * 0.8)
                draw.ellipse(
                    (cx-5+nose_shift, cy-5, cx+5+nose_shift, cy+7),
                    fill=(205, 155, 135),
                )
                image.save(stem.with_suffix(".jpg"), quality=92)

                yy, xx = np.mgrid[:image_size, :image_size]
                nx = (xx - cx) / max(rx, 1)
                ny = (yy - cy) / max(ry, 1)
                inside = nx * nx + ny * ny <= 1.0
                nz = np.sqrt(np.clip(1.0 - nx * nx - ny * ny, 0.0, 1.0))
                normal = np.stack([nx, -ny, nz], axis=-1).astype(np.float32)
                normal[~inside] = 0.0
                depth = (200.0 - 35.0 * nz).astype(np.float32)
                depth[~inside] = 0.0
                Image.fromarray(
                    inside.astype(np.uint8) * 255,
                    mode="L",
                ).save(str(stem) + "_mask.png")
                np.save(str(stem) + "_normal.npy", normal)
                np.save(str(stem) + "_depth.npy", depth)

                views.append({
                    "image": str(stem.relative_to(root).with_suffix(".jpg")),
                    "normal": str(Path(str(stem) + "_normal.npy").relative_to(root)),
                    "depth": str(Path(str(stem) + "_depth.npy").relative_to(root)),
                    "mask": str(Path(str(stem) + "_mask.png").relative_to(root)),
                    "yaw": float(yaw),
                    "pitch": 0.0,
                })
            records.append({
                "id": sid,
                "vertices": mesh_rel,
                "views": views,
            })

        with (root / f"{name}.jsonl").open("w", encoding="utf-8") as handle:
            for row in records:
                handle.write(json.dumps(row) + "\n")

    make_split("train", train_subjects)
    make_split("valid", val_subjects)
    return root
