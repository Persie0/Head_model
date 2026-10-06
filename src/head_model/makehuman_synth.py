from __future__ import annotations

import json
import math
from pathlib import Path
import random
import re
from typing import Iterable

import cv2
import numpy as np


# Only MakeHuman core targets are used. The MakeHuman project explicitly releases
# the base mesh, targets and other core assets under CC0. We do not import or copy
# MakeHuman's AGPL application code.
FINE_TARGET_DIRS = (
    "head",
    "cheek",
    "chin",
    "ears",
    "eyebrows",
    "eyes",
    "forehead",
    "mouth",
    "neck",
    "nose",
)
HEAD_ASYM_TERMS = (
    "brow",
    "cheek",
    "ear",
    "eye",
    "jaw",
    "mouth",
    "nose",
    "temple",
    "top",
)
OPPOSITE_SUFFIXES = {
    "incr",
    "decr",
    "up",
    "down",
    "in",
    "out",
    "forward",
    "backward",
    "min",
    "max",
    "round",
    "square",
    "triangle",
    "pointed",
}
DEFAULT_YAWS = (0.0, 45.0, 90.0, 135.0, 180.0, -135.0, -90.0, -45.0)


def read_obj(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read MakeHuman's OBJ base mesh and triangulate polygon faces."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if line.startswith("v "):
                parts = line.split()
                if len(parts) >= 4:
                    vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                parts = line.split()[1:]
                indices: list[int] = []
                for token in parts:
                    value = token.split("/", 1)[0]
                    index = int(value)
                    if index < 0:
                        index = len(vertices) + index
                    else:
                        index -= 1
                    indices.append(index)
                for i in range(1, len(indices) - 1):
                    faces.append((indices[0], indices[i], indices[i + 1]))
    if not vertices or not faces:
        raise ValueError(f"Could not read vertices/faces from {path}")
    return np.asarray(vertices, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def read_target(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a MakeHuman .target exactly as the application does."""
    indices: list[int] = []
    vectors: list[tuple[float, float, float]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for raw in handle:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith('"'):
                continue
            parts = line.split()
            if len(parts) != 4:
                continue
            indices.append(int(parts[0]))
            vectors.append((float(parts[1]), float(parts[2]), float(parts[3])))
    return np.asarray(indices, dtype=np.int32), np.asarray(vectors, dtype=np.float32)


def discover_targets(makehuman_root: str | Path) -> tuple[list[Path], list[Path], list[Path]]:
    root = Path(makehuman_root) / "makehuman" / "data" / "targets"
    fine: list[Path] = []
    structural: list[Path] = []
    for directory in FINE_TARGET_DIRS:
        folder = root / directory
        if not folder.is_dir():
            continue
        files = sorted(folder.rglob("*.target"))
        fine.extend(files)
        structural.extend(files)

    asym = root / "asym"
    if asym.is_dir():
        for path in sorted(asym.glob("*.target")):
            name = path.stem.lower()
            if any(term in name for term in HEAD_ASYM_TERMS):
                fine.append(path)
                structural.append(path)

    macro_root = root / "macrodetails"
    macro = sorted(
        path
        for path in macro_root.glob("*.target")
        if re.match(
            r"^(african|asian|caucasian)-(female|male)-(baby|child|young|old)\.target$",
            path.name,
            flags=re.IGNORECASE,
        )
    )
    if not fine or not macro:
        raise RuntimeError(
            "MakeHuman core target tree is incomplete. Expected the normal "
            "makehuman/data/targets head/face + macrodetails targets."
        )
    return fine, macro, structural


def _load_target_cache(paths: Iterable[Path]) -> dict[Path, tuple[np.ndarray, np.ndarray]]:
    return {path: read_target(path) for path in paths}


def _head_vertex_set(
    faces: np.ndarray,
    structural_targets: list[Path],
    cache: dict[Path, tuple[np.ndarray, np.ndarray]],
    *,
    adjacency_rings: int = 2,
) -> np.ndarray:
    selected: set[int] = set()
    for path in structural_targets:
        indices, _ = cache[path]
        selected.update(int(value) for value in indices)

    # Include neighboring neutral scalp/skin vertices which might not move in a
    # particular target. Expansion through mesh adjacency keeps topology fixed.
    for _ in range(max(0, adjacency_rings)):
        mask = np.isin(faces, np.fromiter(selected, dtype=np.int32))
        touching = faces[mask.any(axis=1)]
        selected.update(int(value) for value in touching.reshape(-1))

    return np.asarray(sorted(selected), dtype=np.int32)


def _extract_head_topology(
    vertices: np.ndarray,
    faces: np.ndarray,
    head_indices: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    full_to_head = np.full(len(vertices), -1, dtype=np.int32)
    full_to_head[head_indices] = np.arange(len(head_indices), dtype=np.int32)
    mapped = full_to_head[faces]
    keep = (mapped >= 0).all(axis=1)
    head_faces = mapped[keep]
    if len(head_faces) < 100 or len(head_indices) < 200:
        raise RuntimeError(
            f"Head subset unexpectedly small: {len(head_indices)} vertices, "
            f"{len(head_faces)} triangles"
        )
    used = np.unique(head_faces.reshape(-1))
    compact_map = np.full(len(head_indices), -1, dtype=np.int32)
    compact_map[used] = np.arange(len(used), dtype=np.int32)
    compact_faces = compact_map[head_faces]
    compact_full_indices = head_indices[used]
    return vertices[compact_full_indices], compact_faces, compact_full_indices


def _feature_key(path: Path) -> str:
    name = path.stem.lower()
    name = re.sub(r"^[lr]-", "", name)
    name = re.sub(r"^asym-", "", name)
    parts = name.split("-")
    if parts and parts[-1] in OPPOSITE_SUFFIXES:
        parts = parts[:-1]
    return "/".join((path.parent.name.lower(), "-".join(parts)))


def _sample_fine_targets(
    rng: np.random.Generator,
    fine_targets: list[Path],
    *,
    min_features: int = 10,
    max_features: int = 22,
) -> list[tuple[Path, float]]:
    groups: dict[str, list[Path]] = {}
    for path in fine_targets:
        groups.setdefault(_feature_key(path), []).append(path)
    keys = list(groups)
    rng.shuffle(keys)
    count = min(len(keys), int(rng.integers(min_features, max_features + 1)))
    chosen: list[tuple[Path, float]] = []

    for key in keys[:count]:
        variants = groups[key]
        path = variants[int(rng.integers(0, len(variants)))]
        weight = float(rng.uniform(0.12, 0.72))
        chosen.append((path, weight))

        # Most humans are approximately bilateral. If the chosen target has an
        # obvious L/R counterpart, apply it with nearly the same strength.
        stem = path.stem
        counterpart: Path | None = None
        if stem.startswith("l-"):
            counterpart = path.with_name("r-" + stem[2:] + path.suffix)
        elif stem.startswith("r-"):
            counterpart = path.with_name("l-" + stem[2:] + path.suffix)
        elif "-l" in stem:
            counterpart = path.with_name(stem.replace("-l", "-r") + path.suffix)
        elif "-r" in stem:
            counterpart = path.with_name(stem.replace("-r", "-l") + path.suffix)
        if counterpart is not None and counterpart.is_file() and rng.random() < 0.88:
            chosen.append((counterpart, weight * float(rng.uniform(0.90, 1.10))))

    return chosen


def _apply_targets(
    base_vertices: np.ndarray,
    applications: Iterable[tuple[Path, float]],
    cache: dict[Path, tuple[np.ndarray, np.ndarray]],
) -> np.ndarray:
    vertices = base_vertices.copy()
    for path, weight in applications:
        indices, vectors = cache[path]
        vertices[indices] += vectors * float(weight)
    return vertices


def _rotation_matrix(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    pitch = math.radians(pitch_deg)
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.asarray([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]], dtype=np.float32)
    rx = np.asarray([[1.0, 0.0, 0.0], [0.0, cp, -sp], [0.0, sp, cp]], dtype=np.float32)
    return rx @ ry


def _skin_color(rng: np.random.Generator) -> np.ndarray:
    # Broad procedural range: intentionally not tied to a demographic label.
    anchors = np.asarray(
        [
            [242, 205, 178],
            [222, 170, 135],
            [185, 123, 88],
            [128, 78, 52],
            [82, 49, 35],
        ],
        dtype=np.float32,
    )
    i = int(rng.integers(0, len(anchors) - 1))
    t = float(rng.uniform(0.0, 1.0))
    color = anchors[i] * (1.0 - t) + anchors[i + 1] * t
    color *= float(rng.uniform(0.90, 1.08))
    return np.clip(color, 20, 250)


def render_head(
    vertices_mm: np.ndarray,
    faces: np.ndarray,
    *,
    yaw: float,
    pitch: float,
    image_size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Fast painter-style CPU rasterizer for synthetic domain-randomized RGB."""
    rotation = _rotation_matrix(yaw, pitch)
    v = vertices_mm @ rotation.T
    distance = float(rng.uniform(500.0, 650.0))
    focal = float(image_size * rng.uniform(1.45, 1.85))
    denom = np.maximum(distance - v[:, 2], 120.0)
    x = image_size * 0.5 + focal * v[:, 0] / denom
    y = image_size * 0.52 - focal * v[:, 1] / denom
    projected = np.stack([x, y], axis=1)

    tri3 = v[faces]
    e1 = tri3[:, 1] - tri3[:, 0]
    e2 = tri3[:, 2] - tri3[:, 0]
    normals = np.cross(e1, e2)
    norms = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(norms, 1e-6)
    center_z = tri3[:, :, 2].mean(axis=1)

    # Whichever winding the OBJ uses, retain the side facing the +Z camera.
    visible = normals[:, 2] > 0.01
    if visible.sum() < len(faces) * 0.05:
        visible = normals[:, 2] < -0.01
        normals = -normals

    face_ids = np.flatnonzero(visible)
    polys = np.rint(projected[faces[face_ids]]).astype(np.int32)
    depths = center_z[face_ids]
    n = normals[face_ids]

    bg0 = rng.integers(10, 245, size=3, dtype=np.uint8)
    bg1 = rng.integers(10, 245, size=3, dtype=np.uint8)
    alpha = np.linspace(0.0, 1.0, image_size, dtype=np.float32)[:, None, None]
    image = (
        bg0[None, None, :].astype(np.float32) * (1.0 - alpha)
        + bg1[None, None, :].astype(np.float32) * alpha
    )
    image = np.repeat(image, image_size, axis=1).astype(np.uint8)
    mask = np.zeros((image_size, image_size), dtype=np.uint8)

    light = rng.normal(size=3).astype(np.float32)
    light[2] = abs(light[2]) + 0.7
    light /= max(float(np.linalg.norm(light)), 1e-6)
    shade = np.clip(0.38 + 0.72 * np.maximum(0.0, n @ light), 0.22, 1.12)
    skin = _skin_color(rng)

    # Draw far-to-near in coarse depth bins, batching equal shade levels with
    # cv2.fillPoly. This is much faster than one Python call per triangle.
    depth_lo, depth_hi = float(depths.min(initial=0.0)), float(depths.max(initial=1.0))
    depth_bins = np.clip(
        ((depths - depth_lo) / max(depth_hi - depth_lo, 1e-6) * 23.999).astype(np.int32),
        0,
        23,
    )
    shade_bins = np.clip((shade * 14.0).astype(np.int32), 0, 15)
    for db in range(24):
        in_depth = depth_bins == db
        if not in_depth.any():
            continue
        for sb in np.unique(shade_bins[in_depth]):
            sel = in_depth & (shade_bins == sb)
            if not sel.any():
                continue
            value = (float(sb) + 1.0) / 15.0
            color = tuple(int(v) for v in np.clip(skin * value, 0, 255))
            cv2.fillPoly(image, list(polys[sel]), color, lineType=cv2.LINE_AA)
    if len(polys):
        cv2.fillPoly(mask, list(polys), 255, lineType=cv2.LINE_AA)

    # Hair is an occluder, not part of the geometry target. A procedural cap
    # teaches the geometry network that dark top-of-head pixels need not be scalp.
    if rng.random() < 0.58:
        hair_color = tuple(int(v) for v in rng.integers(8, 90, size=3))
        center = (int(image_size * rng.uniform(0.48, 0.52)), int(image_size * rng.uniform(0.27, 0.34)))
        axes = (int(image_size * rng.uniform(0.23, 0.34)), int(image_size * rng.uniform(0.15, 0.25)))
        cv2.ellipse(image, center, axes, float(rng.uniform(-10, 10)), 180, 360, hair_color, -1, cv2.LINE_AA)

    if rng.random() < 0.45:
        k = int(rng.choice([3, 3, 5]))
        image = cv2.GaussianBlur(image, (k, k), float(rng.uniform(0.25, 0.9)))
    noise_sigma = float(rng.uniform(0.0, 7.0))
    if noise_sigma > 0.2:
        noise = rng.normal(0.0, noise_sigma, image.shape).astype(np.float32)
        image = np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)

    return image, mask


def _write_jpeg(path: Path, image_rgb: np.ndarray, rng: np.random.Generator) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    quality = int(rng.integers(78, 98))
    cv2.imwrite(
        str(path),
        cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
        [int(cv2.IMWRITE_JPEG_QUALITY), quality],
    )


def _write_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), mask)


def prepare_makehuman_assets(makehuman_root: str | Path) -> dict[str, object]:
    """Parse MakeHuman's CC0 mesh/targets and build a fixed head topology."""
    root = Path(makehuman_root)
    base_path = root / "makehuman" / "data" / "3dobjs" / "base.obj"
    if not base_path.is_file():
        raise FileNotFoundError(f"MakeHuman base mesh not found: {base_path}")

    vertices, faces = read_obj(base_path)
    fine, macro, structural = discover_targets(root)
    all_target_paths = sorted(set(fine + macro + structural))
    cache = _load_target_cache(all_target_paths)
    head_indices = _head_vertex_set(faces, structural, cache)
    neutral_head, head_faces, full_indices = _extract_head_topology(
        vertices, faces, head_indices
    )

    # MakeHuman uses Y-up. Keep a fixed dataset-wide physical scale so shape and
    # head-size differences remain meaningful. Neutral adult head height ~=240 mm.
    neutral_height = float(np.ptp(neutral_head[:, 1]))
    scale_to_mm = 240.0 / max(neutral_height, 1e-6)

    nose_paths = [p for p in structural if p.parent.name.lower() == "nose"]
    nose_indices: list[int] = []
    for path in nose_paths:
        idx, _ = cache[path]
        nose_indices.extend(int(v) for v in idx)
    front_sign = 1.0
    if nose_indices:
        nose_mean_z = float(vertices[np.unique(nose_indices), 2].mean())
        head_center_z = float(neutral_head[:, 2].mean())
        front_sign = 1.0 if nose_mean_z >= head_center_z else -1.0

    return {
        "base_vertices": vertices,
        "faces": faces,
        "fine_targets": fine,
        "macro_targets": macro,
        "target_cache": cache,
        "head_full_indices": full_indices,
        "head_faces": head_faces,
        "scale_to_mm": scale_to_mm,
        "front_sign": front_sign,
    }


def generate_identity(
    assets: dict[str, object],
    *,
    identity_index: int,
    output_root: Path,
    views_per_identity: int,
    image_size: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed + identity_index * 10007)
    base_vertices = assets["base_vertices"]
    fine_targets = assets["fine_targets"]
    macro_targets = assets["macro_targets"]
    cache = assets["target_cache"]
    head_full_indices = assets["head_full_indices"]
    head_faces = assets["head_faces"]

    macro_path = macro_targets[int(rng.integers(0, len(macro_targets)))]
    applications: list[tuple[Path, float]] = [(macro_path, float(rng.uniform(0.85, 1.0)))]
    applications.extend(_sample_fine_targets(rng, fine_targets))
    morphed = _apply_targets(base_vertices, applications, cache)
    head = morphed[head_full_indices].astype(np.float32)

    # Fix the front of the canonical representation to +Z.
    head[:, 2] *= float(assets["front_sign"])
    head *= float(assets["scale_to_mm"])
    bbox_center = (head.min(axis=0) + head.max(axis=0)) * 0.5
    head -= bbox_center

    subject = f"mh_{identity_index:06d}"
    vertex_rel = Path("vertices") / f"{subject}.npy"
    (output_root / "vertices").mkdir(parents=True, exist_ok=True)
    np.save(output_root / vertex_rel, head)

    views: list[dict[str, object]] = []
    yaws = list(DEFAULT_YAWS[: min(views_per_identity, len(DEFAULT_YAWS))])
    if views_per_identity > len(yaws):
        extra = np.linspace(-180.0, 180.0, views_per_identity, endpoint=False).tolist()
        yaws = extra[:views_per_identity]

    for view_index, base_yaw in enumerate(yaws):
        yaw = float(base_yaw + rng.uniform(-5.0, 5.0))
        pitch = float(rng.uniform(-8.0, 8.0))
        image, mask = render_head(
            head,
            head_faces,
            yaw=yaw,
            pitch=pitch,
            image_size=image_size,
            rng=rng,
        )
        image_rel = Path("images") / subject / f"view_{view_index:02d}.jpg"
        mask_rel = Path("masks") / subject / f"view_{view_index:02d}.png"
        _write_jpeg(output_root / image_rel, image, rng)
        _write_mask(output_root / mask_rel, mask)
        views.append({
            "image": str(image_rel),
            "mask": str(mask_rel),
            "yaw": yaw,
            "pitch": pitch,
        })

    return {
        "id": subject,
        "vertices": str(vertex_rel),
        "views": views,
        "generator": {
            "macro_target": str(macro_path.name),
            "fine_target_count": len(applications) - 1,
        },
    }


def generate_makehuman_dataset(
    makehuman_root: str | Path,
    output_root: str | Path,
    *,
    identities: int = 2500,
    views_per_identity: int = 8,
    image_size: int = 320,
    validation_fraction: float = 0.10,
    seed: int = 42,
    progress_every: int = 25,
) -> dict[str, object]:
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    records_dir = output_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)

    assets = prepare_makehuman_assets(makehuman_root)
    records: list[dict[str, object]] = []
    for index in range(int(identities)):
        record_path = records_dir / f"mh_{index:06d}.json"
        if record_path.is_file():
            row = json.loads(record_path.read_text(encoding="utf-8"))
        else:
            row = generate_identity(
                assets,
                identity_index=index,
                output_root=output_root,
                views_per_identity=views_per_identity,
                image_size=image_size,
                seed=seed,
            )
            record_path.write_text(json.dumps(row), encoding="utf-8")
        records.append(row)
        if (index + 1) % max(1, progress_every) == 0 or index + 1 == identities:
            print(
                f"[makehuman] generated {index+1}/{identities} identities "
                f"({100.0*(index+1)/max(1,identities):.1f}%)",
                flush=True,
            )

    rng = random.Random(seed)
    order = list(range(len(records)))
    rng.shuffle(order)
    valid_count = max(1, int(round(len(records) * validation_fraction)))
    valid_ids = set(order[:valid_count])
    train_rows = [row for i, row in enumerate(records) if i not in valid_ids]
    valid_rows = [row for i, row in enumerate(records) if i in valid_ids]

    for split, rows in (("train", train_rows), ("valid", valid_rows)):
        with (output_root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")

    np.save(output_root / "head_faces.npy", assets["head_faces"])
    report = {
        "dataset": "MakeHuman CC0 synthetic heads",
        "identities": len(records),
        "train_subjects": len(train_rows),
        "valid_subjects": len(valid_rows),
        "views_per_identity": views_per_identity,
        "image_size": image_size,
        "head_vertices": int(len(assets["head_full_indices"])),
        "head_triangles": int(len(assets["head_faces"])),
        "fine_target_count": len(assets["fine_targets"]),
        "macro_target_count": len(assets["macro_targets"]),
        "scale_to_mm": float(assets["scale_to_mm"]),
        "license_note": (
            "Generated only from MakeHuman core base mesh and targets. "
            "MakeHuman documents these core assets as CC0. No third-party "
            "community assets are used."
        ),
    }
    (output_root / "makehuman_generation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(
        f"[makehuman] dataset ready | train={len(train_rows)} | "
        f"valid={len(valid_rows)} | vertices={report['head_vertices']} | "
        f"triangles={report['head_triangles']}",
        flush=True,
    )
    return report
