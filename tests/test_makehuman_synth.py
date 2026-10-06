from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from head_model.makehuman_synth import (
    generate_makehuman_dataset,
    read_target,
    render_head,
)


def _write_uv_sphere_obj(path: Path, rings: int = 16, sectors: int = 24) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = []
    for i in range(rings):
        phi = math.pi * (i + 0.5) / rings
        for j in range(sectors):
            theta = 2.0 * math.pi * j / sectors
            vertices.append(
                (
                    0.75 * math.sin(phi) * math.cos(theta),
                    1.10 * math.cos(phi),
                    0.90 * math.sin(phi) * math.sin(theta),
                )
            )

    faces = []
    for i in range(rings - 1):
        for j in range(sectors):
            a = i * sectors + j
            b = i * sectors + (j + 1) % sectors
            c = (i + 1) * sectors + (j + 1) % sectors
            d = (i + 1) * sectors + j
            faces.append((a, b, c))
            faces.append((a, c, d))

    with path.open("w", encoding="utf-8") as handle:
        for x, y, z in vertices:
            handle.write(f"v {x:.7f} {y:.7f} {z:.7f}\n")
        for a, b, c in faces:
            handle.write(f"f {a+1} {b+1} {c+1}\n")
    return len(vertices)


def _write_target(path: Path, vertex_count: int, *, z_delta: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("# CC0 synthetic test target\n")
        for index in range(vertex_count):
            handle.write(f"{index} 0 0 {z_delta:.6f}\n")


def _fake_makehuman_tree(root: Path) -> None:
    vertex_count = _write_uv_sphere_obj(
        root / "makehuman" / "data" / "3dobjs" / "base.obj"
    )
    target_root = root / "makehuman" / "data" / "targets"
    _write_target(target_root / "head" / "head-round.target", vertex_count, z_delta=0.001)
    _write_target(target_root / "nose" / "nose-width-incr.target", vertex_count, z_delta=0.002)
    _write_target(
        target_root / "macrodetails" / "caucasian-female-young.target",
        vertex_count,
        z_delta=0.001,
    )
    _write_target(
        target_root / "macrodetails" / "african-male-old.target",
        vertex_count,
        z_delta=-0.001,
    )


def test_read_target_matches_makehuman_text_format(tmp_path: Path):
    target = tmp_path / "shape.target"
    target.write_text(
        "# comment\n"
        "3 0.1 -0.2 0.3\n"
        "9 -0.4 0.5 0\n",
        encoding="utf-8",
    )
    indices, vectors = read_target(target)
    assert indices.tolist() == [3, 9]
    assert np.allclose(vectors[0], [0.1, -0.2, 0.3])


def test_render_head_returns_rgb_and_mask():
    vertices = np.asarray(
        [
            [-60, -70, 20],
            [60, -70, 20],
            [0, 80, 20],
            [0, 0, 90],
        ],
        dtype=np.float32,
    )
    faces = np.asarray(
        [[0, 1, 3], [1, 2, 3], [2, 0, 3], [0, 2, 1]],
        dtype=np.int32,
    )
    image, mask = render_head(
        vertices,
        faces,
        yaw=0.0,
        pitch=0.0,
        image_size=64,
        rng=np.random.default_rng(1),
    )
    assert image.shape == (64, 64, 3)
    assert mask.shape == (64, 64)
    assert image.dtype == np.uint8
    assert mask.max() == 255


def test_tiny_makehuman_dataset_end_to_end(tmp_path: Path):
    source = tmp_path / "makehuman"
    _fake_makehuman_tree(source)
    output = tmp_path / "dataset"

    report = generate_makehuman_dataset(
        source,
        output,
        identities=4,
        views_per_identity=2,
        image_size=64,
        validation_fraction=0.25,
        seed=7,
        progress_every=99,
    )

    assert report["identities"] == 4
    assert report["head_vertices"] >= 200
    assert report["head_triangles"] >= 100
    assert (output / "head_faces.npy").is_file()
    assert (output / "makehuman_generation.json").is_file()

    rows = []
    for split in ("train.jsonl", "valid.jsonl"):
        rows.extend(
            json.loads(line)
            for line in (output / split).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    assert len(rows) == 4
    assert all(len(row["views"]) == 2 for row in rows)
    assert all((output / row["vertices"]).is_file() for row in rows)
    assert all(
        (output / view["image"]).is_file()
        and (output / view["mask"]).is_file()
        for row in rows
        for view in row["views"]
    )
