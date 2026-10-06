from __future__ import annotations

import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image

from head_model.headspace import (
    extract_relevant_headspace_archives,
    prepare_headspace_dataset,
    vertices_to_mm,
)


def _write_obj(path: Path, scale: float = 0.2) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    vertices = [
        (-0.4 * scale, -0.5 * scale, -0.3 * scale),
        (0.4 * scale, -0.5 * scale, -0.3 * scale),
        (0.4 * scale, 0.5 * scale, 0.3 * scale),
        (-0.4 * scale, 0.5 * scale, 0.3 * scale),
    ]
    path.write_text(
        "".join(f"v {x} {y} {z}\n" for x, y, z in vertices),
        encoding="utf-8",
    )


def _write_png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 60), (120, 80, 60)).save(path)


def test_vertices_to_mm_converts_flame_meter_scale():
    vertices = np.asarray(
        [[-0.1, 0.0, 0.0], [0.1, 0.2, 0.15]],
        dtype=np.float32,
    )
    converted, scale = vertices_to_mm(vertices)
    assert scale == 1000.0
    assert np.isclose(np.ptp(converted, axis=0).max(), 200.0)


def test_prepare_headspace_pairs_registration_and_color_views(tmp_path: Path):
    registrations = tmp_path / "src" / "registrations"
    images = tmp_path / "src" / "images"
    for subject in ("00001", "00002", "00003"):
        _write_obj(registrations / subject / "neutral.obj")
        _write_png(images / subject / "1C.png")
        _write_png(images / subject / "2C.png")
        _write_png(images / subject / "1A.png")  # IR/non-color-style name: ignored

    output = tmp_path / "prepared"
    report = prepare_headspace_dataset(
        [tmp_path / "src"],
        output,
        image_size=64,
        max_views=8,
        validation_fraction=0.34,
        min_subjects=3,
    )

    assert report["usable_subjects"] == 3
    assert report["vertex_count"] == 4
    assert report["max_views"] == 2
    rows = []
    for name in ("train.jsonl", "valid.jsonl"):
        rows.extend(
            json.loads(line)
            for line in (output / name).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    assert len(rows) == 3
    assert all(len(row["views"]) == 2 for row in rows)
    assert all("yaw" not in view for row in rows for view in row["views"])
    assert all((output / row["vertices"]).is_file() for row in rows)


def test_archive_extraction_keeps_only_needed_headspace_files(tmp_path: Path):
    archive_path = tmp_path / "headspace.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "FLAME/registrations/00001/neutral.obj",
            "v 0 0 0\nv 0.1 0 0\nv 0 0.1 0\n",
        )
        archive.writestr("3dMD/00001/1C.png", b"color")
        archive.writestr("3dMD/00001/1A.png", b"infrared")
        archive.writestr("3dMD/00001/calibration.TKA", b"calibration")

    destination = tmp_path / "out"
    result = extract_relevant_headspace_archives(tmp_path, destination)

    assert result["archives"] == 1
    assert result["files"] == 2
    assert (destination / "FLAME/registrations/00001/neutral.obj").is_file()
    assert (destination / "3dMD/00001/1C.png").is_file()
    assert not (destination / "3dMD/00001/1A.png").exists()
    assert not (destination / "3dMD/00001/calibration.TKA").exists()
