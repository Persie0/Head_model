"""Tests for the Colab-only cache / progress logic without Google Drive."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys


def load_colab():
    path = Path(__file__).resolve().parents[1] / "colab" / "train_headscan_lite_colab.py"
    spec = importlib.util.spec_from_file_location("headscan_colab_recipe", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def setup_paths(module, monkeypatch, tmp_path: Path):
    work = tmp_path / "work"
    drive = tmp_path / "drive"
    work.mkdir()
    drive.mkdir()
    monkeypatch.setattr(module, "WORK", work)
    monkeypatch.setattr(module, "LOCAL_PREPARED", work / "restored")
    monkeypatch.setattr(module, "MAKEHUMAN_CACHE_ZIP", drive / "dataset.zip")
    monkeypatch.setattr(module, "MAKEHUMAN_PARTIAL_ZIP", drive / "dataset.partial.zip")
    monkeypatch.setattr(module, "SYNTH_IDENTITIES", 2)
    monkeypatch.setattr(module, "SYNTH_VIEWS", 1)
    monkeypatch.setattr(module, "SYNTH_IMAGE_SIZE", 64)
    return work, drive


def make_rows(source: Path, count: int):
    rows = []
    for idx in range(count):
        sid = f"mh_{idx:06d}"
        (source / "vertices").mkdir(exist_ok=True)
        (source / "images" / sid).mkdir(parents=True, exist_ok=True)
        (source / "masks" / sid).mkdir(parents=True, exist_ok=True)
        v = f"vertices/{sid}.npy"
        image = f"images/{sid}/view_00.jpg"
        mask = f"masks/{sid}/view_00.png"
        (source / v).write_bytes(b"fake_np")
        (source / image).write_bytes(b"fake_jpg")
        (source / mask).write_bytes(b"fake_png")
        row = {"id": sid, "vertices": v, "views": [{"image": image, "mask": mask}]}
        (source / "records").mkdir(exist_ok=True)
        (source / "records" / f"{sid}.json").write_text(json.dumps(row), encoding="utf-8")
        rows.append(row)
    return rows


def test_complete_cache_zip_restored_without_regeneration(tmp_path, monkeypatch, capsys):
    module = load_colab()
    _, drive = setup_paths(module, monkeypatch, tmp_path)
    source = tmp_path / "synth"
    source.mkdir()
    rows = make_rows(source, 2)
    for name, row in (("train", rows[0]), ("valid", rows[1])):
        (source / f"{name}.jsonl").write_text(json.dumps(row) + "\n", encoding="utf-8")
    (source / "head_faces.npy").write_bytes(b"fake_faces")
    (source / "makehuman_generation.json").write_text(
        json.dumps({
            "identities": 2,
            "train_subjects": 1,
            "valid_subjects": 1,
            "views_per_identity": 1,
            "image_size": 64,
        }),
        encoding="utf-8",
    )
    module.save_prepared_cache(source)
    assert (drive / "dataset.zip").is_file()
    result = module.restore_prepared_cache()
    assert result is not None
    assert (result / rows[1]["views"][0]["image"]).is_file()
    assert "[dataset-cache] RESTORED complete dataset" in capsys.readouterr().out
    assert not list(drive.glob("*.uploading"))


def test_partial_cache_zip_restores_generated_identity_records(tmp_path, monkeypatch, capsys):
    module = load_colab()
    _, drive = setup_paths(module, monkeypatch, tmp_path)
    source = tmp_path / "synth"
    source.mkdir()
    rows = make_rows(source, 1)
    module.save_partial_cache(source, 1, 2)
    assert (drive / "dataset.partial.zip").is_file()
    result = module.restore_partial_cache()
    assert result is not None
    assert (result / "records" / "mh_000000.json").is_file()
    assert (result / rows[0]["views"][0]["image"]).is_file()
    assert "[dataset-cache] RESTORED partial dataset" in capsys.readouterr().out


def test_trainer_output_is_streamed(tmp_path, capsys):
    module = load_colab()
    module.run_with_heartbeat(
        [sys.executable, "-u", "-c", "print('[basis-load] 2/2 meshes'); print('[epoch] 1/2 starting')"],
        cwd=tmp_path,
        heartbeat_seconds=5,
    )
    text = capsys.readouterr().out
    assert "[basis-load] 2/2 meshes" in text
    assert "[epoch] 1/2 starting" in text
    assert "process completed successfully" in text
