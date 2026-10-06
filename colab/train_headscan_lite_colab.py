"""Google Colab recipe for HeadScan-Lite trained on Headspace/LYHM.

Required licensed Headspace packages:
  1) FLAME registrations (OBJ files + model parameters package)
  2) 3dMD package (raw PNG color-camera views + TKA calibration package)

Place the downloaded archives, or their extracted folders, anywhere under:
  MyDrive/head_model/headspace/

The script automatically extracts only the files needed for this trainer,
converts registered FLAME targets to metric NumPy arrays, pairs them with
Headspace color-camera images, writes train/validation manifests, caches the
prepared dataset in Drive, and then trains/resumes.

Headspace/LYHM is non-commercial research/education data. This recipe does not
download or redistribute the licensed dataset.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import zipfile

HEADSCAN_COLAB_VERSION = "2026-10-06-v2-headspace"

IMAGE_SIZE = 256
MAX_VIEWS = 8
MIN_VIEWS = 1
SHAPE_COMPONENTS = 128
TOKEN_DIM = 256
TRANSFORMER_LAYERS = 3
TRANSFORMER_HEADS = 4
EPOCHS = 60
BATCH_SIZE = 2
PROGRESS_EVERY = 1
CHECKPOINT_EVERY = 5
HEARTBEAT_SECONDS = 5
USE_PRETRAINED_BACKBONE = True

HEADSPACE_PREP_IMAGE_SIZE = 512
HEADSPACE_VALIDATION_FRACTION = 0.10
HEADSPACE_MIN_SUBJECTS = 20
HEADSPACE_CACHE_VERSION = "v1"

ALLOW_SYNTHETIC_SMOKE_TEST = True
SMOKE_EPOCHS = 2
SMOKE_IMAGE_SIZE = 128
SMOKE_COMPONENTS = 8

WORK = Path("/content/headscan_lite")
REPO = WORK / "Head_model"
LOCAL_HEADSPACE_SOURCE = WORK / "headspace_source"
LOCAL_PREPARED = WORK / "headspace_prepared"
LOCAL_RESUME = WORK / "resume-last.pt"
DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive" / "head_model"
HEADSPACE_DRIVE_DIR = DRIVE_ROOT / "headspace"
HEADSPACE_CACHE_ZIP = DRIVE_ROOT / f"headspace_prepared_{HEADSPACE_CACHE_VERSION}.zip"
REAL_RUN_DIR = DRIVE_ROOT / "headscan-lite-headspace"
SMOKE_RUN_DIR = DRIVE_ROOT / "headscan-lite-smoke"


def run(command: list[str], *, cwd: Path | None = None) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        check=True,
    )


def run_with_heartbeat(
    command: list[str],
    *,
    cwd: Path | None = None,
    heartbeat_seconds: float = HEARTBEAT_SECONDS,
) -> None:
    print("+", " ".join(map(str, command)), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
    )
    next_heartbeat = started + max(1.0, heartbeat_seconds)
    while True:
        code = process.poll()
        if code is not None:
            if code != 0:
                raise subprocess.CalledProcessError(code, command)
            return
        now = time.monotonic()
        if now >= next_heartbeat:
            print(
                f"[trainer] process alive | waiting for next trainer log line | "
                f"elapsed {now-started:.0f}s",
                flush=True,
            )
            next_heartbeat = now + max(1.0, heartbeat_seconds)
        time.sleep(0.5)


def copy_with_progress(source: Path, destination: Path, label: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    total = source.stat().st_size
    copied = 0
    report_step = max(16 * 1024 * 1024, total // 20 if total else 1)
    next_report = report_step
    print(
        f"[{label}] {source} -> {destination} ({total/1024/1024:.1f} MiB)",
        flush=True,
    )
    with source.open("rb") as src, destination.open("wb") as dst:
        while True:
            chunk = src.read(16 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
            if copied >= next_report or copied == total:
                print(
                    f"[{label}] {copied/1024/1024:.1f}/"
                    f"{total/1024/1024:.1f} MiB "
                    f"({100.0*copied/max(1,total):.0f}%)",
                    flush=True,
                )
                next_report = copied + report_step


def require_gpu() -> None:
    try:
        output = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception as exc:
        raise RuntimeError(
            "No NVIDIA GPU detected. In Colab choose "
            "Runtime -> Change runtime type -> GPU."
        ) from exc
    print("GPU:", output, flush=True)


def mount_drive() -> None:
    try:
        from google.colab import drive
    except ImportError as exc:
        raise RuntimeError("This recipe is intended for Google Colab.") from exc
    drive.mount(str(DRIVE_MOUNT), force_remount=False)
    DRIVE_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Persistent Drive root: {DRIVE_ROOT}", flush=True)


def clone_repo() -> None:
    WORK.mkdir(parents=True, exist_ok=True)
    if REPO.exists():
        shutil.rmtree(REPO)
    run([
        "git",
        "clone",
        "--depth",
        "1",
        "https://github.com/Persie0/Head_model.git",
        str(REPO),
    ])


def install_dependencies() -> None:
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        f"{REPO}[train,export]",
    ])


def _prepared_ok(root: Path) -> bool:
    return (
        (root / "train.jsonl").is_file()
        and (root / "valid.jsonl").is_file()
        and (root / "headspace_conversion.json").is_file()
    )


def restore_prepared_cache() -> Path | None:
    if not HEADSPACE_CACHE_ZIP.is_file():
        return None
    if LOCAL_PREPARED.exists():
        shutil.rmtree(LOCAL_PREPARED)
    LOCAL_PREPARED.mkdir(parents=True, exist_ok=True)
    local_zip = WORK / HEADSPACE_CACHE_ZIP.name
    copy_with_progress(HEADSPACE_CACHE_ZIP, local_zip, "headspace-cache-copy")
    print("[headspace-cache] extracting prepared dataset...", flush=True)
    with zipfile.ZipFile(local_zip) as archive:
        archive.extractall(LOCAL_PREPARED)
    if not _prepared_ok(LOCAL_PREPARED):
        print(
            "[headspace-cache] cache is incomplete; rebuilding from licensed source files.",
            flush=True,
        )
        shutil.rmtree(LOCAL_PREPARED)
        return None
    report = json.loads(
        (LOCAL_PREPARED / "headspace_conversion.json").read_text(encoding="utf-8")
    )
    print(
        f"[headspace-cache] ready | usable={report.get('usable_subjects')} | "
        f"train={report.get('train_subjects')} | valid={report.get('valid_subjects')}",
        flush=True,
    )
    return LOCAL_PREPARED


def save_prepared_cache(root: Path) -> None:
    local_zip = WORK / HEADSPACE_CACHE_ZIP.name
    if local_zip.exists():
        local_zip.unlink()
    print("[headspace-cache] packing prepared dataset for future Colab sessions...", flush=True)
    shutil.make_archive(
        str(local_zip.with_suffix("")),
        "zip",
        root_dir=root,
    )
    copy_with_progress(local_zip, HEADSPACE_CACHE_ZIP, "headspace-cache-save")
    print(f"[headspace-cache] saved: {HEADSPACE_CACHE_ZIP}", flush=True)


def prepare_headspace() -> tuple[Path, bool]:
    cached = restore_prepared_cache()
    if cached is not None:
        return cached, False

    if HEADSPACE_DRIVE_DIR.exists():
        from head_model.headspace import (
            extract_relevant_headspace_archives,
            prepare_headspace_dataset,
        )

        if LOCAL_HEADSPACE_SOURCE.exists():
            shutil.rmtree(LOCAL_HEADSPACE_SOURCE)
        LOCAL_HEADSPACE_SOURCE.mkdir(parents=True, exist_ok=True)

        print(f"[headspace] licensed source folder: {HEADSPACE_DRIVE_DIR}", flush=True)
        extraction = extract_relevant_headspace_archives(
            HEADSPACE_DRIVE_DIR,
            LOCAL_HEADSPACE_SOURCE,
        )
        print(
            f"[headspace] relevant archive extraction: "
            f"{extraction['archives']} archive(s), {extraction['files']} file(s)",
            flush=True,
        )

        if LOCAL_PREPARED.exists():
            shutil.rmtree(LOCAL_PREPARED)
        LOCAL_PREPARED.mkdir(parents=True, exist_ok=True)

        try:
            report = prepare_headspace_dataset(
                [HEADSPACE_DRIVE_DIR, LOCAL_HEADSPACE_SOURCE],
                LOCAL_PREPARED,
                image_size=HEADSPACE_PREP_IMAGE_SIZE,
                max_views=MAX_VIEWS,
                validation_fraction=HEADSPACE_VALIDATION_FRACTION,
                min_subjects=HEADSPACE_MIN_SUBJECTS,
            )
        except (FileNotFoundError, RuntimeError) as exc:
            print("", flush=True)
            print("[headspace] Source files were found but are incomplete:", flush=True)
            print(f"  {exc}", flush=True)
            print("", flush=True)
            print(
                "Expected licensed data under MyDrive/head_model/headspace/:",
                flush=True,
            )
            print(
                "  - Headspace FLAME registrations package "
                "(registrations/<actor_id>/*.obj)",
                flush=True,
            )
            print(
                "  - Headspace 3dMD PNG package "
                "(subject folders containing *C.png color-camera views)",
                flush=True,
            )
            raise

        print(json.dumps(report, indent=2), flush=True)
        save_prepared_cache(LOCAL_PREPARED)
        return LOCAL_PREPARED, False

    if not ALLOW_SYNTHETIC_SMOKE_TEST:
        raise FileNotFoundError(
            f"No Headspace data found at {HEADSPACE_DRIVE_DIR}. "
            "Place the licensed FLAME-registration and 3dMD packages there."
        )

    print("", flush=True)
    print("=" * 78, flush=True)
    print("[SMOKE TEST] NO LICENSED HEADSPACE DATA FOUND", flush=True)
    print(
        f"[SMOKE TEST] Expected packages under: {HEADSPACE_DRIVE_DIR}",
        flush=True,
    )
    print(
        "[SMOKE TEST] Generating a tiny procedural dataset only to verify "
        "setup/training/checkpoint/export.",
        flush=True,
    )
    print("[SMOKE TEST] THE RESULTING MODEL IS NOT A REAL HEAD SCANNER.", flush=True)
    print("=" * 78, flush=True)

    from head_model.data import create_synthetic_smoke_dataset

    if LOCAL_PREPARED.exists():
        shutil.rmtree(LOCAL_PREPARED)
    create_synthetic_smoke_dataset(
        LOCAL_PREPARED,
        train_subjects=16,
        val_subjects=4,
        views_per_subject=8,
        image_size=160,
    )
    return LOCAL_PREPARED, True


def find_resume_checkpoint(run_dir: Path) -> Path | None:
    checkpoint = run_dir / "last.pt"
    return checkpoint if checkpoint.is_file() and checkpoint.stat().st_size > 0 else None


def stage_resume_checkpoint(
    source: Path | None,
    destination: Path = LOCAL_RESUME,
) -> Path | None:
    if source is None:
        if destination.exists():
            destination.unlink()
        return None
    if destination.exists():
        destination.unlink()
    copy_with_progress(source, destination, "resume-copy")
    print("[resume-copy] local checkpoint ready.", flush=True)
    return destination


def training_command(
    dataset_root: Path,
    run_dir: Path,
    resume: Path | None,
    *,
    smoke: bool,
) -> list[str]:
    image_size = SMOKE_IMAGE_SIZE if smoke else IMAGE_SIZE
    epochs = SMOKE_EPOCHS if smoke else EPOCHS
    components = SMOKE_COMPONENTS if smoke else SHAPE_COMPONENTS

    command = [
        sys.executable,
        "-u",
        "-m",
        "head_model.train",
        "--train-manifest",
        str(dataset_root / "train.jsonl"),
        "--valid-manifest",
        str(dataset_root / "valid.jsonl"),
        "--basis-path",
        str(run_dir / "shape_basis.npz"),
        "--output-dir",
        str(run_dir),
        "--image-size",
        str(image_size),
        "--max-views",
        str(MAX_VIEWS),
        "--min-views",
        str(MIN_VIEWS),
        "--shape-components",
        str(components),
        "--token-dim",
        str(TOKEN_DIM),
        "--transformer-layers",
        str(TRANSFORMER_LAYERS),
        "--transformer-heads",
        str(TRANSFORMER_HEADS),
        "--epochs",
        str(epochs),
        "--batch-size",
        str(BATCH_SIZE),
        "--num-workers",
        "2",
        "--progress-every",
        str(PROGRESS_EVERY),
        "--checkpoint-every",
        str(CHECKPOINT_EVERY),
        "--geometry-only-training",
    ]
    if USE_PRETRAINED_BACKBONE:
        command.append("--pretrained-backbone")
    if resume is not None:
        command.extend(["--resume", str(resume)])
    return command


def print_run_configuration(
    dataset_root: Path,
    run_dir: Path,
    resume_source: Path | None,
    local_resume: Path | None,
    *,
    smoke: bool,
) -> None:
    image_size = SMOKE_IMAGE_SIZE if smoke else IMAGE_SIZE
    epochs = SMOKE_EPOCHS if smoke else EPOCHS
    components = SMOKE_COMPONENTS if smoke else SHAPE_COMPONENTS

    print("\nHeadScan-Lite training configuration", flush=True)
    print(
        "  dataset: "
        + ("procedural smoke test" if smoke else "Headspace / LYHM"),
        flush=True,
    )
    print(
        "  model: MobileNetV3-Small + variable-view Transformer + PCA FLAME prior",
        flush=True,
    )
    print(f"  prepared data: {dataset_root}", flush=True)
    print(f"  input: {image_size}x{image_size}", flush=True)
    print(f"  views: {MIN_VIEWS}..{MAX_VIEWS} slots; random available subset each sample", flush=True)
    print(f"  PCA components: up to {components}", flush=True)
    print(f"  token dim: {TOKEN_DIM}", flush=True)
    print(
        f"  transformer: {TRANSFORMER_LAYERS} layers / "
        f"{TRANSFORMER_HEADS} heads",
        flush=True,
    )
    print(f"  epochs: {epochs}", flush=True)
    print(f"  batch size: {BATCH_SIZE} identities", flush=True)
    print("  training heads: geometry only (registered FLAME supervision)", flush=True)
    print(f"  pretrained MobileNet backbone: {USE_PRETRAINED_BACKBONE}", flush=True)
    print(f"  progress: every {PROGRESS_EVERY} batch", flush=True)
    print(
        f"  heartbeat: every {HEARTBEAT_SECONDS}s during silent trainer startup",
        flush=True,
    )
    print(
        f"  numbered checkpoints: every {CHECKPOINT_EVERY} epochs",
        flush=True,
    )
    print(f"  persistent outputs: {run_dir}", flush=True)
    if resume_source is None:
        print("  resume: no checkpoint found; starting fresh", flush=True)
    else:
        print(f"  resume source: {resume_source}", flush=True)
        print(f"  resume local: {local_resume}", flush=True)
    print("", flush=True)


def train_and_export(
    dataset_root: Path,
    *,
    smoke: bool,
) -> tuple[Path, Path | None]:
    run_dir = SMOKE_RUN_DIR if smoke else REAL_RUN_DIR
    run_dir.mkdir(parents=True, exist_ok=True)
    drive_resume = find_resume_checkpoint(run_dir)
    local_resume = stage_resume_checkpoint(drive_resume)
    print_run_configuration(
        dataset_root,
        run_dir,
        drive_resume,
        local_resume,
        smoke=smoke,
    )

    run_with_heartbeat(
        training_command(
            dataset_root,
            run_dir,
            local_resume,
            smoke=smoke,
        ),
        cwd=REPO,
    )

    onnx_path = run_dir / "headscan_lite.onnx"
    try:
        run([
            sys.executable,
            "-m",
            "head_model.export",
            "--checkpoint",
            str(run_dir / "best.pt"),
            "--output",
            str(onnx_path),
        ], cwd=REPO)
    except subprocess.CalledProcessError as exc:
        (run_dir / "onnx_export_error.txt").write_text(
            str(exc),
            encoding="utf-8",
        )
        print(
            "[export] ONNX export failed; checkpoints and metrics remain safely in Drive.",
            flush=True,
        )
        onnx_path = None

    print(f"Best checkpoint: {run_dir / 'best.pt'}", flush=True)
    print(f"Latest checkpoint: {run_dir / 'last.pt'}", flush=True)
    print(f"Shape basis: {run_dir / 'shape_basis.npz'}", flush=True)
    print(f"Metrics: {run_dir / 'metrics.jsonl'}", flush=True)
    if onnx_path is not None:
        print(f"ONNX model: {onnx_path}", flush=True)
    return run_dir, onnx_path


def package_and_download(run_dir: Path, *, smoke: bool) -> None:
    name = "headscan-lite-smoke" if smoke else "headscan-lite-headspace"
    zip_path = WORK / f"{name}.zip"
    if zip_path.exists():
        zip_path.unlink()
    shutil.make_archive(
        str(zip_path.with_suffix("")),
        "zip",
        root_dir=run_dir,
    )
    print(
        f"Packaged run outputs: {zip_path} "
        f"({zip_path.stat().st_size/1024/1024:.1f} MiB)",
        flush=True,
    )
    try:
        from google.colab import files as colab_files

        colab_files.download(str(zip_path))
    except Exception:
        print(
            f"Automatic download unavailable. Outputs remain in Drive at {run_dir}",
            flush=True,
        )


def main() -> None:
    print(
        f"HeadScan-Lite Colab version: {HEADSCAN_COLAB_VERSION}",
        flush=True,
    )
    print(
        "Headspace/LYHM mode: licensed non-commercial research/education data only.",
        flush=True,
    )
    require_gpu()
    mount_drive()
    clone_repo()
    install_dependencies()
    dataset_root, smoke = prepare_headspace()
    run_dir, _ = train_and_export(dataset_root, smoke=smoke)
    package_and_download(run_dir, smoke=smoke)


if __name__ == "__main__":
    main()
