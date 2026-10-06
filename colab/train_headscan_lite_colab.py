"""Google Colab recipe for HeadScan-Lite variable-view 3D head reconstruction.

Real data:
  MyDrive/head_model/headscan_dataset.zip
  -> archive containing train.jsonl + valid.jsonl + referenced files.

If no real dataset is found, a tiny procedural smoke dataset is generated. That
only validates the pipeline; it does not create a useful real-world model.
"""

from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import time
import zipfile

HEADSCAN_COLAB_VERSION = "2026-10-06-v1"

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
USE_PRETRAINED_BACKBONE = False

ALLOW_SYNTHETIC_SMOKE_TEST = True
SMOKE_EPOCHS = 2
SMOKE_IMAGE_SIZE = 128
SMOKE_COMPONENTS = 8

WORK = Path("/content/headscan_lite")
REPO = WORK / "Head_model"
LOCAL_DATASET = WORK / "dataset"
LOCAL_RESUME = WORK / "resume-last.pt"
DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive" / "head_model"
DATASET_ZIP = DRIVE_ROOT / "headscan_dataset.zip"
DATASET_DIR = DRIVE_ROOT / "dataset"
REAL_RUN_DIR = DRIVE_ROOT / "headscan-lite"
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
                raise subprocess.CalledProcessError(
                    code,
                    command,
                )
            return
        now = time.monotonic()
        if now >= next_heartbeat:
            print(
                f"[trainer] process alive | waiting for next trainer log line | "
                f"elapsed {now-started:.0f}s",
                flush=True,
            )
            next_heartbeat = now + max(
                1.0,
                heartbeat_seconds,
            )
        time.sleep(0.5)


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
        raise RuntimeError(
            "This recipe is intended for Google Colab."
        ) from exc
    drive.mount(
        str(DRIVE_MOUNT),
        force_remount=False,
    )
    DRIVE_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )
    print(
        f"Persistent Drive root: {DRIVE_ROOT}",
        flush=True,
    )


def clone_repo() -> None:
    WORK.mkdir(
        parents=True,
        exist_ok=True,
    )
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
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "--upgrade",
        "pip",
    ])
    run([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "-e",
        f"{REPO}[train,export]",
    ])


def _contains_manifests(root: Path) -> bool:
    return (
        (root / "train.jsonl").is_file()
        and (root / "valid.jsonl").is_file()
    )


def _find_dataset_root(root: Path) -> Path:
    if _contains_manifests(root):
        return root
    candidates = sorted({
        path.parent
        for path in root.rglob("train.jsonl")
        if (path.parent / "valid.jsonl").is_file()
    })
    if len(candidates) != 1:
        raise RuntimeError(
            "Could not uniquely locate train.jsonl + valid.jsonl "
            f"under {root}; found {candidates}"
        )
    return candidates[0]


def prepare_dataset() -> tuple[Path, bool]:
    if LOCAL_DATASET.exists():
        shutil.rmtree(LOCAL_DATASET)
    LOCAL_DATASET.mkdir(
        parents=True,
        exist_ok=True,
    )

    if DATASET_ZIP.is_file():
        size = DATASET_ZIP.stat().st_size
        print(
            f"[dataset] found {DATASET_ZIP} "
            f"({size/1024/1024:.1f} MiB)",
            flush=True,
        )
        local_zip = WORK / "headscan_dataset.zip"
        print(
            "[dataset] copying archive from Drive "
            "to local Colab storage...",
            flush=True,
        )
        shutil.copy2(
            DATASET_ZIP,
            local_zip,
        )
        print(
            "[dataset] extracting...",
            flush=True,
        )
        with zipfile.ZipFile(local_zip) as archive:
            archive.extractall(LOCAL_DATASET)
        root = _find_dataset_root(LOCAL_DATASET)
        print(
            f"[dataset] real dataset ready: {root}",
            flush=True,
        )
        return root, False

    if _contains_manifests(DATASET_DIR):
        print(
            f"[dataset] found unpacked Drive dataset: "
            f"{DATASET_DIR}",
            flush=True,
        )
        print(
            "[dataset] using Drive files directly. For faster "
            f"training, zip it as {DATASET_ZIP.name} next time.",
            flush=True,
        )
        return DATASET_DIR, False

    if not ALLOW_SYNTHETIC_SMOKE_TEST:
        raise FileNotFoundError(
            f"No dataset found. Expected {DATASET_ZIP} "
            f"or {DATASET_DIR}/train.jsonl"
        )

    print("", flush=True)
    print("=" * 78, flush=True)
    print(
        "[SMOKE TEST] NO REAL HEAD DATASET FOUND",
        flush=True,
    )
    print(
        "[SMOKE TEST] Generating a tiny procedural dataset "
        "only to verify training/checkpoint/export.",
        flush=True,
    )
    print(
        "[SMOKE TEST] THE RESULTING MODEL IS NOT "
        "A REAL HEAD SCANNER.",
        flush=True,
    )
    print("=" * 78, flush=True)

    from head_model.data import create_synthetic_smoke_dataset

    create_synthetic_smoke_dataset(
        LOCAL_DATASET,
        train_subjects=16,
        val_subjects=4,
        views_per_subject=8,
        image_size=160,
    )
    return LOCAL_DATASET, True


def find_resume_checkpoint(
    run_dir: Path,
) -> Path | None:
    checkpoint = run_dir / "last.pt"
    if (
        checkpoint.is_file()
        and checkpoint.stat().st_size > 0
    ):
        return checkpoint
    return None


def stage_resume_checkpoint(
    source: Path | None,
    destination: Path = LOCAL_RESUME,
) -> Path | None:
    if source is None:
        if destination.exists():
            destination.unlink()
        return None

    destination.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if destination.exists():
        destination.unlink()

    total = source.stat().st_size
    print(
        f"[resume-copy] staging {source} -> {destination} "
        f"({total/1024/1024:.1f} MiB)",
        flush=True,
    )
    copied = 0
    report_step = max(
        8 * 1024 * 1024,
        total // 10 if total else 1,
    )
    next_report = report_step
    with source.open("rb") as src, destination.open("wb") as dst:
        while True:
            chunk = src.read(8 * 1024 * 1024)
            if not chunk:
                break
            dst.write(chunk)
            copied += len(chunk)
            if copied >= next_report or copied == total:
                print(
                    f"[resume-copy] "
                    f"{copied/1024/1024:.1f}/"
                    f"{total/1024/1024:.1f} MiB "
                    f"({100.0*copied/max(1,total):.0f}%)",
                    flush=True,
                )
                next_report = copied + report_step
    print(
        "[resume-copy] local checkpoint ready.",
        flush=True,
    )
    return destination


def training_command(
    dataset_root: Path,
    run_dir: Path,
    resume: Path | None,
    *,
    smoke: bool,
) -> list[str]:
    image_size = (
        SMOKE_IMAGE_SIZE if smoke else IMAGE_SIZE
    )
    epochs = (
        SMOKE_EPOCHS if smoke else EPOCHS
    )
    components = (
        SMOKE_COMPONENTS
        if smoke
        else SHAPE_COMPONENTS
    )

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
    ]
    if USE_PRETRAINED_BACKBONE:
        command.append("--pretrained-backbone")
    if resume is not None:
        command.extend([
            "--resume",
            str(resume),
        ])
    return command


def print_run_configuration(
    dataset_root: Path,
    run_dir: Path,
    resume_source: Path | None,
    local_resume: Path | None,
    *,
    smoke: bool,
) -> None:
    image_size = (
        SMOKE_IMAGE_SIZE if smoke else IMAGE_SIZE
    )
    epochs = (
        SMOKE_EPOCHS if smoke else EPOCHS
    )
    components = (
        SMOKE_COMPONENTS
        if smoke
        else SHAPE_COMPONENTS
    )

    print(
        "\nHeadScan-Lite training configuration",
        flush=True,
    )
    print(
        "  model: MobileNetV3-Small + variable-view "
        "Transformer + PCA mesh prior",
        flush=True,
    )
    print(
        f"  dataset: {dataset_root}",
        flush=True,
    )
    print(
        f"  mode: "
        f"{'SYNTHETIC SMOKE TEST' if smoke else 'REAL TRAINING'}",
        flush=True,
    )
    print(
        f"  input: {image_size}x{image_size}",
        flush=True,
    )
    print(
        f"  views: {MIN_VIEWS}..{MAX_VIEWS} per identity",
        flush=True,
    )
    print(
        f"  PCA components: up to {components}",
        flush=True,
    )
    print(
        f"  token dim: {TOKEN_DIM}",
        flush=True,
    )
    print(
        f"  transformer: {TRANSFORMER_LAYERS} layers / "
        f"{TRANSFORMER_HEADS} heads",
        flush=True,
    )
    print(
        f"  epochs: {epochs}",
        flush=True,
    )
    print(
        f"  batch size: {BATCH_SIZE} identities",
        flush=True,
    )
    print(
        f"  progress: every {PROGRESS_EVERY} batch",
        flush=True,
    )
    print(
        f"  heartbeat: every {HEARTBEAT_SECONDS}s during "
        "silent trainer startup",
        flush=True,
    )
    print(
        f"  numbered checkpoints: every "
        f"{CHECKPOINT_EVERY} epochs",
        flush=True,
    )
    print(
        f"  pretrained backbone: {USE_PRETRAINED_BACKBONE}",
        flush=True,
    )
    print(
        f"  persistent outputs: {run_dir}",
        flush=True,
    )
    if resume_source is None:
        print(
            "  resume: no checkpoint found; starting fresh",
            flush=True,
        )
    else:
        print(
            f"  resume source: {resume_source}",
            flush=True,
        )
        print(
            f"  resume local: {local_resume}",
            flush=True,
        )
    print("", flush=True)


def train_and_export(
    dataset_root: Path,
    *,
    smoke: bool,
) -> tuple[Path, Path | None]:
    run_dir = (
        SMOKE_RUN_DIR if smoke else REAL_RUN_DIR
    )
    run_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    drive_resume = find_resume_checkpoint(
        run_dir
    )
    local_resume = stage_resume_checkpoint(
        drive_resume
    )
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

    onnx_path = (
        run_dir / "headscan_lite.onnx"
    )
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
        (
            run_dir
            / "onnx_export_error.txt"
        ).write_text(
            str(exc),
            encoding="utf-8",
        )
        print(
            "[export] ONNX export failed; checkpoints and "
            "metrics remain safely in Drive.",
            flush=True,
        )
        onnx_path = None

    print(
        f"Best checkpoint: {run_dir / 'best.pt'}",
        flush=True,
    )
    print(
        f"Latest checkpoint: {run_dir / 'last.pt'}",
        flush=True,
    )
    print(
        f"Shape basis: {run_dir / 'shape_basis.npz'}",
        flush=True,
    )
    print(
        f"Metrics: {run_dir / 'metrics.jsonl'}",
        flush=True,
    )
    if onnx_path is not None:
        print(
            f"ONNX model: {onnx_path}",
            flush=True,
        )
    return run_dir, onnx_path


def package_and_download(
    run_dir: Path,
    *,
    smoke: bool,
) -> None:
    name = (
        "headscan-lite-smoke"
        if smoke
        else "headscan-lite"
    )
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

        colab_files.download(
            str(zip_path)
        )
    except Exception:
        print(
            f"Automatic download unavailable. "
            f"Outputs remain in Drive at {run_dir}",
            flush=True,
        )


def main() -> None:
    print(
        f"HeadScan-Lite Colab version: "
        f"{HEADSCAN_COLAB_VERSION}",
        flush=True,
    )
    require_gpu()
    mount_drive()
    clone_repo()
    install_dependencies()
    dataset_root, smoke = prepare_dataset()
    run_dir, _ = train_and_export(
        dataset_root,
        smoke=smoke,
    )
    package_and_download(
        run_dir,
        smoke=smoke,
    )


if __name__ == "__main__":
    main()
