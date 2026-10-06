"""Google Colab recipe for request-free HeadScan-Lite training.

Primary dataset:
  Synthetic registered heads generated directly from MakeHuman's official core
  base mesh + morph targets. MakeHuman documents these core assets as CC0.

No dataset request, account, API key, university approval, Blender, MakeHuman GUI
or manually downloaded training archive is required.

The Colab recipe:
  1. mounts Google Drive,
  2. clones this repository,
  3. restores a prepared synthetic cache if present,
  4. otherwise sparse-clones only the required MakeHuman asset directories,
  5. generates registered heads + 8 domain-randomized RGB views + masks,
  6. caches that prepared dataset in Drive,
  7. trains/resumes HeadScan-Lite,
  8. exports the geometry-only mobile graph to ONNX.
"""

from __future__ import annotations

from pathlib import Path
import json
import shutil
import subprocess
import sys
import time
import zipfile

HEADSCAN_COLAB_VERSION = "2026-10-06-v3-makehuman"

# Free-Colab-oriented defaults. Increase SYNTH_IDENTITIES on later runs after
# validating the pipeline. Changing it automatically uses a different cache.
SYNTH_IDENTITIES = 2500
SYNTH_VIEWS = 8
SYNTH_IMAGE_SIZE = 320
SYNTH_VALIDATION_FRACTION = 0.10
SYNTH_SEED = 42

IMAGE_SIZE = 256
MAX_VIEWS = 8
MIN_VIEWS = 1
SHAPE_COMPONENTS = 128
TOKEN_DIM = 256
TRANSFORMER_LAYERS = 3
TRANSFORMER_HEADS = 4
EPOCHS = 35
BATCH_SIZE = 2
PROGRESS_EVERY = 1
CHECKPOINT_EVERY = 5
HEARTBEAT_SECONDS = 5

# False is the cleanest licensing path: the model learns entirely from the CC0
# MakeHuman-derived synthetic dataset. Set True only if you intentionally want
# torchvision's ImageNet-pretrained MobileNetV3 weights and have reviewed their
# provenance/terms for your use case.
USE_PRETRAINED_BACKBONE = False

WORK = Path("/content/headscan_lite")
REPO = WORK / "Head_model"
MAKEHUMAN_ROOT = WORK / "makehuman_cc0"
LOCAL_PREPARED = WORK / "makehuman_prepared"
LOCAL_RESUME = WORK / "resume-last.pt"
DRIVE_MOUNT = Path("/content/drive")
DRIVE_ROOT = DRIVE_MOUNT / "MyDrive" / "head_model"

CACHE_TAG = f"v1_{SYNTH_IDENTITIES}ids_{SYNTH_VIEWS}views_{SYNTH_IMAGE_SIZE}px"
MAKEHUMAN_CACHE_ZIP = DRIVE_ROOT / f"makehuman_synth_{CACHE_TAG}.zip"
REAL_RUN_DIR = DRIVE_ROOT / f"headscan-lite-makehuman-{SYNTH_IDENTITIES}"


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
        and (root / "makehuman_generation.json").is_file()
        and (root / "head_faces.npy").is_file()
    )


def restore_prepared_cache() -> Path | None:
    if not MAKEHUMAN_CACHE_ZIP.is_file():
        return None
    if LOCAL_PREPARED.exists():
        shutil.rmtree(LOCAL_PREPARED)
    LOCAL_PREPARED.mkdir(parents=True, exist_ok=True)
    local_zip = WORK / MAKEHUMAN_CACHE_ZIP.name
    copy_with_progress(MAKEHUMAN_CACHE_ZIP, local_zip, "dataset-cache-copy")
    print("[dataset-cache] extracting prepared MakeHuman dataset...", flush=True)
    with zipfile.ZipFile(local_zip) as archive:
        archive.extractall(LOCAL_PREPARED)
    if not _prepared_ok(LOCAL_PREPARED):
        print("[dataset-cache] cache incomplete; rebuilding.", flush=True)
        shutil.rmtree(LOCAL_PREPARED)
        return None
    report = json.loads(
        (LOCAL_PREPARED / "makehuman_generation.json").read_text(encoding="utf-8")
    )
    print(
        f"[dataset-cache] ready | identities={report.get('identities')} | "
        f"train={report.get('train_subjects')} | valid={report.get('valid_subjects')}",
        flush=True,
    )
    return LOCAL_PREPARED


def save_prepared_cache(root: Path) -> None:
    local_zip = WORK / MAKEHUMAN_CACHE_ZIP.name
    if local_zip.exists():
        local_zip.unlink()
    print("[dataset-cache] packing generated dataset for later sessions...", flush=True)
    shutil.make_archive(
        str(local_zip.with_suffix("")),
        "zip",
        root_dir=root,
    )
    copy_with_progress(local_zip, MAKEHUMAN_CACHE_ZIP, "dataset-cache-save")
    print(f"[dataset-cache] saved: {MAKEHUMAN_CACHE_ZIP}", flush=True)


def clone_makehuman_cc0_assets() -> tuple[Path, str]:
    """Sparse-clone only official MakeHuman mesh/target asset directories."""
    if MAKEHUMAN_ROOT.exists():
        shutil.rmtree(MAKEHUMAN_ROOT)

    run([
        "git",
        "clone",
        "--depth",
        "1",
        "--filter=blob:none",
        "--sparse",
        "https://github.com/makehumancommunity/makehuman.git",
        str(MAKEHUMAN_ROOT),
    ])
    run([
        "git",
        "-C",
        str(MAKEHUMAN_ROOT),
        "sparse-checkout",
        "set",
        "makehuman/data/3dobjs",
        "makehuman/data/targets",
    ])
    commit = subprocess.check_output(
        ["git", "-C", str(MAKEHUMAN_ROOT), "rev-parse", "HEAD"],
        text=True,
    ).strip()

    license_path = MAKEHUMAN_ROOT / "LICENSE.md"
    if license_path.is_file():
        license_text = license_path.read_text(encoding="utf-8", errors="ignore").lower()
        if "cc0" not in license_text or "base mesh" not in license_text:
            raise RuntimeError(
                "MakeHuman repository license text did not contain the expected "
                "CC0 core-asset statement. Stop instead of assuming asset terms."
            )
    else:
        raise RuntimeError("MakeHuman LICENSE.md missing from sparse clone.")

    print(f"[makehuman] source commit: {commit}", flush=True)
    print(
        "[makehuman] using only core base mesh + targets documented as CC0; "
        "no MakeHuman application code or third-party community assets are used.",
        flush=True,
    )
    return MAKEHUMAN_ROOT, commit


def prepare_makehuman_dataset() -> Path:
    cached = restore_prepared_cache()
    if cached is not None:
        return cached

    source_root, commit = clone_makehuman_cc0_assets()
    if LOCAL_PREPARED.exists():
        shutil.rmtree(LOCAL_PREPARED)
    LOCAL_PREPARED.mkdir(parents=True, exist_ok=True)

    from head_model.makehuman_synth import generate_makehuman_dataset

    print("", flush=True)
    print("=" * 78, flush=True)
    print("[makehuman] GENERATING REQUEST-FREE SYNTHETIC TRAINING DATA", flush=True)
    print(
        f"[makehuman] identities={SYNTH_IDENTITIES} | views={SYNTH_VIEWS} | "
        f"render={SYNTH_IMAGE_SIZE}px",
        flush=True,
    )
    print(
        "[makehuman] This is synthetic pretraining. Real-world accuracy must be "
        "measured separately on real scans/photos.",
        flush=True,
    )
    print("=" * 78, flush=True)

    report = generate_makehuman_dataset(
        source_root,
        LOCAL_PREPARED,
        identities=SYNTH_IDENTITIES,
        views_per_identity=SYNTH_VIEWS,
        image_size=SYNTH_IMAGE_SIZE,
        validation_fraction=SYNTH_VALIDATION_FRACTION,
        seed=SYNTH_SEED,
        progress_every=25,
    )
    report["makehuman_commit"] = commit
    (LOCAL_PREPARED / "makehuman_generation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2), flush=True)
    save_prepared_cache(LOCAL_PREPARED)
    return LOCAL_PREPARED


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
) -> list[str]:
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
        str(IMAGE_SIZE),
        "--max-views",
        str(MAX_VIEWS),
        "--min-views",
        str(MIN_VIEWS),
        "--shape-components",
        str(SHAPE_COMPONENTS),
        "--token-dim",
        str(TOKEN_DIM),
        "--transformer-layers",
        str(TRANSFORMER_LAYERS),
        "--transformer-heads",
        str(TRANSFORMER_HEADS),
        "--epochs",
        str(EPOCHS),
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
        command.extend(["--resume", str(resume)])
    return command


def print_run_configuration(
    dataset_root: Path,
    resume_source: Path | None,
    local_resume: Path | None,
) -> None:
    print("\nHeadScan-Lite training configuration", flush=True)
    print("  dataset: MakeHuman core-asset synthetic heads (request-free)", flush=True)
    print(
        "  model: MobileNetV3-Small + variable-view Transformer + PCA head prior",
        flush=True,
    )
    print(f"  identities: {SYNTH_IDENTITIES}", flush=True)
    print(f"  prepared data: {dataset_root}", flush=True)
    print(f"  input: {IMAGE_SIZE}x{IMAGE_SIZE}", flush=True)
    print(f"  views: random 1..{MAX_VIEWS} from {SYNTH_VIEWS} rendered views", flush=True)
    print(f"  PCA components: up to {SHAPE_COMPONENTS}", flush=True)
    print(f"  token dim: {TOKEN_DIM}", flush=True)
    print(
        f"  transformer: {TRANSFORMER_LAYERS} layers / "
        f"{TRANSFORMER_HEADS} heads",
        flush=True,
    )
    print(f"  epochs: {EPOCHS}", flush=True)
    print(f"  batch size: {BATCH_SIZE} identities", flush=True)
    print(
        "  auxiliary supervision: synthetic foreground mask + confidence",
        flush=True,
    )
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
    print(f"  persistent outputs: {REAL_RUN_DIR}", flush=True)
    if resume_source is None:
        print("  resume: no checkpoint found; starting fresh", flush=True)
    else:
        print(f"  resume source: {resume_source}", flush=True)
        print(f"  resume local: {local_resume}", flush=True)
    print("", flush=True)


def train_and_export(dataset_root: Path) -> tuple[Path, Path | None]:
    REAL_RUN_DIR.mkdir(parents=True, exist_ok=True)
    drive_resume = find_resume_checkpoint(REAL_RUN_DIR)
    local_resume = stage_resume_checkpoint(drive_resume)
    print_run_configuration(dataset_root, drive_resume, local_resume)

    run_with_heartbeat(
        training_command(dataset_root, REAL_RUN_DIR, local_resume),
        cwd=REPO,
    )

    topology_source = dataset_root / "head_faces.npy"
    if topology_source.is_file():
        shutil.copy2(topology_source, REAL_RUN_DIR / "head_faces.npy")
        print(f"Head topology: {REAL_RUN_DIR / 'head_faces.npy'}", flush=True)

    onnx_path = REAL_RUN_DIR / "headscan_lite.onnx"
    try:
        run([
            sys.executable,
            "-m",
            "head_model.export",
            "--checkpoint",
            str(REAL_RUN_DIR / "best.pt"),
            "--output",
            str(onnx_path),
        ], cwd=REPO)
    except subprocess.CalledProcessError as exc:
        (REAL_RUN_DIR / "onnx_export_error.txt").write_text(
            str(exc),
            encoding="utf-8",
        )
        print(
            "[export] ONNX export failed; checkpoints and metrics remain safely in Drive.",
            flush=True,
        )
        onnx_path = None

    print(f"Best checkpoint: {REAL_RUN_DIR / 'best.pt'}", flush=True)
    print(f"Latest checkpoint: {REAL_RUN_DIR / 'last.pt'}", flush=True)
    print(f"Shape basis: {REAL_RUN_DIR / 'shape_basis.npz'}", flush=True)
    print(f"Metrics: {REAL_RUN_DIR / 'metrics.jsonl'}", flush=True)
    if onnx_path is not None:
        print(f"ONNX model: {onnx_path}", flush=True)
    return REAL_RUN_DIR, onnx_path


def package_and_download(run_dir: Path) -> None:
    zip_path = WORK / "headscan-lite-makehuman.zip"
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
        "Dataset mode: automatic MakeHuman CC0 synthetic generation; no request required.",
        flush=True,
    )
    require_gpu()
    mount_drive()
    clone_repo()
    install_dependencies()
    dataset_root = prepare_makehuman_dataset()
    run_dir, _ = train_and_export(dataset_root)
    package_and_download(run_dir)


if __name__ == "__main__":
    main()
