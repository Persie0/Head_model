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
import importlib
import importlib.util
import json
import queue
import threading
import shutil
import subprocess
import sys
import time
import zipfile

HEADSCAN_COLAB_VERSION = "2026-10-16-v5-progress-cache"
# Kept so already-open v3 notebook bootstrap cells accept this compatible update.
COLAB_COMPATIBILITY_TOKENS = ("v3-makehuman", "v4-makehuman-import-fix")

# Free-Colab-oriented defaults. Increase SYNTH_IDENTITIES on later runs after
# validating the pipeline. Changing it automatically uses a different cache.
SYNTH_IDENTITIES = 2500
SYNTH_VIEWS = 8
SYNTH_IMAGE_SIZE = 320
SYNTH_VALIDATION_FRACTION = 0.10
SYNTH_SEED = 42
SYNTH_SNAPSHOT_EVERY = 500  # Restartable intermediate ZIP every 500 identities.

IMAGE_SIZE = 256
MAX_VIEWS = 8
MIN_VIEWS = 1
SHAPE_COMPONENTS = 128
TOKEN_DIM = 256
TRANSFORMER_LAYERS = 3
TRANSFORMER_HEADS = 4
EPOCHS = 35
BATCH_SIZE = 2
PROGRESS_EVERY = 25  # One line per 25 batches, plus the final batch of each phase.
CHECKPOINT_EVERY = 5
HEARTBEAT_SECONDS = 30

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
MAKEHUMAN_PARTIAL_ZIP = DRIVE_ROOT / f"makehuman_synth_{CACHE_TAG}.partial.zip"
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
    """Stream child output immediately, and diagnose genuinely silent stages.

    Unlike Popen with inherited output, this relays the trainer's individual
    lines through the active notebook process. Heartbeat resets after every
    real trainer line instead of printing useless five-second repetitions.
    """
    print("+", " ".join(map(str, command)), flush=True)
    started = time.monotonic()
    process = subprocess.Popen(
        command,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    output: queue.Queue[str | None] = queue.Queue()

    def reader() -> None:
        assert process.stdout is not None
        try:
            for line in process.stdout:
                output.put(line.rstrip("\n"))
        finally:
            output.put(None)

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    last_line_at = started
    last_stage = "waiting for process to emit first line"
    interval = max(5.0, float(heartbeat_seconds))
    next_heartbeat = started + interval

    while True:
        try:
            item = output.get(timeout=0.5)
        except queue.Empty:
            item = False

        if item is None:
            break
        if item is not False:
            print(item, flush=True)
            last_line_at = time.monotonic()
            if item.strip():
                last_stage = str(item).strip()[:160]
            next_heartbeat = last_line_at + interval

        now = time.monotonic()
        if now >= next_heartbeat and process.poll() is None:
            rss = "?"
            try:
                status_file = Path(f"/proc/{process.pid}/status")
                for line in status_file.read_text(encoding="utf-8").splitlines():
                    if line.startswith("VmRSS:"):
                        rss = line.partition(":")[2].strip()
                        break
            except OSError:
                pass
            print(
                f"[trainer-heartbeat] no child output for "
                f"{now-last_line_at:.0f}s | elapsed {(now-started)/60:.1f}m | "
                f"pid={process.pid} | RAM={rss} | last: {last_stage}",
                flush=True,
            )
            next_heartbeat = now + interval

    code = process.wait()
    thread.join(timeout=2)
    if code:
        raise subprocess.CalledProcessError(code, command)
    print(
        f"[trainer] process completed successfully in "
        f"{(time.monotonic()-started)/60:.1f} min",
        flush=True,
    )


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


def activate_repo_import_path() -> None:
    """Make the editable src-layout package visible in the live Colab kernel.

    pip is launched in a subprocess. For a PEP 660 editable install it writes a
    .pth file into site-packages, but Python processes .pth files only during
    interpreter startup. The notebook kernel is already running, so explicitly
    prepend the repository's src directory instead of requiring a runtime
    restart.
    """
    src_dir = (REPO / "src").resolve()
    package_dir = src_dir / "head_model"
    if not package_dir.is_dir():
        raise RuntimeError(
            f"Expected Head_model package directory is missing: {package_dir}"
        )
    src_text = str(src_dir)
    if src_text not in sys.path:
        sys.path.insert(0, src_text)
    importlib.invalidate_caches()

    spec = importlib.util.find_spec("head_model")
    if spec is None:
        raise RuntimeError(
            "Head_model was installed but the live Colab kernel still cannot "
            f"resolve it. Added source path: {src_dir}. sys.path={sys.path[:6]}"
        )
    print(
        f"[python-path] head_model import ready: {spec.origin}",
        flush=True,
    )


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
    activate_repo_import_path()


def _prepared_ok(root: Path) -> bool:
    """Check complete restored cache, including settings and sample files."""
    try:
        required = ("train.jsonl", "valid.jsonl", "makehuman_generation.json", "head_faces.npy")
        if any(not (root / name).is_file() for name in required):
            return False
        report = json.loads((root / "makehuman_generation.json").read_text(encoding="utf-8"))
        if (
            report.get("identities") != SYNTH_IDENTITIES
            or report.get("views_per_identity") != SYNTH_VIEWS
            or report.get("image_size") != SYNTH_IMAGE_SIZE
        ):
            return False
        for split in ("train", "valid"):
            path = root / f"{split}.jsonl"
            with path.open("r", encoding="utf-8") as handle:
                first = next((json.loads(line) for line in handle if line.strip()), None)
            if first is None:
                return False
            files = [root / first["vertices"]]
            for view in first["views"]:
                files.append(root / view["image"])
                files.append(root / view["mask"])
            if any(not item.is_file() or item.stat().st_size == 0 for item in files):
                return False
        return True
    except (OSError, KeyError, TypeError, ValueError, StopIteration):
        return False


def _zip_local_dataset(root: Path, destination: Path, label: str) -> None:
    """Write a restart-safe ZIP with explicit packing progress, without extra copies."""
    started = time.monotonic()
    files = [path for path in root.rglob("*") if path.is_file()]
    total_files = len(files)
    total_bytes = sum(path.stat().st_size for path in files)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    if temporary.exists():
        temporary.unlink()
    print(
        f"[{label}] packing {total_files} files | "
        f"{total_bytes/1024/1024:.1f} MiB source",
        flush=True,
    )
    try:
        with zipfile.ZipFile(
            temporary, "w", compression=zipfile.ZIP_DEFLATED,
            compresslevel=1, allowZip64=True,
        ) as archive:
            for i, path in enumerate(files, 1):
                archive.write(path, arcname=str(path.relative_to(root)))
                if i % 1000 == 0 or i == total_files:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    eta = elapsed / i * (total_files - i)
                    print(
                        f"[{label}] {i}/{total_files} files "
                        f"({100*i/max(1,total_files):.1f}%) | "
                        f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
                        flush=True,
                    )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    print(
        f"[{label}] ZIP ready: {destination.stat().st_size/1024/1024:.1f} MiB",
        flush=True,
    )


def _atomic_drive_copy(source: Path, destination: Path, label: str) -> None:
    """Never replace a good Drive cache with a half-copied ZIP."""
    temp = destination.with_suffix(destination.suffix + ".uploading")
    try:
        copy_with_progress(source, temp, label)
        if temp.stat().st_size != source.stat().st_size:
            raise IOError(f"Incomplete cache upload: {temp}")
        temp.replace(destination)
    except BaseException:
        temp.unlink(missing_ok=True)
        raise
    print(f"[{label}] Drive ZIP saved atomically: {destination}", flush=True)


def _restore_archive(source: Path, *, complete: bool) -> Path | None:
    if not source.is_file():
        return None
    if LOCAL_PREPARED.exists():
        shutil.rmtree(LOCAL_PREPARED)
    LOCAL_PREPARED.mkdir(parents=True, exist_ok=True)
    local_zip = WORK / source.name
    print(
        f"[dataset-cache] Found {'complete' if complete else 'partial'} "
        f"Drive ZIP: {source}",
        flush=True,
    )
    try:
        copy_with_progress(source, local_zip, "dataset-cache-copy")
        started = time.monotonic()
        with zipfile.ZipFile(local_zip) as archive:
            # archive.extract validates CRC while writing, so avoid reading
            # the entire 500 MiB ZIP once with testzip() before extracting.
            members = archive.infolist()
            total = len(members)
            for i, member in enumerate(members, 1):
                target = (LOCAL_PREPARED / member.filename).resolve()
                if not target.is_relative_to(LOCAL_PREPARED.resolve()):
                    raise zipfile.BadZipFile(f"unsafe ZIP path: {member.filename}")
                archive.extract(member, LOCAL_PREPARED)
                if i % 2000 == 0 or i == total:
                    elapsed = max(time.monotonic() - started, 1e-6)
                    eta = elapsed * (total - i) / max(1, i)
                    print(
                        f"[dataset-cache-restore] {i}/{total} files "
                        f"({100*i/max(1,total):.1f}%) | "
                        f"elapsed {elapsed:.0f}s | ETA {eta:.0f}s",
                        flush=True,
                    )
        if complete:
            if not _prepared_ok(LOCAL_PREPARED):
                raise ValueError("restored full dataset does not match expected settings/files")
            report = json.loads(
                (LOCAL_PREPARED / "makehuman_generation.json").read_text(encoding="utf-8")
            )
            print(
                f"[dataset-cache] RESTORED complete dataset: "
                f"{report['identities']} identities, "
                f"{report['train_subjects']} train, "
                f"{report['valid_subjects']} validation | "
                "no regeneration required",
                flush=True,
            )
        else:
            progress_path = LOCAL_PREPARED / ".partial-progress.json"
            if not progress_path.is_file():
                raise ValueError("partial dataset snapshot lacks progress metadata")
            data = json.loads(progress_path.read_text(encoding="utf-8"))
            if data.get("cache_tag") != CACHE_TAG or data.get("seed") != SYNTH_SEED:
                raise ValueError("partial dataset settings mismatch")
            completed = len(list((LOCAL_PREPARED / "records").glob("mh_*.json")))
            if completed == 0:
                raise ValueError("partial snapshot contains no completed identities")
            print(
                f"[dataset-cache] RESTORED partial dataset: "
                f"{completed}/{SYNTH_IDENTITIES} record(s) found; "
                "generation will skip complete restored identities",
                flush=True,
            )
        return LOCAL_PREPARED
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        print(
            f"[dataset-cache] restore rejected ({type(exc).__name__}: {exc}); "
            "will use the next available source",
            flush=True,
        )
        shutil.rmtree(LOCAL_PREPARED, ignore_errors=True)
        return None
    finally:
        local_zip.unlink(missing_ok=True)


def restore_prepared_cache() -> Path | None:
    return _restore_archive(MAKEHUMAN_CACHE_ZIP, complete=True)


def restore_partial_cache() -> Path | None:
    return _restore_archive(MAKEHUMAN_PARTIAL_ZIP, complete=False)


def save_partial_cache(root: Path, completed: int, total: int) -> None:
    (root / ".partial-progress.json").write_text(
        json.dumps(
            {"cache_tag": CACHE_TAG, "seed": SYNTH_SEED, "completed": completed, "total": total}
        ),
        encoding="utf-8",
    )
    local_zip = WORK / f"makehuman_synth_{CACHE_TAG}.partial.zip"
    _zip_local_dataset(root, local_zip, "dataset-snapshot-pack")
    _atomic_drive_copy(local_zip, MAKEHUMAN_PARTIAL_ZIP, "dataset-snapshot-save")
    local_zip.unlink(missing_ok=True)
    print(
        f"[dataset-snapshot] {completed}/{total} identities backed up; "
        "a new Colab session can resume generation",
        flush=True,
    )


def save_prepared_cache(root: Path) -> None:
    local_zip = WORK / MAKEHUMAN_CACHE_ZIP.name
    print("[dataset-cache] packing completed synthetic dataset for later sessions...", flush=True)
    _zip_local_dataset(root, local_zip, "dataset-cache-pack")
    _atomic_drive_copy(local_zip, MAKEHUMAN_CACHE_ZIP, "dataset-cache-save")
    local_zip.unlink(missing_ok=True)
    # The full ZIP is now safely committed. No need to retain a stale partial.
    MAKEHUMAN_PARTIAL_ZIP.unlink(missing_ok=True)
    print(
        f"[dataset-cache] complete dataset persisted in Drive: {MAKEHUMAN_CACHE_ZIP}",
        flush=True,
    )


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

    # If a Colab runtime was interrupted mid-generation, restore the latest
    # safely-uploaded partial ZIP and continue from its per-identity records.
    partial = restore_partial_cache()
    if partial is not None:
        print("[dataset-cache] continuing interrupted synthesis from partial ZIP", flush=True)
    else:
        if LOCAL_PREPARED.exists():
            shutil.rmtree(LOCAL_PREPARED)
        LOCAL_PREPARED.mkdir(parents=True, exist_ok=True)

    source_root, commit = clone_makehuman_cc0_assets()
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
        snapshot_every=SYNTH_SNAPSHOT_EVERY,
        on_snapshot=save_partial_cache,
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
    print(f"  progress: every {PROGRESS_EVERY} training batch", flush=True)
    print(f"  dataset snapshot: every {SYNTH_SNAPSHOT_EVERY} generated identities", flush=True)
    print(f"  complete data cache: {MAKEHUMAN_CACHE_ZIP}", flush=True)
    print(f"  partial data cache: {MAKEHUMAN_PARTIAL_ZIP}", flush=True)
    print(
        f"  heartbeat: every {HEARTBEAT_SECONDS}s of actual child silence",
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
