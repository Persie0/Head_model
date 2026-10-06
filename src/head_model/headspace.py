from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
import shutil
import tarfile
import zipfile

import numpy as np
from PIL import Image, ImageOps


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp"}
COLOR_CAMERA_RE = re.compile(r"^(?:\d+)?C\.(?:png|jpg|jpeg|bmp)$", re.IGNORECASE)
SUBJECT_RE = re.compile(r"(?<!\d)(\d{5})(?!\d)")


def read_obj_vertices(path: str | Path) -> np.ndarray:
    """Read only vertex positions from an OBJ without pulling in a mesh library."""
    vertices: list[tuple[float, float, float]] = []
    with Path(path).open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            if not line.startswith("v "):
                continue
            parts = line.split()
            if len(parts) >= 4:
                vertices.append((float(parts[1]), float(parts[2]), float(parts[3])))
    if not vertices:
        raise ValueError(f"no OBJ vertices found in {path}")
    return np.asarray(vertices, dtype=np.float32)


def vertices_to_mm(vertices: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize common FLAME/scan units to millimetres using head extent."""
    extent = float(np.ptp(vertices, axis=0).max())
    if extent <= 0:
        raise ValueError("degenerate mesh extent")
    if extent < 1.0:
        scale = 1000.0  # metres -> mm (typical FLAME/MICA registration scale)
    elif extent < 10.0:
        scale = 100.0
    elif extent < 100.0:
        scale = 10.0    # centimetres -> mm
    else:
        scale = 1.0     # already millimetres
    return (vertices * scale).astype(np.float32), scale


def _subject_id(path: Path) -> str | None:
    for part in reversed(path.parts):
        match = SUBJECT_RE.search(part)
        if match:
            return match.group(1)
    return None


def _wanted_member(name: str) -> bool:
    normalized = name.replace("\\", "/")
    lower = normalized.lower()
    if "/registrations/" in f"/{lower}" and lower.endswith(".obj"):
        return True
    base = Path(normalized).name
    return bool(COLOR_CAMERA_RE.match(base) and _subject_id(Path(normalized)))


def _safe_target(root: Path, member_name: str) -> Path:
    relative = Path(member_name.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"unsafe archive member: {member_name}")
    target = (root / relative).resolve()
    if root.resolve() not in target.parents and target != root.resolve():
        raise ValueError(f"archive member escapes target: {member_name}")
    return target


def extract_relevant_headspace_archives(
    source_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, int]:
    """Extract only FLAME registration OBJs and 3dMD color-camera images.

    ZIP archives are indexed before extraction. TAR/TAR.GZ archives are handled
    in one streaming pass so a 30+ GB source is not decompressed twice.
    """
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = [
        path for path in source_dir.rglob("*")
        if path.is_file()
        and path.name.lower().endswith((".zip", ".tar", ".tar.gz", ".tgz"))
    ]
    extracted = 0
    used_archives = 0

    for archive_path in sorted(archives):
        lower_name = archive_path.name.lower()
        if lower_name.endswith(".zip"):
            with zipfile.ZipFile(archive_path) as archive:
                wanted = [
                    info for info in archive.infolist()
                    if not info.is_dir() and _wanted_member(info.filename)
                ]
                if not wanted:
                    continue
                used_archives += 1
                print(
                    f"[headspace] {archive_path.name}: extracting "
                    f"{len(wanted)} relevant files",
                    flush=True,
                )
                for index, info in enumerate(wanted, 1):
                    target = _safe_target(output_dir, info.filename)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as src, target.open("wb") as dst:
                        shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                    extracted += 1
                    if index % 100 == 0 or index == len(wanted):
                        print(
                            f"[headspace] {archive_path.name}: "
                            f"{index}/{len(wanted)}",
                            flush=True,
                        )
            continue

        matched = 0
        with tarfile.open(archive_path, "r:*") as archive:
            for member in archive:
                if not member.isfile() or not _wanted_member(member.name):
                    continue
                if matched == 0:
                    used_archives += 1
                    print(
                        f"[headspace] {archive_path.name}: "
                        "streaming relevant files from TAR archive",
                        flush=True,
                    )
                target = _safe_target(output_dir, member.name)
                target.parent.mkdir(parents=True, exist_ok=True)
                src = archive.extractfile(member)
                if src is None:
                    continue
                with src, target.open("wb") as dst:
                    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                matched += 1
                extracted += 1
                if matched % 100 == 0:
                    print(
                        f"[headspace] {archive_path.name}: "
                        f"{matched} relevant files extracted",
                        flush=True,
                    )
        if matched:
            print(
                f"[headspace] {archive_path.name}: "
                f"{matched} relevant files extracted",
                flush=True,
            )

    return {"archives": used_archives, "files": extracted}


def find_registration_root(*roots: str | Path) -> Path:
    candidates: list[tuple[int, Path]] = []
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in [root, *root.rglob("registrations")]:
            if path.name != "registrations":
                continue
            actors = [p for p in path.iterdir() if p.is_dir() and any(p.glob("*.obj"))]
            if actors:
                candidates.append((len(actors), path))
    if not candidates:
        raise FileNotFoundError(
            "Could not find Headspace FLAME registrations. Expected a directory "
            "containing registrations/<actor_id>/*.obj."
        )
    candidates.sort(key=lambda item: (item[0], str(item[1])), reverse=True)
    return candidates[0][1]


def index_color_images(*roots: str | Path) -> dict[str, list[Path]]:
    by_subject: dict[str, list[Path]] = {}
    for root_value in roots:
        root = Path(root_value)
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not COLOR_CAMERA_RE.match(path.name):
                continue
            subject = _subject_id(path)
            if subject is None:
                continue
            by_subject.setdefault(subject, []).append(path)
    for subject in by_subject:
        # If a user keeps both an extracted package in Drive and the original
        # archive, extraction to /content can expose the same camera image twice.
        # Deduplicate by camera filename so repeated copies cannot consume view slots.
        deduped: dict[str, Path] = {}
        for path in sorted(by_subject[subject], key=lambda p: (p.name.lower(), str(p))):
            deduped.setdefault(path.name.lower(), path)
        by_subject[subject] = list(deduped.values())
    return by_subject


def _letterbox(image: Image.Image, size: int) -> Image.Image:
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((size, size), Image.Resampling.LANCZOS)
    background = Image.new("RGB", (size, size), (0, 0, 0))
    x = (size - image.width) // 2
    y = (size - image.height) // 2
    background.paste(image, (x, y))
    return background


def _deterministic_validation(subject: str, fraction: float) -> bool:
    digest = hashlib.sha1(subject.encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") / float(2**64)
    return value < fraction


def prepare_headspace_dataset(
    source_roots: list[str | Path],
    output_root: str | Path,
    *,
    image_size: int = 512,
    max_views: int = 8,
    validation_fraction: float = 0.10,
    min_subjects: int = 20,
) -> dict[str, object]:
    """Convert licensed Headspace/LYHM files into HeadScan-Lite manifests."""
    roots = [Path(root) for root in source_roots if Path(root).exists()]
    if not roots:
        raise FileNotFoundError("no Headspace source roots exist")

    registrations = find_registration_root(*roots)
    images = index_color_images(*roots)
    output_root = Path(output_root)
    vertices_dir = output_root / "vertices"
    images_dir = output_root / "images"
    vertices_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    actor_dirs = sorted(
        path for path in registrations.iterdir()
        if path.is_dir() and any(path.glob("*.obj"))
    )
    rows: list[dict[str, object]] = []
    vertex_count: int | None = None
    scales: dict[str, int] = {}
    missing_images: list[str] = []

    print(
        f"[headspace] registrations={registrations} | actors={len(actor_dirs)} | "
        f"subjects with color images={len(images)}",
        flush=True,
    )

    for actor_index, actor_dir in enumerate(actor_dirs, 1):
        subject = _subject_id(actor_dir) or actor_dir.name
        camera_images = images.get(subject, [])[:max_views]
        if not camera_images:
            missing_images.append(subject)
            continue

        obj_files = sorted(actor_dir.glob("*.obj"))
        vertices, scale = vertices_to_mm(read_obj_vertices(obj_files[0]))
        scales[f"x{scale:g}"] = scales.get(f"x{scale:g}", 0) + 1
        if vertex_count is None:
            vertex_count = int(vertices.shape[0])
        elif vertices.shape[0] != vertex_count:
            raise ValueError(
                f"Headspace registration topology mismatch for {subject}: "
                f"{vertices.shape[0]} vs expected {vertex_count}"
            )

        vertex_rel = Path("vertices") / f"{subject}.npy"
        np.save(output_root / vertex_rel, vertices)

        view_rows: list[dict[str, object]] = []
        subject_image_dir = images_dir / subject
        subject_image_dir.mkdir(parents=True, exist_ok=True)
        for view_index, source_image in enumerate(camera_images):
            dest = subject_image_dir / f"{view_index:02d}_{source_image.stem}.jpg"
            if not dest.is_file():
                with Image.open(source_image) as image:
                    _letterbox(image, image_size).save(dest, quality=94, subsampling=0)
            view_rows.append({
                "image": str(dest.relative_to(output_root)),
                # Headspace camera labels are kept in source_name. We intentionally
                # do not invent yaw/pitch calibration when it is not known.
                "source_name": source_image.name,
            })

        rows.append({
            "id": subject,
            "vertices": str(vertex_rel),
            "views": view_rows,
        })
        if actor_index % 100 == 0 or actor_index == len(actor_dirs):
            print(
                f"[headspace] converted {actor_index}/{len(actor_dirs)} registration actors | "
                f"usable={len(rows)}",
                flush=True,
            )

    if len(rows) < min_subjects:
        raise RuntimeError(
            f"Only {len(rows)} Headspace subjects had both a FLAME registration and "
            f"a color-camera image; expected at least {min_subjects}. "
            "Check that both the FLAME registrations package and the 3dMD PNG package "
            "are present/extracted under the configured Drive folder."
        )

    train_rows = [row for row in rows if not _deterministic_validation(str(row["id"]), validation_fraction)]
    valid_rows = [row for row in rows if _deterministic_validation(str(row["id"]), validation_fraction)]
    if not valid_rows and len(train_rows) > 1:
        valid_rows.append(train_rows.pop())

    for split, split_rows in (("train", train_rows), ("valid", valid_rows)):
        with (output_root / f"{split}.jsonl").open("w", encoding="utf-8") as handle:
            for row in split_rows:
                handle.write(json.dumps(row) + "\n")

    report = {
        "dataset": "Headspace/LYHM",
        "registration_root": str(registrations),
        "actors_with_registrations": len(actor_dirs),
        "usable_subjects": len(rows),
        "train_subjects": len(train_rows),
        "valid_subjects": len(valid_rows),
        "subjects_missing_color_images": len(missing_images),
        "vertex_count": vertex_count,
        "unit_scales": scales,
        "max_views": max((len(row["views"]) for row in rows), default=0),
        "image_size": image_size,
        "note": (
            "Camera yaw/pitch is intentionally omitted because Headspace camera labels "
            "are not assumed to encode calibrated angles. The fusion network therefore "
            "treats these source views as an unordered set."
        ),
    }
    (output_root / "headspace_conversion.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(
        f"[headspace] ready | train={len(train_rows)} | valid={len(valid_rows)} | "
        f"vertices={vertex_count} | max views found={report['max_views']}",
        flush=True,
    )
    return report
