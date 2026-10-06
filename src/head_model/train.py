from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import RegisteredHeadDataset, build_shape_basis
from .model import HeadScanLite, load_shape_basis


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset, *, batch_size: int, shuffle: bool, workers: int, device: torch.device):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        persistent_workers=workers > 0,
    )


def masked_mean(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    while valid.ndim < value.ndim:
        valid = valid.unsqueeze(-1)
    valid = valid.to(dtype=value.dtype)
    denom = valid.expand_as(value).sum().clamp_min(1.0)
    return (value * valid).sum() / denom


def compute_losses(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    *,
    vertex_scale_mm: float,
    weights: dict[str, float],
) -> tuple[torch.Tensor, dict[str, float]]:
    target_vertices = batch["vertices"]
    target_coeff = batch["coeff_norm"]

    coeff_loss = torch.nn.functional.mse_loss(output["coeff_norm"], target_coeff)
    vertex_loss = torch.nn.functional.smooth_l1_loss(
        output["vertices"] / vertex_scale_mm,
        target_vertices / vertex_scale_mm,
    )

    zero = coeff_loss.new_zeros(())
    normal_loss = zero
    depth_loss = zero
    mask_loss = zero
    confidence_loss = zero

    # Headspace/LYHM training can run geometry-only because its licensed 3dMD
    # package supplies RGB views + registered FLAME targets, not our custom
    # dense supervision maps. Generic datasets may still provide them.
    if "normal" in output:
        normal_target = batch["normal"]
        normal_dot = (output["normal"] * normal_target).sum(dim=2).clamp(-1.0, 1.0)
        normal_valid = batch["normal_valid"].unsqueeze(-1).unsqueeze(-1)
        if "mask_valid" in batch:
            supervised_mask = torch.where(
                batch["mask_valid"].unsqueeze(-1).unsqueeze(-1),
                batch["mask"][:, :, 0] > 0.5,
                torch.ones_like(batch["mask"][:, :, 0], dtype=torch.bool),
            )
            normal_valid = normal_valid & supervised_mask
        normal_loss = masked_mean(1.0 - normal_dot, normal_valid)

        depth_valid = batch["depth_valid"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        depth_pixel_valid = depth_valid & (batch["depth"] > 0)
        depth_loss = masked_mean(
            torch.abs(output["depth"] - batch["depth"]),
            depth_pixel_valid,
        )

        mask_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            output["mask_logits"],
            batch["mask"],
            reduction="none",
        )
        mask_valid = batch["mask_valid"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        mask_loss = masked_mean(mask_bce, mask_valid)

        conf_bce = torch.nn.functional.binary_cross_entropy_with_logits(
            output["confidence_logits"],
            batch["confidence"],
            reduction="none",
        )
        conf_valid = batch["confidence_valid"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        confidence_loss = masked_mean(conf_bce, conf_valid)

    total = (
        weights["coeff"] * coeff_loss
        + weights["vertex"] * vertex_loss
        + weights["normal"] * normal_loss
        + weights["depth"] * depth_loss
        + weights["mask"] * mask_loss
        + weights["confidence"] * confidence_loss
    )
    pieces = {
        "loss": float(total.detach().item()),
        "coeff": float(coeff_loss.detach().item()),
        "vertex": float(vertex_loss.detach().item()),
        "normal": float(normal_loss.detach().item()),
        "depth": float(depth_loss.detach().item()),
        "mask": float(mask_loss.detach().item()),
        "confidence": float(confidence_loss.detach().item()),
    }
    return total, pieces


def move_batch(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def forward_batch(model: HeadScanLite, batch: dict, *, geometry_only: bool) -> dict[str, torch.Tensor]:
    if not geometry_only:
        return model(batch["images"], batch["view_angles"], batch["view_valid"])
    vertices, coeff_norm, view_quality = model.forward_geometry(
        batch["images"],
        batch["view_angles"],
        batch["view_valid"],
    )
    return {
        "vertices": vertices,
        "coeff_norm": coeff_norm,
        "view_quality": view_quality,
    }


@torch.no_grad()
def geometry_metrics(
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, float]:
    distance = torch.linalg.vector_norm(
        output["vertices"] - batch["vertices"],
        dim=-1,
    )
    coeff_rmse = torch.sqrt(
        torch.mean((output["coeff_norm"] - batch["coeff_norm"]) ** 2)
    )
    return {
        "vertex_mae_mm": float(distance.mean().item()),
        "vertex_p95_mm": float(torch.quantile(distance.flatten(), 0.95).item()),
        "coeff_rmse": float(coeff_rmse.item()),
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    weights: dict[str, float],
    vertex_scale_mm: float,
    progress_every: int,
    label: str = "validation",
    geometry_only: bool = False,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    total_batches = len(loader)
    started = time.perf_counter()
    for batch_index, batch in enumerate(loader, 1):
        batch = move_batch(batch, device)
        output = forward_batch(model, batch, geometry_only=geometry_only)
        _, pieces = compute_losses(
            output,
            batch,
            vertex_scale_mm=vertex_scale_mm,
            weights=weights,
        )
        metrics = geometry_metrics(output, batch)
        combined = {**pieces, **metrics}
        for key, value in combined.items():
            totals[key] = totals.get(key, 0.0) + value
        count += 1
        if progress_every > 0 and (
            batch_index % progress_every == 0
            or batch_index == total_batches
        ):
            current = {k: v / count for k, v in totals.items()}
            print(
                f"[{label}] batch {batch_index}/{total_batches} "
                f"({100.0 * batch_index / max(1,total_batches):.1f}%) | "
                f"loss {current['loss']:.4f} | "
                f"vertex {current['vertex_mae_mm']:.2f} mm | "
                f"p95 {current['vertex_p95_mm']:.2f} mm | "
                f"{time.perf_counter() - started:.1f}s",
                flush=True,
            )
    return {k: v / max(1, count) for k, v in totals.items()}


def save_checkpoint(
    path: Path,
    *,
    model: HeadScanLite,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    best_vertex_mae_mm: float,
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "architecture": "headscan_lite_mobilenetv3_small",
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "epoch": epoch,
            "best_vertex_mae_mm": best_vertex_mae_mm,
            "config": {
                "image_size": args.image_size,
                "max_views": args.max_views,
                "token_dim": args.token_dim,
                "transformer_layers": args.transformer_layers,
                "transformer_heads": args.transformer_heads,
                "depth_scale_mm": args.depth_scale_mm,
                "vertex_scale_mm": args.vertex_scale_mm,
                "basis_path": str(args.basis_path),
                "geometry_only_training": bool(args.geometry_only_training),
            },
        },
        path,
    )


def train(args: argparse.Namespace) -> dict[str, float]:
    seed_everything(args.seed)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not Path(args.basis_path).is_file():
        print(
            f"[basis] building PCA basis from {args.train_manifest} "
            f"with up to {args.shape_components} components",
            flush=True,
        )
        build_shape_basis(
            args.train_manifest,
            args.basis_path,
            components=args.shape_components,
        )
    basis = load_shape_basis(args.basis_path)
    print(
        f"[basis] vertices={basis.vertex_count} | "
        f"components={basis.component_count} | "
        f"path={args.basis_path}",
        flush=True,
    )

    train_dataset = RegisteredHeadDataset(
        args.train_manifest,
        args.basis_path,
        image_size=args.image_size,
        max_views=args.max_views,
        min_views=args.min_views,
        random_views=True,
        augment=True,
        depth_scale_mm=args.depth_scale_mm,
    )
    val_dataset = RegisteredHeadDataset(
        args.valid_manifest,
        args.basis_path,
        image_size=args.image_size,
        max_views=args.max_views,
        min_views=1,
        random_views=False,
        augment=False,
        depth_scale_mm=args.depth_scale_mm,
    )
    train_loader = make_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        workers=args.num_workers,
        device=device,
    )
    val_loader = make_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        workers=args.num_workers,
        device=device,
    )

    model = HeadScanLite(
        basis,
        token_dim=args.token_dim,
        transformer_layers=args.transformer_layers,
        transformer_heads=args.transformer_heads,
        pretrained_backbone=args.pretrained_backbone,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, args.epochs),
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    except (TypeError, AttributeError):
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    weights = {
        "coeff": args.coeff_weight,
        "vertex": args.vertex_weight,
        "normal": args.normal_weight,
        "depth": args.depth_weight,
        "mask": args.mask_weight,
        "confidence": args.confidence_weight,
    }

    start_epoch = 0
    best_vertex = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        if "optimizer" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer"])
        if "scheduler" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler"])
        if "scaler" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        best_vertex = float(
            checkpoint.get("best_vertex_mae_mm", float("inf"))
        )
        print(
            f"[resume] {args.resume} | "
            f"completed epoch {start_epoch}/{args.epochs} | "
            f"best vertex {best_vertex:.2f} mm",
            flush=True,
        )

    params = sum(p.numel() for p in model.parameters())
    trainable = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(
        f"[setup] device={device} | "
        f"train={len(train_dataset)} identities | "
        f"validation={len(val_dataset)} | "
        f"image={args.image_size} | "
        f"views=1..{args.max_views} | "
        f"batch={args.batch_size} | epochs={args.epochs}",
        flush=True,
    )
    print(
        f"[setup] parameters={params/1e6:.2f}M | "
        f"trainable={trainable/1e6:.2f}M | AMP={amp_enabled} | "
        f"geometry-only={args.geometry_only_training}",
        flush=True,
    )
    print(f"[setup] outputs={output_dir}", flush=True)
    print(
        f"[setup] progress every {args.progress_every} batch(es) | "
        f"checkpoint every {args.checkpoint_every} epoch(s)",
        flush=True,
    )

    metrics_path = output_dir / "metrics.jsonl"
    for epoch in range(start_epoch, args.epochs):
        completed = epoch + 1
        print(
            f"\n[epoch] {completed}/{args.epochs} starting",
            flush=True,
        )
        model.train()
        epoch_started = time.perf_counter()
        running_loss = 0.0
        running_vertex = 0.0
        batches = 0
        total_batches = len(train_loader)

        for batch_index, batch in enumerate(train_loader, 1):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                output = forward_batch(
                    model,
                    batch,
                    geometry_only=args.geometry_only_training,
                )
                loss, _ = compute_losses(
                    output,
                    batch,
                    vertex_scale_mm=args.vertex_scale_mm,
                    weights=weights,
                )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    args.grad_clip,
                )
            scaler.step(optimizer)
            scaler.update()

            metrics = geometry_metrics(output, batch)
            running_loss += float(loss.detach().item())
            running_vertex += metrics["vertex_mae_mm"]
            batches += 1

            if args.progress_every > 0 and (
                batch_index % args.progress_every == 0
                or batch_index == total_batches
            ):
                print(
                    f"[train] epoch {completed}/{args.epochs} | "
                    f"batch {batch_index}/{total_batches} "
                    f"({100.0 * batch_index / max(1,total_batches):.1f}%) | "
                    f"loss {running_loss/max(1,batches):.4f} | "
                    f"vertex {running_vertex/max(1,batches):.2f} mm | "
                    f"lr {optimizer.param_groups[0]['lr']:.2e} | "
                    f"{time.perf_counter()-epoch_started:.1f}s",
                    flush=True,
                )

        validation = evaluate(
            model,
            val_loader,
            device,
            weights=weights,
            vertex_scale_mm=args.vertex_scale_mm,
            progress_every=args.progress_every,
            geometry_only=args.geometry_only_training,
        )
        scheduler.step()
        elapsed = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "completed_epoch": completed,
            "train_loss": running_loss / max(1, batches),
            "train_vertex_mae_mm": running_vertex / max(1, batches),
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_seconds": elapsed,
            "validation": validation,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row) + "\n")

        current_vertex = validation["vertex_mae_mm"]
        if current_vertex < best_vertex:
            best_vertex = current_vertex
            save_checkpoint(
                output_dir / "best.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_vertex_mae_mm=best_vertex,
                args=args,
            )
            print(
                f"[checkpoint] best.pt updated | "
                f"vertex {best_vertex:.2f} mm",
                flush=True,
            )

        save_checkpoint(
            output_dir / "last.pt",
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            best_vertex_mae_mm=best_vertex,
            args=args,
        )
        print(
            f"[checkpoint] last.pt saved after epoch {completed}",
            flush=True,
        )

        if args.checkpoint_every > 0 and (
            completed % args.checkpoint_every == 0
            or completed == args.epochs
        ):
            periodic = (
                output_dir
                / "checkpoints"
                / f"epoch_{completed:03d}.pt"
            )
            save_checkpoint(
                periodic,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                epoch=epoch,
                best_vertex_mae_mm=best_vertex,
                args=args,
            )
            print(
                f"[checkpoint] saved {periodic}",
                flush=True,
            )

        print(
            f"[epoch] {completed}/{args.epochs} complete | "
            f"train loss {row['train_loss']:.4f} | "
            f"val loss {validation['loss']:.4f} | "
            f"vertex {validation['vertex_mae_mm']:.2f} mm | "
            f"p95 {validation['vertex_p95_mm']:.2f} mm | "
            f"{elapsed:.1f}s",
            flush=True,
        )

    summary = {
        "best_vertex_mae_mm": best_vertex,
        "basis_components": basis.component_count,
        "vertices": basis.vertex_count,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        f"[done] summary written to {output_dir / 'summary.json'}",
        flush=True,
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train HeadScan-Lite variable-view "
            "registered-head reconstruction"
        )
    )
    parser.add_argument(
        "--train-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--valid-manifest",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--basis-path",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("runs/headscan_lite"),
    )
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--max-views", type=int, default=8)
    parser.add_argument("--min-views", type=int, default=1)
    parser.add_argument("--shape-components", type=int, default=128)
    parser.add_argument("--token-dim", type=int, default=256)
    parser.add_argument("--transformer-layers", type=int, default=3)
    parser.add_argument("--transformer-heads", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--depth-scale-mm", type=float, default=250.0)
    parser.add_argument("--vertex-scale-mm", type=float, default=100.0)
    parser.add_argument("--coeff-weight", type=float, default=1.0)
    parser.add_argument("--vertex-weight", type=float, default=5.0)
    parser.add_argument("--normal-weight", type=float, default=1.0)
    parser.add_argument("--depth-weight", type=float, default=0.5)
    parser.add_argument("--mask-weight", type=float, default=0.5)
    parser.add_argument("--confidence-weight", type=float, default=0.25)
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=5)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument(
        "--geometry-only-training",
        action="store_true",
        help=(
            "Skip dense normal/depth/mask/confidence heads during training. "
            "Recommended for Headspace/LYHM, which supplies registered FLAME "
            "geometry plus RGB camera views but not these custom dense targets."
        ),
    )
    parser.add_argument(
        "--no-amp",
        dest="amp",
        action="store_false",
    )
    parser.set_defaults(amp=True)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = train(args)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
