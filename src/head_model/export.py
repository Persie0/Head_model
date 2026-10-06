from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn

from .model import HeadScanLite, ShapeBasis


class GeometryExport(nn.Module):
    def __init__(self, model: HeadScanLite) -> None:
        super().__init__()
        self.model = model

    def forward(
        self,
        images: torch.Tensor,
        view_angles: torch.Tensor,
        view_valid: torch.Tensor,
    ):
        return self.model.forward_geometry(
            images,
            view_angles,
            view_valid,
        )


def load_from_checkpoint(
    path: str | Path,
) -> tuple[HeadScanLite, dict]:
    checkpoint = torch.load(path, map_location="cpu")
    state = checkpoint["model"]
    basis = ShapeBasis(
        mean=state["shape_mean"].detach().cpu(),
        components=state["shape_components"].detach().cpu(),
        coeff_std=state["shape_coeff_std"].detach().cpu(),
    )
    config = checkpoint.get("config", {})
    model = HeadScanLite(
        basis,
        token_dim=int(config.get("token_dim", 256)),
        transformer_layers=int(
            config.get("transformer_layers", 3)
        ),
        transformer_heads=int(
            config.get("transformer_heads", 4)
        ),
        pretrained_backbone=False,
    )
    model.load_state_dict(state)
    model.eval()
    return model, config


def export_onnx(
    checkpoint: str | Path,
    output: str | Path,
    *,
    opset: int = 17,
) -> Path:
    model, config = load_from_checkpoint(checkpoint)
    wrapper = GeometryExport(model).eval()
    image_size = int(config.get("image_size", 256))
    max_views = int(config.get("max_views", 8))

    images = torch.zeros(
        1,
        max_views,
        3,
        image_size,
        image_size,
    )
    view_angles = torch.zeros(1, max_views, 2)
    view_valid = torch.ones(
        1,
        max_views,
        dtype=torch.bool,
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        wrapper,
        (images, view_angles, view_valid),
        str(output),
        input_names=[
            "images",
            "view_angles",
            "view_valid",
        ],
        output_names=[
            "vertices_mm",
            "shape_coefficients",
            "view_quality",
        ],
        dynamic_axes={
            "images": {0: "batch"},
            "view_angles": {0: "batch"},
            "view_valid": {0: "batch"},
            "vertices_mm": {0: "batch"},
            "shape_coefficients": {0: "batch"},
            "view_quality": {0: "batch"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export HeadScan-Lite geometry path to ONNX"
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
    )
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    output = export_onnx(
        args.checkpoint,
        args.output,
        opset=args.opset,
    )
    print(
        f"[export] ONNX: {output} "
        f"({output.stat().st_size / 1024 / 1024:.1f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
