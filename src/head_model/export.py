from __future__ import annotations

import argparse
import inspect
from pathlib import Path

import numpy as np
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
        return self.model.forward_geometry(images, view_angles, view_valid)


def load_from_checkpoint(
    path: str | Path,
) -> tuple[HeadScanLite, dict]:
    print(f"[export] loading trained checkpoint: {path}", flush=True)
    # Checkpoints written by HeadScan-Lite use tensors and primitive metadata,
    # so weights_only=True is compatible and avoids unpickling arbitrary objects.
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
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
        transformer_layers=int(config.get("transformer_layers", 3)),
        transformer_heads=int(config.get("transformer_heads", 4)),
        pretrained_backbone=False,
    )
    model.load_state_dict(state)
    model.eval()
    print(
        f"[export] model loaded | vertices={basis.vertex_count} | "
        f"PCA={basis.component_count} | "
        f"parameters={sum(p.numel() for p in model.parameters())/1e6:.2f}M",
        flush=True,
    )
    return model, config


def _validate_onnx(
    output: Path,
    wrapper: GeometryExport,
    inputs: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> None:
    import onnx
    import onnxruntime as ort

    print("[export] verifying ONNX graph with onnx.checker...", flush=True)
    onnx.checker.check_model(str(output))
    print("[export] checking ONNX Runtime against PyTorch outputs...", flush=True)

    # ORT is also the mobile deployment runtime; catch unsupported attention
    # operators and numerical discrepancies before advertising a successful ZIP.
    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        reference = tuple(t.detach().cpu().numpy() for t in wrapper(*inputs))
    actual = session.run(
        None,
        {
            "images": inputs[0].detach().cpu().numpy(),
            "view_angles": inputs[1].detach().cpu().numpy(),
            "view_valid": inputs[2].detach().cpu().numpy(),
        },
    )
    if len(actual) != len(reference):
        raise RuntimeError(
            f"Exported ONNX model has {len(actual)} outputs, expected {len(reference)}."
        )
    for name, expected, result in zip(
        ("vertices_mm", "shape_coefficients", "view_quality"),
        reference,
        actual,
    ):
        np.testing.assert_allclose(
            result, expected, rtol=5e-3, atol=5e-3,
            err_msg=f"ONNX Runtime parity failed for {name}",
        )
        print(
            f"[export] {name} | {result.shape} | "
            f"max abs diff {np.max(np.abs(result-expected)):.5g}",
            flush=True,
        )


def export_onnx(
    checkpoint: str | Path,
    output: str | Path,
    *,
    opset: int = 17,
    validate: bool = True,
) -> Path:
    model, config = load_from_checkpoint(checkpoint)
    wrapper = GeometryExport(model).eval()
    image_size = int(config.get("image_size", 256))
    max_views = int(config.get("max_views", 8))

    images = torch.zeros(1, max_views, 3, image_size, image_size)
    view_angles = torch.zeros(1, max_views, 2)
    view_valid = torch.ones(1, max_views, dtype=torch.bool)
    example_inputs = (images, view_angles, view_valid)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"[export] exporting opset {opset} | image={image_size}x{image_size} "
        f"| view slots={max_views}",
        flush=True,
    )
    # The new PyTorch exporter (default on modern Colab) depends on onnxscript,
    # and its torch.export-based treatment of Transformer masks varies across
    # versions. Use the established TorchScript ONNX exporter for opset 17.
    # Disable fused MHA so that ONNX sees portable standard attention ops.
    opts = {
        "input_names": ["images", "view_angles", "view_valid"],
        "output_names": ["vertices_mm", "shape_coefficients", "view_quality"],
        "dynamic_axes": {
            "images": {0: "batch"},
            "view_angles": {0: "batch"},
            "view_valid": {0: "batch"},
            "vertices_mm": {0: "batch"},
            "shape_coefficients": {0: "batch"},
            "view_quality": {0: "batch"},
        },
        "opset_version": opset,
        "do_constant_folding": True,
    }
    if "dynamo" in inspect.signature(torch.onnx.export).parameters:
        opts["dynamo"] = False
    old_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with torch.no_grad():
            torch.onnx.export(wrapper, example_inputs, str(output), **opts)
        print(f"[export] wrote ONNX ({output.stat().st_size / 1024 / 1024:.1f} MiB)", flush=True)
        if validate:
            _validate_onnx(output, wrapper, example_inputs)
    except BaseException:
        # Do not leave behind a misleading, incomplete model on failure.
        output.unlink(missing_ok=True)
        raise
    finally:
        torch.backends.mha.set_fastpath_enabled(old_fastpath)
    print("[export] verified ONNX model ready", flush=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export trained HeadScan-Lite geometry to ONNX (no retraining)"
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--opset", type=int, default=17)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    output = export_onnx(
        args.checkpoint, args.output, opset=args.opset,
        validate=not args.skip_validation,
    )
    print(
        f"[export] ONNX: {output} "
        f"({output.stat().st_size / 1024 / 1024:.1f} MiB)",
        flush=True,
    )


if __name__ == "__main__":
    main()
