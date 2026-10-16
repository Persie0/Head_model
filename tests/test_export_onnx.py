"""End-to-end export smoke test on a tiny registered-head model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from head_model.model import HeadScanLite, ShapeBasis
from head_model.export import export_onnx


def test_export_onnx_runtime_matches_pytorch(tmp_path):
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    torch.manual_seed(4)
    basis = ShapeBasis(
        mean=torch.randn(12, 3) * 30,
        components=torch.randn(4, 12, 3),
        coeff_std=torch.ones(4),
    )
    model = HeadScanLite(
        basis,
        token_dim=32,
        transformer_layers=1,
        transformer_heads=4,
    ).eval()
    checkpoint = tmp_path / "best.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "image_size": 64,
                "max_views": 2,
                "token_dim": 32,
                "transformer_layers": 1,
                "transformer_heads": 4,
            },
        },
        checkpoint,
    )
    result = export_onnx(checkpoint, tmp_path / "mobile.onnx")
    assert result.is_file() and result.stat().st_size > 1000
    import onnxruntime as ort

    sess = ort.InferenceSession(str(result), providers=["CPUExecutionProvider"])
    feeds = {
        "images": np.zeros((1, 2, 3, 64, 64), dtype=np.float32),
        "view_angles": np.zeros((1, 2, 2), dtype=np.float32),
        "view_valid": np.ones((1, 2), dtype=bool),
    }
    results = sess.run(None, feeds)
    assert [x.shape for x in results] == [(1, 12, 3), (1, 4), (1, 2)]
    assert all(np.isfinite(x).all() for x in results)
