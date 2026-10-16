"""HeadScan-Lite: variable-view full-head reconstruction for mobile deployment.

Avoid importing heavyweight PyTorch/torchvision until the caller actually asks
for a model; especially important for visible startup progress in free Colab.
"""

__all__ = ["HeadScanLite", "load_shape_basis"]


def __getattr__(name: str):
    if name in __all__:
        from .model import HeadScanLite, load_shape_basis
        return {"HeadScanLite": HeadScanLite, "load_shape_basis": load_shape_basis}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
