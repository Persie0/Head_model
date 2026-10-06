import torch

from head_model.model import HeadScanLite, ShapeBasis


def make_basis(
    vertices: int = 32,
    components: int = 4,
) -> ShapeBasis:
    return ShapeBasis(
        mean=torch.zeros(vertices, 3),
        components=torch.randn(
            components,
            vertices,
            3,
        ) * 0.01,
        coeff_std=torch.ones(components),
    )


def test_variable_view_shapes():
    model = HeadScanLite(
        make_basis(),
        token_dim=64,
        transformer_layers=1,
        transformer_heads=4,
    ).eval()
    images = torch.randn(2, 3, 3, 64, 64)
    angles = torch.tensor([
        [
            [0.0, 0.0],
            [45.0, 0.0],
            [90.0, 0.0],
        ],
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    ])
    valid = torch.tensor([
        [True, True, True],
        [True, False, False],
    ])

    with torch.no_grad():
        out = model(images, angles, valid)

    assert out["vertices"].shape == (2, 32, 3)
    assert out["coeff_norm"].shape == (2, 4)
    assert out["view_quality"].shape == (2, 3)
    assert out["normal"].shape == (2, 3, 3, 64, 64)
    assert torch.allclose(
        out["view_quality"].sum(dim=1),
        torch.ones(2),
        atol=1e-5,
    )
    assert out["view_quality"][1, 1:].max().item() < 1e-4


def test_geometry_only_matches_full_geometry():
    model = HeadScanLite(
        make_basis(),
        token_dim=64,
        transformer_layers=1,
        transformer_heads=4,
    ).eval()
    images = torch.randn(1, 2, 3, 64, 64)
    angles = torch.zeros(1, 2, 2)
    valid = torch.ones(
        1,
        2,
        dtype=torch.bool,
    )

    with torch.no_grad():
        full = model(images, angles, valid)
        vertices, coeff, quality = model.forward_geometry(
            images,
            angles,
            valid,
        )

    assert torch.allclose(
        vertices,
        full["vertices"],
        atol=1e-6,
    )
    assert torch.allclose(
        coeff,
        full["coeff_norm"],
        atol=1e-6,
    )
    assert torch.allclose(
        quality,
        full["view_quality"],
        atol=1e-6,
    )
