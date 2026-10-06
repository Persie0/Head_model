# Registered head dataset format

HeadScan-Lite expects registered full-head meshes plus one or more RGB views.

## Directory example

```text
dataset/
  train.jsonl
  valid.jsonl

  train_0001/
    vertices.npy
    view_00.jpg
    view_00_normal.npy
    view_00_depth.npy
    view_00_mask.png
    ...

  valid_0001/
    vertices.npy
    ...
```

Paths inside each JSONL record are relative to the manifest directory.

## Mesh

`vertices.npy`

```text
dtype: float32 recommended
shape: [V, 3]
unit: millimetres recommended
```

Every identity must use identical topology and vertex ordering. Meshes should be
in a common canonical coordinate frame. The current trainer does not perform
mesh registration for you.

The PCA prior is generated only from `train.jsonl`; validation identities are
never included when constructing it.

## Manifest

One JSON object per line:

```json
{
  "id": "person_000001",
  "vertices": "person_000001/vertices.npy",
  "views": [
    {
      "image": "person_000001/view_00.jpg",
      "normal": "person_000001/view_00_normal.npy",
      "depth": "person_000001/view_00_depth.npy",
      "mask": "person_000001/view_00_mask.png",
      "confidence": "person_000001/view_00_confidence.png",
      "yaw": 0.0,
      "pitch": 0.0
    }
  ]
}
```

Required:

- `id`
- `vertices`
- one or more `views`
- `views[].image`

Recommended:

- `normal`
- `depth`
- `mask`
- `yaw`
- `pitch`

Optional:

- `confidence`

If confidence is absent and a mask is present, the foreground mask is used as
the dense confidence target.

## RGB

Any Pillow-readable RGB image.

The loader:

1. converts to RGB,
2. resizes to the configured square input,
3. applies mild color augmentation during training,
4. applies ImageNet mean/std normalization.

Do not pre-normalize the JPEG/PNG.

## Normal map

`.npy`, float32.

Accepted layout:

```text
[H, W, 3]
or
[3, H, W]
```

Values should be unit vectors in camera coordinates.

Background can be zero. When a mask is available, normal loss is evaluated only
on foreground pixels.

## Depth map

`.npy`, float32.

Accepted layout:

```text
[H, W]
[H, W, 1]
or
[1, H, W]
```

Recommended units: millimetres.

Zero depth is treated as invalid/background. Internally depth is divided by
`--depth-scale-mm` (default 250) for numerical stability.

## Mask / confidence

8-bit grayscale PNG is recommended.

```text
0   = background / no confidence
255 = foreground / full confidence
```

Intermediate values are valid for confidence maps.

## Angles

`yaw` and `pitch` are degrees. They are encoded as sine/cosine values before
being added to the view token.

Suggested capture positions for 8 views:

```text
-157.5, -112.5, -67.5, -22.5,
 +22.5,  +67.5, +112.5, +157.5 degrees
```

A front-centered alternative is also fine. Consistency matters more than the
exact convention.

## Variable-view training

For every training sample the loader randomly chooses between `min_views` and
the available number up to `max_views`.

This means a subject with eight views automatically creates training examples
such as:

```text
front only
front + left
front + right + rear
6 views
all 8 views
```

without duplicating the dataset on disk.

Validation uses a deterministic prefix of the stored views.

## Recommended dataset preparation

For each registered 3D scan:

1. canonicalize scale/orientation,
2. render 8–16 known camera views,
3. save RGB, metric depth, camera-space normals and foreground mask,
4. write yaw/pitch into the manifest,
5. keep the original registered metric vertices as the target.

Real photographs can also be used, but dense supervision is easiest and most
accurate when RGB/depth/normal/mask are rendered from the corresponding scan.
A later fine-tuning stage can mix real phone images with weaker 2D consistency
losses.
