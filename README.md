# Head_model — HeadScan-Lite

Colab-friendly variable-view 3D head reconstruction designed to train on a single
free-tier GPU and later run on a phone.

The model accepts **1–8 RGB views** of the same head and predicts an explicit,
registered full-head mesh. It combines:

- a shared **MobileNetV3-Small** image encoder,
- dense **normal / depth / silhouette / confidence** supervision,
- a small angle-aware **Set/Transformer fusion** block,
- a registered **PCA full-head shape prior**, and
- an explicit metric mesh output in millimetres.

This intentionally avoids NeRF/SDF/diffusion training in the normal path. Those
methods are much more expensive and are a poor fit for free Colab + on-device
deployment.

## Colab

Open:

`colab/train_headscan_lite_colab.ipynb`

Select a GPU runtime and choose **Runtime → Run all**.

The notebook follows the same pattern as the `resistor_model` Colab trainers:

- fresh shallow clone of `main`,
- explicit trainer version print,
- GPU check,
- Google Drive mount,
- visible setup/configuration prints,
- batch-by-batch progress,
- 5-second heartbeat during silent subprocess startup,
- `last.pt` resume staging from Drive with byte progress,
- `best.pt`, `last.pt`, numbered checkpoints and `metrics.jsonl`,
- ONNX export after training,
- final ZIP packaging/download.

Persistent output:

`MyDrive/head_model/headscan-lite/`

Recommended dataset archive:

`MyDrive/head_model/headscan_dataset.zip`

The archive should extract to a directory containing `train.jsonl` and
`valid.jsonl`.

If no real dataset is found, the Colab recipe creates a **tiny procedural smoke
dataset** and runs a short end-to-end test. The notebook prints a large warning:
that synthetic data is only for verifying the pipeline and does **not** produce
a useful real-world head scanner.

## Why there is no automatic FaceScape download

High-quality head-scan datasets commonly require registration, institutional
access, non-commercial terms, or redistribution restrictions. This repository
therefore does not silently download or redistribute one of them.

Use scans that you are licensed to train on and convert them to the neutral
registered-head manifest described in [docs/dataset-format.md](docs/dataset-format.md).

## Dataset requirement

All target meshes must have the **same vertex ordering/topology** and should be
in a canonical pose. The recommended unit is **millimetres**.

Each subject can have one or more rendered/captured views. RGB is required;
depth, normals and masks are optional but strongly recommended.

Example:

```json
{
  "id": "subject_0001",
  "vertices": "subject_0001/vertices.npy",
  "views": [
    {
      "image": "subject_0001/front.jpg",
      "normal": "subject_0001/front_normal.npy",
      "depth": "subject_0001/front_depth.npy",
      "mask": "subject_0001/front_mask.png",
      "yaw": 0.0,
      "pitch": 0.0
    }
  ]
}
```

One JSON object is stored per line in `train.jsonl` / `valid.jsonl`.

See the full specification in `docs/dataset-format.md`.

## Architecture

```text
1..8 RGB images
      │
      ├─ shared MobileNetV3-Small
      │      ├─ depth
      │      ├─ normals
      │      ├─ silhouette
      │      └─ confidence
      │
      └─ pooled per-view tokens
               + yaw/pitch encoding
                       │
                3-layer Transformer
                       │
              learned view weighting
                       │
                canonical head latent
                       │
                 PCA coefficients
                       │
              registered metric mesh
```

The number of input photographs varies during training. A sample may contain
1, 2, 3, ... up to `MAX_VIEWS`, forcing the same network to learn both
single-view completion and genuine multi-view refinement.

## Training objective

The default objective is:

```text
L =
  1.00 * coefficient MSE
+ 5.00 * metric vertex Smooth-L1
+ 1.00 * normal cosine loss
+ 0.50 * depth L1
+ 0.50 * silhouette BCE
+ 0.25 * confidence BCE
```

Validation reports explicit geometry error:

- mean vertex error in mm,
- 95th-percentile vertex error in mm,
- normalized PCA coefficient RMSE.

For a registered mesh dataset these are more meaningful scanner metrics than
photometric quality alone.

## Shape prior

At the first run the trainer builds `shape_basis.npz` from training meshes using
PCA/SVD. It stores:

- mean head `[V,3]`,
- PCA components `[K,V,3]`,
- per-component coefficient standard deviation.

The network predicts normalized coefficients. The reconstructed mesh is:

```text
V = mean + Σ(coeff_normalized[k] * std[k] * basis[k])
```

This keeps the final output explicit, metric and cheap on mobile.

## Mobile export

After training:

```bash
python -m head_model.export \
  --checkpoint runs/headscan_lite/best.pt \
  --output headscan_lite.onnx
```

The exported graph intentionally skips dense training-only heads.

Inputs:

- `images`: `[B, 8, 3, H, W]`
- `view_angles`: `[B, 8, 2]` yaw/pitch degrees
- `view_valid`: `[B, 8]` bool

Outputs:

- `vertices_mm`: `[B, V, 3]`
- `shape_coefficients`: `[B, K]`
- `view_quality`: `[B, 8]`

Unused view slots are zero-padded and marked `False` in `view_valid`.

## Local install

```bash
pip install -e ".[train,export]"
```

Train:

```bash
python -m head_model.train \
  --train-manifest /data/heads/train.jsonl \
  --valid-manifest /data/heads/valid.jsonl \
  --basis-path runs/headscan_lite/shape_basis.npz \
  --output-dir runs/headscan_lite \
  --image-size 256 \
  --max-views 8 \
  --epochs 60 \
  --batch-size 2
```

## Recommended real training data

For a high-quality model, prioritize data with:

1. true full-head coverage rather than face-only crops,
2. registered topology across identities,
3. calibrated or known view angles,
4. metric scale,
5. dense normals/depth rendered from the ground-truth mesh,
6. broad head-shape, age, hairstyle and appearance diversity,
7. explicit rights compatible with the intended use of the resulting model.

Hair is a separate problem from skull/skin geometry. For a first production
model, train the geometric mesh on head/skin geometry and add hair as a separate
surface/appearance stage later instead of forcing a low-rank head prior to model
individual hair strands.

## License

Repository code: MIT.

Training data and any pretrained weights keep their own terms. Verify those
separately before redistribution or commercial use.
