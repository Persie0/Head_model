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

`MyDrive/head_model/headscan-lite-headspace/`

## Headspace / LYHM setup

The Colab notebook now uses **Headspace / LYHM as its primary real dataset**.
Obtain the licensed packages from the University of York Headspace/LYHM
distribution and place the archives, or their extracted folders, anywhere under:

`MyDrive/head_model/headspace/`

For the current trainer request these two packages:

1. **Headspace FLAME registrations** — registered OBJ meshes + FLAME parameters.
2. **Headspace 3dMD package** — raw PNG camera images + TKA calibration data.

The converter automatically:

- finds `registrations/<actor_id>/*.obj`,
- finds subject-matched `*C.png` color-camera images such as
  `00001/1C.png` and `00001/2C.png`,
- ignores IR/TKA files during this first training stage,
- converts registered geometry to millimetres,
- verifies identical topology across subjects,
- makes deterministic train/validation splits,
- letterboxes RGB inputs to a consistent resolution, and
- caches the prepared dataset at
  `MyDrive/head_model/headspace_prepared_v1.zip`.

The first run can therefore work directly from the original licensed downloads;
later free-Colab sessions restore the much smaller prepared cache before
resuming training.

Headspace/LYHM is distributed for **non-commercial research and education**
under its own agreement. This repository does not download or redistribute the
dataset itself.

If no licensed Headspace folder is found, the notebook runs a **tiny procedural
smoke test** only to verify setup/checkpoint/export. Those weights are not a
usable head scanner.

The generic manifest format remains supported for other licensed datasets; see
[docs/dataset-format.md](docs/dataset-format.md).

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

For generic datasets with dense supervision, the default objective is:

```text
L =
  1.00 * coefficient MSE
+ 5.00 * metric vertex Smooth-L1
+ 1.00 * normal cosine loss
+ 0.50 * depth L1
+ 0.50 * silhouette BCE
+ 0.25 * confidence BCE
```

For **Headspace/LYHM**, Colab uses `--geometry-only-training`: only the
coefficient and metric-vertex terms are evaluated, and the dense
normal/depth/mask/confidence heads are skipped. This is both more faithful to
the licensed source data and substantially cheaper on a free Colab GPU.

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
  --batch-size 2 \
  --geometry-only-training
```

## Recommended real training data

Headspace/LYHM is the default starting point for this repository. For additional
or replacement datasets, prioritize data with:

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
