# Evaluation

This directory contains the script for precomputing random masks and the ADM evaluator required for image generation and inpainting scripts, corresponding to `scripts/sample_generation.py` and `scripts/sample_completion.py`. Run all commands below from the repository root. The released checkpoints and tokenizer are downloaded automatically; add the local checkpoint to override this behavior.

## Image Generation

Use the corresponding config for `STAR-B`, `STAR-L`, `STAR-XL`, or `STAR-XXL`. The config contains that checkpoint's reported temperature and classifier-free guidance settings. The reported runs use one GPU, a per-process batch size of 50, and a seed of 42. For `STAR-B`:

```bash
python scripts/sample_generation.py \
  config=configs/star_b.yaml \
  experiment.output_dir=/path/to/generation/star_b \
  inference.num_samples=50000 \
  inference.per_process_batch_size=50 \
  inference.num_order_workers=0 \
  inference.seed=42
```

Repeat with `star_l.yaml`, `star_xl.yaml`, and `star_xxl.yaml`, using a distinct output directory for each run. Without a fixed `inference.class_id`, global sample `i` receives ImageNet class `i % 1000`, yielding 50 samples per class. The output NPZ is written adjacent to the image directory; the example above produces `/path/to/generation/star_b.npz`.

## Image Inpainting

Run the script for precomputing random masks before starting the image inpainting script:

```bash
python evaluations/precompute_masks.py \
  --output-dir /path/to/random_masks \
  --count 50000 \
  --grid-height 16 \
  --grid-width 16 \
  --workers 12 \
  --seed 42 \
  --mask-ratios 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9
```

This script precomputes random masks in a separate CPU process pool. Each mask ratio produces `masks_ratio_<ratio>.jsonl`; every row has the canonical form `{"indices":[...]}` with zero-based row-major token indices. `manifest.json` records the grid, seed, target mask cardinality, proposal counts, acceptance rate, and elapsed time. `--proposals-per-task` controls CPU task granularity, while `--max-proposals` sets a fail-closed search cap. Their defaults are normally sufficient.

We evaluated `STAR-L` and `STAR-XL` using the pretokenized single-crop ImageNet-1k validation set, which can be computed as follows. The input must follow the ImageFolder layout `/path/to/imagenet/val/<class>/<image>`:

```bash
python scripts/pretokenization.py \
  --data-path /path/to/imagenet \
  --output-dir /path/to/tokenized_validation \
  --image-size 256 \
  --single-crop
```

The output is `/path/to/tokenized_validation/pretokenized.jsonl`. Every record contains its `class_id` and flattened `tokens`. Run the completion script once for each mask file. The reported runs use 1 GPU, a per-process batch size of 50, and a seed of 42:

```bash
python scripts/sample_completion.py \
  config=configs/star_l.yaml \
  experiment.output_dir=/path/to/inpainting/star_l_ratio_0.1 \
  inference.pretokenized_jsonl=/path/to/tokenized_validation/pretokenized.jsonl \
  inference.masks=/path/to/random_masks/masks_ratio_0.1.jsonl \
  inference.num_samples=50000 \
  inference.per_process_batch_size=50 \
  inference.seed=42
```

Repeat for ratios 0.2 through 0.9, then repeat the nine runs with `configs/star_xl.yaml`. Each run writes numbered PNGs to experiment.output_dir and creates an evaluator-ready `experiment.output_dir.npz` next to that directory.

## ADM Evaluator

First, install the evaluator-only dependencies:

```bash
pip install -r evaluations/requirements.txt
```

Then, download the [official ADM reference batch](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) and evaluate the generation or inpainting NPZ with:

```bash
python evaluations/evaluator.py \
  /path/to/VIRTUAL_imagenet256_labeled.npz \
  /path/to/generated_samples.npz
```

## Reproducing Main Results

The following tables present reproduced results using the release protocol described above, since the code release includes a sampling interface for user convenience and provides better control over RNG seeds during parallel sampling of uniform spanning trees.

#### Image Generation

| Model | FID | IS | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| `STAR-B` | 2.26 | 296.3 | 0.82 | 0.57 |
| `STAR-L` | 1.97 | 318.5 | 0.81 | 0.58 |
| `STAR-XL` | 1.64 | 333.9 | 0.80 | 0.62 |
| `STAR-XXL` | 1.56 | 341.4 | 0.81 | 0.61 |

#### Image Inpainting

All metrics are arithmetic means over masking ratios 0.1 through 0.9. IS is reported using the 1-GPU source order.

| Model | FID | IS | Precision | Recall |
| :--- | :---: | :---: | :---: | :---: |
| `STAR-L` | 2.16 | **66.5** | 0.82 | 0.58 |
| `STAR-XL` | 2.03 | **65.8** | 0.81 | 0.60 |


We reproduced performance nearly identical to the results reported in the paper using the released code; however, we observed that the Inception Scores (IS) for RAR and STAR in the inpainting evaluation were inflated relative to other baselines because we **parallelized pre-tokenization**. This difference arises because the ADM evaluator does not compute IS once over the entire set of 50,000 images. Instead, it splits the predictions into 5,000-image chunks, computes IS for each chunk, and reports the mean. 

In the 1-GPU source order, the 50 validation images from each ImageNet class are contiguous, so each split contains approximately 100 classes, but 2-GPU rank-major order distributes each class across the two shards, causing each split to contain approximately 200 classes. This produces a more diverse marginal class distribution within each split and consequently a higher IS. The table below reports the average IS over masking ratios 0.1 through 0.9 under different numbers of GPUs used for pretokenization:

| Model | 1-GPU source order | 2-GPU rank-major order |
| :--- | :---: | :---: |
| `RAR-L` | 55.5 | 86.2 |
| `RAR-XL` | 56.2 | 87.6 |
| `STAR-L` | 66.5 | 110.5 |
| `STAR-XL` | 65.8 | 108.6 |

We find that RAR and STAR achieve IS values similar to those reported in the paper under the 2-GPU rank-major order. However, the IS trend across masking ratios and STAR's advantage over baselines remain consistent under the 1-GPU source order. The documented release evaluation protocol uses 1-GPU pre-tokenization and therefore preserves source order, while the evaluator performs no implicit reordering. To reproduce the 2-GPU rank-major order evaluation, create a separate reordered NPZ following the script below and evaluate:

```bash
SAMPLE_NPZ=/path/to/inpainting/star_l_ratio_0.1.npz \
REORDERED_NPZ=/path/to/inpainting/star_l_ratio_0.1_paper_is_order.npz \
python - <<'PY'
import os
import numpy as np

samples = np.load(os.environ["SAMPLE_NPZ"])["arr_0"]
rank_major = np.concatenate((samples[0::2], samples[1::2]), axis=0)
np.savez(os.environ["REORDERED_NPZ"], arr_0=rank_major)
PY
```
