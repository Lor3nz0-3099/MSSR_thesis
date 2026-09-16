# Expert dataset visualization

Reproducible thesis-figure generator for `datasets/expert_v1_compact`.

Default representative episodes:

- Stairs / Snake8: seed 3101
- Gap / Snake8: seed 4107
- RC-Car8: seed 5101

## Run

```bash
scripts/visualization/expert_dataset/run.sh
```

## Custom seeds

```bash
scripts/visualization/expert_dataset/run.sh \
  --stairs-seed 3105 \
  --gap-seed 4109 \
  --rc-car-seed 5104
```

## Output

Generated under `figures/expert_dataset_v1/` in:

- PNG (300 dpi)
- PDF
- SVG

`figure_manifest.json` records the representative episodes and the SHA-256
of the compact-dataset manifest used for generation.

The wrapper deliberately uses Ubuntu's compatible NumPy 1.x / Matplotlib
stack and does not modify the user's Python, ROS, or Isaac environments.

The expert dataset is read-only.
