# KD-distillation
# Potato Knowledge Distillation

Experiments for knowledge distillation on the Potato Leaf Disease Dataset.

## Final comparable experiment

All variants below use the same protocol:

- same student architecture: about 1.05M parameters;
- input resolution: 224x224;
- same train/validation/test split: 2492 / 276 / 308 images;
- student trained from scratch;
- checkpoint selected using validation performance;
- test evaluated once after model selection.

| Variant | Accuracy | Macro-F1 | Note |
|---|---:|---:|---|
| Paper baseline reference | 84.57% | reported in the paper | original comparison point |
| Recipe reference (`ref`) | 89.94% | 88.06% | strongest clean reference run |
| MixUp | 87.99% | 85.96% | did not help on this split |
| Class-weighted hard loss | 89.29% | 88.85% | accuracy/F1 trade-off; needs multi-seed confirmation |

The weighted run should not be claimed as a definitive improvement from one seed alone. The reference run is the main result; weighted CE is an exploratory ablation.

## Files

- `notebooks/potato_kd_sweep.ipynb`: reproducible sweep of the comparable variants.

The dataset, checkpoints, cache files and generated outputs are intentionally excluded from the repository.

## Reproducibility

Run the notebook on Kaggle with the Potato dataset attached. Keep the split manifest, seed, image size and checkpoint-selection rule unchanged when comparing variants.
