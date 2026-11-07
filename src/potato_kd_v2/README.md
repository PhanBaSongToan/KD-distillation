# Training source

This folder contains the newer trainer and its exact fixed split manifest.

- `prepare_split.py`: materialize the fixed train/validation/test split from the downloaded class folders.
- `make_cache.py`: resize the split for faster training.
- `models.py`, `data.py`, `losses.py`, `metrics.py`: model, input pipeline, objectives, and metrics.
- `train.py`: ResNet18 teacher or hybrid student trainer; supports online and offline KD.
- `train_mt.py`: multi-teacher trainer used by the exploratory multi-teacher study.
- `split_manifest.json`: filename-to-split mapping for the fixed 2,492/276/308 experiment.

The complete command line and recipe are in [`docs/REPRODUCIBILITY.md`](../../docs/REPRODUCIBILITY.md). The benchmark dataset and trained weights are intentionally not included.
