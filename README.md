# Potato leaf disease classification with knowledge distillation

This repository collects the paper-style baseline, follow-up experiments, saved notebook evidence, and a reconciled report for the Potato Leaf Disease Dataset in Uncontrolled Environment. The project asks whether the compact hybrid student can be improved through training recipes and knowledge-distillation changes.

> **Scope:** this repository is only for the potato-leaf KD study. The Bee Pollen, Varroa object-detection, and Hưng Yên water-level assignments are separate projects and are not mixed into these results.

## Main finding

The strongest fixed-split scratch student currently recorded is **89.83 ± 0.68% Accuracy and 88.17 ± 1.44% Macro-F1** across three seeds. It uses the paper's 1.051M-parameter student at 224×224 (about 0.630 GFLOPs) and offline KD from a separately trained frozen teacher. This is a promising recipe-level result, **not an isolated adaptive-loss improvement**: optimizer/schedule, augmentation, EMA, label smoothing, training duration, and teacher protocol differ from the paper's online setup.

The proposed teacher-confidence adaptive KD did not improve the local baseline in the saved run (82.47% Accuracy / 79.93% Macro-F1 versus baseline Accuracy 84.09%). There is also an unresolved Macro-F1 discrepancy for the baseline checkpoint (82.65% in its run JSON versus 81.61% in a later evaluator). See the report before quoting any gain.

## Dataset and split

- 3,076 images across seven classes.
- Main fixed manifest: 2,492 train / 276 validation / 308 test.
- The archived early Kaggle notebooks built a different split: 2,487 / 274 / 315. Their metrics are kept separate in the report.
- The local MD5 audit found exact duplicate files crossing the fixed split: 17 test files had identical copies in train or validation. This limits claims about generalization to unseen images.
- The dataset is not included. Obtain it from [Mendeley Data](https://data.mendeley.com/datasets/ptz377bwb8/1) and follow its terms. Checkpoint files and large caches are also excluded.

## Results and methods

- [Complete experiment ledger](docs/RESULTS.md): completed runs, variants, alternative splits, postprocessing, failures, and unverified drafts.
- [Methods, advantages, and limitations](docs/METHODS_AND_LIMITATIONS.md): what each method was intended to test and what conclusions the evidence supports.
- [Reproducibility guide](docs/REPRODUCIBILITY.md): data layout, commands, settings, and run interpretation.
- [Primary comparison figure](docs/figures/primary_comparison.png).

## Code and notebooks

- `src/potato_kd_v2/`: training implementation, model/loss/data code, fixed split manifest, and requirements for the newer experiments.
- `notebooks/potato_kd_sweep_clean.ipynb`: cleaned rerunnable version of the fixed-split Kaggle sweep.
- `notebooks/potato_kd_sweep.ipynb`: previously saved notebook, retained unchanged as historical context.
- `notebooks/archive/`: original executed notebooks retained as evidence, including the alternative-split early KD run, three-teacher run, and interrupted capacity sweep. Notebook outputs can contain old errors; use the ledger for status and the cleaned sweep to reproduce.

## Primary model sizes

| Model | Parameters | Approx. compute at 224×224 |
|---|---:|---:|
| ResNet18 teacher | 11.276M | 1.818 GFLOPs |
| Hybrid student | 1.051M | 0.630 GFLOPs |

## Quick reproduction outline

1. Download the dataset, then run `src/potato_kd_v2/prepare_split.py` with the downloaded class-folder root and a destination folder. It applies the fixed manifest and verifies counts.
2. Install the packages listed in `src/potato_kd_v2/requirements.txt` in a compatible PyTorch/torchvision environment.
3. Run the cleaned Kaggle notebook or follow the exact command/settings documented in `docs/REPRODUCIBILITY.md`.
4. Preserve validation for checkpoint and tuning choices. Read test metrics only after the run is fixed.

The saved notebook uses Kaggle paths and requires a GPU for practical training time. The data and checkpoints are not bundled in this Git repository.

## Important reporting cautions

- Do not compare the 2,487/274/315 notebook scores directly with the 2,492/276/308 fixed-split scores.
- Report raw single-model inference separately from test-time augmentation, calibration, and ensembles.
- Label scratch-trained and warm-start results distinctly.
- State that the paper uses online KD whereas the leading local student uses offline KD.
- Do not quote a precise Macro-F1 gain over the paper-style baseline until its 82.65% vs 81.61% metric discrepancy is reconciled.
