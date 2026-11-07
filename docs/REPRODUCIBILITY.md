# Reproducing the fixed-split experiments

## 1. Obtain the data

Download the Potato Leaf Disease Dataset in Uncontrolled Environment from [Mendeley Data](https://data.mendeley.com/datasets/ptz377bwb8/1). The dataset is not committed to this repository.

The raw dataset should have one folder per class (or those folders one level below the directory passed as `--source`). The fixed manifest expects these seven class names:

```text
Bacteria, Fungi, Healthy, Nematode, Pest, Phytopthora, Virus
```

Materialize the exact split used by the fixed-split runs:

```powershell
cd src/potato_kd_v2
python prepare_split.py --source D:/data/PotatoDataset --output D:/data/potato_split
```

Expected counts are train 2,492 / validation 276 / test 308. The script fails if a manifest file is missing. It copies the images so the source archive can remain untouched.

## 2. Install dependencies

Use a compatible Python, PyTorch, and torchvision installation; then install the project requirements:

```powershell
python -m pip install -r requirements.txt
```

GPU training is recommended. Kaggle notebook users can open `notebooks/potato_kd_sweep_clean.ipynb`, attach the dataset, and run all cells in order.

## 3. Cache resized images

Images in the source data are large. Resizing once to 448×448 speeds up repeated augmentation and training:

```powershell
python make_cache.py --source D:/data/potato_split --output D:/data/potato_split_448 --size 448
```

## 4. Recorded main scratch-student recipe

The three-seed scratch student used a separately trained teacher, then froze the teacher during student training. This is offline KD, not the paper's online teacher-student update.

Teacher example (seed 42, 224 input):

```powershell
python train.py --mode teacher --data D:/data/potato_split_448 `
  --output D:/runs/teacher224_s42 --seed 42 --epochs 50 --batch-size 32 `
  --lr 1e-3 --weight-decay 0.05 --label-smoothing 0.1 `
  --aug strong --ema 0.999 --image-size 224 --track mean
```

Student example:

```powershell
python train.py --mode student --data D:/data/potato_split_448 `
  --teacher D:/runs/teacher224_s42/best.pt `
  --output D:/runs/student224_s42 --seed 42 --epochs 250 --batch-size 32 `
  --lr 1.5e-3 --weight-decay 0.05 --label-smoothing 0.05 `
  --aug fast --ema 0.999 --kd dkd --feature map `
  --lambda-hard 1.0 --lambda-feature 0.3 --lambda-response 1.0 `
  --lambda-self 0.1 --kd-ramp 20 --image-size 224 --track mean
```

Repeat using seeds 43 and 44 to reproduce the reported three-seed summary. Save each run directory; it contains its arguments, validation history, checkpoint, and test metrics.

## 5. Interpretation and selection

- Select checkpoints using validation; use the held-out test only for final reporting.
- Read `test` as raw single-view inference. Report `test_tta`, calibrated metrics, and ensembles separately.
- State the seed count and report mean ± sample standard deviation for repeated runs.
- Keep warm-start refinement separate from scratch-trained results.
- Do not combine results from the 2,487/274/315 historical split with the fixed manifest results.
- Disclose the exact-duplicate audit and the baseline Macro-F1 discrepancy described in the experiment ledger.

## 6. Expected artifacts

The training code writes `best.pt`, `history.json`, and `test_metrics.json` into each output directory. Checkpoint weights and raw datasets are not stored in Git to keep the repository compact and avoid redistributing the dataset.
