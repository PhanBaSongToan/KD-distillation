"""Pre-resize the potato dataset once so each epoch does not decode 1500x1500 JPEGs.

The raw images are all 1500x1500. Training with RandomResizedCrop/affine on a
448x448 cached copy is visually indistinguishable from resizing the original,
but decoding is ~15x faster (important because the DSH sandbox forbids
DataLoader worker processes here, so --num-workers must stay 0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
from tqdm import tqdm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="data/potato_split")
    ap.add_argument("--output", default="data/potato_split_448")
    ap.add_argument("--size", type=int, default=448)
    args = ap.parse_args()

    src = Path(args.source)
    dst = Path(args.output)
    for split in ("train", "val", "test"):
        for cls_dir in sorted((src / split).iterdir()):
            if not cls_dir.is_dir():
                continue
            out_dir = dst / split / cls_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for f in tqdm(list(cls_dir.glob("*")), desc=f"{split}/{cls_dir.name}", leave=False):
                if not f.is_file():
                    continue
                im = Image.open(f).convert("RGB")
                im = im.resize((args.size, args.size), Image.BICUBIC)
                im.save(out_dir / (f.stem + ".jpg"), quality=95)
    print("cache written to", dst)


if __name__ == "__main__":
    main()
