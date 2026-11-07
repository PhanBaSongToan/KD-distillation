"""Materialize the experiment's fixed split from class/image folders.

Example:
    python prepare_split.py --source /path/to/PotatoDataset --output /path/to/potato_split

The source must contain one directory per class (directly or one level below
the given path). The split manifest is fixed in this repository. The script
copies images and fails if any manifest entry is missing or if counts differ.
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path


def find_class_root(source: Path, classes: set[str]) -> Path:
    candidates = [source, *[p for p in source.rglob("*") if p.is_dir()]]
    for candidate in candidates:
        present = {p.name for p in candidate.iterdir() if p.is_dir()}
        if classes.issubset(present):
            return candidate
    raise FileNotFoundError(
        f"Could not find a directory with all class folders under {source}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, required=True,
                    help="Dataset root or parent folder containing class folders")
    ap.add_argument("--output", type=Path, required=True,
                    help="Destination with train/val/test/class folders")
    ap.add_argument("--manifest", type=Path,
                    default=Path(__file__).with_name("split_manifest.json"))
    args = ap.parse_args()

    manifest: dict[str, str] = json.loads(args.manifest.read_text(encoding="utf-8"))
    classes = {entry.split("/", 1)[0] for entry in manifest}
    class_root = find_class_root(args.source, classes)
    missing = [rel for rel in manifest if not (class_root / rel).is_file()]
    if missing:
        preview = "\n".join(missing[:12])
        raise FileNotFoundError(
            f"{len(missing)} manifest files are missing under {class_root}.\n{preview}"
        )

    counts: Counter[tuple[str, str]] = Counter()
    for rel, split in manifest.items():
        cls, filename = rel.split("/", 1)
        src = class_root / cls / filename
        dst = args.output / split / cls / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
        counts[(split, cls)] += 1

    totals = Counter(manifest.values())
    expected = {"train": 2492, "val": 276, "test": 308}
    actual = {split: totals[split] for split in ("train", "val", "test")}
    if actual != expected:
        raise RuntimeError(f"Manifest count mismatch: {actual} != {expected}")
    print(f"Source class folders: {class_root}")
    print(f"Prepared split: {args.output}")
    print("Counts:", actual)
    print("Classes:", ", ".join(sorted(classes)))


if __name__ == "__main__":
    main()
