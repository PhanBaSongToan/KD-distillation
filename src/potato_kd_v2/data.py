"""Dataset pipeline for the potato leaf disease KD study.

All raw images are 1500x1500 field photographs. `make_cache.py` in the baseline
folder writes a 448x448 copy (`data/potato_split_448`) so an epoch does not
spend most of its time decoding JPEGs. This module works with either root.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(image_size: int, aug: str = "strong") -> transforms.Compose:
    if aug == "paper":
        ops = [
            transforms.Resize((image_size, image_size)),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
            transforms.RandomAffine(degrees=30),
        ]
    elif aug == "field":
        ops = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.70, 1.0), ratio=(0.85, 1.15),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.RandomAffine(0, translate=(0.08, 0.08), scale=(0.90, 1.10)),
            transforms.ColorJitter(0.25, 0.25, 0.20, 0.04),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.15),
        ]
    elif aug == "strong":
        # The dataset is uncontrolled: photos of whole plants, single leaves in
        # a hand, plastic sheeting backgrounds, variable zoom and lighting.
        # Colour is a genuine cue (yellowing vs. necrosis) so hue jitter stays
        # small; geometric augmentation does most of the work.
        ops = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.45, 1.0), ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.RandomVerticalFlip()], p=0.15),
            transforms.RandomApply(
                [transforms.RandomAffine(
                    degrees=20, translate=(0.06, 0.06), scale=(0.92, 1.08)
                )], p=0.6,
            ),
            transforms.ColorJitter(0.25, 0.25, 0.18, 0.03),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.10),
        ]
    elif aug == "fast":
        # Same geometric coverage as "strong" but drops the two most expensive
        # PIL ops: HSV-based saturation/hue jitter and Gaussian blur. Profiling
        # showed those two dominate the data-path cost while the GPU sat idle
        # ~75% of the wall clock.
        ops = [
            transforms.RandomResizedCrop(
                image_size, scale=(0.45, 1.0), ratio=(0.75, 1.33),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.RandomAffine(
                    degrees=20, translate=(0.06, 0.06), scale=(0.92, 1.08)
                )], p=0.5,
            ),
            transforms.ColorJitter(brightness=0.25, contrast=0.25),
        ]
    else:
        raise ValueError(f"unknown aug mode {aug!r}")
    ops += [
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15), ratio=(0.3, 3.3)),
    ]
    return transforms.Compose(ops)


def build_eval_transform(image_size: int, crop_pct: float = 1.14) -> transforms.Compose:
    resize = int(round(image_size * crop_pct))
    return transforms.Compose([
        transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


class RamImageFolder(datasets.ImageFolder):
    """ImageFolder that keeps the decoded 448x448 images in RAM.

    JPEG decoding was ~5 of the ~12 ms/image spent in the data path, and the
    data path was the bottleneck (the GPU sat idle). 2493 images at 448x448x3
    is ~1.5 GB, which fits comfortably.
    """

    def __init__(self, root, transform=None, preload: bool = True):
        super().__init__(root, transform=transform)
        self._cache = []
        if preload:
            from PIL import Image
            for path, _ in self.samples:
                with Image.open(path) as im:
                    self._cache.append(im.convert("RGB"))

    def __getitem__(self, index: int):
        _, target = self.samples[index]
        img = self._cache[index] if self._cache else Image.open(self.samples[index][0]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, target


class PrefetchLoader:
    """Thread-parallel batch prefetcher for environments where DataLoader
    worker *processes* are unavailable.

    The GPU was idle ~99% of the time while augmenting 448x448 field photos in
    the main Python thread. PIL/torchvision ops release the GIL, so a handful of
    shard readers running in threads overlaps augmentation with training and
    gives a 3-4x epoch speed-up. Shuffling is per shard, which is equivalent to
    standard SGD sampling for practical purposes.
    """

    def __init__(self, dataset, batch_size: int, n_threads: int = 6,
                 shuffle: bool = True, seed: int = 0):
        from torch.utils.data import Subset
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_threads = max(1, n_threads)
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0
        self._gen = torch.Generator()
        if hasattr(dataset, "targets"):
            self._targets = list(dataset.targets)
        else:
            self._targets = None

    @property
    def targets(self):
        return self._targets

    def __len__(self):
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self):
        import queue
        import threading
        from torch.utils.data import Subset

        n = len(self.dataset)
        self._gen.manual_seed(self.seed + self.epoch)
        perm = torch.randperm(n, generator=self._gen).tolist() if self.shuffle else list(range(n))
        self.epoch += 1
        shards = [perm[i::self.n_threads] for i in range(self.n_threads)]
        q: "queue.Queue" = queue.Queue(maxsize=self.n_threads * 3)
        errors: list = []
        stop = threading.Event()

        def work(indices):
            try:
                loader = DataLoader(
                    Subset(self.dataset, indices), batch_size=self.batch_size,
                    shuffle=self.shuffle, drop_last=False, num_workers=0,
                )
                for batch in loader:
                    if stop.is_set():
                        return
                    q.put(batch)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)
                stop.set()
            finally:
                q.put(None)

        threads = [threading.Thread(target=work, args=(s,), daemon=True) for s in shards]
        for t in threads:
            t.start()
        finished = 0
        while finished < len(threads):
            item = q.get()
            if item is None:
                finished += 1
                continue
            if errors:
                stop.set()
                raise errors[0]
            yield item
        stop.set()
        if errors:
            raise errors[0]


class TTADataset(Dataset):
    """Deterministic multi-view test-time augmentation."""

    def __init__(self, root: Path, image_size: int, views: str = "flip2"):
        self.base = datasets.ImageFolder(str(root))
        self.samples = self.base.samples
        self.classes = self.base.classes
        self.class_to_idx = self.base.class_to_idx
        self.targets = self.base.targets
        self.image_size = image_size
        self.views = views

        resize = int(round(image_size * 1.14))
        norm = [transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)]
        self.center = transforms.Compose([
            transforms.Resize(resize, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size), *norm,
        ])
        self.full = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
            *norm,
        ])
        self.zoom = transforms.Compose([
            transforms.Resize(int(round(image_size * 1.35)),
                              interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(image_size), *norm,
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def _views(self, img):
        if self.views == "center":
            return [self.center(img)]
        if self.views == "flip2":
            c = self.center(img)
            return [c, torch.flip(c, dims=[2])]
        if self.views == "rich":
            out = []
            for tf in (self.center, self.full, self.zoom):
                v = tf(img)
                out += [v, torch.flip(v, dims=[2])]
            return out
        raise ValueError(self.views)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        views = self._views(img)
        return torch.stack(views), target


def make_loaders(
    data_root: str,
    batch_size: int = 32,
    num_workers: int = 0,
    image_size: int = 224,
    aug: str = "strong",
    train_on_val: bool = False,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    root = Path(data_root)
    for split in ("train", "val", "test"):
        if not (root / split).is_dir():
            raise FileNotFoundError(f"missing split: {root / split}")

    train_set = RamImageFolder(str(root / "train"), transform=build_train_transform(image_size, aug))
    val_set = datasets.ImageFolder(str(root / "val"), transform=build_eval_transform(image_size))
    test_set = datasets.ImageFolder(str(root / "test"), transform=build_eval_transform(image_size))

    if train_on_val:
        train_set = torch.utils.data.ConcatDataset([train_set, RamImageFolder(
            str(root / "val"), transform=build_train_transform(image_size, aug))])

    classes = datasets.ImageFolder(str(root / "train")).classes
    kw = dict(num_workers=num_workers, pin_memory=True)
    g = torch.Generator()
    g.manual_seed(seed)
    train_loader = (
        PrefetchLoader(train_set, batch_size, n_threads=6, shuffle=True, seed=seed)
        if num_workers == 0 else
        DataLoader(train_set, batch_size=batch_size, shuffle=True, generator=g,
                   drop_last=False, **kw)
    )
    return (
        train_loader,
        DataLoader(val_set, batch_size=batch_size, shuffle=False, **kw),
        DataLoader(test_set, batch_size=batch_size, shuffle=False, **kw),
        classes,
    )
