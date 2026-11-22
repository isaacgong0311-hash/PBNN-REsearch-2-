"""Thin wrapper to run BiPer's own CIFAR-10 scripts unchanged on a cluster without internet.
usage: python biper_run.py <split> <script.py> [script args ...]
  split "paper": train = 50k, "val/test" = the 10k test split (exactly what BiPer's scripts do; the best epoch is chosen on the test set).
  split "ours" : train = 45k, "val/test" = a 5k held-out validation set (same seed-0 permutation as the LogicTreeNet runs), so the best epoch is chosen
                 on validation data; evaluate the best checkpoint on the real test set afterwards with split "paper" and -e.
It only replaces torchvision.datasets.CIFAR10 by the uoft-cs/cifar10 Hugging Face mirror (offline cache). Training math is not touched."""

import logging
import os
import runpy
import sys

# BiPer sets the root logger to DEBUG and logs to a file; PIL then logs every PNG chunk of every decoded image (about 2 GB per hour per run). Silence it.
logging.getLogger("PIL").setLevel(logging.WARNING)
import torch
from torchvision import datasets

SPLIT = sys.argv[1]
SCRIPT = os.path.abspath(sys.argv[2])
REST = sys.argv[3:]
assert SPLIT in ("paper", "ours")


class HFCifar10(torch.utils.data.Dataset):
    def __init__(self, root=None, train=True, download=False, transform=None):
        from datasets import load_dataset

        self.transform = transform
        if SPLIT == "paper":
            self.ds = load_dataset("uoft-cs/cifar10", split="train" if train else "test")
            self.idx = list(range(len(self.ds)))
        else:
            self.ds = load_dataset("uoft-cs/cifar10", split="train")
            perm = torch.randperm(len(self.ds), generator=torch.Generator().manual_seed(0))
            self.idx = (perm[5000:] if train else perm[:5000]).tolist()

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        row = self.ds[self.idx[i]]
        img = row["img"]
        if self.transform is not None:
            img = self.transform(img)
        return img, row["label"]


datasets.CIFAR10 = HFCifar10
os.chdir(os.path.dirname(SCRIPT))
sys.path.insert(0, os.path.dirname(SCRIPT))
sys.argv = [SCRIPT] + REST
runpy.run_path(SCRIPT, run_name="__main__")
