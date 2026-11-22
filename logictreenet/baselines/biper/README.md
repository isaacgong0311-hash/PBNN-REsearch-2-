# BiPer ResNet-18 baseline (CIFAR-10)

BiPer: *Binary Neural Networks using a Periodic Function* (CVPR 2024, [arXiv:2404.01278](https://arxiv.org/abs/2404.01278)). Their repo is cloned separately, pinned at commit `0c07fdd6f7ada649b1bb4576174e54731f615ff6`; these two helpers run it unchanged.

- `biper_eval_cifar.py`: evaluates a released or retrained checkpoint with BiPer's own model code and counts MACs per image.
- `biper_run.py`: runs BiPer's training scripts offline with the Hugging Face CIFAR-10 mirror. `paper` split = their setup (best epoch picked on the 10k test set); `ours` = 45k/5k split, best epoch on validation, test touched once.

| | Top-1 |
|---|---|
| Paper | 93.75 |
| Released checkpoint (freq 30) | 93.76 |
| Our retrain, paper split, freq 20 / 30 (3 seeds) | 93.32 / 93.35 |
| Our retrain, clean 45k/5k split, test once (3 seeds) | 92.65 |

Per-seed numbers: `../../results/biper_r18_cifar10_retrain.csv`, `../../results/biper_r18_cifar10_f30_and_splits.csv`. Details: `../../docs/report_residual_biper_2026-11-19.md`.

BiPer is not a pure binary network: the first conv, final linear layer, every BatchNorm, per-channel scaling and the residual sums are real-valued (binary convs are 99.7% of MACs). Compare it with LogicTreeNet on accuracy against inference cost, not accuracy alone.
