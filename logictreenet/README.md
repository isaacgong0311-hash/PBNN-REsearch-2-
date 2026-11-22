# LogicTreeNet on CIFAR-10 (PyTorch reimplementation)

A reimplementation of **LogicTreeNet** (Petersen et al., *Convolutional Differentiable Logic Gate Networks*, NeurIPS 2024, [arXiv:2411.04732](https://arxiv.org/abs/2411.04732)), written as the logic-gate baseline for the PBNN project (Prof. Liping Liu's lab, Tufts). The authors never released code for the convolutional model, so this is written from the paper. The same file also has the dense **DiffLogicNet** (Petersen et al., NeurIPS 2022, [arXiv:2210.08277](https://arxiv.org/abs/2210.08277)).

Isaac Gong, September to November 2026.

**Terms.** *Soft* is the relaxed network that is trained (each neuron is a softmax mix of the 16 two-input gates). *Hard* is the same network with each neuron snapped to its most likely gate: a pure binary network, and the number that counts. *Gap* is soft minus hard.

## Results (CIFAR-10 test set, as of 2026-11-21)

Random crop and flip, 200 epochs, best epoch chosen on a 5k validation split, test set evaluated once. Mean over seeds.

| Model | Seeds | Hard | Soft | Gap | Paper |
|---|---|---|---|---|---|
| Scale S, plain reimplementation | 5 | 57.83 | 61.28 | 3.46 | 60.38 |
| Scale S, balanced head wiring | 5 | **58.99** | 61.89 | 2.90 | 60.38 |
| Scale M, plain reimplementation | 6 | 70.17 | 72.20 | 2.03 | 71.01 |
| Scale M, balanced head wiring | 7 | **70.80** | 72.58 | 1.78 | 71.01 |
| Scale M, balanced head wiring, 400 epochs | 3 | 70.35 | 72.50 | 2.15 | 71.01 |

Balanced head wiring is the only change that has helped consistently: +1.17 at S (p = 0.049) and +0.62 at M (p = 0.023, uncorrected). At 400 epochs at M the gain is gone (baseline 70.49). Everything else that was tried is summarized below; per-seed numbers are in [`results/`](results/) and write-ups in [`docs/`](docs/).

| Tried | Outcome | Where |
|---|---|---|
| Textbook dropout (head, conv, all layers; S and M) | No gain beyond seed noise | `docs/report_gumbel_freeze_dropout.md`, `docs/addendum_bias_alldropout.md` |
| Permutation dropout (binary-preserving) | Hurts in soft training (56.0 / 52.0 / 46.4 at p 0.05 / 0.1 / 0.2); neutral in the binary phase | `docs/addendum3_perm_bias_scaleM.md`, `results/ste_permx_results.csv` |
| Gumbel straight-through (Yousefi et al. 2025) | Fails: 17.4 at the paper setting, 43.5 at best (pass-through bias 20) | `docs/diagnosis_gumbel_freeze.md` |
| TaskFreeze / GlobalFreeze (Gerlach 2026) | Closes the gap (M 0.58) by lowering soft accuracy; hard does not rise | `docs/report_gumbel_freeze_dropout.md` |
| Concat skip connections | Within noise; nothing on top of balanced wiring | `results/combo_results_S_M.csv` |
| Residual units in conv stages | No arm beats balanced wiring alone (best 58.91); skip prevents the loss from extra depth (53.16 without it) | `results/res_phase1_S.csv` |
| Longer training (2x, 4x) | +0.85 / +1.27 at S, within noise | `figures/fig_budget_sweep.png` |

## Differences from the paper's recipe

- Random crop and flip (`--augment`). The paper uses none; without it scale M reaches about 68.3.
- 45k/5k train/validation split with best-epoch selection on validation.
- Balanced head wiring, skips, residual units, dropout, freezing and Gumbel are all off by default. Defaults reproduce the paper's architecture (Appendix A.1.1) and optimizer (Appendix A.2).

## Layout

```
difflogic.py          model: 16 relaxed gates, LogicLayer, LogicTree, LogicTreeConv2d, OrPool2d,
                      ResidualUnit, LogicTreeNet, DiffLogicNet, gate counting
train.py              CLI trainer: CIFAR-10 loading, checkpoint/resume, JSONL logs, all options below
test_difflogic.py     test suite (459 checks; run from this folder)
pyproject.toml        ruff lint/format settings
baselines/biper/      BiPer ResNet-18 (CVPR 2024) evaluation and run wrapper
cluster/              Slurm job scripts and analysis scripts (see cluster/README.md)
results/              per-seed and summary CSVs behind every number above
figures/              result figures; make_figures.py redraws figA to figC
docs/                 write-ups, newest last (see docs/README.md)
```

## Setup and tests

```bash
pip install -r requirements.txt
python test_difflogic.py
```

The suite has 459 checks. 22 of them need two files that live on the Tufts cluster, not in this repo yet: `cluster/layer_histograms.py` with `cluster/gate_activity.py`, and `ref/difflogic_before_res.py` (a frozen pre-residual copy of `difflogic.py` used for backward-compatibility checks). Without them the suite prints `437 passed, 0 failed` and names the skipped groups.

CIFAR-10 is read from the Hugging Face mirror `uoft-cs/cifar10` (cached after the first run).

## Training

Scale S, paper architecture with crop/flip (the control used in every comparison):

```bash
python train.py --model conv --scale S --tree-depth 3 --kernel-size 3 --batch-size 128 \
    --optimizer adamw --lr 0.02 --weight-decay 0.002 --thresholds 0.25,0.5,0.75 \
    --val-split 5000 --eval-every 5 --pool max --groups -1 --channels-per-tree 2 \
    --epochs 200 --augment --seed 0 --save-every-epoch --ckpt ckpt.pt --log log.jsonl
```

Use `--scale M` for the large network (add `--accum-steps N` if it runs out of GPU memory; same training, smaller micro-batches). Add options to change one thing at a time:

| Option | What it does | In the paper? |
|---|---|---|
| `--head-wiring balanced` | Head layers draw inputs from shuffled lists so every input is read about equally often | No (best result) |
| `--skip conv\|head\|both` | Binary-safe concatenation skips | No |
| `--res-units R --res-body-layers L --res-merge gate\|xor\|or\|none` | Residual units in the conv stages; `none` is the depth-only ablation | No |
| `--dropout P --dropout-mode half\|scale\|runmean\|perm\|permx --dropout-where all\|conv1\|head\|convlate` | Logic dropout variants; `perm`/`permx` are the binary-preserving permutation versions | No |
| `--ste`, `--ste-start-epoch N --ste-lr LR`, `--init-from CKPT` | Straight-through training: hard forward pass, soft gradients | No |
| `--gumbel --gumbel-tau T --gumbel-scale S` | Gumbel straight-through sampling (Yousefi et al.) | No |
| `--freeze global\|task` | Progressive discretization (Gerlach 2026) | No |
| `--pool max\|prob`, `--groups -1`, `--channels-per-tree 2` | Paper's OR-pooling and channel grouping (Sec. 3.1, 3.4) | Yes |

`python train.py --help` lists every option. Checkpoints store the architecture and refuse to load into a different one; rerunning the same command resumes.

## Not in this repo

- Checkpoints and per-epoch logs: on the Tufts cluster (`~/logictreenet/ckpt/`, `~/biper/runs/`), several GB.
- The October 2026 Colab/Kaggle notebooks and run history (before the pooling fix) are in [`../archive/logictreenet-oct2026/`](../archive/logictreenet-oct2026/). Their numbers are superseded by the table above.
