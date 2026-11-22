# Residual units in the conv stages, and BiPer ResNet-18 baseline: status (2026-11-19)

> **Update (Nov 21).** Phase 1 finished. No arm beat balanced wiring alone (58.99): A 58.91, B 58.24, C (no skip) 53.16, D (xor) 29.10, E 58.05. See `../results/res_phase1_S.csv`. BiPer retrains: 93.32 (freq 20), 93.35 (freq 30), 92.65 on the clean split.

Isaac Gong. Scale S / M of LogicTreeNet on CIFAR-10 unless noted. Hard = fully binary network, soft = relaxed network, gap = soft minus hard. Paper: S 60.38, M 71.01.

## 1. Residual units (Prof. Liu's suggestion from the Nov 18 meeting; not in the paper)

**What was built.** Inside each conv stage, after the stage's conv block and before its OR-pool, a stack of shape-preserving logic conv layers (the body), then a merge gate that combines the stage input x with the body output y. The merge is one learnable 16-op gate per channel (shared over positions), initialized as pass-through of x, so a unit starts as the identity. Options (all off by default): `--res-units R`, `--res-body-layers L`, `--res-tree-depth`, `--res-merge gate|xor|or|none`, `--res-stages all|2,3,4`, `--res-merge-bias`. `--res-merge none` is the depth-only ablation (no skip). The existing `--skip conv` is unchanged. The merge works in the soft, hard, straight-through, Gumbel and TaskFreeze paths and with freezing; it raises a clear error with `--skip conv/both` and `--dropout-mode runmean`.

**Checks.** The suite is now 459 checks (419 old, 40 new), all passing locally and on the cluster. They include: `--res-units 0` gives the identical state_dict and identical soft and hard outputs as the previous code; shapes are preserved at S and M channel counts; at init the hard unit output equals x exactly; XOR and OR truth tables; hard output is exactly 0/1 end to end and equals the straight-through forward; gradients reach the stage-1 body; TaskFreeze probes and freezing work with merge gates; checkpoint round trip and old checkpoints load; gate counts match a hand count (S default stays 83,552 trainable gates).

**Phase 0 (scale S, one A100, balanced head wiring, crop/flip, seed 0).** Per-run table: `res_phase0_S.csv`.

| Config | Seconds per epoch | Peak GPU memory | Trainable gates | Gates counted the paper's way |
|---|---|---|---|---|
| No residual units | 27.7 | 1.4 GB | 83,552 | 960,512 |
| 1 unit, 1 layer, gate | 60.2 | 2.7 GB | 97,120 | 1,878,016 |
| 1 unit, 2 layers, gate | 91.9 | 3.9 GB | 108,992 | 2,680,832 |
| 1 unit, 2 layers, no skip | 90.6 | 3.8 GB | 107,296 | 2,566,144 |
| 2 units, 1 layer, gate | 92.8 | 4.1 GB | 110,688 | 2,795,520 |
| 1 unit, 2 layers, gate, stages 2-4 only | 74.9 | 3.2 GB | 108,512 | 2,189,312 |

A 200-epoch run takes about 3.3 h (1 unit, 1 layer) to 5.1 h (2 layers), so no arm needed the stages 2-4 fallback. The 2-epoch smoke runs (hard accuracy after 2 epochs, so only a sanity check): gate 35.1%, or 34.3%, none 39.3%, **xor 10.4% (chance)**. A fixed XOR merge starts as a non-identity and may be a poor choice; arm D tests it.

**Phase 1 (running).** Slurm array 7386584, 25 runs: arm A 1 unit/1 layer/gate, B 1/2/gate, C 1/2/none (same depth, no skip), D 1/2/xor, E 2/1/gate; 5 seeds each (0-4), crop/flip, 200 epochs, balanced head wiring. Control: balanced wiring alone, existing runs `screenS_balanced_aug1_s0..4` (58.99; same recipe). Rule: an arm goes to scale M only if it beats the control by more than about 0.5 hard with Welch p < 0.1 and also beats arm C. **No Phase 1 results exist yet.**

**Differences from the paper's recipe.** The residual units themselves; balanced head wiring; random crop/flip; 45k/5k train/validation split with best-validation selection.

## 2. BiPer ResNet-18 (CVPR 2024, arXiv:2404.01278): Phase 0 on CIFAR-10

**Setup.** Repo cloned into its own folder (not vendored), pinned commit `0c07fdd6f7ada649b1bb4576174e54731f615ff6` (2024-08-11). Released CIFAR-10 and ImageNet ResNet-18 checkpoints downloaded from the README's Google Drive links. The CIFAR-10 checkpoint was evaluated with BiPer's own model code (`models_cifar/resnet2.py`, `modules/binarized_modules.py`, stage-2 mode), their test transform, on a laptop CPU, with the CIFAR-10 test split from the Hugging Face `uoft-cs/cifar10` mirror. Script: `biper_eval_cifar.py`; outputs: `biper_cifar_eval_freq30.json`, `biper_cifar_eval_freq20.json`.

| | Top-1 | Paper |
|---|---|---|
| Released checkpoint, frequency 30 | **93.76** | 93.75 |
| Same checkpoint, frequency 20 | 93.38 | 93.75 |

The checkpoint stores epoch 285 and best accuracy 93.75, so the 0.01 difference is rounding on our evaluation of the same test set.

**Findings.**
1. **Frequency.** The paper and the run script use frequency 20, but this CIFAR-10 checkpoint was trained at **30** (its folder name says `freq=30`) and scores 93.76 at 30 versus 93.38 at 20. The repo's `eval.sh` passes no `--freq`, and the default is 30, so CIFAR-10 evaluation works as released. The ImageNet checkpoint's config says `freq: 20.0`, so evaluating it with the default would use the wrong frequency; pass `--freq 20` explicitly. The README's ImageNet commands also use names that do not match the repo (`main_stg2.py`, `--load_ckpt_2tage`; the repo has `main_stage2.py` and `--load_ckpt_stage1`).
2. **The binary weights are really +-1.** In all 16 binary conv layers, sign(sin(30 w)) has no zeros and about 50% +1 in every layer (0.498 to 0.503).
3. **It is not a pure binary network.** Full precision: the first conv (3 to 64 channels), every BatchNorm, the final linear layer and its BatchNorm. Also real-valued: the per-channel scale `alpha` after each binary conv, hardtanh, and the real residual adds (the shortcut is an average pool, no binary conv). Counted per image: binary conv MACs 547,356,672 (99.7% of all MACs), first conv 1,769,472 and linear 5,120 full-precision MACs (computed from the layer shapes and cross-checked against a hook count). Parameters 11.0M.
4. **The 93.75 is selected on the test set.** In BiPer's code the "validation" loader is the CIFAR-10 test split, and the best epoch is chosen by it. Our LogicTreeNet numbers pick the best epoch on a held-out 5k validation split and touch the test set once, so the two numbers are not selected the same way.

**Not done.** The ImageNet checkpoint is downloaded but not evaluated (no ImageNet validation data here, and the cluster check is not done). No retraining (600 + 300 epochs on CIFAR-10, 3 seeds, two splits): not benchmarked yet. No second-split ("our-split", 45k/5k/10k) numbers.

**Comparison caveat.** 93.75 for BiPer versus 70.8 for LogicTreeNet M is not like for like: BiPer keeps full-precision first and last layers, BatchNorm, scaling and real residual sums. The fair axis is accuracy against inference cost.

## 3. Open questions for Liping
1. Is BiPer the baseline he meant, or a generic binary ResNet-18 (Bi-Real, IR-Net, ReCU)?
2. ImageNet: for the binary ResNet baseline only, or should LogicTreeNet also go on ImageNet (the LogicTreeNet paper reports none; downsampled 32x32 or 64x64 ImageNet would keep our architecture nearly unchanged)?
3. Accuracy only, or accuracy against inference cost (recommended, since BiPer is not pure binary)?
4. Merge gate learnable or fixed, units per stage, and all stages or only the later ones (defaults used: learnable, 1 to 2, all)?
