# LogicTreeNet-S, CIFAR-10: Gumbel sampling, training length and freezing (diagnosis, 1 page)

Setup for every row: scale S, crop/flip, AdamW lr 0.02, wd 0.002, batch 128, best-validation-hard checkpoint, test set evaluated once.
Budget is stated in epochs (1x = 200 epochs, about 70k steps; Gerlach trains TreeLogicNet-S for 200k steps, about 2.8x our 1x).
Soft = plain softmax network, hard = argmax gates, gap = soft minus hard (percentage points). Mean +- SD over seeds; p = Welch t-test vs the control with the same budget.

| Arm | Budget | n | Hard | Soft | Gap | p (hard) | p (gap) |
|---|---|---|---|---|---|---|---|
| Control (relaxed training, argmax at the end) | 200 ep | 3 | 57.77 +- 0.85 | 61.20 +- 0.64 | 3.43 +- 0.27 | | |
| Control | 400 ep | 3 | 58.62 +- 1.22 | 61.96 +- 0.47 | 3.34 +- 0.78 | 0.38 (vs 200 ep) | 0.87 |
| Control | 800 ep | 3 | 59.04 +- 0.52 | 62.25 +- 0.61 | 3.21 +- 0.30 | 0.11 (vs 200 ep) | 0.40 |
| Gumbel, paper setting (noise 1, tau 1) | 200 ep | 3 | 17.39 +- 5.16 | 18.36 +- 7.33 | 0.97 +- 2.45 | 0.004 | 0.22 |
| Gumbel, noise 0.1 (chosen by validation) | 200 ep | 3 | 40.92 +- 0.15 | 37.90 +- 3.78 | -3.02 +- 3.69 | 0.001 | 0.09 |
| Gumbel, noise 0.1 | 400 ep | 3 | 41.88 +- 0.40 | 37.90 +- 6.10 | -3.98 +- 5.88 | 0.001 | 0.16 |
| Gumbel, noise 0.1 | 800 ep | 3 | 43.22 +- 0.43 | 41.34 +- 3.64 | -1.88 +- 3.88 | <0.001 | 0.15 |
| TaskFreeze | 200 ep | 3 | 57.72 +- 1.15 | 58.50 +- 0.84 | 0.78 +- 0.38 | 0.95 | 0.001 |
| TaskFreeze | 400 ep | 2 | 58.47 +- 0.37 | 59.52 +- 1.17 | 1.05 +- 1.55 | 0.86 | 0.25 |
| GlobalFreeze (literal: all gates fixed at epoch 25) | 200 ep | 3 | 54.78 +- 0.47 | 56.07 +- 1.80 | 1.29 +- 2.23 | 0.012 | 0.24 |
| GlobalFreeze, straight-through reading (our interpretation) | 200 ep | 3 | 55.85 +- 0.36 | 52.72 +- 1.79 | -3.13 +- 2.08 | 0.044 | 0.03 |
| Same, straight-through reading | 400 ep | 2 | 56.74 +- 0.90 | 54.19 +- 2.95 | -2.55 +- 2.05 | 0.15 | 0.13 |

Paper reference for scale S (Petersen et al. 2024): 60.38. Every number above comes from a logged run (all_runs_gumbel_freeze.csv).

## Diagnosis

1. **Undertraining is not the explanation for Gumbel.** Going from 200 to 800 epochs moves Gumbel (noise 0.1) from 40.9 to 43.2 hard, still 15.8 points below the control at 800 epochs. The control itself gains only 1.3 points (p = 0.11) and its gap changes from 3.43 to 3.21 (p = 0.40, inside seed noise). Gerlach's observation that the gap shrinks with training has the right sign here but is not distinguishable from noise with 3 seeds.
2. **Gumbel noise breaks learning in this recipe, monotonically in the noise scale.** Hard accuracy at seed 0, 200 epochs: noise 0 (straight-through, no noise) 49.50, noise 0.1: 41.09, 0.25: 41.10, 0.5: 35.20, 1 (paper): 20.20, 2: 12.80. Even zero noise (discrete forward pass from step 0) is 8 points below the control. tau 0.25 / 2 at noise 1: 25.04 / 23.66. The paper's own optimizer (Adam, lr 0.01, no weight decay) at noise 1 gives 23.04 and at noise 0.5 gives 40.66 vs 35.20 with AdamW: a small effect, so the optimizer is not the main cause.
3. **Likely structural, not proven.** With the pass-through initialization (logit 5 for one gate), each gate is replaced by a random gate about 9% of the time at the start, and the tree network has 15 sequential gate layers, so most paths are corrupted from the first step. The noise is also shared across the batch and all positions, so it is not averaged out. We did not test this directly. Plesner said tree kernels need no special treatment; with the paper's noise as is, they do not train here, so this needs his check (dense nets use N(0,1) init and Adam lr 0.01, ours uses the residual init).
4. **Soft/hard gap for Gumbel is not a clean measure.** Soft accuracy of Gumbel-trained networks is erratic across seeds (31.0 to 43.9 at 400 and 800 epochs), so gaps range from -10.5 to +1.2. Do not read "gap closed" off these rows; the hard accuracy is the number that matters and it is far below the control.
5. **Unused gates (paper definition: softmax entropy above the 2.5th percentile of N(0,1)-initialised neurons):** control 45.8% (range 45.1 to 46.5 over seeds 0-2); Gumbel noise 0.1 at 200 epochs: 83.8% (n=3: 81.5, 84.5, 85.2). The paper reports 0.00% for Gumbel on dense nets; we do not reproduce that. The wiring-based unread-gate share (27.1%) depends on the fixed wiring only and is unchanged.
6. **Schedule (freezing) closes the gap by lowering soft, not by raising hard.** TaskFreeze at 200 epochs: gap 0.78 vs 3.43 (p = 0.001) with hard unchanged (57.72 vs 57.77, p = 0.95). Gerlach reports TaskFreeze +1.5 over FinalArgmax on TreeLogicNet-S (60.76 vs 59.29); we do not see a hard-accuracy gain at 1x or 2x (58.47 vs 58.62, n = 2). Our budget is 2.8x smaller than his and the 2x row has 2 seeds, so this is not a refutation.
7. **GlobalFreeze is ambiguous for this architecture.** Fixing every gate at epoch 25 leaves nothing trainable (wiring is fixed), so the literal version is just an argmax at epoch 25 (54.78 hard). The straight-through reading (discrete forward pass from epoch 25, gates keep learning, weight decay 0 in that phase) gives 55.85 hard and a negative gap (soft below hard). Both are below the control (p = 0.012 and 0.044). Ask Gerlach what GlobalFreeze keeps trainable in TreeLogicNet.
8. **Deeper and dropout (Prof. Liu):** 9 layers with both skips, 400 epochs: 58.78 +- 0.11 (n = 2) vs 58.62 control at 400 epochs, p = 0.85; at 200 epochs 58.52 vs 57.77, p = 0.32. Dropout p = 0.05 on conv blocks 2 and 3 outputs only: 55.82 +- 0.43, 1.96 below the control (p = 0.039), gap 4.77.

## What is within seed noise
Control budget effects (all p > 0.1), deeper-with-skips at either budget, TaskFreeze hard accuracy at either budget. Seed SD of hard accuracy is 0.85 (n = 10 control seeds, 57.70).

## Not done / caveats
- torchlogix reproduction: script and README written (repro_torchlogix.py) but NOT run; the cluster connection dropped before it could be launched, so there is no torchlogix number yet.
- Gumbel noise level was chosen on validation hard accuracy (noise 0.1, val 41.5%), so the 2x/4x Gumbel runs are not at the paper's setting. The paper setting itself was only run at 200 epochs.
- Seeds: 3 at most; several rows have n = 2. The Gumbel runs use per-forward-call noise shared over batch and positions (as F.gumbel_softmax in the paper's code).
- Gerlach's schedule parameters (start at 12.5% of training, K = 1000, P = 3, beta = 2^(-K/10000)) are from his Appendix B. We used them unchanged.
