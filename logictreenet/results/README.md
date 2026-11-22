# Results tables

Every number in the READMEs and write-ups comes from these files. CIFAR-10 test accuracy in percent; hard = binary network, soft = relaxed, gap = soft minus hard. Crop/flip and 200 epochs unless the name says otherwise.

| File | Contents |
|---|---|
| `scaleM_balanced_vs_baseline.csv` | Scale M per seed: baseline (6) vs balanced head wiring (7) |
| `scaleM_400epoch.csv` | Scale M at 400 epochs, baseline vs balanced (3 seeds each) |
| `scaleM_freezetask_balanced.csv` | Scale M, TaskFreeze + balanced wiring (3 seeds) |
| `scaleM_dropout_crop_flip.csv` | Scale M dropout placements and p (2 seeds each) |
| `combo_results_S_M.csv` | Balanced wiring and conv skip combinations, S and M |
| `extra_runs_dropout_balanced_freeze.csv` | Scale S per run: base, balanced, head dropout, TaskFreeze, 2x budget |
| `welch_extra_runs.csv` | Welch comparisons for the runs above |
| `perm_bias_welch.csv` | Permutation vs textbook dropout; Gumbel bias 15 and 20 |
| `ste_permx_results.csv` | Binary (straight-through) phase with exact-count permutation dropout |
| `gate_ones_dead_share_S.csv` | Dead-gate share by layer (seed-0 models) |
| `res_phase0_S.csv` | Residual units: speed, memory, gate counts, 2-epoch smoke runs |
| `res_phase1_S.csv`, `res_phase1_S_per_run.csv` | Residual units, 5 arms x 5 seeds, vs balanced wiring |
| `biper_r18_cifar10_retrain.csv`, `biper_r18_cifar10_f30_and_splits.csv` | BiPer ResNet-18 retrains |
