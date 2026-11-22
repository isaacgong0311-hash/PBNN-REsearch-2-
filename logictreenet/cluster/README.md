# Cluster scripts (not yet copied here)

These live on the Tufts HPC in `~/logictreenet/cluster/` and should be committed to this folder:

- Slurm job scripts used for every run in `../results/` (S and M seeds, dropout, Gumbel, freezing, residual units, 400-epoch runs).
- `layer_histograms.py` and `gate_activity.py`: per-layer gate-output histograms and dead-gate counts (`../figures/fig_gate_ones_by_layer_S.png`, `../figures/fig_perm_bias_dead.png`). `../test_difflogic.py` checks them when they are present.

Also copy `~/logictreenet/ref/difflogic_before_res.py` to `../ref/` so the residual-unit backward-compatibility checks run.
