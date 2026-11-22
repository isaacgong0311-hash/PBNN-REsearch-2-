# Write-ups (oldest first)

| File | Date | Covers |
|---|---|---|
| `report_gumbel_freeze_dropout.md` | Nov 3 | Gumbel sampling, TaskFreeze/GlobalFreeze, training length, dropout at S and M, torchlogix check |
| `diagnosis_gumbel_freeze.md` | Nov 3 | One-page diagnosis of the Gumbel failure (written before the torchlogix runs finished) |
| `addendum_bias_alldropout.md` | Nov 10 | Pass-through bias with Gumbel, dropout on all layers, head-dropout confirmation seeds |
| `addendum3_perm_bias_scaleM.md` | Nov 12 | Permutation dropout, Gumbel bias 15/20, scale M skips, dead gates over 10 seeds |
| `report_residual_biper_2026-11-19.md` | Nov 19 | Residual units design and speed, BiPer setup. Written before the residual runs finished: final numbers are in `../results/res_phase1_S.csv` and the main README |

p-values throughout are Welch tests, not corrected for multiple comparisons.
