# Handoff to Claude Science (cluster work), 2026-11-22

Paste this into Claude Science. Do the items in order. Report hard, soft and gap for every run, mean and SD over seeds, and Welch p against the named control. Never invent a number.

## 0. Sync the code first (no GPU)

The repo copy of `difflogic.py` / `train.py` (cleaned 2026-11-21, behavior-identical to the Drive copy) has **no option for the pass-through starting bias**: `_residual_gate_init` hard-codes `bias=5.0`. The Nov 12 bias 7/10/15/20 runs and the Nov 18 meeting doc ("starting-bias option") say the cluster copy has one. So:

1. Diff the cluster `~/logictreenet/{difflogic,train,test_difflogic}.py` against `logictreenet/` in the repo. If the cluster copy is newer, it is the source of truth: run `ruff format` with the repo's `pyproject.toml` on it and copy it into the repo.
2. Copy into the repo: `cluster/` (all Slurm job scripts, `layer_histograms.py`, `gate_activity.py`) and `ref/difflogic_before_res.py`. Then `python test_difflogic.py` must report 459 passed with nothing skipped.
3. Push the `logictreenet/` folder to `tufts-ml/pbnn` in its own folder (Liping's Nov 11 request).

## 1. Scale M balanced wiring: more seeds (main priority)

Why: balanced wiring is +0.62 hard at 200 epochs (70.80, n = 7, vs 70.17, n = 6; Welch p = 0.023, uncorrected). Pooled seed SD is 0.43, which gives only about 65% power at the current n. 9 seeds per arm gives 80% power for a 0.62 difference.

- Run: baseline seeds 6, 7, 8 and balanced seeds 7, 8. Scale M, crop/flip, 200 epochs, same job script and recipe as `results/scaleM_balanced_vs_baseline.csv`. 5 runs.
- Analysis, fixed now: Welch test on all 9 vs 9 seeds, hard accuracy. Report the effect with a 95% CI. Do not drop or rerun any seed.
- Append rows to `results/scaleM_balanced_vs_baseline.csv` in the same format.

## 2. Gumbel failure: direct test of the initialization explanation (lower priority)

The reports' untested hypothesis: Gumbel fails here because of the pass-through init (one gate at logit 5, 15 gate layers in series) and because the noise is shared over the batch and positions. Yousefi et al. use N(0, 1) init on dense nets.

- Add, behind flags that are off by default, with tests:
  - `--gate-init residual|gaussian` (gaussian = logits ~ N(0, 1), no pass-through bias), threaded to all six `_residual_gate_init` call sites.
  - `--gumbel-per-example`: one noise draw per example (noise shape gains the batch dimension), still shared over spatial positions of a conv tree.
- Scale S, crop/flip, 200 epochs, 3 seeds each:
  - A: `--gumbel` (noise 1, tau 1) + `--gate-init gaussian`
  - B: `--gumbel` + `--gumbel-per-example` (default residual init)
  - C: `--gate-init gaussian`, no Gumbel (control for A; may not train at all, which is itself the answer)
- Compare with: Gumbel at the paper setting 17.39 (n = 3) and the control 57.77 (n = 3).

## 3. Wait for Liping before

- Any ImageNet run (BiPer only, or LogicTreeNet at 32x32 / 64x64).
- A no-augmentation baseline at scale M.
- If BiPer ImageNet goes ahead: evaluate the released checkpoint with `--freq 20` (its config says 20; the script default is 30), and use `main_stage2.py` / `--load_ckpt_stage1` (the README's names are wrong).

## Already done here (no cluster needed)

- BiPer MAC counts: `biper_eval_cifar.py` is correct; the two JSONs in Drive were written by an older version that divided by the batch size (250). Corrected copies are in `results/`.
- Cost comparison: `docs/cost_comparison.md`.
