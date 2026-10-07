# PBNN — Status as of Aug 10, 3:XXpm session

## Where things actually stand

**The real blocker (core method):** the LP relaxation is degenerate. Subgradient
comes out exactly zero, loss flatlines at -1.0 — found July 30 in
`toy_lp_testbed.py` (built against Liping's formulation doc: McCormick envelope
for AND, OR-reduce bounds, XOR linearization, per-sample LP solve via
`scipy.optimize.linprog`, subgradient via dual/marginal values). This is still
unsolved as of the last session touching it (Aug 9). **This is the highest-value
thing to spend time on.**

**Baselines (scaffolding, not the research question):**
- Tree bit-op baseline: ~86% MNIST acc, ~15-25 bit ops/sample. Done (Aug 3).
- Full-precision MLP testbed: 97.68% (small run). Done (Aug 3-4). Recovered
  script: `code/mnist_testbed.py` / `code/pbnn_baseline_exploration.ipynb`.
- Faithful BNN literature baseline (Courbariaux et al., arXiv:1602.02830,
  Algorithm 1): built Aug 6 as `code/benchmark/` (binary_layers.py, model.py,
  train.py, bitops.py, verify_binary.py) — **git push kept failing (no auth in
  sandbox), so this was never confirmed to land in
  github.com/tufts-ml/pbnn. Check the repo directly.** Validation run hit
  94.61% at epoch 11/12 (hidden=512, reduced from paper's 4096). See
  `notes/PBNN_BNN_Baseline_Summary.md` for full numbers.
- Aug 6 meeting with Liping + Alex happened. Q1 (tree comparison legitimacy)
  and Q2 (scale up MLP vs. move to LP debugging) were both resolved in favor
  of moving to LP debugging — baselines are considered "good enough" scaffolding.

**Aug 9 session:** read the BEP paper (Colombo et al., arXiv:2512.04189) for a
structural comparison against PBNN's LP relaxation. Key finding, in
`notes/BEP_vs_LP_and_Courbariaux_sanity_check.md`: BEP's relaxation is
provably tight *because it's separable* — each layer's subproblem decouples
into independent 1-D sign problems. **If PBNN's LP couples variables across
neurons/layers (which the McCormick/OR/XOR construction likely does), that
coupling is a plausible root cause of the zero-subgradient degeneracy** —
worth checking directly against the actual constraint structure.

## The gap I can't close from memory alone
- `toy_lp_testbed.py` itself (the actual degenerate LP code) and Liping's
  original formulation doc (the `u_j` / Related Work material) live in the
  `tufts-ml/pbnn` repo and/or shared docs — not recoverable from chat memory
  or your Drive. I don't have current read access to the private repo from
  this sandbox (tried — no git credentials, same issue as every push
  attempt).
- Practical options for this hour: (a) paste/upload the current
  `toy_lp_testbed.py` and formulation doc content directly and I'll dig into
  the coupling-vs-separability question concretely against your actual
  constraints, or (b) if you don't have it handy, I can rebuild a toy LP
  relaxation from the documented structure (McCormick AND, OR-reduce, XOR
  linearization) and test the separability hypothesis on that — useful as a
  sanity check, but it's a reconstruction, not your actual code, so any fix
  found there needs porting back and re-verifying against the real repo.

## Files in this folder
- `code/mnist_testbed.py`, `code/pbnn_baseline_exploration.ipynb` — recovered baseline testbed
- `notes/BEP_vs_LP_and_Courbariaux_sanity_check.md` — the separability lead on the degeneracy
- `notes/baseline_candidates.md`, `notes/PBNN_BNN_Baseline_Summary.md` — baseline reference material
