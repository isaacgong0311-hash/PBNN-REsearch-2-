# PBNN — Baseline Candidates & Testbed Status
_Updated Aug 4, 2026_

## Correction from last session
Primary baseline arXiv ID was wrong: **1602.02505 is an obsolete/withdrawn
version** ("This is an obsolete version, up to date version is available
here: arXiv:1602.02830" — per the arXiv API record itself). The correct,
current paper is:

**arXiv:1602.02830** — "Binarized Neural Networks: Training Deep Neural
Networks with Weights and Activations Constrained to +1 or -1"
Courbariaux, Hubara, Soudry, El-Yaniv, Bengio.

## Paper's reported MNIST numbers (Table 1)
Architecture: MLP, 3 hidden layers, 4096 binary units (Theano) / 2048
(Torch7), L2-SVM output layer, BatchNorm, Adam, 1000 epochs, no conv/
augmentation/pretraining.

| Variant | MNIST test error | Accuracy |
|---|---|---|
| BNN (Theano) | 0.96% | 99.04% |
| BNN (Torch7) | 1.40% | 98.60% |
| No binarization (Maxout, for reference) | 0.94% | 99.06% |

Their headline result: full binarization costs ~0.02–0.46pp accuracy vs.
full-precision, on this benchmark.

## Our testbed — now with a working binary model, not just full-precision
`mnist_testbed.py` implements both models and trains them end-to-end:

- **Full-precision MLP** — 2 hidden layers, 256 units, ReLU, Adam.
- **Binary MLP** — same shape, weights binarized via a straight-through
  estimator (STE: sign() forward, clipped-identity backward, per
  Hubara et al.), activations also binarized, BatchNorm between layers,
  real-valued shadow weights clamped to [-1,1] after each optimizer step.

Verified run, 5 epochs, CPU (~10-12s/epoch):

| Model | Test acc after 5 epochs |
|---|---|
| Full-precision | 97.68% |
| Binary (STE) | 96.34% |
| Gap | 1.34pp |

Our architecture is much smaller than the paper's (256 vs 4096 units, 5 vs
1000 epochs) so these aren't comparable to Table 1 directly — but the STE
training mechanics are confirmed working end-to-end, and the gap is
already in the same ballpark as the paper's, and closing epoch over epoch
(1.34pp at epoch 5, was wider early on). Scaling up units/epochs should
close it further if we want a paper-comparable number later.

## Still open (from bit-op counting side, separate track)
Decision-tree bit-op count (~15–25 bit ops/sample) vs. BNN forward-pass
bit-op count (~537,600 bit ops/sample) — completed last session, flagged
for discussion: does the tree count as a legitimate efficiency baseline
given the accuracy gap (~86% tree vs ~96–99% BNN)?

## Next steps
- Decide with Liping/Alex: scale up the binary MLP (more units/epochs) to
  get a paper-comparable number, or is the small-scale demo sufficient to
  move on to the actual PBNN (LP-relaxation) training method next?
- Sync with Alex — check if he found the same candidates or others.
- If moving to PBNN training next: reuse the STE binary layers here as the
  "hard" baseline to compare the LP-relaxation training method against.
