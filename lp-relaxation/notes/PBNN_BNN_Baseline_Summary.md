# PBNN Project — BNN Literature Baseline: Implementation & Status

Faithful re-implementation of Courbariaux, Hubara, Soudry, El-Yaniv & Bengio (2016), *Binarized Neural Networks*, arXiv:1602.02830 — Section 2.1 ("MLP on MNIST, Theano") — as the primary literature baseline for the PBNN accuracy comparison.

## What was built (Aug 6 session)
Intended to live at `code/benchmark/` in `github.com/tufts-ml/pbnn`:
- `binary_layers.py` — Sign() binarization + saturating straight-through estimator
- `model.py` — 784→4096→4096→4096→10, BatchNorm + Dropout, L2-SVM/squared-hinge output
- `train.py` — Algorithm 1's training loop: per-layer Glorot-scaled ADAM learning rates, exponential decay, weight clipping to [-1,1] after every step, validation-based model selection
- `bitops.py` — bit-operation counter for the accuracy/cost comparison against the decision-tree baseline
- `verify_binary.py` — standalone script checking which tensors are exactly ±1 at forward-pass time (built in response to Liping's question about whether the implementation is "fully binary")

**Status note:** git push from the sandbox was blocked (no auth) every time this was attempted (Aug 4, Aug 6). Files were delivered as chat outputs instead — **not confirmed to be in the actual repo.** Check `github.com/tufts-ml/pbnn/tree/main/code/benchmark` directly before trusting this description or re-deriving the code.

## Binary verification (what the check found)

| Component | Status | Value at forward pass |
|---|---|---|
| Hidden-layer weights (all 3 layers) | Binary | {-1.0, +1.0} exactly |
| Output-layer weights | Binary | {-1.0, +1.0} exactly |
| Hidden-layer activations (all 3 layers) | Binary | {-1.0, +1.0} exactly |
| Input pixels | Real (by design) | Sec 1.6: only layer 1's input is exempt |
| Final output score | Real (by design) | Algorithm 1: a_L feeds the loss directly, never binarized |
| Shadow weights (SGD accumulator) | Real (by design) | Sec 1.2: required for gradient accumulation |
| Gradients | Real (by design) | Stated explicitly in Algorithm 1 |
| BatchNorm γ/β | Real (by design) | Vanilla BN used for Theano MNIST MLP in the paper |

## Validation run (hidden=512, reduced from paper's 4096 to fit a single CPU core)

| Epoch | Train loss | Val acc | Test acc @ best val |
|---|---|---|---|
| 1 | 0.5305 | 85.35% | 84.63% |
| 5 | 0.1001 | 92.76% | 92.23% |
| 11 | 0.0701 | 94.94% | 94.61% |

Loss decreases monotonically, accuracy climbs every epoch through 11/12 epochs (12th cut off by sandbox time limit) — confirms STE gradient, weight clipping, and squared-hinge loss are all correctly wired.

## Numbers at a glance

| Model | Test accuracy | Bit-ops/inference | Source |
|---|---|---|---|
| Decision tree baseline | ~86% | ~20 | This project |
| Small demo Binary MLP (784-256-256-10) | 96.34% | ~537,600 | This project |
| Faithful BNN, hidden=512 (in progress, 11/12 epochs) | 94.61% | 930,816 | This project |
| Faithful BNN, hidden=4096, 1000ep | 99.04% (0.96% error) | 36,806,656 | Paper Table 1 (not reproduced here) |

## Open next step from this thread
Run `train.py --hidden 4096 --epochs 1000` on real GPU compute for a paper-comparable number — **not done, needs real compute, not a sandbox task.**
