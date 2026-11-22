
# PBNN research (Liu lab, Tufts)

Isaac Gong's work for the pure binary neural network (PBNN) project with Prof. Liping Liu.

## Headline result

A PyTorch reimplementation of LogicTreeNet (Petersen et al., NeurIPS 2024), the logic-gate baseline for the lab's method, reaches **70.80% hard (pure binary) accuracy on CIFAR-10 at scale M** (7 seeds) against the paper's 71.01%, and 58.99% at scale S against 60.38%. Details in [`logictreenet/`](logictreenet/).

| Model (CIFAR-10 test, crop/flip, 200 epochs) | Paper | Soft | Hard (pure binary) | Gap |
|---|---|---|---|---|
| LogicTreeNet S, balanced head wiring (5 seeds) | 60.38% | 61.89% | 58.99% | 2.90 pts |
| LogicTreeNet M, balanced head wiring (7 seeds) | 71.01% | 72.58% | **70.80%** | 1.78 pts |
| DiffLogicNet, dense, small (Aug) | 51.27% | 49.65% | 49.56% | 0.1 pts |

## What's in here

| Folder | What it is | When |
|---|---|---|
| [`lp-relaxation/`](lp-relaxation/) | The original PBNN method (LP relaxation of the binary training problem), a toy testbed, the MNIST testbed, and the diagnosis of why the relaxation is degenerate | Jul–Aug |
| [`difflogicnet-dense/`](difflogicnet-dense/) | First CIFAR-10 logic-gate baseline: dense DiffLogicNet (Petersen et al., NeurIPS 2022) | Aug–Sep |
| [`logictreenet/`](logictreenet/) | Main work: LogicTreeNet (Petersen et al., NeurIPS 2024) reimplementation, results tables, figures, write-ups, BiPer baseline | Sep–Nov |
| [`archive/`](archive/) | October Colab/Kaggle notebooks and run history, superseded | Sep–Oct |
| [`notes/`](notes/) | Meeting notes and write-ups | |

## Timeline

1. **Jul 29–Aug 10: LP relaxation.** Built a toy testbed of the method in `lp-relaxation/documentation.md`. Found the relaxation is degenerate: subgradient exactly zero, loss flat at −1.0, and a trained model that the LP scores as perfect gets 0/8 toy samples right under the real binary forward pass. Root cause: the intermediate layer variables are free LP variables only loosely tied by per-gate McCormick/OR/XOR constraints. Interval-bound tightening did not fix it, because the OR-reduce upper bound widens intervals back to [0, 1] within two layers. This is an open research question, not a bug.
2. **Aug 3–13: MNIST baselines.** Full-precision MLP (97.3%), decision-tree bit-op baseline (~86%), and a binarized MLP with a straight-through estimator following Courbariaux et al. 2016 (95.0% val / 95.4% test). Liping then redirected the benchmark to CIFAR-10 and bigger models.
3. **Aug 18–Sep 14: DiffLogicNet.** Dense logic gate network on CIFAR-10, 49.6% hard vs. the paper's 51.3%. Fixed a dead-gradient bug from zero-initialized gate logits.
4. **Sep 14–Oct 7: LogicTreeNet, first pass.** Per Liping, moved to the best-performing logic gate model on CIFAR-10. Fixed two architecture mismatches with the paper, ran seven training variants, measured ±3 pt run-to-run noise in hard accuracy, then found that a straight-through fine-tune reliably closes the soft/hard gap.
5. **Oct 8–Nov 21: LogicTreeNet on the Tufts cluster.** Fixed OR-pooling (max t-conorm) and channel grouping to match the paper, which closed most of the soft/hard gap. Added crop/flip. Tested balanced head wiring (the one consistent gain), dropout variants including Liping's permutation dropout, Gumbel sampling, TaskFreeze, skips, residual units, and a BiPer ResNet-18 baseline. See `logictreenet/README.md`.

## Not in this repo

- Checkpoints, CIFAR-10 data and per-epoch logs: on the Tufts cluster, several GB.
- Cluster job scripts and the gate-histogram scripts: see `logictreenet/cluster/README.md`.
- The Courbariaux BNN benchmark script (`code/benchmark/`) from August; its results are in `lp-relaxation/notes/`.
