
## Headline result

A straight-through fine-tune of LogicTreeNet (a convolutional logic gate network) brings the discrete, pure-binary network to **60.43% on CIFAR-10** at scale M, with the soft/hard gap down from 6.1 points to 1.0. Details in [`logictreenet/`](logictreenet/).

| Model (CIFAR-10 test) | Paper | Soft | Hard (pure binary) | Gap |
|---|---|---|---|---|
| DiffLogicNet, dense, small | 51.27% | 49.65% | 49.56% | 0.1 pts |
| LogicTreeNet S, paper settings | 60.38% | 57.40% | 53.54% | 3.9 pts |
| LogicTreeNet S + straight-through fine-tune | | 57.34% | 54.16% | 3.2 pts |
| LogicTreeNet M, paper settings | 71.01% | 62.57% | 56.45% | 6.1 pts |
| **LogicTreeNet M + straight-through fine-tune** | | 61.39% | **60.43%** | **1.0 pts** |

## What's in here

| Folder | What it is | When |
|---|---|---|
| [`lp-relaxation/`](lp-relaxation/) | The original PBNN method (LP relaxation of the binary training problem), a toy testbed, the MNIST testbed, and the diagnosis of why the relaxation is degenerate | Jul–Aug |
| [`difflogicnet-dense/`](difflogicnet-dense/) | First CIFAR-10 logic-gate baseline: dense DiffLogicNet (Petersen et al., NeurIPS 2022) | Aug–Sep |
| [`logictreenet/`](logictreenet/) | Main work: LogicTreeNet (Petersen et al., NeurIPS 2024) reimplementation, every training run, and the straight-through fine-tune | Sep–Oct |
| [`notes/`](notes/) | Meeting notes and write-ups | |

## Timeline

1. **Jul 29–Aug 10: LP relaxation.** Built a toy testbed of the method in `lp-relaxation/documentation.md`. Found the relaxation is degenerate: subgradient exactly zero, loss flat at −1.0, and a trained model that the LP scores as perfect gets 0/8 toy samples right under the real binary forward pass. Root cause: the intermediate layer variables are free LP variables only loosely tied by per-gate McCormick/OR/XOR constraints. Interval-bound tightening did not fix it, because the OR-reduce upper bound widens intervals back to [0, 1] within two layers. This is an open research question, not a bug.
2. **Aug 3–13: MNIST baselines.** Full-precision MLP (97.3%), decision-tree bit-op baseline (~86%), and a binarized MLP with a straight-through estimator following Courbariaux et al. 2016 (95.0% val / 95.4% test). Liping then redirected the benchmark to CIFAR-10 and bigger models.
3. **Aug 18–Sep 14: DiffLogicNet.** Dense logic gate network on CIFAR-10, 49.6% hard vs. the paper's 51.3%. Fixed a dead-gradient bug from zero-initialized gate logits.
4. **Sep 14–Oct 4: LogicTreeNet.** Per Liping, moved to the best-performing logic gate model on CIFAR-10. Fixed two architecture mismatches with the paper, ran seven training variants, measured ±3 pt run-to-run noise in hard accuracy, then found that a straight-through fine-tune reliably closes the soft/hard gap.

## Not in this repo

- The faithful Courbariaux BNN benchmark script (`code/benchmark/`) was delivered as files in August but isn't in any of the source folders this repo was assembled from. Its results are recorded in `lp-relaxation/notes/`.
- Checkpoints (`*.pt`) and CIFAR-10 data are excluded; they're large and regenerate from the notebooks.
- Executed notebooks for LogicTreeNet S runs 6–8 weren't saved locally. Their results are in `logictreenet/README.md`.
