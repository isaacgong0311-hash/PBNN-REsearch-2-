# LP relaxation (original PBNN method)

Prof. Liu's proposed method for training a pure binary neural network: relax the binary training problem to a linear program, solve it per sample, and get a subgradient for the parameters from the LP duals. The full method is in [`documentation.md`](documentation.md) / [`documentation.pdf`](documentation.pdf).

## Finding: the relaxation is degenerate

On a toy problem the LP always reaches its best possible objective, whatever the parameters are. The subgradient is exactly zero and the loss stays flat at −1.0 every epoch. A trained model the LP scores as perfect on all 8 toy samples gets **0 of 8** right under the real binary forward pass.

**Why:** the intermediate layer variables (z, g, h) are free LP variables, tied to each other only by per-gate McCormick (AND), OR-reduce and XOR inequalities. They are never forced to equal the true forward pass, so the solver finds a slack solution that satisfies every constraint without the parameters being a working classifier.

**Ruled out:**

- *Interval-bound tightening* (standard LP/MIP preprocessing, stays inside the method). Still fully degenerate: the OR-reduce upper bound, a sum over inputs, widens intervals so fast that by layer 2 the output interval is back to [0, 1].
- *Sequential soft forward pass* (product t-norm AND, probabilistic-sum OR). Trains, and gets 8/8, but it isn't an LP anymore, so it's out of scope for this method.

**Open question:** does a tighter valid LP relaxation exist for this composed AND / OR-reduce / XOR architecture, or does the project need a different method? Full write-up: [`notes/LP_degeneracy_root_cause_aug10.md`](notes/LP_degeneracy_root_cause_aug10.md).

## Files

| File | What it is |
|---|---|
| `toy_lp_testbed.py` | The LP relaxation on a toy task: McCormick AND, OR-reduce bounds, XOR linearization, per-sample `scipy.optimize.linprog`, subgradient from the duals |
| `toy_lp_testbed_entropy.py` | Variant with an entropy penalty term (one of the candidate fixes) |
| `run_all.sh` | Runs both testbeds |
| `mnist_testbed.py` | MNIST baselines: full-precision MLP and binarized MLP with a straight-through estimator |
| `pbnn_baseline_exploration.ipynb` | Early baseline exploration notebook |
| `step2_beginner_guide.md` | Plain-language walkthrough of the method |
| `notes/` | Baseline candidates, BEP vs. LP comparison, Courbariaux reference numbers, the Aug 10 status, the degeneracy write-up, meeting positions |

## MNIST baselines (Aug)

| Model | Test accuracy | Notes |
|---|---|---|
| Full-precision MLP | 97.3% | upper reference |
| Binarized MLP, straight-through estimator (Courbariaux et al. 2016, [arXiv:1602.02830](https://arxiv.org/abs/1602.02830)) | 95.4% | hidden 512, 930,816 bit-ops per sample |
| Decision tree, bit-op count | ~86% | ~25 bit-ops per sample; Liping later dropped this comparison |

## Running

```bash
pip install numpy scipy torch torchvision
bash run_all.sh
python mnist_testbed.py
```
