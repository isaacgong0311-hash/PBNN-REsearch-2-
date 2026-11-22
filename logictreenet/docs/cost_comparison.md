# Accuracy against inference cost: LogicTreeNet vs BiPer ResNet-18 (CIFAR-10)

Written 2026-11-22. All counts are per image at inference. Source: `../results/cost_comparison.csv`.

| Model | Hard top-1 | Binary ops | Full-precision MACs | Other full-precision ops | OPs (Bi-Real convention) |
|---|---|---|---|---|---|
| LogicTreeNet S, balanced wiring (5 seeds) | 58.99 | 0.96M two-input gates | 0 | 0 | 0.015M |
| LogicTreeNet M, balanced wiring (7 seeds) | 70.80 | 7.68M two-input gates | 0 | 0 | 0.12M |
| BiPer ResNet-18, released checkpoint | 93.76 | 547.4M 1-bit MACs | 1.77M | 1.29M | 10.3M |
| BiPer ResNet-18, our retrain, clean split (3 seeds) | 92.65 | 547.4M 1-bit MACs | 1.77M | 1.29M | 10.3M |

**How each column was counted.**
- LogicTreeNet gates: `count_gates` in `../difflogic.py`, the paper's method (every spatial placement of a conv tree counts, 3 ORs per 2x2 pool window, one gate per head neuron; GroupSum adders not counted). Built with the run options; balanced wiring does not change the count. These are before logic synthesis; the paper reports synthesized counts, which are lower.
- BiPer binary MACs: the 16 binary 3x3 convs, from layer shapes (547,356,672) and confirmed by forward hooks on BiPer's own `resnet18_1w1a` at commit `0c07fdd`.
- BiPer full-precision MACs: first conv 1,769,472 + final linear 5,120.
- BiPer other full-precision ops: BatchNorm outputs 557,568 + per-channel alpha scaling 491,520 + residual adds 245,760 (hooks and block shapes). Hardtanh clipping not counted. Alpha can be folded into the following BatchNorm at inference, which would remove 0.49M.
- OPs (Bi-Real Net / ReActNet convention): binary ops / 64 + full-precision MACs. For LogicTreeNet this treats one logic gate like one binary MAC, which favors BiPer (a 1-bit MAC is an XNOR plus a popcount accumulate). Element-wise ops are left out, as that convention does.

**Reading.** BiPer is about 23 points more accurate and uses about 86x the OPs of LogicTreeNet M by this convention, and about 71x the raw binary operation count. It also needs full-precision arithmetic (about 3.1M operations), which LogicTreeNet does not. The two models sit at very different points of the cost curve, so a single accuracy comparison says little. A fair comparison would need either a cheaper binary ResNet (for example ResNet-20, which BiPer also reports: 86.98) or a larger LogicTreeNet (scale B, 5-bit input, not implemented).

**Caveats.** The BiPer 93.76 is selected on the test set by their code; 92.65 is our clean number. Gate counts are not hardware cost (no wiring, memory or latency), and logic gates and binary MACs are not the same unit.
