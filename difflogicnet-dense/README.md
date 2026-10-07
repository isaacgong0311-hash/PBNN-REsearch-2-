# DiffLogicNet implementation (arXiv:2210.08277)

Faithful implementation of Petersen, Borgelt, Kuehne, Deussen, "Deep Differentiable
Logic Gate Networks" (NeurIPS 2022) — the second CIFAR-10 literature baseline for
the PBNN benchmark repo.

## Files
- **difflogic_net.py** — the algorithm itself:
  - all 16 real-valued logic-gate relaxations (Table 1)
  - `LogicLayer`: fixed random 2-input wiring per neuron + softmax-weighted mixture
    over the 16 gates (Eq. 2), with a `discretized_forward` for the hard, bit-op-only
    inference path (argmax gate per neuron)
  - `GroupSum`: k-group output aggregation with temperature `tau` (Eq. 3)
  - `DiffLogicNet`: stacks `num_layers` `LogicLayer`s ("straight network," matching
    the paper's 4–8 layer, constant-width setup) + `GroupSum`
- **smoke_test.py** — correctness check on sklearn's `digits` dataset (fast, CPU-only,
  no download needed). Confirms: loss decreases, soft net trains to ~88% test acc,
  and the **discretized net matches the soft net to 0.00 pp** — consistent with the
  paper's claim that discretization only costs a small fraction of a percent.
- **cifar10_difflogicnet.py** — the actual CIFAR-10 baseline script: paper's Sec. 6.4
  binary embedding (3 thresholds per color channel → 9216 input bits), Adam lr=0.01,
  GroupSum classification loss, soft vs. discretized eval at the end. Wiring verified
  with synthetic CIFAR-shaped tensors (forward, backward, discretized forward all run
  cleanly). Not run end-to-end on real CIFAR-10 in this session — the download from
  the CIFAR mirror was too slow in the sandbox (~100 KB/s; killed after 170s at 10%).
  Run it yourself with:
  ```
  python cifar10_difflogicnet.py --data-root ./data --epochs 30 \
      --layer-width 8000 --num-layers 5
  ```
  (drop `--layer-width`/`--num-layers` toward the paper's Table 5 configs, or use
  `--limit-train-batches` for a quick smoke test on real data.)

## Status vs. the paper
Implemented: real-valued logic Table 1, Eq. 2 differentiable neuron, fixed random
connectivity, GroupSum (Eq. 3), N(0,1) weight init, Adam lr=0.01, discretization.
Not implemented (paper's speed-only engineering, not needed for a correctness
baseline): the custom CUDA kernels / int64 bit-packed SIMD inference path, and
hardware adder-tree output aggregation. Those only affect inference *speed*, not
the accuracy numbers you need for the PBNN comparison.


## Results
Results. On sklearn's digits dataset (CPU-only, no downloads), the soft DiffLogicNet trains to 88.3% test accuracy over 60 epochs with the loss decreasing smoothly (2.30 → 1.18), and the discretized hard-gate network matches the soft network to 0.00 percentage points (88.33% vs 88.33%) — reproducing the paper's claim that the argmax-gate discretization costs only a small fraction of a percent of accuracy. Gate distributions converge as expected, with mean max-gate probability rising from 0.65 (layer 0) to 0.90 (layer 4). This validates the end-to-end pipeline: relaxed real-valued training, GroupSum classification, and bit-op inference all agree. The CIFAR-10 baseline script is fully wired (forward, backward, and discretized forward all verified on synthetic CIFAR-shaped tensors) but has not yet been run end-to-end on real data; run it with python cifar10_difflogicnet.py --data-root ./data --epochs 30 --layer-width 8000 --num-layers 5 (drop --layer-width/--num-layers toward the paper's Table 5 configs, or use --limit-train-batches for a quick smoke test on real data).



