"""
Faithful implementation of Deep Differentiable Logic Gate Networks
(Petersen, Borgelt, Kuehne, Deussen — NeurIPS 2022, arXiv:2210.08277).

Implements:
  - Table 1: all 16 real-valued relaxations of binary logic gates
  - Eq. 2: differentiable logic-gate neuron (softmax-weighted mixture of the 16 gates)
  - Fixed pseudo-random 2-input connectivity per layer (Sec. 3 / Fig. 1)
  - Eq. 3: GroupSum aggregation with temperature tau and offset beta
  - Discretization to a hard (bit-op only) network for inference (Sec. 4.1)

This is a plain PyTorch reference implementation (no custom CUDA kernels like the
official `difflogic` package) -- correct and differentiable, but not optimized for
the >1M img/s inference speed the paper reports. Good enough to validate the
algorithm and produce a CIFAR-10 baseline number for the PBNN comparison.
"""
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Table 1: the 16 real-valued binary logic operators, f_i(a, b), i = 0..15
# ---------------------------------------------------------------------------
def bin_op(a: torch.Tensor, b: torch.Tensor, i: int) -> torch.Tensor:
    if i == 0:
        return torch.zeros_like(a)                      # False
    elif i == 1:
        return a * b                                     # A and B
    elif i == 2:
        return a - a * b                                 # not(A -> B)   = A and not B
    elif i == 3:
        return a                                          # A
    elif i == 4:
        return b - a * b                                 # not(A <- B)   = not A and B
    elif i == 5:
        return b                                          # B
    elif i == 6:
        return a + b - 2 * a * b                          # A xor B
    elif i == 7:
        return a + b - a * b                              # A or B
    elif i == 8:
        return 1 - (a + b - a * b)                        # nor
    elif i == 9:
        return 1 - (a + b - 2 * a * b)                     # xnor
    elif i == 10:
        return 1 - b                                       # not B
    elif i == 11:
        return 1 - b + a * b                                # A <- B  (B -> A)
    elif i == 12:
        return 1 - a                                       # not A
    elif i == 13:
        return 1 - a + a * b                                # A -> B
    elif i == 14:
        return 1 - a * b                                   # nand
    elif i == 15:
        return torch.ones_like(a)                          # True
    else:
        raise ValueError(f"invalid gate id {i}")


N_OPS = 16
GATE_NAMES = [
    "False", "A and B", "A and not B", "A", "not A and B", "B", "A xor B",
    "A or B", "nor", "xnor", "not B", "B -> A", "not A", "A -> B", "nand", "True",
]


# ---------------------------------------------------------------------------
# A single logic layer: n output neurons, each wired to 2 fixed random inputs
# from the previous layer, each parameterized by a learned distribution over
# the 16 gates (Eq. 2).
# ---------------------------------------------------------------------------
class LogicLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, seed: int | None = None):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        g = torch.Generator().manual_seed(seed) if seed is not None else None
        # fixed pseudo-random connectivity, drawn once and never trained (Sec. 3)
        idx_a = torch.randint(0, in_dim, (out_dim,), generator=g)
        idx_b = torch.randint(0, in_dim, (out_dim,), generator=g)
        self.register_buffer("idx_a", idx_a)
        self.register_buffer("idx_b", idx_b)

        # w in R^16 per neuron, initialized N(0,1) as in the paper (Sec. 4.1 "Training")
        self.weights = nn.Parameter(torch.randn(out_dim, N_OPS))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim) in [0, 1] (relaxed) -- soft/continuous forward pass
        a = x[:, self.idx_a]   # (batch, out_dim)
        b = x[:, self.idx_b]
        p = torch.softmax(self.weights, dim=-1)          # (out_dim, 16), Eq. 2's p_i
        stacked = torch.stack([bin_op(a, b, i) for i in range(N_OPS)], dim=-1)  # (batch, out_dim, 16)
        return (stacked * p.unsqueeze(0)).sum(-1)         # (batch, out_dim)

    @torch.no_grad()
    def discretized_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, in_dim) hard {0,1} -- discretized (mode of the distribution) forward pass
        a = x[:, self.idx_a]
        b = x[:, self.idx_b]
        gate_ids = self.weights.argmax(dim=-1)             # (out_dim,) -- the "4 bits" per neuron
        out = torch.empty_like(a)
        for i in range(N_OPS):
            mask = gate_ids == i
            if mask.any():
                out[:, mask] = bin_op(a[:, mask], b[:, mask], i)
        return out

    def hard_gate_ids(self) -> torch.Tensor:
        return self.weights.argmax(dim=-1)


# ---------------------------------------------------------------------------
# Eq. 3: GroupSum aggregation -- k groups of n/k neurons each, summed and
# rescaled by temperature tau (+ optional offset beta).
# ---------------------------------------------------------------------------
class GroupSum(nn.Module):
    def __init__(self, k: int, tau: float = 1.0, beta: float = 0.0):
        super().__init__()
        self.k = k
        self.tau = tau
        self.beta = beta

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, n = x.shape
        assert n % self.k == 0, f"n_outputs ({n}) must be divisible by k ({self.k})"
        x = x.view(batch, self.k, n // self.k).sum(dim=-1)
        return x / self.tau + self.beta


# ---------------------------------------------------------------------------
# Full network: a stack of LogicLayers ("straight network", same width per
# layer, 4-8 layers per the paper) followed by GroupSum.
# ---------------------------------------------------------------------------
class DiffLogicNet(nn.Module):
    def __init__(self, in_dim: int, layer_width: int, num_layers: int,
                 num_classes: int, tau: float = 10.0, beta: float = 0.0,
                 seed: int | None = 0):
        super().__init__()
        assert layer_width % num_classes == 0, \
            "layer_width must be divisible by num_classes for GroupSum"
        assert num_layers >= 1

        layers = []
        dim = in_dim
        for l in range(num_layers):
            layer_seed = None if seed is None else seed + l
            layers.append(LogicLayer(dim, layer_width, seed=layer_seed))
            dim = layer_width
        self.layers = nn.ModuleList(layers)
        self.group_sum = GroupSum(num_classes, tau=tau, beta=beta)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.group_sum(x)

    @torch.no_grad()
    def discretized_forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer.discretized_forward(x)
        return self.group_sum(x)

    def num_parameters(self) -> int:
        # "parameters" here = number of neurons (each stores a 4-bit gate id once discretized);
        # matches the paper's #Parameters column convention (one param per logic gate/neuron).
        return sum(l.out_dim for l in self.layers)

    def bit_op_count_per_inference(self) -> int:
        # every neuron performs exactly one binary gate op at inference (discretized net)
        return self.num_parameters()
