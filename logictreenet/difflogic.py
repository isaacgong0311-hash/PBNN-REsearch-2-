"""
Differentiable Logic Gate Networks (dense) and Convolutional LogicTreeNet.

References:
  - Petersen et al., "Deep Differentiable Logic Gate Networks", NeurIPS 2022
    (arXiv:2210.08277) -- dense DiffLogicNet.
  - Petersen et al., "Convolutional Differentiable Logic Gate Networks",
    NeurIPS 2024 (arXiv:2411.04732) -- convolutional LogicTreeNet.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# The 16 possible 2-input binary Boolean gates, represented as bilinear
# polynomials of the two (soft, in [0,1]) inputs a, b:
#
#   gate(a, b) = k0 + k1*a + k2*b + k3*a*b
#
# For hard (0/1) inputs this recovers the exact truth table.  We build the
# truth tables programmatically (index i -> 4-bit truth table over
# (a,b) in {(0,0),(0,1),(1,0),(1,1)}) rather than hand-typing them, since a
# hand-typed table is an easy place to introduce an off-by-one gate-index bug.
# ---------------------------------------------------------------------------

def _bit(i, k):
    return (i >> k) & 1


# _ALL_TRUTH_TABLES[i] = (out(0,0), out(0,1), out(1,0), out(1,1)) for gate i
_ALL_TRUTH_TABLES = [
    tuple(_bit(i, k) for k in range(4)) for i in range(16)
]


def _truth_table_to_bilinear(tt):
    # tt = (f00, f01, f10, f11); solve for k0,k1,k2,k3 s.t.
    #   f(a,b) = k0 + k1*a + k2*b + k3*a*b
    f00, f01, f10, f11 = tt
    k0 = f00
    k1 = f10 - f00
    k2 = f01 - f00
    k3 = f11 - f10 - f01 + f00
    return (float(k0), float(k1), float(k2), float(k3))


GATE_COEFFS = torch.tensor(
    [_truth_table_to_bilinear(tt) for tt in _ALL_TRUTH_TABLES],
    dtype=torch.float32,
)  # (16, 4)

_NAME_FROM_TT = {
    (0, 0, 0, 0): "FALSE",
    (0, 0, 0, 1): "AND",
    (0, 0, 1, 0): "A_AND_NOT_B",
    (0, 0, 1, 1): "A",
    (0, 1, 0, 0): "NOT_A_AND_B",
    (0, 1, 0, 1): "B",
    (0, 1, 1, 0): "XOR",
    (0, 1, 1, 1): "OR",
    (1, 0, 0, 0): "NOR",
    (1, 0, 0, 1): "XNOR",
    (1, 0, 1, 0): "NOT_B",
    (1, 0, 1, 1): "A_OR_NOT_B",
    (1, 1, 0, 0): "NOT_A",
    (1, 1, 0, 1): "NOT_A_OR_B",
    (1, 1, 1, 0): "NAND",
    (1, 1, 1, 1): "TRUE",
}

GATE_NAMES = [_NAME_FROM_TT[tt] for tt in _ALL_TRUTH_TABLES]

_PASS_A_IDX = GATE_NAMES.index("A")
_PASS_B_IDX = GATE_NAMES.index("B")


def gate_probs(logits, sharpness=1.0, ste=False):
    """Softmax over the 16 gates. With ste=True (straight-through estimator) the
    forward value is the one-hot argmax gate -- exactly the discretized network --
    while gradients flow as if it were the softmax."""
    probs = F.softmax(logits * sharpness, dim=-1)
    if ste:
        hard = F.one_hot(probs.argmax(dim=-1), probs.shape[-1]).to(probs.dtype)
        probs = hard + probs - probs.detach()
    return probs


def gate_forward(logits, a, b, sharpness=1.0, ste=False):
    """logits: (..., 16); a, b: (...,) broadcastable soft values in [0,1]."""
    coeffs = GATE_COEFFS.to(device=logits.device, dtype=logits.dtype)
    probs = gate_probs(logits, sharpness, ste)  # (..., 16)
    k = probs @ coeffs  # (..., 4) -> k0,k1,k2,k3
    k0, k1, k2, k3 = k.unbind(-1)
    return k0 + k1 * a + k2 * b + k3 * a * b


def gate_forward_hard(gate_idx, a, b):
    """gate_idx: (...,) long tensor of chosen gate index; a,b in {0,1}."""
    coeffs = GATE_COEFFS.to(device=gate_idx.device)
    k = coeffs[gate_idx]  # (..., 4)
    k0, k1, k2, k3 = k.unbind(-1)
    out = k0 + k1 * a + k2 * b + k3 * a * b
    return out.round().clamp(0, 1)


def gate_confidence(logits):
    """Mean max-softmax-probability over all gates in a logits tensor.
    1.0 = fully one-hot (confident); 1/16 = uniform (no preference)."""
    probs = F.softmax(logits, dim=-1)
    return probs.max(dim=-1).values.mean().item()


def _residual_gate_init(out_features, generator, bias=5.0, noise=0.1):
    """Residual/pass-through initialization (paper Sec 3.2, z3=5): biases
    each neuron's gate-choice logits toward the pass-through gate ('A' or
    'B', chosen at random per neuron) so that at init the network behaves
    close to identity, preventing vanishing gradients in deep stacks.
    Without this bias, zero/small-random init gives a near-uniform softmax
    over all 16 gates, whose *averaged* output is a constant independent of
    the inputs -- i.e. zero Jacobian and dead backprop through all but the
    last layer.
    """
    logits = torch.randn(out_features, 16, generator=generator) * noise
    choice = torch.randint(0, 2, (out_features,), generator=generator)
    pass_idx = torch.where(choice == 0, _PASS_A_IDX, _PASS_B_IDX)
    logits[torch.arange(out_features), pass_idx] += bias
    return logits


class LogicLayer(nn.Module):
    """A dense differentiable-logic-gate layer with fixed random wiring."""

    def __init__(self, in_features, out_features, seed=0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        g = torch.Generator().manual_seed(seed)

        idx_a = torch.empty(out_features, dtype=torch.long)
        idx_b = torch.empty(out_features, dtype=torch.long)
        for i in range(out_features):
            a = torch.randint(0, in_features, (1,), generator=g).item()
            b = torch.randint(0, in_features, (1,), generator=g).item()
            tries = 0
            while b == a and in_features > 1 and tries < 100:
                b = torch.randint(0, in_features, (1,), generator=g).item()
                tries += 1
            idx_a[i] = a
            idx_b[i] = b
        self.register_buffer("idx_a", idx_a)
        self.register_buffer("idx_b", idx_b)

        self.gate_logits = nn.Parameter(_residual_gate_init(out_features, g))

    def forward(self, x, hard=False, gate_idx=None, sharpness=1.0, ste=False):
        a = x[..., self.idx_a]
        b = x[..., self.idx_b]
        if hard:
            idx = self.gate_logits.argmax(dim=-1) if gate_idx is None else gate_idx
            return gate_forward_hard(idx, a, b)
        return gate_forward(self.gate_logits, a, b, sharpness=sharpness, ste=ste)


class GroupSum(nn.Module):
    def __init__(self, num_classes, tau=1.0):
        super().__init__()
        self.num_classes = num_classes
        self.tau = tau

    def forward(self, x):
        b = x.shape[0]
        group = x.view(b, self.num_classes, -1).sum(dim=-1)
        return group / self.tau


def binarize(x, thresholds=(0.5,), flatten=True):
    """x: (B, C, H, W) float in [0,1]. Returns thresholded binary channels."""
    chans = [(x >= t).float() for t in thresholds]
    out = torch.cat(chans, dim=1)
    if flatten:
        b = out.shape[0]
        return out.view(b, -1)
    return out


class DiffLogicNet(nn.Module):
    def __init__(self, in_features, width, depth, num_classes, tau=1.0, seed=0):
        super().__init__()
        layers = []
        f_in = in_features
        for i in range(depth):
            layers.append(LogicLayer(f_in, width, seed=seed + i))
            f_in = width
        self.layers = nn.ModuleList(layers)
        self.groupsum = GroupSum(num_classes, tau=tau)

    def forward(self, x, hard=False, sharpness=1.0, ste=False):
        for layer in self.layers:
            x = layer(x, hard=hard, sharpness=sharpness, ste=ste)
        return self.groupsum(x)

    def gate_confidences(self):
        return {f"layer_{i}": gate_confidence(l.gate_logits) for i, l in enumerate(self.layers)}


DIFFLOGICNET_CIFAR = {
    "small": dict(width=12000, depth=4, tau=1 / 0.03, reported=51.27),
    "medium": dict(width=48000, depth=4, tau=1 / 0.03, reported=57.68),
    "large": dict(width=128000, depth=4, tau=1 / 0.03, reported=59.38),
}


class LogicTree(nn.Module):
    """A balanced binary tree of logic gates: n_leaves = 2**depth soft
    inputs are folded pairwise, depth times, down to a single output, for
    each of n_trees independent trees (used as a conv kernel)."""

    def __init__(self, n_trees, depth, seed=0):
        super().__init__()
        self.n_trees = n_trees
        self.depth = depth
        self.n_leaves = 2 ** depth
        g = torch.Generator().manual_seed(seed)
        n_gates = self.n_leaves - 1
        self.gate_logits = nn.Parameter(
            _residual_gate_init(n_trees * n_gates, g).view(n_trees, n_gates, 16)
        )

    def forward(self, leaves, hard=False, sharpness=1.0, ste=False):
        # leaves: (..., n_trees, n_leaves)
        x = leaves
        gate_offset = 0
        level_size = self.n_leaves
        while level_size > 1:
            n_pairs = level_size // 2
            a = x[..., 0::2]
            b = x[..., 1::2]
            logits = self.gate_logits[:, gate_offset:gate_offset + n_pairs, :]
            if hard:
                idx = logits.argmax(dim=-1)
                x = gate_forward_hard(idx, a, b)
            else:
                probs = gate_probs(logits, sharpness, ste)
                coeffs = GATE_COEFFS.to(device=logits.device, dtype=logits.dtype)
                k = probs @ coeffs
                k0, k1, k2, k3 = k.unbind(-1)
                x = k0 + k1 * a + k2 * b + k3 * a * b
            gate_offset += n_pairs
            level_size = n_pairs
        return x[..., 0]  # (..., n_trees)


class LogicTreeConv2d(nn.Module):
    """Uses a LogicTree per output channel as a convolutional kernel, via
    F.unfold to extract sliding-window patches.

    channels_per_tree: paper Sec 3.1 -- "each tree observes only 2 (rather
    than up to 8) input channels", which forces spatial comparisons within a
    channel. Each tree picks that many distinct input channels at random,
    then draws its 2**depth leaves (without replacement, when possible) from
    those channels' kernel_size x kernel_size windows. 0/None = draw leaves
    from all input channels. Default 0: on CIFAR-10 scale S, 2-channel trees
    gave the same soft accuracy (57.39% vs 57.40%) but much worse hard
    accuracy (46.23% vs 53.54%) than all-channel trees.
    """

    def __init__(self, in_channels, out_channels, kernel_size, tree_depth, seed=0,
                 channels_per_tree=0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.tree_depth = tree_depth
        self.channels_per_tree = channels_per_tree
        n_leaves = 2 ** tree_depth
        kk = kernel_size * kernel_size
        window = in_channels * kk
        g = torch.Generator().manual_seed(seed)
        leaf_idx = torch.empty(out_channels, n_leaves, dtype=torch.long)
        if not channels_per_tree:
            for c in range(out_channels):
                leaf_idx[c] = torch.randint(0, window, (n_leaves,), generator=g)
        else:
            n_ch = min(channels_per_tree, in_channels)
            offsets = torch.arange(kk)
            for c in range(out_channels):
                chans = torch.randperm(in_channels, generator=g)[:n_ch]
                # F.unfold lays patches out channel-major: index = ch * kk + pos
                cand = (chans[:, None] * kk + offsets[None, :]).reshape(-1)
                for _ in range(100):
                    if cand.numel() >= n_leaves:
                        pick = torch.randperm(cand.numel(), generator=g)[:n_leaves]
                    else:
                        pick = torch.randint(0, cand.numel(), (n_leaves,), generator=g)
                    # redraw in the rare case every leaf landed in one channel
                    if len(set((cand[pick] // kk).tolist())) == n_ch:
                        break
                leaf_idx[c] = cand[pick]
        self.register_buffer("leaf_idx", leaf_idx)
        self.tree = LogicTree(out_channels, tree_depth, seed=seed + 1)

    def forward(self, x, hard=False, sharpness=1.0, ste=False):
        b, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        # patches: (B, window, L)
        l = patches.shape[-1]
        patches = patches.transpose(1, 2)  # (B, L, window)
        # gather leaves: (B, L, out_channels, n_leaves)
        leaves = patches[:, :, self.leaf_idx]
        out = self.tree(leaves, hard=hard, sharpness=sharpness, ste=ste)  # (B, L, out_channels)
        out = out.transpose(1, 2).contiguous()  # (B, out_channels, L)
        out_h = h  # padding = kernel_size//2 keeps spatial size
        out_w = w
        return out.view(b, self.out_channels, out_h, out_w)


class OrPool2d(nn.Module):
    """Logical-OR pooling: soft = probabilistic-sum-OR = 1 - prod(1-x) over
    the window; hard = max-pool (since OR of bits == max of bits)."""

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x, hard=False):
        if hard:
            return F.max_pool2d(x, self.kernel_size, self.stride)
        # Product over the window written as explicit multiplies of strided slices.
        # (torch.prod on CUDA is JIT-compiled via NVRTC, which is broken on some
        # Colab images: "failed to open libnvrtc-builtins.so".)
        b, c, h, w = x.shape
        k, s = self.kernel_size, self.stride
        out_h = (h - k) // s + 1
        out_w = (w - k) // s + 1
        keep = None
        for i in range(k):
            for j in range(k):
                sl = x[:, :, i:i + s * (out_h - 1) + 1:s, j:j + s * (out_w - 1) + 1:s]
                keep = (1 - sl) if keep is None else keep * (1 - sl)
        return 1 - keep


class LogicTreeNet(nn.Module):
    """Convolutional differentiable logic gate network (paper: LogicTreeNet).

    Paper Appendix A.1.1: after the conv+pool blocks, the head is exactly
    3 dense LogicLayers of width 1280k -> 640k -> 320k, where k =
    channels[0] (the base scale, e.g. 32 for scale S). This is NOT a free
    architectural choice.
    """

    def __init__(self, in_channels, in_size, channels, num_classes,
                 tree_depth=3, kernel_size=3, output_gate_factor=1,
                 tau=20.0, seed=0, channels_per_tree=0):
        super().__init__()
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        c_in = in_channels
        size = in_size
        for i, c_out in enumerate(channels):
            self.blocks.append(
                LogicTreeConv2d(c_in, c_out, kernel_size, tree_depth, seed=seed + i,
                                channels_per_tree=channels_per_tree)
            )
            self.pools.append(OrPool2d(2, 2))
            c_in = c_out
            size = size // 2
        flat_dim = c_in * size * size

        # Paper Appendix A.1.1: k = channels[0]; head is exactly 3 dense
        # LogicLayers of width 1280k -> 640k -> 320k (not a free choice).
        k = channels[0]
        ox = output_gate_factor
        head_widths = [1280 * k * ox, 640 * k * ox, 320 * k * ox]
        assert head_widths[-1] % num_classes == 0, (
            f"final head width {head_widths[-1]} must be divisible by num_classes={num_classes}"
        )

        head_layers = []
        f_in = flat_dim
        for i, w in enumerate(head_widths):
            head_layers.append(LogicLayer(f_in, w, seed=seed + 100 + i))
            f_in = w
        self.head = nn.ModuleList(head_layers)
        self.groupsum = GroupSum(num_classes, tau=tau)
        self.flat_dim = flat_dim

    def forward(self, x, hard=False, sharpness=1.0, ste=False):
        # ste=True: forward pass is exactly the discretized network (one-hot gates
        # on binary inputs give binary outputs, and soft OR-pooling of bits is
        # exact OR), but gradients flow through the gate softmaxes.
        for block, pool in zip(self.blocks, self.pools):
            x = block(x, hard=hard, sharpness=sharpness, ste=ste)
            x = pool(x, hard=hard)
        x = x.flatten(1)
        for layer in self.head:
            x = layer(x, hard=hard, sharpness=sharpness, ste=ste)
        return self.groupsum(x)

    def gate_confidences(self):
        confs = {}
        for i, block in enumerate(self.blocks):
            confs[f"conv_block_{i}"] = gate_confidence(block.tree.gate_logits)
        for i, layer in enumerate(self.head):
            confs[f"head_{i}"] = gate_confidence(layer.gate_logits)
        return confs


# Paper Table 6 (CIFAR-10). Scale B removed: needs unimplemented 5-bit
# input + edge/curvature-detector preprocessing (paper Sec 3.5).
LOGICTREENET_CIFAR = {
    "S": dict(channels=(32, 128, 512, 1024), tau=20.0, reported=60.38),
    "M": dict(channels=(256, 1024, 4096, 8192), tau=40.0, reported=71.01),
}