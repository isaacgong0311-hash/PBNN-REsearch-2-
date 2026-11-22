"""
Differentiable Logic Gate Networks (dense) and Convolutional LogicTreeNet.

References:
  - Petersen et al., "Deep Differentiable Logic Gate Networks", NeurIPS 2022
    (arXiv:2210.08277) -- dense DiffLogicNet.
  - Petersen et al., "Convolutional Differentiable Logic Gate Networks",
    NeurIPS 2024 (arXiv:2411.04732) -- convolutional LogicTreeNet.
"""

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
_ALL_TRUTH_TABLES = [tuple(_bit(i, k) for k in range(4)) for i in range(16)]


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


def gumbel_noise(like):
    """Gumbel(0, 1) noise with the shape of `like`: g = -log(-log u), u ~ U(0, 1) (computed as -log of an Exp(1) draw, as in
    torch.nn.functional.gumbel_softmax)."""
    return -torch.empty_like(like).exponential_().log()


def _gate_probs_raw(logits, sharpness=1.0, ste=False, gumbel=None):
    """Softmax over the 16 gates. With ste=True (straight-through estimator) the
    forward value is the one-hot argmax gate -- exactly the discretized network --
    while gradients flow as if it were the softmax.

    gumbel = (scale, tau) (not in Petersen et al.; method of Yousefi et al., "Mind the Gap", arXiv:2506.07500, Sec. 4 and
    Algorithm 1 in App. E): draw g ~ Gumbel(0, 1) with the shape of `logits` (one draw per gate vector, so neurons, tree nodes
    and conv channels get their own noise, shared over the batch and over spatial positions), soft = softmax((logits + scale*g)
    / tau), forward value = one-hot(argmax(logits + scale*g)), gradient = gradient of `soft` (straight-through). scale = 1 is the
    paper's Gumbel(0, 1). The forward value does not depend on tau; tau only changes the backward softmax."""
    if gumbel is not None:
        scale, tau = gumbel
        y = logits + scale * gumbel_noise(logits)
        soft = F.softmax(y * sharpness / tau, dim=-1)
        hard = F.one_hot(y.argmax(dim=-1), y.shape[-1]).to(soft.dtype)
        return hard + soft - soft.detach()
    probs = F.softmax(logits * sharpness, dim=-1)
    if ste:
        hard = F.one_hot(probs.argmax(dim=-1), probs.shape[-1]).to(probs.dtype)
        probs = hard + probs - probs.detach()
    return probs


FREEZE_LOGIT = 20.0  # logit given to a frozen gate (all others 0) so that argmax(logits) stays the frozen gate


def gate_probs(logits, sharpness=1.0, ste=False, gumbel=None, frozen=None):
    """gate_probs with optional frozen gates (progressive discretization, Gerlach 2026, arXiv:2609.33747): frozen is a long tensor
    (logits.shape[:-1]) with the frozen gate index, or -1 for gates that are still relaxed. A frozen neuron always uses the one-hot
    distribution of its gate (no noise, no gradient to its logits); the others are computed as before."""
    probs = _gate_probs_raw(logits, sharpness, ste, gumbel)
    if frozen is None:
        return probs
    oh = F.one_hot(frozen.clamp_min(0), probs.shape[-1]).to(probs.dtype)
    return torch.where((frozen >= 0).unsqueeze(-1), oh, probs)


def _hard_idx(logits, frozen=None):
    idx = logits.argmax(dim=-1)
    return idx if frozen is None else torch.where(frozen >= 0, frozen, idx)


class _FreezeMixin:
    """Per-neuron freezing state for LogicLayer / LogicTree (only exists after enable_freeze(); off by default).
    Buffers (so they are saved in checkpoints and resumed): frozen_idx (-1 = relaxed), freeze_mu (EMA of the first-order freeze
    shock, TaskFreeze), freeze_n (evaluations seen), freeze_cnt (consecutive evaluations with mu < 0)."""

    def enable_freeze(self):
        shape, dev = self.gate_logits.shape[:-1], self.gate_logits.device
        self.register_buffer("frozen_idx", torch.full(shape, -1, dtype=torch.long, device=dev))
        self.register_buffer("freeze_mu", torch.zeros(shape, device=dev))
        self.register_buffer("freeze_n", torch.zeros(shape, dtype=torch.long, device=dev))
        self.register_buffer("freeze_cnt", torch.zeros(shape, dtype=torch.long, device=dev))

    def freeze_where(self, mask):
        """Freeze every still-relaxed neuron selected by mask (bool, logits.shape[:-1]) to its current argmax gate."""
        with torch.no_grad():
            m = mask & (self.frozen_idx < 0)
            fi = self.gate_logits.argmax(dim=-1)
            self.frozen_idx[m] = fi[m]
            new = FREEZE_LOGIT * F.one_hot(fi, self.gate_logits.shape[-1]).to(self.gate_logits.dtype)
            self.gate_logits.copy_(torch.where(m.unsqueeze(-1), new, self.gate_logits))
            return int(m.sum())

    def freeze_update(self, idx, s, beta, patience):
        """TaskFreeze bookkeeping for the neurons self.<buffer>[idx] given their minibatch first-order scores s (same shape):
        EMA mu <- beta*mu + (1-beta)*s (the first evaluation sets mu = s); a neuron freezes once mu < 0 for `patience`
        consecutive evaluations. Returns the number of newly frozen neurons."""
        with torch.no_grad():
            mu, n, cnt = self.freeze_mu[idx], self.freeze_n[idx], self.freeze_cnt[idx]
            mu = torch.where(n == 0, s, beta * mu + (1 - beta) * s)
            cnt = torch.where(mu < 0, cnt + 1, torch.zeros_like(cnt))
            self.freeze_mu[idx] = mu
            self.freeze_n[idx] = n + 1
            self.freeze_cnt[idx] = cnt
            mask = torch.zeros_like(self.frozen_idx, dtype=torch.bool)
            mask[idx] = cnt >= patience
            return self.freeze_where(mask)


def gate_forward(logits, a, b, sharpness=1.0, ste=False, gumbel=None, frozen=None):
    """logits: (..., 16); a, b: (...,) broadcastable soft values in [0,1]."""
    coeffs = GATE_COEFFS.to(device=logits.device, dtype=logits.dtype)
    probs = gate_probs(logits, sharpness, ste, gumbel, frozen)  # (..., 16)
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


def logic_dropout_perm(x, p, exact=False):
    """Binary-preserving dropout (proposed by Prof. Liu, Slack 2026-11-11; NOT in the paper). For each example, the dropped
    positions (each element dropped independently with probability p) keep the same SET of values but are randomly
    permuted among themselves. Every output is still 0 or 1, the number of ones per example is unchanged exactly,
    and the expected output is (1 - p) * h + p * (mean of h over the whole example), per element (checked by simulation).
    Operates per example over all positions and channels of that example (x.shape[0] = batch)."""
    shape = x.shape
    b = shape[0]
    h = x.reshape(b, -1)
    n = h.shape[1]
    if (
        exact
    ):  # exactly k = round(p * n) dropped positions per example (Prof. Liu, Slack 2026-11-11: 'there are dp entries')
        k = int(round(p * n))
        sel = torch.rand_like(h).argsort(dim=1)[:, :k]
        mask = torch.zeros_like(h, dtype=torch.bool).scatter_(1, sel, True)
    else:
        mask = torch.rand_like(h) < p  # dropped positions, each independently with probability p
    keys = torch.where(mask, torch.rand_like(h), torch.full_like(h, 2.0))
    order_rand = torch.argsort(keys, dim=1)  # dropped positions first, in random order
    pos = torch.arange(n, device=h.device).expand_as(h)
    order_pos = torch.argsort(torch.where(mask, pos, pos + n), dim=1)  # dropped positions first, in position order
    vals_rand = h.gather(1, order_rand)  # the dropped values come first, in random order
    permuted = h.scatter(1, order_pos, vals_rand)  # write them back to the dropped positions in position order
    out = torch.where(mask, permuted, h)
    return out.reshape(shape)


def logic_dropout(x, p, training=True, mode="half"):
    """Dropout for soft logic activations (NOT in the paper; Prof. Liu's suggestion).

    mode "half" (default): each activation is replaced, with probability p, by 0.5
    ("unknown bit"). There is no 1/(1-p) rescaling, which would push values outside
    [0, 1]. Because every gate is bilinear in each input, feeding 0.5 gives exactly
    the average of the gate's outputs for that input = 0 and = 1, i.e. the input is
    marginalized as a fair coin. The expected activation is pulled toward 0.5.

    mode "scale": the textbook version (d2l.ai, "Dropout"): each activation is
    replaced by 0 with probability p and by x / (1 - p) otherwise, so the expected
    activation is unchanged. Values can exceed 1 (up to 1 / (1 - p)).

    Identity when p <= 0 or not training; callers also skip it for the hard network
    and the straight-through (STE) phase, whose forward pass must be the discretized
    network.
    """
    if mode not in ("half", "scale", "perm", "permx"):
        raise ValueError(f"dropout mode must be 'half', 'scale', 'perm' or 'permx', got {mode!r}")
    if not training or p <= 0:
        return x
    if mode == "perm":
        return logic_dropout_perm(x, p)
    if mode == "permx":  # exact-count variant: exactly round(p * n) positions dropped per example
        return logic_dropout_perm(x, p, exact=True)
    mask = torch.rand_like(x) < p
    if mode == "scale":
        return torch.where(mask, torch.zeros_like(x), x / (1.0 - p))
    return torch.where(mask, torch.full_like(x, 0.5), x)


def _sample_two_source_wiring(out_features, in_group, in_source, groups, skip_frac, generator):
    """Random 2-input wiring over a concatenation of a main source (in_source == 0)
    and a skip source (in_source == 1). Each of a neuron's two input slots comes
    from the skip source with probability skip_frac, otherwise from the main
    source, always from the neuron's own group (j % groups); a != b."""
    g = generator
    out_group = torch.arange(out_features) % groups
    pools = {}
    for k in range(groups):
        for s in (0, 1):
            m = torch.nonzero((in_group == k) & (in_source == s)).flatten()
            if m.numel() < 2:
                raise ValueError(f"group {k}, source {s} has {m.numel()} input features; need >= 2")
            pools[(k, s)] = m

    def draw(neurons, src):
        out = torch.empty(neurons.numel(), dtype=torch.long)
        for k in range(groups):
            for s in (0, 1):
                sel = torch.nonzero((out_group[neurons] == k) & (src == s)).flatten()
                if sel.numel():
                    m = pools[(k, s)]
                    out[sel] = m[torch.randint(0, m.numel(), (sel.numel(),), generator=g)]
        return out

    all_n = torch.arange(out_features)
    src_a = (torch.rand(out_features, generator=g) < skip_frac).long()
    src_b = (torch.rand(out_features, generator=g) < skip_frac).long()
    idx_a = draw(all_n, src_a)
    idx_b = draw(all_n, src_b)
    for _ in range(100):  # redraw b (same pool) wherever it collides with a
        clash = torch.nonzero(idx_a == idx_b).flatten()
        if clash.numel() == 0:
            break
        idx_b[clash] = draw(clash, src_b[clash])
    return idx_a, idx_b


def _balanced_pairs(num_inputs, n, generator):
    """Balanced 2-input wiring for n gates over num_inputs inputs (NOT in the paper, which draws
    every connection at random, Sec. 2 and Sec. 3.4). The 2n input slots are filled from
    concatenated random permutations of the inputs, so every input is read floor(2n / num_inputs)
    or ceil(2n / num_inputs) times. If 2n >= num_inputs, every input is read at least once; with
    plain random wiring about 1/e (36.8%) of the inputs are never read when 2n == num_inputs.
    Returns (ia, ib), LongTensors of length n with positions into the input list, ia != ib wherever
    that is possible."""
    need = 2 * n
    reps = -(-need // num_inputs)
    stream = torch.cat([torch.randperm(num_inputs, generator=generator) for _ in range(reps)])[:need]
    ia, ib = stream[:n].clone(), stream[n:].clone()
    for c in torch.nonzero(ia == ib).flatten().tolist():  # rare: only when 2n > num_inputs
        for _ in range(100):  # swap b with another gate's b (keeps every input's read count)
            j = int(torch.randint(0, n, (1,), generator=generator))
            if ia[c] != ib[j] and ia[j] != ib[c]:
                tmp = int(ib[c])
                ib[c] = ib[j]
                ib[j] = tmp
                break
    return ia, ib


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


class LogicLayer(_FreezeMixin, nn.Module):
    """A dense differentiable-logic-gate layer with fixed random wiring.

    groups / in_group (paper App. A.3): with groups=G > 1, output neuron j
    belongs to group j % G and draws both inputs only from input features
    whose group id (in_group[f]) is the same, so the network splits into G
    separate models that only meet at GroupSum. Interleaving (j % G) means
    every class block of the final GroupSum gets an equal share of every
    group. groups=1 keeps the original wiring exactly."""

    def __init__(
        self, in_features, out_features, seed=0, groups=1, in_group=None, in_source=None, skip_frac=0.0, wiring="random"
    ):
        # in_source (optional, skip connections): (in_features,) tensor, 0 = main input,
        # 1 = skip input; each input slot reads the skip part with prob skip_frac.
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.groups = groups
        g = torch.Generator().manual_seed(seed)
        # wiring: "random" (default, paper) or "balanced" (every input read about equally often, see
        # _balanced_pairs; not in the paper). "random" builds exactly the same layer as before.
        if wiring not in ("random", "balanced"):
            raise ValueError(f"wiring must be 'random' or 'balanced', got {wiring!r}")
        if wiring == "balanced" and in_source is not None:
            raise ValueError("wiring='balanced' does not support skip sources")
        self.wiring = wiring

        if wiring == "balanced":
            G = max(groups, 1)
            if in_group is None:
                in_group = torch.arange(in_features) % G
            out_group = torch.arange(out_features) % G
            idx_a = torch.empty(out_features, dtype=torch.long)
            idx_b = torch.empty(out_features, dtype=torch.long)
            for k in range(G):
                m = torch.nonzero(in_group == k).flatten()
                if m.numel() < 2:
                    raise ValueError(f"group {k} has {m.numel()} input features; need >= 2")
                sel = torch.nonzero(out_group == k).flatten()
                ia, ib = _balanced_pairs(m.numel(), sel.numel(), g)
                idx_a[sel] = m[ia]
                idx_b[sel] = m[ib]
            self.register_buffer("idx_a", idx_a)
            self.register_buffer("idx_b", idx_b)
            self.gate_logits = nn.Parameter(_residual_gate_init(out_features, g))
            return

        if in_source is not None:
            if in_group is None:
                in_group = (
                    torch.arange(in_features) % groups if groups > 1 else torch.zeros(in_features, dtype=torch.long)
                )
            idx_a, idx_b = _sample_two_source_wiring(out_features, in_group, in_source, max(groups, 1), skip_frac, g)
            self.register_buffer("idx_a", idx_a)
            self.register_buffer("idx_b", idx_b)
            self.gate_logits = nn.Parameter(_residual_gate_init(out_features, g))
            return

        idx_a = torch.empty(out_features, dtype=torch.long)
        idx_b = torch.empty(out_features, dtype=torch.long)
        if groups > 1:
            if in_group is None:
                in_group = torch.arange(in_features) % groups
            members = [torch.nonzero(in_group == k).flatten() for k in range(groups)]
            for k, m in enumerate(members):
                if m.numel() < 2:
                    raise ValueError(f"group {k} has {m.numel()} input features; need >= 2")
            out_group = torch.arange(out_features) % groups
            for k in range(groups):
                sel = torch.nonzero(out_group == k).flatten()
                m = members[k]
                n = sel.numel()
                ia = torch.randint(0, m.numel(), (n,), generator=g)
                # second input: a different member of the same group
                ib = (ia + torch.randint(1, m.numel(), (n,), generator=g)) % m.numel()
                idx_a[sel] = m[ia]
                idx_b[sel] = m[ib]
            self.register_buffer("idx_a", idx_a)
            self.register_buffer("idx_b", idx_b)
            self.gate_logits = nn.Parameter(_residual_gate_init(out_features, g))
            return

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

    def forward(self, x, hard=False, gate_idx=None, sharpness=1.0, ste=False, collect=None, gumbel=None, probe=None):
        a = x[..., self.idx_a]
        b = x[..., self.idx_b]
        frozen = getattr(self, "frozen_idx", None)
        if hard:
            idx = _hard_idx(self.gate_logits, frozen) if gate_idx is None else gate_idx
            return gate_forward_hard(idx, a, b)
        y = gate_forward(self.gate_logits, a, b, sharpness=sharpness, ste=ste, gumbel=gumbel, frozen=frozen)
        if (
            probe is not None
        ):  # TaskFreeze: keep (module, index, a, b, y) and the gradient of y for the first-order score
            y.retain_grad()
            probe.append((self, (slice(None),), a.detach(), b.detach(), y))
        if collect is not None:  # per-gate mean output over the batch, shape (out_features,)
            collect.append(y.reshape(-1, y.shape[-1]).mean(0))
        return y


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


class LogicTree(_FreezeMixin, nn.Module):
    """A balanced binary tree of logic gates: n_leaves = 2**depth soft
    inputs are folded pairwise, depth times, down to a single output, for
    each of n_trees independent trees (used as a conv kernel)."""

    def __init__(self, n_trees, depth, seed=0):
        super().__init__()
        self.n_trees = n_trees
        self.depth = depth
        self.n_leaves = 2**depth
        g = torch.Generator().manual_seed(seed)
        n_gates = self.n_leaves - 1
        self.gate_logits = nn.Parameter(_residual_gate_init(n_trees * n_gates, g).view(n_trees, n_gates, 16))

    def forward(self, leaves, hard=False, sharpness=1.0, ste=False, collect=None, gumbel=None, probe=None):
        # leaves: (..., n_trees, n_leaves)
        # collect (optional list): appends this tree layer's per-gate mean output over all leading dims,
        # shape (n_trees, n_leaves - 1) in gate_logits order (used by the balance penalty, see balance_penalty)
        x = leaves
        gate_offset = 0
        level_size = self.n_leaves
        level_means = []
        while level_size > 1:
            n_pairs = level_size // 2
            a = x[..., 0::2]
            b = x[..., 1::2]
            logits = self.gate_logits[:, gate_offset : gate_offset + n_pairs, :]
            fz = self.frozen_idx[:, gate_offset : gate_offset + n_pairs] if hasattr(self, "frozen_idx") else None
            if hard:
                idx = _hard_idx(logits, fz)
                x = gate_forward_hard(idx, a, b)
            else:
                probs = gate_probs(logits, sharpness, ste, gumbel, fz)
                coeffs = GATE_COEFFS.to(device=logits.device, dtype=logits.dtype)
                k = probs @ coeffs
                k0, k1, k2, k3 = k.unbind(-1)
                x = k0 + k1 * a + k2 * b + k3 * a * b
                if probe is not None:
                    x.retain_grad()
                    probe.append(
                        (self, (slice(None), slice(gate_offset, gate_offset + n_pairs)), a.detach(), b.detach(), x)
                    )
            if collect is not None:
                level_means.append(x.reshape(-1, *x.shape[-2:]).mean(0))
            gate_offset += n_pairs
            level_size = n_pairs
        if collect is not None:
            collect.append(torch.cat(level_means, dim=1))
        return x[..., 0]  # (..., n_trees)


class LogicTreeConv2d(nn.Module):
    """Uses a LogicTree per output channel as a convolutional kernel, via
    F.unfold to extract sliding-window patches.

    channels_per_tree: paper Sec 3.4 -- "each tree observes only 2 (rather
    than up to 8) input channels", which forces spatial comparisons within a
    channel. Each tree picks that many distinct input channels at random,
    then draws its 2**depth leaves (without replacement, when possible) from
    those channels' kernel_size x kernel_size windows. 0/None = draw leaves
    from all input channels. Default 0: on CIFAR-10 scale S, 2-channel trees
    gave the same soft accuracy (57.39% vs 57.40%) but much worse hard
    accuracy (46.23% vs 53.54%) than all-channel trees.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        tree_depth,
        seed=0,
        channels_per_tree=0,
        groups=1,
        group_inputs=True,
    ):
        super().__init__()
        # groups (paper App. A.3): output channel c is in group c % groups and,
        # when group_inputs, only reads input channels i with i % groups == c % groups.
        # The first block reads the raw thresholded image, which has no groups,
        # so it is built with group_inputs=False.
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.tree_depth = tree_depth
        self.channels_per_tree = channels_per_tree
        n_leaves = 2**tree_depth
        kk = kernel_size * kernel_size
        window = in_channels * kk
        g = torch.Generator().manual_seed(seed)
        leaf_idx = torch.empty(out_channels, n_leaves, dtype=torch.long)
        if not channels_per_tree and groups > 1 and group_inputs:
            offsets = torch.arange(kk)
            for c in range(out_channels):
                chans = torch.nonzero(torch.arange(in_channels) % groups == c % groups).flatten()
                cand = (chans[:, None] * kk + offsets[None, :]).reshape(-1)
                leaf_idx[c] = cand[torch.randint(0, cand.numel(), (n_leaves,), generator=g)]
        elif not channels_per_tree:
            for c in range(out_channels):
                leaf_idx[c] = torch.randint(0, window, (n_leaves,), generator=g)
        else:
            offsets = torch.arange(kk)
            for c in range(out_channels):
                if groups > 1 and group_inputs:
                    pool = torch.nonzero(torch.arange(in_channels) % groups == c % groups).flatten()
                else:
                    pool = torch.arange(in_channels)
                n_ch = min(channels_per_tree, pool.numel())
                chans = pool[torch.randperm(pool.numel(), generator=g)[:n_ch]]
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

    def forward(self, x, hard=False, sharpness=1.0, ste=False, collect=None, gumbel=None, probe=None):
        b, c, h, w = x.shape
        patches = F.unfold(x, kernel_size=self.kernel_size, padding=self.kernel_size // 2)
        # patches: (B, window, L)
        patches = patches.transpose(1, 2)  # (B, L, window)
        # gather leaves: (B, L, out_channels, n_leaves)
        leaves = patches[:, :, self.leaf_idx]
        out = self.tree(
            leaves, hard=hard, sharpness=sharpness, ste=ste, collect=collect, gumbel=gumbel, probe=probe
        )  # (B, L, out_channels)
        out = out.transpose(1, 2).contiguous()  # (B, out_channels, L)
        out_h = h  # padding = kernel_size//2 keeps spatial size
        out_w = w
        return out.view(b, self.out_channels, out_h, out_w)


class OrPool2d(nn.Module):
    """Logical-OR pooling. Hard: max-pool (OR of bits == max of bits).
    Soft, mode="max" (paper Sec 3.1, the default): the maximum
    t-conorm max(a, b), i.e. max-pool in training too. Soft, mode="prob"
    (our earlier runs): probabilistic sum 1 - prod(1 - x), which pushes soft
    activations toward 1 and makes the soft network differ from the hard one."""

    def __init__(self, kernel_size=2, stride=2, mode="max"):
        super().__init__()
        if mode not in ("max", "prob"):
            raise ValueError(f"OrPool2d mode must be 'max' or 'prob', got {mode!r}")
        self.kernel_size = kernel_size
        self.stride = stride
        self.mode = mode

    def forward(self, x, hard=False):
        if hard or self.mode == "max":
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
                sl = x[:, :, i : i + s * (out_h - 1) + 1 : s, j : j + s * (out_w - 1) + 1 : s]
                keep = (1 - sl) if keep is None else keep * (1 - sl)
        return 1 - keep


# ---------------------------------------------------------------------------
# Residual units inside a conv stage (NOT in the paper; Prof. Liu's suggestion, Nov 18 2026 meeting). A unit keeps (B, C, h, w), runs a
# stack of L shape-preserving LogicTreeConv2d layers (the body) and merges the stage's input x with the body output y by a 2-input
# logic gate per channel (shared over positions). Binary safe: the merge is a gate, not an addition.
# ---------------------------------------------------------------------------

RES_MERGES = ("gate", "xor", "or", "none")


def _merge_gate_init(channels, generator, bias=5.0, noise=0.1):
    """Logits of the per-channel merge gates, biased toward pass-through of the skip input x (gate 'A', x is the first gate input), so a unit
    starts as the identity: the binary analogue of x + F(x) with F close to 0. Same bias/noise convention as _residual_gate_init."""
    logits = torch.randn(channels, 16, generator=generator) * noise
    logits[:, _PASS_A_IDX] += bias
    return logits


class MergeGate(_FreezeMixin, nn.Module):
    """One learnable 16-op gate per channel, applied elementwise to (x_c, y_c) at every position. Works in the soft, hard, STE, Gumbel
    and TaskFreeze-probe paths exactly like LogicLayer (it reuses gate_forward / gate_forward_hard / _FreezeMixin)."""

    def __init__(self, channels, seed=0, bias=5.0):
        super().__init__()
        self.channels = channels
        g = torch.Generator().manual_seed(seed)
        self.gate_logits = nn.Parameter(_merge_gate_init(channels, g, bias))

    def forward(self, x, y, hard=False, sharpness=1.0, ste=False, gumbel=None, probe=None):
        a = x.permute(0, 2, 3, 1)  # channels last, so the (C, 16) logits broadcast over (B, h, w, C)
        b = y.permute(0, 2, 3, 1)
        frozen = getattr(self, "frozen_idx", None)
        if hard:
            out = gate_forward_hard(_hard_idx(self.gate_logits, frozen), a, b)
        else:
            out = gate_forward(self.gate_logits, a, b, sharpness=sharpness, ste=ste, gumbel=gumbel, frozen=frozen)
            if probe is not None:  # TaskFreeze: same record as LogicLayer, idx = all channels
                out.retain_grad()
                probe.append((self, (slice(None),), a.detach(), b.detach(), out))
        return out.permute(0, 3, 1, 2).contiguous()


def merge_xor(x, y):
    """XOR of two soft/binary values: exact on {0, 1}, bilinear (x + y - 2xy) on [0, 1]."""
    return x + y - 2 * x * y


def merge_or(x, y):
    """OR of two soft/binary values: exact on {0, 1}, probabilistic sum (x + y - xy) on [0, 1]."""
    return x + y - x * y


class ResidualUnit(nn.Module):
    """body: L LogicTreeConv2d(C -> C, 3x3) layers, then merge(x, body(x)). merge: 'gate' (learnable per channel, starts as pass-through
    of x), 'xor', 'or' (fixed gates, no parameters) or 'none' (x = body(x): the depth-only ablation, no skip)."""

    def __init__(
        self,
        channels,
        body_layers=1,
        tree_depth=3,
        kernel_size=3,
        merge="gate",
        merge_bias=5.0,
        seed=0,
        channels_per_tree=0,
        groups=1,
    ):
        super().__init__()
        if merge not in RES_MERGES:
            raise ValueError(f"merge must be one of {RES_MERGES}, got {merge!r}")
        if body_layers < 1:
            raise ValueError(f"body_layers must be >= 1, got {body_layers}")
        self.merge_mode = merge
        self.body = nn.ModuleList(
            [
                LogicTreeConv2d(
                    channels,
                    channels,
                    kernel_size,
                    tree_depth,
                    seed=seed + 2 * j,
                    channels_per_tree=channels_per_tree,
                    groups=groups,
                    group_inputs=True,
                )
                for j in range(body_layers)
            ]
        )
        self.merge = MergeGate(channels, seed=seed + 4000, bias=merge_bias) if merge == "gate" else None

    def forward(self, x, hard=False, sharpness=1.0, ste=False, gumbel=None, probe=None):
        y = x
        for layer in self.body:
            y = layer(y, hard=hard, sharpness=sharpness, ste=ste, collect=None, gumbel=gumbel, probe=probe)
        if self.merge_mode == "gate":
            return self.merge(x, y, hard=hard, sharpness=sharpness, ste=ste, gumbel=gumbel, probe=probe)
        if self.merge_mode == "xor":
            return merge_xor(x, y)
        if self.merge_mode == "or":
            return merge_or(x, y)
        return y


def parse_res_stages(res_stages, n_stages):
    """'all' -> every stage; '2,3,4' -> those stages (1-based, stage 1 = the first conv block). Returns a sorted tuple."""
    if res_stages == "all":
        return tuple(range(1, n_stages + 1))
    try:
        st = sorted({int(s) for s in str(res_stages).split(",") if s.strip()})
    except ValueError:
        raise ValueError(f"res_stages must be 'all' or a comma list of stage numbers, got {res_stages!r}") from None
    if not st or st[0] < 1 or st[-1] > n_stages:
        raise ValueError(f"res_stages must be within 1..{n_stages}, got {res_stages!r}")
    return tuple(st)


class LogicTreeNet(nn.Module):
    """Convolutional differentiable logic gate network (paper: LogicTreeNet).

    Paper Appendix A.1.1: after the conv+pool blocks, the head is exactly
    3 dense LogicLayers of width 1280k -> 640k -> 320k, where k =
    channels[0] (the base scale, e.g. 32 for scale S). This is NOT a free
    architectural choice.
    """

    def __init__(
        self,
        in_channels,
        in_size,
        channels,
        num_classes,
        tree_depth=3,
        kernel_size=3,
        output_gate_factor=1,
        tau=20.0,
        seed=0,
        channels_per_tree=0,
        pool="max",
        groups=1,
        dropout=0.0,
        skip="none",
        skip_frac=0.25,
        dropout_mode="half",
        head_wiring="random",
        head_layers=3,
        dropout_where="all",
        dropout_in_ste=False,
        dropout_soft=True,
        res_units=0,
        res_body_layers=1,
        res_tree_depth=3,
        res_merge="gate",
        res_stages="all",
        res_merge_bias=5.0,
    ):
        # dropout / skip are NOT in the paper (Prof. Liu's suggestions); defaults
        # (0.0 / "none") build exactly the same model, same wiring, as before.
        #   dropout (+ dropout_mode half/scale, see logic_dropout): prob on the input of every conv block after the
        #     first and every head layer (soft, non-STE training only).
        #   skip "conv": block i >= 2 also reads block i-2's output, OR-pooled 2x2
        #     to the right size, as extra input channels (binary-safe concat skip;
        #     the skip part is ~20% of each block's input channels at S/M ratios).
        #   skip "head": head layer i >= 1 also reads the input of the layer before
        #     it (layer 1: flattened conv features; layer 2: layer 0's output);
        #     each gate input slot reads the skip part with prob skip_frac.
        #   skip "both": both.
        #   head_wiring "balanced" (not in the paper, which wires the head at random, Sec. 3.4): every head layer's
        #     inputs are read about equally often (see _balanced_pairs), so no head gate is left unread.
        #     Cannot be combined with skip head/both. "random" (default) builds the same model as before.
        #   head_layers (not in the paper, which fixes 3 head layers of width 1280k/640k/320k, App. A.1.1): number of dense head
        #     layers. 3 (default) = the paper's head. N > 3 keeps the total head gate count (2240k) by using N-1 equal layers of
        #     1920k/(N-1) followed by a last layer of 320k (the last width must divide the 10 classes); N = 5 -> 480k x4 + 320k,
        #     N = 7 -> 320k x7. Total layers = 4 conv blocks + head_layers.
        #   dropout_where (not in the paper): which inputs get dropout. "all" (default) = the input of every conv block after the
        #     first and of every head layer; "conv1" = only conv block 1's output (the input of block 2; block 1 reads the raw
        #     image bits); "head" = only the inputs of the head layers.
        #   dropout_mode "runmean" (not in the paper): a dropped input is replaced by that input's running mean (EMA, momentum 0.01,
        #     per channel for conv inputs and per feature for head inputs, initialized at 0.5), so the expected activation over the
        #     data is unchanged and values stay in [0, 1]. Needs no skip connections. Adds buffers rm_conv<i> / rm_head<j>.
        #   res_units (not in the paper; Prof. Liu's suggestion): residual units per conv stage, after the stage's LogicTreeConv2d and before
        #     its OR-pool, see ResidualUnit. 0 (default) builds exactly the same model as before (no extra modules or state_dict keys).
        #     res_stages 'all' or a comma list of 1-based stage numbers. Not combinable with skip conv/both or dropout_mode runmean.
        super().__init__()
        if int(res_units) != res_units or res_units < 0:
            raise ValueError(f"res_units must be an integer >= 0, got {res_units!r}")
        if res_units > 0 and skip in ("conv", "both"):
            raise ValueError("res_units > 0 cannot be combined with skip conv/both (for now)")
        if res_units > 0 and dropout_mode == "runmean":
            raise ValueError("res_units > 0 cannot be combined with dropout_mode='runmean' (for now)")
        self.res_units_n = int(res_units)
        if dropout_where not in ("all", "conv1", "head", "convlate"):
            raise ValueError(f"dropout_where must be all/conv1/head/convlate, got {dropout_where!r}")
        self.dropout_where = dropout_where
        self.dropout_soft = bool(
            dropout_soft
        )  # False (with dropout_in_ste): dropout only in the straight-through phase, not in soft training
        self.dropout_in_ste = bool(
            dropout_in_ste
        )  # not in the paper: also apply dropout in the straight-through phase (activations are binary there)
        if head_layers < 3 or int(head_layers) != head_layers:
            raise ValueError(f"head_layers must be an integer >= 3, got {head_layers!r}")
        self.head_layers_n = int(head_layers)
        if head_wiring not in ("random", "balanced"):
            raise ValueError(f"head_wiring must be 'random' or 'balanced', got {head_wiring!r}")
        if head_wiring == "balanced" and skip in ("head", "both"):
            raise ValueError("head_wiring='balanced' cannot be combined with skip head/both")
        self.head_wiring = head_wiring
        if skip not in ("none", "conv", "head", "both"):
            raise ValueError(f"skip must be none/conv/head/both, got {skip!r}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")
        if dropout_mode not in ("half", "scale", "runmean", "perm", "permx"):
            raise ValueError(
                f"dropout_mode must be 'half', 'scale', 'runmean', 'perm' or 'permx', got {dropout_mode!r}"
            )
        if dropout_mode == "runmean" and skip != "none":
            raise ValueError("dropout_mode='runmean' cannot be combined with skip connections")
        self.dropout_mode = dropout_mode
        if groups > 1:
            for c in channels:
                if c % groups:
                    raise ValueError(f"groups={groups} must divide every block's channel count {channels}")
        self.groups = groups
        self.in_size = in_size
        self.channels = tuple(channels)
        self.dropout = dropout
        self.skip = skip
        self.conv_skip = skip in ("conv", "both")
        self.head_skip = skip in ("head", "both")
        self.skip_frac = skip_frac
        self.blocks = nn.ModuleList()
        self.pools = nn.ModuleList()
        self.skip_pools = nn.ModuleList()
        c_in = in_channels
        size = in_size
        for i, c_out in enumerate(channels):
            if self.conv_skip and i >= 2:
                c_in = channels[i - 1] + channels[i - 2]
                self.skip_pools.append(OrPool2d(2, 2, mode=pool))
            self.blocks.append(
                LogicTreeConv2d(
                    c_in,
                    c_out,
                    kernel_size,
                    tree_depth,
                    seed=seed + i,
                    channels_per_tree=channels_per_tree,
                    groups=groups,
                    group_inputs=(i > 0),
                )
            )
            self.pools.append(OrPool2d(2, 2, mode=pool))
            c_in = c_out
            size = size // 2
        flat_dim = c_in * size * size
        self.res = nn.ModuleList()
        if self.res_units_n > 0:
            if res_merge not in RES_MERGES:
                raise ValueError(f"res_merge must be one of {RES_MERGES}, got {res_merge!r}")
            self.res_stages_sel = parse_res_stages(res_stages, len(channels))
            for i, c_ch in enumerate(channels):
                units = []
                if (i + 1) in self.res_stages_sel:
                    for u in range(self.res_units_n):
                        units.append(
                            ResidualUnit(
                                c_ch,
                                body_layers=res_body_layers,
                                tree_depth=res_tree_depth,
                                kernel_size=kernel_size,
                                merge=res_merge,
                                merge_bias=res_merge_bias,
                                seed=seed + 1000 + 100 * i + 10 * u,
                                channels_per_tree=channels_per_tree,
                                groups=groups,
                            )
                        )
                self.res.append(nn.ModuleList(units))

        # Paper Appendix A.1.1: k = channels[0]; head is exactly 3 dense
        # LogicLayers of width 1280k -> 640k -> 320k (not a free choice).
        k = channels[0]
        ox = output_gate_factor
        if head_layers == 3:
            head_widths = [1280 * k * ox, 640 * k * ox, 320 * k * ox]
        else:
            rest = 1920 * k * ox
            if rest % (head_layers - 1):
                raise ValueError(
                    f"1920*k*output_gate_factor = {rest} is not divisible by head_layers - 1 = {head_layers - 1}"
                )
            head_widths = [rest // (head_layers - 1)] * (head_layers - 1) + [320 * k * ox]
        assert head_widths[-1] % num_classes == 0, (
            f"final head width {head_widths[-1]} must be divisible by num_classes={num_classes}"
        )

        head_list = []
        f_in = flat_dim
        # flatten of (C, h, w) puts feature f in channel f // (h*w), whose group is channel % groups
        in_group = (torch.arange(flat_dim) // (size * size)) % groups if groups > 1 else None
        if not self.head_skip:
            for i, w in enumerate(head_widths):
                head_list.append(
                    LogicLayer(f_in, w, seed=seed + 100 + i, groups=groups, in_group=in_group, wiring=head_wiring)
                )
                f_in = w
                in_group = None  # later layers: neuron j is in group j % groups (LogicLayer default)
        else:
            G = max(groups, 1)
            flat_group = (torch.arange(flat_dim) // (size * size)) % G
            prev_dim, prev_group = flat_dim, flat_group  # input of the previous layer (skip source)
            for i, w in enumerate(head_widths):
                if i == 0:
                    head_list.append(LogicLayer(f_in, w, seed=seed + 100 + i, groups=groups, in_group=in_group))
                else:
                    main_group = torch.arange(f_in) % G
                    cat_group = torch.cat([main_group, prev_group])
                    cat_source = torch.cat(
                        [torch.zeros(f_in, dtype=torch.long), torch.ones(prev_dim, dtype=torch.long)]
                    )
                    head_list.append(
                        LogicLayer(
                            f_in + prev_dim,
                            w,
                            seed=seed + 100 + i,
                            groups=G,
                            in_group=cat_group,
                            in_source=cat_source,
                            skip_frac=skip_frac,
                        )
                    )
                    prev_dim, prev_group = f_in, main_group
                f_in = w
        self.head = nn.ModuleList(head_list)
        # inputs that get dropout (block index >= 1, head layer index)
        self._drop_conv_sites = {
            "all": set(range(1, len(self.blocks))),
            "conv1": {1},
            "head": set(),
            "convlate": {2, 3},
        }[dropout_where]
        self._drop_head_sites = {
            "all": set(range(len(self.head))),
            "conv1": set(),
            "head": set(range(len(self.head))),
            "convlate": set(),
        }[dropout_where]
        if dropout_mode == "runmean" and dropout > 0:
            for i in sorted(self._drop_conv_sites):
                self.register_buffer(f"rm_conv{i}", torch.full((self.blocks[i].in_channels,), 0.5))
            for j in sorted(self._drop_head_sites):
                self.register_buffer(f"rm_head{j}", torch.full((self.head[j].in_features,), 0.5))
        self.groupsum = GroupSum(num_classes, tau=tau)
        self.flat_dim = flat_dim

    def forward(self, x, hard=False, sharpness=1.0, ste=False, collect=None, gumbel=None, probe=None):
        # probe (optional list, soft training only): TaskFreeze, see freeze_update_from_probe. Off (None) by default.
        # gumbel (optional (scale, tau), soft training only, see gate_probs): sample every gate choice with Gumbel noise and use
        # the straight-through estimator. Ignored when the model is in eval mode, hard=True or ste=True. Off (None) by default.
        # collect (optional list, soft training only): each trainable layer appends its per-gate mean output
        # (conv block: (channels, 7); head layer: (width,)), in layer order. Used by the balance penalty.
        # ste=True: forward pass is exactly the discretized network (one-hot gates
        # on binary inputs give binary outputs, and soft OR-pooling of bits is
        # exact OR), but gradients flow through the gate softmaxes.
        # Dropout only in soft, non-STE training (never in the hard network).
        drop = (
            self.dropout
            if (self.training and not hard and (self.dropout_in_ste if ste else self.dropout_soft))
            else 0.0
        )
        gum = gumbel if (self.training and not hard and not ste) else None
        probe_ = probe if (self.training and not hard and not ste) else None
        outs = []  # pooled output of each conv block
        for i, (block, pool) in enumerate(zip(self.blocks, self.pools)):
            if self.conv_skip and i >= 2:
                x = torch.cat([x, self.skip_pools[i - 2](outs[i - 2], hard=hard)], dim=1)
            if i > 0:
                x = self._drop(x, drop if i in self._drop_conv_sites else 0.0, f"conv{i}")
            x = block(x, hard=hard, sharpness=sharpness, ste=ste, collect=collect, gumbel=gum, probe=probe_)
            if len(self.res):
                for unit in self.res[i]:
                    x = unit(x, hard=hard, sharpness=sharpness, ste=ste, gumbel=gum, probe=probe_)
            x = pool(x, hard=hard)
            outs.append(x)
        x = x.flatten(1)
        prev = None  # input of the previous head layer (skip source)
        for i, layer in enumerate(self.head):
            inp = x if (not self.head_skip or i == 0) else torch.cat([x, prev], dim=1)
            prev = x
            x = layer(
                self._drop(inp, drop if i in self._drop_head_sites else 0.0, f"head{i}"),
                hard=hard,
                sharpness=sharpness,
                ste=ste,
                collect=collect,
                gumbel=gum,
                probe=probe_,
            )
        return self.groupsum(x)

    # ---- progressive discretization (Gerlach 2026, arXiv:2609.33747): GlobalFreeze / TaskFreeze; off unless enable_freeze() ----
    def _freeze_modules(self):
        return [m for m in self.modules() if isinstance(m, (LogicLayer, LogicTree, MergeGate))]

    def enable_freeze(self):
        for m in self._freeze_modules():
            m.enable_freeze()

    def freeze_all(self):
        """GlobalFreeze: fix every neuron to its current argmax gate. Returns the number of newly frozen neurons."""
        return sum(m.freeze_where(torch.ones_like(m.frozen_idx, dtype=torch.bool)) for m in self._freeze_modules())

    def frozen_fraction(self):
        n = sum(m.frozen_idx.numel() for m in self._freeze_modules())
        return sum(int((m.frozen_idx >= 0).sum()) for m in self._freeze_modules()) / n

    def freeze_update_from_probe(self, probe, beta, patience):
        """TaskFreeze: first-order freeze shock of the argmax gate for every neuron, from one forward/backward pass with
        probe=list: s_i = mean over batch (and positions) of dl/da_i * (a_i,k - a_i), where a_i is the relaxed output and a_i,k
        the output under the argmax gate k with the same inputs (Sec. 4.2/4.3). Updates the EMA / patience counters and freezes
        neurons whose EMA stayed negative for `patience` evaluations. Neurons whose output is unused get s = 0 and never freeze."""
        newly = 0
        coeffs = GATE_COEFFS
        for mod, idx, a, b, y in probe:
            g = y.grad if y.grad is not None else torch.zeros_like(y)
            logits = mod.gate_logits[idx]
            k = logits.argmax(dim=-1)
            fz = mod.frozen_idx[idx]
            k = torch.where(fz >= 0, fz, k)
            c = coeffs.to(device=a.device, dtype=a.dtype)[k]  # (..., 4)
            c0, c1, c2, c3 = c.unbind(-1)
            ak = c0 + c1 * a + c2 * b + c3 * a * b
            s = (g * (ak - y.detach())).reshape(-1, *k.shape).mean(0)
            newly += mod.freeze_update(idx, s, beta, patience)
        return newly

    def _drop(self, x, p, key):
        """Dropout at one site. half/scale: logic_dropout (unchanged). runmean: with probability p a value is replaced by the
        running mean of its input (channel for conv inputs, feature for head inputs); the running mean is updated from the
        undropped batch (EMA, momentum 0.01) every time the site is active in soft training."""
        if self.dropout_mode != "runmean":
            return logic_dropout(x, p, mode=self.dropout_mode)
        if p <= 0.0:
            return x
        buf = getattr(self, "rm_" + key)
        with torch.no_grad():
            bm = x.mean(dim=(0, 2, 3)) if x.dim() == 4 else x.mean(dim=0)
            buf.mul_(0.99).add_(0.01 * bm)
        mean = buf.view(1, -1, 1, 1) if x.dim() == 4 else buf.view(1, -1)
        return torch.where(torch.rand_like(x) < p, mean.expand_as(x).to(x.dtype), x)

    @torch.no_grad()
    def unread_masks(self):
        """Per trainable layer, a bool tensor (True = this gate's output is never read by any later layer, so it gets no
        gradient) with the same shape as the entries that forward(collect=...) appends: conv block (channels, 7) and head
        layer (width,). A conv channel is unread if no later conv leaf (or, for the last block, no head-1 input) takes it;
        all 7 gates of an unread channel's tree are unread. A head gate is unread if no gate of the next head layer takes
        it; the last head layer feeds GroupSum and is fully read. Same rule as cluster/unread_gates.py. Models with skip
        connections are not supported."""
        if self.conv_skip or self.head_skip:
            raise NotImplementedError("unread_masks does not support skip connections")
        if self.res_units_n > 0:
            raise NotImplementedError("unread_masks does not support residual units")
        out = []
        for i, block in enumerate(self.blocks):
            C = block.out_channels
            if i < len(self.blocks) - 1:
                nxt = self.blocks[i + 1]
                read_ch = torch.unique(nxt.leaf_idx // (nxt.kernel_size**2))
            else:
                h0 = self.head[0]
                read_ch = torch.unique(torch.cat([h0.idx_a, h0.idx_b]) // (self.flat_dim // C))
            unread = ~torch.isin(torch.arange(C, device=read_ch.device), read_ch)
            out.append(unread[:, None].expand(C, block.tree.n_leaves - 1).clone())
        for j, layer in enumerate(self.head):
            n = layer.out_features
            dev = layer.idx_a.device
            if j < len(self.head) - 1:
                nxt = self.head[j + 1]
                out.append(~torch.isin(torch.arange(n, device=dev), torch.unique(torch.cat([nxt.idx_a, nxt.idx_b]))))
            else:
                out.append(torch.zeros(n, dtype=torch.bool, device=dev))
        return out

    def layer_gate_logits(self):
        """[(name, gate_logits parameter)] for the trainable layers in order: conv1..conv4, head1..headN."""
        if self.res_units_n > 0:
            raise NotImplementedError("layer_gate_logits does not support residual units")
        return [(f"conv{i + 1}", b.tree.gate_logits) for i, b in enumerate(self.blocks)] + [
            (f"head{j + 1}", l.gate_logits) for j, l in enumerate(self.head)
        ]

    def gate_confidences(self):
        confs = {}
        for i, block in enumerate(self.blocks):
            confs[f"conv_block_{i}"] = gate_confidence(block.tree.gate_logits)
            if len(self.res):
                for u, unit in enumerate(self.res[i]):
                    for j, bl in enumerate(unit.body):
                        confs[f"res_{i}_{u}_body{j}"] = gate_confidence(bl.tree.gate_logits)
                    if unit.merge is not None:
                        confs[f"res_{i}_{u}_merge"] = gate_confidence(unit.merge.gate_logits)
        for i, layer in enumerate(self.head):
            confs[f"head_{i}"] = gate_confidence(layer.gate_logits)
        return confs


def balance_penalty(means, keep):
    """Balance penalty (not in the paper): the mean over read gates g of (m_g - 0.5)^2, where m_g is the gate's mean soft
    output over the batch (and positions, for conv gates). means: list of per-layer tensors as appended by
    LogicTreeNet.forward(collect=...); keep: list of float 0/1 tensors of the same shapes (1 = gate is read by a later
    layer; unread gates are skipped). Every read gate counts once, so the head layers (86% of the gates) dominate.
    Training adds lam * balance_penalty(...) to the loss."""
    total = sum((((m - 0.5) ** 2) * k).sum() for m, k in zip(means, keep))
    count = sum(k.sum() for k in keep)
    return total / count.clamp_min(1.0)


_WIRE_OR_CONST = {GATE_NAMES.index(n) for n in ("A", "B", "FALSE", "TRUE")}


def _nontrivial(idx):
    trivial = torch.tensor(sorted(_WIRE_OR_CONST), device=idx.device)
    return int((~torch.isin(idx, trivial)).sum().item())


def count_gates(model):
    """Logic-gate count of the discretized LogicTreeNet, counted the paper's
    way: every spatial placement of a conv tree counts in full (paper Sec 5.1,
    zero-padded placements included), 3 two-input ORs per 2x2
    pooling window, one gate per head neuron. GroupSum adders are not counted.

    Returns dict(total=..., nontrivial=...). `nontrivial` drops gates whose
    chosen gate is a plain wire (A, B) or a constant (FALSE, TRUE) -- a crude
    lower-bound stand-in for the logic synthesis the paper runs before
    reporting gate counts, so the paper's numbers should sit between the two.
    """
    total = 0
    nontrivial = 0
    size = model.in_size
    for i, (block, pool) in enumerate(zip(model.blocks, model.pools)):
        if getattr(model, "conv_skip", False) and i >= 2:
            # skip input: block i-2's output OR-pooled 2x2 down to this block's input size
            sp = model.skip_pools[i - 2]
            skip_or = model.channels[i - 2] * size * size * (sp.kernel_size**2 - 1)
            total += skip_or
            nontrivial += skip_or
        idx = block.tree.gate_logits.argmax(dim=-1)  # (out_channels, n_gates)
        placements = size * size
        n = idx.numel()
        keep = _nontrivial(idx)
        total += n * placements
        nontrivial += keep * placements
        if len(getattr(model, "res", ())):
            # residual units (not in the paper): body conv trees count like conv trees; the merge is one gate per channel and position
            for unit in model.res[i]:
                for bl in unit.body:
                    bidx = bl.tree.gate_logits.argmax(dim=-1)
                    total += bidx.numel() * placements
                    nontrivial += _nontrivial(bidx) * placements
                if unit.merge_mode == "gate":
                    midx = unit.merge.gate_logits.argmax(dim=-1)
                    total += midx.numel() * placements
                    nontrivial += _nontrivial(midx) * placements
                elif unit.merge_mode in ("xor", "or"):
                    total += block.out_channels * placements
                    nontrivial += block.out_channels * placements
        size = size // 2
        or_gates = block.out_channels * size * size * (pool.kernel_size**2 - 1)
        total += or_gates
        nontrivial += or_gates
    for layer in model.head:
        idx = layer.gate_logits.argmax(dim=-1)
        total += idx.numel()
        nontrivial += _nontrivial(idx)
    return dict(total=total, nontrivial=nontrivial)


def count_trainable_gates(model):
    """Number of trainable gates = sum of the leading size of every gate_logits tensor (one gate per conv-tree node and channel, per head
    neuron, per residual merge channel), without the spatial placements that count_gates multiplies in. S default model: 83,552."""
    return sum(int(p.shape[:-1].numel()) for n, p in model.named_parameters() if n.endswith("gate_logits"))


# Paper Table 6 (CIFAR-10). Scale B removed: needs unimplemented 5-bit
# input + edge/curvature-detector preprocessing (paper Sec 3.5).
LOGICTREENET_CIFAR = {
    "S": dict(channels=(32, 128, 512, 1024), tau=20.0, reported=60.38),
    "M": dict(channels=(256, 1024, 4096, 8192), tau=40.0, reported=71.01),
}
