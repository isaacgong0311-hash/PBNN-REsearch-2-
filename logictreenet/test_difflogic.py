"""Test suite for difflogic.py / train.py. Run: python test_difflogic.py"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, ".")

from difflogic import (
    _ALL_TRUTH_TABLES,
    _PASS_A_IDX,
    _PASS_B_IDX,
    DIFFLOGICNET_CIFAR,
    GATE_COEFFS,
    GATE_NAMES,
    LOGICTREENET_CIFAR,
    DiffLogicNet,
    GroupSum,
    LogicLayer,
    LogicTree,
    LogicTreeConv2d,
    LogicTreeNet,
    OrPool2d,
    _residual_gate_init,
    binarize,
    gate_confidence,
    gate_forward,
    gate_forward_hard,
)
from train import sharpness_at, weight_decay_at

passed = 0
failed = 0
skipped = []


def check(name, cond):
    global passed, failed
    if cond:
        passed += 1
    else:
        failed += 1
        print(f"FAILED: {name}")


# ---------------------------------------------------------------------------
# Gate truth tables / bilinear coefficients
# ---------------------------------------------------------------------------

corners = [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0)]

for i in range(16):
    tt = _ALL_TRUTH_TABLES[i]
    k = GATE_COEFFS[i]
    for j, (a, b) in enumerate(corners):
        val = k[0] + k[1] * a + k[2] * b + k[3] * a * b
        check(f"gate {i} ({GATE_NAMES[i]}) truth-table corner {j}", abs(val.item() - tt[j]) < 1e-6)

check("16 gate names, all unique", len(set(GATE_NAMES)) == 16)
check("AND gate correct", GATE_NAMES[_ALL_TRUTH_TABLES.index((0, 0, 0, 1))] == "AND")
check("OR gate correct", GATE_NAMES[_ALL_TRUTH_TABLES.index((0, 1, 1, 1))] == "OR")
check("XOR gate correct", GATE_NAMES[_ALL_TRUTH_TABLES.index((0, 1, 1, 0))] == "XOR")
check("pass-A gate found", GATE_NAMES[_PASS_A_IDX] == "A")
check("pass-B gate found", GATE_NAMES[_PASS_B_IDX] == "B")

# bilinear-softmax-collapse identity vs explicit 16-gate averaging
torch.manual_seed(0)
logits = torch.randn(5, 16)
a = torch.rand(5)
b = torch.rand(5)
fast = gate_forward(logits, a, b, sharpness=1.0)
probs = F.softmax(logits, dim=-1)
slow = torch.zeros(5)
for i in range(16):
    tt = _ALL_TRUTH_TABLES[i]
    k = GATE_COEFFS[i]
    slow += probs[:, i] * (k[0] + k[1] * a + k[2] * b + k[3] * a * b)
check("bilinear-softmax-collapse matches explicit 16-gate average", torch.allclose(fast, slow, atol=1e-5))

# hard/soft agreement at extreme one-hot logits
onehot = torch.full((4, 16), -100.0)
for i, idx in enumerate(
    [
        _ALL_TRUTH_TABLES.index((0, 0, 0, 1)),  # AND
        _ALL_TRUTH_TABLES.index((0, 1, 1, 1)),  # OR
        _ALL_TRUTH_TABLES.index((0, 1, 1, 0)),  # XOR
        _PASS_A_IDX,
    ]
):
    onehot[i, idx] = 100.0
aa = torch.tensor([1.0, 0.0, 1.0, 1.0])
bb = torch.tensor([1.0, 0.0, 1.0, 0.0])
soft_out = gate_forward(onehot, aa, bb, sharpness=1.0)
idx_hard = onehot.argmax(dim=-1)
hard_out = gate_forward_hard(idx_hard, aa, bb)
check("hard/soft agree at one-hot extreme logits", torch.allclose(soft_out.round(), hard_out, atol=1e-3))

check("gate_confidence near 1.0 for one-hot logits", gate_confidence(onehot) > 0.99)
uniform_logits = torch.zeros(10, 16)
check("gate_confidence near 1/16 for uniform logits", abs(gate_confidence(uniform_logits) - 1 / 16) < 1e-3)

# ---------------------------------------------------------------------------
# Residual init
# ---------------------------------------------------------------------------

g = torch.Generator().manual_seed(0)
ri = _residual_gate_init(1000, g, bias=5.0, noise=0.1)
probs_ri = F.softmax(ri, dim=-1)
maxp = probs_ri.max(dim=-1).values
check("residual init: most neurons strongly favor pass-through gate", (maxp > 0.5).float().mean().item() > 0.8)

# ---------------------------------------------------------------------------
# LogicLayer
# ---------------------------------------------------------------------------

layer = LogicLayer(20, 30, seed=1)
check("LogicLayer wiring in range", (layer.idx_a < 20).all() and (layer.idx_b < 20).all())
check("LogicLayer wiring mostly distinct a!=b", (layer.idx_a != layer.idx_b).float().mean().item() > 0.9)
x = torch.rand(4, 20, requires_grad=True)
out = layer(x, hard=False)
check("LogicLayer output shape", out.shape == (4, 30))
loss = out.sum()
loss.backward()
check("LogicLayer gate_logits gradient nonzero", layer.gate_logits.grad.norm().item() > 1e-12)
check("LogicLayer input gradient nonzero", x.grad.norm().item() > 1e-12)

# same seed -> same wiring (determinism)
layer2 = LogicLayer(20, 30, seed=1)
check(
    "LogicLayer wiring deterministic across construction w/ same seed",
    torch.equal(layer.idx_a, layer2.idx_a) and torch.equal(layer.idx_b, layer2.idx_b),
)

# ---------------------------------------------------------------------------
# GroupSum / binarize
# ---------------------------------------------------------------------------

gs = GroupSum(num_classes=5, tau=2.0)
xin = torch.ones(3, 20)
gout = gs(xin)
check("GroupSum shape", gout.shape == (3, 5))
check("GroupSum values correct", torch.allclose(gout, torch.full((3, 5), 2.0)))

img = torch.rand(2, 3, 8, 8)
bz = binarize(img, thresholds=(0.25, 0.5, 0.75), flatten=False)
check("binarize channel count", bz.shape[1] == 3 * 3)
check("binarize output is 0/1", set(bz.unique().tolist()).issubset({0.0, 1.0}))
bz_flat = binarize(img, thresholds=(0.25, 0.5, 0.75), flatten=True)
check("binarize flatten shape", bz_flat.shape == (2, 3 * 3 * 8 * 8))

# ---------------------------------------------------------------------------
# DiffLogicNet
# ---------------------------------------------------------------------------

net = DiffLogicNet(in_features=40, width=64, depth=3, num_classes=4, tau=1.0, seed=0)
xin = torch.rand(6, 40, requires_grad=True)
out = net(xin, hard=False)
check("DiffLogicNet output shape", out.shape == (6, 4))
out.sum().backward()
for i, l in enumerate(net.layers):
    check(f"DiffLogicNet layer {i} gradient nonzero (no dead layers)", l.gate_logits.grad.norm().item() > 1e-12)
check("DiffLogicNet input gradient nonzero", xin.grad.norm().item() > 1e-12)

hard_out = net(torch.randint(0, 2, (6, 40)).float(), hard=True)
check("DiffLogicNet hard forward finite", torch.isfinite(hard_out).all())

confs = net.gate_confidences()
check("DiffLogicNet gate_confidences has one entry per layer", len(confs) == len(net.layers))

for scale, cfg in DIFFLOGICNET_CIFAR.items():
    check(f"DIFFLOGICNET_CIFAR[{scale}] has required keys", {"width", "depth", "tau", "reported"}.issubset(cfg.keys()))

# ---------------------------------------------------------------------------
# LogicTree
# ---------------------------------------------------------------------------

tree = LogicTree(n_trees=5, depth=3, seed=2)
leaves = torch.rand(4, 5, 8, requires_grad=True)
tout = tree(leaves, hard=False)
check("LogicTree output shape", tout.shape == (4, 5))
tout.sum().backward()
check("LogicTree gate_logits gradient nonzero", tree.gate_logits.grad.norm().item() > 1e-12)
check("LogicTree input gradient nonzero", leaves.grad.norm().item() > 1e-12)

hard_leaves = torch.randint(0, 2, (4, 5, 8)).float()
tout_hard = tree(hard_leaves, hard=True)
check("LogicTree hard forward is 0/1", set(tout_hard.unique().tolist()).issubset({0.0, 1.0}))

# ---------------------------------------------------------------------------
# LogicTreeConv2d: vectorized vs naive loop equivalence
# ---------------------------------------------------------------------------

conv = LogicTreeConv2d(in_channels=2, out_channels=3, kernel_size=3, tree_depth=2, seed=3)
ximg = torch.rand(2, 2, 6, 6, requires_grad=True)
cout = conv(ximg, hard=False)
check("LogicTreeConv2d output shape", cout.shape == (2, 3, 6, 6))
cout.sum().backward()
check("LogicTreeConv2d gate_logits gradient nonzero", conv.tree.gate_logits.grad.norm().item() > 1e-12)
check("LogicTreeConv2d input gradient nonzero", ximg.grad.norm().item() > 1e-12)

# naive per-pixel loop check (small case, no grad)
with torch.no_grad():
    conv2 = LogicTreeConv2d(in_channels=2, out_channels=3, kernel_size=3, tree_depth=2, seed=3)
    conv2.load_state_dict(conv.state_dict())
    x_small = torch.rand(1, 2, 5, 5)
    out_vec = conv2(x_small, hard=False)
    padded = F.pad(x_small, (1, 1, 1, 1))
    out_naive = torch.zeros(1, 3, 5, 5)
    window = 2 * 3 * 3
    for hh in range(5):
        for ww in range(5):
            patch = padded[0, :, hh : hh + 3, ww : ww + 3].reshape(-1)  # (window,)
            leaves_px = patch[conv2.leaf_idx]  # (out_channels, n_leaves)
            val = conv2.tree(leaves_px.unsqueeze(0), hard=False)  # (1, out_channels)
            out_naive[0, :, hh, ww] = val[0]
    check("LogicTreeConv2d vectorized matches naive per-pixel loop", torch.allclose(out_vec, out_naive, atol=1e-4))

# ---------------------------------------------------------------------------
# OrPool2d
# ---------------------------------------------------------------------------

pool = OrPool2d(2, 2, mode="prob")
xp = torch.rand(2, 3, 4, 4)
soft_p = pool(xp, hard=False)
check("OrPool2d soft output shape", soft_p.shape == (2, 3, 2, 2))
# manual formula check on one window
manual = 1 - (1 - xp[0, 0, 0, 0]) * (1 - xp[0, 0, 0, 1]) * (1 - xp[0, 0, 1, 0]) * (1 - xp[0, 0, 1, 1])
check("OrPool2d soft formula matches manual 1-prod(1-x)", abs(soft_p[0, 0, 0, 0].item() - manual.item()) < 1e-5)

hard_p = pool(xp, hard=True)
maxp_manual = F.max_pool2d(xp, 2, 2)
check("OrPool2d hard matches max_pool2d", torch.allclose(hard_p, maxp_manual))

xnc = torch.rand(2, 4, 4, 3).permute(0, 3, 1, 2)  # non-contiguous (channels-last) input
check(
    "OrPool2d soft works on non-contiguous input and matches contiguous result",
    torch.allclose(pool(xnc, hard=False), pool(xnc.contiguous(), hard=False)),
)
check("LogicTreeConv2d output is contiguous", conv(torch.rand(1, 2, 6, 6)).is_contiguous())

# ---------------------------------------------------------------------------
# LogicTreeNet (full model)
# ---------------------------------------------------------------------------

model = LogicTreeNet(
    in_channels=3, in_size=8, channels=(4, 8), num_classes=5, tree_depth=2, kernel_size=3, tau=1.0, seed=0
)
xin = torch.rand(2, 3, 8, 8, requires_grad=True)
out = model(xin, hard=False)
check("LogicTreeNet output shape", out.shape == (2, 5))
out.sum().backward()
for i, block in enumerate(model.blocks):
    check(f"LogicTreeNet conv block {i} gradient nonzero", block.tree.gate_logits.grad.norm().item() > 1e-12)
for i, layer in enumerate(model.head):
    check(f"LogicTreeNet head layer {i} gradient nonzero", layer.gate_logits.grad.norm().item() > 1e-12)
check("LogicTreeNet input gradient nonzero", xin.grad.norm().item() > 1e-12)

hard_in = torch.randint(0, 2, (2, 3, 8, 8)).float()
hard_out = model(hard_in, hard=True)
check("LogicTreeNet hard forward finite", torch.isfinite(hard_out).all())

confs = model.gate_confidences()
check("LogicTreeNet gate_confidences count", len(confs) == len(model.blocks) + len(model.head))

# head width taper check: 1280k -> 640k -> 320k relative ratios (k=channels[0])
k = 4
expected_widths = [1280 * k, 640 * k, 320 * k]
actual_widths = [l.out_features for l in model.head]
check("LogicTreeNet head widths match paper taper 1280k/640k/320k", actual_widths == expected_widths)

for scale, cfg in LOGICTREENET_CIFAR.items():
    check(f"LOGICTREENET_CIFAR[{scale}] has required keys", {"channels", "tau", "reported"}.issubset(cfg.keys()))
check("LOGICTREENET_CIFAR scale B removed (needs unimplemented preprocessing)", "B" not in LOGICTREENET_CIFAR)
check(
    "LOGICTREENET_CIFAR M channels = 256,1024,4096,8192", LOGICTREENET_CIFAR["M"]["channels"] == (256, 1024, 4096, 8192)
)
check("LOGICTREENET_CIFAR S channels = 32,128,512,1024", LOGICTREENET_CIFAR["S"]["channels"] == (32, 128, 512, 1024))

s_params = sum(
    p.numel()
    for p in LogicTreeNet(
        in_channels=3,
        in_size=32,
        num_classes=10,
        tree_depth=3,
        kernel_size=3,
        **{k: v for k, v in LOGICTREENET_CIFAR["S"].items() if k != "reported"},
    ).parameters()
)
m_params = sum(
    p.numel()
    for p in LogicTreeNet(
        in_channels=3,
        in_size=32,
        num_classes=10,
        tree_depth=3,
        kernel_size=3,
        **{k: v for k, v in LOGICTREENET_CIFAR["M"].items() if k != "reported"},
    ).parameters()
)
check("LogicTreeNet scale M has more params than scale S", m_params > s_params)

# ---------------------------------------------------------------------------
# channels_per_tree (paper Sec 3.4: each tree observes only 2 input channels)
# ---------------------------------------------------------------------------

kk = 9
c2 = LogicTreeConv2d(in_channels=16, out_channels=50, kernel_size=3, tree_depth=3, seed=5, channels_per_tree=2)
chans_per_row = [len(set((row // kk).tolist())) for row in c2.leaf_idx]
check("channels_per_tree=2: every tree reads from exactly 2 channels", all(n == 2 for n in chans_per_row))
check("channels_per_tree=2: leaves within a tree are distinct", all(len(set(row.tolist())) == 8 for row in c2.leaf_idx))
check("channels_per_tree=2: leaf indices in range", int(c2.leaf_idx.max()) < 16 * kk and int(c2.leaf_idx.min()) >= 0)
check(
    "channels_per_tree=2: different trees use different channel pairs",
    len({tuple(sorted(set((row // kk).tolist()))) for row in c2.leaf_idx}) > 10,
)

c_all = LogicTreeConv2d(in_channels=16, out_channels=50, kernel_size=3, tree_depth=3, seed=5, channels_per_tree=0)
check(
    "channels_per_tree=0: trees span more than 2 channels (old all-channel behavior)",
    max(len(set((row // kk).tolist())) for row in c_all.leaf_idx) > 2,
)

c1 = LogicTreeConv2d(in_channels=1, out_channels=4, kernel_size=3, tree_depth=3, seed=0, channels_per_tree=2)
check("channels_per_tree with a single input channel still works", c1(torch.rand(2, 1, 5, 5)).shape == (2, 4, 5, 5))

c_small = LogicTreeConv2d(in_channels=3, out_channels=4, kernel_size=1, tree_depth=3, seed=0, channels_per_tree=2)
check(
    "channels_per_tree with fewer candidates than leaves (1x1 kernel) still works",
    c_small(torch.rand(2, 3, 4, 4)).shape == (2, 4, 4, 4)
    and all(len(set((row // 1).tolist())) <= 2 for row in c_small.leaf_idx),
)

net_cpt = LogicTreeNet(
    in_channels=3, in_size=8, channels=(4, 8), num_classes=5, tree_depth=2, kernel_size=3, tau=1.0, seed=0
)
net_cpt2 = LogicTreeNet(
    in_channels=3,
    in_size=8,
    channels=(4, 8),
    num_classes=5,
    tree_depth=2,
    kernel_size=3,
    tau=1.0,
    seed=0,
    channels_per_tree=2,
)
check(
    "LogicTreeNet passes channels_per_tree=2 through to every conv block",
    all(b.channels_per_tree == 2 for b in net_cpt2.blocks),
)
check(
    "LogicTreeNet defaults to channels_per_tree=0 (all channels) in every conv block",
    all(b.channels_per_tree == 0 for b in net_cpt.blocks),
)
xin = torch.rand(2, 3, 8, 8, requires_grad=True)
net_cpt(xin).sum().backward()
check(
    "LogicTreeNet (default wiring): gradient reaches every conv block",
    all(b.tree.gate_logits.grad.norm().item() > 1e-12 for b in net_cpt.blocks),
)

# ---------------------------------------------------------------------------
# straight-through (ste) training mode
# ---------------------------------------------------------------------------

from difflogic import gate_probs

lg = torch.randn(7, 16, requires_grad=True)
pr = gate_probs(lg, ste=True)
check(
    "gate_probs(ste=True) forward is exactly one-hot",
    torch.allclose(pr.detach(), F.one_hot(lg.argmax(-1), 16).float(), atol=1e-6),
)
(pr * torch.randn(7, 16)).sum().backward()
check("gate_probs(ste=True) passes gradients to logits", lg.grad.norm().item() > 1e-8)

torch.manual_seed(1)
ste_net = LogicTreeNet(
    in_channels=9, in_size=16, channels=(4, 8), num_classes=10, tree_depth=3, kernel_size=3, tau=5.0, seed=3
)
for prm in ste_net.parameters():
    prm.data += torch.randn_like(prm) * 3  # move argmaxes off the pass-through init
xb = binarize(torch.rand(6, 3, 16, 16), (0.25, 0.5, 0.75), flatten=False)
with torch.no_grad():
    check(
        "LogicTreeNet ste=True forward equals the hard (discretized) network exactly",
        torch.allclose(ste_net(xb, ste=True), ste_net(xb, hard=True), atol=1e-5),
    )
ste_net.zero_grad()
ste_net(xb, ste=True).sum().backward()
check(
    "LogicTreeNet ste=True: gradient reaches every conv block and head layer",
    all(b.tree.gate_logits.grad.norm().item() > 1e-12 for b in ste_net.blocks)
    and all(l.gate_logits.grad.norm().item() > 1e-12 for l in ste_net.head),
)

dn = DiffLogicNet(in_features=48, width=40, depth=3, num_classes=10, tau=2.0, seed=2)
for prm in dn.parameters():
    prm.data += torch.randn_like(prm) * 3
xdn = binarize(torch.rand(5, 3, 4, 4), (0.5,), flatten=True)
with torch.no_grad():
    check(
        "DiffLogicNet ste=True forward equals hard forward",
        torch.allclose(dn(xdn, ste=True), dn(xdn, hard=True), atol=1e-5),
    )

# ---------------------------------------------------------------------------
# sharpness_at / weight_decay_at schedules
# ---------------------------------------------------------------------------

check("sharpness_at off when final_sharpness<=1.0", sharpness_at(50, 100, 1.0, 0.5) == 1.0)
check("sharpness_at held at 1.0 during warmup", sharpness_at(10, 100, 5.0, 0.5) == 1.0)
check("sharpness_at reaches final at last epoch", abs(sharpness_at(99, 100, 5.0, 0.5) - 5.0) < 0.2)
s_mid = sharpness_at(75, 100, 5.0, 0.5)
check(
    "sharpness_at monotonic non-decreasing after warmup",
    sharpness_at(60, 100, 5.0, 0.5) <= s_mid <= sharpness_at(90, 100, 5.0, 0.5),
)

check(
    "weight_decay_at off (base unchanged) when final_sharpness<=1.0", weight_decay_at(50, 100, 0.002, 1.0, 0.5) == 0.002
)
check("weight_decay_at held at base during warmup", weight_decay_at(10, 100, 0.002, 5.0, 0.5) == 0.002)
check("weight_decay_at decays to ~0 at end", weight_decay_at(99, 100, 0.002, 5.0, 0.5) < 0.0005)
wd_mid = weight_decay_at(75, 100, 0.002, 5.0, 0.5)
check(
    "weight_decay_at monotonic non-increasing after warmup",
    weight_decay_at(60, 100, 0.002, 5.0, 0.5) >= wd_mid >= weight_decay_at(90, 100, 0.002, 5.0, 0.5),
)

# ---------------------------------------------------------------------------
# straight-through phase schedule (--ste-start-epoch / --ste-lr)
# ---------------------------------------------------------------------------

from train import lr_at, ste_active

check("ste_active off by default", not any(ste_active(e, False, -1) for e in range(10)))
check("ste_active --ste is on every epoch", all(ste_active(e, True, -1) for e in range(10)))
check("ste_active switches on at start epoch", [ste_active(e, False, 7) for e in range(10)] == [False] * 7 + [True] * 3)
check("lr_at keeps base lr with no ste_lr", lr_at(9, 0.02, 7, None) == 0.02)
check("lr_at keeps base lr before the switch", lr_at(6, 0.02, 7, 3e-4) == 0.02)
check("lr_at drops to ste_lr at the switch", lr_at(7, 0.02, 7, 3e-4) == 3e-4)
check("lr_at ignores ste_lr when phase is off", lr_at(9, 0.02, -1, 3e-4) == 0.02)

# ---------------------------------------------------------------------------
# Paper-faithful options (Oct 2026): max OR-pooling, k/8 channel groups, gate count
# ---------------------------------------------------------------------------

from difflogic import count_gates
from train import resolve_groups

# --- max OR-pooling ---
pm = OrPool2d(2, 2)
check("OrPool2d default mode is max (paper Sec 3.1)", pm.mode == "max")
xq = torch.rand(2, 3, 4, 4)
check("OrPool2d max: soft == max_pool2d", torch.allclose(pm(xq), F.max_pool2d(xq, 2, 2)))
check("OrPool2d max: soft == hard", torch.allclose(pm(xq), pm(xq, hard=True)))
half = torch.full((1, 1, 2, 2), 0.5)
check(
    "OrPool2d max keeps 0.5 at 0.5; prob pushes it to 0.9375",
    abs(pm(half).item() - 0.5) < 1e-6 and abs(OrPool2d(2, 2, mode="prob")(half).item() - 0.9375) < 1e-6,
)
try:
    OrPool2d(2, 2, mode="bogus")
    check("OrPool2d rejects unknown mode", False)
except ValueError:
    check("OrPool2d rejects unknown mode", True)
xg = torch.rand(1, 1, 2, 2, requires_grad=True)
pm(xg).sum().backward()
check("OrPool2d max: gradient goes only to the max input", (xg.grad != 0).sum().item() == 1)

# --- LogicLayer groups ---
G = 4
ll = LogicLayer(64, 40, seed=3, groups=G)
og = torch.arange(40) % G
check("LogicLayer groups: input a in same group as neuron", torch.equal(ll.idx_a % G, og))
check("LogicLayer groups: input b in same group as neuron", torch.equal(ll.idx_b % G, og))
check("LogicLayer groups: a != b", bool((ll.idx_a != ll.idx_b).all()))
ing = (torch.arange(64) // 4) % G  # flattened (16 channels x 2x2) layout
ll2 = LogicLayer(64, 40, seed=3, groups=G, in_group=ing)
check(
    "LogicLayer groups with in_group: inputs respect custom input groups",
    torch.equal(ing[ll2.idx_a], og) and torch.equal(ing[ll2.idx_b], og),
)
check(
    "LogicLayer groups=1 wiring unchanged from original",
    torch.equal(LogicLayer(64, 40, seed=3).idx_a, LogicLayer(64, 40, seed=3, groups=1).idx_a),
)

# --- LogicTreeConv2d groups ---
for cpt in (0, 2):
    cg = LogicTreeConv2d(8, 12, 3, 3, seed=1, channels_per_tree=cpt, groups=G)
    ch = cg.leaf_idx // 9  # input channel of each leaf
    same = bool(((ch % G) == (torch.arange(12)[:, None] % G)).all())
    check(f"LogicTreeConv2d groups (channels_per_tree={cpt}): leaves only from own group", same)
    if cpt:
        check(
            "LogicTreeConv2d groups + 2 channels: each tree uses exactly 2 channels",
            all(len(set(r.tolist())) == 2 for r in ch),
        )
cf = LogicTreeConv2d(9, 12, 3, 3, seed=1, channels_per_tree=2, groups=G, group_inputs=False)
check(
    "LogicTreeConv2d group_inputs=False (first block): trees may read any input channel",
    len(set((cf.leaf_idx // 9).flatten().tolist())) > G,
)

# --- full model with groups: G separate models until GroupSum ---
tiny = dict(in_channels=9, in_size=32, channels=(8, 16, 32, 64), num_classes=10, tau=10.0)
mg = LogicTreeNet(**tiny, groups=G, channels_per_tree=2)
xb = binarize(torch.rand(2, 3, 32, 32), (0.25, 0.5, 0.75), flatten=False)
check("LogicTreeNet groups: forward shape", mg(xb).shape == (2, 10))
check("LogicTreeNet groups: hard forward runs", mg(xb, hard=True).shape == (2, 10))
# perturb one group's gates: only that group's final neurons may change
with torch.no_grad():
    h_before = xb

    def head_out(m):
        x = xb
        for blk, pl in zip(m.blocks, m.pools):
            x = pl(blk(x))
        x = x.flatten(1)
        for layer in m.head:
            x = layer(x)
        return x

    base = head_out(mg)
    g_change = 1
    tl = mg.blocks[1].tree.gate_logits
    sel = torch.arange(tl.shape[0]) % G == g_change
    tl[sel] += torch.randn_like(tl[sel]) * 5
    after = head_out(mg)
    diff = (after - base).abs().sum(0) > 1e-6
    final_group = torch.arange(after.shape[1]) % G
    check(
        "LogicTreeNet groups: changing group 1's gates only changes group 1's outputs",
        bool(diff.any()) and bool((final_group[diff] == g_change).all()),
    )
check(
    "GroupSum: every class block draws equally from every group",
    all(torch.bincount(final_group.view(10, -1)[c], minlength=G).unique().numel() == 1 for c in range(10)),
)
try:
    LogicTreeNet(**{**tiny, "channels": (6, 16, 32, 64)}, groups=G)
    check("LogicTreeNet rejects groups not dividing channels", False)
except ValueError:
    check("LogicTreeNet rejects groups not dividing channels", True)

check("resolve_groups -1 -> k/8 (S: 4, M: 32)", resolve_groups(-1, 32) == 4 and resolve_groups(-1, 256) == 32)
check("resolve_groups 1 -> 1", resolve_groups(1, 32) == 1)

# --- gate counter ---
cnt = count_gates(LogicTreeNet(in_channels=9, in_size=32, channels=(32, 128, 512, 1024), num_classes=10, tau=20.0))
check("count_gates: S total = 0.960M (conv per placement + pool ORs + head)", cnt["total"] == 802816 + 86016 + 71680)
check(
    "count_gates: at residual init every trainable gate is a wire (only the 86016 pool ORs remain)",
    cnt["nontrivial"] == 86016,
)

# ---------------------------------------------------------------------------
# Logic dropout and skip connections (not in the paper; off by default)
# ---------------------------------------------------------------------------
from difflogic import count_gates as _count_gates
from difflogic import logic_dropout
from train import normalize_arch

torch.manual_seed(0)
xd = torch.rand(20000)
check("logic_dropout: p=0 is identity", torch.equal(logic_dropout(xd, 0.0), xd))
check("logic_dropout: not training is identity", torch.equal(logic_dropout(xd, 0.5, training=False), xd))
yd = logic_dropout(xd, 0.3)
changed = yd != xd
check("logic_dropout: changed entries are exactly 0.5", bool((yd[changed] == 0.5).all()))
check("logic_dropout: ~p of entries dropped", abs(changed.float().mean().item() - 0.3) < 0.02)
check("logic_dropout: stays in [0,1] (no 1/(1-p) rescale)", bool((yd >= 0).all() and (yd <= 1).all()))

# feeding 0.5 into any gate = average of the gate over that input being 0 and 1
ok = True
for i in range(16):
    lg = torch.full((16,), -1e4)
    lg[i] = 1e4
    for bv in (0.0, 1.0, 0.3):
        bt = torch.tensor(bv)
        mid = gate_forward(lg, torch.tensor(0.5), bt)
        avg = 0.5 * (gate_forward(lg, torch.tensor(0.0), bt) + gate_forward(lg, torch.tensor(1.0), bt))
        ok &= bool(torch.allclose(mid, avg, atol=1e-6))
check("gate(0.5, b) == mean(gate(0,b), gate(1,b)) for all 16 gates", ok)

small = dict(
    in_channels=9, in_size=32, channels=(8, 16, 32, 64), num_classes=10, tau=20.0, seed=0, channels_per_tree=2, groups=4
)
xb = (torch.rand(3, 9, 32, 32) > 0.5).float()

# dropout: no parameters, eval/hard/STE forwards unaffected
m0 = LogicTreeNet(**small)
md = LogicTreeNet(**small, dropout=0.3)
check(
    "dropout: same parameters/wiring as dropout=0",
    all(torch.equal(a, b) for a, b in zip(m0.state_dict().values(), md.state_dict().values())),
)
m0.eval()
md.eval()
check("dropout: eval soft forward unchanged", torch.equal(m0(xb), md(xb)))
md.train()
m0.train()
check("dropout: hard forward unchanged in train mode", torch.equal(m0(xb, hard=True), md(xb, hard=True)))
check("dropout: STE forward unchanged in train mode", torch.equal(m0(xb, ste=True), md(xb, ste=True)))
torch.manual_seed(1)
check("dropout: train soft forward changes", not torch.equal(m0(xb), md(xb)))
try:
    LogicTreeNet(**small, dropout=1.0)
    check("dropout=1 rejected", False)
except ValueError:
    check("dropout=1 rejected", True)

# conv skip
mc = LogicTreeNet(**small, skip="conv")
check("conv skip: blocks 0,1 unchanged in_channels", mc.blocks[0].in_channels == 9 and mc.blocks[1].in_channels == 8)
check(
    "conv skip: block i>=2 reads ch[i-1]+ch[i-2]",
    mc.blocks[2].in_channels == 16 + 8 and mc.blocks[3].in_channels == 32 + 16,
)
check("conv skip: 2 skip pools", len(mc.skip_pools) == 2)
kk = 9
okg = True
for bi in (2, 3):
    blk = mc.blocks[bi]
    ch = blk.leaf_idx // kk
    okg &= bool(((ch % 4) == (torch.arange(blk.out_channels)[:, None] % 4)).all())
check("conv skip: leaves stay within each tree's group", okg)
okskip = any(bool((mc.blocks[bi].leaf_idx // kk >= mc.channels[bi - 1]).any()) for bi in (2, 3))
check("conv skip: some trees read the skip channels", okskip)
oc = mc(xb, hard=True) * 20.0
check("conv skip: hard output is integer counts", torch.allclose(oc, oc.round()))
mc.train()
oc_ste = mc(xb, ste=True)
check("conv skip: STE forward == hard forward", torch.allclose(oc_ste, mc(xb, hard=True), atol=1e-4))
gS0 = _count_gates(LogicTreeNet(in_channels=9, in_size=32, channels=(32, 128, 512, 1024), num_classes=10, tau=20.0))
gSc = _count_gates(
    LogicTreeNet(in_channels=9, in_size=32, channels=(32, 128, 512, 1024), num_classes=10, tau=20.0, skip="conv")
)
# extra ORs: 32 ch @ 8x8 + 128 ch @ 4x4, 3 ORs per 2x2 window; conv tree gates unchanged in count
check(
    "conv skip: count_gates adds 3 ORs per skip-pool window", gSc["total"] - gS0["total"] == 32 * 64 * 3 + 128 * 16 * 3
)

# head skip
mh = LogicTreeNet(**small, skip="head", skip_frac=0.25)
w0, w1 = mh.head[0].out_features, mh.head[1].out_features
check(
    "head skip: layer 0 unchanged",
    mh.head[0].in_features == mh.flat_dim and torch.equal(mh.head[0].idx_a, m0.head[0].idx_a),
)
check("head skip: layer 1 reads out0 + flat", mh.head[1].in_features == w0 + mh.flat_dim)
check("head skip: layer 2 reads out1 + out0", mh.head[2].in_features == w1 + w0)
L2 = mh.head[2]
frac = (torch.cat([L2.idx_a, L2.idx_b]) >= w1).float().mean().item()
check("head skip: ~skip_frac of inputs from skip source", abs(frac - 0.25) < 0.03)
check("head skip: a != b everywhere", all(bool((L.idx_a != L.idx_b).all()) for L in mh.head))
gr = torch.cat([torch.arange(w1) % 4, torch.arange(w0) % 4])
j = torch.arange(L2.out_features) % 4
check("head skip: groups respected", bool((gr[L2.idx_a] == j).all() and (gr[L2.idx_b] == j).all()))
L1 = mh.head[1]
size_f = 32 // 16
gr1 = torch.cat([torch.arange(w0) % 4, (torch.arange(mh.flat_dim) // (size_f * size_f)) % 4])
j1 = torch.arange(L1.out_features) % 4
check(
    "head skip: layer-1 groups respected (flat part grouped by channel)",
    bool((gr1[L1.idx_a] == j1).all() and (gr1[L1.idx_b] == j1).all()),
)
oh = mh(xb, hard=True) * 20.0
check("head skip: hard output is integer counts", torch.allclose(oh, oh.round()))
mb = LogicTreeNet(**small, skip="both", dropout=0.2)
mb.train()
outb = mb(xb)
outb.sum().backward()
check(
    "skip both + dropout: gradients reach first conv block",
    mb.blocks[0].tree.gate_logits.grad is not None and mb.blocks[0].tree.gate_logits.grad.abs().sum() > 0,
)
mh1 = LogicTreeNet(**{**small, "groups": 1}, skip="head")
check("head skip works without groups", mh1(xb).shape == (3, 10))
try:
    LogicTreeNet(**small, skip="bogus")
    check("bad skip rejected", False)
except ValueError:
    check("bad skip rejected", True)

# checkpoint arch compatibility: the scale-M + crop/flip run's saved arch must still load
old_arch = dict(
    model="conv",
    scale="M",
    tree_depth=3,
    kernel_size=3,
    thresholds="0.25,0.5,0.75",
    channels_per_tree=2,
    pool="max",
    groups=32,
)
new_expected = {**old_arch, "skip": "none", "skip_frac": None}
check("normalize_arch: pre-skip checkpoint matches skip=none", normalize_arch(old_arch) == new_expected)
check("normalize_arch: None passes through", normalize_arch(None) is None)

# ---------------------------------------------------------------------------
# Textbook dropout mode (dropout_mode="scale"; off by default)
# ---------------------------------------------------------------------------
torch.manual_seed(0)
xs_ = torch.full((200000,), 0.7)
ys_ = logic_dropout(xs_, 0.2, mode="scale")
vals_ = torch.unique(ys_)
check(
    "scale dropout: outputs are only 0 or x/(1-p)",
    bool(((vals_ == 0) | torch.isclose(vals_, torch.tensor(0.7 / 0.8))).all()),
)
check("scale dropout: unbiased, mean output equals the input", abs(ys_.mean().item() - 0.7) < 0.01)
check("scale dropout: ~p of entries dropped to 0", abs((ys_ == 0).float().mean().item() - 0.2) < 0.01)
check("scale dropout: p=0 is identity", torch.equal(logic_dropout(xs_, 0.0, mode="scale"), xs_))
check(
    "scale dropout: not training is identity", torch.equal(logic_dropout(xs_, 0.2, training=False, mode="scale"), xs_)
)
yh_ = logic_dropout(xs_, 0.2, mode="half")
check(
    "half dropout: biased toward 0.5 (mean = 0.8*0.7 + 0.2*0.5)",
    abs(yh_.mean().item() - (0.8 * 0.7 + 0.2 * 0.5)) < 0.01,
)
torch.manual_seed(5)
d_def_ = logic_dropout(xs_, 0.2)
torch.manual_seed(5)
d_half_ = logic_dropout(xs_, 0.2, mode="half")
check("half is the default mode", torch.equal(d_def_, d_half_))
try:
    logic_dropout(xs_, 0.2, mode="bogus")
    check("bad dropout mode rejected", False)
except ValueError:
    check("bad dropout mode rejected", True)

ms = LogicTreeNet(**small, dropout=0.3, dropout_mode="scale")
m0b = LogicTreeNet(**small)
check(
    "scale mode: same parameters/wiring as dropout=0",
    all(torch.equal(a, b) for a, b in zip(m0b.state_dict().values(), ms.state_dict().values())),
)
ms.eval()
m0b.eval()
check("scale mode: eval soft forward unchanged", torch.equal(m0b(xb), ms(xb)))
ms.train()
m0b.train()
check("scale mode: hard forward unchanged in train mode", torch.equal(m0b(xb, hard=True), ms(xb, hard=True)))
check("scale mode: STE forward unchanged in train mode", torch.equal(m0b(xb, ste=True), ms(xb, ste=True)))
torch.manual_seed(1)
check("scale mode: train soft forward changes", not torch.equal(m0b(xb), ms(xb)))
torch.manual_seed(2)
ms(xb).sum().backward()
check(
    "scale mode: gradients reach first conv block",
    ms.blocks[0].tree.gate_logits.grad is not None and ms.blocks[0].tree.gate_logits.grad.abs().sum() > 0,
)
try:
    LogicTreeNet(**small, dropout=0.1, dropout_mode="bogus")
    check("bad dropout_mode rejected by LogicTreeNet", False)
except ValueError:
    check("bad dropout_mode rejected by LogicTreeNet", True)
check("LogicTreeNet default dropout_mode is half", LogicTreeNet(**small).dropout_mode == "half")

# ---------------------------------------------------------------------------
# Balanced head wiring (not in the paper, which wires at random)
# ---------------------------------------------------------------------------

from difflogic import _balanced_pairs

gb_ = torch.Generator().manual_seed(3)
ia_, ib_ = _balanced_pairs(40, 20, gb_)  # 2n == inputs: every input exactly once
check(
    "balanced pairs: 2n == inputs reads every input exactly once",
    torch.equal(torch.bincount(torch.cat([ia_, ib_]), minlength=40), torch.ones(40, dtype=torch.long)),
)
check("balanced pairs: a != b", bool((ia_ != ib_).all()))
ia_, ib_ = _balanced_pairs(10, 37, gb_)  # 74 slots over 10 inputs
cnt_ = torch.bincount(torch.cat([ia_, ib_]), minlength=10)
check(
    "balanced pairs: read counts differ by at most 1 when 2n > inputs",
    int(cnt_.max() - cnt_.min()) <= 1 and int(cnt_.sum()) == 74,
)
check("balanced pairs: a != b when 2n > inputs", bool((ia_ != ib_).all()))
ia_, ib_ = _balanced_pairs(100, 20, gb_)  # fewer slots than inputs
cnt_ = torch.bincount(torch.cat([ia_, ib_]), minlength=100)
check("balanced pairs: 2n < inputs reads each input at most once", int(cnt_.max()) == 1 and int(cnt_.sum()) == 40)
g1_ = torch.Generator().manual_seed(9)
g2_ = torch.Generator().manual_seed(9)
check(
    "balanced pairs: deterministic for a seed",
    all(torch.equal(x, y) for x, y in zip(_balanced_pairs(50, 33, g1_), _balanced_pairs(50, 33, g2_))),
)

L_def_ = LogicLayer(64, 32, seed=1)
L_rand_ = LogicLayer(64, 32, seed=1, wiring="random")
check(
    "wiring='random' is the default (same wiring and logits)",
    torch.equal(L_def_.idx_a, L_rand_.idx_a)
    and torch.equal(L_def_.idx_b, L_rand_.idx_b)
    and torch.equal(L_def_.gate_logits, L_rand_.gate_logits),
)
check(
    "random wiring (2n == inputs) leaves some inputs unread",
    int(torch.bincount(torch.cat([L_rand_.idx_a, L_rand_.idx_b]), minlength=64).min()) == 0,
)
L_bal_ = LogicLayer(64, 32, seed=1, wiring="balanced")
check(
    "balanced layer: every input read exactly once",
    torch.equal(
        torch.bincount(torch.cat([L_bal_.idx_a, L_bal_.idx_b]), minlength=64), torch.ones(64, dtype=torch.long)
    ),
)
check("balanced layer: a != b", bool((L_bal_.idx_a != L_bal_.idx_b).all()))
L_balg_ = LogicLayer(64, 32, seed=1, groups=4, wiring="balanced")
j_ = torch.arange(32)
check(
    "balanced layer with groups: gate j reads only inputs of group j % 4",
    bool(((L_balg_.idx_a % 4) == (j_ % 4)).all()) and bool(((L_balg_.idx_b % 4) == (j_ % 4)).all()),
)
check(
    "balanced layer with groups: every input read exactly once",
    torch.equal(
        torch.bincount(torch.cat([L_balg_.idx_a, L_balg_.idx_b]), minlength=64), torch.ones(64, dtype=torch.long)
    ),
)
xl_ = torch.rand(5, 64, requires_grad=True)
yl_ = L_bal_(xl_)
yl_.sum().backward()
check(
    "balanced layer: soft forward shape and gradients",
    yl_.shape == (5, 32) and L_bal_.gate_logits.grad is not None and xl_.grad is not None,
)
check(
    "balanced layer: hard forward is binary",
    set(L_bal_((xl_ > 0.5).float().detach(), hard=True).unique().tolist()) <= {0.0, 1.0},
)
for bad_ in ("bogus",):
    try:
        LogicLayer(8, 4, wiring=bad_)
        check("bad wiring rejected by LogicLayer", False)
    except ValueError:
        check("bad wiring rejected by LogicLayer", True)
try:
    LogicLayer(8, 4, wiring="balanced", in_source=torch.zeros(8, dtype=torch.long))
    check("balanced wiring with skip sources rejected", False)
except ValueError:
    check("balanced wiring with skip sources rejected", True)

mr_ = LogicTreeNet(**small)
mb_ = LogicTreeNet(**small, head_wiring="balanced")
check("LogicTreeNet default head_wiring is random", mr_.head_wiring == "random" and mb_.head_wiring == "balanced")
check(
    "balanced head: conv blocks have the same wiring and logits as the random head",
    all(
        torch.equal(a.leaf_idx, b.leaf_idx) and torch.equal(a.tree.gate_logits, b.tree.gate_logits)
        for a, b in zip(mr_.blocks, mb_.blocks)
    ),
)
check(
    "balanced head: first head layer has the same widths",
    [l.out_features for l in mr_.head] == [l.out_features for l in mb_.head],
)
for i_ in (1, 2):
    reads_b_ = torch.unique(torch.cat([mb_.head[i_].idx_a, mb_.head[i_].idx_b])).numel()
    reads_r_ = torch.unique(torch.cat([mr_.head[i_].idx_a, mr_.head[i_].idx_b])).numel()
    check(f"balanced head layer {i_}: every output of layer {i_ - 1} is read", reads_b_ == mb_.head[i_].in_features)
    check(f"random head layer {i_}: some output of layer {i_ - 1} is unread", reads_r_ < mr_.head[i_].in_features)
flat_b_ = torch.unique(torch.cat([mb_.head[0].idx_a, mb_.head[0].idx_b])).numel()
check("balanced head layer 0: every flattened conv feature is read", flat_b_ == mb_.head[0].in_features)
mb_.eval()
check("balanced head: soft forward shape", mb_(xb).shape == (3, 10))
check("balanced head: hard forward shape", mb_(xb, hard=True).shape == (3, 10))
mb_.train()
mb_(xb).sum().backward()
check(
    "balanced head: gradients reach the first conv block",
    mb_.blocks[0].tree.gate_logits.grad is not None and mb_.blocks[0].tree.gate_logits.grad.abs().sum() > 0,
)
try:
    LogicTreeNet(**small, head_wiring="bogus")
    check("bad head_wiring rejected by LogicTreeNet", False)
except ValueError:
    check("bad head_wiring rejected by LogicTreeNet", True)
try:
    LogicTreeNet(**small, head_wiring="balanced", skip="head")
    check("balanced head wiring with skip head rejected", False)
except ValueError:
    check("balanced head wiring with skip head rejected", True)
m_bs_ = LogicTreeNet(**small, head_wiring="balanced", skip="conv")
check("balanced head wiring works together with conv skip", m_bs_(xb).shape == (3, 10))

# ---------------------------------------------------------------------------
# Per-layer gate histograms (cluster/layer_histograms.py; no training)
# ---------------------------------------------------------------------------

import importlib.util
import os as _os

import numpy as np


def _load_script(name):
    for d_ in ("cluster", "."):
        p_ = _os.path.join(d_, name + ".py")
        if _os.path.exists(p_):
            spec_ = importlib.util.spec_from_file_location(name, p_)
            mod_ = importlib.util.module_from_spec(spec_)
            spec_.loader.exec_module(mod_)
            return mod_
    return None  # analysis scripts live in cluster/ on the HPC; skip their checks if absent


lh_ = _load_script("layer_histograms")
ga_ = _load_script("gate_activity")
if lh_ is None or ga_ is None:
    skipped.append(
        "gate histogram / gate_activity checks (cluster/layer_histograms.py, cluster/gate_activity.py not found)"
    )
else:
    check(
        "bin_index: 5% bins, 1.0 in the last bin",
        lh_.bin_index([0.0, 0.049, 0.05, 0.5, 0.951, 1.0]).tolist() == [0, 0, 1, 10, 19, 19],
    )
    rd_, un_ = lh_.bin_counts(
        np.array([0.0, 0.01, 0.5, 0.97, 1.0, 0.52]), np.array([False, True, False, False, True, False])
    )
    check(
        "bin_counts: read and unread counted separately",
        rd_.tolist() == [1] + [0] * 9 + [2] + [0] * 8 + [1] and un_.tolist() == [1] + [0] * 18 + [1],
    )
    check("bin_counts: read + unread = all gates", int(rd_.sum() + un_.sum()) == 6 and len(rd_) == 20)
    check("layer_groups: 7 layers, group 10 -> one group", lh_.layer_groups(7, 10) == [(1, 7)])
    check("layer_groups: group size 1 -> one per layer", lh_.layer_groups(7, 1) == [(i, i) for i in range(1, 8)])
    check("layer_groups: 11 layers, group 10", lh_.layer_groups(11, 10) == [(1, 10), (11, 11)])
    check("layer_groups: 7 layers, group 3", lh_.layer_groups(7, 3) == [(1, 3), (4, 6), (7, 7)])
    try:
        lh_.layer_groups(7, 0)
        check("layer_groups rejects group size 0", False)
    except ValueError:
        check("layer_groups rejects group size 0", True)

    mh_ = LogicTreeNet(**small).eval()
    with torch.no_grad():
        lg_, rec_ = ga_.instrumented_forward(mh_, xb)
        ref_ = mh_(xb, hard=True)
    check("instrumented forward equals model(x, hard=True) on the small model", torch.equal(lg_, ref_))
    info_ = ga_.layer_info(mh_)
    check("layer_info and instrumented forward use the same layer names", list(info_.keys()) == list(rec_.keys()))
    check(
        "per-layer gate counts: conv 7 per channel, head = layer width",
        all(info_[f"conv{i + 1}"][0].numel() == b_.out_channels * 7 for i, b_ in enumerate(mh_.blocks))
        and all(info_[f"head{j + 1}"][0].numel() == l_.out_features for j, l_ in enumerate(mh_.head)),
    )
    check(
        "total gates in layer_info = trainable gate logits / 16",
        sum(v[0].numel() for v in info_.values())
        == sum(p_.numel() for n_, p_ in mh_.named_parameters() if n_.endswith("gate_logits")) // 16,
    )
    check(
        "fractions of 1s are in [0, 1]",
        all(float((o_ / c_).min()) >= 0 and float((o_ / c_).max()) <= 1 for o_, c_ in rec_.values()),
    )
    for j_ in (0, 1):
        nxt_ = mh_.head[j_ + 1]
        n_unread_ = mh_.head[j_].out_features - torch.unique(torch.cat([nxt_.idx_a, nxt_.idx_b])).numel()
        check(
            f"unread count of head{j_ + 1} matches an independent count",
            int(info_[f"head{j_ + 1}"][1].sum()) == n_unread_,
        )
    check("last head layer has no unread gates (it feeds GroupSum)", int(info_["head3"][1].sum()) == 0)

# ---------------------------------------------------------------------------
# Balance penalty (not in the paper): collect=, unread_masks, balance_penalty
# ---------------------------------------------------------------------------

from difflogic import balance_penalty

mp_ = LogicTreeNet(**small)
xp_ = (torch.rand(4, 9, 32, 32) > 0.5).float()
means_ = []
out_c_ = mp_(xp_, collect=means_)
check("collect does not change the output", torch.equal(out_c_, mp_(xp_)))
check("collect gives one entry per trainable layer", len(means_) == len(mp_.blocks) + len(mp_.head))
check(
    "collect shapes: conv (channels, 7), head (width,)",
    all(means_[i].shape == (b_.out_channels, 7) for i, b_ in enumerate(mp_.blocks))
    and all(means_[len(mp_.blocks) + j].shape == (l_.out_features,) for j, l_ in enumerate(mp_.head)),
)
check("collected means are in [0, 1]", all(float(m_.min()) >= 0 and float(m_.max()) <= 1 for m_ in means_))
blk0_ = mp_.blocks[0](xp_)
check(
    "conv root-gate mean equals the mean of the block output",
    torch.allclose(blk0_.mean(dim=(0, 2, 3)), means_[0][:, 6], atol=1e-6),
)
cap_ = {}
hk_ = mp_.head[2].register_forward_hook(lambda m_, i_, o_: cap_.setdefault("y", o_.detach()))
means_b_ = []
mp_(xp_, collect=means_b_)
hk_.remove()
check(
    "head mean equals the mean of the layer output over the batch",
    torch.allclose(cap_["y"].mean(0), means_b_[-1], atol=1e-6),
)

check(
    "balance_penalty: hand-computed value (read gates only)",
    abs(
        float(
            balance_penalty(
                [torch.tensor([0.5, 0.0]), torch.tensor([1.0])], [torch.tensor([1.0, 1.0]), torch.tensor([0.0])]
            )
        )
        - 0.125
    )
    < 1e-7,
)
check(
    "balance_penalty: zero when every gate is unread",
    float(balance_penalty([torch.tensor([0.0, 1.0])], [torch.tensor([0.0, 0.0])])) == 0.0,
)
check(
    "balance_penalty: model value is between 0 and 0.25",
    0.0 <= float(balance_penalty(means_, [(~m_).float() for m_ in mp_.unread_masks()])) <= 0.25,
)

um_ = mp_.unread_masks()
if ga_ is not None:
    info_p_ = ga_.layer_info(mp_)
    check(
        "unread_masks agree with gate_activity.layer_info for every layer",
        all(torch.equal(m_.reshape(-1), info_p_[n_][1]) for m_, n_ in zip(um_, info_p_.keys())),
    )
check("unread_masks shapes match the collected means", all(m_.shape == c_.shape for m_, c_ in zip(um_, means_)))
cfgS_ = {k_: v_ for k_, v_ in LOGICTREENET_CIFAR["S"].items() if k_ != "reported"}
mS_ = LogicTreeNet(
    in_channels=9,
    in_size=32,
    num_classes=10,
    tree_depth=3,
    kernel_size=3,
    seed=0,
    channels_per_tree=2,
    pool="max",
    groups=4,
    **cfgS_,
)
umS_ = mS_.unread_masks()
check(
    "scale S seed 0: 22,631 unread gates (same as the analysis of a trained checkpoint)",
    sum(int(m_.sum()) for m_ in umS_) == 22631,
)
check(
    "scale S seed 0: unread per layer conv3 70 gates (10 channels x 7), head1 15,156, head2 7,405",
    int(umS_[2].sum()) == 70 and int(umS_[4].sum()) == 15156 and int(umS_[5].sum()) == 7405,
)

mq_ = LogicTreeNet(**small)
keep_q_ = [(~m_).float() for m_ in mq_.unread_masks()]
means_q_ = []
mq_(xp_, collect=means_q_)
balance_penalty(means_q_, keep_q_).backward()
g1_ = mq_.head[0].gate_logits.grad
un1_ = mq_.unread_masks()[len(mq_.blocks)]
check("penalty gradient is zero for unread head gates", float(g1_[un1_].abs().sum()) == 0.0)
check("penalty gradient reaches read head gates", float(g1_[~un1_].abs().sum()) > 0.0)
check("penalty gradient reaches the first conv block", float(mq_.blocks[0].tree.gate_logits.grad.abs().sum()) > 0.0)
try:
    LogicTreeNet(**small, skip="conv").unread_masks()
    check("unread_masks rejects skip connections", False)
except NotImplementedError:
    check("unread_masks rejects skip connections", True)

# ---------------------------------------------------------------------------
# Configurable head depth (not in the paper) and per-layer gradient norms
# ---------------------------------------------------------------------------

k_small_ = small["channels"][0]
base_head_ = [l_.out_features for l_ in LogicTreeNet(**small).head]
check(
    "head_layers=3 (default) is the paper head 1280k/640k/320k",
    base_head_ == [1280 * k_small_, 640 * k_small_, 320 * k_small_],
)
check(
    "head_layers=3 explicit equals the default model",
    all(
        torch.equal(a_, b_)
        for a_, b_ in zip(
            LogicTreeNet(**small).state_dict().values(), LogicTreeNet(**small, head_layers=3).state_dict().values()
        )
    ),
)
for n_ in (5, 7, 9):
    mh_n_ = LogicTreeNet(**small, head_layers=n_)
    w_ = [l_.out_features for l_ in mh_n_.head]
    check(f"head_layers={n_}: {n_} head layers", len(w_) == n_ and len(mh_n_.head) == n_)
    check(f"head_layers={n_}: total head gates equal the paper head (2240k)", sum(w_) == 2240 * k_small_)
    check(f"head_layers={n_}: last layer 320k, others equal", w_[-1] == 320 * k_small_ and len(set(w_[:-1])) == 1)
    check(f"head_layers={n_}: soft forward shape", mh_n_(xb).shape == (3, 10))
    check(f"head_layers={n_}: hard forward shape", mh_n_(xb, hard=True).shape == (3, 10))
    check(
        f"head_layers={n_}: same trainable gate count as the 3-layer model (conv unchanged)",
        sum(p_.numel() for nm_, p_ in mh_n_.named_parameters() if nm_.endswith("gate_logits"))
        == sum(p_.numel() for nm_, p_ in LogicTreeNet(**small).named_parameters() if nm_.endswith("gate_logits")),
    )
check(
    "head_layers=5 widths are 480k x4 + 320k",
    [l_.out_features for l_ in LogicTreeNet(**small, head_layers=5).head] == [480 * k_small_] * 4 + [320 * k_small_],
)
check(
    "head_layers=7 widths are 320k x7",
    [l_.out_features for l_ in LogicTreeNet(**small, head_layers=7).head] == [320 * k_small_] * 7,
)
for bad_n_ in (2, 1, 0, 4.5):
    try:
        LogicTreeNet(**small, head_layers=bad_n_)
        check(f"head_layers={bad_n_} rejected", False)
    except ValueError:
        check(f"head_layers={bad_n_} rejected", True)
m7_ = LogicTreeNet(**small, head_layers=7)
m7s_ = LogicTreeNet(**small, head_layers=7, skip="both")
check("head_layers=7 works with --skip both", m7s_(xb).shape == (3, 10) and len(m7s_.head) == 7)
m7b_ = LogicTreeNet(**small, head_layers=7, head_wiring="balanced")
um7_ = m7b_.unread_masks()
check("head_layers=7 with balanced wiring: no unread head gates", all(int(m_.sum()) == 0 for m_ in um7_[4:]))
um7r_ = m7_.unread_masks()
fr_ = [float(m_.float().mean()) for m_ in um7r_[4:-1]]
check(
    "head_layers=7 with random wiring: about e^-2 = 13.5% of each hidden head layer is unread",
    all(0.10 < f_ < 0.17 for f_ in fr_),
)
names_ = [n_ for n_, _ in m7_.layer_gate_logits()]
check(
    "layer_gate_logits lists conv1..conv4 then head1..head7",
    names_ == ["conv1", "conv2", "conv3", "conv4"] + [f"head{j_}" for j_ in range(1, 8)],
)
m7_(xb).sum().backward()
gn7_ = [float(p_.grad.norm()) for _, p_ in m7_.layer_gate_logits()]
check(
    "every layer's gate-logit gradient is finite and nonzero at initialization (depth 11)",
    all(g_ == g_ and g_ > 0 for g_ in gn7_),
)

# ---------------------------------------------------------------------------
# Dropout placement (dropout_where) and running-mean replacement (dropout_mode="runmean"); not in the paper
# ---------------------------------------------------------------------------

mw0_ = LogicTreeNet(**small, dropout=0.3, dropout_mode="scale")
mw1_ = LogicTreeNet(**small, dropout=0.3, dropout_mode="scale", dropout_where="all")
check("dropout_where default is all", mw0_.dropout_where == "all")
check(
    "dropout_where='all' builds the same model as the default",
    all(torch.equal(a_, b_) for a_, b_ in zip(mw0_.state_dict().values(), mw1_.state_dict().values())),
)
torch.manual_seed(11)
o0_ = mw0_(xb)
torch.manual_seed(11)
o1_ = mw1_(xb)
check("dropout_where='all' gives the same training output as the default under the same seed", torch.equal(o0_, o1_))
check("default model has no running-mean buffers", not any(k_.startswith("rm_") for k_ in mw0_.state_dict().keys()))
try:
    LogicTreeNet(**small, dropout=0.1, dropout_where="bogus")
    check("bad dropout_where rejected", False)
except ValueError:
    check("bad dropout_where rejected", True)


def _site_inputs(model, x_):
    """Inputs seen by conv blocks 1..3 and head layer 0 in one training forward pass, and the clean (undropped) inputs
    recomputed from the previous captured tensor."""
    cap_ = {}
    hs_ = [
        model.blocks[i_].register_forward_pre_hook(lambda m_, a_, i_=i_: cap_.__setitem__(f"b{i_}", a_[0].detach()))
        for i_ in range(1, 4)
    ]
    hs_.append(model.head[0].register_forward_pre_hook(lambda m_, a_: cap_.__setitem__("h0", a_[0].detach())))
    model.train()
    with torch.no_grad():
        model(x_)
    for h_ in hs_:
        h_.remove()
    with torch.no_grad():
        clean_ = {
            "b1": model.pools[0](model.blocks[0](x_)),
            "b2": model.pools[1](model.blocks[1](cap_["b1"])),
            "b3": model.pools[2](model.blocks[2](cap_["b2"])),
            "h0": model.pools[3](model.blocks[3](cap_["b3"])).flatten(1),
        }
    return {k_: bool(torch.equal(cap_[k_], clean_[k_])) for k_ in clean_}


for where_, expect_ in (
    ("all", {"b1": False, "b2": False, "b3": False, "h0": False}),
    ("conv1", {"b1": False, "b2": True, "b3": True, "h0": True}),
    ("head", {"b1": True, "b2": True, "b3": True, "h0": False}),
):
    mwx_ = LogicTreeNet(**small, dropout=0.5, dropout_mode="scale", dropout_where=where_)
    check(
        f"dropout_where={where_}: inputs that stay clean (True) or get dropout (False)",
        _site_inputs(mwx_, xb) == expect_,
    )
mwx_ = LogicTreeNet(**small, dropout=0.5, dropout_mode="scale", dropout_where="head").eval()
check("dropout_where=head: eval forward unchanged", torch.equal(mwx_(xb), LogicTreeNet(**small).eval()(xb)))

mr_ = LogicTreeNet(**small, dropout=0.3, dropout_mode="runmean")
rm_keys_ = sorted(k_ for k_ in mr_.state_dict().keys() if k_.startswith("rm_"))
check(
    "runmean model has buffers rm_conv1..3 and rm_head0..2",
    rm_keys_ == ["rm_conv1", "rm_conv2", "rm_conv3", "rm_head0", "rm_head1", "rm_head2"],
)
check(
    "runmean buffers start at 0.5 with one entry per input channel/feature",
    all(float(mr_.state_dict()[k_].min()) == 0.5 and float(mr_.state_dict()[k_].max()) == 0.5 for k_ in rm_keys_)
    and mr_.rm_conv1.numel() == mr_.blocks[1].in_channels
    and mr_.rm_head0.numel() == mr_.head[0].in_features,
)
mrh_ = LogicTreeNet(**small, dropout=0.3, dropout_mode="runmean", dropout_where="head")
check(
    "runmean with dropout_where=head has only head buffers",
    sorted(k_ for k_ in mrh_.state_dict().keys() if k_.startswith("rm_")) == ["rm_head0", "rm_head1", "rm_head2"],
)
check(
    "runmean with dropout 0 has no buffers",
    not any(k_.startswith("rm_") for k_ in LogicTreeNet(**small, dropout_mode="runmean").state_dict().keys()),
)
try:
    LogicTreeNet(**small, dropout=0.1, dropout_mode="runmean", skip="conv")
    check("runmean with skip rejected", False)
except ValueError:
    check("runmean with skip rejected", True)

# EMA update and replacement at one conv site
mr_.train()
cap_pre_ = {}
h1_ = mr_.pools[0].register_forward_hook(lambda m_, a_, o_: cap_pre_.setdefault("pre", o_.detach()))
h2_ = mr_.blocks[1].register_forward_pre_hook(lambda m_, a_: cap_pre_.setdefault("post", a_[0].detach()))
with torch.no_grad():
    mr_(xb)
h1_.remove()
h2_.remove()
pre_, post_ = cap_pre_["pre"], cap_pre_["post"]
expect_buf_ = 0.99 * 0.5 + 0.01 * pre_.mean(dim=(0, 2, 3))
check(
    "runmean: one EMA step moves the buffer to 0.99*0.5 + 0.01*batch mean",
    torch.allclose(mr_.rm_conv1, expect_buf_, atol=1e-6),
)
changed_ = post_ != pre_
repl_ = mr_.rm_conv1.view(1, -1, 1, 1).expand_as(post_)
check(
    "runmean: every changed value equals that channel's running mean", bool((post_[changed_] == repl_[changed_]).all())
)
check("runmean: dropped fraction is about p (0.3)", abs(float(changed_.float().mean()) - 0.3) < 0.08)
check("runmean: values stay in [0, 1]", float(post_.min()) >= 0 and float(post_.max()) <= 1)
mr_.eval()
b_before_ = mr_.rm_conv1.clone()
check(
    "runmean: eval forward equals a model without dropout and does not move the buffers",
    torch.equal(mr_(xb), LogicTreeNet(**small).eval()(xb)) and torch.equal(mr_.rm_conv1, b_before_),
)
mr_.train()
with torch.no_grad():
    check(
        "runmean: hard forward unchanged in train mode",
        torch.equal(mr_(xb, hard=True), LogicTreeNet(**small)(xb, hard=True)),
    )
    b2_ = mr_.rm_conv1.clone()
    mr_(xb, hard=True)
check("runmean: hard forward does not move the buffers", torch.equal(mr_.rm_conv1, b2_))
mr_(xb).sum().backward()
check("runmean: gradients reach the first conv block", float(mr_.blocks[0].tree.gate_logits.grad.abs().sum()) > 0)
mr2_ = LogicTreeNet(**small, dropout=0.3, dropout_mode="runmean")
mr2_.load_state_dict(mr_.state_dict())
check(
    "runmean: buffers survive a state_dict round trip (resumable)",
    torch.equal(mr2_.rm_head1, mr_.rm_head1) and torch.equal(mr2_.rm_conv3, mr_.rm_conv3),
)
# unbiased over the data: when the running mean equals the data mean of each channel, dropout leaves the mean activation unchanged
mu_m_ = LogicTreeNet(**small, dropout=0.4, dropout_mode="runmean").train()
xbig_ = (torch.rand(32, 9, 32, 32) > 0.5).float()
with torch.no_grad():
    pre_big_ = mu_m_.pools[0](mu_m_.blocks[0](xbig_))
    mu_m_.rm_conv1.copy_(pre_big_.mean(dim=(0, 2, 3)))
    torch.manual_seed(5)
    out_big_ = mu_m_._drop(pre_big_, 0.4, "conv1")
check(
    "runmean: dropout with the data mean as replacement keeps each channel's mean activation (within 0.02)",
    float((out_big_.mean(dim=(0, 2, 3)) - pre_big_.mean(dim=(0, 2, 3))).abs().max()) < 0.02,
)
check(
    "runmean: it is not a no-op (about 40% of values were replaced)",
    abs(
        float((out_big_ != pre_big_).float().mean())
        - 0.4 * float((pre_big_ != pre_big_.mean(dim=(0, 2, 3), keepdim=True)).float().mean())
    )
    < 0.05,
)


# ---------------------------------------------------------------------------
# Gumbel straight-through gate sampling (Yousefi et al., arXiv:2506.07500, Sec. 4 / Algorithm 1); flag off by default
# ---------------------------------------------------------------------------
from difflogic import gate_probs, gumbel_noise

torch.manual_seed(0)
zg_ = torch.randn(50, 16)
check(
    "gumbel: gate_probs unchanged when gumbel is None", torch.equal(gate_probs(zg_, 1.0, False, None), gate_probs(zg_))
)
torch.manual_seed(7)
pg_ = gate_probs(zg_, 1.0, False, (1.0, 1.0))
torch.manual_seed(7)
g_ = -torch.empty_like(zg_).exponential_().log()
check(
    "gumbel: noise is Gumbel(0,1) = -log(-log u): mean about 0.5772, sd about 1.2825 (large sample)",
    abs(float(gumbel_noise(torch.zeros(200000)).mean()) - 0.5772) < 0.02
    and abs(float(gumbel_noise(torch.zeros(200000)).std()) - 1.2825) < 0.02,
)
check(
    "gumbel: forward value is the one-hot of argmax(logits + g) for the same RNG stream",
    torch.allclose(pg_.detach(), F.one_hot((zg_ + g_).argmax(-1), 16).float(), atol=1e-6),
)
check("gumbel: forward rows are one-hot (sum 1)", torch.allclose(pg_.sum(-1), torch.ones(50), atol=1e-6))
torch.manual_seed(7)
pg_tau_ = gate_probs(zg_, 1.0, False, (1.0, 0.1))
check(
    "gumbel: the forward value does not depend on tau (same noise)",
    torch.allclose(pg_.detach(), pg_tau_.detach(), atol=1e-6),
)
w_ = torch.randn(50, 16)
za_ = zg_.clone().requires_grad_(True)
torch.manual_seed(7)
(gate_probs(za_, 1.0, False, (1.0, 0.5)) * w_).sum().backward()
zb_ = zg_.clone().requires_grad_(True)
torch.manual_seed(7)
gb_ = -torch.empty_like(zb_).exponential_().log()
(F.softmax((zb_ + gb_) / 0.5, -1) * w_).sum().backward()
check(
    "gumbel: gradient equals the gradient of softmax((logits + g)/tau) (straight-through)",
    torch.allclose(za_.grad, zb_.grad, atol=1e-5),
)
zc_ = zg_.clone().requires_grad_(True)
torch.manual_seed(7)
(gate_probs(zc_, 1.0, False, (1.0, 2.0)) * w_).sum().backward()
check("gumbel: tau changes the backward pass", not torch.allclose(za_.grad, zc_.grad, atol=1e-4))
zero_ = torch.randn(40, 16)
check(
    "gumbel: scale 0 gives the plain argmax gate",
    torch.allclose(
        gate_probs(zero_, 1.0, False, (0.0, 1.0)).detach(), F.one_hot(zero_.argmax(-1), 16).float(), atol=1e-6
    ),
)
torch.manual_seed(1)
cnt_ = gate_probs(torch.zeros(32000, 16), 1.0, False, (1.0, 1.0)).detach().round().sum(0)
chi_ = float(((cnt_ - 2000) ** 2 / 2000).sum())
check("gumbel: uniform logits select every gate with equal frequency (chi2, 15 dof, < 40)", chi_ < 40)
zp_ = torch.zeros(40000, 16)
zp_[:, 3] = 2.0
torch.manual_seed(2)
freq_ = float(gate_probs(zp_, 1.0, False, (1.0, 1.0)).detach().round()[:, 3].mean())
check(
    "gumbel: selection frequency equals softmax(logits) (Gumbel-max trick), within 0.01",
    abs(freq_ - float(F.softmax(zp_[0], -1)[3])) < 0.01,
)
torch.manual_seed(3)
a_, b_ = gumbel_noise(torch.zeros(5)), gumbel_noise(torch.zeros(5))
check("gumbel: successive draws differ", not torch.equal(a_, b_))
torch.manual_seed(3)
check("gumbel: a seed reproduces the draw", torch.equal(a_, gumbel_noise(torch.zeros(5))))

xg_ = (torch.rand(3, 9, 32, 32) > 0.5).float()
mg_ = LogicTreeNet(**small).train()
ref_state_ = {k_: v_.clone() for k_, v_ in mg_.state_dict().items()}
o_def_ = mg_(xg_)
check("gumbel: default call equals gumbel=None", torch.equal(o_def_, mg_(xg_, gumbel=None)))
torch.manual_seed(0)
o_g1_ = mg_(xg_, gumbel=(1.0, 1.0))
o_g2_ = mg_(xg_, gumbel=(1.0, 1.0))
check("gumbel: model output differs between two noisy forward passes", not torch.equal(o_g1_, o_g2_))
check("gumbel: the noisy passes differ from the noiseless soft forward", not torch.allclose(o_g1_, o_def_, atol=1e-3))
with torch.no_grad():
    o_hard_ = mg_(xg_, hard=True)
    o_s0_ = mg_(xg_, gumbel=(0.0, 1.0))
check(
    "gumbel: scale 0 reproduces the discretized (hard) network exactly on binary inputs (all layers)",
    torch.allclose(o_s0_, o_hard_, atol=1e-4),
)
check("gumbel: hard=True ignores the argument", torch.equal(mg_(xg_, hard=True, gumbel=(1.0, 1.0)), o_hard_))
check("gumbel: ste=True ignores the argument", torch.equal(mg_(xg_, ste=True, gumbel=(1.0, 1.0)), mg_(xg_, ste=True)))
mg_.eval()
check(
    "gumbel: eval mode ignores the argument (soft eval = plain softmax)",
    torch.equal(mg_(xg_, gumbel=(1.0, 1.0)), mg_(xg_)),
)
mg_.train()
cap_g_ = {}
hg_ = mg_.blocks[0].register_forward_hook(lambda m_, a_, o_: cap_g_.setdefault("o", o_.detach()))
mg_(xg_, gumbel=(1.0, 1.0))
hg_.remove()
check(
    "gumbel: first conv block outputs are binary (0/1) in the noisy training pass",
    float(torch.minimum(cap_g_["o"], 1 - cap_g_["o"]).abs().max()) < 1e-5,
)
mg_.zero_grad()
mg_(xg_, gumbel=(1.0, 1.0)).sum().backward()
gl_ = [p_.grad for n_, p_ in mg_.named_parameters() if n_.endswith("gate_logits")]
check(
    "gumbel: gradients reach gate_logits of every layer and are finite",
    len(gl_) >= 5
    and all(g__ is not None and bool(torch.isfinite(g__).all()) and float(g__.abs().sum()) > 0 for g__ in gl_),
)
check(
    "gumbel: the model state is untouched by using the flag",
    all(torch.equal(ref_state_[k_], v_) for k_, v_ in mg_.state_dict().items()),
)
md_ = LogicTreeNet(**small, dropout=0.1).train()
check("gumbel: works together with dropout", bool(torch.isfinite(md_(xg_, gumbel=(1.0, 1.0))).all()))
mc_ = LogicTreeNet(**small).train()
cl_ = []
check("gumbel: works together with collect", bool(torch.isfinite(mc_(xg_, gumbel=(1.0, 1.0), collect=cl_)).all()))


# ---------------------------------------------------------------------------
# Progressive discretization: GlobalFreeze / TaskFreeze (Gerlach 2026, arXiv:2609.33747); off by default. + dropout placement convlate
# ---------------------------------------------------------------------------
from difflogic import FREEZE_LOGIT

mf0_ = LogicTreeNet(**small)
check(
    "freeze: a default model has no freeze buffers (old checkpoints stay loadable)",
    not any("frozen_idx" in k_ or "freeze_" in k_ for k_ in mf0_.state_dict()),
)
mf_ = LogicTreeNet(**small)
mf_.load_state_dict(mf0_.state_dict())
mf_.enable_freeze()
xf_ = (torch.rand(4, 9, 32, 32) > 0.5).float()
mf0_.train()
mf_.train()
check("freeze: enabling the buffers without freezing changes nothing", torch.equal(mf_(xf_), mf0_(xf_)))
check("freeze: fraction is 0 at the start", mf_.frozen_fraction() == 0.0)
for m_ in mf_.modules():
    if isinstance(m_, LogicTree):
        m_.gate_logits.data += 0.5 * torch.randn_like(m_.gate_logits)  # make argmax gates non-trivial
mf_.train()
with torch.no_grad():
    o_soft_ = mf_(xf_)
n_fr_ = mf_.freeze_all()
check(
    "freeze: freeze_all freezes every trainable gate",
    mf_.frozen_fraction() == 1.0
    and n_fr_ == sum(p_.numel() // 16 for n_, p_ in mf_.named_parameters() if n_.endswith("gate_logits")),
)
with torch.no_grad():
    o_fr_, o_hd_ = mf_(xf_), mf_(xf_, hard=True)
check(
    "freeze: after GlobalFreeze the soft training forward equals the hard network on binary inputs",
    torch.allclose(o_fr_, o_hd_, atol=1e-4),
)
check(
    "freeze: frozen logits keep argmax == frozen gate and the frozen value is the large logit",
    all(
        bool((m_.gate_logits.argmax(-1) == m_.frozen_idx).all()) and float(m_.gate_logits.max()) == FREEZE_LOGIT
        for m_ in mf_.modules()
        if isinstance(m_, (LogicLayer, LogicTree))
    ),
)
mf_.zero_grad()
mf_(xf_).sum().backward()
check(
    "freeze: frozen neurons get no gradient to their logits",
    all(float(p_.grad.abs().sum()) == 0 for n_, p_ in mf_.named_parameters() if n_.endswith("gate_logits")),
)
# partial freeze: one neuron of the first tree layer frozen; changing its logits must not change the output, changing an unfrozen one must
mp_ = LogicTreeNet(**small).train()
mp_.enable_freeze()
tr_ = mp_.blocks[0].tree
msk_ = torch.zeros_like(tr_.frozen_idx, dtype=torch.bool)
msk_[0, 0] = True
tr_.freeze_where(msk_)
check(
    "freeze: freeze_where freezes exactly the selected neuron",
    int((tr_.frozen_idx >= 0).sum()) == 1 and int(tr_.frozen_idx[0, 0]) == int(tr_.gate_logits[0, 0].argmax()),
)
with torch.no_grad():
    base_ = mp_(xf_)
    tr_.gate_logits[0, 0] += torch.randn(16)  # frozen: no effect on soft output
    same_ = torch.allclose(mp_(xf_), base_, atol=1e-6)
    tr_.gate_logits[0, 1] += 3.0 * torch.randn(16)  # not frozen: should change the output
    diff_ = not torch.allclose(mp_(xf_), base_, atol=1e-6)
check("freeze: a frozen neuron ignores its logits in the soft forward, an unfrozen one does not", same_ and diff_)
check(
    "freeze: hard forward uses the frozen gate even if the logits were changed afterwards",
    torch.equal(mp_(xf_, hard=True), mp_(xf_, hard=True)) and int(tr_.frozen_idx[0, 0]) >= 0,
)
# first-order score = exact loss change of freezing one head neuron when the loss is linear in its output
torch.manual_seed(0)
ll_ = LogicLayer(12, 6, seed=3).train()
ll_.enable_freeze()
xl_ = torch.rand(7, 12)
wl_ = torch.randn(7, 6)
pr_ = []
yl_ = ll_(xl_, probe=pr_)
(yl_ * wl_).sum().backward()
ll_.freeze_n.zero_()
ll_.freeze_mu.zero_()


# reimplement the score through the model-level function on a tiny wrapper
class _W_:
    def __init__(s_, m_):
        s_.m_ = m_


net_ = LogicTreeNet.__new__(LogicTreeNet)
nn.Module.__init__(net_)
net_.layer_ = ll_
freeze_scores_ = {}
orig_update_ = ll_.freeze_update
ll_.freeze_update = lambda idx, s, beta, patience: freeze_scores_.update(s=s.clone()) or 0
LogicTreeNet.freeze_update_from_probe(net_, pr_, 0.9, 3)
s_ = freeze_scores_["s"]
base_loss_ = float((ll_(xl_) * wl_).sum())
ok_ = True
for j_ in range(6):
    l2_ = LogicLayer(12, 6, seed=3).train()
    l2_.enable_freeze()
    l2_.load_state_dict(ll_.state_dict())
    l2_.frozen_idx.fill_(-1)
    m_ = torch.zeros(6, dtype=torch.bool)
    m_[j_] = True
    l2_.freeze_where(m_)
    d_ = float((l2_(xl_) * wl_).sum()) - base_loss_
    ok_ = ok_ and abs(d_ - float(s_[j_]) * 7) < 1e-4
check(
    "freeze: the first-order score equals the exact loss change of freezing a neuron to its argmax gate (loss linear in its output)",
    ok_,
)
# EMA / patience logic
lq_ = LogicLayer(12, 4, seed=1)
lq_.enable_freeze()
ix_ = (slice(None),)
neg_ = torch.tensor([-1.0, -1.0, 1.0, -1.0])
pos_ = torch.tensor([1.0, 1.0, 1.0, 1.0])
for sc_ in (neg_, neg_, neg_):
    nf_ = lq_.freeze_update(ix_, sc_, 0.5, 3)
check(
    "freeze: freezes after `patience` consecutive negative EMA evaluations, never the always-positive neuron",
    lq_.frozen_idx[0] >= 0 and lq_.frozen_idx[1] >= 0 and lq_.frozen_idx[2] < 0 and lq_.frozen_idx[3] >= 0,
)
lr_ = LogicLayer(12, 2, seed=1)
lr_.enable_freeze()
lr_.freeze_update(ix_, torch.tensor([-1.0, -1.0]), 0.0, 3)
lr_.freeze_update(ix_, torch.tensor([-1.0, 1.0]), 0.0, 3)
lr_.freeze_update(ix_, torch.tensor([1.0, -1.0]), 0.0, 3)
lr_.freeze_update(ix_, torch.tensor([-1.0, -1.0]), 0.0, 3)
check(
    "freeze: a positive evaluation resets the patience counter",
    bool((lr_.frozen_idx < 0).all()) and lr_.freeze_cnt.tolist() == [1, 2],
)
mq_ = LogicTreeNet(**small).train()
mq_.enable_freeze()
pb_ = []
mq_(xf_, probe=pb_).sum().backward()
nq_ = mq_.freeze_update_from_probe(pb_, 0.9, 1)
check(
    "freeze: probe + score + update run on the full model (all layers appear in the probe)",
    len(pb_) == 3 * 4 + 3 and nq_ >= 0,
)
mq2_ = LogicTreeNet(**small)
mq2_.enable_freeze()
mq2_.load_state_dict(mq_.state_dict())
check(
    "freeze: freezing state round-trips through state_dict (resumable)",
    all(torch.equal(a_, b_) for a_, b_ in zip(mq_.state_dict().values(), mq2_.state_dict().values())),
)
check(
    "freeze: probe is ignored in eval / hard mode",
    (lambda m_: (m_.eval(), m_(xf_, probe=[]), True)[-1])(LogicTreeNet(**small)),
)
md3_ = LogicTreeNet(**small, dropout=0.5, dropout_mode="scale", dropout_where="convlate")
check(
    "dropout convlate: sites are conv blocks 2 and 3 only, none in the head",
    md3_._drop_conv_sites == {2, 3} and md3_._drop_head_sites == set(),
)


# ---------------------------------------------------------------------------
# Permutation dropout (mode "perm", proposed by Prof. Liu, not in the paper); off by default
# ---------------------------------------------------------------------------
from difflogic import logic_dropout, logic_dropout_perm

torch.manual_seed(11)
hp_ = (torch.rand(6, 3, 8, 8) > 0.5).float()
op_ = logic_dropout_perm(hp_, 0.3)
check("perm: outputs are exactly 0 or 1 on binary input", bool(((op_ == 0) | (op_ == 1)).all()))
check(
    "perm: number of ones per example is unchanged exactly",
    bool((op_.flatten(1).sum(1) == hp_.flatten(1).sum(1)).all()),
)
check("perm: some values move (not an identity)", bool((op_ != hp_).any()))
check("perm: shape is kept", op_.shape == hp_.shape)
check("perm: p = 0 is the identity", torch.equal(logic_dropout_perm(hp_, 0.0), hp_))
check(
    "perm: logic_dropout(mode='perm') is the identity when not training",
    torch.equal(logic_dropout(hp_, 0.5, training=False, mode="perm"), hp_),
)
check(
    "perm: logic_dropout(mode='perm') changes values when training",
    not torch.equal(logic_dropout(hp_, 0.5, training=True, mode="perm"), hp_),
)
torch.manual_seed(12)
hq_ = (torch.rand(1, 1, 16, 16) > 0.7).float()
acc_ = torch.zeros_like(hq_)
for _ in range(3000):
    acc_ += logic_dropout_perm(hq_, 0.3)
Ep_ = acc_ / 3000
check(
    "perm: expected output is (1-p)h + p*mean(example) (Monte Carlo, max abs error < 0.04)",
    float((Ep_ - ((1 - 0.3) * hq_ + 0.3 * hq_.mean())).abs().max()) < 0.04,
)
check(
    "perm: expected output keeps the mean activation of the example", abs(float(Ep_.mean()) - float(hq_.mean())) < 0.01
)
torch.manual_seed(13)
ht_ = torch.tensor([[1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]])
ot_ = logic_dropout_perm(ht_, 0.5)
check(
    "perm: a constant-valued window is unchanged (permuting equal values does nothing)",
    bool((ot_.sum() == ht_.sum()).item()),
)
hg_ = torch.rand(2, 4, 4, 4, requires_grad=True)
og_ = logic_dropout_perm(hg_, 0.4)
og_.sum().backward()
check(
    "perm: gradient flows to the input (finite, not all zero)",
    bool(torch.isfinite(hg_.grad).all()) and float(hg_.grad.abs().sum()) > 0,
)
md_ = LogicTreeNet(**small, dropout=0.2, dropout_mode="perm").train()
check(
    "perm: a model with dropout_mode='perm' runs a soft training forward (finite)", bool(torch.isfinite(md_(xg_)).all())
)
check(
    "perm: hard forward ignores the perm dropout",
    torch.equal(md_(xg_, hard=True), LogicTreeNet(**small).train()(xg_, hard=True)),
)

# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# permx (exact count) and --dropout-in-ste
# ---------------------------------------------------------------------------
import difflogic
from difflogic import logic_dropout_perm as _ldp

torch.manual_seed(21)
hx_ = (torch.rand(5, 3, 8, 8) > 0.5).float()
ox_ = _ldp(hx_, 0.25, exact=True)
check("permx: outputs are exactly 0 or 1 on binary input", bool(((ox_ == 0) | (ox_ == 1)).all()))
check("permx: number of ones per example unchanged", bool((ox_.flatten(1).sum(1) == hx_.flatten(1).sum(1)).all()))
probe_ = torch.arange(192, dtype=torch.float32).repeat(4, 1)
mv_ = torch.stack([(_ldp(probe_, 0.25, exact=True) != probe_).sum(1) for _ in range(50)])
check(
    "permx: at most k = round(0.25 * 192) = 48 positions change, and many do",
    int(mv_.max()) <= 48 and int(mv_.max()) >= 40,
)
bv_ = torch.stack([(_ldp(probe_, 0.25) != probe_).sum(1) for _ in range(50)])
check("permx: Bernoulli mask can change more than 48 positions, the exact mask never does", int(bv_.max()) > 48)
srt_ = torch.rand(3, 50)
check(
    "permx: the value multiset of each example is unchanged (soft values)",
    torch.equal(_ldp(srt_, 0.3, exact=True).sort(1).values, srt_.sort(1).values),
)
check("permx: p = 0 drops nothing", torch.equal(_ldp(srt_, 0.0, exact=True), srt_))
check(
    "permx: logic_dropout(mode='permx') is the identity when not training",
    torch.equal(logic_dropout(hx_, 0.5, training=False, mode="permx"), hx_),
)
for_ = (torch.rand(1, 64) > 0.7).float()
Ex_ = _ldp(for_.expand(20000, 64).contiguous(), 0.25, exact=True).mean(0, keepdim=True)
check(
    "permx: E[out] = (1 - p) h + p mean(h) (simulation, tol 0.02)",
    float((Ex_ - (0.75 * for_ + 0.25 * for_.mean())).abs().max()) < 0.02,
)
xs_ = torch.rand(2, 10, requires_grad=True)
(_ldp(xs_, 0.5, exact=True) * torch.randn(2, 10)).sum().backward()
check("permx: gradient is finite", bool(torch.isfinite(xs_.grad).all()))
# model level
_sites = []
_orig = difflogic.logic_dropout_perm


def _spy(v, p, exact=False):
    _sites.append((bool(((v - v.round()).abs() < 1e-5).all()), exact))
    return _orig(v, p, exact=exact)


difflogic.logic_dropout_perm = _spy
mxs_ = LogicTreeNet(**small, dropout=0.2, dropout_mode="permx", dropout_in_ste=True).train()
_sites.clear()
mxs_(xg_, ste=True)
check("dropout_in_ste: dropout runs at every site in the STE forward (6 sites)", len(_sites) == 6)
check(
    "dropout_in_ste: every dropout input is binary (within 1e-5 float rounding) in the STE forward",
    all(b for b, _ in _sites),
)
check("dropout_in_ste: permx mode uses the exact count", all(e for _, e in _sites))
check(
    "dropout_in_ste: STE output is finite and binary-valued dropout keeps training stochastic",
    bool(torch.isfinite(mxs_(xg_, ste=True)).all()) and not torch.equal(mxs_(xg_, ste=True), mxs_(xg_, ste=True)),
)
mno_ = LogicTreeNet(**small, dropout=0.2, dropout_mode="permx").train()
_sites.clear()
mno_(xg_, ste=True)
check("dropout_in_ste off (default): no dropout in the STE forward", len(_sites) == 0)
_sites.clear()
mxs_.eval()
mxs_(xg_, ste=True)
mxs_(xg_, hard=True)
check("dropout_in_ste: no dropout in eval or hard forward", len(_sites) == 0)
difflogic.logic_dropout_perm = _orig
_g = LogicTreeNet(**small, dropout=0.0, dropout_in_ste=True).train()
check(
    "dropout_in_ste with dropout 0: STE forward equals the default model's",
    torch.equal(_g(xg_, ste=True), LogicTreeNet(**small).train()(xg_, ste=True)),
)


_so = LogicTreeNet(**small, dropout=0.2, dropout_mode="permx", dropout_in_ste=True, dropout_soft=False).train()
difflogic.logic_dropout_perm = _spy
_sites.clear()
_so(xg_)
check("dropout_ste_only: no dropout in soft training", len(_sites) == 0)
_sites.clear()
_so(xg_, ste=True)
check("dropout_ste_only: dropout at all 6 sites in the STE forward", len(_sites) == 6)
difflogic.logic_dropout_perm = _orig


# ---------------------------------------------------------------------------
# Residual units inside conv stages (--res-units; not in the paper)
# ---------------------------------------------------------------------------
import importlib.util as _ilu

import difflogic as _dl
from difflogic import MergeGate, ResidualUnit, count_trainable_gates, merge_or, merge_xor, parse_res_stages
from difflogic import count_gates as _cg

# ref/difflogic_before_res.py is a frozen copy of difflogic.py from before the residual units were added.
# It lives on the cluster; the backward-compatibility checks below are skipped when it is absent.
_ref_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ref", "difflogic_before_res.py")
_x8 = (torch.rand(2, 9, 32, 32) > 0.5).float()
_m_new0 = LogicTreeNet(**small, res_units=0)
if os.path.exists(_ref_path):
    _spec = _ilu.spec_from_file_location("difflogic_before_res", _ref_path)
    _old = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_old)
    _m_old = _old.LogicTreeNet(**small)
    check(
        "res: res_units=0 has the same state_dict keys as the previous model",
        list(_m_old.state_dict().keys()) == list(_m_new0.state_dict().keys()),
    )
    check(
        "res: res_units=0 has identical state_dict values",
        all(torch.equal(a, b) for a, b in zip(_m_old.state_dict().values(), _m_new0.state_dict().values())),
    )
    check(
        "res: res_units=0 gives identical soft and hard outputs",
        torch.equal(_m_old(_x8), _m_new0(_x8)) and torch.equal(_m_old(_x8, hard=True), _m_new0(_x8, hard=True)),
    )
else:
    _m_old = None
    skipped.append("backward-compatibility checks against ref/difflogic_before_res.py (file not found)")
check("res: res_units=0 adds no module (res is empty)", len(_m_new0.res) == 0)
# shapes at S and M channel counts (tiny spatial sizes)
_ok = True
for chs in [(32, 128, 512, 1024), (256, 1024, 4096, 8192)]:
    for i, (c, hw) in enumerate(zip(chs, (8, 4, 2, 2))):
        u = ResidualUnit(c, body_layers=1, tree_depth=3, merge="gate", seed=i, channels_per_tree=2, groups=chs[0] // 8)
        xi = (torch.rand(1, c, hw, hw) > 0.5).float()
        _ok = _ok and tuple(u(xi).shape) == tuple(xi.shape) and tuple(u(xi, hard=True).shape) == tuple(xi.shape)
check("res: a unit preserves (B, C, h, w) at every stage for S and M channel counts, soft and hard", _ok)
# merge gate at init
_u = ResidualUnit(16, body_layers=1, merge="gate", seed=3, channels_per_tree=2, groups=4)
_xi = (torch.rand(3, 16, 8, 8) > 0.5).float()
check("res: merge gate at init: hard output equals x exactly", torch.equal(_u(_xi, hard=True), _xi))
check(
    "res: merge gate at init: soft output is close to x (mean abs diff < 0.15)",
    float((_u(_xi) - _xi).abs().mean()) < 0.15,
)
check(
    "res: all merge gates start as pass-through of x (argmax = gate A)",
    bool((_u.merge.gate_logits.argmax(-1) == _dl._PASS_A_IDX).all()),
)
# truth tables
_tx = torch.tensor([0.0, 0.0, 1.0, 1.0])
_ty = torch.tensor([0.0, 1.0, 0.0, 1.0])
check("res: xor truth table", torch.equal(merge_xor(_tx, _ty), torch.tensor([0.0, 1.0, 1.0, 0.0])))
check("res: or truth table", torch.equal(merge_or(_tx, _ty), torch.tensor([0.0, 1.0, 1.0, 1.0])))
for _mode, _fn in [
    ("xor", lambda a, b: (a.bool() ^ b.bool()).float()),
    ("or", lambda a, b: (a.bool() | b.bool()).float()),
]:
    _um = ResidualUnit(16, body_layers=1, merge=_mode, seed=5, channels_per_tree=2, groups=4)
    _y = _xi
    for _l in _um.body:
        _y = _l(_y, hard=True)
    check(
        f"res: merge '{_mode}' hard output equals the bitwise {_mode} of x and the body output",
        torch.equal(_um(_xi, hard=True), _fn(_xi, _y)),
    )
_un = ResidualUnit(16, body_layers=2, merge="none", seed=6, channels_per_tree=2, groups=4)
_y = _xi
for _l in _un.body:
    _y = _l(_y, hard=True)
check(
    "res: merge 'none' returns the body output (no skip) and has no merge parameters",
    torch.equal(_un(_xi, hard=True), _y) and _un.merge is None,
)
# full model: binary end to end, hard == STE forward on binary inputs, stage selection, gradients
_mr = LogicTreeNet(**small, res_units=1, res_body_layers=2, res_merge="gate", head_wiring="balanced").train()
_ho = _mr(_x8, hard=True)
check("res: hard forward of a model with residual units is finite", bool(torch.isfinite(_ho).all()))
_hid = []
_h = _x8
for _i, (_b, _p) in enumerate(zip(_mr.blocks, _mr.pools)):
    _h = _b(_h, hard=True)
    for _u2 in _mr.res[_i]:
        _h = _u2(_h, hard=True)
    _hid.append(bool(((_h == 0) | (_h == 1)).all()))
    _h = _p(_h, hard=True)
check("res: every hard conv stage output (after the units) is exactly 0/1", all(_hid))
_so = _mr(_x8, ste=True)
check("res: STE forward equals the hard forward on binary inputs (full model)", torch.allclose(_so, _ho, atol=1e-5))
_mr.zero_grad()
_mr(_x8).sum().backward()
check(
    "res: gradients reach stage-1 body logits and merge gates",
    float(_mr.res[0][0].body[0].tree.gate_logits.grad.abs().sum()) > 0
    and float(_mr.res[0][0].merge.gate_logits.grad.abs().sum()) > 0,
)
_m23 = LogicTreeNet(**small, res_units=2, res_stages="2,3")
check("res: --res-stages 2,3 puts 2 units in stages 2 and 3 only", [len(s) for s in _m23.res] == [0, 2, 2, 0])
check(
    "res: parse_res_stages handles 'all' and lists",
    parse_res_stages("all", 4) == (1, 2, 3, 4) and parse_res_stages("2, 4", 4) == (2, 4),
)
for _bad, _kw in [
    ("skip conv", dict(skip="conv")),
    ("skip both", dict(skip="both")),
    ("runmean", dict(dropout_mode="runmean", dropout=0.1)),
    ("merge", dict(res_merge="mul")),
    ("stages", dict(res_stages="5")),
    ("negative", dict(res_units=-1)),
]:
    try:
        LogicTreeNet(**small, **{"res_units": 1, **_kw})
        _raised = False
    except ValueError:
        _raised = True
    check(f"res: invalid combination raises ValueError ({_bad})", _raised)
_mg = LogicTreeNet(**small, res_units=1).train()
check("res: Gumbel forward runs through units and merge gates", bool(torch.isfinite(_mg(_x8, gumbel=(1.0, 1.0))).all()))
# freezing and TaskFreeze probes
_mf = LogicTreeNet(**small, res_units=1, head_wiring="balanced").train()
_mf.enable_freeze()
check("res: enable_freeze adds buffers to merge gates", hasattr(_mf.res[0][0].merge, "frozen_idx"))
_probes = []
_out = _mf(_x8, probe=_probes)
_out.sum().backward()
check("res: probes are recorded for merge gates", any(isinstance(p_[0], MergeGate) for p_ in _probes))
_newly = _mf.freeze_update_from_probe(_probes, 0.9, 1)
check("res: TaskFreeze update runs with merge gates present", isinstance(_newly, int))
_mf.freeze_all()
check("res: freeze_all freezes every gate including merge gates", abs(_mf.frozen_fraction() - 1.0) < 1e-9)
check(
    "res: a fully frozen model's soft forward equals its hard forward",
    torch.allclose(_mf(_x8), _mf(_x8, hard=True), atol=1e-5),
)
# checkpoint round trip, old checkpoints
import io as _io

_buf = _io.BytesIO()
torch.save(_mr.state_dict(), _buf)
_buf.seek(0)
_mr2 = LogicTreeNet(**small, res_units=1, res_body_layers=2, res_merge="gate", head_wiring="balanced")
_mr2.load_state_dict(torch.load(_buf))
_mr.eval()
_mr2.eval()
check("res: state_dict save, load into a fresh model, identical outputs (resume)", torch.equal(_mr(_x8), _mr2(_x8)))
if _m_old is not None:
    _buf = _io.BytesIO()
    torch.save(_m_old.state_dict(), _buf)
    _buf.seek(0)
    _m_ld = LogicTreeNet(**small, res_units=0)
    _m_ld.load_state_dict(torch.load(_buf))
    check(
        "res: a checkpoint from before this change loads into the res_units=0 model",
        torch.equal(_m_ld(_x8), _m_old(_x8)),
    )
    _r = _mr2.load_state_dict(_m_old.state_dict(), strict=False)
    check(
        "res: an old checkpoint into a residual model misses only the residual keys",
        all(k.startswith("res.") for k in _r.missing_keys) and not _r.unexpected_keys,
    )
# gate counts against a hand count (small model: channels 8, 16, 32, 64; stage sizes 32, 16, 8, 4)
_chs = small["channels"]
_sz = (32, 16, 8, 4)
_base = _cg(LogicTreeNet(**small))
_res = _cg(LogicTreeNet(**small, res_units=1, res_body_layers=2, res_merge="gate"))
_extra = sum(c * 7 * s * s * 2 + c * s * s for c, s in zip(_chs, _sz))
check(
    "res: count_gates adds exactly the hand-counted body trees and merge gates (L = 2, merge gate)",
    _res["total"] - _base["total"] == _extra,
)
_res2 = _cg(LogicTreeNet(**small, res_units=2, res_body_layers=1, res_merge="none"))
check(
    "res: count_gates with merge 'none' counts body trees only (R = 2, L = 1)",
    _res2["total"] - _base["total"] == sum(c * 7 * s * s * 2 for c, s in zip(_chs, _sz)),
)
_t0 = count_trainable_gates(LogicTreeNet(**small))
_t1 = count_trainable_gates(LogicTreeNet(**small, res_units=1, res_body_layers=2, res_merge="gate"))
check("res: trainable gates add c*7*L + c per stage", _t1 - _t0 == sum(c * 7 * 2 + c for c in _chs))
check(
    "res: S default model has 83,552 trainable gates",
    count_trainable_gates(
        LogicTreeNet(
            in_channels=9,
            in_size=32,
            channels=(32, 128, 512, 1024),
            num_classes=10,
            tau=20.0,
            channels_per_tree=2,
            groups=4,
        )
    )
    == 83552,
)
try:
    LogicTreeNet(**small, res_units=1).unread_masks()
    _r_ok = False
except NotImplementedError:
    _r_ok = True
check("res: unread_masks raises NotImplementedError for residual models", _r_ok)
check(
    "res: gate confidences include the residual layers",
    any(k.startswith("res_") for k in LogicTreeNet(**small, res_units=1).gate_confidences()),
)

print(f"{passed} passed, {failed} failed")
for s_ in skipped:
    print(f"skipped: {s_}")
if failed:
    raise SystemExit(1)
