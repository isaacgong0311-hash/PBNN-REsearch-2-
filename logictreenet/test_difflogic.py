"""Test suite for difflogic.py / train.py. Run: python test_difflogic.py"""

import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, ".")

from difflogic import (
    GATE_COEFFS, GATE_NAMES, _ALL_TRUTH_TABLES,
    gate_forward, gate_forward_hard, gate_confidence,
    _residual_gate_init, _PASS_A_IDX, _PASS_B_IDX,
    LogicLayer, GroupSum, binarize, DiffLogicNet, DIFFLOGICNET_CIFAR,
    LogicTree, LogicTreeConv2d, OrPool2d, LogicTreeNet, LOGICTREENET_CIFAR,
)
from train import sharpness_at, weight_decay_at

passed = 0
failed = 0


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
for i, idx in enumerate([_ALL_TRUTH_TABLES.index((0, 0, 0, 1)),  # AND
                          _ALL_TRUTH_TABLES.index((0, 1, 1, 1)),  # OR
                          _ALL_TRUTH_TABLES.index((0, 1, 1, 0)),  # XOR
                          _PASS_A_IDX]):
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
check("LogicLayer wiring deterministic across construction w/ same seed",
      torch.equal(layer.idx_a, layer2.idx_a) and torch.equal(layer.idx_b, layer2.idx_b))

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
    check(f"DIFFLOGICNET_CIFAR[{scale}] has required keys",
          {"width", "depth", "tau", "reported"}.issubset(cfg.keys()))

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
            patch = padded[0, :, hh:hh + 3, ww:ww + 3].reshape(-1)  # (window,)
            leaves_px = patch[conv2.leaf_idx]  # (out_channels, n_leaves)
            val = conv2.tree(leaves_px.unsqueeze(0), hard=False)  # (1, out_channels)
            out_naive[0, :, hh, ww] = val[0]
    check("LogicTreeConv2d vectorized matches naive per-pixel loop", torch.allclose(out_vec, out_naive, atol=1e-4))

# ---------------------------------------------------------------------------
# OrPool2d
# ---------------------------------------------------------------------------

pool = OrPool2d(2, 2)
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
check("OrPool2d soft works on non-contiguous input and matches contiguous result",
      torch.allclose(pool(xnc, hard=False), pool(xnc.contiguous(), hard=False)))
check("LogicTreeConv2d output is contiguous", conv(torch.rand(1, 2, 6, 6)).is_contiguous())

# ---------------------------------------------------------------------------
# LogicTreeNet (full model)
# ---------------------------------------------------------------------------

model = LogicTreeNet(in_channels=3, in_size=8, channels=(4, 8), num_classes=5,
                      tree_depth=2, kernel_size=3, tau=1.0, seed=0)
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
check("LogicTreeNet head widths match paper taper 1280k/640k/320k",
      actual_widths == expected_widths)

for scale, cfg in LOGICTREENET_CIFAR.items():
    check(f"LOGICTREENET_CIFAR[{scale}] has required keys",
          {"channels", "tau", "reported"}.issubset(cfg.keys()))
check("LOGICTREENET_CIFAR scale B removed (needs unimplemented preprocessing)", "B" not in LOGICTREENET_CIFAR)
check("LOGICTREENET_CIFAR M channels = 256,1024,4096,8192", LOGICTREENET_CIFAR["M"]["channels"] == (256, 1024, 4096, 8192))
check("LOGICTREENET_CIFAR S channels = 32,128,512,1024", LOGICTREENET_CIFAR["S"]["channels"] == (32, 128, 512, 1024))

s_params = sum(p.numel() for p in
                LogicTreeNet(in_channels=3, in_size=32, num_classes=10, tree_depth=3, kernel_size=3,
                             **{k: v for k, v in LOGICTREENET_CIFAR["S"].items() if k != "reported"}).parameters())
m_params = sum(p.numel() for p in
                LogicTreeNet(in_channels=3, in_size=32, num_classes=10, tree_depth=3, kernel_size=3,
                             **{k: v for k, v in LOGICTREENET_CIFAR["M"].items() if k != "reported"}).parameters())
check("LogicTreeNet scale M has more params than scale S", m_params > s_params)

# ---------------------------------------------------------------------------
# channels_per_tree (paper Sec 3.1: each tree observes only 2 input channels)
# ---------------------------------------------------------------------------

kk = 9
c2 = LogicTreeConv2d(in_channels=16, out_channels=50, kernel_size=3, tree_depth=3, seed=5, channels_per_tree=2)
chans_per_row = [len(set((row // kk).tolist())) for row in c2.leaf_idx]
check("channels_per_tree=2: every tree reads from exactly 2 channels", all(n == 2 for n in chans_per_row))
check("channels_per_tree=2: leaves within a tree are distinct",
      all(len(set(row.tolist())) == 8 for row in c2.leaf_idx))
check("channels_per_tree=2: leaf indices in range", int(c2.leaf_idx.max()) < 16 * kk and int(c2.leaf_idx.min()) >= 0)
check("channels_per_tree=2: different trees use different channel pairs",
      len({tuple(sorted(set((row // kk).tolist()))) for row in c2.leaf_idx}) > 10)

c_all = LogicTreeConv2d(in_channels=16, out_channels=50, kernel_size=3, tree_depth=3, seed=5, channels_per_tree=0)
check("channels_per_tree=0: trees span more than 2 channels (old all-channel behavior)",
      max(len(set((row // kk).tolist())) for row in c_all.leaf_idx) > 2)

c1 = LogicTreeConv2d(in_channels=1, out_channels=4, kernel_size=3, tree_depth=3, seed=0, channels_per_tree=2)
check("channels_per_tree with a single input channel still works",
      c1(torch.rand(2, 1, 5, 5)).shape == (2, 4, 5, 5))

c_small = LogicTreeConv2d(in_channels=3, out_channels=4, kernel_size=1, tree_depth=3, seed=0, channels_per_tree=2)
check("channels_per_tree with fewer candidates than leaves (1x1 kernel) still works",
      c_small(torch.rand(2, 3, 4, 4)).shape == (2, 4, 4, 4)
      and all(len(set((row // 1).tolist())) <= 2 for row in c_small.leaf_idx))

net_cpt = LogicTreeNet(in_channels=3, in_size=8, channels=(4, 8), num_classes=5,
                       tree_depth=2, kernel_size=3, tau=1.0, seed=0)
net_cpt2 = LogicTreeNet(in_channels=3, in_size=8, channels=(4, 8), num_classes=5,
                        tree_depth=2, kernel_size=3, tau=1.0, seed=0, channels_per_tree=2)
check("LogicTreeNet passes channels_per_tree=2 through to every conv block",
      all(b.channels_per_tree == 2 for b in net_cpt2.blocks))
check("LogicTreeNet defaults to channels_per_tree=0 (all channels) in every conv block",
      all(b.channels_per_tree == 0 for b in net_cpt.blocks))
xin = torch.rand(2, 3, 8, 8, requires_grad=True)
net_cpt(xin).sum().backward()
check("LogicTreeNet (default wiring): gradient reaches every conv block",
      all(b.tree.gate_logits.grad.norm().item() > 1e-12 for b in net_cpt.blocks))

# ---------------------------------------------------------------------------
# straight-through (ste) training mode
# ---------------------------------------------------------------------------

from difflogic import gate_probs
lg = torch.randn(7, 16, requires_grad=True)
pr = gate_probs(lg, ste=True)
check("gate_probs(ste=True) forward is exactly one-hot",
      torch.allclose(pr.detach(), F.one_hot(lg.argmax(-1), 16).float(), atol=1e-6))
(pr * torch.randn(7, 16)).sum().backward()
check("gate_probs(ste=True) passes gradients to logits", lg.grad.norm().item() > 1e-8)

torch.manual_seed(1)
ste_net = LogicTreeNet(in_channels=9, in_size=16, channels=(4, 8), num_classes=10,
                       tree_depth=3, kernel_size=3, tau=5.0, seed=3)
for prm in ste_net.parameters():
    prm.data += torch.randn_like(prm) * 3  # move argmaxes off the pass-through init
xb = binarize(torch.rand(6, 3, 16, 16), (0.25, 0.5, 0.75), flatten=False)
with torch.no_grad():
    check("LogicTreeNet ste=True forward equals the hard (discretized) network exactly",
          torch.allclose(ste_net(xb, ste=True), ste_net(xb, hard=True), atol=1e-5))
ste_net.zero_grad()
ste_net(xb, ste=True).sum().backward()
check("LogicTreeNet ste=True: gradient reaches every conv block and head layer",
      all(b.tree.gate_logits.grad.norm().item() > 1e-12 for b in ste_net.blocks)
      and all(l.gate_logits.grad.norm().item() > 1e-12 for l in ste_net.head))

dn = DiffLogicNet(in_features=48, width=40, depth=3, num_classes=10, tau=2.0, seed=2)
for prm in dn.parameters():
    prm.data += torch.randn_like(prm) * 3
xdn = binarize(torch.rand(5, 3, 4, 4), (0.5,), flatten=True)
with torch.no_grad():
    check("DiffLogicNet ste=True forward equals hard forward",
          torch.allclose(dn(xdn, ste=True), dn(xdn, hard=True), atol=1e-5))

# ---------------------------------------------------------------------------
# sharpness_at / weight_decay_at schedules
# ---------------------------------------------------------------------------

check("sharpness_at off when final_sharpness<=1.0", sharpness_at(50, 100, 1.0, 0.5) == 1.0)
check("sharpness_at held at 1.0 during warmup", sharpness_at(10, 100, 5.0, 0.5) == 1.0)
check("sharpness_at reaches final at last epoch", abs(sharpness_at(99, 100, 5.0, 0.5) - 5.0) < 0.2)
s_mid = sharpness_at(75, 100, 5.0, 0.5)
check("sharpness_at monotonic non-decreasing after warmup",
      sharpness_at(60, 100, 5.0, 0.5) <= s_mid <= sharpness_at(90, 100, 5.0, 0.5))

check("weight_decay_at off (base unchanged) when final_sharpness<=1.0",
      weight_decay_at(50, 100, 0.002, 1.0, 0.5) == 0.002)
check("weight_decay_at held at base during warmup", weight_decay_at(10, 100, 0.002, 5.0, 0.5) == 0.002)
check("weight_decay_at decays to ~0 at end", weight_decay_at(99, 100, 0.002, 5.0, 0.5) < 0.0005)
wd_mid = weight_decay_at(75, 100, 0.002, 5.0, 0.5)
check("weight_decay_at monotonic non-increasing after warmup",
      weight_decay_at(60, 100, 0.002, 5.0, 0.5) >= wd_mid >= weight_decay_at(90, 100, 0.002, 5.0, 0.5))

# ---------------------------------------------------------------------------
# straight-through phase schedule (--ste-start-epoch / --ste-lr)
# ---------------------------------------------------------------------------

from train import ste_active, lr_at
check("ste_active off by default", not any(ste_active(e, False, -1) for e in range(10)))
check("ste_active --ste is on every epoch", all(ste_active(e, True, -1) for e in range(10)))
check("ste_active switches on at start epoch",
      [ste_active(e, False, 7) for e in range(10)] == [False]*7 + [True]*3)
check("lr_at keeps base lr with no ste_lr", lr_at(9, 0.02, 7, None) == 0.02)
check("lr_at keeps base lr before the switch", lr_at(6, 0.02, 7, 3e-4) == 0.02)
check("lr_at drops to ste_lr at the switch", lr_at(7, 0.02, 7, 3e-4) == 3e-4)
check("lr_at ignores ste_lr when phase is off", lr_at(9, 0.02, -1, 3e-4) == 0.02)

# ---------------------------------------------------------------------------

print(f"{passed} passed, {failed} failed")
if failed:
    raise SystemExit(1)