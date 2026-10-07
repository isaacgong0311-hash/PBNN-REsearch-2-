"""
CIFAR-10 baseline using the DiffLogicNet architecture from
Petersen, Borgelt, Kuehne, Deussen, "Deep Differentiable Logic Gate Networks"
(NeurIPS 2022, arXiv:2210.08277) -- Section 6.4.

Second literature baseline for the PBNN benchmark repo (alongside the
Courbariaux et al. Binarized Neural Networks baseline), matching the
CIFAR-10 pivot direction from Liping.

Preprocessing (paper Sec. 6.4, "color-channel resolution of 4"):
  each of the 3 CIFAR-10 color channels is binarized with 3 thresholds
  (0.25, 0.5, 0.75) -> 32*32*3*3 = 9216 binary input features.

Architecture (paper Table 5, "small" config for this resolution setting):
  a "straight" network: constant width per hidden layer, num_layers in [4,8],
  followed by GroupSum over 10 class groups. This script defaults to a
  modest width so it can be smoke-tested on CPU; bump `layer_width` /
  `num_layers` for a full-scale run (paper's larger CIFAR-10 configs use
  layers with tens of thousands of neurons -- outside what this sandbox
  can run, same constraint as the earlier smoke-test baseline).

Usage:
    python cifar10_difflogicnet.py --data-root ./data --epochs 30 \
        --layer-width 8000 --num-layers 5
"""
import argparse
import sys
import time

sys.path.insert(0, "/home/claude/difflogic")
import torch
import torch.nn.functional as F

from difflogic_net import DiffLogicNet


def binarize_cifar_batch(x: torch.Tensor, thresholds=(0.25, 0.5, 0.75)) -> torch.Tensor:
    """x: (batch, 3, 32, 32) float in [0,1] -> (batch, 3*32*32*len(thresholds)) binary."""
    b = x.shape[0]
    feats = [(x > t).float() for t in thresholds]     # each (batch, 3, 32, 32)
    stacked = torch.stack(feats, dim=1)                # (batch, T, 3, 32, 32)
    return stacked.reshape(b, -1)


def get_cifar10_loaders(data_root: str, batch_size: int):
    import torchvision
    import torchvision.transforms as T

    tfm = T.Compose([T.ToTensor()])
    train_set = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=tfm)
    test_set = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2)
    return train_loader, test_loader


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default="./data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--layer-width", type=int, default=8000)   # must be divisible by 10
    ap.add_argument("--num-layers", type=int, default=5)       # paper: 4-8 for "straight" nets
    ap.add_argument("--tau", type=float, default=100.0)        # paper heuristic: grows with n
    ap.add_argument("--lr", type=float, default=0.01)          # paper: constant Adam lr=0.01
    ap.add_argument("--limit-train-batches", type=int, default=None,
                     help="cap #batches/epoch for a quick sandbox smoke test")
    args = ap.parse_args()

    assert args.layer_width % 10 == 0, "layer_width must be divisible by 10 classes"

    train_loader, test_loader = get_cifar10_loaders(args.data_root, args.batch_size)

    in_dim = 3 * 32 * 32 * 3   # 3 thresholds per channel, per Sec 6.4
    model = DiffLogicNet(in_dim, args.layer_width, args.num_layers, num_classes=10,
                          tau=args.tau, seed=0)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"DiffLogicNet: in_dim={in_dim}, width={args.layer_width}, "
          f"layers={args.num_layers}, total neurons={model.num_parameters()}")

    for epoch in range(args.epochs):
        model.train()
        t0 = time.time()
        total_loss, n_seen = 0.0, 0
        for bi, (x, y) in enumerate(train_loader):
            if args.limit_train_batches and bi >= args.limit_train_batches:
                break
            xb = binarize_cifar_batch(x)
            logits = model(xb)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * x.size(0)
            n_seen += x.size(0)
        print(f"epoch {epoch:3d}  loss {total_loss / max(n_seen,1):.4f}  "
              f"({time.time()-t0:.1f}s)")

    # final evaluation: soft (relaxed) vs hard (discretized, bit-op-only) accuracy
    model.eval()
    with torch.no_grad():
        correct_soft = correct_hard = total = 0
        for x, y in test_loader:
            xb = binarize_cifar_batch(x)
            correct_soft += (model(xb).argmax(-1) == y).sum().item()
            correct_hard += (model.discretized_forward(xb).argmax(-1) == y).sum().item()
            total += x.size(0)
    print(f"\nsoft test acc: {correct_soft/total:.4f}")
    print(f"hard test acc: {correct_hard/total:.4f}  "
          f"(discretization gap: {abs(correct_soft-correct_hard)/total*100:.2f} pp)")
    print(f"bit-op count / inference: {model.bit_op_count_per_inference()}")


if __name__ == "__main__":
    main()
