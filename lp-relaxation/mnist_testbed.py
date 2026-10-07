"""
PBNN project - MNIST testbed.

Two models trained and evaluated on MNIST:
  1. Full-precision MLP baseline
  2. Binary MLP (BNN-style, Hubara et al. 2016, arXiv:1602.02830) -
     weights AND activations binarized to {-1,+1} in forward pass,
     straight-through estimator (STE) for the backward pass, real-valued
     shadow weights kept for the optimizer step.

Architecture is scaled down from the paper (2 hidden layers x 256 units
vs. their 3 x 4096) to keep CPU runtime short. This is a testbed to
validate the training mechanics work end-to-end, not a SOTA reproduction.

Usage:
    python mnist_testbed.py --epochs 5
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Binary building blocks (Hubara et al. straight-through estimator)
# ---------------------------------------------------------------------------
class BinarizeSTE(torch.autograd.Function):
    """sign() forward, straight-through (clipped identity) backward."""

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        out = torch.sign(x)
        out[out == 0] = 1
        return out

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        # gradient passes through only where |x| <= 1 (hard-tanh clip)
        mask = (x.abs() <= 1).float()
        return grad_output * mask


def binarize(x):
    return BinarizeSTE.apply(x)


class BinaryLinear(nn.Module):
    """Linear layer with binarized weights; real-valued weights are the
    optimizer's shadow copy (clamped to [-1, 1] after each step)."""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        nn.init.uniform_(self.weight, -1, 1)

    def forward(self, x):
        w_bin = binarize(self.weight)
        return F.linear(x, w_bin, self.bias)

    @torch.no_grad()
    def clip_weights(self):
        self.weight.clamp_(-1, 1)


class MLP(nn.Module):
    """Full-precision baseline."""

    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class BinaryMLP(nn.Module):
    """BNN-style: binary weights + binary activations, BatchNorm between
    layers (as in Hubara et al. - BN is load-bearing for BNN training)."""

    def __init__(self, hidden=256):
        super().__init__()
        self.fc1 = BinaryLinear(784, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = BinaryLinear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.fc3 = BinaryLinear(hidden, 10)
        self.bn3 = nn.BatchNorm1d(10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = binarize(self.bn1(self.fc1(x)))
        x = binarize(self.bn2(self.fc2(x)))
        x = self.bn3(self.fc3(x))
        return x

    def clip_weights(self):
        for m in self.modules():
            if isinstance(m, BinaryLinear):
                m.clip_weights()


# ---------------------------------------------------------------------------
# Train / eval
# ---------------------------------------------------------------------------
def get_loaders(batch_size=128):
    tfm = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST("/home/claude/pbnn/data", train=True, download=True, transform=tfm)
    test = datasets.MNIST("/home/claude/pbnn/data", train=False, download=True, transform=tfm)
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True),
        DataLoader(test, batch_size=1000, shuffle=False),
    )


def evaluate(model, loader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total


def train(model, loader, test_loader, epochs, lr, binary=False, tag=""):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        model.train()
        t0 = time.time()
        total_loss = 0.0
        for x, y in loader:
            opt.zero_grad()
            out = model(x)
            loss = F.cross_entropy(out, y)
            loss.backward()
            opt.step()
            if binary:
                model.clip_weights()
            total_loss += loss.item() * x.size(0)
        acc = evaluate(model, test_loader)
        print(f"[{tag}] epoch {epoch+1}/{epochs}  loss={total_loss/len(loader.dataset):.4f}  "
              f"test_acc={acc*100:.2f}%  ({time.time()-t0:.1f}s)")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--hidden", type=int, default=256)
    args = ap.parse_args()

    train_loader, test_loader = get_loaders()

    print("=== Full-precision MLP baseline ===")
    fp_model = MLP(hidden=args.hidden)
    fp_acc = train(fp_model, train_loader, test_loader, args.epochs, lr=1e-3, tag="FP")

    print("\n=== Binary MLP (BNN-style, STE) ===")
    bin_model = BinaryMLP(hidden=args.hidden)
    bin_acc = train(bin_model, train_loader, test_loader, args.epochs, lr=1e-2, binary=True, tag="BIN")

    print(f"\nFinal: full-precision={fp_acc*100:.2f}%  binary={bin_acc*100:.2f}%  "
          f"gap={100*(fp_acc-bin_acc):.2f}pp")


if __name__ == "__main__":
    main()
