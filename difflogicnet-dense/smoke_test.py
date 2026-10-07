"""
End-to-end correctness check for DiffLogicNet on sklearn's `digits` dataset
(8x8 grayscale, 10 classes, ~1800 samples) -- small enough to run on CPU in
seconds, just to verify: (1) the relaxed net trains and loss decreases,
(2) the discretized (hard, bit-op-only) net tracks the continuous net closely,
matching the paper's claim of a <0.1%-scale discretization gap.
"""
import sys
sys.path.insert(0, "/home/claude/difflogic")
import torch
import torch.nn.functional as F
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from difflogic_net import DiffLogicNet

torch.manual_seed(0)
np.random.seed(0)

# --- data: binarize 8x8 digit images with 3 thresholds per pixel (like the
# paper's CIFAR-10 color-channel binary embedding, Sec 6.4) ---
digits = load_digits()
X = digits.data / 16.0          # scale to [0,1], shape (n, 64)
y = digits.target

thresholds = [0.25, 0.5, 0.75]
X_bin = np.concatenate([(X > t).astype(np.float32) for t in thresholds], axis=1)  # (n, 192)

X_train, X_test, y_train, y_test = train_test_split(
    X_bin, y, test_size=0.2, random_state=0, stratify=y
)
X_train = torch.tensor(X_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.long)
y_test = torch.tensor(y_test, dtype=torch.long)

in_dim = X_train.shape[1]
num_classes = 10
layer_width = 1000     # must be divisible by num_classes
num_layers = 5          # paper uses 4-8 layers for "straight" networks

model = DiffLogicNet(in_dim, layer_width, num_layers, num_classes, tau=30.0, seed=0)
opt = torch.optim.Adam(model.parameters(), lr=0.01)   # paper: Adam, lr=0.01 constant

n_epochs = 60
batch_size = 128
n_train = X_train.shape[0]

for epoch in range(n_epochs):
    perm = torch.randperm(n_train)
    total_loss = 0.0
    for i in range(0, n_train, batch_size):
        idx = perm[i:i + batch_size]
        xb, yb = X_train[idx], y_train[idx]
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.item() * len(idx)
    if epoch % 10 == 0 or epoch == n_epochs - 1:
        with torch.no_grad():
            soft_acc = (model(X_test).argmax(-1) == y_test).float().mean().item()
        print(f"epoch {epoch:3d}  loss {total_loss/n_train:.4f}  soft_test_acc {soft_acc:.4f}")

with torch.no_grad():
    soft_train_acc = (model(X_train).argmax(-1) == y_train).float().mean().item()
    soft_test_acc = (model(X_test).argmax(-1) == y_test).float().mean().item()
    hard_train_acc = (model.discretized_forward(X_train).argmax(-1) == y_train).float().mean().item()
    hard_test_acc = (model.discretized_forward(X_test).argmax(-1) == y_test).float().mean().item()

print("\n--- final ---")
print(f"soft (relaxed)   train acc: {soft_train_acc:.4f}   test acc: {soft_test_acc:.4f}")
print(f"hard (discrete)  train acc: {hard_train_acc:.4f}   test acc: {hard_test_acc:.4f}")
print(f"discretization gap (test): {abs(soft_test_acc - hard_test_acc)*100:.2f} pp")
print(f"#neurons (bit-op count / inference): {model.num_parameters()}")

# sanity: distribution over gates has mostly converged (peaky), as the paper reports
with torch.no_grad():
    for li, layer in enumerate(model.layers):
        p = torch.softmax(layer.weights, dim=-1)
        max_p = p.max(dim=-1).values.mean().item()
        print(f"layer {li}: mean max-gate-probability = {max_p:.3f} (1.0 = fully converged)")
