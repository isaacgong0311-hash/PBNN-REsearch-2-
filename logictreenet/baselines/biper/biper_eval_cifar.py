"""Phase 0: evaluate the released BiPer ResNet-18 CIFAR-10 checkpoint without editing BiPer's code.
Usage: python biper_eval_cifar.py <freq> <checkpoint> <cifar10_test.parquet> <out.json>
Uses BiPer's own model (models_cifar/resnet2.py, modules/binarized_modules.py) in stage-2 mode (weights = sign(sin(freq * w))),
their test transform (ToTensor + Normalize with the ImageNet mean/std), eval mode. Also checks the binary weights and counts MACs."""

import io
import json
import os
import sys
import time

freq, ckpt, pq, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
BIPER = os.path.join(os.path.dirname(os.path.abspath(__file__)), "BiPer", "cifar10")
sys.path.insert(0, BIPER)
os.chdir(BIPER)
sys.argv = ["x", "--stage2", "--freq", freq]  # BiPer reads these at import time (utils/options.py)
import models_cifar.resnet2 as r2
import numpy as np
import pyarrow.parquet as papq
import torch
import torch.nn as nn
from modules.binarized_modules import BinarizeConv2d
from PIL import Image

torch.set_num_threads(max(1, os.cpu_count() // 2))
model = r2.resnet18_1w1a(num_classes=10)
st = torch.load(ckpt, map_location="cpu", weights_only=False)
sd = {k[7:] if k.startswith("module.") else k: v for k, v in st["state_dict"].items()}
missing = model.load_state_dict(sd, strict=True)
model.eval()
df = papq.read_table(pq).to_pydict()
imgcol = [c for c in df if "im" in c][0]
labcol = [c for c in df if "label" in c][0]
mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
X = np.stack([np.array(Image.open(io.BytesIO(d["bytes"])).convert("RGB")) for d in df[imgcol]])
X = (torch.from_numpy(X).permute(0, 3, 1, 2).float() / 255 - mean) / std
Y = torch.tensor(df[labcol], dtype=torch.long)
# MAC counting by forward hooks
macs = {"binary_conv": 0, "fp_conv": 0, "fp_linear": 0}


def hook(m, i, o):
    if isinstance(m, nn.Conv2d):
        k = o.shape[2] * o.shape[3] * m.out_channels * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]
        macs["binary_conv" if isinstance(m, BinarizeConv2d) else "fp_conv"] += k
    elif isinstance(m, nn.Linear):
        macs["fp_linear"] += m.in_features * m.out_features


hs = [m.register_forward_hook(hook) for m in model.modules() if isinstance(m, (nn.Conv2d, nn.Linear))]
t0 = time.time()
correct = 0
n = 0
with torch.no_grad():
    for i in range(0, len(X), 250):
        o = model(X[i : i + 250])
        correct += int((o.argmax(1) == Y[i : i + 250]).sum())
        n += len(o)
        if i == 0:
            for h in hs:
                h.remove()
            per_img = dict(
                macs
            )  # hooks fire once per forward and ignore the batch dimension, so this is MACs per image
res = dict(
    freq=float(freq),
    n_test=n,
    top1=100.0 * correct / n,
    seconds=round(time.time() - t0, 1),
    ckpt_epoch=st.get("epoch"),
    ckpt_best_prec1=st.get("best_prec1"),
    macs_per_image=per_img,
    n_params=sum(p.numel() for p in model.parameters()),
)
# binary weight check
layers = []
for name, m in model.named_modules():
    if isinstance(m, BinarizeConv2d):
        w = torch.sign(torch.sin(float(freq) * m.weight.data))
        layers.append(
            dict(
                name=name,
                shape=list(m.weight.shape),
                frac_plus=float((w > 0).float().mean()),
                n_zero=int((w == 0).sum()),
                only_pm1=bool(((w == 1) | (w == -1)).all()),
            )
        )
res["binary_conv_layers"] = layers
res["all_binary_conv_weights_pm1"] = all(l["only_pm1"] for l in layers)
res["full_precision_modules"] = [
    f"{name}: {type(m).__name__}"
    for name, m in model.named_modules()
    if isinstance(m, (nn.Conv2d, nn.Linear, nn.BatchNorm2d, nn.BatchNorm1d)) and not isinstance(m, BinarizeConv2d)
]
res["has_alpha_scaling"] = sum(1 for m in model.modules() if isinstance(m, BinarizeConv2d))
json.dump(res, open(out, "w"), indent=1)
print(
    json.dumps(
        {
            k: res[k]
            for k in [
                "freq",
                "top1",
                "seconds",
                "ckpt_epoch",
                "ckpt_best_prec1",
                "macs_per_image",
                "n_params",
                "all_binary_conv_weights_pm1",
            ]
        }
    )
)
