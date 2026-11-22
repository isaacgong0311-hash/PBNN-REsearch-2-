"""CLI trainer for DiffLogicNet (dense) and LogicTreeNet (conv) on CIFAR-10."""

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T

from difflogic import (
    DIFFLOGICNET_CIFAR,
    LOGICTREENET_CIFAR,
    DiffLogicNet,
    LogicTreeNet,
    balance_penalty,
    binarize,
    count_gates,
    count_trainable_gates,
)

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


class _HFCIFAR10(torch.utils.data.Dataset):
    """Wraps the uoft-cs/cifar10 HuggingFace mirror (much faster to fetch
    than torchvision's default download source in many environments)."""

    def __init__(self, split, transform=None):
        from datasets import load_dataset

        self.ds = load_dataset("uoft-cs/cifar10", split=split)
        self.transform = transform

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        img = row["img"]
        label = row["label"]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def load_cifar10(data_dir, val_split, thresholds, augment=False):
    """45k/5k/10k split. With augment=True the training images (only) get a
    random 32x32 crop from a 4-pixel zero-padded image plus a random
    horizontal flip, applied before binarization. The paper does not mention
    augmentation, so it is off by default."""
    plain = T.Compose([T.ToTensor()])
    train_tf = T.Compose([T.RandomCrop(32, padding=4), T.RandomHorizontalFlip(), T.ToTensor()]) if augment else plain
    try:
        train_full = _HFCIFAR10("train", transform=train_tf)
        val_full = _HFCIFAR10("train", transform=plain)
        test_set = _HFCIFAR10("test", transform=plain)
    except Exception as e:
        print(f"HF CIFAR-10 mirror unavailable ({e}); falling back to torchvision download.")
        train_full = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=train_tf)
        val_full = torchvision.datasets.CIFAR10(data_dir, train=True, download=True, transform=plain)
        test_set = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=plain)

    n = len(train_full)
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    val_idx = perm[:val_split]
    train_idx = perm[val_split:]
    train_set = torch.utils.data.Subset(train_full, train_idx.tolist())
    val_set = torch.utils.data.Subset(val_full, val_idx.tolist())
    return train_set, val_set, test_set


def collate_binarize(batch, thresholds):
    imgs = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch])
    return imgs, labels


def pick_device(name="auto"):
    """auto = CUDA GPU, else Apple-silicon GPU (MPS), else CPU."""
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------


def resolve_groups(groups, k):
    """--groups -1 means the paper's k/8 (App. A.3): 4 groups on S, 32 on M."""
    if groups == -1:
        return max(k // 8, 1)
    return max(groups, 1)


def expected_arch_groups(args):
    return resolve_groups(args.groups, LOGICTREENET_CIFAR[args.scale]["channels"][0])


def normalize_arch(saved_arch):
    """Fill in arch fields that older checkpoints predate, with the values those
    runs actually used."""
    if saved_arch is None:
        return None
    a = {
        **saved_arch,
        "channels_per_tree": saved_arch.get("channels_per_tree", 0),
        "pool": saved_arch.get("pool", "prob"),
        "groups": saved_arch.get("groups", 1),
        "skip": saved_arch.get("skip", "none"),
    }
    # head_wiring is only stored when it is not the default, so older checkpoints still match
    a["skip_frac"] = saved_arch.get("skip_frac") if a["skip"] in ("head", "both") else None
    return a


def build_model(args, in_ch, in_size, n_classes, n_thresh):
    if args.model == "dense":
        cfg = dict(DIFFLOGICNET_CIFAR[args.scale])
        cfg.pop("reported", None)
        if args.tau_override is not None:
            cfg["tau"] = args.tau_override
        in_features = in_ch * n_thresh * in_size * in_size
        return DiffLogicNet(in_features=in_features, num_classes=n_classes, seed=args.seed, **cfg)
    elif args.model == "conv":
        cfg = dict(LOGICTREENET_CIFAR[args.scale])
        cfg.pop("reported", None)
        if args.tau_override is not None:
            cfg["tau"] = args.tau_override
        return LogicTreeNet(
            in_channels=in_ch * n_thresh,
            in_size=in_size,
            num_classes=n_classes,
            tree_depth=args.tree_depth,
            kernel_size=args.kernel_size,
            seed=args.seed,
            channels_per_tree=args.channels_per_tree,
            pool=args.pool,
            groups=resolve_groups(args.groups, cfg["channels"][0]),
            dropout=args.dropout,
            dropout_mode=args.dropout_mode,
            dropout_where=args.dropout_where,
            dropout_in_ste=args.dropout_in_ste or args.dropout_ste_only,
            res_units=args.res_units,
            res_body_layers=args.res_body_layers,
            res_tree_depth=args.res_tree_depth,
            res_merge=args.res_merge,
            res_stages=args.res_stages,
            res_merge_bias=args.res_merge_bias,
            dropout_soft=not args.dropout_ste_only,
            skip=args.skip,
            skip_frac=args.skip_frac,
            head_wiring=args.head_wiring,
            head_layers=args.head_layers,
            **cfg,
        )
    else:
        raise ValueError(f"unknown model {args.model}")


# ---------------------------------------------------------------------------
# Annealing schedules
# ---------------------------------------------------------------------------


def sharpness_at(epoch, total_epochs, final_sharpness, warmup_frac):
    """Linear ramp: sharpness=1.0 held during warmup_frac of training, then
    ramps linearly to final_sharpness by the end. Returns 1.0 (off) if
    final_sharpness <= 1.0."""
    if final_sharpness <= 1.0:
        return 1.0
    warmup_epochs = warmup_frac * total_epochs
    if epoch <= warmup_epochs:
        return 1.0
    if total_epochs <= warmup_epochs:
        return final_sharpness
    frac = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    frac = min(max(frac, 0.0), 1.0)
    return 1.0 + frac * (final_sharpness - 1.0)


def weight_decay_at(epoch, total_epochs, base_wd, final_sharpness, warmup_frac):
    """Anneals weight decay to 0 over the same window sharpness ramps up in
    (plain weight decay fights sharpness annealing: it pulls logit
    magnitude toward zero while sharpness tries to grow it). Returns
    base_wd unchanged if final_sharpness <= 1.0 (annealing off)."""
    if final_sharpness <= 1.0:
        return base_wd
    warmup_epochs = warmup_frac * total_epochs
    if epoch <= warmup_epochs:
        return base_wd
    if total_epochs <= warmup_epochs:
        return 0.0
    frac = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
    frac = min(max(frac, 0.0), 1.0)
    return base_wd * (1.0 - frac)


def ste_active(epoch, ste_flag, ste_start_epoch):
    """Whether the straight-through estimator is on for this epoch.
    --ste turns it on for every epoch (the separate fine-tune used so far);
    --ste-start-epoch N turns it on from epoch N to the end, so normal soft
    training and the straight-through phase happen in one run."""
    if ste_flag:
        return True
    return ste_start_epoch is not None and ste_start_epoch >= 0 and epoch >= ste_start_epoch


def lr_at(epoch, base_lr, ste_start_epoch, ste_lr):
    """Learning rate for this epoch: base_lr, switching to ste_lr once the
    straight-through phase starts (if ste_lr is given). The fine-tune results
    say straight-through wants a much smaller step than soft training
    (lr 0.005 made the soft network drift on S; 3e-4 worked on M)."""
    if ste_lr is not None and ste_start_epoch is not None and ste_start_epoch >= 0 and epoch >= ste_start_epoch:
        return ste_lr
    return base_lr


# ---------------------------------------------------------------------------
# Eval
# ---------------------------------------------------------------------------


@torch.no_grad()
def evaluate(model, loader, thresholds, device, hard=False, sharpness=1.0, flatten=True):
    model.eval()
    correct = 0
    total = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)
        x = binarize(imgs, thresholds=thresholds, flatten=flatten)
        out = model(x, hard=hard, sharpness=sharpness)
        pred = out.argmax(dim=-1)
        correct += (pred == labels).sum().item()
        total += labels.numel()
    model.train()
    return correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["dense", "conv"], required=True)
    p.add_argument("--scale", required=True)
    p.add_argument("--tree-depth", type=int, default=3)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--thresholds", type=str, default="0.5")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--optimizer", choices=["adam", "adamw"], default="adam")
    p.add_argument("--tau-override", type=float, default=None)
    p.add_argument("--val-split", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--data-dir", type=str, default="./data")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--log", type=str, required=True)
    p.add_argument("--eval-every", type=int, default=5)
    p.add_argument("--fresh", action="store_true")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--sharpness-final", type=float, default=1.0)
    p.add_argument("--sharpness-warmup-frac", type=float, default=0.5)
    p.add_argument(
        "--channels-per-tree",
        type=int,
        default=0,
        help="input channels each conv logic tree reads from; 0 = all (default). The paper says 2, "
        "but that lowered hard accuracy by ~7 pts on scale S here",
    )
    p.add_argument(
        "--pool",
        choices=["max", "prob"],
        default="max",
        help="soft OR-pooling: max t-conorm (paper Sec 3.1, default) or probabilistic sum "
        "(what runs before Oct 2026 used)",
    )
    p.add_argument(
        "--groups",
        type=int,
        default=1,
        help="split the conv blocks and head into this many groups with no cross-connections "
        "until GroupSum (paper App. A.3). -1 = the paper's k/8; 1 = no groups (default)",
    )
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument(
        "--ste",
        action="store_true",
        help="straight-through training: forward pass uses the discretized (argmax) gates, "
        "gradients flow through the soft gate probabilities",
    )
    p.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="start from this checkpoint's weights (fresh optimizer, epoch 0) when --ckpt "
        "doesn't exist yet; used for fine-tuning",
    )
    p.add_argument(
        "--ste-start-epoch",
        type=int,
        default=-1,
        help="switch to straight-through training from this epoch to the end of the run "
        "(-1 = off). Builds the straight-through phase into normal training instead of "
        "a separate fine-tune",
    )
    p.add_argument(
        "--ste-lr",
        type=float,
        default=None,
        help="learning rate to use once the straight-through phase starts (default: keep --lr)",
    )
    p.add_argument(
        "--accum-steps",
        type=int,
        default=1,
        help="split each batch into this many micro-batches and accumulate gradients "
        "(same training, less GPU memory; use for scale M on an 8 GB GPU)",
    )
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument(
        "--save-every-epoch", action="store_true", help="also save the resume checkpoint after non-eval epochs"
    )
    p.add_argument(
        "--augment",
        action="store_true",
        help="random crop (pad 4) + horizontal flip on training images; not in the paper",
    )
    p.add_argument(
        "--dropout",
        type=float,
        default=0.0,
        help="conv model: logic dropout prob (replace soft activations with 0.5) on the input "
        "of every conv block after the first and every head layer; soft non-STE training "
        "only. Not in the paper. 0 = off (default)",
    )
    p.add_argument(
        "--dropout-mode",
        choices=["half", "scale", "runmean", "perm", "permx"],
        default="half",
        help="with --dropout > 0: half = replace dropped activations with 0.5, no rescaling "
        "(keeps values in [0, 1], default); scale = textbook dropout (d2l.ai): dropped "
        "activations become 0 and kept ones are divided by 1-p (unbiased, values can exceed 1); runmean = dropped values "
        "are replaced by that input's running mean (EMA; keeps values in [0, 1], unbiased over the data; not in the paper)",
    )
    p.add_argument(
        "--dropout-in-ste",
        action="store_true",
        help="not in the paper: also apply --dropout in the straight-through phase (--ste / --ste-start-epoch), where the layer inputs are "
        "binary, so --dropout-mode perm/permx keep every value in {0,1}. Off by default (dropout is soft training only)",
    )
    p.add_argument(
        "--dropout-ste-only",
        action="store_true",
        help="not in the paper: apply --dropout only in the straight-through phase (implies --dropout-in-ste); no dropout in soft training",
    )
    p.add_argument(
        "--dropout-where",
        choices=["all", "conv1", "head", "convlate"],
        default="all",
        help="with --dropout > 0 (conv model): all = input of every conv block after the first and of every head layer "
        "(default); conv1 = only conv block 1's output (the input of block 2); head = only the head layers' inputs; "
        "convlate = only the outputs of conv blocks 2 and 3 (the inputs of blocks 3 and 4; not in the paper)",
    )
    p.add_argument(
        "--res-units",
        type=int,
        default=0,
        help="not in the paper (Prof. Liu's suggestion): residual units per conv stage, between the stage's conv block and its OR-pool. "
        "Each unit = --res-body-layers shape-preserving logic conv layers + a --res-merge gate of (stage input, body output). 0 = off (default)",
    )
    p.add_argument(
        "--res-body-layers", type=int, default=1, help="with --res-units > 0: logic conv layers inside one unit"
    )
    p.add_argument("--res-tree-depth", type=int, default=3, help="with --res-units > 0: tree depth of the body layers")
    p.add_argument(
        "--res-merge",
        choices=["gate", "xor", "or", "none"],
        default="gate",
        help="with --res-units > 0: gate = learnable 16-op gate per channel, starts as pass-through of the skip input; xor / or = fixed gate; "
        "none = no skip (x = body output), the depth-only ablation",
    )
    p.add_argument(
        "--res-stages",
        default="all",
        help="with --res-units > 0: 'all' or a comma list of 1-based conv stages, e.g. 2,3,4",
    )
    p.add_argument(
        "--res-merge-bias",
        type=float,
        default=5.0,
        help="with --res-merge gate: initial logit bias toward pass-through of the skip input",
    )
    p.add_argument(
        "--skip",
        choices=["none", "conv", "head", "both"],
        default="none",
        help="conv model: binary-safe concat skip connections (not in the paper). conv: block "
        "i>=2 also reads block i-2's OR-pooled output; head: head layer i>=1 also reads "
        "the previous layer's input. none = paper architecture (default)",
    )
    p.add_argument(
        "--skip-frac",
        type=float,
        default=0.25,
        help="with --skip head/both: prob that each head gate input reads the skip source",
    )
    p.add_argument(
        "--head-wiring",
        choices=["random", "balanced"],
        default="random",
        help="conv model: how the 3 dense head layers pick their inputs. random = paper (Sec. 2, 3.4, default); "
        "balanced = every input is read about equally often, so no head gate is unread (not in the paper)",
    )
    p.add_argument(
        "--balance-lam",
        type=float,
        default=0.0,
        help="conv model: weight of the balance penalty lam * mean_g (m_g - 0.5)^2 over gates that a later layer reads "
        "(m_g = gate's mean soft output over the batch and positions; not in the paper). 0 = off (default). "
        "Not used in the straight-through phase; with --accum-steps > 1 m_g is per micro-batch; no skip connections",
    )
    p.add_argument(
        "--head-layers",
        type=int,
        default=3,
        help="conv model: number of dense head layers. 3 = the paper (App. A.1.1: 1280k/640k/320k, default). N > 3 keeps the "
        "total head gate count by using N-1 equal layers of 1920k/(N-1) and a last layer of 320k (not in the paper)",
    )
    p.add_argument(
        "--log-grad-norms",
        action="store_true",
        help="conv model: log the mean (over batches) L2 norm of the gradient of every trainable layer's gate logits each epoch",
    )
    p.add_argument(
        "--gumbel",
        action="store_true",
        help="conv model: Gumbel straight-through gate sampling (Yousefi et al., arXiv:2506.07500, not in Petersen et al.): in soft "
        "training every gate choice is sampled with Gumbel noise, the forward pass uses the one-hot sampled gate and the "
        "gradient is that of softmax((logits + scale*g) / tau). Evaluation is unchanged (soft = plain softmax, hard = argmax). "
        "Off by default; cannot be combined with --ste / --ste-start-epoch",
    )
    p.add_argument(
        "--gumbel-tau",
        type=float,
        default=1.0,
        help="with --gumbel: softmax temperature tau of the backward pass (paper: 1)",
    )
    p.add_argument(
        "--gumbel-scale",
        type=float,
        default=1.0,
        help="with --gumbel: multiplier on the Gumbel(0, 1) noise (paper: 1; other values are our extension, the paper has no such knob)",
    )
    p.add_argument(
        "--freeze",
        choices=["none", "global", "task"],
        default="none",
        help="progressive discretization (Gerlach 2026, arXiv:2609.33747, not in Petersen et al.): global = GlobalFreeze, at "
        "--freeze-start-epoch every neuron is fixed to its current argmax gate for the rest of training; task = TaskFreeze, "
        "from that epoch on, every --freeze-k steps each neuron's first-order freeze shock of its argmax gate is scored on the "
        "minibatch, EMA-smoothed (beta), and the neuron is frozen once the EMA stays negative for --freeze-patience "
        "evaluations. none (default) = FinalArgmax, i.e. relaxed training and argmax only at the end. conv model only",
    )
    p.add_argument(
        "--freeze-start-epoch",
        type=int,
        default=-1,
        help="first epoch of freezing; -1 = 12.5%% of --epochs (paper App. B: 25k of 200k steps)",
    )
    p.add_argument("--freeze-k", type=int, default=1000, help="TaskFreeze scoring interval in steps (paper: 1000)")
    p.add_argument("--freeze-patience", type=int, default=3, help="TaskFreeze patience (paper: 3)")
    p.add_argument(
        "--freeze-beta", type=float, default=None, help="TaskFreeze EMA factor; default 2^(-K/10000) (paper)"
    )
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")
    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    n_thresh = len(thresholds)

    train_set, val_set, test_set = load_cifar10(args.data_dir, args.val_split, thresholds, augment=args.augment)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.eval_batch_size, shuffle=False, num_workers=args.num_workers
    )

    n_classes = 10
    in_ch = 3
    in_size = 32
    flatten = args.model == "dense"

    model = build_model(args, in_ch, in_size, n_classes, n_thresh).to(device)
    if args.freeze != "none" and args.model == "conv":
        model.enable_freeze()  # adds the frozen_idx / EMA buffers before any checkpoint is loaded
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"model: {args.model} scale {args.scale}, {n_params / 1e6:.2f}M params"
        + (
            f", pool={args.pool}, groups={expected_arch_groups(args)}, channels_per_tree={args.channels_per_tree}"
            f", dropout={args.dropout}"
            + (
                f" ({args.dropout_mode}"
                + (f", where={args.dropout_where}" if args.dropout_where != "all" else "")
                + ")"
                if args.dropout > 0
                else ""
            )
            + f", skip={args.skip}"
            + (f", skip_frac={args.skip_frac}" if args.skip in ("head", "both") else "")
            + (f", head_wiring={args.head_wiring}" if args.head_wiring != "random" else "")
            + (f", head_layers={args.head_layers}" if args.head_layers != 3 else "")
            + (
                f", res_units={args.res_units} (body_layers={args.res_body_layers}, tree_depth={args.res_tree_depth}, merge={args.res_merge}, stages={args.res_stages}, merge_bias={args.res_merge_bias})"
                if args.res_units > 0
                else ""
            )
            if args.model == "conv"
            else ""
        )
    )

    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, foreach=False)

    ckpt_path = Path(args.ckpt)
    log_path = Path(args.log)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    expected_arch = dict(
        model=args.model,
        scale=args.scale,
        tree_depth=args.tree_depth,
        kernel_size=args.kernel_size,
        thresholds=args.thresholds,
        channels_per_tree=(args.channels_per_tree if args.model == "conv" else 0),
        pool=(args.pool if args.model == "conv" else "prob"),
        groups=(
            resolve_groups(args.groups, LOGICTREENET_CIFAR[args.scale]["channels"][0]) if args.model == "conv" else 1
        ),
        skip=(args.skip if args.model == "conv" else "none"),
    )
    expected_arch["skip_frac"] = args.skip_frac if expected_arch["skip"] in ("head", "both") else None
    if args.model == "conv" and args.head_wiring != "random":
        expected_arch["head_wiring"] = args.head_wiring
    if args.model == "conv" and args.head_layers != 3:
        expected_arch["head_layers"] = args.head_layers
    if (
        args.model == "conv" and args.res_units > 0
    ):  # stored only when used, so checkpoints of models without residual units still match
        expected_arch.update(
            res_units=args.res_units,
            res_body_layers=args.res_body_layers,
            res_tree_depth=args.res_tree_depth,
            res_merge=args.res_merge,
            res_stages=args.res_stages,
            res_merge_bias=args.res_merge_bias,
        )
    if args.freeze != "none":
        expected_arch["freeze"] = (
            args.freeze
        )  # buffers frozen_idx etc. exist only then (checkpoints without freeze stay compatible)

    start_epoch = 0
    best_val = -1.0

    if ckpt_path.exists() and not args.fresh:
        state = torch.load(ckpt_path, map_location=device)
        saved_arch = state.get("arch")
        # checkpoints from before channels_per_tree / skip existed: fill in what they used
        saved_arch = normalize_arch(saved_arch)
        if saved_arch != expected_arch:
            raise SystemExit(
                f"Checkpoint architecture mismatch!\n"
                f"  checkpoint arch: {saved_arch}\n"
                f"  requested arch:  {expected_arch}\n"
                f"Pass --fresh to discard the incompatible checkpoint and start over."
            )
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        start_epoch = state["epoch"] + 1
        best_val = state.get("best_val", -1.0)
        print(f"Resumed from {ckpt_path} at epoch {start_epoch}")
    elif args.init_from:
        state = torch.load(args.init_from, map_location=device)
        saved_arch = state.get("arch")
        saved_arch = normalize_arch(saved_arch)
        if saved_arch != expected_arch:
            raise SystemExit(
                f"--init-from checkpoint has a different architecture:\n"
                f"  checkpoint arch: {saved_arch}\n  requested arch:  {expected_arch}"
            )
        model.load_state_dict(state["model"])
        print(
            f"Initialized weights from {args.init_from} (its epoch {state.get('epoch')}); "
            f"fine-tuning from epoch 0 with a fresh optimizer"
        )
    elif args.fresh:
        print("`--fresh` passed: starting from scratch (ignoring any existing checkpoint).")

    log_f = open(log_path, "a")

    loss_fn = nn.CrossEntropyLoss()

    gl_names, gl_params = None, None  # --log-grad-norms
    if args.log_grad_norms:
        if args.model != "conv":
            raise SystemExit("--log-grad-norms needs --model conv")
        gl = model.layer_gate_logits()
        gl_names, gl_params = [n for n, _ in gl], [p for _, p in gl]
    if args.freeze != "none":
        if args.model != "conv" or args.gumbel or args.ste or args.ste_start_epoch >= 0 or args.balance_lam > 0:
            raise SystemExit("--freeze needs --model conv and cannot be combined with --gumbel / --ste / --balance-lam")
        if args.freeze_start_epoch < 0:
            args.freeze_start_epoch = int(round(0.125 * args.epochs))
        if args.freeze_beta is None:
            args.freeze_beta = 2.0 ** (-args.freeze_k / 10000.0)
        print(
            f"progressive discretization: {args.freeze}, start epoch {args.freeze_start_epoch}, K={args.freeze_k}, "
            f"P={args.freeze_patience}, beta={args.freeze_beta:.4f}"
        )
    gum_cfg = None
    if args.gumbel:
        if args.model != "conv":
            raise SystemExit("--gumbel needs --model conv")
        if args.ste or args.ste_start_epoch >= 0:
            raise SystemExit("--gumbel cannot be combined with --ste / --ste-start-epoch")
        if not args.gumbel_tau > 0 or args.gumbel_scale < 0:
            raise SystemExit("--gumbel-tau must be > 0 and --gumbel-scale >= 0")
        gum_cfg = (args.gumbel_scale, args.gumbel_tau)
        print(f"gumbel straight-through gate sampling: scale={args.gumbel_scale}, tau={args.gumbel_tau}")
    keep = None  # balance penalty: float masks, 1 = gate is read by a later layer
    if args.balance_lam < 0:
        raise SystemExit("--balance-lam must be >= 0")
    if args.balance_lam > 0:
        if args.model != "conv" or args.skip != "none" or args.res_units > 0:
            raise SystemExit("--balance-lam needs --model conv without --skip and without --res-units")
        keep = [(~m).float().to(device) for m in model.unread_masks()]
        print(
            f"balance penalty: lam={args.balance_lam}, {int(sum(k.sum() for k in keep))} read gates of "
            f"{sum(k.numel() for k in keep)} trainable gates"
        )

    probes = []
    steps_per_epoch = len(train_loader)
    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        if args.freeze == "global" and epoch >= args.freeze_start_epoch and model.frozen_fraction() == 0.0:
            print(f"epoch {epoch}: GlobalFreeze froze {model.freeze_all()} neurons")
        sharpness = sharpness_at(epoch, args.epochs, args.sharpness_final, args.sharpness_warmup_frac)
        wd_now = weight_decay_at(
            epoch, args.epochs, args.weight_decay, args.sharpness_final, args.sharpness_warmup_frac
        )
        ste_now = ste_active(epoch, args.ste, args.ste_start_epoch)
        lr_now = lr_at(epoch, args.lr, args.ste_start_epoch, args.ste_lr)
        if ste_now and not args.ste:
            # match the fine-tune recipe that worked (run with --weight-decay 0)
            wd_now = 0.0
        for g in opt.param_groups:
            g["weight_decay"] = wd_now
            g["lr"] = lr_now
        if ste_now and not args.ste and epoch == args.ste_start_epoch:
            print(f"epoch {epoch}: switching to straight-through training (lr {lr_now})")

        model.train()
        total_loss = 0.0
        n_batches = 0
        n_correct = 0
        n_seen = 0
        pen_total = 0.0
        newly_frozen = 0
        gn_acc = torch.zeros(len(gl_params), device=device) if gl_params is not None else None
        for step_in_epoch, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            x = binarize(imgs, thresholds=thresholds, flatten=flatten)
            # gradient accumulation: same gradient as one full batch (no batch-dependent
            # layers in the model), but peak GPU memory of one micro-batch
            opt.zero_grad()
            batch_loss = 0.0
            for ci_, (xs, ys) in enumerate(zip(x.chunk(args.accum_steps), labels.chunk(args.accum_steps))):
                frac = ys.numel() / labels.numel()
                means = [] if (keep is not None and not ste_now) else None
                kw = dict(hard=False, sharpness=sharpness, ste=ste_now)
                if means is not None:
                    kw["collect"] = means
                if gum_cfg is not None:
                    kw["gumbel"] = gum_cfg
                # TaskFreeze probe on the first micro-batch only: probing every micro-batch would count 4 evaluations per probe step at --accum-steps 4
                probe_now = (
                    args.freeze == "task"
                    and ci_ == 0
                    and epoch >= args.freeze_start_epoch
                    and (epoch * steps_per_epoch + step_in_epoch) % args.freeze_k == 0
                )
                if probe_now:
                    kw["probe"] = probes
                out = model(xs, **kw)
                ce = loss_fn(out, ys) * frac
                loss = ce
                if means is not None:
                    pen = balance_penalty(means, keep)
                    loss = ce + args.balance_lam * pen * frac
                    pen_total += pen.item() * frac
                loss.backward()
                if probes:
                    newly_frozen += model.freeze_update_from_probe(probes, args.freeze_beta, args.freeze_patience)
                    probes.clear()
                batch_loss += ce.item()  # logged loss is the cross-entropy only; the penalty is logged separately
                n_correct += (out.argmax(dim=-1) == ys).sum().item()
                del out, loss
            if gn_acc is not None:
                gn_acc = gn_acc + torch.stack([p.grad.detach().norm() for p in gl_params])
            opt.step()
            total_loss += batch_loss
            n_batches += 1
            n_seen += labels.numel()

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        train_acc = n_correct / max(n_seen, 1)
        log_entry = dict(
            epoch=epoch,
            loss=avg_loss,
            train_acc=train_acc,
            sharpness=sharpness,
            weight_decay=wd_now,
            lr=lr_now,
            ste=ste_now,
            elapsed=elapsed,
        )
        if device.type == "cuda":
            log_entry["peak_mem_mb"] = round(torch.cuda.max_memory_allocated(device) / 2**20, 1)
        if args.freeze != "none":
            log_entry["frozen_fraction"] = model.frozen_fraction()
            log_entry["newly_frozen"] = newly_frozen
        if keep is not None:
            log_entry["balance_pen"] = pen_total / max(n_batches, 1)
        if gn_acc is not None:
            log_entry["grad_norm"] = dict(zip(gl_names, (gn_acc / max(n_batches, 1)).tolist()))

        msg = (
            f"epoch {epoch} loss={avg_loss:.4f} train_acc={train_acc:.4f} sharpness={sharpness:.2f} "
            f"weight_decay={wd_now:.5f} lr={lr_now:g} ste={int(ste_now)} elapsed={elapsed:.1f}s"
        )

        # also evaluate right before the straight-through phase starts, so the log
        # holds an in-run "before" number to compare the phase against
        is_eval_epoch = (
            (epoch % args.eval_every == 0)
            or (epoch == args.epochs - 1)
            or (args.ste_start_epoch is not None and epoch == args.ste_start_epoch - 1)
        )
        if is_eval_epoch:
            val_acc = evaluate(model, val_loader, thresholds, device, hard=False, sharpness=1.0, flatten=flatten)
            val_hard_acc = evaluate(model, val_loader, thresholds, device, hard=True, flatten=flatten)
            confs = model.gate_confidences()
            mean_conf = sum(confs.values()) / len(confs)
            log_entry.update(val_acc=val_acc, val_hard_acc=val_hard_acc, mean_gate_confidence=mean_conf)
            msg += f" val_acc={val_acc:.4f} val_hard_acc={val_hard_acc:.4f} gate_conf={mean_conf:.3f}"

            is_best = val_hard_acc > best_val
            if is_best:
                best_val = val_hard_acc
            state = dict(
                model=model.state_dict(),
                opt=opt.state_dict(),
                epoch=epoch,
                best_val=best_val,
                arch=expected_arch,
            )
            torch.save(state, ckpt_path)
            if is_best:
                torch.save(state, str(ckpt_path) + ".best")
        elif args.save_every_epoch:
            # checkpoint every epoch so a crash loses at most one epoch (scale M epochs are slow)
            torch.save(
                dict(
                    model=model.state_dict(), opt=opt.state_dict(), epoch=epoch, best_val=best_val, arch=expected_arch
                ),
                ckpt_path,
            )

        print(msg)
        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()

    log_f.close()

    # Final eval: prefer the best checkpoint if present.
    best_path = Path(str(ckpt_path) + ".best")
    if best_path.exists():
        state = torch.load(best_path, map_location=device)
        model.load_state_dict(state["model"])
        print(f"Loaded best checkpoint (epoch {state['epoch']}, best_val={state.get('best_val')}) for final eval.")

    soft_acc = evaluate(model, test_loader, thresholds, device, hard=False, sharpness=1.0, flatten=flatten)
    hard_acc = evaluate(model, test_loader, thresholds, device, hard=True, flatten=flatten)
    confs = model.gate_confidences()
    mean_conf = sum(confs.values()) / len(confs)
    gap = soft_acc - hard_acc
    gates = count_gates(model) if args.model == "conv" else None
    if gates:
        print(
            f"gates (discretized, paper counting): total={gates['total'] / 1e6:.3f}M "
            f"non-wire/non-constant={gates['nontrivial'] / 1e6:.3f}M"
        )
    print(
        f"FINAL test soft_acc(t=1)={soft_acc:.4f} hard_acc={hard_acc:.4f} "
        f"mean_gate_confidence={mean_conf:.3f} soft/hard gap={gap:.4f}"
    )
    with open(log_path, "a") as f:
        f.write(
            json.dumps(
                dict(
                    final=True,
                    test_soft_acc=soft_acc,
                    test_hard_acc=hard_acc,
                    gap=gap,
                    mean_gate_confidence=mean_conf,
                    ste=args.ste,
                    ste_start_epoch=args.ste_start_epoch,
                    ste_lr=args.ste_lr,
                    pool=args.pool,
                    groups=args.groups,
                    channels_per_tree=args.channels_per_tree,
                    dropout=args.dropout,
                    dropout_mode=args.dropout_mode,
                    dropout_where=args.dropout_where,
                    dropout_in_ste=args.dropout_in_ste,
                    dropout_ste_only=args.dropout_ste_only,
                    skip=args.skip,
                    skip_frac=args.skip_frac,
                    head_wiring=args.head_wiring,
                    head_layers=args.head_layers,
                    balance_lam=args.balance_lam,
                    res_units=args.res_units,
                    res_body_layers=args.res_body_layers,
                    res_tree_depth=args.res_tree_depth,
                    res_merge=args.res_merge,
                    res_stages=args.res_stages,
                    res_merge_bias=args.res_merge_bias,
                    gumbel=args.gumbel,
                    gumbel_tau=args.gumbel_tau,
                    gumbel_scale=args.gumbel_scale,
                    freeze=args.freeze,
                    freeze_start_epoch=args.freeze_start_epoch,
                    frozen_fraction=(model.frozen_fraction() if args.freeze != "none" else None),
                    augment=args.augment,
                    seed=args.seed,
                    gates=gates,
                    trainable_gates=(count_trainable_gates(model) if args.model == "conv" else None),
                    init_from=args.init_from,
                )
            )
            + "\n"
        )


if __name__ == "__main__":
    main()
