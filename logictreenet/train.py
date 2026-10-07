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
    DiffLogicNet, DIFFLOGICNET_CIFAR,
    LogicTreeNet, LOGICTREENET_CIFAR,
    binarize,
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
    p.add_argument("--channels-per-tree", type=int, default=0,
                   help="input channels each conv logic tree reads from; 0 = all (default). The paper says 2, "
                        "but that lowered hard accuracy by ~7 pts on scale S here")
    p.add_argument("--device", default="auto", help="auto | cuda | mps | cpu")
    p.add_argument("--ste", action="store_true",
                   help="straight-through training: forward pass uses the discretized (argmax) gates, "
                        "gradients flow through the soft gate probabilities")
    p.add_argument("--init-from", type=str, default=None,
                   help="start from this checkpoint's weights (fresh optimizer, epoch 0) when --ckpt "
                        "doesn't exist yet; used for fine-tuning")
    p.add_argument("--ste-start-epoch", type=int, default=-1,
                   help="switch to straight-through training from this epoch to the end of the run "
                        "(-1 = off). Builds the straight-through phase into normal training instead of "
                        "a separate fine-tune")
    p.add_argument("--ste-lr", type=float, default=None,
                   help="learning rate to use once the straight-through phase starts "
                        "(default: keep --lr)")
    p.add_argument("--augment", action="store_true",
                   help="random crop (pad 4) + horizontal flip on training images; not in the paper")
    args = p.parse_args()

    device = pick_device(args.device)
    print(f"device: {device}")
    thresholds = tuple(float(t) for t in args.thresholds.split(","))
    n_thresh = len(thresholds)

    train_set, val_set, test_set = load_cifar10(args.data_dir, args.val_split, thresholds, augment=args.augment)
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=256, shuffle=False, num_workers=args.num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=256, shuffle=False, num_workers=args.num_workers)

    n_classes = 10
    in_ch = 3
    in_size = 32
    flatten = (args.model == "dense")

    model = build_model(args, in_ch, in_size, n_classes, n_thresh).to(device)

    if args.optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    ckpt_path = Path(args.ckpt)
    log_path = Path(args.log)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    expected_arch = dict(
        model=args.model, scale=args.scale, tree_depth=args.tree_depth,
        kernel_size=args.kernel_size, thresholds=args.thresholds,
        channels_per_tree=(args.channels_per_tree if args.model == "conv" else 0),
    )

    start_epoch = 0
    best_val = -1.0

    if ckpt_path.exists() and not args.fresh:
        state = torch.load(ckpt_path, map_location=device)
        saved_arch = state.get("arch")
        if saved_arch is not None:
            # checkpoints from before channels_per_tree existed used all channels
            saved_arch = {**saved_arch, "channels_per_tree": saved_arch.get("channels_per_tree", 0)}
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
        if saved_arch is not None:
            saved_arch = {**saved_arch, "channels_per_tree": saved_arch.get("channels_per_tree", 0)}
        if saved_arch != expected_arch:
            raise SystemExit(
                f"--init-from checkpoint has a different architecture:\n"
                f"  checkpoint arch: {saved_arch}\n  requested arch:  {expected_arch}")
        model.load_state_dict(state["model"])
        print(f"Initialized weights from {args.init_from} (its epoch {state.get('epoch')}); "
              f"fine-tuning from epoch 0 with a fresh optimizer")
    elif args.fresh:
        print("`--fresh` passed: starting from scratch (ignoring any existing checkpoint).")

    log_f = open(log_path, "a")

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        sharpness = sharpness_at(epoch, args.epochs, args.sharpness_final, args.sharpness_warmup_frac)
        wd_now = weight_decay_at(epoch, args.epochs, args.weight_decay, args.sharpness_final, args.sharpness_warmup_frac)
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
        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            x = binarize(imgs, thresholds=thresholds, flatten=flatten)
            out = model(x, hard=False, sharpness=sharpness, ste=ste_now)
            loss = loss_fn(out, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()
            n_batches += 1
            n_correct += (out.argmax(dim=-1) == labels).sum().item()
            n_seen += labels.numel()

        avg_loss = total_loss / max(n_batches, 1)
        elapsed = time.time() - t0

        train_acc = n_correct / max(n_seen, 1)
        log_entry = dict(epoch=epoch, loss=avg_loss, train_acc=train_acc, sharpness=sharpness,
                          weight_decay=wd_now, lr=lr_now, ste=ste_now, elapsed=elapsed)

        msg = (f"epoch {epoch} loss={avg_loss:.4f} train_acc={train_acc:.4f} sharpness={sharpness:.2f} "
               f"weight_decay={wd_now:.5f} lr={lr_now:g} ste={int(ste_now)} elapsed={elapsed:.1f}s")

        # also evaluate right before the straight-through phase starts, so the log
        # holds an in-run "before" number to compare the phase against
        is_eval_epoch = ((epoch % args.eval_every == 0) or (epoch == args.epochs - 1)
                         or (args.ste_start_epoch is not None and epoch == args.ste_start_epoch - 1))
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
                model=model.state_dict(), opt=opt.state_dict(), epoch=epoch,
                best_val=best_val, arch=expected_arch,
            )
            torch.save(state, ckpt_path)
            if is_best:
                torch.save(state, str(ckpt_path) + ".best")

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
    print(f"FINAL test soft_acc(t=1)={soft_acc:.4f} hard_acc={hard_acc:.4f} "
          f"mean_gate_confidence={mean_conf:.3f} soft/hard gap={gap:.4f}")
    with open(log_path, "a") as f:
        f.write(json.dumps(dict(final=True, test_soft_acc=soft_acc, test_hard_acc=hard_acc,
                                gap=gap, mean_gate_confidence=mean_conf, ste=args.ste,
                                ste_start_epoch=args.ste_start_epoch, ste_lr=args.ste_lr,
                                init_from=args.init_from)) + "\n")


if __name__ == "__main__":
    main()