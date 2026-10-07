"""Generates the Colab/Kaggle .ipynb notebooks for DiffLogicNet (dense) and
LogicTreeNet (conv), scales S and M, by embedding the contents of
difflogic.py / test_difflogic.py / train.py as %%writefile cells.

Run: python make_notebooks.py
"""

import json
from pathlib import Path

HERE = Path(__file__).parent

DIFFLOGIC_SRC = (HERE / "difflogic.py").read_text()
TEST_SRC = (HERE / "test_difflogic.py").read_text()
TRAIN_SRC = (HERE / "train.py").read_text()


# ---------------------------------------------------------------------------
# notebook-building helpers
# ---------------------------------------------------------------------------

def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text):
    return {
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.splitlines(keepends=True),
    }


def writefile_cell(filename, content):
    return code(f"%%writefile {filename}\n" + content)


def nb(cells, platform="colab"):
    metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    if platform == "colab":
        metadata["accelerator"] = "GPU"
        metadata["colab"] = {"provenance": [], "gpuType": "T4"}
    return {
        "cells": cells,
        "metadata": metadata,
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def write_nb(path, cells, platform="colab"):
    import nbformat
    notebook = nbformat.from_dict(nb(cells, platform=platform))
    nbformat.validator.normalize(notebook)
    nbformat.validate(notebook)
    (HERE / "notebooks").mkdir(exist_ok=True)
    (HERE / "notebooks" / path).write_text(nbformat.writes(notebook))
    print(f"wrote notebooks/{path} ({len(cells)} cells)")


# ---------------------------------------------------------------------------
# shared snippets
# ---------------------------------------------------------------------------

SETUP_MD = """## Setup

Installs the `datasets` library (used to fetch CIFAR-10 from a fast HuggingFace mirror
instead of torchvision's default, often-slow download source)."""

SETUP_MD_KAGGLE = SETUP_MD + """

**Before running:** in the notebook's right sidebar, under *Settings*, toggle **Internet: On**
(off by default on Kaggle — required for `pip install` and the HuggingFace download) and
select a **GPU** accelerator."""

SETUP_CODE = "!pip install -q datasets\n"

TEST_CODE = "!python test_difflogic.py"

DRIVE_MD = """## Checkpoint storage (Google Drive)

Mounts Google Drive so checkpoints/logs survive across Colab sessions. If the mount fails
for any reason, falls back to local (ephemeral) disk automatically rather than hard-failing."""

DRIVE_CODE = """import os

CKPT_DIR = "/content/ckpt"
try:
    from google.colab import drive
    drive.mount("/content/drive")
    CKPT_DIR = "/content/drive/MyDrive/pbnn_checkpoints"
except Exception as e:
    print(f"Drive mount unavailable ({e}); using local disk instead.")

os.makedirs(CKPT_DIR, exist_ok=True)
print("checkpoints ->", CKPT_DIR)
"""

DRIVE_MD_KAGGLE = """## Checkpoint storage (Kaggle working directory)

Kaggle notebooks have no `google.colab.drive` — checkpoints go to `/kaggle/working`,
which persists only for the interactive session unless you click
**Save Version -> Save & Run All (Commit)** before you close the tab."""

DRIVE_CODE_KAGGLE = """import os

CKPT_DIR = "/kaggle/working/pbnn_checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)
print("checkpoints ->", CKPT_DIR)
"""

RESULTS_CODE = """import json
import matplotlib.pyplot as plt

epochs, losses, val_accs, val_hard_accs, gate_confs = [], [], [], [], []
with open(LOG_PATH) as f:
    for line in f:
        row = json.loads(line)
        epochs.append(row["epoch"])
        losses.append(row["loss"])
        if "val_acc" in row:
            val_accs.append((row["epoch"], row["val_acc"]))
            val_hard_accs.append((row["epoch"], row["val_hard_acc"]))
            gate_confs.append((row["epoch"], row["mean_gate_confidence"]))

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(epochs, losses)
axes[0].set_title("train loss")
axes[0].set_xlabel("epoch")

if val_accs:
    xs, ys = zip(*val_accs)
    _, ys_hard = zip(*val_hard_accs)
    axes[1].plot(xs, ys, label="soft")
    axes[1].plot(xs, ys_hard, label="hard")
    axes[1].set_title("val accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].legend()

    xs, ys_conf = zip(*gate_confs)
    axes[2].plot(xs, ys_conf)
    axes[2].set_title("mean gate confidence")
    axes[2].set_xlabel("epoch")

plt.tight_layout()
plt.show()

if val_accs:
    last_soft = val_accs[-1][1]
    last_hard = val_hard_accs[-1][1]
    print(f"latest val soft_acc={last_soft:.4f} hard_acc={last_hard:.4f} gap={last_soft - last_hard:.4f}")
"""


def notes_cell(scale_tag):
    return md(f"""## Notes

- If the soft/hard accuracy gap comes out large after training, `--sharpness-final` and
  `--sharpness-warmup-frac` (both present in `train.py`, off by default) can anneal the
  gate-choice softmax toward its own hard/argmax behavior late in training. This is *not*
  part of the paper's own protocol — keep it as a fallback, not a default, unless the gap
  is actually a problem at full scale.
- `--fresh` discards an existing (or incompatible) checkpoint and starts over. Use it
  **once** after any architecture-breaking code change, then remove it from the command —
  leaving it in wipes progress back to epoch 0 on every re-run.
{scale_tag}""")


# ---------------------------------------------------------------------------
# 1) Dense DiffLogicNet, Colab
# ---------------------------------------------------------------------------

dense_cells = [
    md("""# DiffLogicNet (dense) on CIFAR-10 — Colab

Dense Differentiable Logic Gate Network (Petersen et al., NeurIPS 2022, arXiv:2210.08277),
trained with the paper's real parameters (not a toy config)."""),
    md(SETUP_MD), code(SETUP_CODE),
    md("## Source files"),
    writefile_cell("difflogic.py", DIFFLOGIC_SRC),
    writefile_cell("test_difflogic.py", TEST_SRC),
    writefile_cell("train.py", TRAIN_SRC),
    md("## Run tests"), code(TEST_CODE),
    md(DRIVE_MD), code(DRIVE_CODE),
    md("## Train\n\nScale: `small` (width=12000, depth=4) — paper-reported 51.27% test accuracy."),
    code("""CKPT_PATH = f"{CKPT_DIR}/dense_small_ckpt.pt"
LOG_PATH = f"{CKPT_DIR}/dense_small_log.jsonl"

!python train.py --model dense --scale small \\
    --epochs 200 --batch-size 128 --optimizer adam --lr 0.01 \\
    --thresholds 0.5 --val-split 5000 \\
    --ckpt "{CKPT_PATH}" --log "{LOG_PATH}" --eval-every 5
    # add --fresh above (after --model dense) to discard an incompatible old checkpoint
"""),
    md("## Results"), code(RESULTS_CODE),
    notes_cell(""),
]
write_nb("difflogicnet_dense_colab.ipynb", dense_cells, platform="colab")


# ---------------------------------------------------------------------------
# 2) LogicTreeNet scale S, Colab
# ---------------------------------------------------------------------------

CONV_INTRO_S = """# LogicTreeNet (convolutional LGN) scale S on CIFAR-10 — Colab

Convolutional Differentiable Logic Gate Network (Petersen et al., NeurIPS 2024,
arXiv:2411.04732), scale S, trained with the paper's real parameters.

**Architecture spec (paper Table 6 / Appendix A.1.1):**

| | |
|---|---|
| channels | (32, 128, 512, 1024) |
| tree depth | 3 (7 gates per tree, 8 leaves) |
| kernel size | 3x3 |
| pooling | logical OR-pool, 2x2 |
| head | 3 dense LogicLayers, widths 1280k / 640k / 320k (k=32) |
| residual init | z3=5 (~90% initial probability on pass-through gate) |
| tau | 20.0 |
| input | 3 threshold-binarized channels (0.25, 0.5, 0.75) |
| paper-reported test accuracy | 60.38% |

**Note on a prior bug:** an earlier version of this notebook used a uniform head width
repeated 3 times and a weaker residual-init bias (z3=3). Both are now fixed to match the
paper exactly (head taper 1280k/640k/320k, z3=5). Checkpoints saved by the old code are
architecturally incompatible — use `--fresh` once if resuming from one."""

conv_cells = [
    md(CONV_INTRO_S),
    md(SETUP_MD), code(SETUP_CODE),
    md("## Source files"),
    writefile_cell("difflogic.py", DIFFLOGIC_SRC),
    writefile_cell("test_difflogic.py", TEST_SRC),
    writefile_cell("train.py", TRAIN_SRC),
    md("## Run tests"), code(TEST_CODE),
    md(DRIVE_MD), code(DRIVE_CODE),
    md("## Train"),
    code("""CKPT_PATH = f"{CKPT_DIR}/conv_S_ckpt.pt"
LOG_PATH = f"{CKPT_DIR}/conv_S_log.jsonl"

!python train.py --model conv --scale S --tree-depth 3 --kernel-size 3 \\
    --epochs 200 --batch-size 128 --optimizer adamw --lr 0.02 --weight-decay 0.002 \\
    --thresholds 0.25,0.5,0.75 --val-split 5000 \\
    --ckpt "{CKPT_PATH}" --log "{LOG_PATH}" --eval-every 5
    # add --fresh above (after --model conv) to discard an incompatible old checkpoint
"""),
    md("## Results"), code(RESULTS_CODE),
    notes_cell("- Next step once this validates: scale M (`logictreenet_convM_colab.ipynb`), "
               "which the paper reports at 71.01% vs. S's 60.38%."),
]
write_nb("logictreenet_conv_colab.ipynb", conv_cells, platform="colab")


# ---------------------------------------------------------------------------
# 3) LogicTreeNet scale S, Kaggle
# ---------------------------------------------------------------------------

conv_cells_kaggle = [
    md(CONV_INTRO_S.replace("— Colab", "— Kaggle")),
    md(SETUP_MD_KAGGLE), code(SETUP_CODE),
    md("## Source files"),
    writefile_cell("difflogic.py", DIFFLOGIC_SRC),
    writefile_cell("test_difflogic.py", TEST_SRC),
    writefile_cell("train.py", TRAIN_SRC),
    md("## Run tests"), code(TEST_CODE),
    md(DRIVE_MD_KAGGLE), code(DRIVE_CODE_KAGGLE),
    md("## Train"),
    code("""CKPT_PATH = f"{CKPT_DIR}/conv_S_ckpt.pt"
LOG_PATH = f"{CKPT_DIR}/conv_S_log.jsonl"

!python train.py --model conv --scale S --tree-depth 3 --kernel-size 3 \\
    --epochs 200 --batch-size 128 --optimizer adamw --lr 0.02 --weight-decay 0.002 \\
    --thresholds 0.25,0.5,0.75 --val-split 5000 \\
    --ckpt "{CKPT_PATH}" --log "{LOG_PATH}" --eval-every 5
    # add --fresh above (after --model conv) to discard an incompatible old checkpoint
"""),
    md("## Results"), code(RESULTS_CODE),
    notes_cell("- Kaggle GPU sessions have a runtime limit (~12h) — commit a version "
               "(**Save Version -> Save & Run All**) before it expires so `/kaggle/working` "
               "checkpoints aren't lost."),
]
write_nb("logictreenet_conv_kaggle.ipynb", conv_cells_kaggle, platform="kaggle")


# ---------------------------------------------------------------------------
# 4) LogicTreeNet scale M, Colab
# ---------------------------------------------------------------------------

CONV_INTRO_M = """# LogicTreeNet (convolutional LGN) scale M on CIFAR-10 — Colab

Same architecture as the scale-S notebook, scaled up (k=256 instead of k=32).

**Architecture spec (paper Table 6 / Appendix A.1.1):**

| | |
|---|---|
| channels | (256, 1024, 4096, 8192) |
| tree depth | 3 |
| kernel size | 3x3 |
| pooling | logical OR-pool, 2x2 |
| head | 3 dense LogicLayers, widths 1280k / 640k / 320k (k=256) |
| residual init | z3=5 |
| tau | 40.0 |
| input | 3 threshold-binarized channels (0.25, 0.5, 0.75) |
| paper-reported test accuracy | 71.01% |

Uses a **separate checkpoint path** (`conv_M_ckpt.pt` / `conv_M_log.jsonl`) from the
scale-S notebook so the two runs don't collide in the same Drive folder."""

conv_cells_M = [
    md(CONV_INTRO_M),
    md(SETUP_MD), code(SETUP_CODE),
    md("## Source files"),
    writefile_cell("difflogic.py", DIFFLOGIC_SRC),
    writefile_cell("test_difflogic.py", TEST_SRC),
    writefile_cell("train.py", TRAIN_SRC),
    md("## Run tests"), code(TEST_CODE),
    md(DRIVE_MD), code(DRIVE_CODE),
    md("## Train"),
    code("""CKPT_PATH = f"{CKPT_DIR}/conv_M_ckpt.pt"
LOG_PATH = f"{CKPT_DIR}/conv_M_log.jsonl"

!python train.py --model conv --scale M --tree-depth 3 --kernel-size 3 \\
    --epochs 200 --batch-size 128 --optimizer adamw --lr 0.02 --weight-decay 0.002 \\
    --thresholds 0.25,0.5,0.75 --val-split 5000 \\
    --ckpt "{CKPT_PATH}" --log "{LOG_PATH}" --eval-every 5
    # add --fresh above (after --model conv) to discard an incompatible old checkpoint
"""),
    md("## Results"), code(RESULTS_CODE),
    notes_cell("- Scale M is ~8x the parameter count of scale S — expect each epoch to take "
               "noticeably longer; budget for multiple Colab sessions and resume via the "
               "Drive checkpoint between them."),
]
write_nb("logictreenet_convM_colab.ipynb", conv_cells_M, platform="colab")


# ---------------------------------------------------------------------------
# 5) LogicTreeNet scale M, Kaggle — RESUME from a Colab checkpoint uploaded
#    as a Kaggle Dataset (new: Kaggle and Colab Drive are separate storage,
#    so this notebook imports the checkpoint before training starts).
# ---------------------------------------------------------------------------

CONV_INTRO_M_KAGGLE = """# LogicTreeNet (convolutional LGN) scale M on CIFAR-10 — Kaggle (resume)

Continues an in-progress scale-M run that was started on **Colab** and ran out of compute
there (e.g. interrupted at epoch 148/200). Kaggle and Colab's Google Drive are **separate
storage systems** — a fresh Kaggle notebook cannot see a checkpoint sitting in Colab's
mounted Drive, so this notebook imports one manually via a Kaggle Dataset.

**Architecture spec — must match the original Colab M-run exactly, or the checkpoint
will be refused as incompatible (this is enforced by `train.py`'s own arch guard):**

| | |
|---|---|
| model / scale | conv / M |
| channels | (256, 1024, 4096, 8192) |
| tree depth | 3 |
| kernel size | 3x3 |
| thresholds | 0.25, 0.5, 0.75 |
| tau | 40.0 |

## One-time setup (do this before running this notebook)

1. In your **Colab** notebook, locate the checkpoint in Google Drive:
   `MyDrive/pbnn_checkpoints/conv_M_ckpt.pt` (and, if present, `conv_M_ckpt.pt.best` and
   `conv_M_log.jsonl` — bring those along too so the log/plot picks up where it left off).
2. Download those file(s) to your computer.
3. On Kaggle, click **+ New Dataset**, upload the downloaded file(s), and give the dataset
   a name (e.g. `pbnn-conv-m-ckpt`). Note the dataset's slug — Kaggle will mount it at
   `/kaggle/input/<slug>/`.
4. In **this** notebook's right sidebar, under *Input*, click **Add Input** and attach that
   dataset. Also toggle **Internet: On** and pick a **GPU** accelerator.
5. Set `CKPT_DATASET_DIR` in the cell below to match the mounted path (the default guess is
   `/kaggle/input/pbnn-conv-m-ckpt` — check the *Input* panel for the exact slug Kaggle
   assigned and adjust if it differs)."""

CKPT_IMPORT_CODE = """import glob
import os
import shutil

# Adjust this to match the dataset you attached under "Input" (see the sidebar for the
# exact mounted path/slug Kaggle assigned).
CKPT_DATASET_DIR = "/kaggle/input/pbnn-conv-m-ckpt"

CKPT_PATH = f"{CKPT_DIR}/conv_M_ckpt.pt"
LOG_PATH = f"{CKPT_DIR}/conv_M_log.jsonl"

if not os.path.isdir(CKPT_DATASET_DIR):
    # Kaggle sometimes mounts datasets under /kaggle/input/datasets/<user>/<slug>;
    # fall back to searching all of /kaggle/input for the checkpoint.
    hits = glob.glob("/kaggle/input/**/conv_M_ckpt.pt", recursive=True)
    if hits:
        CKPT_DATASET_DIR = os.path.dirname(hits[0])
        print(f"found checkpoint dataset at {CKPT_DATASET_DIR}")

if os.path.isdir(CKPT_DATASET_DIR):
    found_any = False
    for fname in ["conv_M_ckpt.pt", "conv_M_ckpt.pt.best", "conv_M_log.jsonl"]:
        matches = glob.glob(os.path.join(CKPT_DATASET_DIR, "**", fname), recursive=True)
        if matches:
            src = matches[0]
            dst = f"{CKPT_DIR}/{fname}"
            shutil.copy(src, dst)
            print(f"imported {src} -> {dst}")
            found_any = True
    if not found_any:
        print(f"WARNING: no checkpoint files found under {CKPT_DATASET_DIR}. "
              f"Check the dataset is attached and CKPT_DATASET_DIR matches its mounted path "
              f"(see the Input panel in the sidebar). Training will start from scratch "
              f"(epoch 0) if nothing is imported.")
else:
    print(f"WARNING: {CKPT_DATASET_DIR} does not exist. "
          f"Attach the uploaded checkpoint dataset under Input, then update "
          f"CKPT_DATASET_DIR above to match its mounted path. "
          f"Training will start from scratch (epoch 0) unless a checkpoint is imported "
          f"into {CKPT_DIR} before the training cell runs.")

if os.path.exists(CKPT_PATH):
    print(f"Checkpoint present at {CKPT_PATH} -- train.py will resume from it automatically "
          f"(as long as the architecture flags below match what produced it).")
else:
    print(f"No checkpoint at {CKPT_PATH} -- the training cell will start a fresh run at epoch 0.")
"""

conv_cells_M_kaggle_resume = [
    md(CONV_INTRO_M_KAGGLE),
    md(SETUP_MD_KAGGLE), code(SETUP_CODE),
    md("## Source files"),
    writefile_cell("difflogic.py", DIFFLOGIC_SRC),
    writefile_cell("test_difflogic.py", TEST_SRC),
    writefile_cell("train.py", TRAIN_SRC),
    md("## Run tests"), code(TEST_CODE),
    md(DRIVE_MD_KAGGLE), code(DRIVE_CODE_KAGGLE),
    md("""## Import the Colab checkpoint

Copies the checkpoint (and log, if uploaded) from the attached Kaggle Dataset into
`CKPT_DIR`, where `train.py`'s existing resume-by-`ckpt_path.exists()` logic will pick it
up naturally — no code changes to `train.py` needed. If the arch flags below don't exactly
match what produced the checkpoint, `train.py`'s own arch-mismatch guard will refuse to
resume (with a clear error) rather than silently loading incompatible weights."""),
    code(CKPT_IMPORT_CODE),
    md("## Train\n\nResumes automatically from the imported checkpoint (e.g. continuing "
       "from epoch 148/200) — do **not** pass `--fresh` here, since that would discard "
       "the imported progress."),
    code("""!python train.py --model conv --scale M --tree-depth 3 --kernel-size 3 \\
    --epochs 200 --batch-size 128 --optimizer adamw --lr 0.02 --weight-decay 0.002 \\
    --thresholds 0.25,0.5,0.75 --val-split 5000 \\
    --ckpt "{CKPT_PATH}" --log "{LOG_PATH}" --eval-every 5
    # do NOT add --fresh here -- it would discard the imported checkpoint and restart at epoch 0
"""),
    md("## Results"), code(RESULTS_CODE),
    md("""## When you're done

Checkpoints/logs are in `/kaggle/working/pbnn_checkpoints` — click **Save Version ->
Save & Run All (Commit)** before closing the tab, or download `conv_M_ckpt.pt` /
`conv_M_ckpt.pt.best` from the *Output* panel so you can resume again later (on Kaggle or
back on Colab, following the same import steps)."""),
    notes_cell("- Kaggle GPU sessions have a runtime limit (~12h)."),
]
write_nb("logictreenet_convM_kaggle.ipynb", conv_cells_M_kaggle_resume, platform="kaggle")

print("\nAll notebooks generated.")
