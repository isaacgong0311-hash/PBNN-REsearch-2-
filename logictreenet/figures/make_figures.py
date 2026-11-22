"""Redraw the three figures of the PBNN meeting script. Numbers are hard test accuracy, scale S or M, crop/flip.
Run: python make_figures.py   (needs numpy, matplotlib). Writes figA_*.png, figB_*.png, figC_*.png next to this file."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).resolve().parent

BLUE, GREY, ORANGE, RED = "#1f5fa8", "#7f7f7f", "#d9731a", "#b2182b"
# Figure A: scale S, mean and SD over 5 seeds
arms = [
    ("No dropout (control)", 57.83, 0.61, GREY),
    ("Permutation dropout p=0.05,\nsoft training", 56.01, 1.40, RED),
    ("Permutation dropout p=0.1,\nsoft training", 52.01, 1.03, RED),
    ("Permutation dropout p=0.2,\nsoft training", 46.39, 0.81, RED),
    ("Binary phase, no dropout", 58.25, 0.49, BLUE),
    ("Binary phase, exact-count\npermutation dropout p=0.1", 57.92, 0.36, BLUE),
    ("Binary phase, exact-count\npermutation dropout p=0.2", 58.31, 0.32, BLUE),
]
fig, ax = plt.subplots(figsize=(7.2, 4.4))
y = np.arange(len(arms))[::-1]
for yi, (n, m, s, c) in zip(y, arms):
    ax.errorbar(m, yi, xerr=s, fmt="o", color=c, capsize=3, ms=6)
ax.axvline(60.38, color="k", ls="--", lw=1)
ax.text(60.38, y[0] + 0.55, "paper 60.38", ha="center", fontsize=7)
ax.set_yticks(y)
ax.set_yticklabels([a[0] for a in arms])
ax.set_xlim(44, 62)
ax.set_xlabel("Hard (fully binary) test accuracy, % (mean +- 1 SD of 5 seeds)")
ax.set_title("Permutation dropout hurts in soft training and is neutral in the binary phase", loc="left")
fig.tight_layout()
fig.savefig(OUT / "figA_permutation_dropout.png", dpi=200)
# Figure B: Gumbel starting bias, 3 seeds
b = [5, 7, 10, 15, 20]
m = [17.39, 31.28, 40.03, 41.98, 43.50]
s = [5.16, 0.39, 1.00, 0.91, 0.37]
fig, ax = plt.subplots(figsize=(5.6, 3.8))
ax.errorbar(b, m, yerr=s, fmt="-o", color=RED, capsize=3, label="With Gumbel noise")
ax.errorbar([10], [57.87], yerr=[0.44], fmt="s", color=BLUE, capsize=3, label="Same bias, no Gumbel noise")
ax.axhline(57.83, color=GREY, lw=1, ls=":")
ax.text(3.7, 57.4, "control 57.8", fontsize=7, va="top", color=GREY)
ax.axhline(60.38, color="k", lw=1, ls="--")
ax.text(3.7, 60.8, "paper 60.4", fontsize=7, va="bottom")
ax.set_xlabel("Starting bias of the pass-through gate")
ax.set_ylabel("Hard test accuracy, %")
ax.set_title("A larger starting bias helps Gumbel training but does not fix it", loc="left")
ax.set_xlim(3.5, 22)
ax.set_ylim(10, 64)
ax.legend(frameon=False, loc="lower right")
fig.tight_layout()
fig.savefig(OUT / "figB_gumbel_bias.png", dpi=200)
# Figure C: scale M, per-seed hard accuracy
d200 = {
    "Baseline": [70.53, 69.85, 70.14, 69.80, 70.27, 70.44],
    "Balanced head\nwiring": [70.95, 71.54, 70.76, 69.91, 70.77, 70.47, 71.17],
    "Balanced wiring\n+ TaskFreeze": [70.55, 70.17, 69.60],
}
d400 = {"Baseline": [70.76, 70.05, 70.67], "Balanced head\nwiring": [70.56, 70.36, 70.14]}
cols = {"Baseline": GREY, "Balanced head\nwiring": BLUE, "Balanced wiring\n+ TaskFreeze": ORANGE}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 4.0), sharey=True, gridspec_kw=dict(width_ratios=[3, 2], wspace=0.06))
rng = np.random.default_rng(0)
for ax, dd, t in [(a1, d200, "200 epochs"), (a2, d400, "400 epochs")]:
    for i, (k, v) in enumerate(dd.items()):
        ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, color=cols[k], s=22, zorder=3)
        ax.plot([i - 0.25, i + 0.25], [np.mean(v)] * 2, color=cols[k], lw=2.2, zorder=4)
        ax.text(i, 68.95, f"n={len(v)}", ha="center", fontsize=7, color=cols[k])
    ax.set_xticks(range(len(dd)))
    ax.set_xticklabels(list(dd.keys()))
    ax.set_title(t, loc="left")
    ax.axhline(71.01, color="k", ls="--", lw=1)
a1.text(2.45, 71.06, "paper 71.01", ha="right", fontsize=7, va="bottom")
a1.set_ylabel("Hard test accuracy, %")
a1.set_ylim(68.8, 71.8)
fig.suptitle("Scale M: balanced wiring gains +0.6 at 200 epochs, not at 400 epochs", x=0.01, ha="left", fontsize=9)
fig.subplots_adjust(top=0.84, bottom=0.2, left=0.1, right=0.98)
fig.savefig(OUT / "figC_scaleM.png", dpi=200)
