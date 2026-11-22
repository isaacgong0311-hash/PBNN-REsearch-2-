# Convolutional Differentiable Logic Gate Networks on CIFAR-10

A PyTorch reimplementation of **LogicTreeNet** (Petersen et al., *Convolutional Differentiable Logic Gate Networks*, NeurIPS 2024, [arXiv:2411.04732](https://arxiv.org/abs/2411.04732)) and the dense **DiffLogicNet** it builds on (Petersen et al., *Deep Differentiable Logic Gate Networks*, NeurIPS 2022, [arXiv:2210.08277](https://arxiv.org/abs/2210.08277)), trained on CIFAR-10 with the papers' hyperparameters.

Part of the PBNN (pure binary neural networks) project with Prof. Liping Liu, Tufts.
Isaac Gong, September–October 2026.

## Results

CIFAR-10 test set (10,000 images). "Soft" is the differentiable network. "Hard" is the same network discretized to one logic gate per neuron, which is what a pure binary network actually runs.

| Model | Gate params | Paper | Soft | Hard | Gap |
|---|---|---|---|---|---|
| DiffLogicNet, small | 768k | 51.27% | 49.65% | 49.56% | 0.1 pts |
| LogicTreeNet, scale S | 1.3M | 60.38% | 57.40% | 53.54% | 3.9 pts |
| LogicTreeNet, scale S + straight-through fine-tune | 1.3M | | 57.34% | 54.16% | 3.2 pts |
| LogicTreeNet, scale M | 10.7M | 71.01% | 62.57% | 56.45% | 6.1 pts |
| LogicTreeNet, scale M + straight-through fine-tune | 10.7M | | 61.39% | **60.43%** | **1.0 pts** |

### Scale S run history

| Run | Change | Soft | Hard | Notebook |
|---|---|---|---|---|
| 1 | Original implementation | 57.55% | 40.62% | `runs/convS_run1_original.ipynb` |
| 2 | Sharpness annealing | 49.47% | 49.87% | `runs/convS_run2_annealing.ipynb` |
| 3 | Gentler annealing (final sharpness 6) | 49.28% | 45.81% | `runs/convS_run3_gentler_annealing.ipynb` |
| 4 | Architecture fixed to match the paper | **57.40%** | **53.54%** | `runs/convS_run4_architecture_fixed.ipynb` |
| 5 | 2-channel trees (paper Sec. 3.1) | 57.39% | 46.23% | `runs/convS_run5_two_channel_trees.ipynb` |
| 6 | No weight decay | 57.22% | 47.80% | not saved |
| 7 | Control: run 4 settings, rerun | 57.69% | 50.56% | not saved |

Run 4 fixed two mismatches with the paper's Appendix A.1.1: the head is three dense layers of width 1280k, 640k and 320k (k = first conv channel count), and the residual initialization strength is z3 = 5. That closed the soft/hard gap from 17 points to 4 without any annealing.

Run 7 is the important control: identical settings to run 4 gave hard accuracy 3 points lower while soft barely moved. Hard accuracy varies about ±3 points between identical runs, so single-run differences smaller than that aren't meaningful.

### Scale M

Trained for 200 epochs: epochs 0 to 148 on Colab, then resumed from the checkpoint on Kaggle for epochs 149 to 199. Validation accuracy stayed around 63% soft / 57 to 58% hard from epoch 150 on while training loss kept falling (0.249 to 0.232), so the remaining gap to the paper is likely a difference in training setup rather than too few epochs.

## Straight-through fine-tuning

Normal training optimizes the soft network, which is thrown away after discretization. The fine-tune instead starts from a trained checkpoint and runs the **hard** network in the forward pass while gradients flow through the soft gate probabilities. Per neuron, with p the softmax over the 16 gates and sg stop-gradient:

```
w = onehot(argmax p) + p - sg(p)
```

Forward, `w` is exactly the one-hot gate, so the output equals the hard network. Backward, gradients act as if `w` were `p`. Settings: AdamW, no weight decay, eval every epoch, best epoch by validation hard accuracy.

| Model | Fine-tune | lr | Test soft | Test hard | Gap | Notebook |
|---|---|---|---|---|---|---|
| S | none (start, run 8 baseline) | – | 57.81% | 51.95% | 5.9 | |
| S | normal, 20 epochs | 0.005 | 57.69% | 50.44% | 7.3 | not saved |
| S | straight-through, 20 epochs | 0.005 | 51.66% | 54.22% | −2.6 | not saved |
| S | straight-through, 20 epochs | 0.001 | 57.41% | 54.17% | 3.2 | `runs/convS_ste_finetune_lr_sweep.ipynb` |
| S | straight-through, 20 epochs | 0.0003 | 57.34% | 54.16% | 3.2 | `runs/convS_ste_finetune_lr_sweep.ipynb` |
| M | none (start) | – | 62.57% | 56.45% | 6.1 | |
| M | straight-through, 10 epochs | 0.0003 | 61.39% | **60.43%** | **1.0** | `runs/convM_ste_finetune_kaggle.ipynb` |

- A normal fine-tune with the same budget barely moved the gates (mean confidence stayed at 0.794), so the gain isn't from extra epochs.
- On M the best epoch was 6 of 10. Training accuracy reached 96% against ~61% validation, so M overfits.
- The M starting checkpoint scores 56.45% hard, not the 57.48% of the final 200-epoch model, which wasn't saved. Before and after numbers come from the same weights.
- Still below the paper (60.38% S, 71.01% M). Soft accuracy is also below, so the remaining gap is in the training setup, not discretization.

```bash
# fine-tune an existing checkpoint with straight-through
python train.py --model conv --scale M --tree-depth 3 --kernel-size 3 --epochs 10 \
    --batch-size 128 --optimizer adamw --lr 0.0003 --weight-decay 0 \
    --thresholds 0.25,0.5,0.75 --val-split 5000 --eval-every 1 \
    --init-from ckpt/conv_M.pt --ste --ckpt ckpt/conv_M_ste.pt --log ckpt/conv_M_ste.jsonl

# not run yet: switch to straight-through partway through normal training
#   --ste-start-epoch N  [--ste-lr LR]   (see notebooks/logictreenet_S_ste_phase_colab.ipynb)
```

## Repo layout

```
difflogic.py        model code: 16 relaxed gates, LogicLayer, LogicTree, LogicTreeConv2d, OrPool2d, LogicTreeNet
train.py            CLI trainer with checkpoint/resume, architecture guard, JSONL logging
test_difflogic.py   155 checks: gate truth tables, gradient flow, conv vs. naive loop, pooling, schedules
make_notebooks.py   generates the notebooks in notebooks/ from the three source files
notebooks/          ready-to-run Colab and Kaggle notebooks (dense, conv S, conv M, M resume on Kaggle,
                    straight-through fine-tunes, noise/sharpness check, straight-through phase)
runs/               the actual training runs, with their outputs, as run at the time
```

The notebooks in `runs/` are records of past runs. Each one contains the code as it was when that run happened, so the earlier ones include the bugs described above.

## Running it

Requires Python 3.10+ and a GPU for the conv models.

```bash
pip install -r requirements.txt
python test_difflogic.py

# LogicTreeNet, scale S (paper settings)
python train.py --model conv --scale S --tree-depth 3 --kernel-size 3 \
    --epochs 200 --batch-size 128 --optimizer adamw --lr 0.02 --weight-decay 0.002 \
    --thresholds 0.25,0.5,0.75 --val-split 5000 \
    --ckpt ckpt/conv_S.pt --log ckpt/conv_S.jsonl --eval-every 5

# scale M: same command with --scale M
# dense DiffLogicNet: --model dense --scale small --optimizer adam --lr 0.01 --thresholds 0.5
```

Training resumes automatically if the `--ckpt` file exists. Checkpoints store the architecture, and `train.py` refuses to load one into a different architecture. Pass `--fresh` once to discard an incompatible checkpoint.

The easiest way to run on a free GPU is to open one of the notebooks in `notebooks/` in Colab or Kaggle. To move a run from Colab to Kaggle, upload the checkpoint from Google Drive as a Kaggle dataset and use `logictreenet_convM_kaggle.ipynb`, which finds and imports it before training.

## Implementation notes

- Each neuron's gate is a softmax over the 16 two-input Boolean functions. Every one of them is bilinear in its inputs, so the softmax-weighted mixture collapses to `k0 + k1*a + k2*b + k3*a*b` and never materializes 16 outputs.
- Wiring is fixed and random (seeded), as in the papers.
- Conv kernels are depth-3 logic-gate trees (8 inputs from a 3x3 window, 7 gates), applied with `F.unfold`.
- Pooling is logical OR: `1 - prod(1 - x)` in the soft network, max-pool in the hard one.
- Inputs are binarized at thresholds 0.25, 0.5 and 0.75 per color channel.
- CIFAR-10 is loaded from the [uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) Hugging Face mirror, falling back to torchvision. 45k train / 5k validation / 10k test; the best checkpoint is picked by validation hard accuracy.
- Optional sharpness annealing (`--sharpness-final`, off by default) is available but was not needed after the architecture fix.

## References

- Petersen, Kuehne, Borgelt, Welzel, Ermon. Convolutional Differentiable Logic Gate Networks. NeurIPS 2024. [arXiv:2411.04732](https://arxiv.org/abs/2411.04732)
- Petersen, Borgelt, Kuehne, Deussen. Deep Differentiable Logic Gate Networks. NeurIPS 2022. [arXiv:2210.08277](https://arxiv.org/abs/2210.08277)
- Authors' reference library: [Felix-Petersen/difflogic](https://github.com/Felix-Petersen/difflogic)
