#!/usr/bin/env bash
# Runs everything from this work session in order:
#   1) the original (degenerate) LP testbed
#   2) the entropy-penalty fix
#   3) a quick sanity check of one sample's LP (McCormick tightness + OR-reduce slack)
set -e

pip3 install -q scipy numpy

echo "=============================================="
echo "1) ORIGINAL TESTBED (should be flat at -1.0000)"
echo "=============================================="
python3 toy_lp_testbed.py

echo
echo "=============================================="
echo "2) ENTROPY-PENALTY FIX (loss should move)"
echo "=============================================="
python3 toy_lp_testbed_entropy.py

echo
echo "=============================================="
echo "3) SANITY CHECK: McCormick tightness + OR-reduce slack on one sample"
echo "=============================================="
python3 -c "
import numpy as np
from toy_lp_testbed import AlphaLayout, build_lp
from scipy.optimize import linprog

dims = [3, 3, 2]
alpha_layout = AlphaLayout(dims)
rng = np.random.default_rng(0)
alpha = rng.uniform(0.3, 0.7, size=alpha_layout.n_params)
W, b = alpha_layout.unpack(alpha)
x = np.array([1, 0, 1])

lp = build_lp(dims, alpha_layout, W, b, x, np.array([-1.0, 1.0]))
layout = lp['layout']
res = linprog(lp['c'], A_ub=lp['A_ub'], b_ub=lp['b_ub'],
               A_eq=lp['A_eq'], b_eq=lp['b_eq'], bounds=lp['bounds'], method='highs')

print('x =', x)
for i in range(3):
    zrow = [res.x[layout.z_idx(1, i, j)] for j in range(3)]
    g1i = res.x[layout.g_idx(1, i)]
    h1i = res.x[layout.h_idx(1, i)]
    print(f'neuron {i}: z={[round(float(v),3) for v in zrow]}  '
          f'g1={g1i:.3f} (max(z)={max(zrow):.3f}, sum(z)={sum(zrow):.3f})  '
          f'h1={h1i:.3f}  b1={b[1][i]:.3f}')
"

echo
echo "Done."
