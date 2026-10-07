# LP Relaxation Degeneracy — Root Cause Found (Aug 10)

## Reproduced
Ran `toy_lp_testbed.py` as-is: loss flatlines at exactly -1.0000 from epoch 0,
subgradient effectively zero from then on — matches the July 30 finding exactly.

## Root cause
**The LP never actually forces intermediate variables (z, g, h at each hidden
layer) to equal the true forward-pass values.** They're free LP decision
variables, only *loosely* linked to alpha and to the previous layer's h via
per-gate McCormick/OR/XOR inequalities. Those inequalities are only tight at
the corners of the hypercube (all-integer points) — for a relaxed multi-layer
composition, there's a large slack region where the solver can pick z/g/h
values that satisfy every local inequality *without those values ever
corresponding to a real forward pass through the actual (rounded) W, b*.

**Proof:** took the trained alpha (the one the optimizer converged to, loss
= -1.0 for all 8 samples), rounded W/b to {0,1}, and ran the *actual* binary
AND→OR→XOR forward pass by hand (not the LP relaxation) for every sample.
Result: **0 of 8 samples are correctly classified** by the real network,
despite the LP insisting every sample achieved the perfect objective value.
The LP's "-1.0" is not a real training signal — it's the solver exploiting
slack in the relaxation to trivially satisfy the objective regardless of
whether alpha encodes a working classifier.

Confirmed further: for random (untrained) alpha, the LP objective is *not*
always -1.0 — it varies per sample. But the "trivial-optimum" region of the
alpha space is large enough that gradient descent finds some alpha landing
every sample in it almost immediately, and once there, the subgradient is
exactly zero because moving alpha within that region doesn't change the
optimal LP value (still -1.0) — a flat plateau, not a fixed point at a
meaningful solution.

## Why this matches the BEP comparison (see BEP_vs_LP_and_Courbariaux_sanity_check.md)
BEP avoids this because it never poses a joint multi-variable LP at all —
each layer's problem is separable into independent 1-D sign problems, so
there's no composed slack to exploit. PBNN's formulation, by contrast, is one
joint LP per sample across *all* layers, with every intermediate gate's
output left as a free variable. The looseness compounds with depth — this is
the same failure mode known in the NN-verification literature as "convex
relaxation looseness under composition" (e.g. why naive per-neuron LP/IBP
bounds for ReLU networks get vacuous after a few layers).

## Candidate fix directions (for Isaac + Liping to pick — not decided here)
1. **Tighten via layer-by-layer sequential solving**: instead of one joint LP
   over all layers, solve layer 1's LP first, fix h^1 at its (near-integral)
   optimal value, then solve layer 2 conditioned on that — trades joint
   optimality for actually constraining the forward pass. Loses the
   "backprop through the whole LP at once" property.
2. **Add integrality/tightening cuts**: strengthen the per-gate McCormick
   envelope with valid inequalities that couple across gates (not just within
   one AND/OR/XOR), shrinking the slack region. More complex, closer to a
   real MIP relaxation.
3. **Borrow BEP's move directly**: make the relaxation separable by stripping
   the sign()/gate nonlinearity from the *objective* rather than relaxing the
   full multi-layer feasible region — i.e., don't relax "is this a valid
   forward pass," relax "which single-layer linear surrogate best matches the
   target," the way BEP does. This is a bigger formulation change, not a
   patch.
4. **Add a penalty/regularization term** discouraging the "cheat" plateau —
   e.g. penalize LP solutions where g/z/h are far from {0,1}, pushing the
   solver toward corners where the relaxation is actually tight. Cheapest to
   try, unclear if it fully closes the gap.

## Files
`code/toy_lp_testbed.py` — the file as provided by Isaac (unmodified).
Diagnostic scripts used to reproduce/prove the above were run inline, not
saved as separate files — rerun by importing `solve_and_subgrad` from
`toy_lp_testbed.py` with random alpha to see the objective vary, or by
forward-propagating the trained (rounded) alpha by hand to see it misclassify
every sample despite LP objective = -1.0.
