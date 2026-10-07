"""
Entropy-penalized version of the LP relaxation testbed.

Diagnosis this fix targets (see toy_lp_testbed.py / conversation with Liping
& Alex): the plain LP relaxation is degenerate. For alpha in the interior of
[0,1]^|alpha|, the LP is free to pick interior (non-corner) values for the
hidden auxiliary variables (z^t, g^t, h^t, t < T) that satisfy the local
McCormick/OR/XOR inequalities without being tied to the real binary forward
pass computed from x. That decouples the optimal LP value from both x and
alpha almost everywhere, giving a subgradient of exactly zero.

Fix tried here: add a concave "confidence" penalty
    lam * sum_i v_i * (1 - v_i)
over every auxiliary variable v_i (all z, g, h for layers 1..T -- not the
fixed h^0 = x, and not excluding h^T). v(1-v) is 0 at the corners (v=0 or 1)
and maximal at v=0.5, so *minimizing* c^T v + lam * sum v(1-v) rewards the
LP for pushing auxiliary variables toward the corners, where the relaxation
is exact and where the real dependence on x/alpha actually lives.

v(1-v) is concave, so this is no longer a plain LP. We use the standard fix
for a concave penalty over a polytope: the concave-convex procedure (CCCP /
DC programming). At each inner iteration we linearize the penalty around the
previous solution v_anchor:
    v(1-v) ~= v_anchor(1-v_anchor) + (1 - 2*v_anchor)(v - v_anchor)
so the linear term added to the objective coefficient of v_i is
    lam * (1 - 2*v_anchor_i)
We resolve the LP with this updated linear objective, use the new solution
as the next anchor, and repeat for a fixed number of inner iterations. Each
inner solve is a legitimate LP (linprog), and CCCP guarantees the objective
is non-increasing across inner iterations (standard DC-programming result).

CAVEAT (flag this to Liping/Alex, don't gloss over it): the subgradient
below is read from the *final* inner-iteration LP's dual values, treating
the converged v_anchor as fixed w.r.t. alpha. That ignores the fact that
v_anchor itself depends on alpha through the CCCP fixed point -- so this is
an approximate/local subgradient, not an exact one. A fully rigorous version
would need implicit differentiation through the fixed point, or unrolling
the inner loop and backpropagating through it.
"""

import numpy as np
from scipy.optimize import linprog

from toy_lp_testbed import AlphaLayout, build_lp

rng = np.random.default_rng(0)


def auxiliary_var_indices(layout, dims):
    """All LP variable indices except h^0 (fixed = x): z, g, h for t=1..T."""
    idxs = []
    T = layout.T
    for t in range(1, T + 1):
        dt, dtm1 = dims[t], dims[t - 1]
        for i in range(dt):
            for j in range(dtm1):
                idxs.append(layout.z_idx(t, i, j))
            idxs.append(layout.g_idx(t, i))
            idxs.append(layout.h_idx(t, i))
    return np.array(idxs)


def solve_and_subgrad_entropy(dims, alpha_layout, alpha, x, u, lam=0.5, n_inner=4):
    W, b = alpha_layout.unpack(alpha)
    lp = build_lp(dims, alpha_layout, W, b, x, u)
    layout = lp["layout"]
    aux_idx = auxiliary_var_indices(layout, dims)

    c_base = lp["c"].copy()
    v_anchor = np.full(layout.n_vars, 0.5)  # start at max-uncertainty prior

    res = None
    for _ in range(n_inner):
        c = c_base.copy()
        c[aux_idx] += lam * (1.0 - 2.0 * v_anchor[aux_idx])
        res = linprog(c, A_ub=lp["A_ub"], b_ub=lp["b_ub"],
                       A_eq=lp["A_eq"], b_eq=lp["b_eq"], bounds=lp["bounds"],
                       method="highs")
        if not res.success:
            raise RuntimeError(f"LP failed: {res.message}")
        v_anchor = res.x

    # Report the *original* task loss (c_base^T v), not the penalized
    # objective, so it's directly comparable to the plain-LP run.
    val = float(c_base @ res.x)
    marginals = res.ineqlin.marginals
    subgrad = lp["M"].T @ marginals
    return val, subgrad


def main():
    dims = [3, 3, 2]  # d0, d1(hidden), d2=T output/classes
    alpha_layout = AlphaLayout(dims)

    xs = np.array([[int(b) for b in format(k, "03b")] for k in range(8)])
    ys = (xs.sum(axis=1) >= 2).astype(int)

    def u_from_y(y, n_classes=2):
        u = np.ones(n_classes)
        u[y] = -1.0
        return u

    alpha = rng.uniform(0.3, 0.7, size=alpha_layout.n_params)

    lr = 0.3
    lam = 0.5
    n_inner = 4
    n_epochs = 40
    print("epoch  avg_loss")
    for epoch in range(n_epochs):
        idx = rng.permutation(len(xs))
        total_loss = 0.0
        grad_sum = np.zeros_like(alpha)
        for k in idx:
            x, y = xs[k], ys[k]
            u = u_from_y(y)
            val, g = solve_and_subgrad_entropy(
                dims, alpha_layout, alpha, x, u, lam=lam, n_inner=n_inner
            )
            total_loss += val
            grad_sum += g
        alpha = alpha - lr * (grad_sum / len(xs))
        alpha = np.clip(alpha, 0.0, 1.0)  # project back into the box
        avg_loss = total_loss / len(xs)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            print(f"{epoch:5d}  {avg_loss:.4f}")

    print("\nFinal alpha (rounded to nearest binary weights):")
    W, b = alpha_layout.unpack(alpha)
    for t in W:
        print(f"W{t} =\n{np.round(W[t])}")
        print(f"b{t} = {np.round(b[t])}")


if __name__ == "__main__":
    main()
