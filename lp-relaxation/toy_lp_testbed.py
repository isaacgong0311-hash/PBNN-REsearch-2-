"""
Toy testbed for the PBNN LP-relaxation training method (documentation.pdf).

Goal: validate, on a tiny architecture, that
  1) the per-sample LP relaxation (McCormick envelope for AND, OR-reduce
     bounds, XOR linearization) is well-posed and solvable, and
  2) the subgradient of the LP value w.r.t. alpha = (W^1..W^T, b^1..b^T),
     obtained via LP sensitivity (dual/marginal values), actually drives
     the training loss down when used in projected subgradient descent.

Architecture: d = [d0, d1, ..., dT] fully-binary feedforward network, as
defined in documentation.pdf. Kept small (e.g. [3, 3, 2]) so each per-sample
LP solves in milliseconds.

NOTE on u_j: the doc leaves u_j (the loss weight derived from label y) as a
TODO. Here we use a simple placeholder: u_j = -1 for the true class j = y,
u_j = +1 for all other classes. Since the LP minimizes sum_j u_j h^T_j, this
pushes h^T_y toward 1 and h^T_{j != y} toward 0 -- a crude margin-style loss.
This should be replaced once u_j is properly defined; it's only here to make
the testbed runnable end to end.
"""

import numpy as np
from scipy.optimize import linprog

rng = np.random.default_rng(0)


class PBNNLayout:
    """Tracks variable offsets for one sample's LP: h^0 (fixed=x), then per
    layer t: z^t (dt x dt-1), g^t (dt), h^t (dt)."""

    def __init__(self, dims):
        self.dims = dims  # [d0, d1, ..., dT]
        self.T = len(dims) - 1
        offset = 0
        self.h_offset = {}
        self.z_offset = {}
        self.g_offset = {}

        self.h_offset[0] = offset
        offset += dims[0]
        for t in range(1, self.T + 1):
            self.z_offset[t] = offset
            offset += dims[t] * dims[t - 1]
            self.g_offset[t] = offset
            offset += dims[t]
            self.h_offset[t] = offset
            offset += dims[t]
        self.n_vars = offset

    def z_idx(self, t, i, j):
        return self.z_offset[t] + i * self.dims[t - 1] + j

    def g_idx(self, t, i):
        return self.g_offset[t] + i

    def h_idx(self, t, i):
        return self.h_offset[t] + i


class AlphaLayout:
    """Tracks offsets for the shared parameter vector alpha = (W^1.., b^1..)."""

    def __init__(self, dims):
        self.dims = dims
        self.T = len(dims) - 1
        offset = 0
        self.W_offset = {}
        self.b_offset = {}
        for t in range(1, self.T + 1):
            self.W_offset[t] = offset
            offset += dims[t] * dims[t - 1]
            self.b_offset[t] = offset
            offset += dims[t]
        self.n_params = offset

    def W_idx(self, t, i, j):
        return self.W_offset[t] + i * self.dims[t - 1] + j

    def b_idx(self, t, i):
        return self.b_offset[t] + i

    def unpack(self, alpha):
        W, b = {}, {}
        for t in range(1, self.T + 1):
            di, dj = self.dims[t], self.dims[t - 1]
            W[t] = alpha[self.W_offset[t]: self.W_offset[t] + di * dj].reshape(di, dj)
            b[t] = alpha[self.b_offset[t]: self.b_offset[t] + di]
        return W, b


def build_lp(dims, alpha_layout, W, b, x, u):
    """Build one sample's LP: min c^T v  s.t. A_ub v <= b_base + M @ alpha,
    A_eq v = x (for h^0), 0 <= v <= 1.
    Returns everything needed to solve and to backprop through duals.
    """
    layout = PBNNLayout(dims)
    n = layout.n_vars
    n_alpha = alpha_layout.n_params
    T = layout.T

    c = np.zeros(n)
    for j in range(dims[T]):
        c[layout.h_idx(T, j)] = u[j]

    rows_A, rows_b0, rows_M = [], [], []  # inequality rows

    def add_row(coeffs, rhs_base, alpha_coeffs=None):
        row = np.zeros(n)
        for idx, val in coeffs:
            row[idx] += val
        m_row = np.zeros(n_alpha)
        if alpha_coeffs:
            for idx, val in alpha_coeffs:
                m_row[idx] += val
        rows_A.append(row)
        rows_b0.append(rhs_base)
        rows_M.append(m_row)

    for t in range(1, T + 1):
        dt, dtm1 = dims[t], dims[t - 1]
        for i in range(dt):
            z_terms = []
            for j in range(dtm1):
                zij = layout.z_idx(t, i, j)
                hprevj = layout.h_idx(t - 1, j)
                Wij = alpha_layout.W_idx(t, i, j)
                z_terms.append(zij)

                # z_ij <= W_ij  ->  z_ij - W_ij <= 0  (rhs = W_ij, dRHS/dW_ij=1)
                add_row([(zij, 1.0)], 0.0, [(Wij, 1.0)])
                # z_ij <= h_prev_j  -> z_ij - h_prev_j <= 0
                add_row([(zij, 1.0), (hprevj, -1.0)], 0.0)
                # z_ij >= W_ij + h_prev_j - 1  -> -z_ij + h_prev_j <= 1 - W_ij
                add_row([(zij, -1.0), (hprevj, 1.0)], 1.0, [(Wij, -1.0)])
                # z_ij >= 0 handled by variable bounds

            gi = layout.g_idx(t, i)
            for zij in z_terms:
                # g_i >= z_ij  -> -g_i + z_ij <= 0
                add_row([(gi, -1.0), (zij, 1.0)], 0.0)
            # g_i <= sum_j z_ij -> g_i - sum(z_ij) <= 0
            add_row([(gi, 1.0)] + [(zij, -1.0) for zij in z_terms], 0.0)

            hi = layout.h_idx(t, i)
            bi = alpha_layout.b_idx(t, i)
            # h_i <= g_i + b_i -> h_i - g_i <= b_i
            add_row([(hi, 1.0), (gi, -1.0)], 0.0, [(bi, 1.0)])
            # h_i <= 2 - g_i - b_i -> h_i + g_i <= 2 - b_i
            add_row([(hi, 1.0), (gi, 1.0)], 2.0, [(bi, -1.0)])
            # h_i >= g_i - b_i -> g_i - h_i <= b_i
            add_row([(gi, 1.0), (hi, -1.0)], 0.0, [(bi, 1.0)])
            # h_i >= b_i - g_i -> -g_i - h_i <= -b_i
            add_row([(gi, -1.0), (hi, -1.0)], 0.0, [(bi, -1.0)])

    A_ub = np.array(rows_A)
    b0 = np.array(rows_b0)
    M = np.array(rows_M)

    W_flat_b_flat = np.concatenate(
        [np.concatenate([W[t].flatten(), b[t]]) for t in range(1, T + 1)]
    )
    b_ub = b0 + M @ W_flat_b_flat

    # h^0 = x (equality)
    A_eq = np.zeros((dims[0], n))
    for j in range(dims[0]):
        A_eq[j, layout.h_idx(0, j)] = 1.0
    b_eq = x.astype(float)

    bounds = [(0.0, 1.0)] * n

    return dict(layout=layout, c=c, A_ub=A_ub, b_ub=b_ub, M=M,
                A_eq=A_eq, b_eq=b_eq, bounds=bounds)


def solve_and_subgrad(dims, alpha_layout, alpha, x, u):
    W, b = alpha_layout.unpack(alpha)
    lp = build_lp(dims, alpha_layout, W, b, x, u)
    res = linprog(lp["c"], A_ub=lp["A_ub"], b_ub=lp["b_ub"],
                   A_eq=lp["A_eq"], b_eq=lp["b_eq"], bounds=lp["bounds"],
                   method="highs")
    if not res.success:
        raise RuntimeError(f"LP failed: {res.message}")

    marginals = res.ineqlin.marginals  # dV/d(b_ub)
    subgrad = lp["M"].T @ marginals    # chain rule: dV/dalpha
    return res.fun, subgrad


def main():
    dims = [3, 3, 2]  # d0, d1(hidden), d2=T output/classes
    alpha_layout = AlphaLayout(dims)

    # toy dataset: 8 samples over {0,1}^3, label = majority bit (0 or 1)
    xs = np.array([[int(b) for b in format(k, "03b")] for k in range(8)])
    ys = (xs.sum(axis=1) >= 2).astype(int)

    def u_from_y(y, n_classes=2):
        u = np.ones(n_classes)
        u[y] = -1.0
        return u

    alpha = rng.uniform(0.3, 0.7, size=alpha_layout.n_params)

    lr = 0.3
    n_epochs = 40
    print("epoch  avg_loss")
    for epoch in range(n_epochs):
        idx = rng.permutation(len(xs))
        total_loss = 0.0
        grad_sum = np.zeros_like(alpha)
        for k in idx:
            x, y = xs[k], ys[k]
            u = u_from_y(y)
            val, g = solve_and_subgrad(dims, alpha_layout, alpha, x, u)
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
