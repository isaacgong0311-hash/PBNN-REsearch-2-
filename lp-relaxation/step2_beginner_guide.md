# Step 2 guide: understanding `toy_lp_testbed.py` from scratch

This assumes you know Python but nothing about linear programming (LP) or
binary neural networks (BNNs) going in. Read it top to bottom once, then use
it as a reference while you re-read `toy_lp_testbed.py` next to
`documentation.md`.

---

## 1. What problem is this even solving?

A normal neural network layer looks like `h = f(W @ x + b)` where `W`, `x`,
`b` are all real (floating-point) numbers, and you train `W`, `b` with
gradient descent because everything is differentiable.

A **pure binary neural network (PBNN)** replaces every number with a single
bit (0 or 1), and replaces multiply/add with AND/OR/XOR:

```
h^t_i = b^t_i  XOR  OR_j( W^t_{i,j} AND h^{t-1}_j )
```

Why bother? Because a chip that only ever does AND/OR/XOR on single bits is
*much* simpler and cheaper to build than one that does floating-point
multiply-accumulate — that's the hardware motivation (see doc, "Introduction").

The problem: **you can't take a gradient of AND/OR/XOR.** They're not
smooth functions — they jump. And searching over all possible 0/1 settings
of `W` and `b` directly is combinatorial (NP-hard) — there's no clever
shortcut in general.

## 2. The LP relaxation trick

The classic trick when you're stuck with discrete 0/1 variables: **pretend
they're allowed to be any real number between 0 and 1** (a "relaxation"),
write down linear inequality constraints that force the relaxed variable to
behave exactly like the boolean operation *whenever the inputs sit at the
corners 0 or 1*, then use a fast, well-understood algorithm (a linear
program, LP) to optimize over this continuous stand-in problem.

Concretely: instead of writing `z = W AND h`, you introduce `z` as a new
free variable in `[0,1]` and add:

```
z <= W
z <= h
z >= W + h - 1
z >= 0
```

Try plugging in `W=1,h=1` → forces `z=1`. Try `W=0,h=1` → forces `z=0`.
It behaves exactly like AND at every combination of 0s and 1s. In between
(say `W=0.5, h=0.5`) it only loosely pins `z` down to a range — that
looseness is the "slack" you'll hear about a lot below, and it turns out to
be the whole story behind why this testbed is broken.

This specific 4-inequality trick is called a **McCormick envelope**, and
it's a known, general technique for linearizing a product of two bounded
variables — not something PBNN-specific. `documentation.md` does the same
thing for OR (called "OR-reduce" there) and for XOR, using different
inequalities, so the entire network becomes one big linear program instead
of a tangle of boolean logic.

**Why "linear program" specifically?** Once every variable is continuous
and every constraint is a linear inequality, and the thing you're minimizing
(the loss) is also linear in those variables, you have an LP — and LPs are
one of the best-understood, fastest-to-solve problems in all of
optimization (that's what `scipy.optimize.linprog` does under the hood,
using an algorithm called HiGHS here).

## 3. Where does a "gradient" come from, if LPs aren't differentiable either?

Good instinct — a raw LP just returns a number (the optimal value) and a
solution vector, no gradient. The trick here is **LP sensitivity analysis**
(also called the envelope theorem): every LP solve also produces **dual
values** (one per constraint), which tell you "if I nudged this
constraint's right-hand side by a tiny amount, how much would the optimal
value change?" If you set things up so that your trainable parameters
(`W`, `b`) only ever appear on the right-hand side of constraints (never as
LP decision variables themselves), then the dual values directly hand you
`d(loss)/d(parameter)` for free, every time you solve the LP. That's a real
theorem, not a hack — but it depends on the LP actually solving to a point
where those parameter-linked constraints are *binding* (tight). If they're
not tight, the dual value is exactly 0, and you get zero gradient signal —
which is exactly the failure mode you found.

## 4. Code walkthrough

Open `toy_lp_testbed.py` alongside this section.

### `PBNNLayout` / `AlphaLayout` (lines 29–90)

Pure bookkeeping, no math. An LP solver wants one big flat array of numbers
for its variables. These two classes are just lookup tables: "what index in
that flat array is `h^1_2`?" / "what index is `W^2_{0,1}`?" `PBNNLayout`
indexes the LP's own decision variables (`z`, `g`, `h` — different for
every training sample). `AlphaLayout` indexes the trainable parameters
(`W`, `b` — shared across all samples).

### `build_lp` (93–175) — builds one sample's LP

- **`c` (103–105):** the objective. `c[h^T_j] = u[j]`, and `u` is `-1` for
  the correct class, `+1` for the rest (see file docstring). Since
  `linprog` *minimizes* `c · v`, this rewards pushing the correct class's
  output toward 1 and the wrong class's output toward 0 — this is
  `L = Σ u_j h^T_j` from `documentation.md`.
- **`add_row` (109–119):** every constraint here is
  `(some LP variables) <= constant + (some parameters)`. This helper
  records the LP-variable part and the parameter part *separately* — that
  split is exactly what makes the "parameters only shift the right-hand
  side" property from §3 true, which is what makes the dual-value gradient
  trick valid at all.
- **AND / McCormick (121–137):** implements the 4 inequalities from §2 for
  every `z^t_{i,j} = W^t_{i,j} AND h^{t-1}_j`.
- **OR-reduce, `g` (139–144):** `g_i >= z_ij` for every `j`, and
  `g_i <= Σ_j z_ij`. This is meant to stand in for
  `g = OR(z_1, z_2, ...)`.
- **XOR, `h` (146–155):** four inequalities that box in
  `h_i = b_i XOR g_i`.
- **Wiring parameters into the right-hand side (157–164):**
  `b_ub = b0 + M @ alpha` — this *is* the "parameters only move the
  boundary" structure from §3, spelled out in code.
- **Pinning the input (166–170):** the *only* equality constraint in the
  whole LP — `h^0 = x`. Every other variable (`z`, `g`, `h^1`, ..., `h^T`)
  is only loosely bounded, not pinned to one value.

### `solve_and_subgrad` (178–189)

1. Unpack `alpha` into concrete numbers `W`, `b` (fixed for this solve —
   you're solving for the LP's own variables, not for `alpha`).
2. Solve the LP.
3. `res.ineqlin.marginals` = the dual value for every inequality
   constraint — 0 if that constraint wasn't binding at the optimum,
   nonzero if it was.
4. `subgrad = M.T @ marginals` — converts "sensitivity to the constraint
   boundary" into "sensitivity to `alpha`" via the chain rule, since
   `b_ub = b0 + M @ alpha`.

### `main` (192–230)

Ordinary stochastic subgradient descent: build an 8-sample toy dataset (all
3-bit inputs, label = majority bit), loop over epochs, solve every sample's
LP + subgradient, average the subgradients, step `alpha` downhill, clip
back into `[0,1]` (the only constraint `alpha` itself has to satisfy).

## 5. Why it's degenerate — the mechanism, confirmed by experiment

I ran a direct check (values below are from an actual solve, not a guess):
for `x = [1,0,1]` and a random mid-range `alpha`, the solver's `z^1`
values matched `W^1_{i,j} * x_j` **exactly** — no slack at all in layer 1's
AND. That's because McCormick is provably *tight* whenever at least one of
the two multiplied variables sits exactly at a corner (0 or 1) — and
`x` is always exactly binary real data, so layer 1's AND can never be loose.

But `g^1` came out with a real gap between `max(z)` and `sum(z)` (e.g.
`max=0.555, sum=0.871`) — because OR-reduce's `g <= Σ z_j` is only equal to
the true OR (`max`) when at most one `z_j` is nonzero. The moment 2+ inputs
"fire" through the same neuron, `sum > max`, and `g` gets real freedom to
be anywhere in between. **This is the first place slack enters, and it
doesn't require any fractional weights at all** — just 2+ simultaneously
active inputs.

Separately — and this turned out to matter just as much — `h^1` came out
fractional too (e.g. `0.119`, nowhere near 0 or 1), even though `g^1` in
that solve was pinned at its tight lower value. That's because the XOR box
constraint only pins `h` to one exact value when *both* `g` and `b` sit at
0/1 corners. `b^1` is a trainable parameter living in continuous `[0,1]` —
it's essentially *never* exactly 0 or 1 during training — so `h^1` gets a
real feasible interval purely from `b` being fractional, independent of
anything upstream.

That fractional, freely-choosable `h^1` then feeds into layer 2's AND
(`z^2 = W^2 AND h^1`) — and now, unlike layer 1, McCormick is genuinely
loose too, because neither `W^2` nor `h^1` is pinned to a corner anymore.

**So the degeneracy isn't caused by one single "too loose" constraint.**
It's a two-step chain: (1) OR-reduce's sum-vs-max gap and the XOR box's
looseness under a fractional bias both introduce real freedom in `h^1`
starting at layer 1, with zero dependence on how fractional the weights
are; then (2) that already-fractional `h^1` breaks McCormick's tightness
guarantee at layer 2, compounding the freedom further. By the output layer,
there's enough accumulated freedom to hit the LP's best-case loss (`-1`)
regardless of `x` or `alpha` — which is exactly the flat `-1.0000` you saw.

This also explains why the entropy-penalty fix helped but stayed noisy: it
pushes `z, g, h` toward corners, but `b` (a fixed parameter within any
single LP solve) can't be pushed by that penalty — so `h` can get closer to
a corner but often can't reach it exactly while `b` stays fractional. The
fix reduces the freedom without eliminating it, which matches the
"unstuck but noisy" loss curve instead of a clean monotonic decrease.
