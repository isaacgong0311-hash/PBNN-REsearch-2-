# PBNN Research — BEP Comparison & Courbariaux Sanity-Check

Prep notes for Thursday's meeting with Liping. Covers: (1) BEP's binary error propagation mechanism vs. the PBNN LP-relaxation angle, (2) reference numbers for sanity-checking the BNN baseline against Courbariaux et al.

Source papers: Colombo et al., "BEP: A Binary Error Propagation Algorithm for Binary Neural Networks Training," arXiv:2512.04189 (ICLR 2026); Courbariaux, Hubara, Soudry, El-Yaniv, Bengio, "Binarized Neural Networks," arXiv:1602.02830.

---

## 1. BEP's mechanism

At each layer, the ideal backward problem is

argmax_{a ∈ {±1}^K} ⟨a*_{l+1}, sign(W_{l+1} a)⟩

— NP-hard, nonconvex, discontinuous. BEP drops the inner sign() to get a linear surrogate:

argmax_{a ∈ {±1}^K} ⟨a*_{l+1}, W_{l+1} a⟩

Appendix A (Lemma 2) then relaxes the domain to the hypercube [-1,1]^K and proves the relaxed optimum sits exactly at a vertex: a* = clip(v/|v|) collapses to sign(v). So the relaxation is provably tight — no integrality gap, no rounding step, closed-form solve via one matrix-vector product + sign. The same recursion extends to BEP-TT for RNNs (target state propagated backward through time).

Everything else in the algorithm stays binary/integer: hidden weights H are integer-valued (synaptic inertia only, à la Baldassi's Clipped Perceptron), updates are a Hebbian outer-product rule (ΔH = a*(a_prev)ᵀ), and a binary gate blocks error flow through saturated neurons — no continuous latent parameters anywhere, unlike QAT/STE.

## 2. Is this actually an LP relaxation?

Yes, in the textbook sense — linear objective, box constraints, relax integrality → solve continuous → recover integral solution. What makes it "free" is that BEP only ever relaxes a single, separable, per-layer linear subproblem: coordinates decouple completely, so the relaxed LP trivially splits into K independent 1-D sign problems. No LP solver is ever invoked — Lemma 2 replaces the solver with sign().

What's actually being approximated: the nonlinearity (sign in the forward pass) is stripped from the objective, not solved for. BEP concedes this explicitly — "a global optimum is unnecessary, the goal is to steer weight updates in the appropriate direction." The relaxation is exact for a proxy objective, not for the true credit-assignment problem.

**Where a more general LP formulation (like PBNN's) likely diverges:** if the LP couples variables across a layer or across layers — e.g. a joint program with cross-neuron/cross-layer constraints, or sign kept inside the objective/constraints rather than stripped out — separability breaks and Lemma 2's "tight for free" property disappears. That reintroduces a real integrality gap and a need for actual rounding/solver machinery: a heavier but more general approach than BEP's closed-form shortcut.

**This is the most direct lead on the PBNN degeneracy** (subgradient exactly zero, loss flat at -1.0, found July 30 in toy_lp_testbed.py): if PBNN's LP couples variables (McCormick envelope over AND, OR-reduce bounds, XOR linearization spanning multiple binary vars jointly) rather than solving K independent 1-D problems, the relaxed optimum may sit in the *interior* of the feasible region regardless of the parameters — which is exactly what a permanently-zero subgradient looks like. Worth checking directly: does the PBNN LP ever reduce to a separable per-coordinate problem, or is coupling structural to the formulation?

Other structural contrasts:
- BEP's gating mechanism (blocks error through saturated neurons) functions like an active-set filter — a plausible parallel to "which constraints are binding" in an LP framework.
- BEP-TT is the paper's headline novelty claim ("first end-to-end binary training for RNNs"). Whether the LP angle can extend to recurrent/temporal architectures is an open comparison point.
- BEP's benchmark suite (Random Prototypes, FashionMNIST, CIFAR-10 via AlexNet features, Imagenette, 30 UCR series) never touches standard MNIST/CIFAR-10 in the Courbariaux sense — it's compared only against Larq QAT and their own prior local-rule paper (Colombo et al., 2025), not against BinaryConnect/BNN numbers.

Caveat: this comparison is built from the BEP paper's own math against the project's one-line description ("LP relaxation of the binary optimization problem"). If there's a formulation doc for the PBNN side, it should sharpen this against the actual constraint structure.

## 3. Courbariaux et al. — reference numbers for the sanity check

Canonical results, Table 1 of arXiv:1602.02830 (test error, lower is better):

| Setting | MNIST | SVHN | CIFAR-10 |
|---|---|---|---|
| BNN (Torch7) — binarized weights+activations, train & test | 1.40% | 2.53% | 10.15% |
| BNN (Theano) — binarized weights+activations, train & test | 0.96% | 2.80% | 11.40% |
| BinaryConnect (Courbariaux 2015) — binarized weights only | 1.29±0.08% | 2.30% | 9.90% |

Architecture / protocol gotchas:
- MNIST MLP (Theano): 3 hidden layers × 4096 units, L2-SVM output layer, no conv/data-aug/unsupervised pretraining, dropout, ADAM, BatchNorm (minibatch 100), 1000 epochs, test error = best validation epoch (no retrain on validation split).
- MNIST MLP (Torch7): same but 2048 units/layer, no dropout, shift-based AdaMax + shift-based BatchNorm.
- BinaryConnect binarizes weights only (activations stay full-precision) — a different regime from the BNN rows, not apples-to-apples with a fully-binary baseline.
- Test error is post-early-stopping on a held-out 10K split of the 60K training set.
