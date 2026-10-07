# PBNN — Positions on Open Questions
_Aug 4, 2026 — for Thursday's meeting with Liping / sync with Alex_

## Q1: Is the tree vs. BNN bit-op comparison a legitimate baseline?

**Position: Yes, framed as an accuracy/cost frontier — not a head-to-head.**

- A decision tree (~15-25 bit ops/sample, ~86% MNIST acc) and a BNN
  (~537,600 bit ops/sample, ~96-99% acc) sit at opposite ends of an
  accuracy-vs-compute tradeoff. PBNN's contribution is to land closer to
  the tree's cost while keeping BNN-level accuracy — that only makes
  sense as a claim if both endpoints are on the same plot.
- The failure mode to avoid: presenting bit-op count alone, without
  accuracy attached, since a method that's "cheap and wrong" is not
  a meaningful comparison point on its own.
- Counterpoint worth acknowledging: bit-op counting may not be a fair
  unit across two different computational paradigms (tree traversal vs.
  matrix ops) — flag this as a caveat rather than a blocker.

## Q2: Scale up the binary MLP for a paper-comparable number, or move to the real LP-relaxation PBNN method now?

**Position: Move on now.**

- The STE training mechanics are already validated end-to-end (96.34%
  binary vs. 97.68% full-precision, small architecture, 5 epochs) — the
  remaining gap to the paper's number is compute time, not open questions.
- The LP relaxation was already found to be degenerate (subgradient
  exactly zero, loss stuck at -1.0) in an earlier session — that's the
  actual blocker on the project's core method and the higher-value use
  of time right now.
- Fallback: cite the paper's own reported number (0.96% error / 99.04%
  acc, Theano variant) as the literature baseline rather than
  re-deriving it ourselves, unless Liping specifically wants an
  in-house matched-architecture number for the writeup.
