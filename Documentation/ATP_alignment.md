# Fractional Semantics vs ATP-Style Guidance

This note compares the current implementation with ATP/sequent-calculus guidance from:

- <https://pqnelson.github.io/2020/03/27/automated-theorem-provers.html>
- `Documentation/atp.pdf`

The goal is coherence with ATP methodology while staying strictly inside
Fractional Semantics (analytic GS4-style decomposition, propositional level).

## What is aligned

- Sequent-calculus proof objects:
  the system builds explicit proof trees (`vee`, `wedge`, `leaf`) and validates
  every decomposition structurally.
- One-sided normalization pipeline:
  two-sided inputs are translated to one-sided form `⊢ ¬Gamma, Delta`, consistent
  with the GS4/Fractional setting.
- Clause-oriented decomposition:
  decomposition rules flatten disjunction in-branch and branch on conjunction,
  ending in literal clauses used for top-sequent counting.
- Exchange compatibility:
  decomposition validation now treats sequent contexts as multisets for rule-step
  matching (order-insensitive), as expected in sequent calculi.
- Cut-compatible belief handling:
  belief clauses can be closed under cut/resolution before scoring leaf kinds.

## What remains intentionally out of scope

- No first-order ATP machinery:
  no quantifier rules, unification, Skolemization, or full resolution saturation.
- No general-purpose prover search strategies:
  no heuristic backward/forward ATP loop beyond deterministic GS4 decomposition.

These limits are intentional: the implementation targets Fractional Semantics
evaluation and belief-sensitive valuation, not a full theorem prover.

