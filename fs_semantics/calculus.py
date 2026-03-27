from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import Iterable, Sequence

from .logic import (
    And,
    Formula,
    Literal,
    Or,
    formula_to_literal,
    to_nnf,
)


Clause = frozenset[Literal]


def clause_to_latex(clause: Clause) -> str:
    ordered = sorted(clause, key=lambda l: (l.atom, l.negated))
    return ", ".join(literal.to_latex() for literal in ordered) if ordered else "\\bot"


def is_tautological_clause(clause: Clause) -> bool:
    return any(literal.complement() in clause for literal in clause)


@dataclass(frozen=True)
class Leaf:
    clause: Clause
    kind: str


@dataclass
class ProofNode:
    sequent: tuple[Formula, ...]
    rule: str
    children: list["ProofNode"]
    leaf: Leaf | None = None


@dataclass(frozen=True)
class EvaluationResult:
    root: ProofNode
    leaves: tuple[Leaf, ...]
    total: int
    top1: int
    top0: int
    topb: int
    value: Fraction
    value_b: Fraction


def _find_decomposable(sequent: Sequence[Formula]) -> int | None:
    for idx, formula in enumerate(sequent):
        if isinstance(formula, (And, Or)):
            return idx
    return None


def _build_clause(sequent: Sequence[Formula]) -> Clause:
    literals = [formula_to_literal(item) for item in sequent]
    return frozenset(literals)


def _classify_clause(clause: Clause, belief_clauses: set[Clause] | None) -> str:
    if is_tautological_clause(clause):
        return "ax"
    if belief_clauses and clause in belief_clauses:
        return "belief"
    return "comp"


def _expand(
    sequent: tuple[Formula, ...],
    belief_clauses: set[Clause] | None,
    leaves: list[Leaf],
) -> ProofNode:
    idx = _find_decomposable(sequent)
    if idx is None:
        clause = _build_clause(sequent)
        kind = _classify_clause(clause, belief_clauses)
        leaf = Leaf(clause=clause, kind=kind)
        leaves.append(leaf)
        return ProofNode(sequent=sequent, rule="leaf", children=[], leaf=leaf)

    focus = sequent[idx]
    assert isinstance(focus, (And, Or))
    prefix = sequent[:idx]
    suffix = sequent[idx + 1 :]

    if isinstance(focus, Or):
        next_sequent = prefix + (focus.left, focus.right) + suffix
        child = _expand(next_sequent, belief_clauses, leaves)
        return ProofNode(sequent=sequent, rule="vee", children=[child])

    left_sequent = prefix + (focus.left,) + suffix
    right_sequent = prefix + (focus.right,) + suffix
    left_child = _expand(left_sequent, belief_clauses, leaves)
    right_child = _expand(right_sequent, belief_clauses, leaves)
    return ProofNode(sequent=sequent, rule="wedge", children=[left_child, right_child])


def clauses_from_formula(formula: Formula) -> tuple[Clause, ...]:
    nnf = to_nnf(formula)
    leaves: list[Leaf] = []
    _ = _expand((nnf,), belief_clauses=None, leaves=leaves)
    return tuple(leaf.clause for leaf in leaves)


def evaluate_sequent(
    sequent: Iterable[Formula],
    belief_clauses: set[Clause] | None = None,
) -> EvaluationResult:
    normalized = tuple(to_nnf(item) for item in sequent)
    leaves: list[Leaf] = []
    root = _expand(normalized, belief_clauses=belief_clauses, leaves=leaves)
    top1 = sum(1 for leaf in leaves if leaf.kind == "ax")
    topb = sum(1 for leaf in leaves if leaf.kind == "belief")
    top0 = sum(1 for leaf in leaves if leaf.kind == "comp")
    total = len(leaves)
    if total == 0:
        raise ValueError("Leafless derivation: invalid input")
    value = Fraction(top1, total)
    value_b = Fraction(top1 + topb, total)
    return EvaluationResult(
        root=root,
        leaves=tuple(leaves),
        total=total,
        top1=top1,
        top0=top0,
        topb=topb,
        value=value,
        value_b=value_b,
    )
