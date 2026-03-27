from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .beliefs import BeliefBase
from .calculus import Clause, EvaluationResult, evaluate_sequent, is_tautological_clause
from .logic import (
    And,
    Atom,
    Formula,
    Not,
    Or,
    formula_to_literal,
    parse_sequent_structured,
    sequent_to_ascii_grouped,
    to_nnf,
)


@dataclass(frozen=True)
class SCLeaf:
    clause: Clause
    kind: str  # "ax" | "open"


@dataclass
class SCNode:
    antecedent: tuple[Formula, ...]
    succedent: tuple[Formula, ...]
    rule: str
    children: list["SCNode"]
    leaf: SCLeaf | None = None


@dataclass(frozen=True)
class SCResult:
    root: SCNode
    leaves: tuple[SCLeaf, ...]
    total: int
    ax: int
    open_: int


@dataclass(frozen=True)
class SCComparison:
    matches: bool
    fractional_counts: Counter[Clause]
    reference_counts: Counter[Clause]
    only_fractional: Counter[Clause]
    only_reference: Counter[Clause]
    reference: SCResult
    fractional: EvaluationResult


def _is_literal(formula: Formula) -> bool:
    return isinstance(formula, Atom) or (
        isinstance(formula, Not) and isinstance(formula.arg, Atom)
    )


def _clause_from_two_sided(
    antecedent: tuple[Formula, ...],
    succedent: tuple[Formula, ...],
) -> Clause:
    one_sided = tuple(to_nnf(Not(item)) for item in antecedent) + tuple(
        to_nnf(item) for item in succedent
    )
    literals = [formula_to_literal(item) for item in one_sided]
    return frozenset(literals)


def _pick_focus(
    antecedent: tuple[Formula, ...],
    succedent: tuple[Formula, ...],
) -> tuple[str, int, Formula] | None:
    for idx, formula in enumerate(antecedent):
        if _is_literal(formula):
            continue
        if isinstance(formula, (And, Or)):
            return ("L", idx, formula)
    for idx, formula in enumerate(succedent):
        if _is_literal(formula):
            continue
        if isinstance(formula, (And, Or)):
            return ("R", idx, formula)
    return None


def _expand(
    antecedent: tuple[Formula, ...],
    succedent: tuple[Formula, ...],
    leaves: list[SCLeaf],
) -> SCNode:
    focus = _pick_focus(antecedent, succedent)
    if focus is None:
        clause = _clause_from_two_sided(antecedent, succedent)
        kind = "ax" if is_tautological_clause(clause) else "open"
        leaf = SCLeaf(clause=clause, kind=kind)
        leaves.append(leaf)
        return SCNode(
            antecedent=antecedent,
            succedent=succedent,
            rule="leaf",
            children=[],
            leaf=leaf,
        )

    side, idx, formula = focus
    if side == "L":
        prefix = antecedent[:idx]
        suffix = antecedent[idx + 1 :]
        assert isinstance(formula, (And, Or))
        if isinstance(formula, And):
            next_left = prefix + (formula.left, formula.right) + suffix
            child = _expand(next_left, succedent, leaves)
            return SCNode(
                antecedent=antecedent,
                succedent=succedent,
                rule="andL",
                children=[child],
            )

        left_branch = prefix + (formula.left,) + suffix
        right_branch = prefix + (formula.right,) + suffix
        child1 = _expand(left_branch, succedent, leaves)
        child2 = _expand(right_branch, succedent, leaves)
        return SCNode(
            antecedent=antecedent,
            succedent=succedent,
            rule="orL",
            children=[child1, child2],
        )

    prefix = succedent[:idx]
    suffix = succedent[idx + 1 :]
    assert isinstance(formula, (And, Or))
    if isinstance(formula, Or):
        next_right = prefix + (formula.left, formula.right) + suffix
        child = _expand(antecedent, next_right, leaves)
        return SCNode(
            antecedent=antecedent,
            succedent=succedent,
            rule="orR",
            children=[child],
        )

    left_branch = prefix + (formula.left,) + suffix
    right_branch = prefix + (formula.right,) + suffix
    child1 = _expand(antecedent, left_branch, leaves)
    child2 = _expand(antecedent, right_branch, leaves)
    return SCNode(
        antecedent=antecedent,
        succedent=succedent,
        rule="andR",
        children=[child1, child2],
    )


def evaluate_reference_sc(sequent_text: str) -> SCResult:
    parsed = parse_sequent_structured(sequent_text)
    left = tuple(to_nnf(item) for item in parsed.antecedent)
    right = tuple(to_nnf(item) for item in parsed.succedent)
    leaves: list[SCLeaf] = []
    root = _expand(left, right, leaves)
    ax = sum(1 for leaf in leaves if leaf.kind == "ax")
    open_ = sum(1 for leaf in leaves if leaf.kind == "open")
    return SCResult(
        root=root,
        leaves=tuple(leaves),
        total=len(leaves),
        ax=ax,
        open_=open_,
    )


def _leaf_counter(leaves: tuple[SCLeaf, ...] | tuple) -> Counter[Clause]:
    counts: Counter[Clause] = Counter()
    for leaf in leaves:
        counts[leaf.clause] += 1
    return counts


def compare_with_fractional(
    sequent_text: str,
    *,
    belief_base: BeliefBase | None = None,
) -> SCComparison:
    parsed = parse_sequent_structured(sequent_text)
    reference = evaluate_reference_sc(sequent_text)
    belief_clauses = set(belief_base.clauses) if belief_base is not None else set()
    fractional = evaluate_sequent(parsed.one_sided, belief_clauses=belief_clauses)

    reference_counts = _leaf_counter(reference.leaves)
    fractional_counts: Counter[Clause] = Counter(leaf.clause for leaf in fractional.leaves)
    only_fractional = fractional_counts - reference_counts
    only_reference = reference_counts - fractional_counts
    return SCComparison(
        matches=(not only_fractional and not only_reference),
        fractional_counts=fractional_counts,
        reference_counts=reference_counts,
        only_fractional=only_fractional,
        only_reference=only_reference,
        reference=reference,
        fractional=fractional,
    )


def sc_sequent_to_ascii(antecedent: tuple[Formula, ...], succedent: tuple[Formula, ...]) -> str:
    left = sequent_to_ascii_grouped(antecedent) if antecedent else "·"
    right = sequent_to_ascii_grouped(succedent) if succedent else "·"
    return f"{left} |- {right}"


def render_reference_tree_ascii(root: SCNode) -> str:
    lines: list[str] = []

    def walk(node: SCNode, prefix: str, is_last: bool) -> None:
        connector = "└─ " if is_last else "├─ "
        sequ = sc_sequent_to_ascii(node.antecedent, node.succedent)
        if node.rule == "leaf":
            assert node.leaf is not None
            label = f"{sequ} [{node.leaf.kind}]"
        else:
            label = f"{sequ} ({node.rule})"
        lines.append(prefix + connector + label)
        next_prefix = prefix + ("   " if is_last else "│  ")
        for idx, child in enumerate(node.children):
            walk(child, next_prefix, idx == len(node.children) - 1)

    walk(root, "", True)
    return "\n".join(lines)

