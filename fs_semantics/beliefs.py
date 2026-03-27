from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

from .calculus import Clause, clauses_from_formula, is_tautological_clause
from .logic import Formula, Not, Or, parse_sequent, to_nnf


def _normalize_formula(formula: Formula) -> Formula:
    return to_nnf(formula)


def _close_under_cut(initial_clauses: set[Clause]) -> set[Clause]:
    result = set(initial_clauses)
    changed = True
    while changed:
        changed = False
        current = list(result)
        for clause_a, clause_b in combinations(current, 2):
            for literal in clause_a:
                comp = literal.complement()
                if comp not in clause_b:
                    continue
                resolvent = frozenset((clause_a - {literal}) | (clause_b - {comp}))
                if is_tautological_clause(resolvent):
                    continue
                if resolvent not in result:
                    result.add(resolvent)
                    changed = True
    return result


def build_belief_clauses(
    formulas: Iterable[Formula],
    close_under_cut: bool = True,
) -> set[Clause]:
    clauses: set[Clause] = set()
    for belief in formulas:
        for clause in clauses_from_formula(_normalize_formula(belief)):
            if is_tautological_clause(clause):
                continue
            clauses.add(clause)
    if close_under_cut:
        clauses = _close_under_cut(clauses)
    return clauses


@dataclass(frozen=True)
class BeliefBase:
    formulas: tuple[Formula, ...]
    clauses: frozenset[Clause]
    close_under_cut: bool = True

    @classmethod
    def from_strings(
        cls,
        beliefs: Iterable[str],
        close_under_cut: bool = True,
    ) -> "BeliefBase":
        formulas = tuple(
            _normalize_formula(_belief_string_to_formula(belief)) for belief in beliefs
        )
        clauses = frozenset(build_belief_clauses(formulas, close_under_cut=close_under_cut))
        return cls(formulas=formulas, clauses=clauses, close_under_cut=close_under_cut)

    @classmethod
    def from_formulas(
        cls,
        formulas: Iterable[Formula],
        close_under_cut: bool = True,
    ) -> "BeliefBase":
        normalized = tuple(_normalize_formula(item) for item in formulas)
        clauses = frozenset(build_belief_clauses(normalized, close_under_cut=close_under_cut))
        return cls(formulas=normalized, clauses=clauses, close_under_cut=close_under_cut)

    def contract(self, target: Formula) -> "BeliefBase":
        target_nnf = _normalize_formula(target)
        remaining = tuple(formula for formula in self.formulas if formula != target_nnf)
        return BeliefBase.from_formulas(remaining, close_under_cut=self.close_under_cut)

    def expand(self, target: Formula) -> "BeliefBase":
        target_nnf = _normalize_formula(target)
        if target_nnf in self.formulas:
            return self
        expanded = self.formulas + (target_nnf,)
        return BeliefBase.from_formulas(expanded, close_under_cut=self.close_under_cut)

    def revise(self, target: Formula) -> "BeliefBase":
        neg_target = _normalize_formula(Not(target))
        contracted = self.contract(neg_target)
        return contracted.expand(target)


def _belief_string_to_formula(text: str) -> Formula:
    parts = parse_sequent(text)
    if len(parts) == 1:
        return parts[0]
    acc = parts[0]
    for formula in parts[1:]:
        acc = Or(acc, formula)
    return acc
