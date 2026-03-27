from __future__ import annotations

import unittest

from fs_semantics.beliefs import BeliefBase
from fs_semantics.calculus import evaluate_sequent
from fs_semantics.logic import parse_formula, parse_sequent


class FractionalSemanticsTests(unittest.TestCase):
    def test_documentation_example_without_beliefs(self) -> None:
        sequent = parse_sequent("(p | q) & (p | ~p) & (~r & ~t)")
        result = evaluate_sequent(sequent, belief_clauses=set())
        self.assertEqual(result.top1, 1)
        self.assertEqual(result.total, 4)
        self.assertEqual(str(result.value), "1/4")

    def test_documentation_example_with_beliefs(self) -> None:
        sequent = parse_sequent("(p | q) & (s | ~s) & (~r & ~t)")
        beliefs = BeliefBase.from_strings(["(p | q)", "~u"])
        result = evaluate_sequent(sequent, belief_clauses=set(beliefs.clauses))
        self.assertEqual(result.top1, 1)
        self.assertEqual(result.topb, 1)
        self.assertEqual(result.total, 4)
        self.assertEqual(str(result.value_b), "1/2")

    def test_levi_identity_style_revision(self) -> None:
        base = BeliefBase.from_strings(["~p", "q"])
        revised = base.revise(parse_formula("p"))
        self.assertNotIn(parse_formula("~p"), set(revised.formulas))
        self.assertIn(parse_formula("p"), set(revised.formulas))

    def test_overline_notation(self) -> None:
        formula = parse_formula(r"\overline{p} | q")
        self.assertEqual(parse_formula("~p | q"), formula)

    def test_belief_clause_with_comma_notation(self) -> None:
        sequent = parse_sequent("(p | q) & (s | ~s) & (~r & ~t)")
        beliefs = BeliefBase.from_strings(["p, q"])
        result = evaluate_sequent(sequent, belief_clauses=set(beliefs.clauses))
        self.assertEqual(str(result.value_b), "1/2")


if __name__ == "__main__":
    unittest.main()
