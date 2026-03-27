from __future__ import annotations

from fractions import Fraction
import unittest

from fs_semantics.beliefs import BeliefBase
from fs_semantics.calculus import evaluate_sequent
from fs_semantics.input_parsing import parse_belief_set_text, parse_gradient_values_text
from fs_semantics.latex import render_analysis_document, render_decomposition_prooftree
from fs_semantics.logic import parse_sequent


class CliAndLatexBehaviorTests(unittest.TestCase):
    def test_parse_belief_set_prompt_style(self) -> None:
        parsed = parse_belief_set_text("B={(p | q); ~u}")
        self.assertEqual(parsed, ["(p | q)", "~u"])

    def test_parse_gradient_values_prompt_style(self) -> None:
        parsed = parse_gradient_values_text("G={0.8; 1/2; 1}")
        self.assertEqual(parsed, [Fraction(4, 5), Fraction(1, 2), Fraction(1, 1)])

    def test_parse_gradient_values_rejects_out_of_range(self) -> None:
        with self.assertRaises(ValueError):
            _ = parse_gradient_values_text("1.2")

    def test_empty_belief_base_has_no_b_turnstile_suffix(self) -> None:
        sequent = parse_sequent("(p | q) & (p | ~p) & (~r & ~t)")
        result = evaluate_sequent(sequent, belief_clauses=set())
        tex = render_analysis_document("(p | q) & (p | ~p) & (~r & ~t)", result, None)
        self.assertNotIn("_{\\mathbb{B}}", tex)

    def test_non_empty_belief_base_has_b_turnstile_suffix(self) -> None:
        sequent = parse_sequent("(p | q) & (s | ~s) & (~r & ~t)")
        base = BeliefBase.from_strings(["(p | q)", "~u"])
        result = evaluate_sequent(sequent, belief_clauses=set(base.clauses))
        tex = render_analysis_document("(p | q) & (s | ~s) & (~r & ~t)", result, base)
        self.assertIn("_{\\mathbb{B}}", tex)

    def test_decomposition_only_renderer(self) -> None:
        sequent = parse_sequent("(p | q) & (p | ~p)")
        result = evaluate_sequent(sequent, belief_clauses=set())
        decomposition = render_decomposition_prooftree(result, with_belief_base=False)
        self.assertIn("\\begin{prooftree}", decomposition)
        self.assertIn("\\end{prooftree}", decomposition)
        self.assertNotIn("\\section*", decomposition)

    def test_decomposition_full_mode_includes_delta(self) -> None:
        sequent = parse_sequent("(p | q) & (s | ~s)")
        base = BeliefBase.from_strings(["(p | q)"])
        result = evaluate_sequent(sequent, belief_clauses=set(base.clauses))
        decomposition = render_decomposition_prooftree(
            result,
            with_belief_base=True,
            belief_mode="full",
        )
        self.assertIn("\\delta_{", decomposition)

    def test_decomposition_full_mode_single_delta_has_no_parentheses(self) -> None:
        sequent = parse_sequent("(p | q)")
        base = BeliefBase.from_strings(["(p | q)"])
        result = evaluate_sequent(sequent, belief_clauses=set(base.clauses))
        decomposition = render_decomposition_prooftree(
            result,
            with_belief_base=True,
            belief_mode="full",
        )
        self.assertIn("1-\\delta_{", decomposition)
        self.assertNotIn("1-(\\delta_{", decomposition)

    def test_decomposition_gradient_mode_uses_numeric_weights(self) -> None:
        sequent = parse_sequent("p & q")
        base = BeliefBase.from_strings(["p", "q"])
        result = evaluate_sequent(sequent, belief_clauses=set(base.clauses))
        strengths = {
            leaf.clause: Fraction(4, 5) if idx == 0 else Fraction(7, 10)
            for idx, leaf in enumerate(result.leaves)
        }
        decomposition = render_decomposition_prooftree(
            result,
            with_belief_base=True,
            belief_mode="gradient",
            belief_strengths=strengths,
        )
        self.assertIn("{1.5}", decomposition)
        self.assertNotIn("\\delta_{", decomposition)


if __name__ == "__main__":
    unittest.main()
