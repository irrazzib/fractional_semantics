from __future__ import annotations

import unittest

from fs_semantics.decomposition_service import generate_decomposition_bundle
from fs_semantics.logic import (
    formula_to_ascii,
    formula_to_latex,
    parse_formula,
    parse_sequent,
    parse_sequent_structured,
    sequent_to_ascii_grouped,
)


class SequentInputSemanticsTests(unittest.TestCase):
    def test_one_sided_comma_is_disjunction(self) -> None:
        parsed = parse_sequent_structured("q, r & s")
        self.assertFalse(parsed.is_two_sided)
        self.assertEqual(sequent_to_ascii_grouped(parsed.one_sided), "q, r & s")

    def test_two_sided_maps_to_one_sided_with_negated_left(self) -> None:
        parsed = parse_sequent_structured("p, q |- r, s & t")
        self.assertTrue(parsed.is_two_sided)
        self.assertEqual(
            sequent_to_ascii_grouped(parsed.one_sided),
            "~p, ~q, r, s & t",
        )

    def test_parse_sequent_accepts_turnstile_notation(self) -> None:
        one_sided = parse_sequent("p |- q, r")
        self.assertEqual(sequent_to_ascii_grouped(one_sided), "~p, q, r")

    def test_decomposition_root_eliminates_commas_via_final_vee(self) -> None:
        bundle = generate_decomposition_bundle(
            "p |- q, r & s",
            decomposition_style="fractional",
        )
        self.assertEqual(bundle["sequent_kind"], "two-sided")
        decomposition = str(bundle["decomposition"])
        root_display = str(bundle["tree"]["display"])
        proof_lines = [line for line in decomposition.splitlines() if "InfC{$\\sststile" in line]
        self.assertTrue(proof_lines)
        final_line = proof_lines[-1]
        self.assertIn("\\neg p \\vee q \\vee r \\wedge s", decomposition)
        self.assertIn("~p | q | r & s", root_display)
        self.assertNotIn(",", final_line)
        self.assertNotIn("~p, q, r & s", root_display)

    def test_ascii_negation_of_compound_has_single_parenthesis_pair(self) -> None:
        formula = parse_formula("~(p | q)")
        self.assertEqual(formula_to_ascii(formula), "~(p | q)")

    def test_latex_negation_of_compound_has_single_parenthesis_pair(self) -> None:
        formula = parse_formula("~(p | q)")
        self.assertEqual(formula_to_latex(formula), "\\neg(p \\vee q)")

    def test_mixed_and_or_conjunction_has_stronger_precedence(self) -> None:
        parsed = parse_formula("u | p & v")
        self.assertEqual(formula_to_ascii(parsed), "u | p & v")
        self.assertEqual(parsed, parse_formula("u | (p & v)"))
        self.assertNotEqual(parsed, parse_formula("(u | p) & v"))


if __name__ == "__main__":
    unittest.main()
