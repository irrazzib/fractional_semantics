from __future__ import annotations

from collections import Counter
import unittest

from fs_semantics.cut_free import build_cut_free_tree, collect_cut_free_leaf_clauses
from fs_semantics.decomposition_service import (
    generate_decomposition,
    generate_decomposition_bundle,
)
from fs_semantics.logic import Or, parse_sequent_structured, to_nnf
from fs_semantics.propositional_sc import evaluate_reference_sc


class DecompositionServiceTests(unittest.TestCase):
    @staticmethod
    def _collapse_or(formulas: tuple) -> tuple:
        if len(formulas) <= 1:
            return formulas
        merged = formulas[0]
        for formula in formulas[1:]:
            merged = Or(merged, formula)
        return (merged,)

    def test_generate_decomposition_without_beliefs(self) -> None:
        out = generate_decomposition("(p | q) & (p | ~p) & (~r & ~t)", "")
        self.assertIn("\\begin{prooftree}", out)
        self.assertNotIn("_{\\mathbb{B}}", out)

    def test_generate_decomposition_with_beliefs(self) -> None:
        out = generate_decomposition("(p | q) & (s | ~s) & (~r & ~t)", "B={(p | q); ~u}")
        self.assertIn("\\begin{prooftree}", out)
        self.assertIn("_{\\mathbb{B}}", out)

    def test_generate_decomposition_full_beliefs_has_delta(self) -> None:
        out = generate_decomposition(
            "(p | q) & (s | ~s) & (~r & ~t)",
            "B={(p | q); ~u}",
            belief_mode="full",
            decomposition_style="fractional",
        )
        self.assertIn("\\delta_{", out)

    def test_fractional_belief_labels_are_numbered_left_to_right(self) -> None:
        out = generate_decomposition(
            "\\vdash p & q",
            "B={p; q}",
            decomposition_style="fractional",
        )
        self.assertIn("\\RightLabel{\\scriptsize$(b_{1})$}", out)
        self.assertIn("\\RightLabel{\\scriptsize$(b_{2})$}", out)
        self.assertLess(out.index("b_{1}"), out.index("b_{2}"))

    def test_cut_free_belief_labels_are_numbered_left_to_right(self) -> None:
        out = generate_decomposition(
            "\\vdash p & q",
            "B={p; q}",
            decomposition_style="cut_free",
        )
        self.assertIn("\\RightLabel{\\scriptsize$(b_{1})$}", out)
        self.assertIn("\\RightLabel{\\scriptsize$(b_{2})$}", out)
        self.assertLess(out.index("b_{1}"), out.index("b_{2}"))

    def test_full_mode_with_gradient_values_activates_hybrid_behavior(self) -> None:
        out = generate_decomposition(
            "\\vdash p & q",
            "B={p; q}",
            belief_mode="full",
            decomposition_style="fractional",
            belief_gradients_text="0.8",
        )
        self.assertIn("1.8-(\\delta_{", out)
        self.assertIn("+\\delta_{", out)

    def test_generate_decomposition_gradient_mode_uses_numeric_weights(self) -> None:
        out = generate_decomposition(
            "\\vdash p & q",
            "B={p; q}",
            belief_mode="gradient",
            decomposition_style="fractional",
            belief_gradients_text="0.8; 0.7",
        )
        self.assertIn("{1.5}", out)
        self.assertNotIn("\\delta_{", out)

    def test_generate_decomposition_gradient_mode_defaults_missing_values_to_one(self) -> None:
        out = generate_decomposition(
            "\\vdash p & q",
            "B={p; q}",
            belief_mode="gradient",
            decomposition_style="cut_free",
            belief_gradients_text="0.8",
        )
        self.assertIn("{1.8}", out)
        self.assertNotIn("\\delta_{", out)

    def test_generate_decomposition_gradient_mode_rejects_extra_values(self) -> None:
        with self.assertRaises(ValueError):
            _ = generate_decomposition_bundle(
                "\\vdash p",
                "B={p}",
                belief_mode="gradient",
                belief_gradients_text="0.4; 0.7",
            )

    def test_generate_bundle_contains_belief_mode(self) -> None:
        bundle = generate_decomposition_bundle(
            "(p | q) & (s | ~s) & (~r & ~t)",
            "B={(p | q); ~u}",
            belief_mode="full",
        )
        self.assertEqual(bundle["belief_mode"], "full")
        self.assertIn("validation", bundle)
        self.assertTrue(bundle["validation"]["ok"])
        self.assertIn("equivalence", bundle)

    def test_root_sequent_keeps_explicit_grouping(self) -> None:
        out = generate_decomposition(
            "(p AND q) OR (r AND not r) OR (s AND NOT s) AND (s AND q)",
            "",
            decomposition_style="fractional",
        )
        self.assertIn(
            "p \\wedge q \\vee r \\wedge \\neg r \\vee s \\wedge \\neg s \\wedge (s \\wedge q)",
            out,
        )

    def test_visual_tree_root_sequent_keeps_explicit_grouping(self) -> None:
        bundle = generate_decomposition_bundle(
            "(p AND q) OR (r AND not r) OR (s AND NOT s) AND (s AND q)",
            "",
            decomposition_style="fractional",
        )
        root = bundle["tree"]
        self.assertIn(
            "p & q | r & ~r | s & ~s & (s & q)",
            str(root.get("display", "")),
        )

    def test_cut_free_decomposition_style_has_no_commas_in_root(self) -> None:
        bundle = generate_decomposition_bundle(
            "p |- q, (r & s)",
            "",
            decomposition_style="cut_free",
        )
        self.assertEqual(bundle["decomposition_style"], "cut_free")
        self.assertTrue(bundle["validation"]["ok"])
        self.assertIn("\\RightLabel{\\scriptsize$(\\vee)$}", str(bundle["decomposition"]))
        self.assertIn("\\RightLabel{\\scriptsize$(\\wedge)$}", str(bundle["decomposition"]))
        root_display = str(bundle["tree"].get("display", ""))
        self.assertIn("(~p | q | r) & (~p | q | s)", root_display)
        self.assertNotIn(",", root_display)
        self.assertIn(
            "root shown in cut-free normal form",
            str(bundle["validation"]["message"]),
        )

    def test_cut_free_or_steps_keep_remaining_comma_context(self) -> None:
        bundle = generate_decomposition_bundle(
            "r & (u & ~s) \\vdash v | (t | p) | (u | s)",
            "",
            decomposition_style="cut_free",
        )
        decomposition = str(bundle["decomposition"])
        self.assertIn("\\neg r \\vee \\neg u, s, v, t, p, u, s", decomposition)
        self.assertIn("\\neg r \\vee \\neg u \\vee s, v, t, p, u, s", decomposition)
        self.assertIn("\\neg r \\vee \\neg u \\vee s \\vee v, t, p, u, s", decomposition)
        self.assertIn("\\neg r \\vee \\neg u \\vee s \\vee v \\vee t, p, u, s", decomposition)
        self.assertIn("\\neg r \\vee \\neg u \\vee s \\vee v \\vee t \\vee p, u, s", decomposition)
        self.assertIn("\\neg r \\vee \\neg u \\vee s \\vee v \\vee t \\vee p \\vee u, s", decomposition)
        self.assertIn("\\neg r \\vee \\neg u \\vee s \\vee v \\vee t \\vee p \\vee u \\vee s", decomposition)
        self.assertIn(
            "root shown in cut-free normal form",
            str(bundle["validation"]["message"]),
        )

    def test_cut_free_leaf_clauses_match_reference_sc(self) -> None:
        examples = [
            "|- (p | q) & (r | ~r)",
            "p \\vdash q, (r & s)",
            "r & (u & ~s) \\vdash v | (t | p) | (u | s)",
        ]
        for sequent in examples:
            parsed = parse_sequent_structured(sequent)
            normalized = tuple(to_nnf(formula) for formula in parsed.one_sided)
            collapsed = self._collapse_or(normalized)
            cut_free_root = build_cut_free_tree(
                collapsed[0],
                belief_clauses=set(),
            )
            cut_counts = Counter(collect_cut_free_leaf_clauses(cut_free_root))
            reference = evaluate_reference_sc(sequent)
            ref_counts = Counter(leaf.clause for leaf in reference.leaves)
            self.assertEqual(cut_counts, ref_counts, sequent)

    def test_cut_free_full_mode_delta_keeps_neg_literal_latex_spacing(self) -> None:
        out = generate_decomposition(
            "\\vdash (r | p | ~r) & ~s & w",
            "B={~s}",
            belief_mode="full",
            decomposition_style="cut_free",
        )
        self.assertIn("\\delta_{", out)
        self.assertIn("\\neg s", out)
        self.assertNotIn("\\negs", out)

    def test_cut_free_full_mode_single_delta_has_no_parentheses(self) -> None:
        out = generate_decomposition(
            "\\vdash p",
            "B={p}",
            belief_mode="full",
            decomposition_style="cut_free",
        )
        self.assertIn("1-\\delta_{", out)
        self.assertNotIn("1-(\\delta_{", out)

    def test_equivalence_chain_present_when_root_changes(self) -> None:
        bundle = generate_decomposition_bundle(
            "p |- q, (r & s)",
            "",
            decomposition_style="cut_free",
        )
        eq = bundle["equivalence"]
        self.assertTrue(eq["changed"])
        self.assertIn("~p | q | r & s", eq["ascii_chain"])
        self.assertIn("(~p | q | r) & (~p | q | s)", eq["ascii_chain"])

    def test_equivalence_chain_not_reported_when_root_unchanged(self) -> None:
        bundle = generate_decomposition_bundle(
            "\\vdash p | q",
            "",
            decomposition_style="cut_free",
        )
        eq = bundle["equivalence"]
        self.assertFalse(eq["changed"])


if __name__ == "__main__":
    unittest.main()
