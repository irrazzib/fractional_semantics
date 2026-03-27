from __future__ import annotations

import unittest

from fs_semantics.beliefs import BeliefBase
from fs_semantics.propositional_sc import (
    compare_with_fractional,
    evaluate_reference_sc,
    render_reference_tree_ascii,
)


class PropositionalSCTests(unittest.TestCase):
    def test_reference_decomposition_counts(self) -> None:
        result = evaluate_reference_sc("(p | q) & (p | ~p)")
        self.assertEqual(result.total, 2)
        self.assertEqual(result.ax, 1)
        self.assertEqual(result.open_, 1)

    def test_reference_tree_rendering(self) -> None:
        result = evaluate_reference_sc("p, q |- r, (s & t)")
        tree = render_reference_tree_ascii(result.root)
        self.assertIn("|-", tree)
        self.assertIn("(andR)", tree)

    def test_compare_matches_without_beliefs(self) -> None:
        comparison = compare_with_fractional("p, q |- r, (s & t)")
        self.assertTrue(comparison.matches)
        self.assertEqual(comparison.reference_counts, comparison.fractional_counts)

    def test_compare_matches_with_beliefs(self) -> None:
        base = BeliefBase.from_strings(["(p | q)", "~u"])
        comparison = compare_with_fractional(
            "(p | q) & (s | ~s) & (~r & ~t)",
            belief_base=base,
        )
        self.assertTrue(comparison.matches)
        self.assertEqual(comparison.reference_counts, comparison.fractional_counts)
        self.assertGreaterEqual(comparison.fractional.topb, 1)


if __name__ == "__main__":
    unittest.main()

