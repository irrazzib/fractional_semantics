from __future__ import annotations

import random
import unittest

from fs_semantics.logic import And, Formula, Or, parse_sequent_structured
from fs_semantics.web_gui import (
    generate_random_coherent_sequent,
    generate_random_incoherent_sequent,
    generate_random_sequent,
    is_coherent_formula,
)


def _has_repeated_binary_subformula(formula: Formula) -> bool:
    if isinstance(formula, And):
        return (
            formula.left == formula.right
            or _has_repeated_binary_subformula(formula.left)
            or _has_repeated_binary_subformula(formula.right)
        )
    if isinstance(formula, Or):
        return (
            formula.left == formula.right
            or _has_repeated_binary_subformula(formula.left)
            or _has_repeated_binary_subformula(formula.right)
        )
    # Atom / Not
    inner = getattr(formula, "arg", None)
    if isinstance(inner, Formula):
        return _has_repeated_binary_subformula(inner)
    return False


class WebGuiRandomSequentTests(unittest.TestCase):
    def test_generate_random_sequent_is_parsable(self) -> None:
        rng = random.Random(20260302)
        for _ in range(80):
            sequent = generate_random_sequent(rng=rng)
            parsed = parse_sequent_structured(sequent)
            self.assertGreaterEqual(len(parsed.succedent), 1)
            self.assertTrue(parsed.is_two_sided)
            self.assertIn("\\vdash", sequent)
            self.assertNotIn(",", sequent)
            self.assertEqual(len(parsed.antecedent), 0)

    def test_generate_random_sequent_has_non_empty_both_sides(self) -> None:
        rng = random.Random(17)
        for _ in range(100):
            sequent = generate_random_sequent(rng=rng)
            parsed = parse_sequent_structured(sequent)
            self.assertTrue(parsed.is_two_sided)
            self.assertEqual(len(parsed.antecedent), 0)
            self.assertGreaterEqual(len(parsed.succedent), 1)

    def test_generate_random_sequent_avoids_trivial_binary_repetitions(self) -> None:
        rng = random.Random(90210)
        for _ in range(120):
            sequent = generate_random_sequent(rng=rng)
            parsed = parse_sequent_structured(sequent)
            for formula in (*parsed.antecedent, *parsed.succedent):
                self.assertFalse(_has_repeated_binary_subformula(formula), sequent)

    def test_generate_random_coherent_sequent_has_no_contradictions(self) -> None:
        rng = random.Random(314159)
        for _ in range(120):
            sequent = generate_random_coherent_sequent(rng=rng)
            parsed = parse_sequent_structured(sequent)
            for formula in (*parsed.antecedent, *parsed.succedent):
                self.assertTrue(is_coherent_formula(formula), sequent)

    def test_generate_random_incoherent_sequent_is_incoherent(self) -> None:
        rng = random.Random(271828)
        for _ in range(80):
            sequent = generate_random_incoherent_sequent(rng=rng)
            parsed = parse_sequent_structured(sequent)
            self.assertTrue(
                any(not is_coherent_formula(formula) for formula in parsed.succedent),
                sequent,
            )

    def test_generate_random_sequent_allows_incoherent_when_requested(self) -> None:
        rng = random.Random(1618033)
        found_incoherent = False
        for _ in range(80):
            sequent = generate_random_sequent(rng=rng, allow_incoherent=True)
            parsed = parse_sequent_structured(sequent)
            if any(not is_coherent_formula(formula) for formula in parsed.succedent):
                found_incoherent = True
                break
        self.assertTrue(found_incoherent)
