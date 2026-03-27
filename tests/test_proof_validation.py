from __future__ import annotations

import unittest

from fs_semantics.calculus import ProofNode, evaluate_sequent
from fs_semantics.logic import parse_formula, parse_sequent, to_nnf
from fs_semantics.proof_validation import validate_proof_tree


class ProofValidationTests(unittest.TestCase):
    def test_parentheses_aware_validation_passes(self) -> None:
        source = parse_sequent("(p | q) & (s | ~s) & (~r & ~t)")
        result = evaluate_sequent(source, belief_clauses=set())
        expected = tuple(to_nnf(formula) for formula in source)
        validation = validate_proof_tree(
            result.root,
            expected_root=expected,
            source_sequent=source,
        )
        self.assertTrue(validation.ok)
        self.assertIn("Parentheses-aware decomposition check passed", validation.message)

    def test_validation_detects_root_mismatch(self) -> None:
        source = parse_sequent("(p | q)")
        result = evaluate_sequent(source, belief_clauses=set())
        wrong_expected = tuple(to_nnf(formula) for formula in parse_sequent("(p & q)"))
        validation = validate_proof_tree(
            result.root,
            expected_root=wrong_expected,
            source_sequent=source,
        )
        self.assertFalse(validation.ok)
        self.assertIn("Root sequent mismatch", validation.message)

    def test_validation_accepts_exchange_on_or_step(self) -> None:
        parent = parse_sequent("p | q, r")
        child = (
            parse_formula("r"),
            parse_formula("q"),
            parse_formula("p"),
        )
        tree = ProofNode(
            sequent=tuple(to_nnf(formula) for formula in parent),
            rule="vee",
            children=[ProofNode(sequent=child, rule="leaf", children=[])],
        )
        validation = validate_proof_tree(
            tree,
            expected_root=tuple(to_nnf(formula) for formula in parent),
            source_sequent=parent,
        )
        self.assertTrue(validation.ok)

    def test_validation_accepts_exchange_on_and_step(self) -> None:
        parent = parse_sequent("p & q, r")
        left = (parse_formula("r"), parse_formula("p"))
        right = (parse_formula("q"), parse_formula("r"))
        tree = ProofNode(
            sequent=tuple(to_nnf(formula) for formula in parent),
            rule="wedge",
            children=[
                ProofNode(sequent=left, rule="leaf", children=[]),
                ProofNode(sequent=right, rule="leaf", children=[]),
            ],
        )
        validation = validate_proof_tree(
            tree,
            expected_root=tuple(to_nnf(formula) for formula in parent),
            source_sequent=parent,
        )
        self.assertTrue(validation.ok)


if __name__ == "__main__":
    unittest.main()
