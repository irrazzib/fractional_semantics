from __future__ import annotations

import unittest

from fs_semantics.afp_certificate import build_afp_certificate
from fs_semantics.calculus import evaluate_sequent
from fs_semantics.logic import parse_sequent


class AFPCertificateTests(unittest.TestCase):
    def test_build_certificate_contains_sc_structure(self) -> None:
        sequent = parse_sequent("(p | q) & (r | ~r)")
        result = evaluate_sequent(sequent, belief_clauses=set())
        cert = build_afp_certificate(
            result,
            source_sequent_label="(p | q) & (r | ~r)",
            theory_name="my-cert",
        )
        self.assertEqual(cert.theory_name, "my_cert")
        self.assertEqual(len(cert.leaf_assumptions), result.top0 + result.topb)
        self.assertIn('imports "Propositional_Proof_Systems.SC"', cert.theory_text)
        self.assertIn("theorem generated_tree_certificate", cert.theory_text)
        self.assertIn("SCp.AndR", cert.theory_text)
        self.assertIn("SCp.OrR", cert.theory_text)


if __name__ == "__main__":
    unittest.main()
