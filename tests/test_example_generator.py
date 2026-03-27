from __future__ import annotations

import unittest
from unittest.mock import patch

from fs_semantics.example_generator import generate_example_from_sequent


class ExampleGeneratorTests(unittest.TestCase):
    def test_generate_example_template_mode(self) -> None:
        result = generate_example_from_sequent(
            "p |- q, (r & s)",
            "B={p}",
            prefer_local_model=False,
        )
        self.assertEqual(result.engine, "template")
        self.assertTrue(result.text.strip())
        self.assertIn("Mapping:", result.text)
        self.assertIn("Check:", result.text)

    def test_generate_example_requires_sequent(self) -> None:
        with self.assertRaises(ValueError):
            generate_example_from_sequent("", prefer_local_model=False)

    def test_generate_example_template_only_mode(self) -> None:
        result = generate_example_from_sequent(
            "p |- q",
            generation_mode="template_only",
        )
        self.assertEqual(result.engine, "template")
        lower = result.text.lower()
        self.assertTrue(
            any(
                marker in lower
                for marker in (
                    "everyday",
                    "daily-life",
                    "morning commute",
                    "grocery",
                    "home routine",
                    "study planning",
                )
            )
        )

    def test_generate_example_with_belief_set_changes_output(self) -> None:
        result = generate_example_from_sequent(
            "\\vdash p | q",
            "B={p}",
            generation_mode="template_only",
        )
        self.assertIn("prior beliefs are true", result.text)
        self.assertIn("B=p", result.text)

    def test_generate_example_invalid_mode_raises(self) -> None:
        with self.assertRaises(ValueError):
            generate_example_from_sequent(
                "p |- q",
                generation_mode="invalid",
            )

    @patch("fs_semantics.example_generator._generate_with_ollama", return_value=None)
    def test_generate_example_local_only_raises_when_model_unavailable(self, _: object) -> None:
        with self.assertRaises(RuntimeError):
            generate_example_from_sequent(
                "p |- q",
                generation_mode="local_only",
            )


if __name__ == "__main__":
    unittest.main()
