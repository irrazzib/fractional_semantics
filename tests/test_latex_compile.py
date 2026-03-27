from __future__ import annotations

import shutil
import unittest

from fs_semantics.latex_compile import (
    compile_decomposition_pdf,
    compile_decomposition_png,
)


class LatexCompileTests(unittest.TestCase):
    def test_compile_pdf_rejects_invalid_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "Scale must be in"):
            compile_decomposition_pdf("\\AxiomC{}", scale=0)

    def test_compile_png_rejects_invalid_scale(self) -> None:
        with self.assertRaisesRegex(ValueError, "Scale must be in"):
            compile_decomposition_png("\\AxiomC{}", scale=2)

    @unittest.skipIf(shutil.which("pdflatex") is None, "pdflatex not available")
    def test_compile_simple_prooftree(self) -> None:
        decomposition = (
            "\\begin{prooftree}\n"
            "\\AxiomC{}\n"
            "\\RightLabel{\\scriptsize$(ax.)$}\n"
            "\\UnaryInfC{$\\sststile{\\mathrm{1}}{1}\\; p, \\neg p$}\n"
            "\\end{prooftree}"
        )
        pdf = compile_decomposition_pdf(decomposition)
        self.assertGreater(len(pdf), 1000)
        self.assertTrue(pdf.startswith(b"%PDF"))

    @unittest.skipIf(
        shutil.which("pdflatex") is None
        or (
            shutil.which("pdftoppm") is None
            and shutil.which("magick") is None
            and shutil.which("sips") is None
            and shutil.which("gs") is None
        ),
        "png preview toolchain not available",
    )
    def test_compile_simple_prooftree_png(self) -> None:
        decomposition = (
            "\\begin{prooftree}\n"
            "\\AxiomC{}\n"
            "\\RightLabel{\\scriptsize$(ax.)$}\n"
            "\\UnaryInfC{$\\sststile{\\mathrm{1}}{1}\\; p, \\neg p$}\n"
            "\\end{prooftree}"
        )
        png = compile_decomposition_png(decomposition)
        self.assertGreater(len(png), 1000)
        self.assertTrue(png.startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
