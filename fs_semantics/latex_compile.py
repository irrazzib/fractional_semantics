from __future__ import annotations

from pathlib import Path
import re
import shutil
import subprocess
import tempfile


_PROOFTREE_RE = re.compile(
    r"^\s*\\begin\{prooftree\}(?P<body>.*?)\\end\{prooftree\}\s*$",
    re.DOTALL,
)


def _extract_prooftree_body(decomposition: str) -> str:
    text = decomposition.strip()
    if not text:
        raise ValueError("Missing decomposition.")
    match = _PROOFTREE_RE.match(text)
    if match:
        return match.group("body").strip()
    return text


def _paper_size_for_prooftree(body: str) -> tuple[float, float]:
    leaves = max(1, body.count("\\AxiomC"))
    branches = body.count("\\UnaryInfC") + body.count("\\BinaryInfC")
    longest_line = max((len(line) for line in body.splitlines()), default=0)

    # Keep the canvas large enough for wide trees while avoiding massive blank space.
    width_in = 2.0 + (1.1 * leaves) + (0.02 * longest_line)
    height_in = 7.0 + (0.22 * max(1, branches))

    width_in = min(140.0, max(22.0, width_in))
    height_in = min(60.0, max(12.0, height_in))
    return width_in, height_in


def _preview_document(decomposition: str, scale: float) -> str:
    body = _extract_prooftree_body(decomposition)
    paper_width_in, paper_height_in = _paper_size_for_prooftree(body)
    return f"""\\documentclass[12pt]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[paperwidth={paper_width_in:.3f}in,paperheight={paper_height_in:.3f}in,margin=0.35in]{{geometry}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{stmaryrd}}
\\usepackage{{turnstile}}
\\usepackage{{bussproofs}}
\\usepackage{{graphicx}}
\\usepackage[active,tightpage]{{preview}}
\\setlength\\PreviewBorder{{18pt}}
\\newenvironment{{scprooftree}}[1]%
{{\\gdef\\scalefactor{{#1}}\\begin{{center}}\\proofSkipAmount \\leavevmode}}%
{{\\scalebox{{\\scalefactor}}{{\\DisplayProof}}\\proofSkipAmount \\end{{center}} }}
\\PreviewEnvironment{{scprooftree}}
\\pagestyle{{empty}}
\\begin{{document}}
\\begin{{scprooftree}}{{{scale}}}
{body}
\\end{{scprooftree}}
\\end{{document}}
"""


def _compile_pdf_in_workdir(workdir: Path, decomposition: str, scale: float) -> Path:
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        raise RuntimeError("pdflatex was not found in PATH.")

    tex_path = workdir / "decomposition_preview.tex"
    tex_path.write_text(_preview_document(decomposition, scale=scale), encoding="utf-8")

    cmd = [
        pdflatex,
        "-interaction=nonstopmode",
        "-halt-on-error",
        tex_path.name,
    ]
    result = subprocess.run(
        cmd,
        cwd=workdir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        tail = "\n".join(result.stdout.splitlines()[-25:])
        raise RuntimeError(f"LaTeX compilation failed.\n{tail}")

    pdf_path = workdir / "decomposition_preview.pdf"
    if not pdf_path.exists():
        raise RuntimeError("LaTeX compilation finished but no PDF was produced.")
    return pdf_path


def compile_decomposition_pdf(decomposition: str, scale: float = 0.88) -> bytes:
    try:
        scale_value = float(scale)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Scale must be a number.") from exc
    if scale_value <= 0 or scale_value > 1.0:
        raise ValueError("Scale must be in (0, 1].")

    with tempfile.TemporaryDirectory(prefix="fs_preview_") as tmp:
        workdir = Path(tmp)
        pdf_path = _compile_pdf_in_workdir(workdir, decomposition, scale=scale_value)
        return pdf_path.read_bytes()


def compile_decomposition_png(
    decomposition: str,
    dpi: int = 220,
    scale: float = 0.88,
) -> bytes:
    try:
        dpi_value = int(dpi)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("DPI must be an integer.") from exc
    if dpi_value < 72 or dpi_value > 1200:
        raise ValueError("DPI must be between 72 and 1200.")
    try:
        scale_value = float(scale)
    except Exception as exc:  # noqa: BLE001
        raise ValueError("Scale must be a number.") from exc
    if scale_value <= 0 or scale_value > 1.0:
        raise ValueError("Scale must be in (0, 1].")

    with tempfile.TemporaryDirectory(prefix="fs_preview_") as tmp:
        workdir = Path(tmp)
        pdf_path = _compile_pdf_in_workdir(workdir, decomposition, scale=scale_value)

        pdftoppm = shutil.which("pdftoppm")
        if pdftoppm:
            out_prefix = workdir / "decomposition_preview"
            cmd = [
                pdftoppm,
                "-singlefile",
                "-png",
                "-rx",
                str(dpi_value),
                "-ry",
                str(dpi_value),
                str(pdf_path),
                str(out_prefix),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                tail = "\n".join(result.stdout.splitlines()[-25:])
                raise RuntimeError(f"PNG conversion failed.\n{tail}")

            png_path = out_prefix.with_suffix(".png")
            if not png_path.exists():
                raise RuntimeError("PNG conversion finished but no PNG was produced.")
            return png_path.read_bytes()

        magick = shutil.which("magick")
        if magick:
            png_path = workdir / "decomposition_preview.png"
            cmd = [
                magick,
                "-density",
                str(dpi_value),
                str(pdf_path),
                "-quality",
                "100",
                str(png_path),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                tail = "\n".join(result.stdout.splitlines()[-25:])
                raise RuntimeError(f"PNG conversion failed.\n{tail}")
            if not png_path.exists():
                raise RuntimeError("PNG conversion finished but no PNG was produced.")
            return png_path.read_bytes()

        gs = shutil.which("gs")
        if gs:
            png_path = workdir / "decomposition_preview.png"
            cmd = [
                gs,
                "-dBATCH",
                "-dNOPAUSE",
                "-sDEVICE=pngalpha",
                f"-r{dpi_value}",
                f"-sOutputFile={png_path}",
                str(pdf_path),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                tail = "\n".join(result.stdout.splitlines()[-25:])
                raise RuntimeError(f"PNG conversion failed.\n{tail}")
            if not png_path.exists():
                raise RuntimeError("PNG conversion finished but no PNG was produced.")
            return png_path.read_bytes()

        sips = shutil.which("sips")
        if sips:
            png_path = workdir / "decomposition_preview.png"
            cmd = [
                sips,
                "-s",
                "format",
                "png",
                str(pdf_path),
                "--out",
                str(png_path),
            ]
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            if result.returncode != 0:
                tail = "\n".join(result.stdout.splitlines()[-25:])
                raise RuntimeError(f"PNG conversion failed.\n{tail}")
            if not png_path.exists():
                raise RuntimeError("PNG conversion finished but no PNG was produced.")
            return png_path.read_bytes()

        raise RuntimeError(
            "No PDF->PNG converter found. Install one of: `pdftoppm` (poppler), "
            "`magick`, `sips`, or `gs`."
        )
