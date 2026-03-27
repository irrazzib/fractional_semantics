from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

from .afp_certificate import build_afp_certificate
from .beliefs import BeliefBase
from .calculus import evaluate_sequent
from .input_parsing import load_belief_strings
from .latex import render_analysis_document, render_revision_document
from .logic import Formula, parse_formula, parse_sequent, parse_sequent_structured
from .propositional_sc import (
    compare_with_fractional,
    evaluate_reference_sc,
    render_reference_tree_ascii,
)


DEFAULT_OUTPUT_DIR = Path("tex_outputs")


def _new_run_dir(tag: str) -> Path:
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = DEFAULT_OUTPUT_DIR / f"{tag}_{stamp}"
    candidate = base
    counter = 1
    while candidate.exists():
        candidate = DEFAULT_OUTPUT_DIR / f"{tag}_{stamp}_{counter:02d}"
        counter += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _resolve_output_in_run(run_dir: Path, requested: str) -> Path:
    name = Path(requested).name
    if not name.lower().endswith(".tex"):
        name += ".tex"
    return run_dir / name


def _resolve_named_output(run_dir: Path, requested: str, extension: str) -> Path:
    name = Path(requested).name
    if extension and not name.lower().endswith(extension.lower()):
        name += extension
    return run_dir / name


def _write_output(path: Path, content: str) -> None:
    out = path
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    print(f"Output written to: {out.resolve()}")


def _compile_tex(path: Path) -> None:
    pdflatex = shutil.which("pdflatex")
    if not pdflatex:
        print("pdflatex not found: automatic compilation skipped.")
        return

    cmd = [pdflatex, "-interaction=nonstopmode", "-halt-on-error", path.name]
    result = subprocess.run(
        cmd,
        cwd=path.parent,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
        check=False,
    )

    if result.returncode != 0:
        print(f"LaTeX compilation failed: {path.resolve()}")
        log_path = path.with_suffix(".log")
        if log_path.exists():
            print(f"Check log: {log_path.resolve()}")
        return

    pdf_path = path.with_suffix(".pdf")
    if pdf_path.exists():
        print(f"PDF compiled: {pdf_path.resolve()}")


def _write_isabelle_root(run_dir: Path, session_name: str) -> Path:
    root_text = (
        "chapter AFP\n\n"
        f"session {session_name} = Propositional_Proof_Systems +\n"
        "  theories\n"
        f"    {session_name}\n"
    )
    path = run_dir / "ROOT"
    path.write_text(root_text, encoding="utf-8")
    return path


def _run_isabelle_build(run_dir: Path, afp_thys: Path, session_name: str) -> bool:
    isabelle = shutil.which("isabelle")
    if not isabelle:
        print("isabelle was not found in PATH; AFP verification skipped.")
        return False
    cmd = [
        isabelle,
        "build",
        "-d",
        str(afp_thys),
        "-D",
        str(run_dir),
        session_name,
    ]
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode == 0:
        print("AFP verification: OK (Isabelle build succeeded).")
        return True

    print("AFP verification failed (Isabelle build returned non-zero).")
    output = result.stdout.strip().splitlines()
    tail = "\n".join(output[-30:]) if output else "(no build output)"
    print(tail)
    return False


def _apply_revision_operation(
    base: BeliefBase,
    operation: str,
    target: Formula,
) -> BeliefBase:
    if operation == "contract":
        return base.contract(target)
    if operation == "expand":
        return base.expand(target)
    if operation == "revise":
        return base.revise(target)
    raise ValueError(f"Unsupported operation: {operation}")


def _format_clause_counter(counter: Counter) -> str:
    if not counter:
        return "∅"

    def lit_key(literal: object) -> tuple[str, int]:
        atom = getattr(literal, "atom", "")
        neg = 1 if getattr(literal, "negated", False) else 0
        return (atom, neg)

    parts: list[str] = []
    for clause, count in sorted(
        counter.items(),
        key=lambda item: (
            len(item[0]),
            tuple(lit_key(lit) for lit in sorted(item[0], key=lit_key)),
        ),
    ):
        lits = sorted(clause, key=lit_key)
        text = " | ".join(
            f"~{lit.atom}" if getattr(lit, "negated", False) else str(lit.atom)
            for lit in lits
        )
        if not text:
            text = "⊥"
        suffix = f" x{count}" if count > 1 else ""
        parts.append(f"{{{text}}}{suffix}")
    return ", ".join(parts)


def run_analyze(args: argparse.Namespace) -> int:
    run_dir = _new_run_dir("analysis")
    beliefs = load_belief_strings(args.belief, args.belief_file, args.belief_set)
    base = BeliefBase.from_strings(beliefs, close_under_cut=not args.no_cut_closure)
    sequent = parse_sequent(args.sequent)
    result = evaluate_sequent(sequent, belief_clauses=set(base.clauses))
    tex = render_analysis_document(args.sequent, result, base if beliefs else None)
    out_path = _resolve_output_in_run(run_dir, args.output)
    _write_output(out_path, tex)
    _compile_tex(out_path)
    print(f"Run folder: {run_dir.resolve()}")
    print(f"Belief set B: {beliefs if beliefs else '∅'}")
    print(
        "Values: "
        f"[[Gamma]] = {result.value} ({float(result.value):.6g}), "
        f"[[Gamma]]_B = {result.value_b} ({float(result.value_b):.6g})"
    )
    return 0


def run_revise(args: argparse.Namespace) -> int:
    run_dir = _new_run_dir("revision")
    beliefs = load_belief_strings(args.belief, args.belief_file, args.belief_set)
    base = BeliefBase.from_strings(beliefs, close_under_cut=not args.no_cut_closure)
    target = parse_formula(args.target)
    sequent = parse_sequent(args.sequent)
    before = evaluate_sequent(sequent, belief_clauses=set(base.clauses))
    after_base = _apply_revision_operation(base, args.operation, target)
    after = evaluate_sequent(sequent, belief_clauses=set(after_base.clauses))
    tex = render_revision_document(
        operation=args.operation,
        target_label=args.target,
        sequent_label=args.sequent,
        before_base=base,
        after_base=after_base,
        before_result=before,
        after_result=after,
    )
    out_path = _resolve_output_in_run(run_dir, args.output)
    _write_output(out_path, tex)
    _compile_tex(out_path)
    print(f"Run folder: {run_dir.resolve()}")
    print(f"Belief set B: {beliefs if beliefs else '∅'}")
    print(
        f"Before: {before.value_b} ({float(before.value_b):.6g}) | "
        f"After: {after.value_b} ({float(after.value_b):.6g})"
    )
    return 0


def run_revision_demo(args: argparse.Namespace) -> int:
    run_dir = _new_run_dir("revision_demo")
    p = args.atom_p
    q = args.atom_q
    sequent_label = f"({p} & {q})"
    sequent = parse_sequent(sequent_label)
    output_prefix = Path(args.output_prefix).name

    scenarios = [
        ("contract", [p, q], p, f"{output_prefix}_contraction.tex"),
        ("expand", [q], p, f"{output_prefix}_expansion.tex"),
        ("revise", [f"~{p}", q], p, f"{output_prefix}_revision.tex"),
    ]

    for operation, beliefs, target_label, output in scenarios:
        base = BeliefBase.from_strings(beliefs, close_under_cut=not args.no_cut_closure)
        target_formula = parse_formula(target_label)
        before = evaluate_sequent(sequent, belief_clauses=set(base.clauses))
        after_base = _apply_revision_operation(base, operation, target_formula)
        after = evaluate_sequent(sequent, belief_clauses=set(after_base.clauses))
        tex = render_revision_document(
            operation=operation,
            target_label=target_label,
            sequent_label=sequent_label,
            before_base=base,
            after_base=after_base,
            before_result=before,
            after_result=after,
        )
        out_path = _resolve_output_in_run(run_dir, output)
        _write_output(out_path, tex)
        _compile_tex(out_path)
        print(
            f"[{operation}] {out_path.name}: "
            f"{before.value_b} -> {after.value_b}"
        )
    print(f"Run folder: {run_dir.resolve()}")
    return 0


def run_gui(_: argparse.Namespace) -> int:
    try:
        from .gui import launch_gui

        return int(launch_gui())
    except ModuleNotFoundError as exc:
        if exc.name not in {"_tkinter", "tkinter"}:
            raise
        print("Tkinter is unavailable in this Python build; starting Web UI fallback...")
        from .web_gui import launch_web_gui

        try:
            return int(launch_web_gui())
        except Exception as web_exc:  # noqa: BLE001
            print(f"Web UI launch failed: {web_exc}")
            return 1


def run_web_gui(_: argparse.Namespace) -> int:
    from .web_gui import launch_web_gui

    try:
        return int(launch_web_gui())
    except Exception as exc:  # noqa: BLE001
        print(f"Web UI launch failed: {exc}")
        return 1


def run_web_app(_: argparse.Namespace) -> int:
    from .web_app import launch_web_app

    try:
        return int(launch_web_app())
    except Exception as exc:  # noqa: BLE001
        print(f"Web app launch failed: {exc}")
        return 1


def run_sc_reference(args: argparse.Namespace) -> int:
    result = evaluate_reference_sc(args.sequent)
    print("Reference propositional sequent-calculus decomposition")
    print(f"Leaves: total={result.total}, ax={result.ax}, open={result.open_}")
    if args.show_tree:
        print(render_reference_tree_ascii(result.root))
    return 0


def run_sc_compare(args: argparse.Namespace) -> int:
    beliefs = load_belief_strings(args.belief, args.belief_file, args.belief_set)
    base = BeliefBase.from_strings(beliefs, close_under_cut=not args.no_cut_closure)
    comparison = compare_with_fractional(
        args.sequent,
        belief_base=base if beliefs else None,
    )
    print("SC comparison (reference propositional sequent calculus vs Fractional engine)")
    print(
        "Reference leaves: "
        f"total={comparison.reference.total}, ax={comparison.reference.ax}, "
        f"open={comparison.reference.open_}"
    )
    print(
        "Fractional leaves: "
        f"total={comparison.fractional.total}, top1={comparison.fractional.top1}, "
        f"topb={comparison.fractional.topb}, top0={comparison.fractional.top0}"
    )
    print(
        "Fractional values: "
        f"[[Gamma]]={comparison.fractional.value} "
        f"[[Gamma]]_B={comparison.fractional.value_b}"
    )
    print(f"Leaf-clause multiset match: {'YES' if comparison.matches else 'NO'}")

    if comparison.matches:
        print(
            "Shared leaf clauses: "
            f"{_format_clause_counter(comparison.reference_counts)}"
        )
    else:
        print("Only in Fractional decomposition:")
        print(f"  {_format_clause_counter(comparison.only_fractional)}")
        print("Only in reference decomposition:")
        print(f"  {_format_clause_counter(comparison.only_reference)}")

    if args.show_tree:
        print("")
        print("Reference tree")
        print(render_reference_tree_ascii(comparison.reference.root))

    return 0


def run_afp_certify(args: argparse.Namespace) -> int:
    run_dir = _new_run_dir("afp_certificate")
    beliefs = load_belief_strings(args.belief, args.belief_file, args.belief_set)
    base = BeliefBase.from_strings(beliefs, close_under_cut=not args.no_cut_closure)
    parsed = parse_sequent_structured(args.sequent)
    sequent = parsed.one_sided
    result = evaluate_sequent(sequent, belief_clauses=set(base.clauses))
    cert = build_afp_certificate(
        result,
        source_sequent_label=args.sequent,
        theory_name=args.theory_name,
    )
    output_name = args.output or cert.theory_name
    thy_path = _resolve_named_output(run_dir, output_name, ".thy")
    _write_output(thy_path, cert.theory_text)
    root_path = _write_isabelle_root(run_dir, cert.theory_name)
    print(f"Isabelle ROOT written to: {root_path.resolve()}")
    print(
        "Leaf assumptions in certificate: "
        f"{len(cert.leaf_assumptions)} "
        f"(top^1={result.top1}, top^b={result.topb}, top^0={result.top0})"
    )
    print(f"Run folder: {run_dir.resolve()}")

    if not args.verify:
        return 0

    afp_thys_raw = args.afp_thys or os.environ.get("AFP_THYS")
    if not afp_thys_raw:
        print(
            "AFP verification requested but no AFP thys directory was provided. "
            "Use --afp-thys or set AFP_THYS."
        )
        return 1
    afp_thys = Path(afp_thys_raw).expanduser().resolve()
    if not afp_thys.exists():
        print(f"AFP thys directory not found: {afp_thys}")
        return 1
    ok = _run_isabelle_build(run_dir, afp_thys, cert.theory_name)
    return 0 if ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Automatic Fractional Semantics calculator with belief base B."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--belief",
        action="append",
        default=[],
        help="Belief formula to add to B (repeatable).",
    )
    common.add_argument(
        "--belief-file",
        default=None,
        help="Text file with one belief per line.",
    )
    common.add_argument(
        "--belief-set",
        default=None,
        help=(
            "Belief set in one argument, e.g. "
            "\"B={(p | q); ~u}\" or \"(p | q); ~u\"."
        ),
    )
    common.add_argument(
        "--no-cut-closure",
        action="store_true",
        help="Disable closure of B under cut (resolution closure).",
    )

    analyze = sub.add_parser(
        "analyze",
        parents=[common],
        help="Analyze a sequent and generate a LaTeX report.",
    )
    analyze.add_argument("--sequent", required=True, help="Sequent to evaluate.")
    analyze.add_argument(
        "--output",
        default="fractional_report.tex",
        help="Output .tex filename inside the auto-created run folder.",
    )
    analyze.set_defaults(func=run_analyze)

    revise = sub.add_parser(
        "revise",
        parents=[common],
        help="Apply a belief revision operation and generate a LaTeX report.",
    )
    revise.add_argument(
        "--operation",
        choices=["contract", "expand", "revise"],
        required=True,
        help="Operation to apply to B.",
    )
    revise.add_argument("--target", required=True, help="Target formula.")
    revise.add_argument("--sequent", required=True, help="Sequent to compare.")
    revise.add_argument(
        "--output",
        default="belief_revision_report.tex",
        help="Output .tex filename inside the auto-created run folder.",
    )
    revise.set_defaults(func=run_revise)

    demo = sub.add_parser(
        "revision-demo",
        help="Generate three automatic examples (contraction/expansion/revision).",
    )
    demo.add_argument("--atom-p", default="p", help="Atom p.")
    demo.add_argument("--atom-q", default="q", help="Atom q.")
    demo.add_argument(
        "--output-prefix",
        default="belief_revision_demo",
        help="Filename prefix inside the auto-created run folder.",
    )
    demo.add_argument(
        "--no-cut-closure",
        action="store_true",
        help="Disable cut-closure in demo examples.",
    )
    demo.set_defaults(func=run_revision_demo)

    gui = sub.add_parser(
        "gui",
        help="Open the visual interface for decomposition generation (Tkinter or Web fallback).",
    )
    gui.set_defaults(func=run_gui)

    web_gui = sub.add_parser(
        "web-gui",
        help="Open the web-based visual interface for decomposition generation.",
    )
    web_gui.set_defaults(func=run_web_gui)

    web_app = sub.add_parser(
        "web-app",
        help="Open the MathJax web app (no LaTeX/pdflatex installation required).",
    )
    web_app.set_defaults(func=run_web_app)

    certify = sub.add_parser(
        "afp-certify",
        parents=[common],
        help=(
            "Generate an Isabelle/AFP certificate (.thy + ROOT) for the current "
            "decomposition tree and optionally run Isabelle build."
        ),
    )
    certify.add_argument("--sequent", required=True, help="Sequent to certify.")
    certify.add_argument(
        "--theory-name",
        default="Generated_FS_Certificate",
        help="Isabelle theory/session name.",
    )
    certify.add_argument(
        "--output",
        default=None,
        help="Output .thy filename inside the auto-created run folder.",
    )
    certify.add_argument(
        "--verify",
        action="store_true",
        help="Run 'isabelle build' for the generated certificate.",
    )
    certify.add_argument(
        "--afp-thys",
        default=None,
        help=(
            "Path to AFP thys directory containing "
            "Propositional_Proof_Systems (or set AFP_THYS)."
        ),
    )
    certify.set_defaults(func=run_afp_certify)

    sc_ref = sub.add_parser(
        "sc-reference",
        help=(
            "Run a pure propositional sequent-calculus decomposition "
            "(without Fractional decorations)."
        ),
    )
    sc_ref.add_argument("--sequent", required=True, help="Sequent to decompose.")
    sc_ref.add_argument(
        "--show-tree",
        action="store_true",
        help="Print the reference decomposition tree in ASCII form.",
    )
    sc_ref.set_defaults(func=run_sc_reference)

    sc_compare = sub.add_parser(
        "sc-compare",
        parents=[common],
        help=(
            "Compare decomposition leaf clauses between the reference propositional "
            "sequent calculus and the Fractional engine."
        ),
    )
    sc_compare.add_argument("--sequent", required=True, help="Sequent to compare.")
    sc_compare.add_argument(
        "--show-tree",
        action="store_true",
        help="Print the reference decomposition tree in ASCII form.",
    )
    sc_compare.set_defaults(func=run_sc_compare)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
