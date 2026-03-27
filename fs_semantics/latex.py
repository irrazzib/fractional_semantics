from __future__ import annotations

from decimal import Decimal
from fractions import Fraction
from typing import Iterable

from .beliefs import BeliefBase
from .calculus import Clause, EvaluationResult, ProofNode, clause_to_latex
from .logic import (
    Formula,
    formula_to_latex,
    sequent_to_latex_grouped,
)


def _fraction_to_latex(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    return f"\\frac{{{value.numerator}}}{{{value.denominator}}}"


def _fraction_to_text(value: Fraction) -> str:
    if value.denominator == 1:
        return str(value.numerator)
    denominator = value.denominator
    while denominator % 2 == 0:
        denominator //= 2
    while denominator % 5 == 0:
        denominator //= 5
    if denominator == 1:
        decimal = Decimal(value.numerator) / Decimal(value.denominator)
        rendered = format(decimal.normalize(), "f")
        if "." in rendered:
            rendered = rendered.rstrip("0").rstrip(".")
        return rendered or "0"
    return _fraction_to_latex(value)


def _decimal(value: Fraction) -> str:
    return f"{float(value):.6g}"


def _escape_text(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("_", "\\_")
        .replace("%", "\\%")
        .replace("&", "\\&")
        .replace("#", "\\#")
    )


def _leaf_kind_label(kind: str, belief_index: int | None = None) -> str:
    if kind == "ax":
        return "ax."
    if kind == "belief":
        if belief_index is not None:
            return f"b_{{{belief_index}}}"
        return "b"
    return "\\overline{ax.}"


def _delta_symbol(node: ProofNode) -> str:
    if node.leaf is None:
        raise ValueError("Leaf node without leaf metadata")
    label = clause_to_latex(node.leaf.clause)
    return f"\\delta_{{\\{{{label}\\}}}}"


def _node_weight(
    node: ProofNode,
    belief_mode: str,
    belief_strengths: dict[Clause, Fraction],
    explicit_gradient_clauses: set[Clause],
) -> tuple[int, Fraction, list[str]]:
    if node.rule == "leaf":
        if node.leaf is None:
            raise ValueError("Leaf node without leaf metadata")
        if node.leaf.kind == "ax":
            return (1, Fraction(1, 1), [])
        if node.leaf.kind == "belief":
            if belief_mode == "gradient":
                return (1, belief_strengths.get(node.leaf.clause, Fraction(1, 1)), [])
            if belief_mode == "full":
                if node.leaf.clause in explicit_gradient_clauses:
                    return (
                        1,
                        belief_strengths.get(node.leaf.clause, Fraction(1, 1)),
                        [_delta_symbol(node)],
                    )
                return (1, Fraction(1, 1), [_delta_symbol(node)])
            return (1, Fraction(1, 1), [])
        return (1, Fraction(0, 1), [])
    if node.rule == "vee":
        if len(node.children) != 1:
            raise ValueError("Vee node must have exactly one child")
        return _node_weight(
            node.children[0],
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
        )
    if node.rule == "wedge":
        if len(node.children) != 2:
            raise ValueError("Wedge node must have exactly two children")
        left_n, left_const, left_deltas = _node_weight(
            node.children[0],
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
        )
        right_n, right_const, right_deltas = _node_weight(
            node.children[1],
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
        )
        return (
            left_n + right_n,
            left_const + right_const,
            left_deltas + right_deltas,
        )
    raise ValueError(f"Unsupported proof node rule: {node.rule}")


def _m_expr(constant_part: Fraction, deltas: list[str]) -> str:
    if not deltas:
        return _fraction_to_text(constant_part)
    if len(deltas) == 1:
        delta_term = deltas[0]
    else:
        delta_term = f"({'+'.join(deltas)})"
    if constant_part == 0:
        return f"-{delta_term}"
    return f"{_fraction_to_text(constant_part)}-{delta_term}"


def _proof_tree_sequent(
    node: ProofNode,
    with_belief_base: bool,
    belief_mode: str,
    belief_strengths: dict[Clause, Fraction],
    explicit_gradient_clauses: set[Clause],
    sequent_override: str | None = None,
) -> str:
    sequent = (
        sequent_override
        if sequent_override is not None
        else sequent_to_latex_grouped(node.sequent)
    )
    n, const, deltas = _node_weight(
        node,
        belief_mode,
        belief_strengths,
        explicit_gradient_clauses,
    )
    m = _m_expr(const, deltas)
    suffix = "_{\\mathbb{B}}" if with_belief_base else ""
    return (
        f"$\\sststile{{\\mathrm{{{n}}}}}{{{m}}}{suffix}"
        f"\\; {sequent}$"
    )


def _render_prooftree_commands(
    node: ProofNode,
    with_belief_base: bool,
    belief_mode: str,
    belief_strengths: dict[Clause, Fraction],
    explicit_gradient_clauses: set[Clause],
    belief_counter: list[int],
    root_node: ProofNode,
    root_sequent_override: str | None,
) -> str:
    current_override = root_sequent_override if node is root_node else None
    if node.rule == "leaf":
        if node.leaf is None:
            raise ValueError("Leaf node without leaf metadata")
        belief_index: int | None = None
        if node.leaf.kind == "belief":
            belief_counter[0] += 1
            belief_index = belief_counter[0]
        kind = _leaf_kind_label(node.leaf.kind, belief_index=belief_index)
        return "\n".join(
            [
                "\\AxiomC{}",
                f"\\RightLabel{{\\scriptsize$({kind})$}}",
                (
                    "\\UnaryInfC{"
                    f"{_proof_tree_sequent(node, with_belief_base, belief_mode, belief_strengths, explicit_gradient_clauses, current_override)}"
                    "}"
                ),
            ]
        )

    if node.rule == "vee":
        if len(node.children) != 1:
            raise ValueError("Vee node must have exactly one child")
        child_block = _render_prooftree_commands(
            node.children[0],
            with_belief_base,
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
            belief_counter,
            root_node,
            root_sequent_override,
        )
        return "\n".join(
            [
                child_block,
                "\\RightLabel{\\scriptsize$(\\vee)$}",
                (
                    "\\UnaryInfC{"
                    f"{_proof_tree_sequent(node, with_belief_base, belief_mode, belief_strengths, explicit_gradient_clauses, current_override)}"
                    "}"
                ),
            ]
        )

    if node.rule == "wedge":
        if len(node.children) != 2:
            raise ValueError("Wedge node must have exactly two children")
        left_block = _render_prooftree_commands(
            node.children[0],
            with_belief_base,
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
            belief_counter,
            root_node,
            root_sequent_override,
        )
        right_block = _render_prooftree_commands(
            node.children[1],
            with_belief_base,
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
            belief_counter,
            root_node,
            root_sequent_override,
        )
        return "\n".join(
            [
                left_block,
                right_block,
                "\\RightLabel{\\scriptsize$(\\wedge)$}",
                (
                    "\\BinaryInfC{"
                    f"{_proof_tree_sequent(node, with_belief_base, belief_mode, belief_strengths, explicit_gradient_clauses, current_override)}"
                    "}"
                ),
            ]
        )

    raise ValueError(f"Unsupported proof node rule: {node.rule}")


def _render_leaves_table(result: EvaluationResult) -> str:
    rows: list[str] = []
    for idx, leaf in enumerate(result.leaves, start=1):
        clause = clause_to_latex(leaf.clause)
        kind = _leaf_kind_label(leaf.kind)
        rows.append(f"{idx} & $\\vdash {clause}$ & ${kind}$ \\\\")
    return "\n".join(rows)


def _render_belief_list_inline(formulas: Iterable[Formula]) -> str:
    parts = [f"${formula_to_latex(formula)}$" for formula in formulas]
    return ", ".join(parts) if parts else "$\\varnothing$"


def _render_belief_set_math(formulas: Iterable[Formula]) -> str:
    parts = [formula_to_latex(formula) for formula in formulas]
    if not parts:
        return "\\varnothing"
    return "\\{" + ", ".join(parts) + "\\}"


def render_decomposition_prooftree(
    result: EvaluationResult,
    with_belief_base: bool,
    belief_mode: str = "standard",
    belief_strengths: dict[Clause, Fraction] | None = None,
    explicit_gradient_clauses: set[Clause] | None = None,
    root_sequent_latex: str | None = None,
) -> str:
    if belief_mode not in {"standard", "full", "gradient"}:
        raise ValueError(f"Unsupported belief mode: {belief_mode}")
    strengths = belief_strengths or {}
    explicit_gradients = explicit_gradient_clauses or set()
    root_override = (
        root_sequent_latex
        if root_sequent_latex is not None
        else sequent_to_latex_grouped(result.root.sequent)
    )
    return (
        "\\begin{prooftree}\n"
        f"{_render_prooftree_commands(result.root, with_belief_base, belief_mode, strengths, explicit_gradients, [0], result.root, root_override)}\n"
        "\\end{prooftree}"
    )


def render_analysis_document(
    sequent_label: str,
    result: EvaluationResult,
    belief_base: BeliefBase | None,
) -> str:
    with_belief_base = belief_base is not None
    belief_section = ""
    if belief_base is not None:
        belief_section = f"""
\\subsection*{{Belief Base $\\mathbb{{B}}$}}
Formulas in $\\mathbb{{B}}$: {_render_belief_list_inline(belief_base.formulas)}.

Derived clauses (closed under cut): {"$\\varnothing$" if not belief_base.clauses else ", ".join(f"$\\vdash {clause_to_latex(clause)}$" for clause in sorted(belief_base.clauses, key=lambda c: (len(c), sorted((l.atom, l.negated) for l in c))))}.
"""

    value_b_block = (
        f"""
\\[
\\llbracket \\Gamma \\rrbracket_\\mathbb{{B}}
= \\frac{{\\#top^1 + \\#top^b}}{{\\#top^1 + \\#top^b + \\#top^0}}
= \\frac{{{result.top1} + {result.topb}}}{{{result.total}}}
= {_fraction_to_latex(result.value_b)}
\\approx {_decimal(result.value_b)}.
\\]
"""
        if belief_base is not None
        else ""
    )

    return f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[margin=2.5cm]{{geometry}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{stmaryrd}}
\\usepackage{{turnstile}}
\\usepackage{{booktabs}}
\\usepackage{{bussproofs}}
\\usepackage{{longtable}}
\\title{{Automatic Fractional Semantics Calculation}}
\\date{{}}
\\begin{{document}}
\\maketitle

\\section*{{Input}}
Input sequent: \\texttt{{{_escape_text(sequent_label)}}}

\\section*{{Fractional Semantics Decomposition (Classic Prooftree)}}
\\begin{{center}}
{render_decomposition_prooftree(result, with_belief_base)}
\\end{{center}}

\\section*{{Leaves (Top-Sequents)}}
\\begin{{longtable}}{{rll}}
\\toprule
\\# & Leaf & Type \\\\
\\midrule
{_render_leaves_table(result)}
\\bottomrule
\\end{{longtable}}

{belief_section}

\\section*{{Fractional Semantics Values}}
\\[
\\llbracket \\Gamma \\rrbracket
= \\frac{{\\#top^1}}{{\\#top^1 + \\#top^0}}
= \\frac{{{result.top1}}}{{{result.total}}}
= {_fraction_to_latex(result.value)}
\\approx {_decimal(result.value)}.
\\]
{value_b_block}

\\end{{document}}
"""


def render_revision_document(
    operation: str,
    target_label: str,
    sequent_label: str,
    before_base: BeliefBase,
    after_base: BeliefBase,
    before_result: EvaluationResult,
    after_result: EvaluationResult,
) -> str:
    op_label = {
        "contract": "Contraction",
        "expand": "Expansion",
        "revise": "Revision (Levi)",
    }.get(operation, operation)
    return f"""\\documentclass[12pt,a4paper]{{article}}
\\usepackage[utf8]{{inputenc}}
\\usepackage[T1]{{fontenc}}
\\usepackage[margin=2.5cm]{{geometry}}
\\usepackage{{amsmath,amssymb}}
\\usepackage{{stmaryrd}}
\\usepackage{{turnstile}}
\\usepackage{{booktabs}}
\\title{{Belief Revision in Fractional Semantics}}
\\date{{}}
\\begin{{document}}
\\maketitle

\\section*{{Operation}}
{op_label} with target \\texttt{{{_escape_text(target_label)}}} on sequent \\texttt{{{_escape_text(sequent_label)}}}.

\\section*{{Before}}
$\\mathbb{{B}}_0 = {_render_belief_set_math(before_base.formulas)}$\\\\
$\\llbracket \\Gamma \\rrbracket_{{\\mathbb{{B}}_0}} = {_fraction_to_latex(before_result.value_b)} \\approx {_decimal(before_result.value_b)}$

\\section*{{After}}
$\\mathbb{{B}}_1 = {_render_belief_set_math(after_base.formulas)}$\\\\
$\\llbracket \\Gamma \\rrbracket_{{\\mathbb{{B}}_1}} = {_fraction_to_latex(after_result.value_b)} \\approx {_decimal(after_result.value_b)}$

\\section*{{Difference}}
\\[
\\Delta =
\\llbracket \\Gamma \\rrbracket_{{\\mathbb{{B}}_1}} -
\\llbracket \\Gamma \\rrbracket_{{\\mathbb{{B}}_0}}
= {_fraction_to_latex(after_result.value_b - before_result.value_b)}.
\\]

\\end{{document}}
"""
