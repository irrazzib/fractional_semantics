from __future__ import annotations

from decimal import Decimal
from fractions import Fraction

from .calculus import Clause, EvaluationResult, ProofNode, clause_to_latex
from .logic import sequent_to_ascii_grouped


def _delta_symbol(node: ProofNode) -> str:
    if node.leaf is None:
        raise ValueError("Leaf node without leaf metadata")
    label = clause_to_latex(node.leaf.clause)
    return f"δ_{{{{{label}}}}}"


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
    return f"{value.numerator}/{value.denominator}"


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
        n1, c1, d1 = _node_weight(
            node.children[0],
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
        )
        n2, c2, d2 = _node_weight(
            node.children[1],
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
        )
        return (n1 + n2, c1 + c2, d1 + d2)
    raise ValueError(f"Unsupported proof node rule: {node.rule}")


def _node_rule_label(node: ProofNode) -> str:
    if node.rule == "vee":
        return "(∨)"
    if node.rule == "wedge":
        return "(∧)"
    if node.rule != "leaf" or node.leaf is None:
        return "(?)"
    if node.leaf.kind == "ax":
        return "(ax.)"
    if node.leaf.kind == "belief":
        return "(b)"
    return "(overline{ax.})"


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


def _node_payload(
    node: ProofNode,
    with_belief_base: bool,
    belief_mode: str,
    belief_strengths: dict[Clause, Fraction],
    explicit_gradient_clauses: set[Clause],
    root_node: ProofNode,
    root_sequent_override: str | None,
) -> dict[str, object]:
    n, constant, deltas = _node_weight(
        node,
        belief_mode,
        belief_strengths,
        explicit_gradient_clauses,
    )
    m = _m_expr(constant, deltas)
    suffix = "_B" if with_belief_base else ""
    sequent = (
        root_sequent_override
        if node is root_node and root_sequent_override is not None
        else sequent_to_ascii_grouped(node.sequent)
    )
    display = f"sststile{{{n}}}{{{m}}}{suffix} |- {sequent}"
    return {
        "rule": _node_rule_label(node),
        "n": n,
        "m": m,
        "belief_mode": belief_mode,
        "with_belief_base": with_belief_base,
        "sequent": sequent,
        "display": display,
        "children": [
            _node_payload(
                child,
                with_belief_base,
                belief_mode,
                belief_strengths,
                explicit_gradient_clauses,
                root_node,
                root_sequent_override,
            )
            for child in node.children
        ],
    }


def build_visual_tree(
    result: EvaluationResult,
    with_belief_base: bool,
    belief_mode: str = "standard",
    belief_strengths: dict[Clause, Fraction] | None = None,
    explicit_gradient_clauses: set[Clause] | None = None,
    root_sequent_ascii: str | None = None,
) -> dict[str, object]:
    if belief_mode not in {"standard", "full", "gradient"}:
        raise ValueError(f"Unsupported belief mode: {belief_mode}")
    strengths = belief_strengths or {}
    explicit_gradients = explicit_gradient_clauses or set()
    return _node_payload(
        result.root,
        with_belief_base,
        belief_mode,
        strengths,
        explicit_gradients,
        result.root,
        root_sequent_ascii,
    )
