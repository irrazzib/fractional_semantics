from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction

from .calculus import Clause, clause_to_latex, is_tautological_clause
from .logic import (
    And,
    Atom,
    Formula,
    Literal,
    Not,
    Or,
    formula_to_ascii,
    formula_to_latex,
    to_nnf,
)


@dataclass
class CutFreeNode:
    rule: str
    formula: Formula | None
    children: list["CutFreeNode"]
    clause_literals: tuple[Formula, ...] = field(default_factory=tuple)
    or_merged_count: int = 0
    clause: Clause | None = None
    kind: str | None = None


def _literal_clause(formula: Formula) -> Clause:
    if isinstance(formula, Atom):
        return frozenset({Literal(formula.name, False)})
    if isinstance(formula, Not) and isinstance(formula.arg, Atom):
        return frozenset({Literal(formula.arg.name, True)})
    raise ValueError("Expected literal formula.")


def _is_literal(formula: Formula) -> bool:
    return isinstance(formula, Atom) or (
        isinstance(formula, Not) and isinstance(formula.arg, Atom)
    )


def _mk_or(left: Formula, right: Formula) -> Formula:
    if left == right:
        return left
    return Or(left, right)


def _mk_and(left: Formula, right: Formula) -> Formula:
    if left == right:
        return left
    return And(left, right)


def _distribute_or(left: Formula, right: Formula) -> Formula:
    if left == right:
        return left
    if isinstance(left, And):
        return _mk_and(
            _distribute_or(left.left, right),
            _distribute_or(left.right, right),
        )
    if isinstance(right, And):
        return _mk_and(
            _distribute_or(left, right.left),
            _distribute_or(left, right.right),
        )
    return _mk_or(left, right)


def _to_cnf(formula: Formula) -> Formula:
    f = to_nnf(formula)
    if isinstance(f, (Atom, Not)):
        return f
    if isinstance(f, And):
        return _mk_and(_to_cnf(f.left), _to_cnf(f.right))
    if isinstance(f, Or):
        return _distribute_or(_to_cnf(f.left), _to_cnf(f.right))
    raise TypeError(f"Unsupported formula node: {f!r}")


def _flatten_or_literals(formula: Formula) -> list[Formula]:
    if _is_literal(formula):
        return [formula]
    if isinstance(formula, Or):
        return _flatten_or_literals(formula.left) + _flatten_or_literals(formula.right)
    raise ValueError("Expected a disjunction of literals.")


def _clause_from_literals(literals: list[Formula]) -> Clause:
    return frozenset(
        lit
        for formula in literals
        for lit in _literal_clause(formula)
    )


def _classify_leaf_clause(clause: Clause, belief_clauses: set[Clause]) -> str:
    if is_tautological_clause(clause):
        return "ax"
    return "belief" if clause in belief_clauses else "comp"


def _build_clause_subtree(
    formula: Formula,
    *,
    belief_clauses: set[Clause],
) -> CutFreeNode:
    literals = _flatten_or_literals(formula)
    clause = _clause_from_literals(literals)
    kind = _classify_leaf_clause(clause, belief_clauses)
    leaf = CutFreeNode(
        rule="leaf",
        formula=None,
        children=[],
        clause_literals=tuple(literals),
        or_merged_count=0,
        clause=clause,
        kind=kind,
    )
    if len(literals) <= 1:
        return leaf

    current = leaf
    combined = literals[0]
    for idx, lit in enumerate(literals[1:], start=2):
        combined = _mk_or(combined, lit)
        current = CutFreeNode(
            rule="or",
            formula=combined,
            children=[current],
            clause_literals=tuple(literals),
            or_merged_count=idx,
            clause=clause,
            kind=kind,
        )
    return current


def _build_cnf_tree(
    formula: Formula,
    *,
    belief_clauses: set[Clause],
) -> CutFreeNode:
    if isinstance(formula, And):
        return CutFreeNode(
            rule="and",
            formula=formula,
            children=[
                _build_cnf_tree(formula.left, belief_clauses=belief_clauses),
                _build_cnf_tree(formula.right, belief_clauses=belief_clauses),
            ],
        )
    return _build_clause_subtree(formula, belief_clauses=belief_clauses)


def build_cut_free_tree(
    formula: Formula,
    *,
    belief_clauses: set[Clause],
) -> CutFreeNode:
    return _build_cnf_tree(_to_cnf(formula), belief_clauses=belief_clauses)


def cut_free_root_formula(root: CutFreeNode) -> Formula:
    if root.formula is not None:
        return root.formula
    if root.rule != "leaf":
        raise ValueError("Non-leaf cut-free root without formula.")
    if not root.clause_literals:
        raise ValueError("Leaf root without clause literals.")
    combined = root.clause_literals[0]
    for formula in root.clause_literals[1:]:
        combined = _mk_or(combined, formula)
    return combined


def collect_cut_free_leaf_clauses(root: CutFreeNode) -> tuple[Clause, ...]:
    out: list[Clause] = []

    def walk(node: CutFreeNode) -> None:
        if node.rule == "leaf":
            if node.clause is None:
                raise ValueError("Leaf without clause.")
            out.append(node.clause)
            return
        for child in node.children:
            walk(child)

    walk(root)
    return tuple(out)


def _delta_symbol(node: CutFreeNode) -> str:
    if node.clause is None:
        raise ValueError("Leaf without clause.")
    label = clause_to_latex(node.clause)
    return f"\\delta_{{\\{{{label}\\}}}}"


def _weight(
    node: CutFreeNode,
    belief_mode: str,
    belief_strengths: dict[Clause, Fraction],
    explicit_gradient_clauses: set[Clause],
) -> tuple[int, Fraction, list[str]]:
    if node.rule == "leaf":
        if node.kind == "ax":
            return (1, Fraction(1, 1), [])
        if node.kind == "belief":
            if belief_mode == "gradient":
                if node.clause is None:
                    raise ValueError("Leaf belief without clause.")
                return (1, belief_strengths.get(node.clause, Fraction(1, 1)), [])
            if belief_mode == "full":
                if node.clause is not None and node.clause in explicit_gradient_clauses:
                    return (
                        1,
                        belief_strengths.get(node.clause, Fraction(1, 1)),
                        [_delta_symbol(node)],
                    )
                return (1, Fraction(1, 1), [_delta_symbol(node)])
            return (1, Fraction(1, 1), [])
        return (1, Fraction(0, 1), [])
    if node.rule == "or":
        if len(node.children) != 1:
            raise ValueError("OR node must have exactly one child.")
        return _weight(
            node.children[0],
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
        )
    if node.rule == "and":
        if len(node.children) != 2:
            raise ValueError("AND node must have exactly two children.")
        left_n, left_c, left_d = _weight(
            node.children[0],
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
        )
        right_n, right_c, right_d = _weight(
            node.children[1],
            belief_mode,
            belief_strengths,
            explicit_gradient_clauses,
        )
        return (left_n + right_n, left_c + right_c, left_d + right_d)
    raise ValueError(f"Unknown cut-free rule: {node.rule}")


def _fraction_to_text(value: Fraction, *, latex: bool) -> str:
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
    if latex:
        return f"\\frac{{{value.numerator}}}{{{value.denominator}}}"
    return f"{value.numerator}/{value.denominator}"


def _m_expr(constant_part: Fraction, deltas: list[str], *, latex: bool) -> str:
    if not deltas:
        return _fraction_to_text(constant_part, latex=latex)
    if len(deltas) == 1:
        delta_term = deltas[0]
    else:
        delta_term = f"({'+'.join(deltas)})"
    if constant_part == 0:
        return f"-{delta_term}"
    return f"{_fraction_to_text(constant_part, latex=latex)}-{delta_term}"


def _rule_label(rule: str, kind: str | None) -> str:
    if rule == "leaf":
        if kind == "ax":
            return "(ax.)"
        if kind == "belief":
            return "(b)"
        return "(overline{ax.})"
    if rule == "or":
        return "(∨)"
    if rule == "and":
        return "(∧)"
    return "(?)"


def _rule_label_latex(
    rule: str,
    kind: str | None,
    belief_index: int | None = None,
) -> str:
    if rule == "leaf":
        if kind == "ax":
            return "(ax.)"
        if kind == "belief":
            if belief_index is not None:
                return f"(b_{{{belief_index}}})"
            return "(b)"
        return "(\\overline{ax.})"
    if rule == "or":
        return "(\\vee)"
    if rule == "and":
        return "(\\wedge)"
    return "(?)"


def _node_formula_latex(node: CutFreeNode) -> str:
    if node.rule == "leaf":
        return ", ".join(formula_to_latex(item) for item in node.clause_literals)
    if node.rule == "or":
        if node.formula is None:
            raise ValueError("OR node requires formula.")
        rendered = formula_to_latex(node.formula)
        remainder = node.clause_literals[node.or_merged_count :]
        if not remainder:
            return rendered
        rem = ", ".join(formula_to_latex(item) for item in remainder)
        return f"{rendered}, {rem}"
    if node.formula is None:
        raise ValueError("Non-leaf node requires formula.")
    return formula_to_latex(node.formula)


def _node_formula_ascii(node: CutFreeNode) -> str:
    if node.rule == "leaf":
        return ", ".join(formula_to_ascii(item) for item in node.clause_literals)
    if node.rule == "or":
        if node.formula is None:
            raise ValueError("OR node requires formula.")
        rendered = formula_to_ascii(node.formula)
        remainder = node.clause_literals[node.or_merged_count :]
        if not remainder:
            return rendered
        rem = ", ".join(formula_to_ascii(item) for item in remainder)
        return f"{rendered}, {rem}"
    if node.formula is None:
        raise ValueError("Non-leaf node requires formula.")
    return formula_to_ascii(node.formula)


def render_cut_free_prooftree(
    root: CutFreeNode,
    *,
    with_belief_base: bool,
    belief_mode: str,
    belief_strengths: dict[Clause, Fraction] | None = None,
    explicit_gradient_clauses: set[Clause] | None = None,
    root_sequent_latex: str | None = None,
) -> str:
    if belief_mode not in {"standard", "full", "gradient"}:
        raise ValueError(f"Unsupported belief mode: {belief_mode}")
    suffix = "_{\\mathbb{B}}" if with_belief_base else ""
    strengths = belief_strengths or {}
    explicit_gradients = explicit_gradient_clauses or set()
    belief_counter = [0]

    def walk(node: CutFreeNode, is_root: bool = False) -> str:
        n, const, deltas = _weight(node, belief_mode, strengths, explicit_gradients)
        m = _m_expr(const, deltas, latex=True)
        sequent = (
            root_sequent_latex
            if is_root and root_sequent_latex is not None
            else _node_formula_latex(node)
        )
        payload = f"$\\sststile{{\\mathrm{{{n}}}}}{{{m}}}{suffix}\\; {sequent}$"

        if node.rule == "leaf":
            belief_index: int | None = None
            if node.kind == "belief":
                belief_counter[0] += 1
                belief_index = belief_counter[0]
            return "\n".join(
                [
                    "\\AxiomC{}",
                    (
                        "\\RightLabel{\\scriptsize$"
                        f"{_rule_label_latex(node.rule, node.kind, belief_index=belief_index)}"
                        "$}"
                    ),
                    f"\\UnaryInfC{{{payload}}}",
                ]
            )
        if node.rule == "or":
            child = walk(node.children[0])
            return "\n".join(
                [
                    child,
                    f"\\RightLabel{{\\scriptsize${_rule_label_latex(node.rule, None)}$}}",
                    f"\\UnaryInfC{{{payload}}}",
                ]
            )

        left = walk(node.children[0])
        right = walk(node.children[1])
        return "\n".join(
            [
                left,
                right,
                f"\\RightLabel{{\\scriptsize${_rule_label_latex(node.rule, None)}$}}",
                f"\\BinaryInfC{{{payload}}}",
            ]
        )

    return "\\begin{prooftree}\n" + walk(root, is_root=True) + "\n\\end{prooftree}"


def build_cut_free_visual_tree(
    root: CutFreeNode,
    *,
    with_belief_base: bool,
    belief_mode: str,
    belief_strengths: dict[Clause, Fraction] | None = None,
    explicit_gradient_clauses: set[Clause] | None = None,
    root_sequent_ascii: str | None = None,
) -> dict[str, object]:
    if belief_mode not in {"standard", "full", "gradient"}:
        raise ValueError(f"Unsupported belief mode: {belief_mode}")
    suffix = "_B" if with_belief_base else ""
    strengths = belief_strengths or {}
    explicit_gradients = explicit_gradient_clauses or set()

    def walk(node: CutFreeNode, is_root: bool = False) -> dict[str, object]:
        n, const, deltas = _weight(node, belief_mode, strengths, explicit_gradients)
        m = _m_expr(const, deltas, latex=False)
        sequent = (
            root_sequent_ascii
            if is_root and root_sequent_ascii is not None
            else _node_formula_ascii(node)
        )
        display = f"sststile{{{n}}}{{{m}}}{suffix} |- {sequent}"
        return {
            "rule": _rule_label(node.rule, node.kind),
            "n": n,
            "m": m,
            "belief_mode": belief_mode,
            "with_belief_base": with_belief_base,
            "sequent": sequent,
            "display": display,
            "children": [walk(child) for child in node.children],
        }

    return walk(root, is_root=True)
