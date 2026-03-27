from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from .calculus import EvaluationResult, ProofNode
from .logic import And, Atom, Formula, Not, Or


@dataclass(frozen=True)
class CertificateBuild:
    theory_name: str
    theory_text: str
    leaf_assumptions: tuple[str, ...]


def _sanitize_theory_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name.strip())
    if not cleaned:
        cleaned = "Generated_FS_Certificate"
    if cleaned[0].isdigit():
        cleaned = f"FS_{cleaned}"
    return cleaned


def _formula_to_isabelle(formula: Formula) -> str:
    if isinstance(formula, Atom):
        return f"Atom ''{formula.name}''"
    if isinstance(formula, Not):
        return f"Not ({_formula_to_isabelle(formula.arg)})"
    if isinstance(formula, And):
        return f"And ({_formula_to_isabelle(formula.left)}) ({_formula_to_isabelle(formula.right)})"
    if isinstance(formula, Or):
        return f"Or ({_formula_to_isabelle(formula.left)}) ({_formula_to_isabelle(formula.right)})"
    raise TypeError(f"Unsupported formula node: {formula!r}")


def _multiset_term(formulas: Iterable[Formula]) -> str:
    items = list(formulas)
    if not items:
        return "{#}"
    return "{# " + ", ".join(_formula_to_isabelle(item) for item in items) + " #}"


def _sc_term(formulas: tuple[Formula, ...]) -> str:
    return f"SCp E ({_multiset_term(formulas)})"


def _bag(formulas: tuple[Formula, ...]) -> Counter[Formula]:
    return Counter(formulas)


def _remove_one(formulas: tuple[Formula, ...], idx: int) -> tuple[Formula, ...]:
    return formulas[:idx] + formulas[idx + 1 :]


def _find_or_focus(parent: tuple[Formula, ...], child: tuple[Formula, ...]) -> tuple[Or, tuple[Formula, ...]]:
    child_bag = _bag(child)
    for idx, formula in enumerate(parent):
        if not isinstance(formula, Or):
            continue
        expected = _bag(_remove_one(parent, idx))
        expected[formula.left] += 1
        expected[formula.right] += 1
        if expected == child_bag:
            return formula, _remove_one(parent, idx)
    raise ValueError("Could not identify a valid (∨) focus for AFP certificate.")


def _find_and_focus(
    parent: tuple[Formula, ...],
    left_child: tuple[Formula, ...],
    right_child: tuple[Formula, ...],
) -> tuple[And, tuple[Formula, ...]]:
    left_bag = _bag(left_child)
    right_bag = _bag(right_child)
    for idx, formula in enumerate(parent):
        if not isinstance(formula, And):
            continue
        context = _remove_one(parent, idx)
        left_expected = _bag(context)
        left_expected[formula.left] += 1
        right_expected = _bag(context)
        right_expected[formula.right] += 1
        if left_expected == left_bag and right_expected == right_bag:
            return formula, context
    raise ValueError("Could not identify a valid (∧) focus for AFP certificate.")


def _collect_leaves(root: ProofNode) -> list[ProofNode]:
    leaves: list[ProofNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        if node.rule == "leaf":
            leaves.append(node)
            continue
        stack.extend(reversed(node.children))
    return leaves


def _render_show_block(
    node: ProofNode,
    assumption_by_leaf_id: dict[int, str],
    *,
    indent: str,
) -> list[str]:
    goal = _sc_term(node.sequent)
    if node.rule == "leaf":
        if node.leaf is None:
            raise ValueError("Leaf node without leaf metadata.")
        if node.leaf.kind != "ax":
            name = assumption_by_leaf_id[id(node)]
            return [f'{indent}show "{goal}" by (rule {name})']

        formulas = list(node.sequent)
        pair_atom: str | None = None
        atom_index = -1
        not_index = -1
        for idx, formula in enumerate(formulas):
            if not isinstance(formula, Atom):
                continue
            for jdx, other in enumerate(formulas):
                if idx == jdx:
                    continue
                if isinstance(other, Not) and isinstance(other.arg, Atom) and other.arg.name == formula.name:
                    pair_atom = formula.name
                    atom_index = idx
                    not_index = jdx
                    break
            if pair_atom is not None:
                break
        if pair_atom is None:
            raise ValueError("Tautological leaf does not contain an atom/complement pair.")

        rest = [item for idx, item in enumerate(formulas) if idx not in {atom_index, not_index}]
        atom_term = f"Atom ''{pair_atom}''"
        atom_arg = f"({atom_term})"
        rest_term = _multiset_term(rest)
        step_right = f"add_mset {atom_arg} ({rest_term})"
        lines = [f'{indent}show "{goal}"']
        lines.append(f"{indent}proof -")
        lines.append(
            f'{indent}  have base: "SCp (add_mset {atom_arg} E) ({step_right})"'
        )
        lines.append(f"{indent}    by (rule Ax_canonical)")
        lines.append(
            f'{indent}  from base have "SCp E (add_mset (Not ({atom_arg})) ({step_right}))"'
        )
        lines.append(f"{indent}    by (rule SCp.NotR)")
        lines.append(f"{indent}  then show ?thesis by simp")
        lines.append(f"{indent}qed")
        return lines

    if node.rule == "vee":
        if len(node.children) != 1:
            raise ValueError("Invalid vee node: expected exactly one child.")
        child = node.children[0]
        focus, context = _find_or_focus(node.sequent, child.sequent)
        focus_l = _formula_to_isabelle(focus.left)
        focus_r = _formula_to_isabelle(focus.right)
        context_term = _multiset_term(context)
        lines = [f'{indent}show "{goal}"']
        lines.append(
            f'{indent}proof (rule SCp.OrR[where Γ="E" and F="{focus_l}" and G="{focus_r}" and Δ="{context_term}"])'
        )
        lines.extend(
            _render_show_block(
                child,
                assumption_by_leaf_id,
                indent=indent + "  ",
            )
        )
        lines.append(f"{indent}qed")
        return lines

    if node.rule == "wedge":
        if len(node.children) != 2:
            raise ValueError("Invalid wedge node: expected exactly two children.")
        left = node.children[0]
        right = node.children[1]
        focus, context = _find_and_focus(node.sequent, left.sequent, right.sequent)
        focus_l = _formula_to_isabelle(focus.left)
        focus_r = _formula_to_isabelle(focus.right)
        context_term = _multiset_term(context)
        lines = [f'{indent}show "{goal}"']
        lines.append(
            f'{indent}proof (rule SCp.AndR[where Γ="E" and F="{focus_l}" and G="{focus_r}" and Δ="{context_term}"])'
        )
        lines.extend(
            _render_show_block(
                left,
                assumption_by_leaf_id,
                indent=indent + "  ",
            )
        )
        lines.extend(
            _render_show_block(
                right,
                assumption_by_leaf_id,
                indent=indent + "  ",
            )
        )
        lines.append(f"{indent}qed")
        return lines

    raise ValueError(f"Unsupported proof node rule: {node.rule!r}")


def build_afp_certificate(
    result: EvaluationResult,
    *,
    source_sequent_label: str,
    theory_name: str = "Generated_FS_Certificate",
) -> CertificateBuild:
    clean_name = _sanitize_theory_name(theory_name)
    leaves = _collect_leaves(result.root)
    assumptions: list[tuple[str, str]] = []
    assumption_by_leaf_id: dict[int, str] = {}
    labels: list[str] = []
    for idx, leaf_node in enumerate(leaves, start=1):
        kind = leaf_node.leaf.kind if leaf_node.leaf is not None else "leaf"
        if kind == "ax":
            continue
        base = {"ax": "leaf_ax", "belief": "leaf_belief", "comp": "leaf_comp"}.get(
            kind,
            "leaf",
        )
        name = f"{base}_{idx}"
        assumption = _sc_term(leaf_node.sequent)
        assumptions.append((name, assumption))
        assumption_by_leaf_id[id(leaf_node)] = name
        labels.append(name)

    assumption_block = ""
    if assumptions:
        lines = []
        first_name, first_term = assumptions[0]
        lines.append(f'  assumes {first_name}: "{first_term}"')
        for name, term in assumptions[1:]:
            lines.append(f'    and {name}: "{term}"')
        assumption_block = "\n" + "\n".join(lines)

    root_term = _sc_term(result.root.sequent)
    body = _render_show_block(result.root, assumption_by_leaf_id, indent="  ")
    theory = (
        f"theory {clean_name}\n"
        f'imports "Propositional_Proof_Systems.SC"\n'
        f"begin\n\n"
        f'text ‹Automatically generated certificate for: "{source_sequent_label}".›\n'
        f"text ‹The theorem verifies that the decomposition tree is a valid SCp-rule composition\n"
        f"from its leaves. Fractional/Belief decorations are managed outside Isabelle.›\n\n"
        f'abbreviation E :: "string formula multiset" where "E ≡ {{#}}"\n\n'
        f"theorem generated_tree_certificate{assumption_block}\n"
        f'  shows "{root_term}"\n'
        f"proof -\n"
        f"{chr(10).join(body)}\n"
        f"qed\n\n"
        f"end\n"
    )
    return CertificateBuild(
        theory_name=clean_name,
        theory_text=theory,
        leaf_assumptions=tuple(labels),
    )
