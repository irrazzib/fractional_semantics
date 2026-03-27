from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from .calculus import ProofNode
from .logic import And, Atom, Formula, Not, Or, sequent_to_ascii_grouped


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    message: str


def _is_literal(formula: Formula) -> bool:
    return isinstance(formula, Atom) or (
        isinstance(formula, Not) and isinstance(formula.arg, Atom)
    )


def _bag(formulas: tuple[Formula, ...]) -> Counter[Formula]:
    return Counter(formulas)


def _matches_or_step(parent: tuple[Formula, ...], child: tuple[Formula, ...]) -> bool:
    child_bag = _bag(child)
    for idx, formula in enumerate(parent):
        if not isinstance(formula, Or):
            continue
        expected = _bag(parent)
        expected[formula] -= 1
        if expected[formula] <= 0:
            expected.pop(formula, None)
        expected[formula.left] += 1
        expected[formula.right] += 1
        if expected == child_bag:
            return True
    return False


def _matches_and_step(
    parent: tuple[Formula, ...],
    left_child: tuple[Formula, ...],
    right_child: tuple[Formula, ...],
) -> bool:
    left_bag = _bag(left_child)
    right_bag = _bag(right_child)
    for idx, formula in enumerate(parent):
        if not isinstance(formula, And):
            continue
        left_expected = _bag(parent)
        left_expected[formula] -= 1
        if left_expected[formula] <= 0:
            left_expected.pop(formula, None)
        left_expected[formula.left] += 1

        right_expected = _bag(parent)
        right_expected[formula] -= 1
        if right_expected[formula] <= 0:
            right_expected.pop(formula, None)
        right_expected[formula.right] += 1

        if left_expected == left_bag and right_expected == right_bag:
            return True
    return False


def _validate_node(node: ProofNode, errors: list[str]) -> None:
    if node.rule == "leaf":
        if node.children:
            errors.append("Leaf node has children.")
        if not all(_is_literal(formula) for formula in node.sequent):
            errors.append(
                "Leaf node contains non-literal formulas: "
                f"{sequent_to_ascii_grouped(node.sequent)}"
            )
        return

    if node.rule == "vee":
        if len(node.children) != 1:
            errors.append("Vee node does not have exactly one child.")
            return
        child = node.children[0]
        if not _matches_or_step(node.sequent, child.sequent):
            errors.append(
                "Invalid (∨) decomposition step from "
                f"{sequent_to_ascii_grouped(node.sequent)} to "
                f"{sequent_to_ascii_grouped(child.sequent)}."
            )
        _validate_node(child, errors)
        return

    if node.rule == "wedge":
        if len(node.children) != 2:
            errors.append("Wedge node does not have exactly two children.")
            return
        left = node.children[0]
        right = node.children[1]
        if not _matches_and_step(node.sequent, left.sequent, right.sequent):
            errors.append(
                "Invalid (∧) decomposition step from "
                f"{sequent_to_ascii_grouped(node.sequent)} to "
                f"{sequent_to_ascii_grouped(left.sequent)} / "
                f"{sequent_to_ascii_grouped(right.sequent)}."
            )
        _validate_node(left, errors)
        _validate_node(right, errors)
        return

    errors.append(f"Unknown rule label: {node.rule!r}.")


def validate_proof_tree(
    root: ProofNode,
    expected_root: tuple[Formula, ...],
    source_sequent: tuple[Formula, ...],
) -> ValidationResult:
    errors: list[str] = []
    if root.sequent != expected_root:
        errors.append(
            "Root sequent mismatch against parsed+normalized input. "
            f"Expected: {sequent_to_ascii_grouped(expected_root)} | "
            f"Found: {sequent_to_ascii_grouped(root.sequent)}"
        )

    _validate_node(root, errors)
    if errors:
        return ValidationResult(
            ok=False,
            message=" | ".join(errors),
        )

    return ValidationResult(
        ok=True,
        message=(
            "Parentheses-aware decomposition check passed. "
            f"Parsed input: {sequent_to_ascii_grouped(source_sequent)}"
        ),
    )
