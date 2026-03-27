from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

from .beliefs import BeliefBase
from .calculus import Clause, EvaluationResult, ProofNode, clauses_from_formula, evaluate_sequent
from .cut_free import (
    build_cut_free_tree,
    build_cut_free_visual_tree,
    cut_free_root_formula,
    render_cut_free_prooftree,
)
from .input_parsing import parse_belief_set_text, parse_gradient_values_text
from .latex import render_decomposition_prooftree
from .logic import (
    Formula,
    Or,
    formula_to_ascii,
    formula_to_latex,
    parse_sequent_structured,
    sequent_to_ascii_grouped,
    sequent_to_latex_grouped,
    to_nnf,
)
from .proof_validation import validate_proof_tree
from .visualization import build_visual_tree


def _collapse_disjunction_sequent(sequent: tuple[Formula, ...]) -> tuple[Formula, ...]:
    if len(sequent) <= 1:
        return sequent
    combined = sequent[0]
    for formula in sequent[1:]:
        combined = Or(combined, formula)
    return (combined,)


def _append_final_vee_chain(root: ProofNode) -> ProofNode:
    current = root
    while len(current.sequent) > 1:
        head = Or(current.sequent[0], current.sequent[1])
        parent_sequent = (head,) + current.sequent[2:]
        current = ProofNode(sequent=parent_sequent, rule="vee", children=[current])
    return current


def _finalize_result_root(result: EvaluationResult) -> EvaluationResult:
    if len(result.root.sequent) <= 1:
        return result
    return replace(result, root=_append_final_vee_chain(result.root))


def _equivalence_payload(
    *,
    source_formula: Formula,
    nnf_formula: Formula,
    final_formula: Formula,
    decomposition_style: str,
    is_two_sided: bool,
) -> dict[str, object]:
    steps: list[tuple[str, Formula]] = [("input", source_formula)]
    if nnf_formula != steps[-1][1]:
        steps.append(("NNF", nnf_formula))
    if final_formula != steps[-1][1]:
        final_label = (
            "cut-free normal form"
            if decomposition_style == "cut_free"
            else "decomposition root"
        )
        steps.append((final_label, final_formula))

    ascii_steps = [
        {"label": label, "formula": formula_to_ascii(formula)}
        for label, formula in steps
    ]
    latex_steps = [
        {"label": label, "formula": formula_to_latex(formula)}
        for label, formula in steps
    ]
    changed = len(steps) > 1
    note = (
        "Input turnstile sequent is interpreted in one-sided form first."
        if is_two_sided
        else ""
    )
    return {
        "changed": changed,
        "ascii_chain": " ≡ ".join(item["formula"] for item in ascii_steps),
        "latex_chain": " \\equiv ".join(item["formula"] for item in latex_steps),
        "steps_ascii": ascii_steps,
        "steps_latex": latex_steps,
        "note": note,
    }


def _belief_clause_gradients(
    formulas: tuple[Formula, ...],
    values: tuple[Fraction, ...],
) -> dict[Clause, Fraction]:
    mapping: dict[Clause, Fraction] = {}
    for formula, value in zip(formulas, values):
        for clause in clauses_from_formula(formula):
            existing = mapping.get(clause)
            if existing is None or value > existing:
                mapping[clause] = value
    return mapping


def _validate_gradient_count(
    *,
    gradients: tuple[Fraction, ...],
    beliefs: tuple[Formula, ...] | tuple[str, ...],
) -> None:
    if len(gradients) > len(beliefs):
        raise ValueError(
            "Too many gradient values: provide at most one gradient per belief."
        )


def generate_decomposition_bundle(
    sequent_text: str,
    belief_set_text: str = "",
    belief_mode: str = "standard",
    decomposition_style: str = "cut_free",
    belief_gradients_text: str = "",
) -> dict[str, object]:
    sequent_raw = sequent_text.strip()
    if not sequent_raw:
        raise ValueError("Please provide a sequent.")
    if belief_mode not in {"standard", "full", "gradient"}:
        raise ValueError("belief_mode must be 'standard', 'full' or 'gradient'.")
    if decomposition_style not in {"fractional", "cut_free"}:
        raise ValueError("decomposition_style must be 'fractional' or 'cut_free'.")

    parsed = parse_sequent_structured(sequent_raw)
    source_sequent = parsed.one_sided
    beliefs = parse_belief_set_text(belief_set_text.strip())
    belief_base = BeliefBase.from_strings(beliefs) if beliefs else None
    belief_strengths: dict[Clause, Fraction] = {}
    explicit_gradient_clauses: set[Clause] = set()
    if belief_mode == "gradient":
        gradients = tuple(parse_gradient_values_text(belief_gradients_text.strip()))
        _validate_gradient_count(gradients=gradients, beliefs=tuple(beliefs))
        if belief_base is not None:
            completed = tuple(
                gradients[idx] if idx < len(gradients) else Fraction(1, 1)
                for idx in range(len(belief_base.formulas))
            )
            belief_strengths = _belief_clause_gradients(
                belief_base.formulas,
                completed,
            )
            explicit_gradient_clauses = set(belief_strengths)
    elif belief_mode == "full":
        gradients = tuple(parse_gradient_values_text(belief_gradients_text.strip()))
        _validate_gradient_count(gradients=gradients, beliefs=tuple(beliefs))
        if gradients and belief_base is not None:
            explicit_formulas = belief_base.formulas[: len(gradients)]
            belief_strengths = _belief_clause_gradients(explicit_formulas, gradients)
            explicit_gradient_clauses = set(belief_strengths)
    belief_clauses = set(belief_base.clauses) if belief_base is not None else set()
    raw_result = evaluate_sequent(source_sequent, belief_clauses=belief_clauses)
    result = _finalize_result_root(raw_result)
    with_b = belief_base is not None
    source_root = _collapse_disjunction_sequent(source_sequent)
    normalized_source = tuple(to_nnf(formula) for formula in source_sequent)
    expected_root = _collapse_disjunction_sequent(normalized_source)
    root_sequent_latex = sequent_to_latex_grouped(expected_root)
    root_sequent_ascii = sequent_to_ascii_grouped(expected_root)
    source_root_latex = sequent_to_latex_grouped(source_root)
    source_root_ascii = sequent_to_ascii_grouped(source_root)
    if decomposition_style == "cut_free":
        cut_free_root = build_cut_free_tree(
            expected_root[0],
            belief_clauses=belief_clauses,
        )
        source_formula = source_root[0]
        root_formula = cut_free_root_formula(cut_free_root)
        use_source_override = root_formula == source_formula
        decomposition = render_cut_free_prooftree(
            cut_free_root,
            with_belief_base=with_b,
            belief_mode=belief_mode,
            belief_strengths=belief_strengths,
            explicit_gradient_clauses=explicit_gradient_clauses,
            root_sequent_latex=source_root_latex if use_source_override else None,
        )
        visual_tree = build_cut_free_visual_tree(
            cut_free_root,
            with_belief_base=with_b,
            belief_mode=belief_mode,
            belief_strengths=belief_strengths,
            explicit_gradient_clauses=explicit_gradient_clauses,
            root_sequent_ascii=source_root_ascii if use_source_override else None,
        )
        validation_ok = True
        if use_source_override:
            validation_message = (
                "Cut-free/context-free decomposition generated; "
                "root kept equal to input (same structure)."
            )
        else:
            validation_message = (
                "Cut-free/context-free decomposition generated; "
                "root shown in cut-free normal form (input-preserving root was not structurally valid)."
            )
        final_formula = source_formula if use_source_override else root_formula
    else:
        decomposition = render_decomposition_prooftree(
            result,
            with_belief_base=with_b,
            belief_mode=belief_mode,
            belief_strengths=belief_strengths,
            explicit_gradient_clauses=explicit_gradient_clauses,
            root_sequent_latex=root_sequent_latex,
        )
        visual_tree = build_visual_tree(
            result,
            with_belief_base=with_b,
            belief_mode=belief_mode,
            belief_strengths=belief_strengths,
            explicit_gradient_clauses=explicit_gradient_clauses,
            root_sequent_ascii=root_sequent_ascii,
        )
        validation = validate_proof_tree(
            result.root,
            expected_root=expected_root,
            source_sequent=source_sequent,
        )
        validation_ok = validation.ok
        validation_message = validation.message
        final_formula = expected_root[0]
    equivalence = _equivalence_payload(
        source_formula=source_root[0],
        nnf_formula=expected_root[0],
        final_formula=final_formula,
        decomposition_style=decomposition_style,
        is_two_sided=parsed.is_two_sided,
    )
    return {
        "decomposition": decomposition,
        "tree": visual_tree,
        "with_belief_base": with_b,
        "beliefs": beliefs,
        "belief_mode": belief_mode,
        "decomposition_style": decomposition_style,
        "sequent_kind": "two-sided" if parsed.is_two_sided else "one-sided",
        "validation": {"ok": validation_ok, "message": validation_message},
        "equivalence": equivalence,
    }


def generate_decomposition(
    sequent_text: str,
    belief_set_text: str = "",
    belief_mode: str = "standard",
    decomposition_style: str = "cut_free",
    belief_gradients_text: str = "",
) -> str:
    bundle = generate_decomposition_bundle(
        sequent_text,
        belief_set_text,
        belief_mode,
        decomposition_style,
        belief_gradients_text,
    )
    return str(bundle["decomposition"])
