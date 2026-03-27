from __future__ import annotations

from dataclasses import dataclass
import json
import os
import random
from urllib import error as urlerror
from urllib import request

from .input_parsing import parse_belief_set_text
from .logic import (
    And,
    Atom,
    Formula,
    Not,
    Or,
    formula_to_ascii,
    parse_formula,
    parse_sequent_structured,
)


@dataclass(frozen=True)
class ExampleResult:
    text: str
    engine: str
    model: str


_MAPPING_PHRASES = (
    "I have my house keys",
    "my phone battery is above 30%",
    "I packed an umbrella",
    "I have enough cash with me",
    "the supermarket is still open",
    "the bus arrives on time",
    "I finished my homework",
    "I set an alarm for tomorrow",
    "I bought groceries",
    "my bike tires are inflated",
    "I charged my laptop",
    "I have a clean shirt ready",
    "my friend confirmed the plan",
    "I have a water bottle with me",
    "the weather forecast says no rain",
)

_SCENARIO_TEMPLATES = (
    "In an everyday {domain} situation, a person decides whether to {action}. "
    "The decision follows the listed boolean conditions.",
    "During a normal {domain} routine, someone checks if they can {action}. "
    "Each mapped atom is treated as a concrete yes/no fact.",
    "A person preparing for {domain} evaluates whether to {action}. "
    "Only combinations of facts that satisfy the rule are accepted.",
    "In a daily-life {domain} context, a person verifies conditions before trying to {action}. "
    "The mapped atoms are the exact facts used in that choice.",
)

_DOMAINS = (
    "morning commute",
    "grocery shopping",
    "study planning",
    "evening workout",
    "meeting preparation",
    "trip organization",
    "home routine",
)

_ACTIONS = (
    "leave home now",
    "buy everything in one trip",
    "finish tasks before dinner",
    "join the plan with friends",
    "go out without delays",
    "start the activity immediately",
)

_SYSTEM_RANDOM = random.SystemRandom()


def _collect_atoms(formula: Formula) -> set[str]:
    if isinstance(formula, Atom):
        return {formula.name}
    if isinstance(formula, Not):
        return _collect_atoms(formula.arg)
    if isinstance(formula, And):
        return _collect_atoms(formula.left) | _collect_atoms(formula.right)
    if isinstance(formula, Or):
        return _collect_atoms(formula.left) | _collect_atoms(formula.right)
    raise TypeError(f"Unsupported formula node: {formula!r}")


def _truncate_text(text: str, limit: int = 1300) -> str:
    cleaned = "\n".join(line.rstrip() for line in text.strip().splitlines())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 3].rstrip() + "..."


def _connective_counts(formula: Formula) -> tuple[int, int, int]:
    if isinstance(formula, Atom):
        return (0, 0, 0)
    if isinstance(formula, Not):
        a, o, n = _connective_counts(formula.arg)
        return (a, o, n + 1)
    if isinstance(formula, And):
        a1, o1, n1 = _connective_counts(formula.left)
        a2, o2, n2 = _connective_counts(formula.right)
        return (a1 + a2 + 1, o1 + o2, n1 + n2)
    if isinstance(formula, Or):
        a1, o1, n1 = _connective_counts(formula.left)
        a2, o2, n2 = _connective_counts(formula.right)
        return (a1 + a2, o1 + o2 + 1, n1 + n2)
    raise TypeError(f"Unsupported formula node: {formula!r}")


def _connective_note(formulas: tuple[Formula, ...]) -> str:
    and_count = 0
    or_count = 0
    not_count = 0
    for formula in formulas:
        a, o, n = _connective_counts(formula)
        and_count += a
        or_count += o
        not_count += n

    descriptors: list[str] = []
    if and_count:
        descriptors.append("joint constraints")
    if or_count:
        descriptors.append("alternative branches")
    if not_count:
        descriptors.append("explicit negations")
    if not descriptors:
        return "atomic condition checks"
    return ", ".join(descriptors)


def _build_prompt(
    sequent_text: str,
    one_sided_text: str,
    antecedent_text: str,
    succedent_text: str,
    belief_text: str,
    atoms: tuple[str, ...],
) -> str:
    atom_hint = ", ".join(atoms) if atoms else "(no atoms detected)"
    belief_hint = belief_text.strip() if belief_text.strip() else "empty"
    return (
        "You are a strict example generator for propositional sequents.\n"
        "Create exactly one concrete micro-scenario from daily life, consistent with the sequent.\n"
        "If the belief set is not empty, explicitly treat those beliefs as fixed background assumptions.\n"
        "No chat, no introductions, no markdown headings.\n"
        "Use this exact 3-part format:\n"
        "Scenario: <2-4 sentences>\n"
        "Mapping: <comma-separated mapping like p=..., q=...>\n"
        "Check: <one sentence explaining why the scenario fits the sequent and beliefs>\n\n"
        f"Input sequent: {sequent_text}\n"
        f"One-sided interpretation: {one_sided_text}\n"
        f"Antecedent (if present): {antecedent_text}\n"
        f"Succedent: {succedent_text}\n"
        f"Belief set B: {belief_hint}\n"
        f"Atoms: {atom_hint}\n"
    )


def _generate_with_ollama(
    prompt: str,
    model: str,
    *,
    timeout_sec: int,
) -> str | None:
    base_url = os.environ.get("FS_OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.6, "num_predict": 240},
    }
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=f"{base_url}/api/generate",
        data=body,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    try:
        with request.urlopen(req, timeout=timeout_sec) as response:
            raw = response.read().decode("utf-8")
    except (urlerror.URLError, TimeoutError, ConnectionError):
        return None
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return None
    text = str(parsed.get("response", "")).strip()
    return _truncate_text(text) if text else None


def _fallback_example(
    *,
    sequent_text: str,
    one_sided_text: str,
    antecedent_text: str,
    succedent_text: str,
    atoms: tuple[str, ...],
    one_sided_formulas: tuple[Formula, ...],
    belief_formulas: tuple[Formula, ...],
) -> str:
    if not atoms:
        atoms = ("p", "q")

    shift = _SYSTEM_RANDOM.randrange(len(_MAPPING_PHRASES))
    mappings: list[str] = []
    for idx, atom in enumerate(atoms):
        phrase = _MAPPING_PHRASES[(shift + idx) % len(_MAPPING_PHRASES)]
        mappings.append(f"{atom}={phrase}")

    scenario_template = _SYSTEM_RANDOM.choice(_SCENARIO_TEMPLATES)
    scenario = scenario_template.format(
        domain=_SYSTEM_RANDOM.choice(_DOMAINS),
        action=_SYSTEM_RANDOM.choice(_ACTIONS),
    )
    connective_note = _connective_note(one_sided_formulas)
    belief_text = ", ".join(formula_to_ascii(formula) for formula in belief_formulas)
    if belief_text:
        scenario += f" The person assumes these prior beliefs are true: {belief_text}."
    check = (
        "This reading includes "
        f"{connective_note}, and the mapped situation is one concrete model of the formula."
    )
    if antecedent_text:
        check = (
            "Assuming the antecedent holds, the same mapped situation illustrates the succedent "
            f"using {connective_note}."
        )
    if belief_text:
        check += f" It is also compatible with B={belief_text}."

    return _truncate_text(
        "\n".join(
            [
                f"Scenario: {scenario}",
                f"Mapping: {', '.join(mappings)}",
                (
                    "Check: "
                    f"Input={sequent_text}; one-sided={one_sided_text}; "
                    f"antecedent={antecedent_text or 'empty'}; succedent={succedent_text}. {check}"
                ),
            ]
        )
    )


def generate_example_from_sequent(
    sequent_text: str,
    belief_set_text: str = "",
    *,
    model: str | None = None,
    prefer_local_model: bool = True,
    generation_mode: str = "auto",
    timeout_sec: int = 18,
) -> ExampleResult:
    raw = sequent_text.strip()
    if not raw:
        raise ValueError("Please provide a sequent.")
    if generation_mode not in {"auto", "local_only", "template_only"}:
        raise ValueError(
            "generation_mode must be one of: auto, local_only, template_only."
        )
    parsed = parse_sequent_structured(raw)
    belief_items = parse_belief_set_text(belief_set_text)
    belief_formulas = tuple(parse_formula(item) for item in belief_items)
    one_sided_text = ", ".join(formula_to_ascii(f) for f in parsed.one_sided)
    antecedent_text = ", ".join(formula_to_ascii(f) for f in parsed.antecedent)
    succedent_text = ", ".join(formula_to_ascii(f) for f in parsed.succedent)
    belief_text = ", ".join(formula_to_ascii(formula) for formula in belief_formulas)

    atom_set: set[str] = set()
    for formula in parsed.one_sided:
        atom_set |= _collect_atoms(formula)
    for formula in belief_formulas:
        atom_set |= _collect_atoms(formula)
    atoms = tuple(sorted(atom_set))

    model_name = (model or "").strip() or os.environ.get("FS_OLLAMA_MODEL", "llama3.2:3b")
    prompt = _build_prompt(
        sequent_text=raw,
        one_sided_text=one_sided_text,
        antecedent_text=antecedent_text,
        succedent_text=succedent_text,
        belief_text=belief_text,
        atoms=atoms,
    )
    should_try_model = (
        generation_mode == "local_only"
        or (generation_mode == "auto" and prefer_local_model)
    )
    if should_try_model:
        generated = _generate_with_ollama(prompt, model_name, timeout_sec=timeout_sec)
        if generated:
            return ExampleResult(text=generated, engine="ollama", model=model_name)
        if generation_mode == "local_only":
            raise RuntimeError(
                "Local model generation failed. "
                "Verify that Ollama is running and the selected model is available."
            )

    return ExampleResult(
        text=_fallback_example(
            sequent_text=raw,
            one_sided_text=one_sided_text,
            antecedent_text=antecedent_text,
            succedent_text=succedent_text,
            atoms=atoms,
            one_sided_formulas=parsed.one_sided,
            belief_formulas=belief_formulas,
        ),
        engine="template",
        model="local-template",
    )
