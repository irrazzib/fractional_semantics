from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Sequence


class Formula:
    """Base class for propositional formulas."""


@dataclass(frozen=True)
class Atom(Formula):
    name: str


@dataclass(frozen=True)
class Not(Formula):
    arg: Formula


@dataclass(frozen=True)
class And(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Or(Formula):
    left: Formula
    right: Formula


@dataclass(frozen=True)
class Literal:
    atom: str
    negated: bool = False

    def complement(self) -> "Literal":
        return Literal(self.atom, not self.negated)

    def to_latex(self) -> str:
        return f"\\neg {self.atom}" if self.negated else self.atom


@dataclass(frozen=True)
class ParsedSequent:
    antecedent: tuple["Formula", ...]
    succedent: tuple["Formula", ...]
    one_sided: tuple["Formula", ...]
    is_two_sided: bool


def _preprocess(text: str) -> str:
    out = text
    out = re.sub(
        r"\\overline\s*\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}",
        r"¬\1",
        out,
    )
    replacements = {
        "\\wedge": "∧",
        "\\vee": "∨",
        "\\neg": "¬",
        "\\lnot": "¬",
        "&&": "∧",
        "||": "∨",
        "/\\": "∧",
        "\\/": "∨",
        "~": "¬",
        "!": "¬",
    }
    for src, dst in replacements.items():
        out = out.replace(src, dst)

    out = re.sub(r"\bAND\b", "∧", out, flags=re.IGNORECASE)
    out = re.sub(r"\bOR\b", "∨", out, flags=re.IGNORECASE)
    out = re.sub(r"\bNOT\b", "¬", out, flags=re.IGNORECASE)
    return out


def _tokenize(text: str) -> list[str]:
    prepared = _preprocess(text)
    tokens: list[str] = []
    i = 0
    while i < len(prepared):
        c = prepared[i]
        if c.isspace():
            i += 1
            continue
        if c in {"(", ")", ",", "¬", "∧", "∨"}:
            tokens.append(c)
            i += 1
            continue
        if c in {"|", "&"}:
            tokens.append("∨" if c == "|" else "∧")
            i += 1
            continue
        if c.isalpha() or c == "_":
            j = i + 1
            while j < len(prepared) and (
                prepared[j].isalnum() or prepared[j] == "_"
            ):
                j += 1
            tokens.append(prepared[i:j])
            i = j
            continue
        raise ValueError(f"Invalid character '{c}' in: {text!r}")
    return tokens


class _Parser:
    def __init__(self, tokens: Sequence[str]) -> None:
        self.tokens = list(tokens)
        self.pos = 0

    def parse_formula(self) -> Formula:
        result = self._parse_or()
        if self.pos != len(self.tokens):
            raise ValueError(
                f"Unexpected token: {self.tokens[self.pos]!r} at position {self.pos}"
            )
        return result

    def _peek(self) -> str | None:
        if self.pos >= len(self.tokens):
            return None
        return self.tokens[self.pos]

    def _consume(self, expected: str | None = None) -> str:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of input")
        if expected is not None and token != expected:
            raise ValueError(f"Expected {expected!r}, found {token!r}")
        self.pos += 1
        return token

    def _parse_or(self) -> Formula:
        left = self._parse_and()
        while self._peek() == "∨":
            self._consume("∨")
            right = self._parse_and()
            left = Or(left, right)
        return left

    def _parse_and(self) -> Formula:
        left = self._parse_not()
        while self._peek() == "∧":
            self._consume("∧")
            right = self._parse_not()
            left = And(left, right)
        return left

    def _parse_not(self) -> Formula:
        if self._peek() == "¬":
            self._consume("¬")
            return Not(self._parse_not())
        return self._parse_atom()

    def _parse_atom(self) -> Formula:
        token = self._peek()
        if token is None:
            raise ValueError("Incomplete formula")
        if token == "(":
            self._consume("(")
            inside = self._parse_or()
            self._consume(")")
            return inside
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", token):
            self._consume()
            return Atom(token)
        raise ValueError(f"Invalid token in formula: {token!r}")


_TURNSTILE_RE = re.compile(r"(\\vdash|⊢|\|-)")


def split_top_level(text: str, separator: str = ",") -> list[str]:
    parts: list[str] = []
    depth = 0
    last = 0
    for idx, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("Unbalanced parentheses: too many closing brackets")
        elif char == separator and depth == 0:
            parts.append(text[last:idx].strip())
            last = idx + 1
    if depth != 0:
        raise ValueError("Unbalanced parentheses")
    tail = text[last:].strip()
    if tail:
        parts.append(tail)
    return parts


def parse_formula(text: str) -> Formula:
    tokens = _tokenize(text)
    if not tokens:
        raise ValueError("Empty formula")
    return _Parser(tokens).parse_formula()


def _parse_formula_list(text: str) -> tuple[Formula, ...]:
    stripped = text.strip()
    if not stripped:
        return tuple()
    pieces = split_top_level(stripped, ",")
    if not pieces:
        return tuple()
    return tuple(parse_formula(piece) for piece in pieces)


def parse_sequent_structured(text: str) -> ParsedSequent:
    raw = text.strip()
    if not raw:
        raise ValueError("Empty sequent")

    markers = list(_TURNSTILE_RE.finditer(raw))
    if len(markers) > 1:
        raise ValueError("Multiple turnstile symbols found in sequent input.")

    if not markers:
        succedent = _parse_formula_list(raw)
        if not succedent:
            raise ValueError("Empty sequent")
        return ParsedSequent(
            antecedent=tuple(),
            succedent=succedent,
            one_sided=succedent,
            is_two_sided=False,
        )

    marker = markers[0]
    left_raw = raw[: marker.start()].strip()
    right_raw = raw[marker.end() :].strip()
    succedent = _parse_formula_list(right_raw)
    if not succedent:
        raise ValueError("The right side of the sequent cannot be empty.")
    antecedent = _parse_formula_list(left_raw)
    one_sided = tuple(Not(formula) for formula in antecedent) + succedent
    return ParsedSequent(
        antecedent=antecedent,
        succedent=succedent,
        one_sided=one_sided,
        is_two_sided=True,
    )


def parse_sequent(text: str) -> tuple[Formula, ...]:
    return parse_sequent_structured(text).one_sided


def to_nnf(formula: Formula) -> Formula:
    if isinstance(formula, Atom):
        return formula
    if isinstance(formula, Not):
        inner = formula.arg
        if isinstance(inner, Atom):
            return formula
        if isinstance(inner, Not):
            return to_nnf(inner.arg)
        if isinstance(inner, And):
            return Or(to_nnf(Not(inner.left)), to_nnf(Not(inner.right)))
        if isinstance(inner, Or):
            return And(to_nnf(Not(inner.left)), to_nnf(Not(inner.right)))
        raise TypeError(f"Unsupported node: {inner!r}")
    if isinstance(formula, And):
        return And(to_nnf(formula.left), to_nnf(formula.right))
    if isinstance(formula, Or):
        return Or(to_nnf(formula.left), to_nnf(formula.right))
    raise TypeError(f"Unsupported node: {formula!r}")


def formula_to_literal(formula: Formula) -> Literal:
    if isinstance(formula, Atom):
        return Literal(formula.name, False)
    if isinstance(formula, Not) and isinstance(formula.arg, Atom):
        return Literal(formula.arg.name, True)
    raise ValueError(f"Formula is not a literal: {formula!r}")


def formula_to_latex(formula: Formula) -> str:
    return _formula_to_latex(formula, parent_prec=0)


def formula_to_latex_grouped(
    formula: Formula,
    *,
    outer_parentheses: bool = False,
) -> str:
    rendered = formula_to_latex(formula)
    if outer_parentheses:
        return f"({rendered})"
    return rendered


def _formula_to_latex(formula: Formula, parent_prec: int) -> str:
    if isinstance(formula, Atom):
        return formula.name
    if isinstance(formula, Not):
        if isinstance(formula.arg, Atom):
            core = _formula_to_latex(formula.arg, parent_prec=3)
            return f"\\neg {core}"
        core = _formula_to_latex(formula.arg, parent_prec=0)
        return f"\\neg({core})"
    if isinstance(formula, And):
        prec = 2
        left = _formula_to_latex(formula.left, parent_prec=prec)
        right = _formula_to_latex(formula.right, parent_prec=prec + 1)
        expr = f"{left} \\wedge {right}"
        if prec < parent_prec:
            return f"({expr})"
        return expr
    if isinstance(formula, Or):
        prec = 1
        left = _formula_to_latex(formula.left, parent_prec=prec)
        right = _formula_to_latex(formula.right, parent_prec=prec + 1)
        expr = f"{left} \\vee {right}"
        if prec < parent_prec:
            return f"({expr})"
        return expr
    raise TypeError(f"Unsupported node: {formula!r}")


def formula_to_ascii(formula: Formula) -> str:
    return _formula_to_ascii(formula, parent_prec=0)


def formula_to_ascii_grouped(
    formula: Formula,
    *,
    outer_parentheses: bool = False,
) -> str:
    rendered = formula_to_ascii(formula)
    if outer_parentheses:
        return f"({rendered})"
    return rendered


def _formula_to_ascii(formula: Formula, parent_prec: int) -> str:
    if isinstance(formula, Atom):
        return formula.name
    if isinstance(formula, Not):
        if isinstance(formula.arg, Atom):
            inner = _formula_to_ascii(formula.arg, parent_prec=3)
            return f"~{inner}"
        inner = _formula_to_ascii(formula.arg, parent_prec=0)
        return f"~({inner})"
    if isinstance(formula, And):
        prec = 2
        left = _formula_to_ascii(formula.left, parent_prec=prec)
        right = _formula_to_ascii(formula.right, parent_prec=prec + 1)
        expr = f"{left} & {right}"
        if prec < parent_prec:
            return f"({expr})"
        return expr
    if isinstance(formula, Or):
        prec = 1
        left = _formula_to_ascii(formula.left, parent_prec=prec)
        right = _formula_to_ascii(formula.right, parent_prec=prec + 1)
        expr = f"{left} | {right}"
        if prec < parent_prec:
            return f"({expr})"
        return expr
    raise TypeError(f"Unsupported node: {formula!r}")


def sequent_to_latex(formulas: Iterable[Formula]) -> str:
    return ", ".join(formula_to_latex(formula) for formula in formulas)


def sequent_to_latex_grouped(formulas: Iterable[Formula]) -> str:
    return ", ".join(
        formula_to_latex_grouped(formula, outer_parentheses=False) for formula in formulas
    )


def sequent_to_ascii(formulas: Iterable[Formula]) -> str:
    return ", ".join(formula_to_ascii(formula) for formula in formulas)


def sequent_to_ascii_grouped(formulas: Iterable[Formula]) -> str:
    return ", ".join(
        formula_to_ascii_grouped(formula, outer_parentheses=False) for formula in formulas
    )
