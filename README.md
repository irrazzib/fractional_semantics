# Fractional Semantics Calculator

Python toolkit to:

- automatically compute Fractional Semantics values from a sequent;
- manage a belief base `B` (including decomposition and cut-closure);
- apply `contraction`, `expansion`, and `revision` (Levi identity);
- generate readable LaTeX output.

## Requirements

- Python 3.10+

## Output Folder

Each new computation creates a fresh subfolder in:

- `tex_outputs/`

Inside that subfolder the tool writes `.tex` files and automatically runs `pdflatex` to produce `.pdf`.

Example subfolders:

- `tex_outputs/analysis_20260302_104500/`
- `tex_outputs/revision_20260302_104612/`
- `tex_outputs/revision_demo_20260302_104700/`

You can still compile manually inside a run folder if needed:

```bash
cd tex_outputs/analysis_YYYYMMDD_HHMMSS
pdflatex fractional_report.tex
```

## Quick Start

### Visual Interface (decomposition only)

```bash
python3 -m fs_semantics gui
```

The GUI provides:

- `Sequent` input
- side-by-side `Belief set B` and `Gradient values` inputs
- `Belief Mode` switch: `Standard B` / `Full Beliefs (delta)` / `Gradient Beliefs`
- configurable PNG `DPI` for high-resolution preview
- visual tree output
- LaTeX `prooftree` decomposition output
- a legend for OR, AND, and negation input syntax
- one-click LaTeX compilation for PNG tree preview (landscape, tree-only)
- run history via dropdown menu
- PNG download button
- decomposition validation feedback (parentheses-aware structural check)

If Tkinter is missing in your Python build, `gui` automatically falls back to a browser-based interface.
You can also launch it directly:

```bash
python3 -m fs_semantics web-gui
```

In `Full Beliefs (delta)` mode, every belief contribution is decorated with
`\\delta_{...}` in the turnstile value (e.g. `1-(\\delta_{...})`).

In `Gradient Beliefs` mode, each belief can be paired with a confidence in
`[0,1]` (same order as the belief list). Missing values default to `1`
(therefore treated as full beliefs).

In `Full Beliefs (delta)` mode, the gradient field can also be used: if you
provide one or more values, the run becomes hybrid: provided beliefs use
graded values with delta decorations (`v-\\delta`), while remaining beliefs
keep the full form (`1-\\delta`).

### 1) Analyze a sequent

```bash
python3 -m fs_semantics analyze \
  --sequent "(p | q) & (p | ~p) & (~r & ~t)"
```

Supported sequent syntaxes:

- one-sided: `A, B, C` (commas are disjunction in GS4);
- two-sided: `Γ |- Δ`, `Γ ⊢ Δ`, or `Γ \vdash Δ` (left commas as conjunction, right commas as disjunction).

Example:

```bash
python3 -m fs_semantics analyze \
  --sequent "p, q |- r, (s & t)"
```

### 2) Analyze with beliefs `B`

```bash
python3 -m fs_semantics analyze \
  --sequent "(p | q) & (s | ~s) & (~r & ~t)" \
  --belief "(p | q)" \
  --belief "~u"
```

You can also provide the whole belief set in one prompt-style argument:

```bash
python3 -m fs_semantics analyze \
  --sequent "(p | q) & (s | ~s) & (~r & ~t)" \
  --belief-set "B={(p | q); ~u}"
```

### 3) Apply Belief Revision on `B`

```bash
python3 -m fs_semantics revise \
  --operation revise \
  --target p \
  --sequent "(p & q)" \
  --belief "~p" \
  --belief "q"
```

### 4) Generate automatic Belief Revision examples

```bash
python3 -m fs_semantics revision-demo
```

This generates:

- `belief_revision_demo_contraction.tex` (in the run folder)
- `belief_revision_demo_expansion.tex` (in the run folder)
- `belief_revision_demo_revision.tex` (in the run folder)

### 5) Generate an AFP/Isabelle certificate for a decomposition

```bash
python3 -m fs_semantics afp-certify \
  --sequent "(p | q) & (p | ~p) & (~r & ~t)" \
  --belief-set "B={(p | q); ~u}" \
  --theory-name FS_Cert_Example
```

This creates a run folder in `tex_outputs/` containing:

- `FS_Cert_Example.thy`
- `ROOT`

The generated theory imports AFP `Propositional_Proof_Systems.SC` and checks that
the decomposition tree is a valid composition of `SCp` rule steps from its leaves.
`ax` leaves are proved classically in the generated theory; `belief` and `\overline{ax.}`
leaves are represented as explicit assumptions (certificate premises).

Optional machine-check run (requires Isabelle + AFP):

```bash
python3 -m fs_semantics afp-certify \
  --sequent "(p | q) & (p | ~p) & (~r & ~t)" \
  --verify \
  --afp-thys "/path/to/afp/thys"
```

You can also set `AFP_THYS=/path/to/afp/thys` instead of `--afp-thys`.

### 6) Pure propositional sequent-calculus reference decomposition

Run a decomposition without Fractional decorations:

```bash
python3 -m fs_semantics sc-reference \
  --sequent "p, q |- r, (s & t)" \
  --show-tree
```

### 7) Compare Fractional decomposition with pure sequent-calculus decomposition

```bash
python3 -m fs_semantics sc-compare \
  --sequent "(p | q) & (s | ~s) & (~r & ~t)" \
  --belief-set "B={(p | q); ~u}" \
  --show-tree
```

The comparison checks whether both systems produce the same multiset of leaf
clauses (decomposition shape), while allowing Fractional Semantics to keep its
own decorated turnstiles and values.

## Model Notes

- Rules used: analytic `GS4` decomposition (`∨` and `∧`).
- Two-sided inputs are normalized to one-sided GS4 form:
  - `Γ |- Δ` is transformed into `⊢ ¬Γ, Δ`;
  - right-side commas are disjunctions (clausal context);
  - left-side commas are conjunctions before negation.
- Rendered sequents use minimal (precedence-based) parentheses, adding brackets only when needed for unambiguous grouping.
- Leaf classification:
  - `ax.` for tautological clauses (`p` and `¬p`);
  - `b` for clauses in the cut-closure of `B`;
  - `\overline{ax.}` otherwise.
- Values:
  - `[[Γ]] = top^1 / (top^1 + top^0)`
  - `[[Γ]]_B = (top^1 + top^b) / (top^1 + top^b + top^0)`

## Empty Belief Set Behavior

If `B` is empty, the generated proof tree uses the undecorated turnstile
without `_B` (i.e., plain Fractional Semantics).

## ATP-Coherence Notes (while staying Fractional Semantics only)

The implementation is intentionally restricted to the analytic GS4-style
fragment used by Fractional Semantics, but remains coherent with standard
ATP/sequent-calculus expectations:

- context/exchange invariance in decomposition validation:
  permutations of formulas in sequents are accepted as equivalent rule steps;
- explicit sequent normalization:
  two-sided sequents are mapped to one-sided GS4 sequents (`⊢ ¬Γ, Δ`);
- parentheses-aware rendering/validation:
  grouped output keeps comma semantics explicit (`Γ` on left as conjunction,
  `Δ` on right as disjunction);
- cut-compatible belief handling:
  belief clauses can be closed under cut/resolution before scoring leaves;
- proof objects are checkable:
  every generated decomposition is structurally validated (`∨`, `∧`, literal
  leaves, expected normalized root).

Out of scope by design: first-order ATP features (unification, quantifiers,
resolution saturation loops, focusing strategies beyond this fragment).
