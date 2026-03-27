from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import product
import random
import os
import json
import secrets
from urllib.parse import unquote
import webbrowser

from .decomposition_service import generate_decomposition_bundle
from .example_generator import generate_example_from_sequent
from .latex_compile import compile_decomposition_png
from .logic import And, Atom, Formula, Not, Or, formula_to_ascii


_ATOM_POOL = ("p", "q", "r", "s", "t", "u", "v", "w")


def _random_formula(rng: random.Random, depth: int) -> Formula:
    if depth <= 0:
        return Atom(rng.choice(_ATOM_POOL))
    if depth == 1:
        atom = Atom(rng.choice(_ATOM_POOL))
        if rng.random() < 0.22:
            return Not(atom)
        return atom

    roll = rng.random()
    if roll < 0.2:
        return Not(_random_formula(rng, depth - 1))

    left = _random_formula(rng, depth - 1)
    right = _random_formula(rng, depth - 1)
    retries = 0
    while right == left and retries < 12:
        right = _random_formula(rng, depth - 1)
        retries += 1
    if right == left:
        # Deterministic fallback to avoid trivial repetitions such as (u | u).
        right = Not(right) if not isinstance(right, Not) else right.arg
    if rng.random() < 0.5:
        return And(left, right)
    return Or(left, right)


def _random_distinct_formulas(
    rng: random.Random,
    count: int,
    max_depth: int,
) -> list[Formula]:
    formulas: list[Formula] = []
    while len(formulas) < count:
        candidate = _random_formula(rng, rng.randint(1, max_depth))
        if candidate not in formulas:
            formulas.append(candidate)
    return formulas


def _merge_formulas(formulas: list[Formula], connective: str) -> Formula:
    if not formulas:
        raise ValueError("Cannot merge an empty formula list.")
    merged = formulas[0]
    for formula in formulas[1:]:
        if connective == "and":
            merged = And(merged, formula)
        elif connective == "or":
            merged = Or(merged, formula)
        else:
            raise ValueError(f"Unsupported connective: {connective}")
    return merged


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


def _eval_formula(formula: Formula, assignment: dict[str, bool]) -> bool:
    if isinstance(formula, Atom):
        return bool(assignment[formula.name])
    if isinstance(formula, Not):
        return not _eval_formula(formula.arg, assignment)
    if isinstance(formula, And):
        return _eval_formula(formula.left, assignment) and _eval_formula(
            formula.right, assignment
        )
    if isinstance(formula, Or):
        return _eval_formula(formula.left, assignment) or _eval_formula(
            formula.right, assignment
        )
    raise TypeError(f"Unsupported formula node: {formula!r}")


def is_coherent_formula(formula: Formula) -> bool:
    atoms = sorted(_collect_atoms(formula))
    if not atoms:
        return True
    for values in product((False, True), repeat=len(atoms)):
        assignment = dict(zip(atoms, values))
        if _eval_formula(formula, assignment):
            return True
    return False


def _random_right_formula(active_rng: random.Random) -> Formula:
    max_depth = active_rng.randint(2, 3)
    right_count = active_rng.randint(1, 3)
    right_formulas = _random_distinct_formulas(active_rng, right_count, max_depth)
    connective = "or" if active_rng.random() < 0.7 else "and"
    return _merge_formulas(right_formulas, connective)


def generate_random_coherent_sequent(
    *,
    rng: random.Random | None = None,
    max_attempts: int = 250,
) -> str:
    active_rng = rng if rng is not None else random.Random()
    for _ in range(max_attempts):
        formula = _random_right_formula(active_rng)
        if is_coherent_formula(formula):
            return f"\\vdash {formula_to_ascii(formula)}"
    # Deterministic satisfiable fallback.
    fallback = Atom(active_rng.choice(_ATOM_POOL))
    return f"\\vdash {fallback.name}"


def generate_random_incoherent_sequent(
    *,
    rng: random.Random | None = None,
) -> str:
    active_rng = rng if rng is not None else random.Random()
    atom = active_rng.choice(_ATOM_POOL)
    contradiction = And(Atom(atom), Not(Atom(atom)))
    if active_rng.random() < 0.5:
        other = _random_right_formula(active_rng)
        contradiction = And(other, contradiction)
    return f"\\vdash {formula_to_ascii(contradiction)}"


def generate_random_sequent(
    *,
    rng: random.Random | None = None,
    allow_incoherent: bool = False,
) -> str:
    active_rng = rng if rng is not None else random.Random()
    if allow_incoherent and active_rng.random() < 0.5:
        return generate_random_incoherent_sequent(rng=active_rng)
    return generate_random_coherent_sequent(rng=active_rng)


_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fractional Semantics Decomposition</title>
  <style>
    :root {
      --bg: #f6f2ec;
      --panel: #fffdfa;
      --line: #d8cdc0;
      --ink: #182026;
      --muted: #4f5964;
      --accent: #1c7c6b;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      padding: 20px;
      background: radial-gradient(circle at 0% 0%, #fffdf8 0%, var(--bg) 60%);
      color: var(--ink);
      font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
    }
    .app {
      max-width: 1200px;
      margin: 0 auto;
      display: grid;
      gap: 16px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      min-width: 0;
    }
    h1 {
      margin: 0 0 8px;
      font-size: 1.35rem;
      letter-spacing: 0.01em;
    }
    p { margin: 0; color: var(--muted); }
    label {
      font-size: 0.92rem;
      font-weight: 700;
      display: block;
      margin-bottom: 6px;
    }
    input, textarea, select {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 12px;
      font: inherit;
      background: #fff;
      color: var(--ink);
    }
    textarea { min-height: 90px; resize: vertical; }
    .belief-grid {
      display: grid;
      grid-template-columns: 1fr 260px;
      gap: 12px;
      margin-top: 12px;
      align-items: start;
    }
    .field-hint {
      margin-top: 6px;
      color: var(--muted);
      font-size: 0.85rem;
    }
    .form-grid {
      display: grid;
      grid-template-columns: 1fr 220px 180px;
      gap: 12px;
      margin-top: 10px;
      align-items: end;
    }
    @media (max-width: 700px) {
      .belief-grid {
        grid-template-columns: 1fr;
      }
      .form-grid {
        grid-template-columns: 1fr;
      }
    }
    .controls {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 10px;
    }
    .history-row {
      display: grid;
      grid-template-columns: 1fr;
      gap: 6px;
      margin-top: 10px;
    }
    .example-tools {
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      align-items: end;
      margin-bottom: 10px;
    }
    .example-tools .mode-wrap {
      min-width: 260px;
      flex: 1;
    }
    button {
      border: none;
      border-radius: 8px;
      padding: 10px 14px;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
    }
    .primary {
      background: var(--accent);
      color: #fff;
    }
    .ghost {
      background: #ece6dd;
      color: var(--ink);
    }
    pre {
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.35;
      font-family: "IBM Plex Mono", "Menlo", monospace;
      min-height: 260px;
    }
    #exampleOut {
      min-height: 140px;
    }
    .grid-two {
      display: grid;
      gap: 16px;
      grid-template-columns: 1.2fr 1fr;
      align-items: start;
    }
    .grid-two > * {
      min-width: 0;
    }
    @media (max-width: 900px) {
      .grid-two { grid-template-columns: 1fr; }
    }
    .legend {
      display: grid;
      gap: 6px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 0.92rem;
    }
    .legend code {
      background: #ece6dd;
      padding: 2px 6px;
      border-radius: 6px;
    }
    .tree-wrap {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 10px;
      overflow: auto;
      min-height: 280px;
      max-height: 62vh;
    }
    .tree {
      min-width: max-content;
      font-family: "IBM Plex Mono", "Menlo", monospace;
    }
    .tree ul {
      position: relative;
      padding-top: 22px;
      margin: 0;
      display: flex;
      justify-content: center;
    }
    .tree li {
      list-style: none;
      text-align: center;
      position: relative;
      padding: 20px 8px 0 8px;
    }
    .tree li::before, .tree li::after {
      content: "";
      position: absolute;
      top: 0;
      right: 50%;
      border-top: 1px solid #b8aa98;
      width: 50%;
      height: 20px;
    }
    .tree li::after {
      right: auto;
      left: 50%;
      border-left: 1px solid #b8aa98;
    }
    .tree li:only-child::before, .tree li:only-child::after {
      display: none;
    }
    .tree li:only-child {
      padding-top: 0;
    }
    .tree li:first-child::before, .tree li:last-child::after {
      border: 0;
    }
    .tree li:last-child::before {
      border-right: 1px solid #b8aa98;
      border-radius: 0 6px 0 0;
    }
    .tree li:first-child::after {
      border-radius: 6px 0 0 0;
    }
    .tree ul ul::before {
      content: "";
      position: absolute;
      top: 0;
      left: 50%;
      border-left: 1px solid #b8aa98;
      width: 0;
      height: 22px;
    }
    .node {
      display: inline-block;
      min-width: 260px;
      max-width: 520px;
      padding: 8px 10px;
      border: 1px solid #bcae9c;
      border-radius: 8px;
      background: #fffdfa;
      text-align: left;
      box-shadow: 0 1px 1px rgba(0,0,0,0.03);
    }
    .node .meta {
      display: flex;
      gap: 8px;
      align-items: center;
      margin-bottom: 5px;
      font-size: 0.82rem;
      color: var(--muted);
    }
    .node .badge {
      padding: 2px 8px;
      border-radius: 999px;
      background: #ece6dd;
      color: var(--ink);
      font-weight: 700;
    }
    .node .seq {
      font-size: 0.92rem;
      color: var(--ink);
      word-break: break-word;
    }
    .error { color: #a32222; margin-top: 8px; }
    .ok { color: #1c7c6b; margin-top: 8px; }
    .warn { color: #ab5f00; margin-top: 8px; }
    .preview-tools {
      margin-top: 10px;
      margin-bottom: 8px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 10px;
      flex-wrap: wrap;
    }
    .zoom-group {
      display: flex;
      align-items: center;
      gap: 8px;
      flex-wrap: wrap;
    }
    .zoom-group input[type="range"] {
      width: 170px;
      padding: 0;
    }
    .zoom-label {
      min-width: 56px;
      text-align: right;
      font-size: 0.9rem;
      color: var(--muted);
      font-family: "IBM Plex Mono", "Menlo", monospace;
    }
    .preview-hint {
      color: var(--muted);
      font-size: 0.9rem;
    }
    #imgWrap {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      min-height: 320px;
      height: min(62vh, 560px);
      overflow: auto;
      padding: 8px;
      position: relative;
    }
    #imgWrap.empty::before {
      content: "Preview will appear here";
      color: var(--muted);
      font-size: 0.95rem;
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
    }
    #imgCanvas {
      min-width: max-content;
      min-height: max-content;
    }
    img#pngView {
      width: auto;
      height: auto;
      max-width: none;
      max-height: none;
      display: block;
    }
    .loading {
      position: fixed;
      right: 20px;
      bottom: 20px;
      display: none;
      align-items: center;
      gap: 10px;
      background: #182026;
      color: #fff;
      border-radius: 999px;
      padding: 9px 14px;
      box-shadow: 0 6px 22px rgba(0,0,0,0.25);
      z-index: 9999;
      font-size: 0.92rem;
    }
    .loading.show { display: inline-flex; }
    .spinner {
      width: 16px;
      height: 16px;
      border: 2px solid rgba(255,255,255,0.25);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin {
      to { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <main class="app">
    <section class="panel">
      <h1>Fractional Semantics Decomposition</h1>
      <p>Provide a sequent and optional beliefs. Output shows decomposition only.</p>
    </section>
    <section class="panel">
      <label for="sequent">Sequent</label>
      <input id="sequent" placeholder="(p | q) & (s | ~s) & (~r & ~t)  or  p \\vdash q, (r & s)" />

      <div class="belief-grid">
        <div>
          <label for="belief" style="margin-top:0;">Belief set B (optional)</label>
          <textarea id="belief" placeholder="B={(p | q); ~u}"></textarea>
        </div>
        <div>
          <label for="beliefGradient" style="margin-top:0;">Gradient values (0..1)</label>
          <textarea id="beliefGradient" placeholder="0.8; 0.7&#10;(empty = 1)"></textarea>
          <div class="field-hint">One value per belief, same order. In Full mode values enable hybrid behavior; in Gradient mode missing values default to 1.</div>
        </div>
      </div>

      <div class="form-grid">
        <div>
          <label for="mode">Belief Mode</label>
          <select id="mode">
            <option value="standard" selected>Standard B</option>
            <option value="full">Full Beliefs (delta)</option>
            <option value="gradient">Gradient Beliefs</option>
          </select>
        </div>
        <div>
          <label for="decompStyle">Decomposition</label>
          <select id="decompStyle">
            <option value="cut_free" selected>Cut-Free Context-Free</option>
            <option value="fractional">Fractional (GS4)</option>
          </select>
        </div>
        <div>
          <label for="dpi">PNG DPI</label>
          <input id="dpi" type="number" min="72" max="1200" step="1" value="320" />
        </div>
      </div>

      <div class="controls">
        <button class="primary" id="go">Generate Decomposition</button>
        <button class="ghost" id="random">Random Sequent</button>
        <button class="ghost" id="compile" disabled>Render PNG Preview</button>
        <button class="ghost" id="download" disabled>Download PNG</button>
        <button class="ghost" id="clear">Clear</button>
      </div>
      <div class="history-row">
        <label for="randomMode">Random Mode</label>
        <select id="randomMode">
          <option value="coherent" selected>Only coherent formulas</option>
          <option value="allow_incoherent">Allow incoherent formulas</option>
        </select>
      </div>
      <div class="history-row">
        <label for="history">History</label>
        <select id="history">
          <option value="">No history yet</option>
        </select>
      </div>
      <div class="legend">
        <div><strong>Legend</strong></div>
        <div>Turnstile forms: <code>Γ |- Δ</code>, <code>Γ ⊢ Δ</code>, <code>\\vdash Δ</code></div>
        <div>Comma in one-sided/right side is disjunction: <code>q, (r &amp; s)</code> = <code>q OR (r AND s)</code></div>
        <div>Comma in left side is conjunction: <code>p, q |- r</code> means <code>(p AND q) |- r</code></div>
        <div>OR: <code>|</code>, <code>∨</code>, <code>\\vee</code>, <code>OR</code></div>
        <div>AND: <code>&amp;</code>, <code>∧</code>, <code>\\wedge</code>, <code>AND</code></div>
        <div>Negation: <code>~</code>, <code>!</code>, <code>¬</code>, <code>\\neg</code>, <code>\\overline{p}</code>, <code>NOT</code></div>
        <div>Precedence: <code>NOT</code> binds strongest, then <code>AND</code>, then <code>OR</code>; <code>u | p &amp; v</code> = <code>u | (p &amp; v)</code></div>
      </div>
      <div id="error" class="error"></div>
      <div id="status" class="ok"></div>
      <div id="validation" class="ok"></div>
      <div id="equivalence" class="warn"></div>
    </section>

    <section class="panel">
      <label>Compiled PNG Preview (Landscape)</label>
      <div class="preview-tools">
        <div class="zoom-group">
          <button class="ghost" id="zoomOut" type="button">-</button>
          <input id="zoom" type="range" min="20" max="200" step="5" value="120" />
          <button class="ghost" id="zoomIn" type="button">+</button>
          <button class="ghost" id="zoomReset" type="button">Reset</button>
          <span id="zoomLabel" class="zoom-label">120%</span>
        </div>
        <div class="preview-hint">Use zoom and scroll to browse the tree preview.</div>
      </div>
      <div id="imgWrap" class="empty">
        <div id="imgCanvas">
          <img id="pngView" alt="Tree preview" />
        </div>
      </div>
    </section>

    <section class="grid-two">
      <div class="panel">
        <label>Visual Tree</label>
        <div class="tree-wrap">
          <div id="tree" class="tree"></div>
        </div>
      </div>
      <div class="panel">
        <label>Decomposition (LaTeX only)</label>
        <pre id="out"></pre>
      </div>
    </section>

    <section class="panel">
      <label>Generated Example (LLM)</label>
      <div class="example-tools">
        <div class="mode-wrap">
          <label for="exampleMode">Example Mode</label>
          <select id="exampleMode">
            <option value="auto" selected>Auto (local model, fallback template)</option>
            <option value="local_only">Local model only</option>
            <option value="template_only">Template only</option>
          </select>
        </div>
        <button class="ghost" id="exampleBtn" type="button">Generate Example</button>
      </div>
      <pre id="exampleOut"></pre>
      <div id="exampleMeta" class="ok"></div>
    </section>
  </main>
  <div id="loading" class="loading">
    <div class="spinner"></div>
    <span id="loadingText">Working...</span>
  </div>

  <script>
    const out = document.getElementById("out");
    const error = document.getElementById("error");
    const status = document.getElementById("status");
    const tree = document.getElementById("tree");
    const sequent = document.getElementById("sequent");
    const belief = document.getElementById("belief");
    const beliefGradient = document.getElementById("beliefGradient");
    const mode = document.getElementById("mode");
    const decompStyle = document.getElementById("decompStyle");
    const dpiInput = document.getElementById("dpi");
    const pngView = document.getElementById("pngView");
    const imgWrap = document.getElementById("imgWrap");
    const zoomInput = document.getElementById("zoom");
    const zoomLabel = document.getElementById("zoomLabel");
    const zoomInBtn = document.getElementById("zoomIn");
    const zoomOutBtn = document.getElementById("zoomOut");
    const zoomResetBtn = document.getElementById("zoomReset");
    const compileBtn = document.getElementById("compile");
    const downloadBtn = document.getElementById("download");
    const goBtn = document.getElementById("go");
    const randomBtn = document.getElementById("random");
    const exampleBtn = document.getElementById("exampleBtn");
    const exampleMode = document.getElementById("exampleMode");
    const exampleOut = document.getElementById("exampleOut");
    const exampleMeta = document.getElementById("exampleMeta");
    const randomMode = document.getElementById("randomMode");
    const clearBtn = document.getElementById("clear");
    const loading = document.getElementById("loading");
    const loadingText = document.getElementById("loadingText");
    const historySelect = document.getElementById("history");
    const validation = document.getElementById("validation");
    const equivalence = document.getElementById("equivalence");

    const runHistory = [];
    let runCounter = 0;
    let currentRunId = null;
    const DEFAULT_ZOOM = 120;
    const MIN_ZOOM = 20;
    const MAX_ZOOM = 200;
    const ZOOM_STEP = 5;
    let zoomPct = DEFAULT_ZOOM;
    let pendingAutoZoom = null;

    function esc(text) {
      return String(text)
        .replaceAll("&", "&amp;")
        .replaceAll("<", "&lt;")
        .replaceAll(">", "&gt;");
    }

    function renderNode(node) {
      const li = document.createElement("li");
      const box = document.createElement("div");
      box.className = "node";
      box.innerHTML = `
        <div class="meta">
          <span class="badge">${esc(node.rule || "")}</span>
          <span>N=${esc(node.n)} M=${esc(node.m)}</span>
        </div>
        <div class="seq">${esc(node.display || "")}</div>
      `;
      li.appendChild(box);

      if (node.children && node.children.length > 0) {
        const ul = document.createElement("ul");
        for (const child of node.children) {
          ul.appendChild(renderNode(child));
        }
        li.appendChild(ul);
      }
      return li;
    }

    function renderTree(rootNode) {
      tree.innerHTML = "";
      if (!rootNode) return;
      const ul = document.createElement("ul");
      ul.appendChild(renderNode(rootNode));
      tree.appendChild(ul);
    }

    function clip(text, maxLen = 52) {
      const t = String(text || "");
      if (t.length <= maxLen) return t;
      return t.slice(0, maxLen - 1) + "…";
    }

    function findRun(id) {
      return runHistory.find((r) => r.id === id) || null;
    }

    function clampZoom(value) {
      return Math.max(MIN_ZOOM, Math.min(MAX_ZOOM, value));
    }

    function applyZoomToImage() {
      if (!pngView.getAttribute("src") || !pngView.naturalWidth || !pngView.naturalHeight) {
        pngView.style.width = "";
        pngView.style.height = "";
        return;
      }
      const ratio = zoomPct / 100;
      pngView.style.width = `${Math.max(1, Math.round(pngView.naturalWidth * ratio))}px`;
      pngView.style.height = `${Math.max(1, Math.round(pngView.naturalHeight * ratio))}px`;
    }

    function centerPreviewScroll() {
      const maxLeft = Math.max(0, imgWrap.scrollWidth - imgWrap.clientWidth);
      const maxTop = Math.max(0, imgWrap.scrollHeight - imgWrap.clientHeight);
      imgWrap.scrollLeft = Math.round(maxLeft / 2);
      imgWrap.scrollTop = Math.round(maxTop / 2);
    }

    function countBinaryInfCommands(decomposition) {
      const text = String(decomposition || "");
      const matches = text.match(/\\\\BinaryInfC(?![A-Za-z])/g);
      return matches ? matches.length : 0;
    }

    function zoomForBinaryCount(binaryCount) {
      if (binaryCount <= 0) return 120;
      if (binaryCount === 1) return 110;
      if (binaryCount === 2) return 80;
      if (binaryCount === 3) return 55;
      return clampZoom(55 - 10 * (binaryCount - 3));
    }

    function recommendedZoomForDecomposition(decomposition) {
      const binaryCount = countBinaryInfCommands(decomposition);
      return zoomForBinaryCount(binaryCount);
    }

    function setZoom(value, persistRun = true) {
      const parsed = Number(value);
      if (!Number.isFinite(parsed)) return;
      zoomPct = clampZoom(Math.round(parsed / ZOOM_STEP) * ZOOM_STEP);
      zoomInput.value = String(zoomPct);
      zoomLabel.textContent = `${zoomPct}%`;
      applyZoomToImage();
      if (persistRun && currentRunId) {
        const run = findRun(currentRunId);
        if (run) {
          run.zoom = zoomPct;
        }
      }
    }

    function renderHistorySelect() {
      const previousValue = historySelect.value;
      historySelect.innerHTML = "";
      if (runHistory.length === 0) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No history yet";
        historySelect.appendChild(opt);
        return;
      }

      const placeholder = document.createElement("option");
      placeholder.value = "";
      placeholder.textContent = "Select a past run";
      historySelect.appendChild(placeholder);

      for (const run of runHistory) {
        const opt = document.createElement("option");
        opt.value = String(run.id);
        opt.textContent = `${run.timestamp} | ${run.mode} | ${clip(run.sequent)}`;
        historySelect.appendChild(opt);
      }

      if (previousValue && findRun(Number(previousValue))) {
        historySelect.value = previousValue;
      } else if (currentRunId && findRun(currentRunId)) {
        historySelect.value = String(currentRunId);
      }
    }

    function setValidationInfo(v) {
      if (!v) {
        validation.textContent = "";
        validation.className = "ok";
        return;
      }
      validation.textContent = `Validation: ${v.ok ? "OK" : "FAILED"} — ${v.message}`;
      validation.className = v.ok ? "ok" : "warn";
    }

    function setEquivalenceInfo(info) {
      if (!info || !info.changed) {
        equivalence.textContent = "";
        equivalence.className = "warn";
        return;
      }
      const chain = String(info.ascii_chain || "");
      const note = String(info.note || "");
      equivalence.textContent = note
        ? `Equivalence chain: ${chain} (${note})`
        : `Equivalence chain: ${chain}`;
      equivalence.className = "warn";
    }

    function sequentKindLabel(kind) {
      return kind === "two-sided" ? "two-sided (right commas = OR)" : "one-sided";
    }

    function updateGradientInputState() {
      const active = mode.value === "gradient" || mode.value === "full";
      beliefGradient.disabled = !active;
      if (!active) {
        beliefGradient.placeholder = "Used in Full (hybrid) or Gradient mode";
        return;
      }
      if (mode.value === "full") {
        beliefGradient.placeholder = "Optional hybrid mode: 0.8; 0.6\\n(values use v-δ, rest stay 1-δ)";
      } else {
        beliefGradient.placeholder = "0.8; 0.7\\n(empty = 1)";
      }
    }

    function setLoading(active, text = "Working...") {
      loadingText.textContent = text;
      loading.classList.toggle("show", active);
      goBtn.disabled = active;
      randomBtn.disabled = active;
      exampleBtn.disabled = active;
      compileBtn.disabled = active || !out.textContent.trim();
      downloadBtn.disabled = active || !pngView.getAttribute("src");
      clearBtn.disabled = active;
    }

    async function generateExample() {
      error.textContent = "";
      const seq = String(sequent.value || "").trim();
      if (!seq) {
        error.textContent = "Please provide a sequent first.";
        return;
      }
      setLoading(true, "Generating example...");
      try {
        const res = await fetch("/api/example", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            sequent: seq,
            belief_set: belief.value || "",
            example_mode: exampleMode.value || "auto",
          }),
        });
        const data = await res.json();
        if (!data.ok) {
          error.textContent = data.error || "Unable to generate example.";
          return;
        }
        exampleOut.textContent = String(data.example || "");
        exampleMeta.textContent = `Engine: ${String(data.engine || "unknown")} | Model: ${String(data.model || "n/a")}`;
        if (currentRunId) {
          const run = findRun(currentRunId);
          if (run) {
            run.example = exampleOut.textContent;
            run.example_meta = exampleMeta.textContent;
            run.example_mode = String(exampleMode.value || "auto");
          }
        }
      } catch (e) {
        error.textContent = String(e);
      } finally {
        setLoading(false);
      }
    }

    async function compilePreview(decomposition, auto = false) {
      error.textContent = "";
      status.textContent = "";
      if (!decomposition) {
        error.textContent = "Generate decomposition first.";
        return false;
      }
      const dpi = parseInt(dpiInput.value || "320", 10);
      if (!Number.isFinite(dpi) || dpi < 72 || dpi > 1200) {
        error.textContent = "DPI must be an integer between 72 and 1200.";
        return false;
      }
      setLoading(true, "Rendering PNG preview...");
      try {
        const res = await fetch("/api/compile", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ decomposition, dpi }),
        });
        const data = await res.json();
        if (!data.ok) {
          error.textContent = data.error || "Compilation failed";
          return false;
        }
        const src = `/api/png/${encodeURIComponent(data.png_id)}?t=${Date.now()}`;
        pngView.style.width = "";
        pngView.style.height = "";
        pendingAutoZoom = recommendedZoomForDecomposition(decomposition);
        pngView.src = src;
        imgWrap.classList.remove("empty");
        downloadBtn.disabled = false;
        if (currentRunId) {
          const run = findRun(currentRunId);
          if (run) {
            run.pngSrc = src;
            run.dpi = dpi;
            run.zoom = zoomPct;
          }
        }
        status.textContent = auto ? "PNG preview generated automatically." : "PNG preview rendered.";
        return true;
      } catch (e) {
        error.textContent = String(e);
        return false;
      } finally {
        setLoading(false);
      }
    }

    goBtn.addEventListener("click", async () => {
      error.textContent = "";
      status.textContent = "";
      out.textContent = "";
      tree.innerHTML = "";
      pngView.removeAttribute("src");
      pngView.style.width = "";
      pngView.style.height = "";
      imgWrap.classList.add("empty");
      setZoom(DEFAULT_ZOOM, false);
      compileBtn.disabled = true;
      downloadBtn.disabled = true;
      setValidationInfo(null);
      setEquivalenceInfo(null);
      const payload = {
        sequent: sequent.value,
        belief_set: belief.value,
        belief_gradients: beliefGradient.value,
        belief_mode: mode.value,
        decomposition_style: decompStyle.value,
      };
      setLoading(true, "Generating decomposition...");
      try {
        const res = await fetch("/api/decompose", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        });
        const data = await res.json();
        if (!data.ok) {
          error.textContent = data.error || "Unknown error";
          return;
        }
        out.textContent = data.decomposition;
        renderTree(data.tree);
        setValidationInfo(data.validation || null);
        setEquivalenceInfo(data.equivalence || null);
        exampleOut.textContent = "";
        exampleMeta.textContent = "";
        const kindLabel = sequentKindLabel(data.sequent_kind || "one-sided");
        status.textContent = `Decomposition generated (${kindLabel}). Building PNG preview...`;

        const run = {
          id: ++runCounter,
          timestamp: new Date().toLocaleTimeString(),
          sequent: payload.sequent,
          belief_set: payload.belief_set,
          belief_gradients: payload.belief_gradients,
          random_mode: randomMode.value,
          mode: payload.belief_mode,
          decomposition_style: payload.decomposition_style,
          dpi: parseInt(dpiInput.value || "320", 10),
          decomposition: data.decomposition,
          tree: data.tree,
          sequent_kind: data.sequent_kind || "one-sided",
          validation: data.validation || null,
          equivalence: data.equivalence || null,
          example: "",
          example_meta: "",
          example_mode: String(exampleMode.value || "auto"),
          pngSrc: "",
          zoom: zoomPct,
        };
        runHistory.unshift(run);
        if (runHistory.length > 40) {
          runHistory.pop();
        }
        currentRunId = run.id;
        renderHistorySelect();
      } catch (e) {
        error.textContent = String(e);
        return;
      } finally {
        setLoading(false);
      }

      compileBtn.disabled = false;
      await compilePreview(out.textContent.trim(), true);
    });

    compileBtn.addEventListener("click", async () => {
      await compilePreview(out.textContent.trim(), false);
    });

    randomBtn.addEventListener("click", async () => {
      error.textContent = "";
      status.textContent = "";
      setLoading(true, "Generating random sequent...");
      try {
        const res = await fetch("/api/random-sequent", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            allow_incoherent: randomMode.value === "allow_incoherent",
          }),
        });
        const data = await res.json();
        if (!data.ok) {
          error.textContent = data.error || "Unable to generate a random sequent.";
          return;
        }
        sequent.value = String(data.sequent || "");
        out.textContent = "";
        tree.innerHTML = "";
        pngView.removeAttribute("src");
        pngView.style.width = "";
        pngView.style.height = "";
        imgWrap.classList.add("empty");
        setZoom(DEFAULT_ZOOM, false);
        compileBtn.disabled = true;
        downloadBtn.disabled = true;
        setValidationInfo(null);
        exampleOut.textContent = "";
        exampleMeta.textContent = "";
        status.textContent = randomMode.value === "allow_incoherent"
          ? "Random sequent ready (incoherent formulas allowed). Click Generate Decomposition."
          : "Random sequent ready (coherent only). Click Generate Decomposition.";
      } catch (e) {
        error.textContent = String(e);
      } finally {
        setLoading(false);
      }
    });

    exampleBtn.addEventListener("click", async () => {
      await generateExample();
    });

    downloadBtn.addEventListener("click", () => {
      const src = pngView.getAttribute("src");
      if (!src) {
        error.textContent = "No PNG preview available.";
        return;
      }
      const a = document.createElement("a");
      a.href = src;
      a.download = `fractional_tree_${Date.now()}.png`;
      document.body.appendChild(a);
      a.click();
      a.remove();
    });

    historySelect.addEventListener("change", () => {
      const selected = historySelect.value;
      if (!selected) return;
      const run = findRun(Number(selected));
      if (!run) return;

      currentRunId = run.id;
      sequent.value = run.sequent;
      belief.value = run.belief_set;
      beliefGradient.value = run.belief_gradients || "";
      randomMode.value = run.random_mode || "coherent";
      exampleMode.value = run.example_mode || "auto";
      mode.value = run.mode;
      decompStyle.value = run.decomposition_style || "cut_free";
      dpiInput.value = String(run.dpi || 320);
      updateGradientInputState();
      out.textContent = run.decomposition;
      renderTree(run.tree);
      setValidationInfo(run.validation || null);
      setEquivalenceInfo(run.equivalence || null);
      exampleOut.textContent = run.example || "";
      exampleMeta.textContent = run.example_meta || "";
      error.textContent = "";
      status.textContent = `Loaded from history (${sequentKindLabel(run.sequent_kind || "one-sided")}).`;
      compileBtn.disabled = false;
      if (run.pngSrc) {
        setZoom(run.zoom || DEFAULT_ZOOM);
        pngView.style.width = "";
        pngView.style.height = "";
        pendingAutoZoom = null;
        pngView.src = run.pngSrc;
        imgWrap.classList.remove("empty");
        downloadBtn.disabled = false;
      } else {
        pngView.removeAttribute("src");
        pngView.style.width = "";
        pngView.style.height = "";
        imgWrap.classList.add("empty");
        downloadBtn.disabled = true;
      }
    });

    mode.addEventListener("change", () => {
      updateGradientInputState();
    });

    zoomInput.addEventListener("input", () => {
      setZoom(zoomInput.value);
    });

    zoomInBtn.addEventListener("click", () => {
      setZoom(zoomPct + ZOOM_STEP);
    });

    zoomOutBtn.addEventListener("click", () => {
      setZoom(zoomPct - ZOOM_STEP);
    });

    zoomResetBtn.addEventListener("click", () => {
      setZoom(DEFAULT_ZOOM);
    });

    pngView.addEventListener("load", () => {
      if (pendingAutoZoom !== null) {
        const target = pendingAutoZoom;
        pendingAutoZoom = null;
        setZoom(target);
      } else {
        applyZoomToImage();
      }
      centerPreviewScroll();
    });

    clearBtn.addEventListener("click", () => {
      sequent.value = "";
      belief.value = "";
      beliefGradient.value = "";
      mode.value = "standard";
      decompStyle.value = "cut_free";
      dpiInput.value = "320";
      randomMode.value = "coherent";
      exampleMode.value = "auto";
      out.textContent = "";
      exampleOut.textContent = "";
      exampleMeta.textContent = "";
      tree.innerHTML = "";
      error.textContent = "";
      status.textContent = "";
      pngView.removeAttribute("src");
      pngView.style.width = "";
      pngView.style.height = "";
      imgWrap.classList.add("empty");
      setZoom(DEFAULT_ZOOM, false);
      compileBtn.disabled = true;
      downloadBtn.disabled = true;
      setValidationInfo(null);
      setEquivalenceInfo(null);
      updateGradientInputState();
    });

    updateGradientInputState();
    setZoom(DEFAULT_ZOOM, false);
  </script>
</body>
</html>
"""


class _Handler(BaseHTTPRequestHandler):
    _png_store: dict[str, bytes] = {}

    def _send_json(self, status: int, data: dict[str, object]) -> None:
        payload = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, status: int, html: str) -> None:
        payload = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send_html(200, _HTML)
            return
        if self.path.startswith("/api/png/"):
            png_id = unquote(self.path.rsplit("/", 1)[-1].split("?", 1)[0])
            payload = self._png_store.get(png_id)
            if payload is None:
                self.send_error(404, "PNG not found")
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Disposition", "inline; filename=decomposition_preview.png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404, "Not found")

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in {"/api/decompose", "/api/compile", "/api/random-sequent", "/api/example"}:
            self.send_error(404, "Not found")
            return

        content_len = int(self.headers.get("Content-Length", "0"))
        raw = self.rfile.read(content_len)
        try:
            body = json.loads(raw.decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"ok": False, "error": f"Invalid JSON: {exc}"})
            return

        if self.path == "/api/decompose":
            try:
                sequent = str(body.get("sequent", ""))
                belief_set = str(body.get("belief_set", ""))
                belief_gradients = str(body.get("belief_gradients", ""))
                belief_mode = str(body.get("belief_mode", "standard"))
                decomposition_style = str(body.get("decomposition_style", "cut_free"))
                bundle = generate_decomposition_bundle(
                    sequent,
                    belief_set,
                    belief_mode=belief_mode,
                    decomposition_style=decomposition_style,
                    belief_gradients_text=belief_gradients,
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"ok": False, "error": str(exc)})
                return

            self._send_json(
                200,
                {
                    "ok": True,
                    "decomposition": bundle["decomposition"],
                    "tree": bundle["tree"],
                    "sequent_kind": bundle.get("sequent_kind", "one-sided"),
                    "validation": bundle["validation"],
                    "equivalence": bundle.get("equivalence"),
                },
            )
            return

        if self.path == "/api/random-sequent":
            try:
                seed_value = body.get("seed")
                rng: random.Random | None = None
                if seed_value is not None:
                    try:
                        rng = random.Random(int(seed_value))
                    except Exception as exc:  # noqa: BLE001
                        raise ValueError("Seed must be an integer.") from exc
                allow_incoherent_raw = body.get("allow_incoherent", False)
                if isinstance(allow_incoherent_raw, bool):
                    allow_incoherent = allow_incoherent_raw
                elif isinstance(allow_incoherent_raw, (int, float)):
                    allow_incoherent = bool(allow_incoherent_raw)
                elif isinstance(allow_incoherent_raw, str):
                    allow_incoherent = allow_incoherent_raw.strip().lower() in {
                        "1",
                        "true",
                        "yes",
                        "on",
                    }
                else:
                    raise ValueError("allow_incoherent must be boolean-like.")
                sequent = generate_random_sequent(
                    rng=rng,
                    allow_incoherent=allow_incoherent,
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"ok": False, "error": str(exc)})
                return

            self._send_json(200, {"ok": True, "sequent": sequent})
            return

        if self.path == "/api/example":
            try:
                sequent = str(body.get("sequent", ""))
                belief_set = str(body.get("belief_set", ""))
                model = str(body.get("model", "")).strip() or None
                example_mode = str(body.get("example_mode", "auto")).strip() or "auto"
                result = generate_example_from_sequent(
                    sequent,
                    belief_set_text=belief_set,
                    model=model,
                    prefer_local_model=True,
                    generation_mode=example_mode,
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {"ok": False, "error": str(exc)})
                return

            self._send_json(
                200,
                {
                    "ok": True,
                    "example": result.text,
                    "engine": result.engine,
                    "model": result.model,
                },
            )
            return

        try:
            decomposition = str(body.get("decomposition", "")).strip()
            if not decomposition:
                raise ValueError("Missing decomposition.")
            dpi_raw = body.get("dpi", 320)
            try:
                dpi_value = int(dpi_raw)
            except Exception as exc:  # noqa: BLE001
                raise ValueError("DPI must be an integer.") from exc
            png_bytes = compile_decomposition_png(decomposition, dpi=dpi_value)
            png_id = secrets.token_urlsafe(12)
            self._png_store[png_id] = png_bytes
            # Keep memory bounded.
            if len(self._png_store) > 30:
                for key in list(self._png_store.keys())[:10]:
                    self._png_store.pop(key, None)
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {"ok": False, "error": str(exc)})
            return

        self._send_json(200, {"ok": True, "png_id": png_id})


def launch_web_gui() -> int:
    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    except OSError as exc:
        raise RuntimeError(
            "Unable to start local Web UI server (cannot bind localhost socket)."
        ) from exc

    host, port = server.server_address
    url = f"http://{host}:{port}"
    print(f"PID: {os.getpid()}")
    print(f"Web UI available at: {url}")
    try:
        webbrowser.open(url)
    except Exception:
        pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping Web UI...")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(launch_web_gui())
