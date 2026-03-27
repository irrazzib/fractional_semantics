"""Web application with MathJax-based proof tree rendering (no pdflatex required)."""
from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import secrets
import webbrowser

from .decomposition_service import generate_decomposition_bundle
from .example_generator import generate_example_from_sequent
from .web_gui import (
    generate_random_coherent_sequent,
    generate_random_incoherent_sequent,
    generate_random_sequent,
)


# ---------------------------------------------------------------------------
# HTML template (raw string – backslashes are literal)
# ---------------------------------------------------------------------------
_HTML = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Fractional Semantics</title>

  <!-- MathJax configuration MUST precede the script tag -->
  <script>
  MathJax = {
    tex: {
      packages: {'[+]': ['bussproofs']},
      macros: {
        // \sststile{n}{m}  —  closest MathJax approximation of the LaTeX
        // turnstile package.  The real package renders n and m in \scriptstyle
        // and overlays them on the horizontal bar via \raisebox.  MathJax
        // lacks \relbar / \joinrel (used to build long arrows from parts), so
        // we place \scriptstyle labels above/below \vdash with \overset /
        // \underset instead — the only difference from the LaTeX original is
        // the labels reference the full glyph height rather than just the bar.
        // Argument order confirmed from turnstile.sty source:
        //   raisedown (negative) → \first = #1 goes BELOW the bar
        //   raiseup   (positive) → \second = #2 goes ABOVE the bar
        // i.e. \sststile{n}{m}: n below, m above — opposite of intuition.
        sststile: [
          '\\mathrel{' +
            '\\underset{\\scriptstyle #1}{' +
              '\\overset{\\scriptstyle #2}{' +
                '\\vdash' +
              '}' +
            '}' +
          '}',
          2
        ],
        llbracket: '{\\unicode{x27E6}}',
        rrbracket: '{\\unicode{x27E7}}'
      }
    },
    loader: {load: ['[tex]/bussproofs']},
    options: {
      // Do not auto-process content inside <pre> (raw LaTeX display)
      skipHtmlTags: ['script', 'noscript', 'style', 'textarea', 'pre', 'code']
    }
  };
  </script>
  <script id="MathJax-script" async
    src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>

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
    .app { max-width: 1200px; margin: 0 auto; display: grid; gap: 16px; }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 14px;
      min-width: 0;
    }
    h1 { margin: 0 0 8px; font-size: 1.35rem; letter-spacing: 0.01em; }
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
    .field-hint { margin-top: 6px; color: var(--muted); font-size: 0.85rem; }
    .form-grid {
      display: grid;
      grid-template-columns: 1fr 220px;
      gap: 12px;
      margin-top: 10px;
      align-items: end;
    }
    @media (max-width: 700px) {
      .belief-grid { grid-template-columns: 1fr; }
      .form-grid { grid-template-columns: 1fr; }
    }
    .controls { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 10px; }
    .history-row { display: grid; grid-template-columns: 1fr; gap: 6px; margin-top: 10px; }
    .example-tools {
      display: flex; gap: 10px; flex-wrap: wrap;
      align-items: end; margin-bottom: 10px;
    }
    .example-tools .mode-wrap { min-width: 260px; flex: 1; }
    button {
      border: none; border-radius: 8px; padding: 10px 14px;
      font: inherit; font-weight: 700; cursor: pointer;
    }
    .primary { background: var(--accent); color: #fff; }
    .ghost { background: #ece6dd; color: var(--ink); }
    pre {
      margin: 0; white-space: pre-wrap; word-break: break-word;
      line-height: 1.35;
      font-family: "IBM Plex Mono", "Menlo", monospace;
      min-height: 80px; font-size: 0.85rem;
    }
    #exampleOut { min-height: 140px; }
    .grid-two {
      display: grid; gap: 16px;
      grid-template-columns: 1.2fr 1fr;
      align-items: start;
    }
    .grid-two > * { min-width: 0; }
    @media (max-width: 900px) { .grid-two { grid-template-columns: 1fr; } }
    .legend {
      display: grid; gap: 6px; margin-top: 10px;
      color: var(--muted); font-size: 0.92rem;
    }
    .legend code {
      background: #ece6dd; padding: 2px 6px; border-radius: 6px;
    }

    /* ── Proof tree (MathJax) ── */
    .math-tree-outer {
      border: 1px solid var(--line);
      border-radius: 8px;
      background: #fff;
      min-height: 220px;
      max-height: 65vh;
      overflow: auto;
      position: relative;
    }
    .math-tree-outer.empty::before {
      content: "Proof tree will appear here";
      color: var(--muted);
      font-size: 0.95rem;
      position: absolute;
      top: 50%; left: 50%;
      transform: translate(-50%, -50%);
      pointer-events: none;
      white-space: nowrap;
    }
    #mathTree {
      padding: 16px;
      min-width: max-content;
      transform-origin: top left;
    }
    .zoom-row {
      display: flex; gap: 8px; align-items: center;
      flex-wrap: wrap; margin-bottom: 8px;
    }
    .zoom-row input[type="range"] { width: 150px; padding: 0; }
    .zoom-lbl {
      min-width: 48px; text-align: right;
      font-size: 0.9rem; color: var(--muted);
      font-family: "IBM Plex Mono", "Menlo", monospace;
    }

    /* ── Proof step navigator ── */
    .step-row {
      display: flex; gap: 8px; align-items: center;
      flex-wrap: wrap; margin-bottom: 8px;
    }
    .step-row input[type="range"] { flex: 1; max-width: 260px; padding: 0; }
    .step-lbl {
      min-width: 110px; font-size: 0.9rem; color: var(--muted);
      font-family: "IBM Plex Mono", "Menlo", monospace;
    }

    /* ── Values (MathJax) ── */
    #statsOutput {
      min-height: 2.5rem;
      font-size: 1.05rem;
    }
    #statsOutput .MathJax { font-size: 1.1em !important; }

    /* ── Visual tree ── */
    .tree-wrap {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 8px;
      padding: 10px;
      overflow: auto;
      min-height: 220px;
      max-height: 55vh;
    }
    .tree { min-width: max-content; font-family: "IBM Plex Mono", "Menlo", monospace; }
    .tree ul {
      position: relative; padding-top: 22px;
      margin: 0; display: flex; justify-content: center;
    }
    .tree li {
      list-style: none; text-align: center;
      position: relative; padding: 20px 8px 0 8px;
    }
    .tree li::before, .tree li::after {
      content: ""; position: absolute; top: 0;
      right: 50%; border-top: 1px solid #b8aa98;
      width: 50%; height: 20px;
    }
    .tree li::after { right: auto; left: 50%; border-left: 1px solid #b8aa98; }
    .tree li:only-child::before, .tree li:only-child::after { display: none; }
    .tree li:only-child { padding-top: 0; }
    .tree li:first-child::before, .tree li:last-child::after { border: 0; }
    .tree li:last-child::before {
      border-right: 1px solid #b8aa98; border-radius: 0 6px 0 0;
    }
    .tree li:first-child::after { border-radius: 6px 0 0 0; }
    .tree ul ul::before {
      content: ""; position: absolute; top: 0; left: 50%;
      border-left: 1px solid #b8aa98; width: 0; height: 22px;
    }
    .node {
      display: inline-block; min-width: 220px; max-width: 480px;
      padding: 8px 10px; border: 1px solid #bcae9c;
      border-radius: 8px; background: #fffdfa;
      text-align: left; box-shadow: 0 1px 1px rgba(0,0,0,0.03);
    }
    .node .meta {
      display: flex; gap: 8px; align-items: center;
      margin-bottom: 5px; font-size: 0.82rem; color: var(--muted);
    }
    .node .badge {
      padding: 2px 8px; border-radius: 999px;
      background: #ece6dd; color: var(--ink); font-weight: 700;
    }
    .node .seq { font-size: 0.88rem; color: var(--ink); word-break: break-word; }

    /* ── Raw LaTeX (collapsible) ── */
    details > summary {
      cursor: pointer; font-weight: 700;
      font-size: 0.92rem; user-select: none;
      margin-bottom: 6px;
    }

    /* ── Messages ── */
    .error { color: #a32222; margin-top: 8px; }
    .ok { color: #1c7c6b; margin-top: 8px; }
    .warn { color: #ab5f00; margin-top: 8px; }

    /* ── Loading spinner ── */
    .loading {
      position: fixed; right: 20px; bottom: 20px;
      display: none; align-items: center; gap: 10px;
      background: #182026; color: #fff;
      border-radius: 999px; padding: 9px 14px;
      box-shadow: 0 6px 22px rgba(0,0,0,0.25);
      z-index: 9999; font-size: 0.92rem;
    }
    .loading.show { display: inline-flex; }
    .spinner {
      width: 16px; height: 16px;
      border: 2px solid rgba(255,255,255,0.25);
      border-top-color: #fff;
      border-radius: 50%;
      animation: spin 0.8s linear infinite;
    }
    @keyframes spin { to { transform: rotate(360deg); } }
  </style>
</head>
<body>
<main class="app">

  <!-- ── Header ── -->
  <section class="panel">
    <h1>Fractional Semantics Decomposition</h1>
    <p>Proof trees rendered in-browser with MathJax — no LaTeX installation required.</p>
  </section>

  <!-- ── Input ── -->
  <section class="panel">
    <label for="sequent">Sequent</label>
    <input id="sequent"
      placeholder="(p | q) & (s | ~s) & (~r & ~t)  or  p \vdash q, (r & s)" />

    <div class="belief-grid">
      <div>
        <label for="belief" style="margin-top:0;">Belief set B (optional)</label>
        <textarea id="belief" placeholder="B={(p | q); ~u}"></textarea>
      </div>
      <div>
        <label for="beliefGradient" style="margin-top:0;">Gradient values (0..1)</label>
        <textarea id="beliefGradient" placeholder="0.8; 0.7&#10;(empty = 1)"></textarea>
        <div class="field-hint">One value per belief, same order. In Full mode values
          enable hybrid behaviour; in Gradient mode missing values default to 1.</div>
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
    </div>

    <div class="controls">
      <button class="primary" id="go">Generate Decomposition</button>
      <button class="ghost" id="random">Random Sequent</button>
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
      <select id="history"><option value="">No history yet</option></select>
    </div>

    <div class="legend">
      <div><strong>Legend</strong></div>
      <div>Turnstile forms: <code>Γ |- Δ</code>, <code>Γ ⊢ Δ</code>, <code>\vdash Δ</code></div>
      <div>Comma in one-sided/right side is disjunction: <code>q, (r &amp; s)</code> = <code>q OR (r AND s)</code></div>
      <div>OR: <code>|</code>, <code>∨</code>, <code>\vee</code>, <code>OR</code> &nbsp;|&nbsp;
           AND: <code>&amp;</code>, <code>∧</code>, <code>\wedge</code>, <code>AND</code> &nbsp;|&nbsp;
           NOT: <code>~</code>, <code>!</code>, <code>¬</code>, <code>\neg</code></div>
    </div>
    <div id="error" class="error"></div>
    <div id="status" class="ok"></div>
    <div id="validation" class="ok"></div>
    <div id="equivalence" class="warn"></div>
  </section>

  <!-- ── Proof Tree (MathJax) ── -->
  <section class="panel">
    <label>Proof Tree</label>
    <div class="zoom-row">
      <button class="ghost" id="zoomOut" type="button">−</button>
      <input id="zoom" type="range" min="30" max="200" step="5" value="100" />
      <button class="ghost" id="zoomIn" type="button">+</button>
      <button class="ghost" id="zoomReset" type="button">Reset</button>
      <span id="zoomLbl" class="zoom-lbl">100%</span>
    </div>
    <div id="stepRow" class="step-row" style="display:none;">
      <span style="font-size:0.85rem;font-weight:700;color:var(--muted);">Step</span>
      <button class="ghost" id="stepPrev" type="button" style="padding:6px 10px;">◀</button>
      <input id="depthSlider" type="range" min="0" max="1" step="1" value="1" />
      <button class="ghost" id="stepNext" type="button" style="padding:6px 10px;">▶</button>
      <span id="depthLbl" class="step-lbl">Full tree</span>
    </div>
    <div id="mathTreeOuter" class="math-tree-outer empty">
      <div id="mathTree"></div>
    </div>
  </section>

  <!-- ── Fractional Values ── -->
  <section class="panel">
    <label>Fractional Semantics Value</label>
    <div id="statsOutput"></div>
    <div id="leavesInfo" style="margin-top:8px;font-size:0.9rem;color:var(--muted);"></div>
  </section>

  <!-- ── Visual tree + Raw LaTeX ── -->
  <section class="grid-two">
    <div class="panel">
      <label>Visual Tree (ASCII)</label>
      <div class="tree-wrap">
        <div id="tree" class="tree"></div>
      </div>
    </div>
    <div class="panel">
      <details>
        <summary>Raw LaTeX Source</summary>
        <pre id="out"></pre>
      </details>
    </div>
  </section>

  <!-- ── Example ── -->
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
  <span id="loadingText">Working…</span>
</div>

<script>
// ── DOM refs ──
const out          = document.getElementById('out');
const error        = document.getElementById('error');
const status       = document.getElementById('status');
const tree         = document.getElementById('tree');
const mathTree     = document.getElementById('mathTree');
const mathTreeOuter= document.getElementById('mathTreeOuter');
const statsOutput  = document.getElementById('statsOutput');
const leavesInfo   = document.getElementById('leavesInfo');
const sequentEl    = document.getElementById('sequent');
const beliefEl     = document.getElementById('belief');
const beliefGradEl = document.getElementById('beliefGradient');
const modeEl       = document.getElementById('mode');
const decompStyleEl= document.getElementById('decompStyle');
const historySelect= document.getElementById('history');
const validation   = document.getElementById('validation');
const equivalence  = document.getElementById('equivalence');
const randomMode   = document.getElementById('randomMode');
const zoomInput    = document.getElementById('zoom');
const zoomLbl      = document.getElementById('zoomLbl');
const stepRow      = document.getElementById('stepRow');
const depthSlider  = document.getElementById('depthSlider');
const depthLbl     = document.getElementById('depthLbl');
const loading      = document.getElementById('loading');
const loadingText  = document.getElementById('loadingText');
const goBtn        = document.getElementById('go');
const randomBtn    = document.getElementById('random');
const exampleBtn   = document.getElementById('exampleBtn');
const exampleMode  = document.getElementById('exampleMode');
const exampleOut   = document.getElementById('exampleOut');
const exampleMeta  = document.getElementById('exampleMeta');
const clearBtn     = document.getElementById('clear');

// ── State ──
const runHistory = [];
let runCounter   = 0;
let currentRunId = null;
let treeScale    = 1.0;
// Slider state
let sliderTree      = null;   // current tree JSON from API
let sliderMaxDepth  = 0;      // depth of full tree
let sliderDepth     = 0;      // currently displayed depth
const SCALE_STEP = 0.05;
const SCALE_MIN  = 0.30;
const SCALE_MAX  = 2.00;

// ── Utilities ──
function esc(text) {
  return String(text)
    .replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
}

function clip(text, maxLen = 52) {
  const t = String(text || '');
  return t.length <= maxLen ? t : t.slice(0, maxLen - 1) + '…';
}

function findRun(id) { return runHistory.find(r => r.id === id) || null; }

function setLoading(active, text = 'Working…') {
  loadingText.textContent = text;
  loading.classList.toggle('show', active);
  goBtn.disabled    = active;
  randomBtn.disabled = active;
  exampleBtn.disabled = active;
  clearBtn.disabled  = active;
}

// ── Zoom (CSS transform on tree content) ──
function applyScale(scale) {
  treeScale = Math.max(SCALE_MIN, Math.min(SCALE_MAX, scale));
  const pct = Math.round(treeScale * 100);
  zoomInput.value = String(pct);
  zoomLbl.textContent = pct + '%';
  mathTree.style.transform = `scale(${treeScale})`;
  mathTree.style.transformOrigin = 'top left';
  // Adjust outer container height so scrollbar reflects scaled content
  const inner = mathTree.getBoundingClientRect();
  mathTreeOuter.style.minHeight = '';  // let CSS max-height govern
}

document.getElementById('zoomOut').addEventListener('click', () => applyScale(treeScale - SCALE_STEP));
document.getElementById('zoomIn').addEventListener('click',  () => applyScale(treeScale + SCALE_STEP));
document.getElementById('zoomReset').addEventListener('click',() => applyScale(1.0));
zoomInput.addEventListener('input', () => applyScale(Number(zoomInput.value) / 100));

// ── Proof-step slider: rebuild bussproofs LaTeX from tree JSON at any depth ──

/** Compute the maximum depth of a tree node. */
function treeDepth(node) {
  if (!node || !node.children || node.children.length === 0) return 0;
  return 1 + Math.max(...node.children.map(treeDepth));
}

/**
 * Convert ASCII sequent (uses ~, &, |) coming from sequent_to_ascii_grouped
 * into LaTeX (uses \neg, \wedge, \vee).
 */
function asciiToLatex(str) {
  return str
    .replace(/~/g,          '\\neg ')
    .replace(/\s*&\s*/g,    ' \\wedge ')
    .replace(/\s*\|\s*/g,   ' \\vee ');
}

/**
 * Parse a display string of the form
 *   "sststile{n}{m}[_B] |- sequent"
 * into its components.  Uses brace-counting so nested braces in m
 * (e.g. delta symbols) are handled correctly.
 */
function parseSststileDisplay(display) {
  if (!display || !display.startsWith('sststile')) return null;

  function extractBraced(pos) {
    let depth = 0, start = pos + 1, end = -1;
    for (let j = pos; j < display.length; j++) {
      if (display[j] === '{') depth++;
      else if (display[j] === '}') { depth--; if (!depth) { end = j; break; } }
    }
    return end >= 0 ? { content: display.slice(start, end), next: end + 1 } : null;
  }

  const i1 = display.indexOf('{');
  if (i1 < 0) return null;
  const r1 = extractBraced(i1);
  if (!r1) return null;

  const i2 = display.indexOf('{', r1.next);
  if (i2 < 0) return null;
  const r2 = extractBraced(i2);
  if (!r2) return null;

  const rest  = display.slice(r2.next);
  const bm    = rest.match(/^(_B)?\s*\|-\s*([\s\S]*)$/);
  if (!bm) return null;
  return { n: r1.content, m: r2.content, withB: !!bm[1], seq: bm[2].trim() };
}

/** Convert a tree-node display string into a LaTeX sequent for \UnaryInfC / \BinaryInfC. */
function displayToLatex(display) {
  const p = parseSststileDisplay(display);
  if (!p) return display;
  const suffix = p.withB ? '_{\\mathbb{B}}' : '';
  return `\\sststile{\\mathrm{${p.n}}}{${p.m}}${suffix}\\; ${asciiToLatex(p.seq)}`;
}

/** Map a tree-node rule label to the LaTeX \RightLabel content. */
function ruleToRightLabel(rule) {
  const map = {
    '(∨)':              '(\\vee)',
    '(∧)':              '(\\wedge)',
    '(ax.)':            '(ax.)',
    '(b)':              '(b)',
    '(overline{ax.})':  '(\\overline{ax.})',
  };
  return `\\RightLabel{$\\scriptstyle ${map[rule] || rule}$}`;
}

/**
 * Recursively build bussproofs LaTeX for `node` up to `maxDepth`.
 * Nodes at the frontier (not yet expanded) are shown as  AxiomC{$\cdots$}.
 */
function nodeToLatex(node, depth, maxDepth) {
  const display  = displayToLatex(node.display);
  const isLeaf   = !node.children || node.children.length === 0;
  const isFrontier = !isLeaf && depth >= maxDepth;

  if (isLeaf) {
    return [
      '\\AxiomC{}',
      ruleToRightLabel(node.rule),
      `\\UnaryInfC{$${display}$}`,
    ].join('\n');
  }

  if (isFrontier) {
    return [
      '\\AxiomC{$\\cdots$}',
      `\\UnaryInfC{$${display}$}`,
    ].join('\n');
  }

  const label = ruleToRightLabel(node.rule);

  if (node.children.length === 1) {
    return [
      nodeToLatex(node.children[0], depth + 1, maxDepth),
      label,
      `\\UnaryInfC{$${display}$}`,
    ].join('\n');
  }

  // 2 children (wedge / binary rule)
  return [
    nodeToLatex(node.children[0], depth + 1, maxDepth),
    nodeToLatex(node.children[1], depth + 1, maxDepth),
    label,
    `\\BinaryInfC{$${display}$}`,
  ].join('\n');
}

/** Wrap a node's LaTeX in a prooftree environment. */
function treeToLatex(tree, maxDepth) {
  return '\\begin{prooftree}\n' + nodeToLatex(tree, 0, maxDepth) + '\n\\end{prooftree}';
}

/** Human-readable depth label. */
function depthLabel(depth, max) {
  if (depth >= max) return 'Full tree';
  if (depth === 0)  return 'Root only';
  return `Depth ${depth} / ${max}`;
}

/** Render the proof tree at the given depth and update UI controls. */
async function applySliderDepth(depth) {
  if (!sliderTree) return;
  sliderDepth = Math.max(0, Math.min(sliderMaxDepth, depth));
  depthSlider.value   = String(sliderDepth);
  depthLbl.textContent = depthLabel(sliderDepth, sliderMaxDepth);
  await renderProofTree(treeToLatex(sliderTree, sliderDepth));
}

/** Initialise the slider for a newly arrived tree. */
function setSliderTree(tree) {
  sliderTree     = tree;
  sliderMaxDepth = tree ? treeDepth(tree) : 0;
  sliderDepth    = sliderMaxDepth;
  depthSlider.max   = String(sliderMaxDepth);
  depthSlider.value = String(sliderMaxDepth);
  depthLbl.textContent = depthLabel(sliderMaxDepth, sliderMaxDepth);
  // Only show the slider when the tree has more than one level
  stepRow.style.display = sliderMaxDepth > 0 ? '' : 'none';
}

depthSlider.addEventListener('input', () => applySliderDepth(Number(depthSlider.value)));
document.getElementById('stepPrev').addEventListener('click',
  () => applySliderDepth(sliderDepth - 1));
document.getElementById('stepNext').addEventListener('click',
  () => applySliderDepth(sliderDepth + 1));

// ── MathJax helpers ──

/**
 * Transform the backend-generated LaTeX bussproofs code so that it renders
 * correctly in MathJax.
 *
 * MathJax bussproofs treats InfC/AxiomC arguments as TEXT mode (same as
 * LaTeX bussproofs), so the $…$ delimiters that switch to math mode MUST
 * be kept — without them, \neg, \vee, \sststile etc. are left in text mode
 * and render as raw characters.
 *
 * \RightLabel also gets text-mode content: keep $…$ but replace \scriptsize
 * (a text-mode size declaration) with \scriptstyle (the math-mode equivalent).
 */
function prepareForMathJax(decomp) {
  return decomp
    // \RightLabel{\scriptsize$(label)$}  →  \RightLabel{$\scriptstyle(label)$}
    .replace(/\\RightLabel\{\\scriptsize\$([^$]*)\$\}/g,
             (_, inner) => `\\RightLabel{$\\scriptstyle ${inner}$}`);
}

/** Wait until MathJax is fully initialised, then typeset element. */
async function typesetElement(el) {
  if (!window.MathJax) return;
  try {
    await MathJax.startup.promise;
    await MathJax.typesetPromise([el]);
  } catch (e) {
    console.error('MathJax typeset error:', e);
  }
}

/** Render the proof tree in the MathJax panel. */
async function renderProofTree(decomposition) {
  // Clear previous render
  await MathJax.startup.promise.then(() =>
    MathJax.typesetClear([mathTree])
  ).catch(() => {});

  if (!decomposition) {
    mathTree.innerHTML = '';
    mathTreeOuter.classList.add('empty');
    return;
  }

  const adapted = prepareForMathJax(decomposition);
  // Wrap in display-math delimiters so MathJax processes the prooftree env
  mathTree.innerHTML = '\\[' + adapted + '\\]';
  mathTreeOuter.classList.remove('empty');

  await typesetElement(mathTree);
  applyScale(treeScale);  // re-apply zoom after render
}

/** Render  ⟦Γ⟧_B = m/n ≈ …  in the stats panel. */
async function renderStats(root) {
  if (!root || !root.n) { statsOutput.innerHTML = ''; leavesInfo.textContent = ''; return; }

  const n = root.n;
  const m = String(root.m);
  const withB = root.with_belief_base;
  const suffix = withB ? '_{\\mathbb{B}}' : '';

  const mNum = parseFloat(m);
  const mIsNum = !isNaN(mNum) && String(mNum) === m.trim();

  let approxPart = '';
  if (mIsNum && n > 0) {
    const val = mNum / n;
    const dec = val.toFixed(6).replace(/\.?0+$/, '') || '0';
    approxPart = ` \\approx ${dec}`;
  }

  const latex = `\\[\\llbracket \\Gamma \\rrbracket${suffix} = \\frac{${m}}{${n}}${approxPart}\\]`;
  statsOutput.innerHTML = latex;
  await typesetElement(statsOutput);

  // Plain-text leaf summary
  const bMode = root.belief_mode || 'standard';
  const beliefNote = withB
    ? ` | mode: ${bMode}`
    : '';
  leavesInfo.textContent = `Total leaves: ${n}  |  numerator m = ${m}${beliefNote}`;
}

// ── Visual (HTML) tree ──
function nodeDisplayText(display) {
  // "sststile{n}{m}_B |- formula"  →  "[n/m]_B ⊢ formula"
  return String(display || '').replace(
    /sststile\{([^}]*)\}\{([^}]*)\}(_B)?\s*\|-/,
    (_, n, m, b) => `[${n}/${m}]${b || ''} \u22a2`
  );
}

function renderNode(node) {
  const li  = document.createElement('li');
  const box = document.createElement('div');
  box.className = 'node';
  box.innerHTML = `
    <div class="meta">
      <span class="badge">${esc(node.rule || '')}</span>
      <span>N=${esc(node.n)} M=${esc(node.m)}</span>
    </div>
    <div class="seq">${esc(nodeDisplayText(node.display))}</div>`;
  li.appendChild(box);
  if (node.children && node.children.length > 0) {
    const ul = document.createElement('ul');
    for (const child of node.children) ul.appendChild(renderNode(child));
    li.appendChild(ul);
  }
  return li;
}

function renderVisualTree(rootNode) {
  tree.innerHTML = '';
  if (!rootNode) return;
  const ul = document.createElement('ul');
  ul.appendChild(renderNode(rootNode));
  tree.appendChild(ul);
}

// ── History ──
function renderHistorySelect() {
  const prev = historySelect.value;
  historySelect.innerHTML = '';
  if (runHistory.length === 0) {
    historySelect.innerHTML = '<option value="">No history yet</option>';
    return;
  }
  const ph = document.createElement('option');
  ph.value = ''; ph.textContent = 'Select a past run';
  historySelect.appendChild(ph);
  for (const run of runHistory) {
    const opt = document.createElement('option');
    opt.value = String(run.id);
    opt.textContent = `${run.timestamp} | ${run.mode} | ${clip(run.sequent)}`;
    historySelect.appendChild(opt);
  }
  if (prev && findRun(Number(prev))) historySelect.value = prev;
  else if (currentRunId) historySelect.value = String(currentRunId);
}

// ── Validation / equivalence banners ──
function setValidationInfo(v) {
  if (!v) { validation.textContent = ''; validation.className = 'ok'; return; }
  validation.textContent = `Validation: ${v.ok ? 'OK' : 'FAILED'} — ${v.message}`;
  validation.className = v.ok ? 'ok' : 'warn';
}
function setEquivalenceInfo(info) {
  if (!info || !info.changed) { equivalence.textContent = ''; return; }
  const note = info.note ? ` (${info.note})` : '';
  equivalence.textContent = `Equivalence chain: ${info.ascii_chain}${note}`;
  equivalence.className = 'warn';
}

function sequentKindLabel(kind) {
  return kind === 'two-sided' ? 'two-sided (right commas = OR)' : 'one-sided';
}

function updateGradientState() {
  const active = modeEl.value === 'gradient' || modeEl.value === 'full';
  beliefGradEl.disabled = !active;
  beliefGradEl.placeholder = active
    ? (modeEl.value === 'full'
        ? 'Optional hybrid: 0.8; 0.6\n(values use v-δ, rest stay 1-δ)'
        : '0.8; 0.7\n(empty = 1)')
    : 'Used in Full (hybrid) or Gradient mode';
}

// ── Main: generate decomposition ──
goBtn.addEventListener('click', async () => {
  error.textContent = '';
  status.textContent = '';
  out.textContent = '';
  tree.innerHTML = '';
  mathTree.innerHTML = '';
  mathTreeOuter.classList.add('empty');
  statsOutput.innerHTML = '';
  leavesInfo.textContent = '';
  setValidationInfo(null);
  setEquivalenceInfo(null);

  const payload = {
    sequent: sequentEl.value,
    belief_set: beliefEl.value,
    belief_gradients: beliefGradEl.value,
    belief_mode: modeEl.value,
    decomposition_style: decompStyleEl.value,
  };
  setLoading(true, 'Generating decomposition…');
  try {
    const res  = await fetch('/api/decompose', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if (!data.ok) { error.textContent = data.error || 'Unknown error'; return; }

    out.textContent = data.decomposition;
    renderVisualTree(data.tree);
    setSliderTree(data.tree || null);
    setValidationInfo(data.validation || null);
    setEquivalenceInfo(data.equivalence || null);

    const kindLabel = sequentKindLabel(data.sequent_kind || 'one-sided');
    status.textContent = `Decomposition generated (${kindLabel}). Rendering proof tree…`;

    const run = {
      id: ++runCounter,
      timestamp: new Date().toLocaleTimeString(),
      sequent: payload.sequent,
      belief_set: payload.belief_set,
      belief_gradients: payload.belief_gradients,
      mode: payload.belief_mode,
      decomposition_style: payload.decomposition_style,
      decomposition: data.decomposition,
      tree: data.tree,
      sequent_kind: data.sequent_kind || 'one-sided',
      validation: data.validation || null,
      equivalence: data.equivalence || null,
      example: '', example_meta: '',
      example_mode: String(exampleMode.value || 'auto'),
    };
    runHistory.unshift(run);
    if (runHistory.length > 40) runHistory.pop();
    currentRunId = run.id;
    renderHistorySelect();

    exampleOut.textContent = '';
    exampleMeta.textContent = '';
  } catch (e) {
    error.textContent = String(e);
    return;
  } finally {
    setLoading(false);
  }

  // MathJax rendering (after loading spinner is gone)
  setLoading(true, 'Rendering proof tree…');
  try {
    await renderProofTree(out.textContent.trim());
    await renderStats(findRun(currentRunId)?.tree || null);
    status.textContent = `Decomposition ready (${sequentKindLabel(
      findRun(currentRunId)?.sequent_kind || 'one-sided')}).`;
  } catch (e) {
    error.textContent = 'MathJax rendering failed: ' + String(e);
  } finally {
    setLoading(false);
  }
});

// ── Random sequent ──
randomBtn.addEventListener('click', async () => {
  error.textContent = '';
  status.textContent = '';
  setLoading(true, 'Generating random sequent…');
  try {
    const res  = await fetch('/api/random-sequent', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({allow_incoherent: randomMode.value === 'allow_incoherent'}),
    });
    const data = await res.json();
    if (!data.ok) { error.textContent = data.error || 'Error'; return; }
    sequentEl.value = String(data.sequent || '');
    status.textContent = 'Random sequent ready. Click Generate Decomposition.';
    // Clear outputs
    out.textContent = '';
    tree.innerHTML = '';
    mathTree.innerHTML = '';
    mathTreeOuter.classList.add('empty');
    statsOutput.innerHTML = '';
    leavesInfo.textContent = '';
    setValidationInfo(null);
    setEquivalenceInfo(null);
  } catch (e) {
    error.textContent = String(e);
  } finally {
    setLoading(false);
  }
});

// ── Generate example ──
async function generateExample() {
  error.textContent = '';
  const seq = String(sequentEl.value || '').trim();
  if (!seq) { error.textContent = 'Please provide a sequent first.'; return; }
  setLoading(true, 'Generating example…');
  try {
    const res  = await fetch('/api/example', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        sequent: seq,
        belief_set: beliefEl.value || '',
        example_mode: exampleMode.value || 'auto',
      }),
    });
    const data = await res.json();
    if (!data.ok) { error.textContent = data.error || 'Unable to generate example.'; return; }
    exampleOut.textContent = String(data.example || '');
    exampleMeta.textContent = `Engine: ${data.engine || 'unknown'} | Model: ${data.model || 'n/a'}`;
    if (currentRunId) {
      const run = findRun(currentRunId);
      if (run) { run.example = exampleOut.textContent; run.example_meta = exampleMeta.textContent; }
    }
  } catch (e) {
    error.textContent = String(e);
  } finally {
    setLoading(false);
  }
}
exampleBtn.addEventListener('click', generateExample);

// ── History restore ──
historySelect.addEventListener('change', async () => {
  const run = findRun(Number(historySelect.value));
  if (!run) return;
  currentRunId = run.id;
  sequentEl.value    = run.sequent;
  beliefEl.value     = run.belief_set;
  beliefGradEl.value = run.belief_gradients || '';
  modeEl.value       = run.mode;
  decompStyleEl.value = run.decomposition_style || 'cut_free';
  randomMode.value   = run.random_mode || 'coherent';
  exampleMode.value  = run.example_mode || 'auto';
  updateGradientState();
  out.textContent    = run.decomposition;
  renderVisualTree(run.tree);
  setValidationInfo(run.validation || null);
  setEquivalenceInfo(run.equivalence || null);
  exampleOut.textContent  = run.example || '';
  exampleMeta.textContent = run.example_meta || '';
  error.textContent  = '';
  status.textContent = `Loaded from history (${sequentKindLabel(run.sequent_kind || 'one-sided')}).`;

  setLoading(true, 'Re-rendering proof tree…');
  try {
    await renderProofTree(run.decomposition);
    await renderStats(run.tree);
  } catch (e) {
    error.textContent = 'Re-render error: ' + String(e);
  } finally {
    setLoading(false);
  }
});

// ── Clear ──
clearBtn.addEventListener('click', () => {
  sequentEl.value = ''; beliefEl.value = ''; beliefGradEl.value = '';
  modeEl.value = 'standard'; decompStyleEl.value = 'cut_free';
  randomMode.value = 'coherent'; exampleMode.value = 'auto';
  out.textContent = ''; exampleOut.textContent = ''; exampleMeta.textContent = '';
  tree.innerHTML = ''; mathTree.innerHTML = '';
  mathTreeOuter.classList.add('empty');
  statsOutput.innerHTML = ''; leavesInfo.textContent = '';
  error.textContent = ''; status.textContent = '';
  setValidationInfo(null); setEquivalenceInfo(null);
  setSliderTree(null);
  updateGradientState();
  applyScale(1.0);
});

// ── Mode change ──
modeEl.addEventListener('change', updateGradientState);

// ── Init ──
updateGradientState();
applyScale(1.0);
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# HTTP handler
# ---------------------------------------------------------------------------
class _Handler(BaseHTTPRequestHandler):
    def _send_json(self, status: int, data: dict) -> None:
        payload = json.dumps(data).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _send_html(self, status: int, html: str) -> None:
        payload = html.encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        if self.path == '/':
            self._send_html(200, _HTML)
        else:
            self.send_error(404, 'Not found')

    def do_POST(self) -> None:  # noqa: N802
        allowed = {'/api/decompose', '/api/random-sequent', '/api/example'}
        if self.path not in allowed:
            self.send_error(404, 'Not found')
            return

        content_len = int(self.headers.get('Content-Length', '0'))
        raw = self.rfile.read(content_len)
        try:
            body = json.loads(raw.decode('utf-8'))
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {'ok': False, 'error': f'Invalid JSON: {exc}'})
            return

        if self.path == '/api/decompose':
            try:
                bundle = generate_decomposition_bundle(
                    str(body.get('sequent', '')),
                    str(body.get('belief_set', '')),
                    belief_mode=str(body.get('belief_mode', 'standard')),
                    decomposition_style=str(body.get('decomposition_style', 'cut_free')),
                    belief_gradients_text=str(body.get('belief_gradients', '')),
                )
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {'ok': False, 'error': str(exc)})
                return
            self._send_json(200, {
                'ok': True,
                'decomposition': bundle['decomposition'],
                'tree': bundle['tree'],
                'sequent_kind': bundle.get('sequent_kind', 'one-sided'),
                'validation': bundle['validation'],
                'equivalence': bundle.get('equivalence'),
            })
            return

        if self.path == '/api/random-sequent':
            import random as _random
            try:
                seed = body.get('seed')
                rng: _random.Random | None = None
                if seed is not None:
                    rng = _random.Random(int(seed))
                allow = body.get('allow_incoherent', False)
                if isinstance(allow, str):
                    allow = allow.strip().lower() in {'1', 'true', 'yes', 'on'}
                sequent = generate_random_sequent(rng=rng, allow_incoherent=bool(allow))
            except Exception as exc:  # noqa: BLE001
                self._send_json(400, {'ok': False, 'error': str(exc)})
                return
            self._send_json(200, {'ok': True, 'sequent': sequent})
            return

        # /api/example
        try:
            result = generate_example_from_sequent(
                str(body.get('sequent', '')),
                belief_set_text=str(body.get('belief_set', '')),
                model=str(body.get('model', '')).strip() or None,
                prefer_local_model=True,
                generation_mode=str(body.get('example_mode', 'auto')).strip() or 'auto',
            )
        except Exception as exc:  # noqa: BLE001
            self._send_json(400, {'ok': False, 'error': str(exc)})
            return
        self._send_json(200, {
            'ok': True,
            'example': result.text,
            'engine': result.engine,
            'model': result.model,
        })


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def launch_web_app() -> int:
    # Render (and most cloud platforms) set the PORT env-var and require the
    # server to bind on 0.0.0.0.  Fall back to localhost:8000 for local use.
    port = int(os.environ.get('PORT', 8000))
    host = '0.0.0.0' if os.environ.get('PORT') else '127.0.0.1'

    try:
        server = ThreadingHTTPServer((host, port), _Handler)
    except OSError as exc:
        raise RuntimeError(f'Unable to bind {host}:{port}') from exc

    actual_host, actual_port = server.server_address
    url = f'http://{actual_host}:{actual_port}'
    print(f'PID: {os.getpid()}')
    print(f'Web app available at: {url}')

    # Only try to open a browser when running locally
    if not os.environ.get('PORT'):
        try:
            webbrowser.open(url)
        except Exception:
            pass

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print('\nStopping Web App…')
    finally:
        server.server_close()
    return 0


if __name__ == '__main__':
    raise SystemExit(launch_web_app())
