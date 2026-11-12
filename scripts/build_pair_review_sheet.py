"""Render the 40 controlled pairs as a self-contained HTML review sheet.

The sheet never fabricates a reviewer identity: the reviewer types their own name,
and the export carries that name into the canonical columns
(review_status, reviewer, review_notes).

Two reviewers must work independently. Each reviewer's decisions live in their own
browser localStorage bucket keyed by reviewer name, so the same file can be opened
twice (or by two people) without overwriting each other, and each exports their own CSV.

Usage:
    python scripts/build_pair_review_sheet.py
    python scripts/build_pair_review_sheet.py --pairs data/diagnostics/neutral_scope_pairs.csv \
        --output data/diagnostics/neutral_scope_pairs_review.html
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

DEFAULT_PAIRS = Path("data/diagnostics/neutral_scope_pairs.csv")
DEFAULT_OUTPUT = Path("data/diagnostics/neutral_scope_pairs_review.html")
COLUMNS = [
    "pair_id", "target", "expected_label", "distractor", "positive_expression",
    "negative_expression", "text_positive_neighbor", "text_negative_neighbor",
    "review_status", "reviewer", "review_notes", "source",
]

TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Neutral scope pairs - human review</title>
<style>
  :root { color-scheme: light dark; --line:#d0d7de; --ok:#1a7f37; --bad:#cf222e; }
  body { font: 15px/1.55 -apple-system, "Segoe UI", sans-serif; margin: 0; padding: 24px 32px 96px; max-width: 1100px; }
  h1 { font-size: 22px; margin: 0 0 4px; }
  .sub { color:#6e7781; margin-bottom: 20px; }
  .bar { position: sticky; top:0; background: Canvas; padding: 12px 0; border-bottom:1px solid var(--line); z-index:5; }
  .bar input[type=text] { padding:6px 8px; width: 220px; }
  .bar button { padding:6px 12px; margin-left:8px; cursor:pointer; }
  .pair { border:1px solid var(--line); border-radius:8px; padding:14px 16px; margin:16px 0; }
  .head { display:flex; gap:12px; align-items:baseline; flex-wrap:wrap; }
  .pid { font-weight:700; font-family: ui-monospace, monospace; }
  .tag { font-size:12px; padding:2px 8px; border-radius:999px; border:1px solid var(--line); }
  .neutral { color:var(--ok); border-color:var(--ok); }
  .control { color:#0969da; border-color:#0969da; }
  .sents { margin:10px 0 2px; }
  .sents div { padding:4px 0; }
  .marker { display:inline-block; width:74px; font-size:12px; color:#6e7781; }
  em { background: rgba(255,193,7,.28); font-style: normal; padding:0 2px; border-radius:3px; }
  .checks { display:flex; gap:22px; flex-wrap:wrap; margin-top:10px; font-size:14px; }
  .checks label { cursor:pointer; }
  .verdict { margin-top:10px; }
  .verdict button { padding:5px 12px; margin-right:8px; cursor:pointer; }
  .verdict button.sel-ok { background:var(--ok); color:#fff; border-color:var(--ok); }
  .verdict button.sel-bad { background:var(--bad); color:#fff; border-color:var(--bad); }
  .notes { width:100%; margin-top:8px; padding:6px; box-sizing:border-box; }
  .hidden { display:none; }
  .done { opacity:.55; }
  .count { font-weight:600; }
  textarea { font:inherit; }
  ol { padding-left: 22px; }
</style>
</head>
<body>
<h1>Neutral scope pairs &mdash; human review</h1>
<div class="sub">80 sentences, 40 pairs. Read the instructions before starting, then work alone.</div>

<div class="pair" id="instructions">
  <div class="head"><span class="pid">How to judge each pair</span></div>
  <ol>
    <li><b>Target is neutral.</b> The highlighted target aspect is only <i>mentioned</i>; the sentence expresses no opinion about it.</li>
    <li><b>Only the distractor changed.</b> The two sentences differ only in the opinion word about the other aspect, and the meaning of the target is unchanged.</li>
    <li><b>Natural English.</b> Both sentences read like something a reviewer could plausibly write.</li>
    <li><b>Distractor polarity.</b> In the positive sentence the other aspect is clearly positive, in the negative sentence clearly negative.</li>
  </ol>
  <div class="sub" style="margin:8px 0 0">Approve only if all four hold. If one fails, reject and say why in the notes.
  Do not look at any model output while judging.</div>
</div>

<div class="bar">
  <label>Reviewer name: <input id="reviewer" type="text" placeholder="e.g. Chen" autocomplete="off"></label>
  <button id="export">Export my CSV</button>
  <button id="reset">Clear my progress</button>
  <span class="count" id="count"></span>
</div>

<div id="pairs"></div>

<script>
const ROWS = __ROWS__;
const COLUMNS = __COLUMNS__;
const storeKey = (name) => 'neutralScopePairs::' + name;

function load() {
  const name = document.getElementById('reviewer').value.trim();
  if (!name) return {};
  try { return JSON.parse(localStorage.getItem(storeKey(name)) || '{}'); }
  catch (e) { return {}; }
}
function save(state) {
  const name = document.getElementById('reviewer').value.trim();
  if (!name) return;
  localStorage.setItem(storeKey(name), JSON.stringify(state));
  renderCount(state);
}
function esc(text) {
  return String(text).replace(/[&<>"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
}
function highlight(text, term) {
  if (!term) return esc(text);
  const i = text.toLowerCase().indexOf(String(term).toLowerCase());
  if (i < 0) return esc(text);
  return esc(text.slice(0, i)) + '<em>' + esc(text.slice(i, i + term.length)) + '</em>' + esc(text.slice(i + term.length));
}

const state = load();
const container = document.getElementById('pairs');

ROWS.forEach((row) => {
  const isControl = row.expected_label !== 'neutral';
  const div = document.createElement('div');
  div.className = 'pair';
  div.dataset.pid = row.pair_id;
  const current = state[row.pair_id] || {checks: [], verdict: '', notes: ''};
  const checks = [
    ['neutral', 'target is neutral'],
    ['only_distractor', 'only the distractor opinion changed'],
    ['natural', 'both sentences are natural'],
    ['polarity', 'distractor polarity is clear'],
  ];
  div.innerHTML =
    '<div class="head"><span class="pid">' + esc(row.pair_id) + '</span>' +
    '<span class="tag ' + (isControl ? 'control' : 'neutral') + '">expected: ' + esc(row.expected_label) + '</span>' +
    '<span class="tag">target: ' + esc(row.target) + '</span>' +
    '<span class="tag">distractor: ' + esc(row.distractor) + '</span></div>' +
    '<div class="sents">' +
      '<div><span class="marker">positive</span>' + highlight(row.text_positive_neighbor, row.distractor) + '</div>' +
      '<div><span class="marker">negative</span>' + highlight(row.text_negative_neighbor, row.distractor) + '</div>' +
    '</div>' +
    '<div class="checks">' + checks.map(([key, label]) =>
      '<label><input type="checkbox" data-check="' + key + '"' +
      (current.checks.includes(key) ? ' checked' : '') + '> ' + label + '</label>').join('') + '</div>' +
    '<div class="verdict">' +
      '<button data-verdict="approved" class="' + (current.verdict === 'approved' ? 'sel-ok' : '') + '">approve</button>' +
      '<button data-verdict="rejected" class="' + (current.verdict === 'rejected' ? 'sel-bad' : '') + '">reject</button>' +
    '</div>' +
    '<textarea class="notes" rows="1" placeholder="notes (required when rejecting)">' + esc(current.notes) + '</textarea>';

  const write = () => {
    const entry = {
      checks: [...div.querySelectorAll('input[data-check]')].filter((x) => x.checked).map((x) => x.dataset.check),
      verdict: div.dataset.verdict || '',
      notes: div.querySelector('.notes').value.trim(),
    };
    if (!entry.verdict && !entry.checks.length && !entry.notes) delete state[row.pair_id];
    else state[row.pair_id] = entry;
    save(state);
  };

  div.querySelectorAll('input[data-check]').forEach((box) => box.addEventListener('change', write));
  div.querySelector('.notes').addEventListener('input', write);
  div.querySelectorAll('button[data-verdict]').forEach((btn) => btn.addEventListener('click', () => {
    if (div.dataset.verdict === btn.dataset.verdict) delete div.dataset.verdict;
    else div.dataset.verdict = btn.dataset.verdict;
    div.querySelectorAll('button[data-verdict]').forEach((b) =>
      b.classList.toggle('sel-ok', div.dataset.verdict === 'approved' && b.dataset.verdict === 'approved'));
    div.querySelectorAll('button[data-verdict]').forEach((b) =>
      b.classList.toggle('sel-bad', div.dataset.verdict === 'rejected' && b.dataset.verdict === 'rejected'));
    div.classList.toggle('done', Boolean(div.dataset.verdict));
    write();
  }));
  div.classList.toggle('done', Boolean(current.verdict));
  container.appendChild(div);
});

function renderCount(st) {
  const done = ROWS.filter((r) => st[r.pair_id] && st[r.pair_id].verdict).length;
  const approved = ROWS.filter((r) => st[r.pair_id] && st[r.pair_id].verdict === 'approved').length;
  document.getElementById('count').textContent = done + '/' + ROWS.length + ' decided (' + approved + ' approved)';
}

if (document.getElementById('reviewer').value) renderCount(state);
document.getElementById('reviewer').addEventListener('input', () => {
  const fresh = load();
  location.hash = '';
  ROWS.forEach((row) => {
    const div = container.querySelector('[data-pid="' + row.pair_id + '"]');
    const entry = fresh[row.pair_id] || {checks: [], verdict: '', notes: ''};
    div.dataset.verdict = entry.verdict;
    div.classList.toggle('done', Boolean(entry.verdict));
    div.querySelectorAll('input[data-check]').forEach((box) =>
      box.checked = entry.checks.includes(box.dataset.check));
    div.querySelector('.notes').value = entry.notes;
    div.querySelectorAll('button[data-verdict]').forEach((b) => {
      b.classList.toggle('sel-ok', entry.verdict === 'approved' && b.dataset.verdict === 'approved');
      b.classList.toggle('sel-bad', entry.verdict === 'rejected' && b.dataset.verdict === 'rejected');
    });
  });
  renderCount(fresh);
});

document.getElementById('export').addEventListener('click', () => {
  const name = document.getElementById('reviewer').value.trim();
  if (!name) { alert('Type your reviewer name first so the export is attributable.'); return; }
  const missing = ROWS.filter((r) => !(state[r.pair_id] || {}).verdict);
  if (missing.length) {
    if (!confirm(missing.length + ' pair(s) have no verdict yet. Export anyway?')) return;
  }
  const q = (v) => '"' + String(v == null ? '' : v).replace(/"/g, '""') + '"';
  const lines = [COLUMNS.map(q).join(',')];
  ROWS.forEach((row) => {
    const entry = state[row.pair_id] || {checks: [], verdict: '', notes: ''};
    const notes = [entry.notes, 'checks:' + entry.checks.join('+')].filter(Boolean).join('; ');
    const out = {...row, review_status: entry.verdict || 'pending', reviewer: name, review_notes: notes};
    lines.push(COLUMNS.map((c) => q(out[c])).join(','));
  });
  const blob = new Blob([lines.join('\\n') + '\\n'], {type: 'text/csv;charset=utf-8'});
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = 'neutral_scope_pairs_reviewed_' + name.replace(/[^A-Za-z0-9_-]+/g, '_') + '.csv';
  a.click();
  URL.revokeObjectURL(a.href);
});

document.getElementById('reset').addEventListener('click', () => {
  const name = document.getElementById('reviewer').value.trim();
  if (!name) return;
  if (!confirm('Clear every decision saved for "' + name + '"?')) return;
  localStorage.removeItem(storeKey(name));
  location.reload();
});

renderCount(state);
</script>
</body>
</html>
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=Path, default=DEFAULT_PAIRS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    with args.pairs.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"No rows in {args.pairs}")
    missing = set(COLUMNS) - set(rows[0])
    if missing:
        raise SystemExit(f"{args.pairs} is missing columns: {sorted(missing)}")

    html = (TEMPLATE
            .replace("__ROWS__", json.dumps(rows, ensure_ascii=False))
            .replace("__COLUMNS__", json.dumps(COLUMNS)))
    args.output.write_text(html, encoding="utf-8")
    print(f"Wrote {args.output} with {len(rows)} pairs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
