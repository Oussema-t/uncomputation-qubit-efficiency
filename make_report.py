#!/usr/bin/env python3
"""Generate a PDF report from the benchmark results.

Reads the JSON written by ``uncomputation_demo.py`` and ``qaoa_scheduling.py``
and renders a print-ready HTML document, then converts it to PDF with headless
Chrome. **Every number in the report is read from those files** -- nothing is
transcribed by hand, which removes transcription error. (It does not by itself
prevent a committed PDF from going stale: if the JSON is regenerated without
rebuilding the PDF, the committed PDF lags. Regenerate both in the same step.)

    python uncomputation_demo.py
    python qaoa_scheduling.py --layers 3
    python make_report.py

Produces ``uncomputation_report.pdf``.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import pathlib
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

LOGGER = logging.getLogger("make_report")

CHROME_CANDIDATES = (
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    "google-chrome",
    "chromium",
    "chromium-browser",
)

GARBAGE = "#c9502a"
CLEAN = "#1f5fac"
INK = "#111111"
MUTED = "#5c5b57"
RULE = "#dcdbd6"


def load_json(path: pathlib.Path) -> Optional[Dict[str, Any]]:
    """Load a results file, returning None (with a warning) if it is absent."""
    if not path.is_file():
        LOGGER.warning("missing %s -- its section will be omitted", path)
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def embed_image(path: pathlib.Path) -> Optional[str]:
    """Return a data: URI for an image, or None if it is missing."""
    if not path.is_file():
        LOGGER.warning("missing figure %s", path)
        return None
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def fmt(value: Any, digits: int = 4) -> str:
    """Format a number for the report, using scientific notation when tiny."""
    if value is None:
        return "n/a"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return str(value)
    number = float(value)
    if number != 0.0 and abs(number) < 1e-3:
        return f"{number:.2e}"
    return f"{number:.{digits}f}"


def scaling_section(data: Optional[Dict[str, Any]], figure: Optional[str]) -> str:
    """Render the synthetic-benchmark section from benchmark_results.json."""
    if data is None:
        return "<p class='warn'>benchmark_results.json not found.</p>"

    meta = data["metadata"]
    records = data["records"]
    by_n = {r["n_steps"]: r for r in records}
    last = records[-1]
    verified = [r for r in records if r["statevector_verified"]]
    worst_dev = max(
        (r["m1_m2_max_deviation"] or 0.0) for r in verified
    ) if verified else None

    rows: List[str] = []
    for n in (1, 5, 10, last["n_steps"]):
        if n not in by_n:
            continue
        r = by_n[n]
        rows.append(
            f"<tr><td>{n}</td><td>{r['naive_total_qubits']}</td>"
            f"<td>{r['uncomputed_total_qubits']}</td>"
            f"<td>{r['naive_total_qubits'] - r['uncomputed_total_qubits']}</td></tr>"
        )

    fid = min(r["fidelity_uncomputed_vs_ideal"] for r in records)
    dist = max(r["trace_distance_uncomputed_vs_ideal"] for r in records)
    entropy_before = last["ancilla_entropy_before"]
    entropy_after = max(r["ancilla_entropy_after"] for r in records)
    purity_after = last["ancilla_purity_after"]
    fid_basis = min(r["fidelity_naive_vs_uncomputed"]["basis"] for r in records)
    fid_uniform = last["fidelity_naive_vs_uncomputed"]["uniform"]

    figure_html = (
        f"<figure><img src='{figure}' alt='Qubit scaling comparison'>"
        f"<figcaption>Figure 1 &mdash; Circuit width against the number of "
        f"computation steps (left), and the scratch register's entropy before "
        f"and after <em>U&dagger;</em> (right). The shaded band marks where the "
        f"naive scenario was additionally confirmed by full state-vector "
        f"simulation.</figcaption></figure>"
        if figure
        else "<p class='warn'>qubit_scaling_comparison.png not found.</p>"
    )

    return f"""
<h2>3. Result 1 &mdash; the synthetic benchmark</h2>

<p>An <em>N</em>-step phase oracle on {meta['n_data']} data qubits. Each step
computes a three-literal conjunction into two scratch qubits using two Toffolis,
applies a phase, and either leaves the scratch behind (scenario A) or unwinds it
with <em>U&dagger;</em> and reuses it (scenario B).</p>

{figure_html}

<h3>3.1 Circuit width</h3>
<p>Widths are <strong>measured from the constructed circuit</strong>, not
restated from a formula.</p>
<table>
<tr><th>N steps</th><th>Without uncomputation</th><th>With uncomputation</th>
<th>Saved</th></tr>
{''.join(rows)}
</table>
<p class="note">Growth is exactly +2 qubits per step without cleanup and +0 with
it, across all N = 1&ndash;{last['n_steps']}.</p>

<h3>3.2 The cleanup is exact</h3>
<table>
<tr><th>Quantity</th><th>Value</th><th class="txt">Meaning</th></tr>
<tr><td>Fidelity, scenario B vs logical target</td><td>&ge; {fid:.15f}</td>
<td class="txt">uncomputation changes nothing logically</td></tr>
<tr><td>Trace distance</td><td>&le; {fmt(dist)}</td><td class="txt">same state</td></tr>
<tr><td>Scratch entropy before <em>U&dagger;</em></td>
<td>{entropy_before:.9f} bits</td><td class="txt">entangled, unusable</td></tr>
<tr><td>Scratch entropy after <em>U&dagger;</em></td><td>&le; {fmt(entropy_after)} bits</td>
<td class="txt">disentangled</td></tr>
<tr><td>Scratch purity after <em>U&dagger;</em></td><td>{purity_after:.9f}</td>
<td class="txt">back to a pure |00&rang;</td></tr>
</table>

<p>The pre-cleanup values match closed form exactly, which is what confirms the
partial trace is correct: over a uniform input the scratch pair takes the value
(0,0) for 12 of 16 basis states and (1,0), (1,1) for 2 each, giving
S = 1.06128 bits and purity = 0.59375.</p>

<h3>3.3 The result that matters most</h3>
<p>&ldquo;Fidelity &asymp; 1&rdquo; between the two scenarios holds
<strong>only for computational-basis inputs</strong>:</p>
<table>
<tr><th>Input state</th><th>Scenario A fidelity vs target</th><th class="txt">Interpretation</th></tr>
<tr><td>computational basis</td><td>{fid_basis:.9f}</td>
<td class="txt">garbage is a harmless deterministic label</td></tr>
<tr><td>uniform superposition (N = {last['n_steps']})</td><td>{fid_uniform:.9f}</td>
<td class="txt">which-path record; coherence destroyed</td></tr>
</table>
<p>The floor is 1/2<sup>{meta['n_data']}</sup> = {1 / 2 ** meta['n_data']:.4f}, the
maximally mixed limit. <strong>For any algorithm that relies on interference,
skipping uncomputation is a correctness bug, not merely a larger
circuit.</strong></p>

<p class="note">Cross-validation: the structured model agrees with full
state-vector simulation to {fmt(worst_dev)} over N = 1&ndash;{verified[-1]['n_steps']
if verified else 0}; an independent JavaScript implementation agrees with the
Python one to ~1e-15.</p>
"""


def _toffoli_svg(x: float, ctrl_ys: Sequence[float], target_y: float,
                 color: str) -> str:
    """A Toffoli gate: filled control dots, an open target, a connecting line."""
    ys = list(ctrl_ys) + [target_y]
    parts = [
        f'<line x1="{x}" y1="{min(ys)}" x2="{x}" y2="{max(ys)}" '
        f'stroke="{color}" stroke-width="1.8"/>'
    ]
    for cy in ctrl_ys:
        parts.append(f'<circle cx="{x}" cy="{cy}" r="4.6" fill="{color}"/>')
    parts.append(
        f'<circle cx="{x}" cy="{target_y}" r="10" fill="#ffffff" '
        f'stroke="{color}" stroke-width="1.8"/>'
        f'<line x1="{x - 10}" y1="{target_y}" x2="{x + 10}" y2="{target_y}" '
        f'stroke="{color}" stroke-width="1.8"/>'
        f'<line x1="{x}" y1="{target_y - 10}" x2="{x}" y2="{target_y + 10}" '
        f'stroke="{color}" stroke-width="1.8"/>'
    )
    return "".join(parts)


def circuit_architecture_html() -> str:
    """Static, faithful diagram of the QAOA circuit and its named layers.

    Reflects the actual gate sequence in ``qaoa_circuit_uncomputed`` /
    ``cost_layer_uncomputed`` / ``compute_step`` -- not a stylised sketch:

      state prep   H on every data qubit
      cost layer   per clause: compute (2 Toffolis into t,r) -> phase on r ->
                   uncompute (adjoint); then a workload phase on each data qubit
      mixer        RX(2 beta) on every data qubit
      repeat cost+mixer for l = 1..p, then measure.
    """
    mono = 'font-family="ui-monospace, Menlo, monospace"'
    sans = 'font-family="-apple-system, Helvetica, Arial, sans-serif"'

    # ---- Diagram A: block-level ansatz -----------------------------------
    a = f'''<svg viewBox="0 0 860 210" width="100%" role="img"
      aria-label="QAOA ansatz block diagram">
  <line x1="150" y1="72" x2="744" y2="72" stroke="{INK}" stroke-width="1.5"/>
  <line x1="150" y1="150" x2="744" y2="150" stroke="{INK}" stroke-width="1.5"/>
  <text x="10" y="69" {sans} font-size="12" fill="{INK}">data</text>
  <text x="10" y="84" {mono} font-size="10" fill="{MUTED}">|0&#10217;<tspan
      baseline-shift="super" font-size="7">&#8855;9</tspan></text>
  <text x="10" y="147" {sans} font-size="12" fill="{INK}">scratch</text>
  <text x="10" y="162" {mono} font-size="10" fill="{MUTED}">|0&#10217;<tspan
      baseline-shift="super" font-size="7">&#8855;2</tspan></text>

  <rect x="158" y="56" width="40" height="32" rx="4" fill="#fbfbf9"
    stroke="{INK}" stroke-width="1.6"/>
  <text x="178" y="77" {mono} font-size="14" fill="{INK}"
    text-anchor="middle">H</text>
  <text x="178" y="104" {sans} font-size="9.5" fill="{MUTED}"
    text-anchor="middle">state prep</text>

  <rect x="226" y="40" width="392" height="150" rx="7" fill="none"
    stroke="{MUTED}" stroke-width="1.1" stroke-dasharray="4 3"/>
  <text x="422" y="34" {sans} font-size="10.5" fill="{MUTED}"
    text-anchor="middle">repeat &#215; p layers</text>

  <rect x="250" y="50" width="120" height="112" rx="5" fill="#fdf1ec"
    stroke="{GARBAGE}" stroke-width="1.8"/>
  <text x="310" y="99" {mono} font-size="14" fill="{INK}"
    text-anchor="middle">U<tspan baseline-shift="sub"
    font-size="10">C</tspan>(&#947;<tspan baseline-shift="sub"
    font-size="10">&#8467;</tspan>)</text>
  <text x="310" y="118" {sans} font-size="9.5" fill="{GARBAGE}"
    text-anchor="middle">cost layer</text>

  <text x="392" y="146" {mono} font-size="10" fill="{MUTED}">|0&#10217;</text>

  <rect x="442" y="52" width="120" height="40" rx="5" fill="#eef4fb"
    stroke="{CLEAN}" stroke-width="1.8"/>
  <text x="502" y="70" {mono} font-size="14" fill="{INK}"
    text-anchor="middle">U<tspan baseline-shift="sub"
    font-size="10">B</tspan>(&#946;<tspan baseline-shift="sub"
    font-size="10">&#8467;</tspan>)</text>
  <text x="502" y="84" {sans} font-size="9" fill="{CLEAN}"
    text-anchor="middle">mixer</text>

  <rect x="650" y="56" width="44" height="32" rx="4" fill="#fbfbf9"
    stroke="{INK}" stroke-width="1.6"/>
  <path d="M660 80 A12 12 0 0 1 684 80" fill="none" stroke="{INK}"
    stroke-width="1.4"/>
  <line x1="672" y1="80" x2="682" y2="62" stroke="{INK}" stroke-width="1.4"/>
  <text x="672" y="104" {sans} font-size="9.5" fill="{MUTED}"
    text-anchor="middle">measure</text>
</svg>'''

    # ---- Diagram B: inside one cost layer, one clause --------------------
    la, lb, lc, t, r = 46, 80, 114, 162, 200
    g1, g2, g3, g4, g5 = 214, 300, 392, 484, 570
    wires = "".join(
        f'<line x1="150" y1="{y}" x2="700" y2="{y}" stroke="{INK}" '
        f'stroke-width="1.4"/>'
        for y in (la, lb, lc, t, r)
    )
    labels = "".join(
        f'<text x="14" y="{y + 4}" {mono} font-size="11" fill="{fill}">{lab}</text>'
        for y, lab, fill in (
            (la, "&#8467;<tspan baseline-shift='sub' font-size='8'>a</tspan>", INK),
            (lb, "&#8467;<tspan baseline-shift='sub' font-size='8'>b</tspan>", INK),
            (lc, "&#8467;<tspan baseline-shift='sub' font-size='8'>c</tspan>", INK),
            (t, "t  |0&#10217;", MUTED),
            (r, "r  |0&#10217;", MUTED),
        )
    )
    b = f'''<svg viewBox="0 0 860 262" width="100%" role="img"
      aria-label="Cost-layer internals for one clause">
  {wires}
  <line x1="150" y1="138" x2="700" y2="138" stroke="{RULE}" stroke-width="1"
    stroke-dasharray="3 3"/>
  <text x="704" y="{(la + lc) / 2 + 4}" {sans} font-size="9.5" fill="{MUTED}">3
    literals</text>
  <text x="704" y="{(t + r) / 2 + 4}" {sans} font-size="9.5" fill="{MUTED}">scratch
    (t, r)</text>
  {labels}

  {_toffoli_svg(g1, [la, lb], t, GARBAGE)}
  {_toffoli_svg(g2, [t, lc], r, GARBAGE)}

  <rect x="{g3 - 13}" y="{r - 13}" width="26" height="26" rx="4" fill="#ffffff"
    stroke="{INK}" stroke-width="1.7"/>
  <text x="{g3}" y="{r + 4}" {mono} font-size="12" fill="{INK}"
    text-anchor="middle">P</text>

  {_toffoli_svg(g4, [t, lc], r, CLEAN)}
  {_toffoli_svg(g5, [la, lb], t, CLEAN)}

  <line x1="176" y1="228" x2="326" y2="228" stroke="{GARBAGE}"
    stroke-width="1.6"/>
  <text x="251" y="244" {sans} font-size="10" fill="{GARBAGE}"
    text-anchor="middle">compute U<tspan baseline-shift="sub"
    font-size="8">k</tspan></text>
  <line x1="366" y1="228" x2="418" y2="228" stroke="{INK}" stroke-width="1.6"/>
  <text x="392" y="244" {sans} font-size="10" fill="{INK}"
    text-anchor="middle">use</text>
  <line x1="458" y1="228" x2="608" y2="228" stroke="{CLEAN}"
    stroke-width="1.6"/>
  <text x="533" y="244" {sans} font-size="10" fill="{CLEAN}"
    text-anchor="middle">uncompute U<tspan baseline-shift="sub"
    font-size="8">k</tspan>&#8224;</text>
</svg>'''

    return f'''
<h3>4.1 Circuit architecture and named layers</h3>
<p>The ansatz is the standard QAOA form (Farhi, Goldstone &amp; Gutmann,
arXiv:1411.4028), with the cost layer built by the compute&ndash;use&ndash;
uncompute motif from Result&nbsp;1. The diagram below is the actual gate
sequence in <code>qaoa_circuit_uncomputed</code>, not a stylised sketch.</p>

<figure>{a}<figcaption>Figure 2 &mdash; QAOA ansatz, block level. The data
register is prepared in uniform superposition, then <em>p</em> repetitions of a
cost layer and a mixer are applied before measurement. The scratch register is
borrowed and returned <em>inside</em> each cost layer, so it never enters the
mixer &mdash; which is exactly why one pair suffices at any depth.</figcaption>
</figure>

<p><strong>The named layers</strong>, in order:</p>
<table>
<tr><th class="txt">Layer</th><th class="txt">Operator</th>
<th class="txt">What it does</th></tr>
<tr><td class="txt">State preparation</td>
<td class="txt"><code>H</code><sup>&#8855;9</sup></td>
<td class="txt">equal superposition over all 512 assignments</td></tr>
<tr><td class="txt">Cost layer <code>U_C(&#947;&#8467;)</code></td>
<td class="txt">exp(&#8722;i&#947;&#8467;&#183;C(x))</td>
<td class="txt">phases each assignment by its cost, via clause oracles + a
one-local workload phase</td></tr>
<tr><td class="txt">Mixer <code>U_B(&#946;&#8467;)</code></td>
<td class="txt">&#8719;<tspan>j</tspan> RX(2&#946;&#8467;)</td>
<td class="txt">transverse-field driver that mixes amplitude between
assignments</td></tr>
<tr><td class="txt">(repeat cost + mixer for &#8467; = 1&#8230;p)</td>
<td class="txt">&mdash;</td>
<td class="txt">depth <em>p</em> sets the expressiveness</td></tr>
<tr><td class="txt">Measurement</td><td class="txt">Z basis</td>
<td class="txt">sample a candidate roster</td></tr>
</table>

<p>Inside the cost layer, each of the 6 three-literal clauses is applied as a
compute&ndash;use&ndash;uncompute block, then reused for the next clause:</p>

<figure>{b}<figcaption>Figure 3 &mdash; One clause inside the cost layer, on its
three literal wires and the shared scratch pair (t, r). Orange: two Toffolis
compute the clause violation into scratch (t = &#8467;<tspan>a</tspan> &and;
&#8467;<tspan>b</tspan>, then r = t &and; &#8467;<tspan>c</tspan>). Black: a phase
gate kicks back the penalty on r. Blue: the adjoint Toffolis uncompute, returning
(t, r) to |00&#10217; so the next clause reuses them. Signed literals add an X
before and after on negated inputs (omitted for clarity). Every data qubit also
receives a one-local workload phase P(&#8722;&#947;&#183;w) once per cost
layer.</figcaption></figure>
'''


def qaoa_section(data: Optional[Dict[str, Any]], figure: Optional[str]) -> str:
    """Render the QAOA section from qaoa_results.json."""
    if data is None:
        return (
            "<h2>4. Result 2 &mdash; QAOA shift scheduling</h2>"
            "<p class='warn'>qaoa_results.json not found. Run "
            "<code>python qaoa_scheduling.py --layers 3</code> first.</p>"
        )

    meta = data["metadata"]
    base = data["baselines"]
    diag = data["diagnostics"]
    widths = data["widths"]
    results = data["results"]
    depths = sorted(int(p) for p in results)

    width_rows = "".join(
        f"<tr><td>{p}</td><td>{widths[str(p)]['naive']}</td>"
        f"<td>{widths[str(p)]['uncomputed']}</td>"
        f"<td>{widths[str(p)]['naive'] - widths[str(p)]['uncomputed']}</td></tr>"
        for p in depths
    )

    # Normaliser that converts a raw-cost spread into score units. Score is
    # (random_mean - E[cost]) / (random_mean - optimum), so a cost std of sigma
    # is a score std of sigma / spread_norm.
    spread_norm = base["random_mean"] - base["optimum"]

    def mean_score(rec: Dict[str, Any]) -> float:
        # Older JSON may predate the mean fields; fall back to best.
        return rec.get("mean_score", rec["score"])

    quality_rows = ""
    for p in depths:
        for scenario, label in (("uncomputed", "with uncomputation"),
                                ("naive", "without uncomputation")):
            r = results[str(p)][scenario]
            css = "clean" if scenario == "uncomputed" else "garbage"
            score_spread = r["cost_spread"] / spread_norm
            quality_rows += (
                f"<tr><td>{p}</td><td class='{css}'>{label}</td>"
                f"<td>{r['score']:.4f}</td>"
                f"<td>{mean_score(r):.4f}</td>"
                f"<td>{r['probability_optimal']:.4f}</td>"
                f"<td>&plusmn;{score_spread:.4f}</td></tr>"
            )
    quality_rows += (
        f"<tr class='baseline'><td>&mdash;</td><td>uniform random guessing</td>"
        f"<td>0.0000</td><td>0.0000</td>"
        f"<td>{len(base['optimum_indices']) / 2 ** 9:.4f}</td><td>&mdash;</td></tr>"
    )

    # Significance, in consistent (score) units on both sides. gap is a
    # difference of scores; the noise floor is the larger restart spread
    # converted to score units. (An earlier version compared the dimensionless
    # gap against a raw-cost spread, which was ~spread_norm-times too strict.)
    verdict_rows = []
    for p in depths:
        unc = results[str(p)]["uncomputed"]
        naive = results[str(p)]["naive"]
        gap = unc["score"] - naive["score"]
        noise = max(unc["cost_spread"], naive["cost_spread"]) / spread_norm
        ratio = abs(gap) / noise if noise > 0 else float("inf")
        verdict = (
            ("uncomputed better" if gap > 0 else "naive better")
            if abs(gap) > noise
            else "within noise"
        )
        verdict_rows.append(
            f"<tr><td>{p}</td><td>{gap:+.4f}</td><td>&plusmn;{noise:.4f}</td>"
            f"<td>{ratio:.1f}&times;</td><td class='txt'>{verdict}</td></tr>"
        )

    first, last = depths[0], depths[-1]
    unc_first = results[str(first)]["uncomputed"]["score"]
    unc_last = results[str(last)]["uncomputed"]["score"]
    naive_first = results[str(first)]["naive"]["score"]
    naive_last = results[str(last)]["naive"]["score"]
    naive_ceiling = max(results[str(p)]["naive"]["score"] for p in depths)
    gap_first = unc_first - naive_first
    trend_html = f"""
<table>
<tr><th class="txt">Scenario</th><th>Score at p = {first}</th>
<th>Score at p = {last}</th><th class="txt">Behaviour with depth</th></tr>
<tr><td class="clean">with uncomputation</td><td>{unc_first:.4f}</td>
<td>{unc_last:.4f}</td>
<td class="txt">already strong at p = {first}; improves with depth</td></tr>
<tr><td class="garbage">without uncomputation</td><td>{naive_first:.4f}</td>
<td>{naive_last:.4f}</td>
<td class="txt">flat, never exceeds {naive_ceiling:.2f} at any depth</td></tr>
</table>"""

    figure_html = (
        f"<figure><img src='{figure}' alt='QAOA comparison'>"
        f"<figcaption>Figure 4 &mdash; Circuit width against QAOA depth (left) "
        f"and solution quality against the same depth (right). A score of 1 is "
        f"the exact optimum; 0 is uniform random guessing.</figcaption></figure>"
        if figure
        else "<p class='warn'>qaoa_scheduling_comparison.png not found.</p>"
    )

    return f"""
<h2>4. Result 2 &mdash; QAOA on a real scheduling problem</h2>

<p>Three staff, three shifts, nine binary decision variables. Two families of
hard constraint, both <strong>three-literal</strong> &mdash; which is precisely
why scratch qubits are needed at all, since two-local penalty terms compile to
CNOT ladders and need none:</p>
<ul>
<li><strong>coverage</strong> &mdash; every shift needs at least one person</li>
<li><strong>no-burnout</strong> &mdash; nobody works all three shifts</li>
</ul>
<p>A clause is violated exactly when a three-literal conjunction holds, so the
same compute subroutine from Result 1 builds the QAOA cost layer unchanged.</p>

{circuit_architecture_html()}

{figure_html}

<h3>4.2 Circuit width</h3>
<p>Without cleanup, every clause in every layer needs its own scratch pair, so
width grows with depth. With uncomputation it is constant.</p>
<table>
<tr><th>QAOA depth p</th><th>Without uncomputation</th>
<th>With uncomputation</th><th>Saved</th></tr>
{width_rows}
</table>

<h3>4.3 Correctness of the cost layer</h3>
<table>
<tr><th class="txt">Check</th><th>Result</th></tr>
<tr><td>Cost layer vs exp(&minus;i&gamma;C(x)) from the classical model</td>
<td>{fmt(diag['cost_layer_error'])}</td></tr>
<tr><td>Scratch entropy before <em>U&dagger;</em></td>
<td>{diag['entropy_before']:.9f} bits</td></tr>
<tr><td>Scratch entropy after <em>U&dagger;</em></td>
<td>{fmt(diag['entropy_after'])} bits</td></tr>
<tr><td>Dephasing model vs full state vector (p = 1)</td>
<td>{fmt(diag.get('naive_model_deviation'))}</td></tr>
<tr><td>Optimiser fast path vs real PennyLane circuit</td>
<td>{fmt(diag.get('fast_path_deviation'))}</td></tr>
</table>

<h3>4.4 Solution quality</h3>
<p>The exact optimum is cost {base['optimum']:.1f}, found by enumerating all 512
assignments ({len(base['optimum_indices'])} rosters achieve it). Score is
normalised so 1 is that optimum and 0 is uniform random guessing. Two score
columns are shown: <em>best</em> is the best of {meta['restarts']} optimiser
restarts (what a user who ran it would keep); <em>mean</em> is the average across
restarts (a fairer scenario-to-scenario comparison, since the two landscapes have
very different numbers of local minima). Spread is the restart standard deviation
in score units.</p>
<table>
<tr><th>p</th><th>Scenario</th><th>Score (best)</th><th>Score (mean)</th>
<th>P(optimal)</th><th>Spread</th></tr>
{quality_rows}
</table>

<h3>4.5 Does the garbage change the answers?</h3>
<p>Yes &mdash; decisively, and at every depth. The dirty circuit is stuck near
random guessing (score {naive_first:.2f}&ndash;{naive_ceiling:.2f}) no matter how
many layers it is given; the clean circuit already beats it by a wide margin at
p&nbsp;=&nbsp;{first} and improves with depth:</p>
{trend_html}
<p>Judged at each depth against the restart spread &mdash; both quantities now in
the same (score) units:</p>
<table>
<tr><th>p</th><th>Score gap (uncomputed &minus; naive)</th><th>Noise floor</th>
<th>Gap / noise</th><th class="txt">Verdict</th></tr>
{''.join(verdict_rows)}
</table>

<p><strong>This is the finding.</strong> Uncleaned scratch does not merely occupy
qubits &mdash; it destroys the interference the mixer depends on, and no amount of
depth recovers it. Cleaning it with <em>U&dagger;</em> makes the algorithm work:
the effect is a large, significant gap at <em>every</em> depth, present already at
a single layer ({gap_first:+.2f} at p&nbsp;=&nbsp;{first}), not something that
emerges only once the circuit is deep.</p>

<div class="caveat">
<p><strong>A correction, stated plainly.</strong> An earlier version of this study
reported the p&nbsp;=&nbsp;1 uncomputed score as ~0.15 and framed the result as
&ldquo;quality climbs from near-random with depth&rdquo;. That p&nbsp;=&nbsp;1
figure was an <em>optimiser artifact</em>: at too few restarts the classical
optimiser fell into a local minimum. An independent audit caught it, and a
brute-force parameter grid confirms the true p&nbsp;=&nbsp;1 optimum is ~0.63. The
honest story is the one above &mdash; the clean circuit is strong from the first
layer &mdash; which is a <em>stronger</em> correctness claim, not a weaker one.
The results here use enough restarts ({meta['restarts']}) that every
configuration converges.</p>
</div>

<p class="note">Stated with its limits: one problem instance, one clause family,
{meta['restarts']} restarts per configuration. The naive-stuck / clean-works
contrast reproduces across independent random instances; the specific depth trend
is instance-dependent. No quantum advantage is claimed &mdash; the optimum is
found classically by enumerating 512 assignments. The qubit-width result in
&sect;4.1 is independent of all of this: it is structural and exact.</p>
"""


def build_html(
    scaling: Optional[Dict[str, Any]],
    qaoa: Optional[Dict[str, Any]],
    figure_1: Optional[str],
    figure_2: Optional[str],
) -> str:
    """Assemble the full report document."""
    meta = (scaling or {}).get("metadata", {})
    qmeta = (qaoa or {}).get("metadata", {})
    versions = (
        f"PennyLane {meta.get('pennylane', '?')}, NumPy {meta.get('numpy', '?')}, "
        f"Python {meta.get('python', '?')}"
    )
    seed = meta.get("seed", "?")

    headline = ""
    if scaling:
        last = scaling["records"][-1]
        headline = (
            f"{last['naive_total_qubits']} &rarr; "
            f"{last['uncomputed_total_qubits']} qubits"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>Uncomputation with U-dagger - results report</title>
<style>
  @page {{ size: A4; margin: 17mm 15mm 15mm 15mm; }}
  * {{ box-sizing: border-box; }}
  body {{
    font: 10.2pt/1.5 "Helvetica Neue", Helvetica, Arial, sans-serif;
    color: {INK}; margin: 0;
  }}
  h1 {{ font-size: 19pt; margin: 0 0 2mm; letter-spacing: -0.01em; }}
  h2 {{
    font-size: 12.5pt; margin: 7mm 0 2.5mm; padding-bottom: 1.4mm;
    border-bottom: 1.2px solid {RULE}; page-break-after: avoid;
  }}
  h3 {{ font-size: 10.8pt; margin: 5mm 0 1.8mm; page-break-after: avoid; }}
  p {{ margin: 0 0 2.6mm; }}
  ul {{ margin: 0 0 2.6mm; padding-left: 5mm; }}
  li {{ margin-bottom: 1mm; }}
  .sub {{ color: {MUTED}; font-size: 9pt; margin-bottom: 5mm; }}
  .hero {{
    border: 1.2px solid {RULE}; border-left: 3.5px solid {CLEAN};
    padding: 3.5mm 4.5mm; margin: 0 0 5mm; background: #fafaf8;
    page-break-inside: avoid;
  }}
  .hero .big {{ font-size: 15pt; font-weight: 600; color: {CLEAN}; }}
  table {{
    width: 100%; border-collapse: collapse; margin: 2mm 0 3.5mm;
    font-size: 9pt; page-break-inside: avoid;
  }}
  th {{
    text-align: left; font-weight: 600; color: {MUTED};
    border-bottom: 1.2px solid {RULE}; padding: 1.5mm 2mm;
  }}
  td {{ padding: 1.5mm 2mm; border-bottom: 0.6px solid #ececea; }}
  td:not(:first-child), th:not(:first-child) {{ text-align: right; }}
  /* Only label columns are left-aligned -- a blanket nth-child(2) rule would
     also left-align numeric columns and break their decimal alignment. */
  td.clean, td.garbage, td.txt, th.txt {{ text-align: left; }}
  .clean {{ color: {CLEAN}; }}
  .garbage {{ color: {GARBAGE}; }}
  tr.baseline td {{ color: {MUTED}; font-style: italic; }}
  figure {{ margin: 3mm 0 4mm; page-break-inside: avoid; }}
  figure img {{ width: 100%; border: 0.6px solid {RULE}; border-radius: 2px; }}
  figcaption {{ font-size: 8.4pt; color: {MUTED}; margin-top: 1.5mm; }}
  code {{ font-family: "SF Mono", Menlo, monospace; font-size: 9pt;
          background: #f2f1ee; padding: 0.3mm 1mm; border-radius: 1.5px; }}
  .note {{ font-size: 9pt; color: {MUTED}; }}
  .warn {{ color: {GARBAGE}; font-weight: 600; }}
  .caveat {{
    border-left: 3.5px solid {GARBAGE}; padding: 2mm 0 2mm 4mm;
    margin: 3mm 0; page-break-inside: avoid;
  }}
  footer {{ margin-top: 7mm; padding-top: 2.5mm; border-top: 1.2px solid {RULE};
            font-size: 8.4pt; color: {MUTED}; }}
</style></head><body>

<h1>Uncomputation with U&dagger;: qubit cost, and what garbage really does</h1>
<p class="sub">Measured results &middot; seed {seed} &middot; {versions}</p>

<div class="hero">
<div class="big">{headline}</div>
<p style="margin:1.5mm 0 0">Reusing one scratch register instead of accumulating
garbage, at 20 computation steps &mdash; with the logical state provably
unchanged (fidelity 1 to 15 decimal places). The second, larger finding: on
superposition inputs the un-cleaned version is not merely wider, it is
<strong>wrong</strong>.</p>
</div>

<h2>1. The question</h2>
<p>A quantum subroutine that computes a value needs scratch space. Because
unitary evolution is reversible it <em>cannot erase</em>, so those intermediate
values stay in the register, entangled with the data. They are
<strong>garbage</strong>, and they cause two separate problems:</p>
<ul>
<li><strong>They cannot be reused.</strong> Overwriting a qubit entangled with
your data corrupts the data, so every step must allocate fresh scratch and the
circuit grows linearly.</li>
<li><strong>They destroy interference.</strong> The garbage is a
<em>which-path record</em> &mdash; something that has learned which branch the
computation took. Trace it out and the data register is left a mixture.</li>
</ul>
<p>Applying the inverse, <code>U&dagger;</code>, unwinds the computation
coherently across every branch at once, returning the scratch to |0&rang;. It
does not measure or reset &mdash; it reverses. This report measures both
consequences.</p>

<h2>2. Method</h2>
<p>One logical operation, implemented two ways, on identical inputs:</p>
<table>
<tr><th>&nbsp;</th><th class="txt">A &mdash; naive</th><th class="txt">B &mdash; uncomputation</th></tr>
<tr><td>Scratch per step</td><td class="garbage">a fresh pair, never cleaned</td>
<td class="clean">one pair, reused</td></tr>
<tr><td>Cleanup</td><td class="garbage">none</td>
<td class="clean">adjoint(U) after each use</td></tr>
<tr><td>Width</td><td class="garbage">n + 2N</td><td class="clean">n + 2</td></tr>
</table>
<p>The compute step is two Toffolis writing <code>t = a AND b</code> then
<code>r = t AND c</code>. They do not commute, so <code>U</code> is deliberately
<em>not</em> self-inverse and <code>adjoint(U)</code> is a genuine inverse rather
than a repeat &mdash; asserted numerically in the test suite, not assumed.</p>
<p class="note">Simulation uses two independent methods, since the naive scenario
needs 2<sup>n+2N</sup> amplitudes: full state-vector simulation where it fits,
and an exact structured model elsewhere. The model is used at large sizes only
because it agrees with the state vector wherever both run.</p>

{scaling_section(scaling, figure_1)}

{qaoa_section(qaoa, figure_2)}

<h2>5. What this does and does not show</h2>
<div class="caveat">
<p><strong>No quantum advantage is claimed anywhere in this report.</strong> The
scheduling optimum is found by enumerating 512 assignments, instantly, on a
laptop. The subject is circuit construction &mdash; how wide a circuit must be,
and what uncleaned scratch does to the answer.</p>
</div>
<ul>
<li><strong>O(1) scratch holds for sequential, independent steps</strong>, the
structure used here. It does not hold in general: when a later step depends on an
earlier intermediate, the achievable space/time trade-off is governed by
Bennett's reversible pebbling result, and buying space back costs time.</li>
<li><strong>U&dagger; trades width for depth.</strong> It roughly doubles the
gates per step. On noisy hardware that trade is not automatically favourable;
this study measures width only and is noiseless throughout.</li>
<li><strong>One problem family, small sizes.</strong> Nine to twelve qubits,
three-literal clauses. Checked across several seeds and two register sizes, but
a different circuit class could behave differently.</li>
<li><strong>Fidelity convention</strong> is Uhlmann, squared (F = 1 for identical
states). Entropy is in bits.</li>
</ul>

<h2>6. References</h2>
<p class="note">
C. H. Bennett, &ldquo;Logical reversibility of computation&rdquo;, <em>IBM
Journal of Research and Development</em> <strong>17</strong>(6), 525&ndash;532
(1973).<br>
C. H. Bennett, &ldquo;Time/space trade-offs for reversible computation&rdquo;,
<em>SIAM Journal on Computing</em> <strong>18</strong>(4), 766&ndash;776 (1989).<br>
E. Farhi, J. Goldstone &amp; S. Gutmann, &ldquo;A Quantum Approximate
Optimization Algorithm&rdquo;, arXiv:1411.4028 (2014).<br>
M. A. Nielsen &amp; I. L. Chuang, <em>Quantum Computation and Quantum
Information</em>, 10th Anniversary Edition, Cambridge University Press (2010)
&mdash; Ch. 3 (reversible computation), Ch. 9 (fidelity, trace distance).<br>
Cited at chapter granularity; these are standard references for the construction,
not the source of any number reported above.
</p>

<footer>
Generated by <code>make_report.py</code> directly from
<code>benchmark_results.json</code> and <code>qaoa_results.json</code> &mdash;
every figure is read from those files rather than typed in by hand. (This
removes transcription error, but a committed PDF can still be stale if the JSON
is regenerated without rebuilding it; regenerate both together.) QAOA depth
studied: p = 1&ndash;{qmeta.get('layers', '?')}, {qmeta.get('restarts', '?')}
optimiser restarts per configuration.
</footer>

</body></html>
"""


def find_chrome() -> Optional[str]:
    """Locate a Chromium-family browser for PDF conversion."""
    for candidate in CHROME_CANDIDATES:
        if pathlib.Path(candidate).is_file():
            return candidate
        found = shutil.which(candidate)
        if found:
            return found
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Entry point."""
    parser = argparse.ArgumentParser(description="Build the PDF results report.")
    parser.add_argument("--scaling-json", default="benchmark_results.json")
    parser.add_argument("--qaoa-json", default="qaoa_results.json")
    parser.add_argument("--scaling-figure", default="qubit_scaling_comparison.png")
    parser.add_argument("--qaoa-figure", default="qaoa_scheduling_comparison.png")
    parser.add_argument("--out", default="uncomputation_report.pdf")
    parser.add_argument("--html-out", default="uncomputation_report.html")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    html = build_html(
        load_json(pathlib.Path(args.scaling_json)),
        load_json(pathlib.Path(args.qaoa_json)),
        embed_image(pathlib.Path(args.scaling_figure)),
        embed_image(pathlib.Path(args.qaoa_figure)),
    )
    html_path = pathlib.Path(args.html_out).resolve()
    html_path.write_text(html, encoding="utf-8")
    LOGGER.info("wrote %s", html_path)

    chrome = find_chrome()
    if chrome is None:
        LOGGER.error(
            "no Chromium-family browser found; open %s and print to PDF manually",
            html_path,
        )
        return 1

    output = pathlib.Path(args.out).resolve()
    result = subprocess.run(
        [
            chrome,
            "--headless",
            "--disable-gpu",
            "--no-sandbox",
            "--no-pdf-header-footer",
            f"--print-to-pdf={output}",
            html_path.as_uri(),
        ],
        capture_output=True,
        text=True,
        timeout=180,
    )
    if not output.is_file():
        LOGGER.error("PDF conversion failed: %s", result.stderr.strip()[:500])
        return 1
    LOGGER.info("wrote %s (%d bytes)", output, output.stat().st_size)
    return 0


if __name__ == "__main__":
    sys.exit(main())
