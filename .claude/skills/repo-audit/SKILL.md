---
name: repo-audit
description: >-
  Adversarial multi-agent audit of an entire repository — given only a GitHub URL
  (or the current working directory) — that tests whether the code actually works,
  whether its science, mathematics and algorithms are correct, and whether its
  headline results are real rather than hard-coded, faked, tuned, leaked or
  copied from an earlier run. Clones fresh, reproduces from zero, re-derives every
  claimed number from a live run, mutation-tests the test suite, sweeps every
  magic number, hunts data leakage and circular validation, then adversarially
  verifies each finding before reporting. Use this WHENEVER asked to audit, test,
  review, validate, verify, check, sanity-check or "see if everything is right"
  in a repository, project or codebase — especially scientific, numerical,
  simulation, benchmark, ML or research code, and especially before publishing,
  submitting, open-sourcing or trusting results. Trigger on a bare GitHub link
  plus any of "test this", "audit this", "check everything", "is this correct",
  "find the bugs", "is this hard-coded", "are these results real".
---

# Repository audit (adversarial, multi-agent)

**The failure this exists to prevent:** code that runs, passes its own tests,
produces plausible numbers, and is *wrong*. That is a worse outcome than code
that crashes, because it gets believed — and it gets believed by the person who
wrote it first.

A single-pass review cannot find this. One agent asked "does this work?" finds
reasons it works; confirmation bias is not a character flaw here, it is the
default behaviour of a read-and-summarise loop. The fan-out below is the fix.

## Invocation

The user supplies a repo URL, a path, or nothing (meaning the current directory).
If they add context — the field, or what they care about most — use it to sharpen
Phase 2's dimension J. If they do not, proceed anyway; Phase 1 discovers it.

**Announce the plan and the rough agent count before spawning anything.** A fan-out
costs real tokens; the user should see the shape first. If the repo turns out to be
tiny or trivial, say so and audit it inline instead of spawning a fleet.

## Ground rules for every agent

- **Verify by executing code and reading real output.** Never by reading a
  comment, a docstring or a README and agreeing with it. Documentation is a
  claim to be tested, not evidence.
- **Try to refute before accepting.** "I attacked this and could not break it" and
  "I read it and it looked fine" are different results. Say which one you have.
- **Report what you could NOT check, and why.** Absence of evidence is data.
  Manufacturing evidence is not.
- **Never assert a defect without a reproduction.**
- Trust nothing already on the machine: no existing virtualenv, no cached output,
  no committed figure or result file that was not regenerated during this audit.

---

## Phase 1 — Reconnaissance (before any fan-out)

Clone fresh into a temporary directory. Send a small number of agents to establish:

1. **What does this repo claim?** Extract every falsifiable claim from README,
   docstrings, reports, papers and comments as a numbered list. A claim is a
   number, a comparison ("X outperforms Y"), a mechanism, a complexity assertion,
   or a correctness guarantee. Mark which are **load-bearing** — the ones the repo
   exists to support.
2. **Stack and entry points.** Language, dependency manager, pinning, how to run
   it, test framework, expected runtime, what artefacts are produced.
3. **The critical path.** Which files actually compute the headline results,
   versus scaffolding, plotting and utilities. Audit weight belongs on the
   critical path.
4. **Domain and methods.** What field, which specific methods — so the correctness
   agents know what standard to hold the code to.

Report this map to the user before continuing. If there are no discernible claims,
say so and audit for correctness and reproducibility only.

---

## Phase 2 — Fan out, one or more agents per dimension

Run concurrently. Each agent reads the code fresh.

**A. Reproducibility from zero.** Fresh environment, install exactly what is
pinned, run what the docs say. Does it work on a clean machine? Are versions
pinned or floating? Run the pipeline twice and diff — any nondeterminism must be
explained (seed, threading, hardware) or reported. Flag anything the docs
mis-state, including runtime.

**B. Claims versus reality.** Take Phase 1's claim list and re-derive **every**
number from a live run. Flag: numbers that cannot be reproduced; numbers that
drifted from an older run; claims stated more strongly than the evidence allows;
comparisons with no baseline; claims with no supporting artefact at all.

**C. Domain correctness.** Verify the science, mathematics and algorithms against
the standard of the field, **from first principles — not from the repo's own
explanation of itself**. Conventions and units; conserved quantities and
invariants; boundary and degenerate cases; stated formulas versus what the code
implements. Where a method or paper is cited, confirm the code implements *that*
method. Flag anything unsourced or invented.

**D. Hard-coding, faked results and tuned parameters.** Three distinct checks,
all mandatory:

1. **Is anything faked?** Trace every headline number to the line that produces
   it. Confirm it comes from a real computation — not a hard-coded constant, not
   a closed-form shortcut dressed up as a measurement, not a value copied from an
   earlier run. **Decisive test: short-circuit or delete the core computation and
   see whether the reported number changes. If it does not, it was never
   computed.** Actually run that test.
2. **Are expected answers baked into the path that produces them?** Ground truth,
   known solutions, expected-output tables visible to code that is meant to
   *produce* an answer rather than *score* one. Reference data is legitimate as
   scoring data; it is a defect the moment the prediction path can see it.
3. **Are parameters tuned so the result comes out right?** Enumerate every magic
   number, threshold, cutoff, tolerance, seed and hyperparameter on the critical
   path. For each: does the headline conclusion depend on it? Sweep the ones that
   do. A conclusion holding at only one setting is a **tuned** result and must be
   labelled as one.

**E. Leakage and circular validation.** Does information from the answer reach the
thing predicting it? Train/test contamination; fitting and scoring on the same
data; a "reference model" not actually independent of what it validates; a metric
computed from the quantity being predicted; an approximation validated only in a
regime where it is not actually used.

**F. Statistical honesty.** Is there a proper baseline, including the trivial one
(random, constant, or the obvious classical method)? Is variance reported, or is a
single run presented as a result? Best-of-N reported as typical? Cherry-picked
seed, split, subset or window? Multiple comparisons uncorrected? Were the
significance criteria fixed before or after seeing the numbers?

**G. Code defects and numerical stability.** Off-by-one and indexing errors,
in-place mutation of inputs, aliasing, silent coercion, ignored returns, unhandled
failure paths. Numerically: conditioning, near-degenerate eigenvalues,
catastrophic cancellation, tiny denominators, accumulation, precision assumptions.
Does it fail loudly, or quietly turn `NaN` into `0` and continue?

**H. Test-suite quality — mutation test it.** Do the tests check *properties*, or
assert hard-coded numbers copied from a previous run? Was any tolerance widened
until it passed? Do they cover the critical path? **Introduce a deliberate defect
— a sign flip, an off-by-one, a dropped term — and confirm the suite catches it.**
Tests that pass on broken code are worse than no tests, and this is the most
common way a repo looks validated while being wrong.

**I. Missing baseline.** Would a simpler method get the same result? If a trivial
approach matches the sophisticated one, that is the headline finding, not a
footnote.

**J. Repo-specific dimensions.** From Phase 1: unvalidated approximations, unusual
claims, anything the authors flagged as uncertain, anything that looks too good.
An approximation that is validated in one regime and *used* in another is the
single highest-value thing to attack — go straight at it.

---

## Phase 3 — Adversarially verify every finding

Never report a finding because one agent believes it. For each candidate, spawn
independent verifiers instructed to **refute** it, defaulting to "refuted" when
uncertain. Where a finding could fail in more than one way, give each verifier a
different lens rather than running identical refuters. Kill what does not survive.

A false finding costs more credibility than a missed one.

Separate **confirmed** (reproduced, with a concrete failing case) from
**plausible** (argued but not demonstrated), and label every finding accordingly.

---

## Phase 4 — One consolidated report

```
BLOCKING        — claims that are wrong, unsupported, faked, tuned, leaked, or
                  resting on an unvalidated assumption. Ordered by how badly each
                  damages the repo's conclusions. For each: the claim, the
                  evidence against it, and a concrete reproduction.
REAL BUT MINOR  — genuine defects that change no conclusion.
VERIFIED SOUND  — what was actively attacked and survived. Say what you attacked,
                  or this section means nothing.
NOT CHECKED     — with the reason: no access, too expensive, out of scope.
FIXES           — ordered by value, cheapest high-impact first.
```

Do not soften a verdict to be agreeable. If a load-bearing claim is broken, say so
plainly and early — discovering it now is cheap; a reviewer discovering it is not.
Calibrate every statement to the evidence behind it.

**Diagnose only. Do not fix anything unless the user asks.** They should see the
findings before any code changes.

---

## Scaling the audit

Match the fan-out to the repo and the stakes.

| Situation | Shape |
|---|---|
| Small repo, quick sanity check | dimensions A, B, D, G inline or 3–4 agents |
| Normal audit | the full A–J fan-out, single-vote verification |
| Pre-publication / "is this real" | full fan-out, 3-vote adversarial verification, plus a completeness critic asking what was missed |

If the audit bounds its own coverage — sampling files, capping runtime, skipping a
slow test — **say so explicitly in the report.** Silent truncation reads as full
coverage when it was not.

## How this composes

- `scientific-coding` governs how *new* code should be written; this skill judges
  code that already exists.
- `scientific-fact-check` verifies claims about the outside world against the
  literature; this skill verifies claims a repo makes about itself.
- Where a repo has its own project skill, that skill is authoritative on project
  facts — but a project skill asserting something the code contradicts is itself
  a finding worth reporting.
