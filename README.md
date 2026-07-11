# AI Implementation Readiness Advisor

**Answers the questions an organization should face BEFORE starting an AI or agentic project — architecture fit, guardrail necessity, privacy requirements, and a realistic cost projection. Deliberately simple: most of it runs as plain logic, not an LLM call.**

## The problem this solves

Most organizations approach AI adoption backwards: build first, discover the governance questions later — usually when a cost overrun, a security incident, or a compliance review forces the issue. This tool forces the questions to the front, before implementation starts, the same way a phase-gate review forces risk assessment before a programme milestone.

## The core idea — and the deliberate irony

A tool whose purpose is "should you build an agent, and do you need guardrails?" would be self-defeating if it answered that question by being an over-engineered agent itself. So this tool practices what it recommends:

- **The structured-form path makes zero API calls.** Architecture fit, guardrail necessity, and privacy requirements are answered by a plain decision tree (`rules_engine/engine.py`) — these questions have objective answers once a few facts are known, and spending tokens on them would be exactly the kind of waste this tool exists to flag.
- **The free-text path makes exactly one API call**, not a multi-step agent — reading a loose project description and extracting structured facts + offering nuance a rigid form can't capture is a single well-defined transformation, not a task requiring planning or tool use.
- **Every cost number is plain arithmetic** with explicitly stated assumptions — never an LLM guessing at its own cost.

## What it does

1. **Structured Form** — direct questions (data sensitivity, input trust, task complexity, user-facing or not) feed a deterministic rules engine that flags architecture mismatches, guardrail gaps, and privacy requirements — with severity levels (blocker / caution / info), not vague advice.
2. **Free-Text Input** — describe a project in plain language; proprietary terms you specify are redacted locally before anything reaches the model; one LLM call extracts the same structured facts plus a nuanced judgment call.
3. **Cost Projection** — given one real measured API call (input/output tokens) and expected volume growth (flat/linear/exponential) over a project timeline, produces a full cost projection with every assumption stated explicitly, separating one-time dev/testing spend from recurring production spend.
4. **Combined Report** — all three come together into a single readiness report with a clear bottom line: blocking issues that must be resolved before implementation, versus cautions to review.

## Architecture

```
Structured Form ──┐
                   ├──► ProjectProfile ──► rules_engine (NO LLM call)
Free-Text Input ───┘         │                    │
        │                    │                    ▼
        ▼                    │            [Architecture / Guardrails / Privacy findings]
  ONE LLM call                │                    │
  (redacted first)            │                    ▼
        │                     └──────────► report_generator ◄── cost_estimator (NO LLM call)
        ▼                                          │
  [structured facts +                              ▼
   nuanced judgment]                       Full Readiness Report
```

## Quickstart

```bash
git clone <this-repo>
cd ai-readiness-advisor
pip install -r requirements.txt
cp .env.example .env   # only needed for the Free-Text tab

streamlit run dashboard/app.py
```

The Structured Form and Cost Projection tabs work immediately with **no API key required** — try those first.

```bash
# Run all tests (no API key needed — 15 tests across rules engine, cost estimator, privacy)
python tests/test_advisor.py
```

## Why this is more than a wrapper

- **Judgment over enthusiasm**: the rules engine actively recommends *against* agent architecture and heavy guardrails when the facts don't justify them — it doesn't default to "more AI is better."
- **Cost transparency**: every projection states its assumptions in plain text (baseline source, growth curve, buffer percentage) so a reader can challenge the number, not just trust it.
- **Privacy by design**: proprietary terms are redacted locally before the one LLM call happens, using the same pattern as [the stakeholder-sentiment analyzer](#) this tool's author built previously.
- **Tested where testing is meaningful**: 15 unit tests cover the deterministic logic (rules engine branching, cost math, redaction) without asserting on non-deterministic LLM output.

## Scope — what this does *not* do (yet)

- The rules engine's decision tree reflects one reasonable framework, not a universally agreed standard — treat its verdicts as a structured starting point for a conversation, not a certification.
- Cost projections are only as good as the baseline measurement fed in — garbage in, garbage out, same as any estimation tool.
- No persistence/history — each assessment is a one-off session, not a tracked portfolio of past assessments (a natural v2 addition).
- Pricing must be manually kept current — the tool deliberately does not hardcode model prices, since they change.

## Tech stack

Python · Anthropic API (Claude) · Streamlit · pytest

## Background

Built by [Soumya Ghatak](https://linkedin.com/in/soumyaghatakiimb) — Senior Program/Transformation Manager, IIM Bangalore MBA, PMP®. Extends the privacy-by-design pattern from an earlier [stakeholder sentiment & roadmap risk analyzer](#), applied here to the pre-implementation decision layer: the questions an organization should answer before building an AI system, not just once it's already been built.
