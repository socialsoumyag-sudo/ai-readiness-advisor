"""
Rules Engine — Deterministic Readiness Logic
-----------------------------------------------
Deliberately NOT an LLM call. Many of the "should we build an agent / do we
need guardrails" questions have objective, rule-based answers once you know
a few facts about the project. Spending an API call on questions that a
decision tree can answer for free is exactly the kind of over-engineering
this tool exists to prevent.

The LLM (see advisor/llm_advisor.py) is reserved for the genuinely
judgment-dependent parts: given a free-text description, does THIS specific
use case need an agent, and are there nuances the rules engine can't see.
"""

from dataclasses import dataclass, field
from enum import Enum


class DataSensitivity(str, Enum):
    PUBLIC = "public"
    INTERNAL = "internal"
    CONFIDENTIAL = "confidential"
    REGULATED = "regulated"  # PII, PHI, financial, legally protected


class InputTrust(str, Enum):
    TRUSTED_INTERNAL_ONLY = "trusted_internal_only"
    TRUSTED_EMPLOYEES = "trusted_employees"
    UNTRUSTED_EXTERNAL = "untrusted_external"  # customers, public web, third parties


class TaskComplexity(str, Enum):
    SINGLE_STEP = "single_step"          # one prompt -> one answer
    MULTI_STEP_FIXED = "multi_step_fixed"  # known sequence of steps, no branching
    MULTI_STEP_DYNAMIC = "multi_step_dynamic"  # model must decide next steps, use tools


@dataclass
class ProjectProfile:
    """Structured facts about a proposed AI project -- either filled in
    directly via a form, or extracted from free text by the LLM advisor."""
    data_sensitivity: DataSensitivity
    input_trust: InputTrust
    task_complexity: TaskComplexity
    expected_daily_volume: int
    is_user_facing: bool
    has_existing_deterministic_solution: bool = False  # could rules/regex/traditional code solve this already?


@dataclass
class ReadinessFinding:
    category: str
    verdict: str
    reasoning: str
    severity: str  # "info", "caution", "blocker"


def assess_architecture_fit(profile: ProjectProfile) -> ReadinessFinding:
    if profile.has_existing_deterministic_solution:
        return ReadinessFinding(
            category="Architecture Fit",
            verdict="Reconsider using an LLM at all",
            reasoning="A deterministic solution already exists or is feasible. LLMs add cost, "
                      "latency, and non-determinism -- justify the switch with a specific "
                      "capability gap the deterministic approach can't close, not just novelty.",
            severity="caution",
        )

    if profile.task_complexity == TaskComplexity.SINGLE_STEP:
        return ReadinessFinding(
            category="Architecture Fit",
            verdict="Agent NOT needed -- single prompt/response is sufficient",
            reasoning="The task is one well-defined transformation (classify, summarize, extract, "
                      "generate). An agent adds planning/tool-use overhead with no benefit here.",
            severity="info",
        )
    elif profile.task_complexity == TaskComplexity.MULTI_STEP_FIXED:
        return ReadinessFinding(
            category="Architecture Fit",
            verdict="Agent likely NOT needed -- a fixed pipeline of LLM calls will do",
            reasoning="The steps and their order are known in advance. A scripted sequence of "
                      "prompts (like a pipeline) gives the same result with far more predictability "
                      "and easier debugging than a full agent.",
            severity="info",
        )
    else:  # MULTI_STEP_DYNAMIC
        return ReadinessFinding(
            category="Architecture Fit",
            verdict="Agent architecture is justified",
            reasoning="The system must decide its own next steps based on intermediate results "
                      "(tool selection, dynamic branching). This is the genuine agent use case -- "
                      "but still scope it to the minimum tools/steps needed.",
            severity="info",
        )


def assess_guardrail_need(profile: ProjectProfile) -> ReadinessFinding:
    if profile.input_trust == InputTrust.UNTRUSTED_EXTERNAL:
        severity = "blocker" if profile.is_user_facing else "caution"
        return ReadinessFinding(
            category="Guardrails",
            verdict="Guardrails are REQUIRED, not optional",
            reasoning="Untrusted external input (customers, public web, third-party data) means "
                      "prompt injection and adversarial input are realistic threats, not theoretical "
                      "ones. Budget for this from day one -- retrofitting guardrails after an "
                      "incident costs far more than building them in.",
            severity=severity,
        )
    elif profile.input_trust == InputTrust.TRUSTED_EMPLOYEES:
        return ReadinessFinding(
            category="Guardrails",
            verdict="Lightweight guardrails recommended",
            reasoning="Internal users are lower risk but not zero risk (accidental injection via "
                      "copy-pasted content, disgruntled insiders). Basic input validation is "
                      "proportionate; heavy defensive engineering may not be cost-justified yet.",
            severity="caution",
        )
    else:
        return ReadinessFinding(
            category="Guardrails",
            verdict="Minimal guardrails likely sufficient",
            reasoning="Fully trusted, internal-only input significantly reduces (not eliminates) "
                      "injection risk. Still validate output format/schema, but heavy adversarial "
                      "defense is likely disproportionate cost for this risk profile.",
            severity="info",
        )


def assess_privacy_requirements(profile: ProjectProfile) -> ReadinessFinding:
    if profile.data_sensitivity == DataSensitivity.REGULATED:
        return ReadinessFinding(
            category="Privacy",
            verdict="Pseudonymization/redaction REQUIRED before any external API call",
            reasoning="Regulated data (PII, PHI, financial records) carries legal exposure. "
                      "Real identities and sensitive fields must never reach a third-party model "
                      "unredacted. This is a compliance requirement, not a nice-to-have.",
            severity="blocker",
        )
    elif profile.data_sensitivity == DataSensitivity.CONFIDENTIAL:
        return ReadinessFinding(
            category="Privacy",
            verdict="Pseudonymization strongly recommended",
            reasoning="Confidential business data (strategy, unreleased financials, internal "
                      "conflicts) isn't legally regulated but is commercially sensitive. Local "
                      "redaction before API calls reduces exposure at low engineering cost.",
            severity="caution",
        )
    else:
        return ReadinessFinding(
            category="Privacy",
            verdict="Standard data handling likely sufficient",
            reasoning="Public or general-internal data carries low exposure risk. Still confirm "
                      "with legal/compliance if uncertain -- 'probably fine' isn't a policy.",
            severity="info",
        )


def run_rules_engine(profile: ProjectProfile) -> list[ReadinessFinding]:
    """Runs all deterministic checks. No LLM call involved -- this is
    plain Python logic, intentionally, since these questions have
    objective answers once the facts are known."""
    return [
        assess_architecture_fit(profile),
        assess_guardrail_need(profile),
        assess_privacy_requirements(profile),
    ]


if __name__ == "__main__":
    # Example: a customer-facing chatbot handling regulated financial data,
    # with dynamic tool use -- should trigger blockers/cautions across the board.
    profile = ProjectProfile(
        data_sensitivity=DataSensitivity.REGULATED,
        input_trust=InputTrust.UNTRUSTED_EXTERNAL,
        task_complexity=TaskComplexity.MULTI_STEP_DYNAMIC,
        expected_daily_volume=5000,
        is_user_facing=True,
    )
    for finding in run_rules_engine(profile):
        print(f"[{finding.severity.upper()}] {finding.category}: {finding.verdict}")
        print(f"  → {finding.reasoning}\n")
