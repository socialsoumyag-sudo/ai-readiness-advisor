"""
model_comparison/comparator.py
--------------------------------
Answers three questions a generic per-token pricing calculator doesn't:

  1. Fit scoring     -- which model is worth it for THIS task, at THIS
                         volume (not just which is cheapest)
  2. Self-host        -- at what volume does self-hosting an open-weight
     breakeven            model beat paying per-token API pricing
  3. Vendor risk       -- lock-in / data residency / deprecation cadence,
                         because cheapest isn't automatically right for a
                         regulated org

All plain arithmetic -- consistent with this project's philosophy of using
an LLM call only where it earns its cost. No API calls happen in this module.
"""

from dataclasses import dataclass
from enum import Enum

from model_comparison.catalog import ModelPricing, QUALITY_TIER_RANK, QualityTier, get_vendor_risk


class TaskType(str, Enum):
    CHATBOT = "chatbot_conversational"
    SUMMARIZATION = "summarization"
    CODING_ASSISTANT = "coding_assistant"
    AGENTIC_TOOL_USE = "agentic_tool_use"
    BULK_CLASSIFICATION = "bulk_classification"
    COMPLEX_REASONING = "complex_reasoning"


@dataclass
class TaskProfile:
    quality_weight: float
    cost_weight: float
    min_context_tokens: int


TASK_PROFILES: dict[TaskType, TaskProfile] = {
    TaskType.CHATBOT: TaskProfile(0.40, 0.60, 8_000),
    TaskType.SUMMARIZATION: TaskProfile(0.35, 0.65, 32_000),
    TaskType.CODING_ASSISTANT: TaskProfile(0.65, 0.35, 64_000),
    TaskType.AGENTIC_TOOL_USE: TaskProfile(0.70, 0.30, 64_000),
    TaskType.BULK_CLASSIFICATION: TaskProfile(0.20, 0.80, 4_000),
    TaskType.COMPLEX_REASONING: TaskProfile(0.75, 0.25, 32_000),
}

QUALITY_BAR_FLOOR = {"low": 0, "mid": 2.5, "frontier": 4}


# ---------------------------------------------------------------------------
# 1. Fit scoring
# ---------------------------------------------------------------------------

@dataclass
class ModelFitResult:
    model: ModelPricing
    monthly_cost: float
    fit_score: float
    context_ok: bool


def estimate_monthly_cost(model: ModelPricing, monthly_input_tokens: int, monthly_output_tokens: int) -> float:
    input_cost = (monthly_input_tokens / 1_000_000) * model.input_price_per_million
    output_cost = (monthly_output_tokens / 1_000_000) * model.output_price_per_million
    return round(input_cost + output_cost, 2)


def rank_models_by_fit(
    catalog: list[ModelPricing],
    task_type: TaskType,
    monthly_input_tokens: int,
    monthly_output_tokens: int,
    quality_bar: str = "mid",
) -> list[ModelFitResult]:
    """
    Ranks candidate models by a blended fit_score (quality vs. cost vs.
    context fit) for the given task -- not by cost alone. Filters out
    models below the requested quality bar.
    """
    profile = TASK_PROFILES[task_type]
    quality_floor = QUALITY_BAR_FLOOR[quality_bar]

    costs = [estimate_monthly_cost(m, monthly_input_tokens, monthly_output_tokens) for m in catalog]
    max_cost = max(costs) if costs and max(costs) > 0 else 1.0

    results = []
    for model, cost in zip(catalog, costs):
        q_rank = QUALITY_TIER_RANK.get(model.quality_tier, 0)
        if q_rank < quality_floor:
            continue

        context_ok = model.context_window_tokens >= profile.min_context_tokens
        quality_norm = q_rank / 4
        cost_norm = 1 - (cost / max_cost)

        fit_score = profile.quality_weight * quality_norm + profile.cost_weight * cost_norm
        if not context_ok:
            fit_score *= 0.5

        results.append(ModelFitResult(model=model, monthly_cost=cost, fit_score=round(fit_score, 3),
                                       context_ok=context_ok))

    results.sort(key=lambda r: r.fit_score, reverse=True)
    return results


# ---------------------------------------------------------------------------
# 2. Self-host breakeven
# ---------------------------------------------------------------------------

@dataclass
class BreakevenResult:
    blended_api_price_per_million: float
    breakeven_tokens_millions: float
    verdict: str


def blended_price_per_million(model: ModelPricing, input_output_ratio: float = 3.0) -> float:
    """Blends input/output pricing assuming a typical output:input token ratio."""
    total_parts = input_output_ratio + 1
    return (model.output_price_per_million * input_output_ratio + model.input_price_per_million) / total_parts


def compute_self_host_breakeven(
    model: ModelPricing,
    monthly_infra_cost: float = 1500.0,
    marginal_cost_per_million: float = 0.02,
    input_output_ratio: float = 3.0,
    current_monthly_tokens_millions: float | None = None,
) -> BreakevenResult:
    """
    Finds the monthly token volume (millions) past which self-hosting this
    open-weight model is cheaper than paying its API price. Deliberately
    simple and editable -- meant to anchor a planning conversation, not
    replace a real infra estimate.
    """
    api_price = blended_price_per_million(model, input_output_ratio)

    if api_price <= marginal_cost_per_million:
        return BreakevenResult(
            blended_api_price_per_million=round(api_price, 4),
            breakeven_tokens_millions=float("inf"),
            verdict="API pricing is already at or below raw self-hosting marginal cost. "
                    "Self-hosting is unlikely to pay off at any volume.",
        )

    breakeven = monthly_infra_cost / (api_price - marginal_cost_per_million)

    if current_monthly_tokens_millions is not None:
        if current_monthly_tokens_millions >= breakeven:
            verdict = (f"At ~{current_monthly_tokens_millions:.1f}M tokens/month, self-hosting is likely "
                       f"CHEAPER than the API -- you're past breakeven (~{breakeven:.1f}M tokens/month).")
        else:
            multiple = breakeven / current_monthly_tokens_millions if current_monthly_tokens_millions else float("inf")
            verdict = (f"At ~{current_monthly_tokens_millions:.1f}M tokens/month, the API is likely still "
                       f"cheaper. Breakeven is ~{breakeven:.1f}M tokens/month (~{multiple:.1f}x current volume).")
    else:
        verdict = f"Self-hosting becomes cheaper past approximately {breakeven:.1f}M tokens/month."

    return BreakevenResult(
        blended_api_price_per_million=round(api_price, 4),
        breakeven_tokens_millions=round(breakeven, 2),
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# 3. Vendor risk
# ---------------------------------------------------------------------------

@dataclass
class RiskResult:
    provider: str
    composite_risk: float
    risk_band: str
    notes: str


def _risk_band(score: float) -> str:
    if score <= 2.0:
        return "Low risk"
    if score <= 3.5:
        return "Moderate risk"
    return "High risk"


def score_vendor_risk(provider: str) -> RiskResult:
    profile = get_vendor_risk(provider)
    if not profile:
        return RiskResult(provider=provider, composite_risk=0.0, risk_band="Unassessed",
                           notes="No risk profile on file for this provider -- not the same as zero risk.")

    composite = round((profile.lock_in + (6 - profile.data_residency_flexibility) + profile.deprecation_cadence) / 3, 2)
    return RiskResult(provider=provider, composite_risk=composite, risk_band=_risk_band(composite),
                       notes=profile.notes)


# ---------------------------------------------------------------------------
# 4. Report-ready recommendation
# ---------------------------------------------------------------------------

@dataclass
class ModelRecommendation:
    model_name: str
    provider: str
    monthly_cost: float
    fit_score: float
    risk_band: str
    reasoning: str


def build_recommendation(top: ModelFitResult, risk: RiskResult, task_type: TaskType) -> ModelRecommendation:
    """
    Turns the top-ranked ModelFitResult + its RiskResult into a single,
    report-ready recommendation with a human-readable "because" reasoning
    string -- so report_generator can drop this straight into a report
    without needing to know how fit scoring or risk scoring work.
    """
    task_label = task_type.value.replace("_", " ")
    reasoning = (
        f"Best fit score ({top.fit_score}) among models meeting the requested quality bar "
        f"for {task_label} at this volume, balancing capability against an estimated "
        f"${top.monthly_cost}/month spend. Vendor risk for {top.model.provider} is "
        f"assessed as {risk.risk_band.lower()} ({risk.composite_risk}/5)."
        + (f" Note: {risk.notes}" if risk.notes else "")
    )
    if not top.context_ok:
        reasoning += " Caution: this model's context window is smaller than recommended for this task type."

    return ModelRecommendation(
        model_name=top.model.display_name,
        provider=top.model.provider,
        monthly_cost=top.monthly_cost,
        fit_score=top.fit_score,
        risk_band=risk.risk_band,
        reasoning=reasoning,
    )
