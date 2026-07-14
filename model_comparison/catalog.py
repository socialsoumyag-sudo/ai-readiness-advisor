"""
model_comparison/catalog.py
----------------------------
Model pricing/quality/risk catalog, as typed dataclasses -- consistent with
cost_estimator's TransactionBaseline/PricingModel style rather than a loose
config file, so it's discoverable the same way the rest of this codebase is.

This is a manually-verified snapshot (see LAST_VERIFIED below), not a live
pricing feed. Update it periodically. Several tools already do real-time
pricing aggregation well; this project's job is the decision layer on top
of the price, not being the fastest feed.
"""

from dataclasses import dataclass, field
from enum import Enum


LAST_VERIFIED = "2026-07-14"


class QualityTier(str, Enum):
    FRONTIER = "frontier"
    MID = "mid"
    BUDGET_REASONING = "budget_reasoning"
    BUDGET = "budget"


# Higher = more capable, used only for scoring -- not a benchmark claim.
QUALITY_TIER_RANK = {
    QualityTier.FRONTIER: 4,
    QualityTier.MID: 3,
    QualityTier.BUDGET_REASONING: 2.5,
    QualityTier.BUDGET: 2,
}


@dataclass
class ModelPricing:
    id: str
    provider: str
    display_name: str
    quality_tier: QualityTier
    input_price_per_million: float
    output_price_per_million: float
    context_window_tokens: int
    open_weight: bool = False
    notes: str = ""


@dataclass
class VendorRiskProfile:
    provider: str
    lock_in: int  # 1 (low) - 5 (high)
    data_residency_flexibility: int  # 1 (low) - 5 (high) -- higher is BETTER here
    deprecation_cadence: int  # 1 (slow/stable) - 5 (fast/frequent churn)
    notes: str = ""


MODEL_CATALOG: list[ModelPricing] = [
    ModelPricing("claude-opus", "Anthropic", "Claude Opus (frontier tier)",
                 QualityTier.FRONTIER, 5.00, 25.00, 200_000,
                 notes="Best for hard reasoning, agentic tool-use, high-stakes judgment calls."),
    ModelPricing("claude-sonnet", "Anthropic", "Claude Sonnet (mid tier)",
                 QualityTier.MID, 3.00, 15.00, 200_000,
                 notes="Best price/quality balance for most production workloads; strong tool-use."),
    ModelPricing("gpt-5", "OpenAI", "GPT-5 (frontier tier)",
                 QualityTier.FRONTIER, 10.00, 30.00, 128_000,
                 notes="Frontier reasoning and multimodal; premium pricing."),
    ModelPricing("gpt-4-1", "OpenAI", "GPT-4.1 (mid tier)",
                 QualityTier.MID, 2.00, 8.00, 128_000,
                 notes="Solid mid-tier generalist, good coding performance."),
    ModelPricing("o4-mini", "OpenAI", "o4-mini (reasoning, budget)",
                 QualityTier.BUDGET_REASONING, 1.10, 4.40, 128_000,
                 notes="Budget-friendly reasoning model for narrower reasoning tasks."),
    ModelPricing("gemini-3-flash", "Google", "Gemini 3 Flash (mid tier)",
                 QualityTier.MID, 0.50, 3.00, 1_000_000,
                 notes="Huge context window; strong multimodal; good value."),
    ModelPricing("gemini-flash-lite", "Google", "Gemini Flash-Lite (budget)",
                 QualityTier.BUDGET, 0.10, 0.40, 1_000_000,
                 notes="Cheapest proprietary option with large context; simple tasks only."),
    ModelPricing("deepseek-v3", "DeepSeek", "DeepSeek V3.2 (open-weight, budget)",
                 QualityTier.BUDGET, 0.14, 0.28, 128_000, open_weight=True,
                 notes="Open-weight, cheap, competitive benchmark scores. Self-hostable."),
    ModelPricing("mistral-small", "Mistral", "Mistral Small (budget)",
                 QualityTier.BUDGET, 0.20, 0.60, 128_000, open_weight=True,
                 notes="Lightweight, cheap, EU-hosted option (data residency advantage)."),
    ModelPricing("llama-4-maverick", "Meta (hosted)", "Llama 4 Maverick (open-weight, budget)",
                 QualityTier.BUDGET, 0.15, 0.60, 128_000, open_weight=True,
                 notes="Open-weight; strong self-hosting candidate at scale."),
]

VENDOR_RISK: list[VendorRiskProfile] = [
    VendorRiskProfile("Anthropic", 2, 3, 2, "Enterprise contracts available; moderate multi-region support."),
    VendorRiskProfile("OpenAI", 3, 3, 3, "Fast model turnover means more frequent migration work."),
    VendorRiskProfile("Google", 2, 4, 2, "Strong regional data residency options via Vertex AI."),
    VendorRiskProfile("DeepSeek", 1, 5, 3, "Open-weight: self-host anywhere, no vendor lock-in on the model itself."),
    VendorRiskProfile("Mistral", 1, 5, 3, "EU-based; open-weight options; strong for data residency-sensitive orgs."),
    VendorRiskProfile("Meta (hosted)", 1, 5, 3, "Open-weight; hosted by many providers, reducing single-vendor risk."),
]


def get_vendor_risk(provider: str) -> VendorRiskProfile | None:
    return next((v for v in VENDOR_RISK if v.provider == provider), None)
