"""
Token & Cost Estimator
------------------------
Deterministic math, not an LLM guess. The methodology:

  1. Baseline a single real transaction (input + output tokens) -- this
     MUST come from an actual API call's usage field, not an estimate.
     Estimating estimates compounds error.
  2. Project volume over the schedule using a growth curve, not a flat number.
  3. Separate one-time (dev/testing) spend from recurring (production) spend.
  4. Apply a "prompt creep" buffer -- real prompts grow over a project's
     life as more context/examples/history get added. Default 30%, adjustable.
  5. Re-baseline at each testing milestone rather than trusting one number
     projected at the start.

Model pricing changes over time and varies by provider/tier -- this module
takes pricing as an explicit input rather than hardcoding it, specifically
so estimates don't go stale as prices change.
"""

from dataclasses import dataclass
from enum import Enum


class GrowthCurve(str, Enum):
    FLAT = "flat"              # constant volume from day 1
    LINEAR = "linear"          # steady ramp-up
    EXPONENTIAL = "exponential"  # viral/compounding adoption


@dataclass
class TransactionBaseline:
    """Measured from ONE real API call -- input_tokens and output_tokens
    should come directly from the API response's usage field."""
    input_tokens: int
    output_tokens: int


@dataclass
class PricingModel:
    """Per-million-token pricing. Pass current rates explicitly --
    this module deliberately does not hardcode prices, since they change."""
    input_price_per_million: float
    output_price_per_million: float


@dataclass
class ProjectionInputs:
    baseline: TransactionBaseline
    pricing: PricingModel
    daily_volume_start: int
    daily_volume_end: int          # for linear/exponential growth; equals start for flat
    project_duration_days: int
    prompt_creep_buffer_pct: float = 0.30  # default: expect 30% growth in per-call token usage over time
    dev_testing_calls_estimate: int = 500   # one-time calls during build/test phase


@dataclass
class CostProjection:
    one_time_dev_cost: float
    recurring_daily_cost_start: float
    recurring_daily_cost_end: float
    total_projected_cost_over_duration: float
    total_tokens_over_duration: int
    assumptions: list


def _volume_on_day(day: int, inputs: ProjectionInputs, curve: GrowthCurve) -> int:
    if curve == GrowthCurve.FLAT:
        return inputs.daily_volume_start
    elif curve == GrowthCurve.LINEAR:
        if inputs.project_duration_days <= 1:
            return inputs.daily_volume_start
        fraction = day / (inputs.project_duration_days - 1)
        return int(inputs.daily_volume_start +
                   (inputs.daily_volume_end - inputs.daily_volume_start) * fraction)
    else:  # EXPONENTIAL
        if inputs.daily_volume_start <= 0:
            return inputs.daily_volume_end if day == inputs.project_duration_days - 1 else 0
        ratio = (inputs.daily_volume_end / inputs.daily_volume_start) ** (1 / max(inputs.project_duration_days - 1, 1))
        return int(inputs.daily_volume_start * (ratio ** day))


def _cost_for_tokens(input_tokens: int, output_tokens: int, pricing: PricingModel) -> float:
    return (input_tokens / 1_000_000 * pricing.input_price_per_million +
            output_tokens / 1_000_000 * pricing.output_price_per_million)


def project_cost(inputs: ProjectionInputs, curve: GrowthCurve = GrowthCurve.LINEAR) -> CostProjection:
    buffered_input = int(inputs.baseline.input_tokens * (1 + inputs.prompt_creep_buffer_pct))
    buffered_output = int(inputs.baseline.output_tokens * (1 + inputs.prompt_creep_buffer_pct))

    # One-time dev/testing cost -- uses UNBUFFERED baseline, since this
    # happens early before prompt creep sets in.
    one_time_cost = (
        _cost_for_tokens(inputs.baseline.input_tokens, inputs.baseline.output_tokens, inputs.pricing)
        * inputs.dev_testing_calls_estimate
    )

    total_tokens = 0
    total_recurring_cost = 0.0
    for day in range(inputs.project_duration_days):
        volume = _volume_on_day(day, inputs, curve)
        day_input_tokens = volume * buffered_input
        day_output_tokens = volume * buffered_output
        total_tokens += day_input_tokens + day_output_tokens
        total_recurring_cost += _cost_for_tokens(day_input_tokens, day_output_tokens, inputs.pricing)

    cost_start = _cost_for_tokens(
        inputs.daily_volume_start * buffered_input,
        inputs.daily_volume_start * buffered_output,
        inputs.pricing,
    )
    cost_end = _cost_for_tokens(
        inputs.daily_volume_end * buffered_input,
        inputs.daily_volume_end * buffered_output,
        inputs.pricing,
    )

    assumptions = [
        f"Baseline measured from 1 real call: {inputs.baseline.input_tokens} input / "
        f"{inputs.baseline.output_tokens} output tokens.",
        f"Applied {inputs.prompt_creep_buffer_pct:.0%} prompt-creep buffer -> "
        f"{buffered_input} input / {buffered_output} output tokens assumed per call in production.",
        f"Volume growth modeled as {curve.value}: {inputs.daily_volume_start} -> "
        f"{inputs.daily_volume_end} calls/day over {inputs.project_duration_days} days.",
        f"Dev/testing phase assumed {inputs.dev_testing_calls_estimate} calls at UNBUFFERED baseline "
        f"(prompt creep hasn't happened yet during initial build).",
        "Pricing is caller-supplied and will go stale -- re-check current rates before trusting "
        "this projection for budget approval.",
    ]

    return CostProjection(
        one_time_dev_cost=round(one_time_cost, 2),
        recurring_daily_cost_start=round(cost_start, 2),
        recurring_daily_cost_end=round(cost_end, 2),
        total_projected_cost_over_duration=round(one_time_cost + total_recurring_cost, 2),
        total_tokens_over_duration=total_tokens,
        assumptions=assumptions,
    )


if __name__ == "__main__":
    # Example: a project starting at 100 calls/day, ramping to 2000/day
    # over a 90-day rollout, using illustrative pricing.
    inputs = ProjectionInputs(
        baseline=TransactionBaseline(input_tokens=800, output_tokens=300),
        pricing=PricingModel(input_price_per_million=3.00, output_price_per_million=15.00),
        daily_volume_start=100,
        daily_volume_end=2000,
        project_duration_days=90,
    )
    result = project_cost(inputs, curve=GrowthCurve.LINEAR)
    print(f"One-time dev/testing cost: ${result.one_time_dev_cost}")
    print(f"Recurring cost/day at start: ${result.recurring_daily_cost_start}")
    print(f"Recurring cost/day at end:   ${result.recurring_daily_cost_end}")
    print(f"Total projected cost over {inputs.project_duration_days} days: ${result.total_projected_cost_over_duration}")
    print(f"Total tokens over duration: {result.total_tokens_over_duration:,}")
    print("\nAssumptions:")
    for a in result.assumptions:
        print(f"  - {a}")
