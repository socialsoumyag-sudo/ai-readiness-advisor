"""
Report Generator
-------------------
Combines rules_engine (deterministic) + llm_advisor (one judgment call) +
cost_estimator (deterministic math) into a single readable readiness report.

No new logic lives here beyond formatting -- this module's only job is
assembly and presentation, keeping each upstream module independently
testable and swappable.
"""

from dataclasses import asdict
from rules_engine.engine import ProjectProfile, DataSensitivity, InputTrust, TaskComplexity, run_rules_engine
from cost_estimator.estimator import ProjectionInputs, TransactionBaseline, PricingModel, GrowthCurve, project_cost


def profile_from_llm_assessment(llm_result: dict, expected_daily_volume: int,
                                  has_existing_deterministic_solution: bool = False) -> ProjectProfile:
    """Converts the LLM's extracted facts into the same ProjectProfile type
    the rules engine expects -- so free-text input and structured-form input
    both flow through the identical deterministic logic afterward."""
    return ProjectProfile(
        data_sensitivity=DataSensitivity(llm_result["data_sensitivity"]),
        input_trust=InputTrust(llm_result["input_trust"]),
        task_complexity=TaskComplexity(llm_result["task_complexity"]),
        expected_daily_volume=expected_daily_volume,
        is_user_facing=llm_result["is_user_facing"],
        has_existing_deterministic_solution=has_existing_deterministic_solution,
    )


def generate_report(profile: ProjectProfile, llm_nuance: dict = None,
                     cost_projection=None) -> str:
    findings = run_rules_engine(profile)

    lines = ["# AI Implementation Readiness Report\n"]

    severity_icon = {"blocker": "🔴", "caution": "🟡", "info": "🟢"}
    for f in findings:
        icon = severity_icon.get(f.severity, "•")
        lines.append(f"## {icon} {f.category}: {f.verdict}")
        lines.append(f"{f.reasoning}\n")

    if llm_nuance:
        lines.append("## 🧭 Nuanced Assessment")
        lines.append(llm_nuance.get("nuanced_judgment", ""))
        consideration = llm_nuance.get("additional_consideration", "")
        if consideration:
            lines.append(f"\n**Additional consideration:** {consideration}")
        lines.append("")

    if cost_projection:
        lines.append("## 💰 Cost Projection")
        lines.append(f"- One-time dev/testing cost: ${cost_projection.one_time_dev_cost}")
        lines.append(f"- Recurring cost/day (start of rollout): ${cost_projection.recurring_daily_cost_start}")
        lines.append(f"- Recurring cost/day (end of rollout): ${cost_projection.recurring_daily_cost_end}")
        lines.append(f"- **Total projected cost:** ${cost_projection.total_projected_cost_over_duration}")
        lines.append(f"- Total tokens over duration: {cost_projection.total_tokens_over_duration:,}\n")
        lines.append("**Assumptions behind this projection:**")
        for a in cost_projection.assumptions:
            lines.append(f"  - {a}")
        lines.append("")

    blockers = [f for f in findings if f.severity == "blocker"]
    if blockers:
        lines.append("## ⚠️ Bottom line")
        lines.append(f"{len(blockers)} blocking issue(s) identified. These should be resolved "
                      f"BEFORE implementation begins, not discovered during it.")
    else:
        lines.append("## ✅ Bottom line")
        lines.append("No blocking issues identified from the information provided. "
                      "Review the cautions above before proceeding.")

    return "\n".join(lines)


if __name__ == "__main__":
    profile = ProjectProfile(
        data_sensitivity=DataSensitivity.CONFIDENTIAL,
        input_trust=InputTrust.TRUSTED_EMPLOYEES,
        task_complexity=TaskComplexity.MULTI_STEP_FIXED,
        expected_daily_volume=200,
        is_user_facing=False,
    )

    cost_inputs = ProjectionInputs(
        baseline=TransactionBaseline(input_tokens=900, output_tokens=250),
        pricing=PricingModel(input_price_per_million=3.00, output_price_per_million=15.00),
        daily_volume_start=50,
        daily_volume_end=200,
        project_duration_days=60,
    )
    cost_result = project_cost(cost_inputs, curve=GrowthCurve.LINEAR)

    fake_llm_nuance = {
        "nuanced_judgment": "Although this involves several processing stages, the sequence is "
                            "fixed and known in advance -- a scripted pipeline of prompts will be "
                            "more predictable and easier to debug than a dynamic agent here.",
        "additional_consideration": "Confirm whether the confidential data classification includes "
                                     "any fields that might actually be regulated (e.g. embedded PII) "
                                     "before finalizing the privacy approach.",
    }

    report = generate_report(profile, llm_nuance=fake_llm_nuance, cost_projection=cost_result)
    print(report)
