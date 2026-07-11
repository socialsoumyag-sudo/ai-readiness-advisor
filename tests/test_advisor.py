"""
Unit tests -- rules engine, cost estimator, privacy redaction.
No API key required (the LLM advisor module is the only piece that needs one,
and it's tested separately/manually since asserting on LLM output is brittle).
Run with: python tests/test_advisor.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from rules_engine.engine import (
    ProjectProfile, DataSensitivity, InputTrust, TaskComplexity,
    run_rules_engine, assess_architecture_fit, assess_guardrail_need, assess_privacy_requirements,
)
from cost_estimator.estimator import (
    TransactionBaseline, PricingModel, ProjectionInputs, GrowthCurve, project_cost,
)
from privacy.redaction import redact_generic_pii, redact_custom_terms, prepare_for_llm


# ---------- Rules engine tests ----------

def test_single_step_never_recommends_agent():
    profile = ProjectProfile(
        data_sensitivity=DataSensitivity.PUBLIC, input_trust=InputTrust.TRUSTED_INTERNAL_ONLY,
        task_complexity=TaskComplexity.SINGLE_STEP, expected_daily_volume=10, is_user_facing=False,
    )
    finding = assess_architecture_fit(profile)
    assert "not needed" in finding.verdict.lower() or "NOT needed" in finding.verdict
    print("✅ test_single_step_never_recommends_agent")


def test_dynamic_multi_step_recommends_agent():
    profile = ProjectProfile(
        data_sensitivity=DataSensitivity.PUBLIC, input_trust=InputTrust.TRUSTED_INTERNAL_ONLY,
        task_complexity=TaskComplexity.MULTI_STEP_DYNAMIC, expected_daily_volume=10, is_user_facing=False,
    )
    finding = assess_architecture_fit(profile)
    assert "justified" in finding.verdict.lower()
    print("✅ test_dynamic_multi_step_recommends_agent")


def test_existing_deterministic_solution_overrides_complexity():
    profile = ProjectProfile(
        data_sensitivity=DataSensitivity.PUBLIC, input_trust=InputTrust.TRUSTED_INTERNAL_ONLY,
        task_complexity=TaskComplexity.MULTI_STEP_DYNAMIC, expected_daily_volume=10,
        is_user_facing=False, has_existing_deterministic_solution=True,
    )
    finding = assess_architecture_fit(profile)
    assert "reconsider" in finding.verdict.lower()
    print("✅ test_existing_deterministic_solution_overrides_complexity")


def test_untrusted_external_userfacing_is_blocker():
    profile = ProjectProfile(
        data_sensitivity=DataSensitivity.PUBLIC, input_trust=InputTrust.UNTRUSTED_EXTERNAL,
        task_complexity=TaskComplexity.SINGLE_STEP, expected_daily_volume=10, is_user_facing=True,
    )
    finding = assess_guardrail_need(profile)
    assert finding.severity == "blocker"
    print("✅ test_untrusted_external_userfacing_is_blocker")


def test_regulated_data_is_blocker():
    profile = ProjectProfile(
        data_sensitivity=DataSensitivity.REGULATED, input_trust=InputTrust.TRUSTED_INTERNAL_ONLY,
        task_complexity=TaskComplexity.SINGLE_STEP, expected_daily_volume=10, is_user_facing=False,
    )
    finding = assess_privacy_requirements(profile)
    assert finding.severity == "blocker"
    print("✅ test_regulated_data_is_blocker")


def test_public_data_is_low_severity():
    profile = ProjectProfile(
        data_sensitivity=DataSensitivity.PUBLIC, input_trust=InputTrust.TRUSTED_INTERNAL_ONLY,
        task_complexity=TaskComplexity.SINGLE_STEP, expected_daily_volume=10, is_user_facing=False,
    )
    finding = assess_privacy_requirements(profile)
    assert finding.severity == "info"
    print("✅ test_public_data_is_low_severity")


def test_run_rules_engine_returns_three_findings():
    profile = ProjectProfile(
        data_sensitivity=DataSensitivity.INTERNAL, input_trust=InputTrust.TRUSTED_EMPLOYEES,
        task_complexity=TaskComplexity.MULTI_STEP_FIXED, expected_daily_volume=50, is_user_facing=False,
    )
    findings = run_rules_engine(profile)
    assert len(findings) == 3
    categories = {f.category for f in findings}
    assert categories == {"Architecture Fit", "Guardrails", "Privacy"}
    print("✅ test_run_rules_engine_returns_three_findings")


# ---------- Cost estimator tests ----------

def test_flat_growth_produces_constant_daily_cost():
    inputs = ProjectionInputs(
        baseline=TransactionBaseline(input_tokens=1000, output_tokens=200),
        pricing=PricingModel(input_price_per_million=3.0, output_price_per_million=15.0),
        daily_volume_start=100, daily_volume_end=100, project_duration_days=10,
        prompt_creep_buffer_pct=0.0, dev_testing_calls_estimate=0,
    )
    result = project_cost(inputs, curve=GrowthCurve.FLAT)
    assert result.recurring_daily_cost_start == result.recurring_daily_cost_end
    print("✅ test_flat_growth_produces_constant_daily_cost")


def test_linear_growth_end_cost_higher_than_start():
    inputs = ProjectionInputs(
        baseline=TransactionBaseline(input_tokens=1000, output_tokens=200),
        pricing=PricingModel(input_price_per_million=3.0, output_price_per_million=15.0),
        daily_volume_start=50, daily_volume_end=500, project_duration_days=30,
    )
    result = project_cost(inputs, curve=GrowthCurve.LINEAR)
    assert result.recurring_daily_cost_end > result.recurring_daily_cost_start
    print("✅ test_linear_growth_end_cost_higher_than_start")


def test_prompt_creep_buffer_increases_cost():
    baseline = TransactionBaseline(input_tokens=1000, output_tokens=200)
    pricing = PricingModel(input_price_per_million=3.0, output_price_per_million=15.0)

    no_buffer = ProjectionInputs(baseline=baseline, pricing=pricing, daily_volume_start=100,
                                   daily_volume_end=100, project_duration_days=10,
                                   prompt_creep_buffer_pct=0.0, dev_testing_calls_estimate=0)
    with_buffer = ProjectionInputs(baseline=baseline, pricing=pricing, daily_volume_start=100,
                                     daily_volume_end=100, project_duration_days=10,
                                     prompt_creep_buffer_pct=0.5, dev_testing_calls_estimate=0)

    result_no_buffer = project_cost(no_buffer, curve=GrowthCurve.FLAT)
    result_with_buffer = project_cost(with_buffer, curve=GrowthCurve.FLAT)
    assert result_with_buffer.total_projected_cost_over_duration > result_no_buffer.total_projected_cost_over_duration
    print("✅ test_prompt_creep_buffer_increases_cost")


def test_cost_projection_includes_assumptions():
    inputs = ProjectionInputs(
        baseline=TransactionBaseline(input_tokens=500, output_tokens=100),
        pricing=PricingModel(input_price_per_million=3.0, output_price_per_million=15.0),
        daily_volume_start=10, daily_volume_end=10, project_duration_days=5,
    )
    result = project_cost(inputs, curve=GrowthCurve.FLAT)
    assert len(result.assumptions) >= 4, "Every projection must state its assumptions explicitly"
    print("✅ test_cost_projection_includes_assumptions")


# ---------- Privacy tests ----------

def test_redact_generic_pii_catches_email():
    text = "Contact john@company.com for details."
    redacted, found = redact_generic_pii(text)
    assert "john@company.com" not in redacted
    assert "email" in found
    print("✅ test_redact_generic_pii_catches_email")


def test_redact_custom_terms_catches_codename():
    text = "We plan to use ProjectPhoenix for this."
    redacted, found = redact_custom_terms(text, ["ProjectPhoenix"])
    assert "ProjectPhoenix" not in redacted
    assert "ProjectPhoenix" in found
    print("✅ test_redact_custom_terms_catches_codename")


def test_prepare_for_llm_combines_both():
    text = "Email me at a@b.com about ProjectPhoenix."
    result = prepare_for_llm(text, custom_terms=["ProjectPhoenix"])
    assert result["was_modified"] is True
    assert "a@b.com" not in result["safe_text"]
    assert "ProjectPhoenix" not in result["safe_text"]
    print("✅ test_prepare_for_llm_combines_both")


def test_prepare_for_llm_leaves_clean_text_unmodified():
    text = "We want to classify support tickets by urgency."
    result = prepare_for_llm(text)
    assert result["was_modified"] is False
    assert result["safe_text"] == text
    print("✅ test_prepare_for_llm_leaves_clean_text_unmodified")


def _run_all():
    tests = [
        test_single_step_never_recommends_agent,
        test_dynamic_multi_step_recommends_agent,
        test_existing_deterministic_solution_overrides_complexity,
        test_untrusted_external_userfacing_is_blocker,
        test_regulated_data_is_blocker,
        test_public_data_is_low_severity,
        test_run_rules_engine_returns_three_findings,
        test_flat_growth_produces_constant_daily_cost,
        test_linear_growth_end_cost_higher_than_start,
        test_prompt_creep_buffer_increases_cost,
        test_cost_projection_includes_assumptions,
        test_redact_generic_pii_catches_email,
        test_redact_custom_terms_catches_codename,
        test_prepare_for_llm_combines_both,
        test_prepare_for_llm_leaves_clean_text_unmodified,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except AssertionError as e:
            failed += 1
            print(f"❌ {t.__name__} FAILED: {e}")
        except Exception as e:
            failed += 1
            print(f"❌ {t.__name__} ERRORED: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    _run_all()
