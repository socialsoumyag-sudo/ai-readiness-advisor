"""
Streamlit Dashboard — AI Implementation Readiness Advisor
------------------------------------------------------------
Two input modes, as requested:
  1. Structured form -- direct, precise, no LLM call needed for the facts
  2. Free-text description -- one LLM call extracts the same facts + adds
     nuanced judgment a rigid form can't capture

Both paths converge on the same ProjectProfile -> same rules engine ->
same report generator. The LLM is used exactly once, only in mode 2,
only for the genuinely judgment-dependent part.
"""

import os
import sys

import streamlit as st

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

from rules_engine.engine import ProjectProfile, DataSensitivity, InputTrust, TaskComplexity
from cost_estimator.estimator import TransactionBaseline, PricingModel, ProjectionInputs, GrowthCurve, project_cost
from advisor.report_generator import generate_report, profile_from_llm_assessment
from privacy.redaction import prepare_for_llm
from model_comparison.catalog import MODEL_CATALOG, LAST_VERIFIED
from model_comparison.comparator import (TaskType, rank_models_by_fit, compute_self_host_breakeven,
                                          score_vendor_risk, build_recommendation)

st.set_page_config(page_title="AI Implementation Readiness Advisor", layout="wide")


def render_header():
    st.title("🧭 AI Implementation Readiness Advisor")
    st.caption(
        "Answers the questions an organization should face BEFORE starting an AI/agentic "
        "project — architecture fit, guardrail need, privacy requirements, and realistic "
        "cost projection. Deliberately simple: most of this runs as plain logic, not an LLM call."
    )


def render_cost_inputs():
    st.subheader("💰 Cost Projection Inputs")
    st.caption("All numbers here are plain math — no API call involved in this section.")

    col1, col2 = st.columns(2)
    with col1:
        input_tokens = st.number_input("Baseline input tokens (from ONE real call)", min_value=1, value=800)
        daily_start = st.number_input("Expected daily calls at launch", min_value=0, value=50)
        duration = st.number_input("Project duration (days)", min_value=1, value=90)
        input_price = st.number_input("Input price per million tokens ($)", min_value=0.0, value=3.00, step=0.5)
    with col2:
        output_tokens = st.number_input("Baseline output tokens (from ONE real call)", min_value=1, value=250)
        daily_end = st.number_input("Expected daily calls at steady state", min_value=0, value=500)
        creep_buffer = st.slider("Prompt-creep buffer (%)", 0, 100, 30) / 100
        output_price = st.number_input("Output price per million tokens ($)", min_value=0.0, value=15.00, step=0.5)

    curve_choice = st.selectbox("Growth curve", ["flat", "linear", "exponential"], index=1)

    if st.button("Calculate cost projection"):
        cost_inputs = ProjectionInputs(
            baseline=TransactionBaseline(input_tokens=int(input_tokens), output_tokens=int(output_tokens)),
            pricing=PricingModel(input_price_per_million=input_price, output_price_per_million=output_price),
            daily_volume_start=int(daily_start), daily_volume_end=int(daily_end),
            project_duration_days=int(duration), prompt_creep_buffer_pct=creep_buffer,
        )
        result = project_cost(cost_inputs, curve=GrowthCurve(curve_choice))
        st.session_state["cost_projection"] = result
        st.success("Cost projection calculated below, and will also be included automatically "
                   "if you generate a report from the Structured Form or Free-Text tab.")

    cost_projection = st.session_state.get("cost_projection")
    if cost_projection:
        st.divider()
        st.markdown("### 📊 Projection Results")
        col1, col2, col3 = st.columns(3)
        col1.metric("One-time dev/testing cost", f"${cost_projection.one_time_dev_cost}")
        col2.metric("Daily cost at launch", f"${cost_projection.recurring_daily_cost_start}")
        col3.metric("Daily cost at steady state", f"${cost_projection.recurring_daily_cost_end}")

        st.markdown(f"**Total projected cost over {duration} days: "
                    f"${cost_projection.total_projected_cost_over_duration}**")
        st.caption(f"Total tokens over duration: {cost_projection.total_tokens_over_duration:,}")

        with st.expander("Assumptions behind this projection"):
            for a in cost_projection.assumptions:
                st.markdown(f"- {a}")


def render_model_comparison():
    st.subheader("🔀 Multi-LLM Cost & Fit Comparison")
    st.caption(
        f"Prices last manually verified: **{LAST_VERIFIED}** — a periodically updated "
        "snapshot, not a live feed. Several tools already do real-time pricing "
        "aggregation well; this answers a different question: which model is actually "
        "worth it for this task, this volume, and this org's risk tolerance — not just "
        "which is cheapest per token. No API call involved in this section."
    )

    col1, col2 = st.columns(2)
    with col1:
        task_type = st.selectbox(
            "Task type",
            options=list(TaskType),
            format_func=lambda t: t.value.replace("_", " ").title(),
        )
        quality_bar = st.select_slider("Minimum quality bar", options=["low", "mid", "frontier"], value="mid")
    with col2:
        monthly_input = st.number_input("Estimated monthly input tokens", min_value=0, value=20_000_000,
                                         step=1_000_000, key="mc_input_tokens")
        monthly_output = st.number_input("Estimated monthly output tokens", min_value=0, value=8_000_000,
                                          step=1_000_000, key="mc_output_tokens")

    if st.button("Compare models", type="primary", key="model_comparison_submit"):
        results = rank_models_by_fit(
            catalog=MODEL_CATALOG,
            task_type=task_type,
            monthly_input_tokens=int(monthly_input),
            monthly_output_tokens=int(monthly_output),
            quality_bar=quality_bar,
        )
        st.session_state["model_comparison"] = results
        st.session_state["model_comparison_task"] = task_type
        if results:
            top_risk = score_vendor_risk(results[0].model.provider)
            st.session_state["model_recommendation"] = build_recommendation(results[0], top_risk, task_type)
        else:
            st.session_state["model_recommendation"] = None
        st.success("Comparison calculated below, and the top recommendation will also be included "
                   "automatically if you generate a report from the Structured Form or Free-Text tab.")

    results = st.session_state.get("model_comparison")
    if results:
        st.divider()
        if not results:
            st.warning("No models meet this quality bar. Try lowering it.")
        else:
            st.markdown("**Ranked by Fit Score** (blends quality tier, monthly cost, and context-window fit)")
            table_data = [
                {
                    "Model": r.model.display_name,
                    "Provider": r.model.provider,
                    "Quality Tier": r.model.quality_tier.value,
                    "Est. Monthly Cost ($)": r.monthly_cost,
                    "Fit Score": r.fit_score,
                    "Context OK": "✅" if r.context_ok else "⚠️ Too small for task",
                }
                for r in results
            ]
            st.dataframe(table_data, use_container_width=True, hide_index=True)

            top = results[0]
            st.success(
                f"**Recommended: {top.model.display_name}** — Fit Score {top.fit_score}, "
                f"est. ${top.monthly_cost}/month at this volume. {top.model.notes}"
            )

            st.markdown("#### Self-host breakeven")
            st.caption("Compares the top-ranked open-weight model's API price against self-hosting it.")
            open_weight_results = [r for r in results if r.model.open_weight]
            if open_weight_results:
                best_open = open_weight_results[0]
                be = compute_self_host_breakeven(
                    best_open.model,
                    current_monthly_tokens_millions=(monthly_input + monthly_output) / 1_000_000,
                )
                m1, m2 = st.columns(2)
                m1.metric("Blended API price ($/1M tokens)", f"${be.blended_api_price_per_million}")
                m2.metric("Breakeven volume (M tokens/mo)",
                          f"{be.breakeven_tokens_millions:.1f}" if be.breakeven_tokens_millions != float("inf") else "Never")
                st.info(be.verdict)
            else:
                st.caption("No open-weight model met the quality bar for this task — self-host comparison skipped.")

            st.markdown("#### Vendor risk")
            risk = score_vendor_risk(top.model.provider)
            st.warning(f"**{top.model.provider}** — {risk.risk_band} (composite {risk.composite_risk}/5). {risk.notes}")

            with st.expander("Assumptions behind this comparison"):
                st.markdown(
                    "- Pricing is manually-verified list pricing; does not account for prompt "
                    "caching, batch discounts, or negotiated enterprise rates.\n"
                    "- Self-host breakeven uses simple, editable infra-cost assumptions "
                    "(default: one on-demand GPU instance) — meant to anchor a planning "
                    "conversation, not replace a real infra estimate.\n"
                    "- Vendor risk scores are opinionated starting points for review, not a "
                    "substitute for legal/procurement due diligence."
                )


def render_structured_form():
    st.subheader("📋 Structured Assessment")

    col1, col2 = st.columns(2)
    with col1:
        data_sensitivity = st.selectbox(
            "Data sensitivity",
            [e.value for e in DataSensitivity],
            help="regulated = PII/PHI/financial/legally protected data",
        )
        task_complexity = st.selectbox(
            "Task complexity",
            [e.value for e in TaskComplexity],
            help="single_step = one prompt->answer. multi_step_dynamic = model decides its own next steps.",
        )
    with col2:
        input_trust = st.selectbox("Input trust level", [e.value for e in InputTrust])
        is_user_facing = st.checkbox("Is this user-facing (customers see it directly)?")

    has_deterministic = st.checkbox("Could a traditional rules/regex-based solution already handle this?")
    daily_volume = st.number_input("Expected daily transaction volume", min_value=1, value=100, key="form_volume")

    if st.button("Generate Report", type="primary", key="structured_submit"):
        profile = ProjectProfile(
            data_sensitivity=DataSensitivity(data_sensitivity),
            input_trust=InputTrust(input_trust),
            task_complexity=TaskComplexity(task_complexity),
            expected_daily_volume=int(daily_volume),
            is_user_facing=is_user_facing,
            has_existing_deterministic_solution=has_deterministic,
        )
        cost_projection = st.session_state.get("cost_projection")
        model_recommendation = st.session_state.get("model_recommendation")
        report = generate_report(profile, llm_nuance=None, cost_projection=cost_projection,
                                  model_recommendation=model_recommendation)
        st.session_state["report"] = report


def render_freetext_input():
    st.subheader("✍️ Free-Text Description")
    st.caption(
        "Describe your proposed project in plain language. Any proprietary terms you list "
        "below are redacted locally before anything is sent to the model."
    )

    description = st.text_area(
        "Project description",
        height=150,
        placeholder="e.g. We want a system that reads customer support tickets and decides "
                    "whether to auto-respond, escalate, or trigger a refund, pulling from our "
                    "order database and CRM as needed.",
    )
    proprietary_terms = st.text_input(
        "Proprietary terms to redact before sending (comma-separated, optional)",
        placeholder="ProjectPhoenix, InternalCRM, VendorX",
    )
    daily_volume = st.number_input("Expected daily transaction volume", min_value=1, value=100, key="freetext_volume")
    has_deterministic = st.checkbox("Could a traditional rules/regex-based solution already handle this?",
                                     key="freetext_deterministic")

    if st.button("Analyze & Generate Report", type="primary", key="freetext_submit"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            st.error("ANTHROPIC_API_KEY not set. Add it to your .env file first.")
            return
        if not description.strip():
            st.error("Please enter a project description.")
            return

        terms_list = [t.strip() for t in proprietary_terms.split(",")] if proprietary_terms else []
        redaction_result = prepare_for_llm(description, custom_terms=terms_list)

        if redaction_result["was_modified"]:
            st.info(f"Redacted before sending: {redaction_result['redacted_generic_types'] + redaction_result['redacted_custom_terms']}")
        with st.expander("Preview exactly what will be sent to the model"):
            st.code(redaction_result["safe_text"])

        with st.spinner("Analyzing project description..."):
            from advisor.llm_advisor import get_llm_assessment
            llm_result = get_llm_assessment(redaction_result["safe_text"])

        if "error" in llm_result:
            st.error(f"Assessment failed: {llm_result['error']}")
            return

        profile = profile_from_llm_assessment(llm_result, expected_daily_volume=int(daily_volume),
                                                has_existing_deterministic_solution=has_deterministic)
        cost_projection = st.session_state.get("cost_projection")
        model_recommendation = st.session_state.get("model_recommendation")
        report = generate_report(profile, llm_nuance=llm_result, cost_projection=cost_projection,
                                  model_recommendation=model_recommendation)
        st.session_state["report"] = report

        if "_usage" in llm_result:
            usage = llm_result["_usage"]
            st.caption(f"This assessment call used {usage['input_tokens']} input / "
                       f"{usage['output_tokens']} output tokens — a real baseline you could "
                       f"plug into the cost projector above for a similarly-scoped project.")


def render_report():
    report = st.session_state.get("report")
    if report:
        st.divider()
        st.markdown(report)


def render_philosophy():
    st.subheader("🎯 Why This Tool Is Deliberately Simple")
    st.markdown("""
This tool practices what it preaches. Most of the "should we build an agent" question
has an objective answer once a few facts are known — so **most of this runs as plain
Python logic, not an LLM call.**

The LLM is used exactly **once**, only when you choose free-text input, only for the
genuinely judgment-dependent part: reading a loose description and extracting structured
facts + offering nuance a rigid form can't capture.

**What this means in practice:**
- The structured-form path makes **zero API calls** — completely free, completely deterministic
- The free-text path makes **exactly one API call** — not a multi-step agent, not repeated
  back-and-forth, one classification-style call
- Every number in the cost projection is plain arithmetic with stated assumptions — never
  an LLM guessing at its own cost

If a tool that recommends "don't build an agent unless you need one" built itself as an
unnecessary agent, that would be a contradiction. This is the alternative: use AI exactly
where it earns its cost, and plain code everywhere else.
""")


def main():
    render_header()

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Structured Form", "✍️ Free-Text Input",
                                              "💰 Cost Projection", "🔀 Model Comparison",
                                              "🎯 Philosophy"])

    with tab3:
        render_cost_inputs()

    with tab1:
        render_structured_form()

    with tab2:
        render_freetext_input()

    with tab4:
        render_model_comparison()

    with tab5:
        render_philosophy()

    render_report()


if __name__ == "__main__":
    main()
