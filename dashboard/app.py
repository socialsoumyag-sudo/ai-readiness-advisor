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
        report = generate_report(profile, llm_nuance=None, cost_projection=cost_projection)
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
        report = generate_report(profile, llm_nuance=llm_result, cost_projection=cost_projection)
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

    tab1, tab2, tab3, tab4 = st.tabs(["📋 Structured Form", "✍️ Free-Text Input",
                                        "💰 Cost Projection", "🎯 Philosophy"])

    with tab3:
        render_cost_inputs()

    with tab1:
        render_structured_form()

    with tab2:
        render_freetext_input()

    with tab4:
        render_philosophy()

    render_report()


if __name__ == "__main__":
    main()
