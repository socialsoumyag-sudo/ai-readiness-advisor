"""
LLM Advisor — The ONE Judgment-Dependent Call
-------------------------------------------------
Everything the rules engine can answer deterministically, it already has
(see rules_engine/engine.py). This module handles the one thing rules
can't: reading a free-text project description and (a) extracting the
structured facts a form would have asked for, and (b) offering nuanced
judgment a decision tree can't fully capture (e.g. "this SOUNDS like it
needs an agent, but the actual variability described is low, so a fixed
pipeline would likely suffice").

Deliberately ONE call, not a multi-step agent -- extracting facts and
giving a judgment call is a single well-defined transformation, not a
task requiring autonomous planning or tool use. This module IS the
architecture recommendation this tool would make about itself.
"""

import json
import time
import anthropic

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are an AI-implementation readiness advisor. You receive a \
free-text description of a proposed AI/agentic project from an organization (already \
redacted of PII/proprietary terms before reaching you). Your job:

1. Extract structured facts implied by the description:
   - data_sensitivity: one of "public", "internal", "confidential", "regulated"
   - input_trust: one of "trusted_internal_only", "trusted_employees", "untrusted_external"
   - task_complexity: one of "single_step", "multi_step_fixed", "multi_step_dynamic"
   - is_user_facing: true/false

2. Give a brief, nuanced judgment (2-4 sentences) on architecture fit that a rigid
   decision tree might miss -- e.g. the description sounds complex but the actual
   decision-making required is narrow, or vice versa. Be willing to say "this is
   simpler than it sounds" as often as "this needs more rigor than it sounds like."
   Do not default to recommending agents or heavy architecture out of enthusiasm --
   err toward the simplest approach that would work, and say so explicitly if that's
   your assessment.

3. Note any risk or consideration the structured facts alone wouldn't surface
   (1-2 sentences, or empty string if nothing notable).

Respond with ONLY a JSON object, no markdown fences, no preamble:
{
  "data_sensitivity": "...",
  "input_trust": "...",
  "task_complexity": "...",
  "is_user_facing": true/false,
  "nuanced_judgment": "...",
  "additional_consideration": "..."
}"""


def get_llm_assessment(redacted_description: str, max_retries: int = 3) -> dict:
    client = anthropic.Anthropic()
    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=MODEL,
                max_tokens=600,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user",
                           "content": f"<project_description>{redacted_description}</project_description>"}],
            )
            raw = response.content[0].text.strip()
            raw = raw.replace("```json", "").replace("```", "").strip()
            parsed = json.loads(raw)

            required_keys = {"data_sensitivity", "input_trust", "task_complexity",
                              "is_user_facing", "nuanced_judgment", "additional_consideration"}
            if not required_keys.issubset(parsed.keys()):
                raise ValueError(f"Missing expected keys: {required_keys - parsed.keys()}")

            # Capture actual token usage for the cost estimator's baseline --
            # this is the "measure one real call" step the estimator module requires.
            parsed["_usage"] = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            return parsed

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            last_error = str(e)
            time.sleep(0.5 * (attempt + 1))
        except anthropic.APIError as e:
            last_error = f"API error: {e}"
            time.sleep(1.0 * (attempt + 1))

    return {"error": last_error or "Unknown failure after retries"}


if __name__ == "__main__":
    sample = ("We want a system that reads customer support tickets and decides whether "
              "to auto-respond, escalate to a human, or trigger a refund workflow, pulling "
              "from our order database and CRM as needed.")
    result = get_llm_assessment(sample)
    print(json.dumps(result, indent=2))
