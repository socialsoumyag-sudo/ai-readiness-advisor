"""
Privacy Module — Proprietary Information Handling
-----------------------------------------------------
Orgs describing a proposed AI project in free text will often mention
real product names, internal system names, vendor names, and sometimes
people's names. Same principle as the stakeholder-sentiment project:
redact/pseudonymize before anything reaches the LLM, never send raw
proprietary specifics unless the org explicitly opts in.

This is intentionally simpler than the stakeholder-sentiment version --
no stable cross-reference mapping is needed here, since each readiness
assessment is a one-off consultation, not an ongoing multi-person
conversation to track. A lighter-weight, single-pass redaction is
proportionate here (matches the tool's own philosophy: don't build more
than the problem needs).
"""

import re


GENERIC_REDACTION_PATTERNS = {
    "email": re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'),
    "phone": re.compile(r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3,4}[-.\s]?\d{3,4}\b'),
    "url": re.compile(r'https?://\S+'),
}


def redact_generic_pii(text: str) -> tuple[str, list[str]]:
    """Redacts emails/phones/URLs. Returns (redacted_text, list of types found)."""
    redacted = text
    found = []
    for pii_type, pattern in GENERIC_REDACTION_PATTERNS.items():
        if pattern.search(redacted):
            found.append(pii_type)
            redacted = pattern.sub(f"[REDACTED_{pii_type.upper()}]", redacted)
    return redacted, found


def redact_custom_terms(text: str, custom_terms: list[str]) -> tuple[str, list[str]]:
    """Redacts organization-specified proprietary terms (product codenames,
    internal system names, vendor names under NDA, etc.). The org supplies
    this list themselves -- the tool can't guess what's proprietary to them.
    """
    redacted = text
    found = []
    for term in custom_terms:
        if not term.strip():
            continue
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        if pattern.search(redacted):
            found.append(term)
            redacted = pattern.sub("[REDACTED_PROPRIETARY_TERM]", redacted)
    return redacted, found


def prepare_for_llm(raw_text: str, custom_terms: list[str] = None) -> dict:
    """Single entry point: redacts generic PII + org-specified proprietary
    terms, returns what's safe to send plus an audit summary for the user
    to review before confirming submission.
    """
    custom_terms = custom_terms or []
    text_after_generic, generic_found = redact_generic_pii(raw_text)
    text_after_custom, custom_found = redact_custom_terms(text_after_generic, custom_terms)

    return {
        "safe_text": text_after_custom,
        "redacted_generic_types": generic_found,
        "redacted_custom_terms": custom_found,
        "was_modified": bool(generic_found or custom_found),
    }


if __name__ == "__main__":
    sample = ("We want to use ProjectPhoenix (our internal codename) to process customer "
              "emails like john@acme.com and route them through our VendorX integration.")
    result = prepare_for_llm(sample, custom_terms=["ProjectPhoenix", "VendorX"])
    print("Safe text to send:", result["safe_text"])
    print("Generic PII types redacted:", result["redacted_generic_types"])
    print("Custom proprietary terms redacted:", result["redacted_custom_terms"])
