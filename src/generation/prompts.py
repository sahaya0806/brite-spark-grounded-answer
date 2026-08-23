"""
Prompt construction for grounded answer generation.

Prompts strictly constrain the language model to synthesize answers exclusively
from supplied PolicyClause evidence.
"""

from __future__ import annotations

from typing import Sequence
from src.ingestion.parser import PolicyClause


SYSTEM_PROMPT = """You are a grounded policy manual assistant for the Calder County Household Support Program.
Your job is to answer the user's question clearly, concisely, and accurately based ONLY on the provided authoritative policy clauses.

STRICT GROUNDING RULES:
1. Use ONLY the supplied policy clauses. Never use external knowledge, unstated assumptions, or generalizations.
2. Every substantive policy claim MUST include a citation using the exact format [§<clause_id>], e.g., [§4.3.2].
3. Never cite any clause ID that is not explicitly supplied in the evidence below.
4. If a detail is not explicitly mentioned in the text, do not guess or infer it.
5. Write in plain, professional English."""


def build_grounded_prompt(
    question: str,
    clauses: Sequence[PolicyClause],
) -> list[dict[str, str]]:
    """
    Construct chat completion messages for a SUPPORTED evidence decision.

    Parameters
    ----------
    question:
        The user's question.
    clauses:
        Authoritative PolicyClause objects containing the supporting text.

    Returns
    -------
    list[dict[str, str]]
        Messages formatted for the ChatProvider.
    """
    evidence_blocks = []
    for c in clauses:
        block = (
            f"[§{c.clause_id}] {c.part_title} — {c.section_title}\n"
            f"{c.text}"
        )
        evidence_blocks.append(block)

    evidence_text = "\n\n".join(evidence_blocks)

    user_content = (
        f"QUESTION:\n{question}\n\n"
        f"AUTHORITATIVE POLICY EVIDENCE:\n{evidence_text}\n\n"
        f"INSTRUCTIONS:\n"
        f"Answer the question in plain English using ONLY the policy evidence above. "
        f"Include clause citations [§<clause_id>] for every substantive statement."
    )

    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
