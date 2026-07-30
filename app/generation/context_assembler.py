"""
Context Assembler (Section 6.1): formats our guardrail-passed chunks into
the {retrieved_context_block} text the system prompt expects, including
trust_tier flags so the LLM can note unverified sources per Section 6.1's
SCOPE & TRUST rules.
"""


def assemble_context(reranked_results: list) -> str:
    """
    Turns a list of (chunk, score) tuples into a formatted text block
    the LLM can read, with section/page info for citation purposes.
    """
    if not reranked_results:
        return "No relevant context was found."

    blocks = []
    for point, _score in reranked_results:
        payload = point.payload
        trust_note = " [trust_tier: unverified]" if payload.get("trust_flag") else ""

        block = (
            f"[Section {payload['section_name']}, Page {payload.get('page_number', 'N/A')}]"
            f"{trust_note}\n"
            f"{payload['text']}"
        )
        blocks.append(block)

    return "\n\n---\n\n".join(blocks)


def build_user_message(query: str, context: str, conversation_history: str = "") -> str:
    """
    Combines the query, context, and history into the final message sent
    to the LLM - matches the {CONTEXT} / {CONVERSATION HISTORY} / {USER QUESTION}
    placeholders from Section 6.1's prompt template.
    """
    return f"""CONTEXT:
{context}

CONVERSATION HISTORY:
{conversation_history if conversation_history else "(none - this is the first message)"}

USER QUESTION:
{query}
"""