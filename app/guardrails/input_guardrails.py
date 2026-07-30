"""
Input Guardrails 
"""

import logging
import re

import tiktoken

logger = logging.getLogger(__name__)

TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_QUERY_TOKENS = 500  # reasonable cap for a single user question

# Common prompt injection patterns - 
# Case-insensitive matching since attackers vary capitalization.
INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|above) instructions",
    r"disregard (all )?(previous|prior|above) instructions",
    r"reveal (your |the )?system prompt",
    r"you are now",
    r"act as (if you are|a) (?!a researcher|an academic)",  # allow "act as a researcher" type framing
    r"forget (everything|all) (you know|above)",
]


def check_prompt_injection(query: str) -> tuple[bool, str]:
    """
    Checks if the query contains obvious prompt injection attempts.

    Returns:
        (is_safe, reason) - is_safe=False means the query should be blocked.
    """
    query_lower = query.lower()

    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, query_lower):
            logger.warning(f"Prompt injection pattern matched: '{pattern}' in query")
            return False, "Query contains a potential prompt injection attempt."

    return True, ""


def check_token_budget(query: str) -> tuple[bool, str]:
    """
    Checks if the query is within our token budget.

    Returns:
        (is_safe, reason) - is_safe=False means the query is too long.
    """
    token_count = len(TOKENIZER.encode(query))

    if token_count > MAX_QUERY_TOKENS:
        logger.warning(f"Query exceeds token budget: {token_count} > {MAX_QUERY_TOKENS}")
        return False, f"Query is too long ({token_count} tokens, max {MAX_QUERY_TOKENS})."

    return True, ""

import json

from app.generation.llm_client import generate_response

SCOPE_CHECK_SYSTEM_PROMPT = """You are a scope classifier for an academic CS research assistant.
Classify the user's query into exactly one category:
- "academic_research": directly about CS/ML research, papers, or technical concepts
- "adjacent_technical": general technical/programming questions, not research-specific but reasonable to answer
- "out_of_scope": unrelated to research or technical topics (e.g. recipes, poems, personal advice)

Respond ONLY with JSON: {"category": "<one of the three above>"}
No other text."""


def check_scope(query: str) -> tuple[bool, str]:
    """
    Checks if the query is within academic/CS research scope (Section 4.1),
    using Qwen 3 as a lightweight classifier.

    Returns:
        (is_safe, reason) - is_safe=False if the query is out_of_scope.
    """
    try:
        raw_response = generate_response(SCOPE_CHECK_SYSTEM_PROMPT, query, temperature=0.0)
        result = json.loads(raw_response.strip())
        category = result.get("category", "out_of_scope")
    except (json.JSONDecodeError, KeyError, Exception):
        # If the LLM call fails or returns unparseable output, fail safe:
        # log it and let the query through rather than blocking legitimate
        # users due to a classifier hiccup (this is a soft guardrail, not
        # a security-critical one like injection detection).
        logger.exception("Scope check failed to parse LLM response, allowing query through")
        return True, ""

    if category == "out_of_scope":
        logger.info(f"Query classified out_of_scope: '{query[:50]}'")
        return False, "This question appears to be outside the academic/technical research scope of this assistant."

    return True, ""

def run_input_guardrails(query: str) -> tuple[bool, str]:
    """
    Runs all input guardrails in sequence. Stops at the first failure
    (no point checking further if the query is already rejected).
    """
    if not query or not query.strip():
        return False, "Query is empty."

    is_safe, reason = check_prompt_injection(query)
    if not is_safe:
        return False, reason

    is_safe, reason = check_token_budget(query)
    if not is_safe:
        return False, reason

    is_safe, reason = check_scope(query)
    if not is_safe:
        return False, reason

    return True, ""

import pikepdf


def sanitize_pdf(input_path: str, output_path: str) -> None:
    """
    Strips embedded JavaScript/macros from a PDF before it's parsed.
    Applies to user-uploaded PDFs - papers from  own
    local corpus or arXiv are already trusted sources, so this mainly
    protects against malicious user uploads.

    Raises:
        pikepdf.PdfError: if the file is corrupted or not a valid PDF.
    """
    with pikepdf.open(input_path) as pdf:
        # Remove the /OpenAction and /AA (Additional Actions) entries,
        # which are the common places JavaScript gets triggered automatically.
        if "/OpenAction" in pdf.Root:
            del pdf.Root["/OpenAction"]
        if "/AA" in pdf.Root:
            del pdf.Root["/AA"]

        pdf.save(output_path)

    logger.info(f"Sanitized PDF: {input_path} -> {output_path}")