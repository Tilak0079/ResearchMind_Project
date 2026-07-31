"""
Retrieval Guardrails : checks that run AFTER search results
come back, BEFORE they're sent to the LLM for generation.


"""
import tiktoken
import logging
import json

from app.generation.llm_client import generate_response

logger = logging.getLogger(__name__)


def apply_trust_filter(reranked_results: list) -> list:
    """
    Anti-Poisoning / Source Trust Filter : doesn't remove
    unverified-source chunks, just tags them so the generation step can
    flag them in the Limitations section .

    Returns:
        The same list, with each item's payload annotated with
        'trust_flag' (True if the source needs a Limitations note).
    """
    flagged_count = 0

    for point, _score in reranked_results:
        trust_tier = point.payload.get("trust_tier", "unverified")
        needs_flag = trust_tier == "unverified"
        point.payload["trust_flag"] = needs_flag

        if needs_flag:
            flagged_count += 1

    if flagged_count:
        logger.info(f"{flagged_count} of {len(reranked_results)} chunks flagged as unverified source")

    return reranked_results



TOKENIZER = tiktoken.get_encoding("cl100k_base")
MAX_CONTEXT_TOKENS = 4000  # reasonable budget for assembled context sent to the LLM


def truncate_context(reranked_results: list, max_tokens: int = MAX_CONTEXT_TOKENS) -> list:
    """
    Context Truncation Control
    """
    kept_results = []
    running_total = 0

    for point, score in reranked_results:
        chunk_tokens = len(TOKENIZER.encode(point.payload["text"]))

        if running_total + chunk_tokens > max_tokens:
            logger.info(
                f"Context truncated at {len(kept_results)} chunks "
                f"({running_total} tokens) - remaining chunks dropped"
            )
            break

        kept_results.append((point, score))
        running_total += chunk_tokens

    return kept_results






# Reranker score threshold for relevance filtering. Chunks below this score
# are dropped without an LLM call - the reranker (BGE-reranker-v2-m3) is
# already a purpose-built cross-encoder for query-chunk relevance, so a
# separate LLM-judge pass is redundant for our use case and adds significant
# latency (each LLM call ~10-30s on M1 CPU/Metal - was 1-4+ min for a full
# candidate set). Chosen deliberately over the LLM-judge approach described
# in the architecture doc's Section 4.2 for latency reasons - see README's
# "Key Engineering Decisions" section for the full reasoning.
RELEVANCE_SCORE_THRESHOLD = 0.35


def grade_relevance(query: str, reranked_results: list) -> list:
    
    graded_results = [
        (point, score) for point, score in reranked_results
        if score >= RELEVANCE_SCORE_THRESHOLD
    ]

    dropped = len(reranked_results) - len(graded_results)
    if dropped:
        logger.info(f"Relevance filter (threshold={RELEVANCE_SCORE_THRESHOLD}): dropped {dropped} chunk(s)")

    logger.info(f"Relevance grading: {len(reranked_results)} -> {len(graded_results)} chunks kept")
    return graded_results

CONSISTENCY_CHECK_SYSTEM_PROMPT = """You compare short text excerpts from different research papers for
direct factual contradictions (not just different topics or phrasing - genuine conflicting claims).

Respond ONLY with JSON: {"has_contradiction": true|false, "note": "<one sentence, empty string if false>"}"""


def check_cross_paper_consistency(reranked_results: list) -> list:
    
    distinct_paper_ids = {point.payload["paper_id"] for point, _ in reranked_results}

    if len(distinct_paper_ids) < 2:
        return reranked_results

   
    seen_papers: dict[str, tuple] = {}
    for point, score in reranked_results:
        paper_id = point.payload["paper_id"]
        if paper_id not in seen_papers:
            seen_papers[paper_id] = (point, score)

    representative_chunks = list(seen_papers.values())

    for i in range(len(representative_chunks)):
        for j in range(i + 1, len(representative_chunks)):
            chunk_a = representative_chunks[i][0].payload["text"]
            chunk_b = representative_chunks[j][0].payload["text"]

            user_prompt = f"EXCERPT A:\n{chunk_a}\n\nEXCERPT B:\n{chunk_b}"

            try:
                raw_response = generate_response(CONSISTENCY_CHECK_SYSTEM_PROMPT, user_prompt, temperature=0.0)
                result = json.loads(raw_response.strip())
                if result.get("has_contradiction"):
                    note = result.get("note", "Potential contradiction detected between sources.")
                    representative_chunks[i][0].payload["consistency_note"] = note
                    representative_chunks[j][0].payload["consistency_note"] = note
                    logger.info(f"Contradiction flagged: {note}")
            except (json.JSONDecodeError, KeyError, Exception):
                logger.exception("Consistency check failed for a chunk pair, skipping")

    return reranked_results

def run_retrieval_guardrails(query: str, reranked_results: list) -> list:
    """
    Runs all 4 retrieval guardrails in sequence 
    """
    if not reranked_results:
        return []

    graded = grade_relevance(query, reranked_results)
    trust_flagged = apply_trust_filter(graded)
    consistency_checked = check_cross_paper_consistency(trust_flagged)
    final_results = truncate_context(consistency_checked)

    logger.info(f"Retrieval guardrails: {len(reranked_results)} -> {len(final_results)} chunks after all checks")
    return final_results