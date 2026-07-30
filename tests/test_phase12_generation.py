"""Manual test: full pipeline from query to final generated, cited answer."""

import logging

from app.generation.context_assembler import assemble_context, build_user_message
from app.generation.llm_client import generate_response
from app.generation.prompts import GENERATION_SYSTEM_PROMPT
from app.guardrails.retrieval_guardrails import run_retrieval_guardrails
from app.ingestion.embedder import embed_text
from app.retrieval.qdrant_client import hybrid_search
from app.retrieval.reranker import rerank_chunks

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    query = "How does the attention mechanism work?"

    # Full pipeline: embed -> search -> rerank -> guardrails -> assemble -> generate
    query_embedding = embed_text(query)
    candidates = hybrid_search(query_embedding["dense"], query_embedding["sparse"], top_k=30)
    reranked = rerank_chunks(query, candidates, top_n=8)
    final_chunks = run_retrieval_guardrails(query, reranked)

    context = assemble_context(final_chunks)
    user_message = build_user_message(query, context)

    logger.info("Sending to LLM, this may take 10-30 seconds on M1...")
    answer = generate_response(GENERATION_SYSTEM_PROMPT, user_message)

    print("\n" + "=" * 60)
    print("FINAL ANSWER:")
    print("=" * 60)
    print(answer)