"""
POST /api/v1/query (Section 7.2): the main REST endpoint for asking a
question and getting a complete answer back (non-streaming).
"""

import logging
import uuid

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.query_pipeline import run_query_pipeline
from app.db.postgres import get_db

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")


class QueryRequest(BaseModel):
    """Request body schema - validated automatically by FastAPI/Pydantic."""

    session_id: str | None = None
    query: str


class QueryResponse(BaseModel):
    """Response body schema, matching Section 7.2's structure (simplified)."""

    session_id: str
    success: bool
    route_taken: str = ""
    confidence_score: float = 0.0
    answer: str = ""
    error_reason: str = ""
    sources: list = []


@router.post("/query", response_model=QueryResponse)
def submit_query(request: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    """
    Runs a user query through the full pipeline and returns the answer.

    Note: not async here because our underlying pipeline functions
    (embedding, LLM calls) are themselves synchronous - making this
    route async wouldn't add real concurrency benefit without also
    rewriting those lower-level functions as async, which is a larger
    change we're deferring rather than doing partially/inconsistently.
    """
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"[{session_id}] Query received: '{request.query[:80]}'")

    result = run_query_pipeline(request.query, db)

    return QueryResponse(
        session_id=session_id,
        success=result.success,
        route_taken=result.route_taken,
        confidence_score=result.confidence_score,
        answer=result.answer,
        error_reason=result.error_reason,
        sources=result.sources,
    )