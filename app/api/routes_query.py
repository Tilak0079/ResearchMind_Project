"""
POST /api/v1/query: the main REST endpoint for asking a
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


from app.api.schemas import QueryRequest, QueryResponse

@router.post("/query", response_model=QueryResponse)
def submit_query(request: QueryRequest, db: Session = Depends(get_db)) -> QueryResponse:
    """
    Runs a user query through the full pipeline and returns the answer.

    """
    session_id = request.session_id or str(uuid.uuid4())

    logger.info(f"[{session_id}] Query received: '{request.query[:80]}'")

    result = run_query_pipeline(request.query, db, session_id)

    return result