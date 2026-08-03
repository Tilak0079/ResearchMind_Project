import logging
from fastapi import APIRouter, Depends, HTTPException

from sqlalchemy.orm import Session
from app.db.postgres import get_db

from app.api.schemas_paper import (
    PaperSearchResponse,
    PaperQueryRequest,
    PaperQueryResponse,
    PaperSessionRequest,
    PaperSessionResponse
)
from app.services.paper_service import search_papers, create_paper_session
from app.paper_query_pipeline import run_paper_query_pipeline

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/v1")

@router.get("/papers/search", response_model=list[PaperSearchResponse])
def get_paper_search(q: str = "", db: Session = Depends(get_db)):
    """Search for papers by title, author, or arxiv ID."""
    if not q or len(q) < 2:
        return []
    return search_papers(q, db)

@router.post("/paper/session", response_model=PaperSessionResponse)
def post_paper_session(request: PaperSessionRequest, db: Session = Depends(get_db)):
    """Create a new session tied strictly to a paper_id."""
    try:
        session_id = create_paper_session(request.paper_id, db)
        return PaperSessionResponse(session_id=session_id, paper_id=request.paper_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error creating paper session: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/query/paper", response_model=PaperQueryResponse)
def submit_paper_query(request: PaperQueryRequest, db: Session = Depends(get_db)) -> PaperQueryResponse:
    """Run a query strictly against a single paper using Paper Mode."""
    from app.db.models import Session as DBSession
    
    session = db.query(DBSession).filter(DBSession.session_id == request.session_id).first()
    
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Please create a paper session first.")
        
    if session.mode != "paper":
        raise HTTPException(status_code=400, detail="This session is not a Paper Mode session.")
        
    if str(session.paper_id) != request.paper_id:
        raise HTTPException(status_code=400, detail="Session paper_id does not match the requested paper_id.")
        
    logger.info(f"[{request.session_id}] Paper Query received: '{request.query[:80]}'")
    
    result = run_paper_query_pipeline(request.query, db, request.session_id, request.paper_id)
    return result
