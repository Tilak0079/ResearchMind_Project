import logging
import uuid
from sqlalchemy.orm import Session
from sqlalchemy import or_, cast, String, case, asc

from app.db.models import PaperRegistry, Session as DBSession
from app.api.schemas_paper import PaperSearchResponse

logger = logging.getLogger(__name__)

def search_papers(query: str, db: Session) -> list[PaperSearchResponse]:
    """Search the PostgreSQL PaperRegistry catalogue using ILIKE."""
    search_term = f"%{query}%"
    
    relevance = case(
        (PaperRegistry.title.ilike(query), 1),
        (PaperRegistry.title.ilike(f"{query}%"), 2),
        (PaperRegistry.title.ilike(search_term), 3),
        (cast(PaperRegistry.authors, String).ilike(search_term), 4),
        (PaperRegistry.arxiv_id.ilike(search_term), 5),
        else_=6
    )

    # Cast JSONB to String to search within authors
    results = (
        db.query(PaperRegistry)
        .filter(
            or_(
                PaperRegistry.title.ilike(search_term),
                cast(PaperRegistry.authors, String).ilike(search_term),
                PaperRegistry.arxiv_id.ilike(search_term)
            )
        )
        .order_by(asc(relevance))
        .limit(10)
        .all()
    )
    
    responses = []
    for row in results:
        pub_year = row.publication_date.year if row.publication_date else None
        
        responses.append(PaperSearchResponse(
            paper_id=str(row.paper_id),
            title=row.title,
            authors=row.authors if isinstance(row.authors, list) else [],
            publication_year=pub_year,
            arxiv_id=row.arxiv_id
        ))
    return responses

def create_paper_session(paper_id: str, db: Session) -> str:
    """Create a new session associated with a specific paper in Paper Mode."""
    # Ensure paper exists
    paper = db.query(PaperRegistry).filter(PaperRegistry.paper_id == paper_id).first()
    if not paper:
        raise ValueError(f"Paper with id {paper_id} not found.")

    session_id = str(uuid.uuid4())
    new_session = DBSession(
        session_id=session_id,
        mode="paper",
        paper_id=paper_id
    )
    db.add(new_session)
    db.commit()
    logger.info(f"Created new Paper Mode session: {session_id} for paper: {paper_id}")
    return session_id
