
#Database engine and session factory



import logging
from collections.abc import Generator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings

logger = logging.getLogger(__name__)

engine = create_engine(settings.postgres_url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db() -> Generator[Session, None, None]:
    """
    Yields a DB session and guarantees it's closed afterward,
    even if an exception occurs mid-request.

    Usage (in a FastAPI route, later phases):
        db: Session = Depends(get_db)
    """
    db = SessionLocal()
    try:
        yield db
    except Exception:
        logger.exception("Database session error, rolling back")
        db.rollback()
        raise
    finally:
        db.close()