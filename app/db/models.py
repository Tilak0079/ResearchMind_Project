"""
SQLAlchemy models of the architecture doc exactly:
paper_registry, chunk_registry, session_history.

"""

import uuid

from sqlalchemy import (
    CheckConstraint,
    Column,
    Date,
    ForeignKey,
    Integer,
    String,
    Text,
    Float,
    TIMESTAMP,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class PaperRegistry(Base):
    """One row per paper (local corpus, arXiv-fetched, or user-uploaded)."""

    __tablename__ = "paper_registry"

    paper_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    arxiv_id = Column(String(32), unique=True)
    doi = Column(String(128))
    title = Column(Text, nullable=False)
    abstract = Column(Text)
    authors = Column(JSONB)
    publication_date = Column(Date)
    source_type = Column(String(20))
    trust_tier = Column(String(20), server_default="unverified")
    ingestion_status = Column(String(20), server_default="pending")
    raw_pdf_s3_path = Column(Text)
    parsed_doc_s3_path = Column(Text)  # Docling output, not TEI-XML (renamed per doc)
    citation_count = Column(Integer, server_default="0")
    parent_citations = Column(JSONB)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    chunks = relationship("ChunkRegistry", back_populates="paper", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(
            "source_type IN ('local_corpus','arxiv_fetched','user_uploaded')",
            name="ck_paper_source_type",
        ),
    )


class ChunkRegistry(Base):
    """One row per chunk produced by the chunker ."""

    __tablename__ = "chunk_registry"

    chunk_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    paper_id = Column(UUID(as_uuid=True), ForeignKey("paper_registry.paper_id", ondelete="CASCADE"))
    chunk_type = Column(String(20))  # text | table | equation | figure_caption
    section_name = Column(String(128))
    page_number = Column(Integer)
    part_index = Column(Integer, server_default="0")  # set when a section exceeds the token cap
    token_count = Column(Integer)
    vector_db_id = Column(String(64))  # pointer to the Qdrant point ID
    artifact_path = Column(Text, nullable=True) # path to minio object or generic artifact identifier
    content_hash = Column(String(64))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    paper = relationship("PaperRegistry", back_populates="chunks")


class SessionHistory(Base):
    """One row per query/response pair, used for history + observability."""

    __tablename__ = "session_history"

    session_id = Column(UUID(as_uuid=True), primary_key=True, server_default=func.gen_random_uuid())
    user_id = Column(UUID(as_uuid=True), nullable=False)
    query = Column(Text)
    route_taken = Column(String(20))
    confidence_score = Column(Float)
    response = Column(Text)
    citations = Column(JSONB)
    faithfulness_score = Column(Float)  # retained column, unused in v2 (no runtime check)
    latency_ms = Column(Integer)
    token_cost = Column(Integer)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())