from typing import Any
from pydantic import BaseModel, Field

from app.api.schemas import ConfidenceDetail, Citation, Evidence, ArtifactMetadataModel


class PaperQueryRequest(BaseModel):
    session_id: str
    paper_id: str
    query: str


class PaperQueryResponse(BaseModel):
    session_id: str
    success: bool
    error_reason: str = ""
    
    query: str
    answer: str = ""
    summary: list[str] = Field(default_factory=list)
    artifacts: list[ArtifactMetadataModel] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    
    confidence: ConfidenceDetail | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PaperSearchResponse(BaseModel):
    paper_id: str
    title: str
    authors: list[str]
    publication_year: int | None
    arxiv_id: str | None


class PaperSessionRequest(BaseModel):
    paper_id: str


class PaperSessionResponse(BaseModel):
    session_id: str
    paper_id: str
