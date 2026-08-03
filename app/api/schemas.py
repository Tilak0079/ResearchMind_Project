from typing import Any
from pydantic import BaseModel, Field

class QueryRequest(BaseModel):
    session_id: str | None = None
    query: str

class ConfidenceDetail(BaseModel):
    overall_confidence: float = Field(..., description="Overall confidence score between 0.0 and 1.0")
    explanation: str = Field(..., description="Natural language explanation of the confidence level")
    factors: list[str] = Field(default_factory=list, description="List of factors influencing the confidence")

class Citation(BaseModel):
    title: str | None = None
    authors: list[str] = Field(default_factory=list)
    page: int | None = None
    section: str | None = None
    trust_tier: str | None = None
    retrieval_score: float = 0.0
    url: str | None = None

class Evidence(BaseModel):
    paper: str | None = None
    page: int | None = None
    chunk_text: str = ""
    similarity_score: float = 0.0
    reranker_score: float = 0.0

class ArtifactMetadataModel(BaseModel):
    type: str = Field(..., description="E.g., figure, table, equation, code")
    title: str | None = None
    caption: str | None = None
    page: int | None = None
    paper: str | None = None
    url: str | None = None
    content: str | None = None
    description: str | None = None

class QueryResponse(BaseModel):
    session_id: str
    success: bool
    route_taken: str = ""
    error_reason: str = ""
    
    # LLM Generated + Pipeline fields
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
