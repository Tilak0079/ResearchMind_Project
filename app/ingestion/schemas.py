"""Shared data structures for the ingestion pipeline (parsing → chunking → embedding)."""

from dataclasses import dataclass, field


@dataclass
class ParsedSection:
    """One structural section of a parsed paper (a header + its body content)."""

    header: str
    level: int          # 1 = '#', 2 = '##', etc.
    content_markdown: str
    page_number: int | None = None


@dataclass
class ParsedArtifact:
    """An extracted figure, table, or equation."""
    artifact_type: str       # "figure", "table", "equation"
    content: str             # Caption for figures, markdown for tables, latex for equations
    page_number: int | None = None
    artifact_path: str | None = None  # Path to MinIO for figures, or just empty for others

@dataclass
class ParsedDocument:
    """Full result of parsing one PDF."""

    title: str
    authors: list[str]
    sections: list[ParsedSection] = field(default_factory=list)
    artifacts: list[ParsedArtifact] = field(default_factory=list)
    raw_markdown: str = ""       # full doc as markdown, fallback if section split fails
    parsing_confidence: float = 1.0  # Stage 3 quality gate score


@dataclass
class Chunk:
    """One chunk ready for embedding + storage. Mirrors chunk_registry columns."""

    chunk_id: str
    paper_id: str
    chunk_type: str          # text | figure | table | equation
    section_name: str
    content: str              # the actual text/markdown to embed
    part_index: int = 0       # >0 when a section was split further due to size
    page_number: int | None = None
    token_count: int = 0    
    artifact_path: str | None = None  # path to minio object or generic artifact identifier