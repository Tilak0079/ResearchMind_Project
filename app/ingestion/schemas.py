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
class ParsedDocument:
    """Full result of parsing one PDF."""

    title: str
    authors: list[str]
    sections: list[ParsedSection] = field(default_factory=list)
    raw_markdown: str = ""       # full doc as markdown, fallback if section split fails
    parsing_confidence: float = 1.0  # Stage 3 quality gate score