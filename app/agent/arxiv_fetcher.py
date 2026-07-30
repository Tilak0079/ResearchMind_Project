"""
arXiv Fetcher Agent : searches arXiv, downloads new papers,
and indexes them into  pipeline when local confidence is too low.

"""

import logging
import time
import uuid
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import requests
from sqlalchemy.orm import Session

from app.utils.minio_client import ensure_bucket_exists, upload_pdf_to_minio
from app.config import settings

from app.db.models import PaperRegistry
from app.ingestion.indexer import index_paper

from app.generation.llm_client import generate_response



QUERY_REFORMULATION_SYSTEM_PROMPT = """Rewrite the user's question into a short, effective arXiv search query
(a few key technical terms, not a full sentence). Respond with ONLY the search terms, no explanation."""


def reformulate_query_for_arxiv(user_query: str) -> str:
    """
    Query Reformulation (Section 3.3, Step 1): rewrites a natural-language
    question into arXiv-search-friendly terms using the LLM.

    Falls back to the raw query if the LLM call fails - a degraded search
    is better than a broken pipeline here.
    """
    try:
        reformulated = generate_response(QUERY_REFORMULATION_SYSTEM_PROMPT, user_query, temperature=0.3)
        return reformulated.strip()
    except Exception:
        logger.exception("Query reformulation failed, using raw query as fallback")
        return user_query

logger = logging.getLogger(__name__)

ARXIV_API_URL = "http://export.arxiv.org/api/query"
ARXIV_RATE_LIMIT_SECONDS = 3  # arXiv's usage policy: wait 3s between requests

# arXiv's XML replies use this XML "namespace" - a prefix that identifies
# which XML vocabulary a tag belongs to. We need it to correctly find tags
# like <entry> and <title> inside the response.
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


@dataclass
class ArxivCandidate:
    """One paper found via arXiv search, before we've downloaded it."""

    arxiv_id: str
    title: str
    authors: list[str]
    summary: str
    pdf_url: str
    published_date: str  # ISO format string, e.g. "2024-01-18"


def search_arxiv(query: str, max_results: int = 5) -> list[ArxivCandidate]:
    """
    Searches arXiv's public API for papers matching the query.

    Raises:
        requests.RequestException: if the arXiv API is unreachable.
    """
    params = {
        "search_query": f"all:{query}",
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    }

    logger.info(f"Searching arXiv for: '{query}'")
    response = requests.get(ARXIV_API_URL, params=params, timeout=15)
    response.raise_for_status()

    return _parse_arxiv_response(response.text)


def _parse_arxiv_response(xml_text: str) -> list[ArxivCandidate]:
    """Parses arXiv's Atom XML feed into a list of ArxivCandidate objects."""
    root = ET.fromstring(xml_text)
    candidates: list[ArxivCandidate] = []

    for entry in root.findall("atom:entry", ATOM_NS):
        raw_id_url = entry.find("atom:id", ATOM_NS).text
        arxiv_id = raw_id_url.rsplit("/", 1)[-1]

        title = entry.find("atom:title", ATOM_NS).text.strip().replace("\n", " ")
        summary = entry.find("atom:summary", ATOM_NS).text.strip()
        published = entry.find("atom:published", ATOM_NS).text[:10]

        authors = [
            author.find("atom:name", ATOM_NS).text
            for author in entry.findall("atom:author", ATOM_NS)
        ]

        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href")
                break

        candidates.append(
            ArxivCandidate(
                arxiv_id=arxiv_id,
                title=title,
                authors=authors,
                summary=summary,
                pdf_url=pdf_url,
                published_date=published,
            )
        )

    logger.info(f"Found {len(candidates)} candidates from arXiv")
    return candidates


def filter_new_candidates(candidates: list[ArxivCandidate], db: Session) -> list[ArxivCandidate]:
    """
    Removes candidates whose arxiv_id already exists in paper_registry -
    no point re-downloading and re-indexing a paper we already have.
    """
    existing_ids = {
        row.arxiv_id for row in db.query(PaperRegistry.arxiv_id).filter(PaperRegistry.arxiv_id.isnot(None)).all()
    }

    new_candidates = [c for c in candidates if c.arxiv_id not in existing_ids]

    skipped = len(candidates) - len(new_candidates)
    if skipped:
        logger.info(f"Skipped {skipped} candidate(s) already in paper_registry")

    return new_candidates


def download_pdf(pdf_url: str, arxiv_id: str, download_dir: str = "tests/arxiv_downloads") -> tuple[str, str]:
    """
    Downloads a PDF from arXiv to local disk, then uploads it to MinIO
    for permanent storage (Section 1.1: MinIO handles raw PDF storage).

    Returns:
        A tuple: (local_file_path, minio_path). The local path is needed
        temporarily because Docling (Phase 4) parses from a local file,
        not directly from MinIO. The MinIO path is what gets saved
        permanently in paper_registry.raw_pdf_s3_path.

    Raises:
        requests.RequestException: if the download fails.
    """
    Path(download_dir).mkdir(parents=True, exist_ok=True)
    local_path = f"{download_dir}/{arxiv_id}.pdf"

    logger.info(f"Downloading {arxiv_id} from {pdf_url}")
    response = requests.get(pdf_url, timeout=30)
    response.raise_for_status()

    with open(local_path, "wb") as f:
        f.write(response.content)

    ensure_bucket_exists(settings.minio_bucket_raw_pdfs)
    minio_path = upload_pdf_to_minio(local_path, f"{arxiv_id}.pdf")

    return local_path, minio_path


def fetch_and_index_from_arxiv(query: str, db: Session, max_results: int = 3) -> list[str]:
    """
    Full arXiv fetch loop (Section 3.3, Steps 1-6):
    search -> filter duplicates -> download -> parse -> chunk -> embed -> index.

    Args:
        query: user's search query (used as-is for now - see module docstring).
        db: active SQLAlchemy session.
        max_results: how many arXiv candidates to consider.

    Returns:
        List of paper_ids that were successfully indexed.
    """
    search_query = reformulate_query_for_arxiv(query)
    candidates = search_arxiv(search_query, max_results=max_results)
    new_candidates = filter_new_candidates(candidates, db)

    if not new_candidates:
        logger.info("No new candidates to fetch - all results already indexed")
        return []

    indexed_paper_ids: list[str] = []

    for i, candidate in enumerate(new_candidates):
        if i > 0:
            time.sleep(ARXIV_RATE_LIMIT_SECONDS)

        try:
            local_pdf_path, minio_pdf_path = download_pdf(candidate.pdf_url, candidate.arxiv_id)
        except requests.RequestException:
            logger.exception(f"Failed to download {candidate.arxiv_id}, skipping")
            continue

        

        paper_id = str(uuid.uuid4())
        paper = PaperRegistry(
            paper_id=paper_id,
            arxiv_id=candidate.arxiv_id,
            title=candidate.title,
            authors=candidate.authors,
            publication_date=candidate.published_date,
            source_type="arxiv_fetched",
            trust_tier="unverified",
            abstract=candidate.summary,
            ingestion_status="indexing",
            raw_pdf_s3_path=minio_pdf_path,
        )
        db.add(paper)
        db.commit()

        try:
            chunk_count = index_paper(local_pdf_path, paper_id, db, trust_tier="unverified")
            logger.info(f"Indexed arXiv paper {candidate.arxiv_id}: {chunk_count} chunks")
            indexed_paper_ids.append(paper_id)

            paper.ingestion_status = "complete"
            db.commit()
        except (FileNotFoundError, ValueError):
            logger.exception(f"Failed to index {candidate.arxiv_id}")
            paper.ingestion_status = "failed"
            db.commit()

    return indexed_paper_ids