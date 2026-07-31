"""
Loads real paper metadata from JSON files (matching a PDF by filename)
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


def load_paper_metadata(json_path: str) -> dict:
    """
    Reads one metadata JSON file and returns a dict ready to build a PaperRegistry row.

    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"Metadata JSON not found: {json_path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Authors come as [{"name": "..."}] - we just want the names as strings,
    # matching how Phase 10's arXiv fetcher already stores authors.
    author_names = [author["name"] for author in raw.get("authors", []) if "name" in author]

    # publication_date arrives as a full ISO timestamp - we only need the date part.
    pub_date = None
    if raw.get("publication_date"):
        pub_date = datetime.fromisoformat(raw["publication_date"]).date()

    return {
        "arxiv_id": raw.get("arxiv_id"),
        "doi": raw.get("doi"),
        "title": raw.get("title", path.stem),
        "authors": author_names,
        "publication_date": pub_date,
        "source_type": raw.get("source_type", "local_corpus"),
        "trust_tier": raw.get("trust_tier", "local_corpus"),
        "abstract": raw.get("abstract"),
    }