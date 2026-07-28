"""
Chunking strategy (Section 2.2): markdown header-splitting with a token-size
safety net.

HOW IT WORKS:
1. Docling already split the document into sections by '#'/'##' headers
   (done in Phase 4's parser). Each section becomes one chunk by default.
2. Some sections (e.g. "Related Work", "Experiments") have no sub-headers
   and can run very long — too long to embed/retrieve well as one chunk.
3. For any section over ~600-800 tokens, we split further at PARAGRAPH
   boundaries (blank lines), never mid-sentence. Each resulting piece
   gets a `part_index` (0, 1, 2...) so we can trace it back to its
   parent section later.
4. Rare edge case: if a single PARAGRAPH itself is bigger than the cap
   (e.g. a dense, unbroken references list with no blank lines), we fall
   back to splitting at sentence boundaries instead.
5. No overlap between chunks (unlike older overlap-window approaches) —
   header/paragraph boundaries are already clean split points, so overlap
   isn't needed here. This can be revisited if evaluation shows chunk-edge
   context loss (see Section 2.2 tradeoff note in the architecture doc).
"""

import logging
import uuid

import tiktoken

from app.ingestion.schemas import Chunk, ParsedDocument, ParsedSection

logger = logging.getLogger(__name__)

MAX_TOKENS_PER_CHUNK = 700  # midpoint of the doc's 600-800 token soft cap
TOKENIZER = tiktoken.get_encoding("cl100k_base")  # standard GPT-family tokenizer, good enough for counting


def count_tokens(text: str) -> int:
    """Returns the number of tokens in a string using the cl100k tokenizer."""
    return len(TOKENIZER.encode(text))


def chunk_document(doc: ParsedDocument, paper_id: str) -> list[Chunk]:
    """
    Converts a ParsedDocument into a list of Chunks, applying the
    header-split + size-cap + paragraph-fallback strategy.
    """
    all_chunks: list[Chunk] = []

    for section in doc.sections:
        section_chunks = _chunk_section(section, paper_id)
        all_chunks.extend(section_chunks)

    logger.info(f"Chunked '{doc.title}' into {len(all_chunks)} chunks from {len(doc.sections)} sections")
    return all_chunks


def _chunk_section(section: ParsedSection, paper_id: str) -> list[Chunk]:
    """
    Chunks a single section. If it fits under the token cap, it's one chunk.
    If it's too long, split it at paragraph boundaries (part_index 0, 1, 2...).
    """
    if not section.content_markdown.strip():
        # Empty section (e.g. a header immediately followed by another header) —
        # nothing to embed, skip it rather than creating a useless zero-token chunk.
        return []

    token_count = count_tokens(section.content_markdown)

    if token_count <= MAX_TOKENS_PER_CHUNK:
        return [
            Chunk(
                chunk_id=str(uuid.uuid4()),
                paper_id=paper_id,
                chunk_type="text",
                section_name=section.header,
                content=section.content_markdown,
                part_index=0,
                page_number=section.page_number,
                token_count=token_count,
            )
        ]

    # Section too long — split at paragraph boundaries (blank lines)
    logger.info(f"Section '{section.header}' has {token_count} tokens, splitting at paragraphs")
    paragraphs = [p.strip() for p in section.content_markdown.split("\n\n") if p.strip()]

    parts: list[Chunk] = []
    current_part = ""
    part_index = 0

    for paragraph in paragraphs:
        # Rare case: a single paragraph alone is bigger than the cap
        # (e.g. an unbroken block of references) — split it further first.
        if count_tokens(paragraph) > MAX_TOKENS_PER_CHUNK:
            sub_pieces = _split_oversized_paragraph(paragraph)
        else:
            sub_pieces = [paragraph]

        for piece in sub_pieces:
            candidate = f"{current_part}\n\n{piece}".strip() if current_part else piece

            if count_tokens(candidate) > MAX_TOKENS_PER_CHUNK and current_part:
                # Adding this piece would push us over the cap —
                # close out the current part first, then start a new one.
                parts.append(
                    Chunk(
                        chunk_id=str(uuid.uuid4()),
                        paper_id=paper_id,
                        chunk_type="text",
                        section_name=section.header,
                        content=current_part,
                        part_index=part_index,
                        page_number=section.page_number,
                        token_count=count_tokens(current_part),
                    )
                )
                part_index += 1
                current_part = piece
            else:
                current_part = candidate

    # Don't forget the last part
    if current_part:
        parts.append(
            Chunk(
                chunk_id=str(uuid.uuid4()),
                paper_id=paper_id,
                chunk_type="text",
                section_name=section.header,
                content=current_part,
                part_index=part_index,
                page_number=section.page_number,
                token_count=count_tokens(current_part),
            )
        )

    return parts


def _split_oversized_paragraph(paragraph: str) -> list[str]:
    """
    Fallback for the rare case where a single paragraph itself exceeds the
    token cap (e.g. dense, unbroken reference lists). Splits at sentence
    boundaries ('. ') instead of paragraph boundaries, since paragraph
    splitting alone isn't fine-grained enough here.
    """
    sentences = paragraph.split(". ")
    pieces: list[str] = []
    current = ""

    for sentence in sentences:
        candidate = f"{current}. {sentence}".strip() if current else sentence
        if count_tokens(candidate) > MAX_TOKENS_PER_CHUNK and current:
            pieces.append(current)
            current = sentence
        else:
            current = candidate

    if current:
        pieces.append(current)

    return pieces