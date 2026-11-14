"""Fixed-size and paragraph-aware chunkers with source and section metadata."""
import hashlib

from .models import Chunk, DocumentPage


def _section_at(page: DocumentPage, offset: int) -> str | None:
    spans = page.metadata.get("section_spans", [])
    for span in spans:
        if span["start"] <= offset < span["end"]:
            return span["section"]
    return None


def _make_chunk(page: DocumentPage, start: int, end: int) -> Chunk:
    text = page.text[start:end].strip()
    digest = hashlib.sha256(
        f"{page.document_id}\0{page.page}\0{start}\0{end}\0{text}".encode("utf-8")
    ).hexdigest()[:24]
    return Chunk(
        chunk_id=digest,
        document_id=page.document_id,
        source=page.source,
        page=page.page,
        text=text,
        metadata={
            "section": _section_at(page, start),
            "source_start_line": page.metadata.get("source_start_line"),
            "source_end_line": page.metadata.get("source_end_line"),
            "char_start": start,
            "char_end": end,
        },
    )


def _split_oversized(page: DocumentPage, start: int, end: int, max_size: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    while start < end:
        boundary = min(start + max_size, end)
        if boundary < end:
            whitespace = page.text.rfind(" ", start, boundary)
            if whitespace > start + max_size // 2:
                boundary = whitespace
        chunks.append(_make_chunk(page, start, boundary))
        start = boundary
        while start < end and page.text[start].isspace():
            start += 1
    return chunks


def fixed_size(page: DocumentPage, size: int, overlap: int) -> list[Chunk]:
    """Split one source unit into fixed character windows with a configurable overlap."""
    if size < 1 or overlap < 0 or overlap >= size:
        raise ValueError("size must be positive and overlap must be in [0, size)")
    chunks: list[Chunk] = []
    start = 0
    while start < len(page.text):
        end = min(start + size, len(page.text))
        chunks.append(_make_chunk(page, start, end))
        if end == len(page.text):
            break
        start = end - overlap
    return chunks


def by_paragraph(page: DocumentPage, max_size: int) -> list[Chunk]:
    """Group adjacent paragraphs within the active section, splitting long paragraphs."""
    if max_size < 1:
        raise ValueError("max_size must be positive")
    paragraphs: list[tuple[int, int, str | None]] = []
    cursor = 0
    for paragraph in page.text.split("\n\n"):
        start = page.text.find(paragraph, cursor)
        if start < 0:
            continue
        end = start + len(paragraph)
        paragraphs.append((start, end, _section_at(page, start)))
        cursor = end + 2

    chunks: list[Chunk] = []
    group_start: int | None = None
    group_end = 0
    group_section: str | None = None

    def flush() -> None:
        nonlocal group_start, group_end, group_section
        if group_start is not None:
            chunks.extend(_split_oversized(page, group_start, group_end, max_size))
        group_start = None
        group_section = None

    for start, end, section in paragraphs:
        if end - start > max_size:
            flush()
            chunks.extend(_split_oversized(page, start, end, max_size))
        elif (
            group_start is None
            or section != group_section
            or end - group_start > max_size
        ):
            flush()
            group_start, group_end, group_section = start, end, section
        else:
            group_end = end
    flush()
    return [chunk for chunk in chunks if chunk.text]
