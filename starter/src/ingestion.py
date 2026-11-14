"""Read, normalize, and retain source locations for the selected corpus."""
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from pypdf import PdfReader

from .models import DocumentPage


def normalize_text(text: str) -> str:
    """Normalize Unicode and whitespace while preserving paragraph structure."""
    normalized = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    normalized = normalized.replace("\u00a0", " ")
    lines = [line.rstrip() for line in normalized.split("\n")]
    normalized = "\n".join(lines).strip()
    return re.sub(r"\n{3,}", "\n\n", normalized)


def load_pdf(path: Path, document_id: str, source: str) -> list[DocumentPage]:
    """Extract text from each physical PDF page; report empty pages to the caller."""
    reader = PdfReader(str(path))
    pages: list[DocumentPage] = []
    for number, pdf_page in enumerate(reader.pages, start=1):
        text = normalize_text(pdf_page.extract_text() or "")
        pages.append(
            DocumentPage(
                document_id=document_id,
                source=source,
                page=str(number),
                text=text,
                metadata={"extraction_error": "no extractable text" if not text else None},
            )
        )
    return pages


def _section_heading(text: str) -> str | None:
    candidate = text.strip().strip("~*_ ").strip()
    letters = re.sub(r"[^A-Za-z]", "", candidate)
    if (
        "\n" not in candidate
        and 3 <= len(candidate) <= 90
        and len(letters) >= 3
        and re.fullmatch(r"[A-Z][A-Z0-9]*(?:[ .'-]+[A-Z0-9]+)*", candidate)
        and not candidate.startswith("CASE ")
        and candidate not in {"EXAMPLES:", "PROBLEMS"}
    ):
        return candidate
    return None


def _annotate_sections(pages: list[DocumentPage]) -> None:
    active_section: str | None = None
    for page in pages:
        spans: list[dict[str, Any]] = []
        cursor = 0
        for paragraph in page.text.split("\n\n"):
            start = page.text.find(paragraph, cursor)
            if start < 0:
                continue
            end = start + len(paragraph)
            heading = _section_heading(paragraph)
            if heading is not None:
                active_section = heading
            spans.append({"start": start, "end": end, "section": active_section})
            cursor = end + 2
        page.metadata["section_spans"] = spans


def load_jsonl(path: Path, errors_path: Path) -> tuple[list[DocumentPage], list[dict[str, Any]]]:
    """Load selected Gutenberg units; log malformed records without losing valid ones."""
    pages: list[DocumentPage] = []
    errors: list[dict[str, Any]] = []
    manifest_path = path.with_name("gutenberg_38536_manifest.json")
    manifest: dict[str, Any] = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    source_id = str(manifest.get("source_id", path.stem))
    source = str(manifest.get("source_path_or_url", path.name))
    try:
        with path.open(encoding="utf-8-sig") as corpus:
            for line_number, raw_line in enumerate(corpus, start=1):
                try:
                    record = json.loads(raw_line)
                    text = normalize_text(record["text"])
                    if not text:
                        raise ValueError("record has empty text")
                    unit_id = str(record["unit_id"])
                    pages.append(
                        DocumentPage(
                            document_id=source_id,
                            source=source,
                            page=unit_id,
                            text=text,
                            metadata={
                                "source_start_line": int(record["source_start_line"]),
                                "source_end_line": int(record["source_end_line"]),
                            },
                        )
                    )
                except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                    errors.append({"file": str(path), "line": line_number, "error": str(error)})
    except (OSError, UnicodeError) as error:
        errors.append({"file": str(path), "error": str(error)})

    errors_path.parent.mkdir(parents=True, exist_ok=True)
    errors_path.write_text(
        "".join(json.dumps(error, ensure_ascii=False) + "\n" for error in errors),
        encoding="utf-8",
    )
    if not pages:
        raise RuntimeError(f"No valid corpus records found in {path}; see {errors_path}")
    _annotate_sections(pages)
    return pages, errors
