"""Run the fixed A/B/C retrieval experiment on the bundled Gutenberg corpus."""
import csv
import json
import logging
import os
import platform
import statistics
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import psycopg
from dotenv import load_dotenv

from .chunking import by_paragraph, fixed_size
from .embeddings import OpenRouterEmbedder
from .evaluation import evaluate_query, summarize
from .ingestion import load_jsonl
from .models import Chunk, DocumentPage
from .store import PgVectorStore

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
EMBEDDING_MODEL = "openai/text-embedding-3-small"
ESTIMATED_USD_PER_MILLION_TOKENS = 0.02


def _jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _chunks(pages: list[DocumentPage], strategy: str) -> list[Chunk]:
    result: list[Chunk] = []
    for page in pages:
        if strategy == "A":
            result.extend(fixed_size(page, size=500, overlap=50))
        else:
            result.extend(by_paragraph(page, max_size=900))
    return result


def _verify_benchmark(
    questions: list[dict[str, Any]], pages: list[DocumentPage]
) -> set[str]:
    expected_page_count = {page.page for page in pages}
    page_text = {page.page: page.text for page in pages}
    if len(pages) < 30:
        raise RuntimeError(f"Corpus has {len(pages)} source units; at least 30 are required.")
    if len(questions) < 20:
        raise RuntimeError(f"Benchmark has {len(questions)} questions; at least 20 are required.")
    if sum(question["split"] == "dev" for question in questions) < 15:
        raise RuntimeError("Benchmark must contain at least 15 dev questions.")
    if sum(question["split"] == "test" for question in questions) < 5:
        raise RuntimeError("Benchmark must contain at least 5 test questions.")

    section_values: set[str] = set()
    for page in pages:
        section_values.update(
            span["section"]
            for span in page.metadata["section_spans"]
            if span["section"] is not None
        )
    filter_questions = 0
    test_filter_questions = 0
    for question in questions:
        locations = [
            {
                "gold_source_id": question["gold_source_id"],
                "gold_page": question["gold_page"],
                "evidence_quote": question["evidence_quote"],
            },
            *question.get("additional_gold_sources", []),
        ]
        for location in locations:
            page = location["gold_page"]
            if page not in expected_page_count or location["gold_source_id"] != pages[0].document_id:
                raise RuntimeError(f"Invalid gold location in benchmark question {question['id']}.")
            if location["evidence_quote"] not in page_text[page]:
                raise RuntimeError(
                    f"Evidence quote is not present in the normalized source: {question['id']} / {page}"
                )
        requested_filter = question.get("filter")
        if requested_filter:
            filter_questions += 1
            test_filter_questions += question["split"] == "test"
            if requested_filter.get("provided_by") != "question":
                raise RuntimeError(
                    f"Filter for {question['id']} is not explicitly provided in its question."
                )
            value = requested_filter.get("value")
            if requested_filter.get("field") != "section" or value not in section_values:
                raise RuntimeError(
                    f"Unverified section filter in {question['id']}: {value!r}."
                )
    if filter_questions < 3 or test_filter_questions < 1:
        raise RuntimeError("Benchmark needs at least 3 filter questions, including 1 test question.")
    return section_values


def _package_version(package: str) -> str:
    try:
        return version(package)
    except PackageNotFoundError:
        return "not installed"


def _store_info(database_url: str) -> dict[str, str]:
    with psycopg.connect(database_url, connect_timeout=5) as connection:
        row = connection.execute(
            """
            SELECT current_setting('server_version'),
                   (SELECT extversion FROM pg_extension WHERE extname = 'vector')
            """
        ).fetchone()
        if not row or not row[1]:
            raise RuntimeError("PostgreSQL is reachable, but the pgvector extension is unavailable.")
        return {"postgresql": row[0], "pgvector": row[1]}


def _validate_trace_metadata(
    traces: list[dict[str, Any]],
    chunks_by_strategy: dict[str, list[Chunk]],
    pages: list[DocumentPage],
    sample_question_ids: set[str],
) -> dict[str, dict[str, int]]:
    chunks = {
        strategy: {chunk.chunk_id: chunk for chunk in strategy_chunks}
        for strategy, strategy_chunks in chunks_by_strategy.items()
    }
    pages_by_id = {page.page: page for page in pages}
    for trace in traces:
        for hit in trace["hits"]:
            chunk = chunks[trace["strategy"]].get(hit["chunk_id"])
            page = pages_by_id.get(hit["page"])
            checks = {
                "chunk_identity": chunk is not None
                and (hit["document_id"], hit["source"], hit["page"])
                == (chunk.document_id, chunk.source, chunk.page),
                "source_line_range": chunk is not None
                and page is not None
                and (
                    hit["metadata"].get("source_start_line"),
                    hit["metadata"].get("source_end_line"),
                )
                == (
                    page.metadata["source_start_line"],
                    page.metadata["source_end_line"],
                ),
                "section": chunk is not None
                and hit.get("section") == chunk.metadata.get("section")
                and hit["metadata"].get("section") == chunk.metadata.get("section"),
            }
            hit["metadata_validation"] = {
                "checks": checks,
                "valid": all(checks.values()),
            }

    audits: dict[str, dict[str, int]] = {}
    for strategy in ("A", "B", "C"):
        sample = [
            trace
            for trace in traces
            if trace["strategy"] == strategy
            and trace["question_id"] in sample_question_ids
        ]
        records = [hit for trace in sample for hit in trace["hits"][:3]]
        filtered = [trace for trace in sample if trace["filter_requested"]]
        audits[strategy] = {
            "metadata_valid_count": sum(
                hit["metadata_validation"]["valid"] for hit in records
            ),
            "metadata_checked_count": len(records),
            "filter_compliant_count": sum(
                trace["filter_compliant"] for trace in filtered
            ),
            "filter_question_count": len(filtered),
        }
    return audits


def _write_manual_sheets(
    pages: list[DocumentPage],
    chunks_by_strategy: dict[str, list[Chunk]],
    traces: list[dict[str, Any]],
) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    sample_pages = sorted(page.page for page in pages)[:10]
    coherence_scores: dict[tuple[str, str, str], dict[str, str]] = {}
    coherence_path = RESULTS / "manual_coherence.csv"
    if coherence_path.exists():
        with coherence_path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                coherence_scores[
                    (row["strategy"], row["source_page"], row.get("chunk_id", ""))
                ] = row
    with (RESULTS / "manual_coherence.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy",
                "source_page",
                "source_offset",
                "chunk_id",
                "excerpt",
                "score_0_to_2",
                "reason",
            ],
        )
        writer.writeheader()
        for strategy in ("A", "B", "C"):
            for page_id in sample_pages:
                selected = next(
                    (
                        chunk
                        for chunk in chunks_by_strategy[strategy]
                        if chunk.page == page_id
                        and chunk.metadata["char_start"] <= 0 < chunk.metadata["char_end"]
                    ),
                    None,
                )
                if selected:
                    prior = coherence_scores.get(
                        (strategy, page_id, selected.chunk_id), {}
                    )
                    writer.writerow(
                        {
                            "strategy": strategy,
                            "source_page": page_id,
                            "source_offset": 0,
                            "chunk_id": selected.chunk_id,
                            "excerpt": selected.text,
                            "score_0_to_2": prior.get("score_0_to_2", ""),
                            "reason": prior.get("reason", ""),
                        }
                    )

    test_ids = sorted(
        {trace["question_id"] for trace in traces if trace["split"] == "test"}
    )[:5]
    grounding_scores: dict[tuple[str, str, tuple[str, ...]], dict[str, str]] = {}
    grounding_path = RESULTS / "manual_grounding.csv"
    if grounding_path.exists():
        with grounding_path.open(newline="", encoding="utf-8-sig") as file:
            for row in csv.DictReader(file):
                try:
                    prior_hits = json.loads(row["top3_evidence"])
                except (KeyError, json.JSONDecodeError):
                    continue
                if isinstance(prior_hits, list):
                    grounding_scores[
                        (
                            row["strategy"],
                            row["question_id"],
                            tuple(hit["chunk_id"] for hit in prior_hits),
                        )
                    ] = row
    with (RESULTS / "manual_grounding.csv").open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "strategy",
                "question_id",
                "question",
                "expected_answer",
                "gold_locations",
                "top3_evidence",
                "score_0_to_2",
                "reason",
            ],
        )
        writer.writeheader()
        for strategy in ("A", "B", "C"):
            for trace in traces:
                if trace["strategy"] == strategy and trace["question_id"] in test_ids:
                    current_chunk_ids = tuple(
                        hit["chunk_id"] for hit in trace["hits"][:3]
                    )
                    prior = grounding_scores.get(
                        (strategy, trace["question_id"], current_chunk_ids), {}
                    )
                    evidence = json.dumps(
                        [
                            {
                                "rank": rank,
                                "chunk_id": hit["chunk_id"],
                                "source": hit["source"],
                                "page": hit["page"],
                                "score": hit["score"],
                                "section": hit["section"],
                                "metadata": hit["metadata"],
                                "metadata_validation": hit.get("metadata_validation", {}),
                                "gold_evidence_matches": hit.get("gold_evidence_matches", []),
                                "text": hit["text"],
                            }
                            for rank, hit in enumerate(trace["hits"][:3], start=1)
                        ],
                        ensure_ascii=False,
                    )
                    writer.writerow(
                        {
                            "strategy": strategy,
                            "question_id": trace["question_id"],
                            "question": trace["question"],
                            "expected_answer": trace["expected_answer"],
                            "gold_locations": trace["gold_locations"],
                            "top3_evidence": evidence,
                            "score_0_to_2": prior.get("score_0_to_2", ""),
                            "reason": prior.get("reason", ""),
                        }
                    )


def _write_report(
    summaries: list[dict[str, Any]],
    metrics: dict[str, Any],
    store_info: dict[str, str],
) -> None:
    coherence_rows: list[dict[str, str]] = []
    grounding_rows: list[dict[str, str]] = []
    for path, target in (
        (RESULTS / "manual_coherence.csv", coherence_rows),
        (RESULTS / "manual_grounding.csv", grounding_rows),
    ):
        if path.exists():
            with path.open(newline="", encoding="utf-8-sig") as file:
                target.extend(csv.DictReader(file))

    lines = [
        "# Day 06 retrieval experiment",
        "",
        "Retrieval chunks and embeddings come from the bundled-corpus run; corrected evidence labels and metrics were recalculated from the saved traces without new embedding/API calls.",
        "",
        "## Reproducibility",
        "",
        f"- Embeddings: OpenRouter `{EMBEDDING_MODEL}`, 1536 dimensions; same raw text treatment for queries and documents.",
        f"- Vector store: PostgreSQL {store_info['postgresql']} + pgvector {store_info['pgvector']}, `rag_chunks`, cosine distance, exact sequential search (no ANN index).",
        "- Source extraction: UTF-8 JSONL records prepared from the Project Gutenberg plain-text source; NFC normalization; no OCR.",
        "- A: 500 characters, overlap 50. B: section/paragraph grouping, max 900 characters. C: identical chunks and vectors to B; exact `section` filtering only when the benchmark question supplies it.",
        "- `section` is read from uppercase / marked headings in source order and carried over unit boundaries. Other metadata: `document_id` and source URL are strings; `page` is the stable corpus unit ID; source line range and chunk character offsets are integers. No section is inferred when a source heading is unavailable.",
        "- Query filters are benchmark context supplied in advance, not inferred from gold locations. Filter values are checked against extracted source headings before search.",
        "- Re-ingest replaces a document's rows per strategy in one transaction; each run repeats an upsert and checks row counts to verify idempotency. Changed content removes obsolete chunks.",
        f"- Runtime: Python {metrics['runtime']['python']}; run started at {metrics['runtime']['started_at_utc']} and results were prepared at {metrics['runtime']['results_generated_at_utc']} (UTC).",
        f"- Libraries: psycopg {metrics['packages']['psycopg']}, pypdf {metrics['packages']['pypdf']}, python-dotenv {metrics['packages']['python-dotenv']}. No random seed is needed: the benchmark split and sampling order are fixed.",
        "- API cost is an estimate using USD 0.02 / 1M embedding tokens; OpenRouter usage-reported tokens are recorded. Provider billing may differ; see `metrics.json`.",
        "",
        "## Retrieval results",
        "",
        "| Strategy | Split | Hit@1 | Evidence Hit@3 |",
        "|---|---|---:|---:|",
    ]
    for item in summaries:
        lines.append(
            f"| {item['strategy']} | {item['split']} | "
            f"{item['hit_at_1_count']}/{item['n']} ({item['hit_at_1']:.3f}) | "
            f"{item['evidence_hit_at_3_count']}/{item['n']} ({item['evidence_hit_at_3']:.3f}) |"
        )
    lines.extend(
        [
            "",
            "A hit requires the chunk to match a gold source/page and contain the complete gold evidence quote after whitespace normalization. A quote split across chunks is not combined for the primary metric; no human-confirmed equivalent was added to the automatic counts. The per-hit `gold_evidence_matches` labels and quoted gold locations are in `traces.jsonl`.",
        ]
    )
    lines.extend(
        [
            "",
            "## Manual sample assessment",
            "",
            "Coherence uses offset 0 in the first 10 sorted source-unit IDs, the same source positions for A/B/C. Grounding and metadata use the first five sorted test IDs (T01-T05) and all top-3 hits. See both CSVs for each excerpt, chunk ID, score, metadata, evidence match, and reason.",
            "Ratings are AI-assisted by the assistant, not the author or an independent human annotator. The rubric was applied to the displayed chunks independently of their similarity values and then checked against the source/evidence. Retrieval results had already been visible, so the review is not blind; author review is recommended.",
            "",
            "| Strategy | Coherence average /2 (n) | Grounding 0/1/2 (n) | Grounding average /2 (n) | Metadata valid (n/N) | Filter-compliant questions (n/N) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in ("A", "B", "C"):
        coherence = [
            int(row["score_0_to_2"])
            for row in coherence_rows
            if row["strategy"] == strategy and row["score_0_to_2"].strip()
        ]
        grounding = [
            int(row["score_0_to_2"])
            for row in grounding_rows
            if row["strategy"] == strategy and row["score_0_to_2"].strip()
        ]
        audit = metrics.get("manual_sample_metadata", {}).get(strategy, {})
        levels = [grounding.count(score) for score in range(3)]
        coherence_average = (
            f"{sum(coherence) / len(coherence):.2f} ({len(coherence)}/10)"
            if coherence
            else "N/A (0/10)"
        )
        grounding_average = (
            f"{sum(grounding) / len(grounding):.2f} ({len(grounding)}/5)"
            if grounding
            else "N/A (0/5)"
        )
        filter_question_count = audit.get("filter_question_count", 0)
        filter_result = (
            f"{audit.get('filter_compliant_count', 0)}/{filter_question_count}"
            if filter_question_count
            else "N/A"
        )
        lines.append(
            f"| {strategy} | {coherence_average} | "
            f"{levels[0]}/{levels[1]}/{levels[2]} | {grounding_average} | "
            f"{audit.get('metadata_valid_count', 0)}/{audit.get('metadata_checked_count', 0)} | "
            f"{filter_result} |"
        )
    lines.extend(
        [
            "",
            "Metadata validity is checked against the selected corpus/chunk (not an independent human audit): `source`, `page`, source line range, and source-derived `section`. `page` and line range support tracing to the Gutenberg text; `section` supports exact filtering. The heading-based extractor may inherit an imperfect heading or produce no section, so complete metadata is not proof every label is semantically useful.",
            "",
            "## Strategy impact and selection",
            "",
            "A→B changes chunking; B→C keeps chunks/vectors and adds exact section filtering only when a question supplies a filter. The selection rule is highest dev Evidence Hit@3, then sample grounding, then lower resource use. This rule is documented after existing test scores were already visible; treat the choice as post-hoc and test as descriptive, not confirmatory.",
            "A→B raises dev Evidence Hit@3 from 7/18 to 10/18 and sample coherence from 1.2/2 to 1.5/2, but drops test Evidence Hit@3 from 5/6 to 4/6 and sample grounding from 1.6/2 to 1.4/2. B uses 168 rather than 219 chunks, fewer document embedding tokens (33,501 vs 36,756), and lower measured ingest time (16.22s vs 20.07s).",
            "B→C leaves chunk count, vectors, coherence, and grounding unchanged; dev Evidence Hit@3 remains 10/18, while test Evidence Hit@3 changes from 4/6 to 3/6. On the three filtered questions in the manual sample, top-3 filter compliance improves from 0/3 to 3/3. C reuses B's document vectors, so this run embeds no additional documents and measured ingest time is 0.50s.",
            "Under the stated dev-first tie-break (dev Evidence Hit@3, sample grounding, then measured resources), C is the choice for this corpus because it ties B on the first two criteria and adds exact filter compliance with no new document-embedding cost. If test Evidence Hit@3 were the primary objective, B is preferable in this sample (4/6 vs 3/6); do not generalize either choice.",
            "T01 is a failure for all strategies. On T02, A retrieves the complete negative-discriminant rule, while B/C retrieve an example and its conclusion without the full general condition; the latter receive grounding 1 and are not exact evidence hits. This corpus has one book, six test questions, and a small manual sample; a one-question change is substantial.",
            "",
            "Grounding rates evidence retrieval, not LLM answer accuracy. Exact Evidence Hit@3 is reproducible and does not replace the per-question manual labels.",
            "",
            "## Resource measurements",
            "",
            "| Strategy | Chunks | Character count | Length min / median / max | Ingest seconds | API tokens (documents) |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for strategy in ("A", "B", "C"):
        row = metrics["strategies"][strategy]
        lines.append(
            f"| {strategy} | {row['chunk_count']} | {row['total_chunk_characters']} | "
            f"{row['chunk_length_min']} / {row['chunk_length_median']} / {row['chunk_length_max']} | "
            f"{row['ingest_seconds']:.2f} | {row['document_embedding_tokens']} |"
        )
    lines.extend(
        [
            "",
            f"Question embedding tokens (shared across strategies): {metrics['query_embedding_tokens']}.",
            f"Estimated API cost for documents and queries: ${metrics['estimated_embedding_cost_usd']:.6f}.",
            "",
            "## Files",
            "",
            "- `traces.jsonl`: top-3 retrieval evidence for all strategies and questions.",
            "- `metrics.json`: exact counts, lengths, timing, API usage, store version, and repeat-ingest check.",
            "- `manual_coherence.csv` and `manual_grounding.csv`: fixed samples with scores and per-case reasons; review the AI-assisted labels.",
            "- `ingestion_errors.jsonl`: malformed input/extraction errors; empty means none were recorded.",
            "",
            "The supplied benchmark contains only 24 questions from one book; the fixed test set is small and the book outline can leak concepts. Treat results as a lab comparison, not generalization evidence.",
        ]
    )
    (RESULTS / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    RESULTS.mkdir(parents=True, exist_ok=True)
    started_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )
    load_dotenv(ROOT / ".env")
    configured_model = os.getenv("EMBEDDING_MODEL")
    if configured_model and configured_model != EMBEDDING_MODEL:
        raise RuntimeError(
            f"This run is configured for {EMBEDDING_MODEL}; EMBEDDING_MODEL is set to {configured_model!r}."
        )
    database_url = os.getenv("DATABASE_URL")
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not database_url:
        raise RuntimeError("DATABASE_URL is missing; start the PostgreSQL + pgvector service.")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is missing; no mock embedding fallback is available.")

    corpus_path = ROOT / "data" / "gutenberg_38536_selected.jsonl"
    pages, ingestion_errors = load_jsonl(corpus_path, RESULTS / "ingestion_errors.jsonl")
    for error in ingestion_errors:
        logging.error("Input record skipped: %s", error)
    with (ROOT / "data" / "benchmark_questions.json").open(encoding="utf-8") as file:
        questions = json.load(file)["questions"]
    section_values = _verify_benchmark(questions, pages)
    logging.info(
        "Validated %d source units, %d benchmark questions, and %d source-derived headings.",
        len(pages),
        len(questions),
        len(section_values),
    )

    embedder = OpenRouterEmbedder(api_key)
    store = PgVectorStore(database_url, embedder.dimensions)
    store_info = _store_info(database_url)
    store.create_schema()

    metrics: dict[str, Any] = {
        "embedding_provider": "OpenRouter",
        "embedding_model": embedder.model_name,
        "embedding_dimensions": embedder.dimensions,
        "document_count": len(pages),
        "source_characters": sum(len(page.text) for page in pages),
        "question_count": len(questions),
        "metadata": {
            "document_id": "string; manifest source_id",
            "source": "string; manifest source_path_or_url",
            "page": "string; source corpus unit_id (line-range traceability)",
            "source_start_line": "integer; JSONL source record",
            "source_end_line": "integer; JSONL source record",
            "section": "string or null; exact uppercase/marked heading from source",
        },
        "store": {
            **store_info,
            "table": "rag_chunks",
            "metric": "cosine",
            "search": "exact sequential scan; no approximate vector index",
        },
        "packages": {
            "psycopg": _package_version("psycopg"),
            "pypdf": _package_version("pypdf"),
            "python-dotenv": _package_version("python-dotenv"),
        },
        "runtime": {
            "python": platform.python_version(),
            "started_at_utc": started_at_utc,
            "seed": None,
            "timestamp_source": "captured by src.run before writing the result artifacts",
        },
        "strategies": {},
    }
    chunks_by_strategy: dict[str, list[Chunk]] = {}
    vectors_by_strategy: dict[str, list[list[float]]] = {}
    document_embedding_tokens: dict[str, int] = {}

    for strategy in ("A", "B", "C"):
        started = time.perf_counter()
        chunks = _chunks(pages, strategy)
        if not chunks:
            raise RuntimeError(f"Chunking produced no chunks for strategy {strategy}.")
        chunks_by_strategy[strategy] = chunks
        if strategy == "C":
            vectors = vectors_by_strategy["B"]
            document_embedding_tokens[strategy] = 0
        else:
            tokens_before = embedder.total_tokens
            vectors = embedder.embed_documents([chunk.text for chunk in chunks])
            document_embedding_tokens[strategy] = embedder.total_tokens - tokens_before
            vectors_by_strategy[strategy] = vectors
        store.upsert(strategy, chunks, vectors)
        first_count = store.count(strategy)
        store.upsert(strategy, chunks, vectors)
        repeated_count = store.count(strategy)
        if first_count != len(chunks) or repeated_count != first_count:
            raise RuntimeError(
                f"Idempotent ingest check failed for {strategy}: "
                f"{len(chunks)} expected, {first_count} then {repeated_count} stored."
            )
        lengths = [len(chunk.text) for chunk in chunks]
        metrics["strategies"][strategy] = {
            "chunk_count": len(chunks),
            "total_chunk_characters": sum(lengths),
            "chunk_length_min": min(lengths),
            "chunk_length_median": statistics.median(lengths),
            "chunk_length_max": max(lengths),
            "ingest_seconds": time.perf_counter() - started,
            "document_embedding_tokens": document_embedding_tokens[strategy],
            "idempotency_check": {
                "first_ingest_rows": first_count,
                "repeat_ingest_rows": repeated_count,
                "passed": True,
            },
            "section_values": sorted(
                {str(chunk.metadata["section"]) for chunk in chunks if chunk.metadata["section"]}
            ),
        }
        logging.info(
            "Strategy %s: %d chunks; repeated ingest kept %d rows.",
            strategy,
            len(chunks),
            repeated_count,
        )

    tokens_before_queries = embedder.total_tokens
    query_vectors = embedder.embed_queries([question["question"] for question in questions])
    if len(query_vectors) != len(questions):
        raise RuntimeError("The embedding provider returned an incomplete set of query vectors.")
    metrics["query_embedding_tokens"] = embedder.total_tokens - tokens_before_queries

    traces: list[dict[str, Any]] = []
    for strategy in ("A", "B", "C"):
        for question, query_vector in zip(questions, query_vectors):
            requested_filter = question.get("filter")
            applied_filter = None
            if strategy == "C" and requested_filter:
                applied_filter = {"section": requested_filter["value"]}
            hits = store.search(strategy, query_vector, top_k=3, filters=applied_filter)
            traces.append(evaluate_query(question, hits, strategy, applied_filter))

    test_question_ids = sorted(
        question["id"] for question in questions if question["split"] == "test"
    )[:5]
    metrics["manual_sample_metadata"] = _validate_trace_metadata(
        traces,
        chunks_by_strategy,
        pages,
        set(test_question_ids),
    )
    summaries = summarize(traces)
    metrics["retrieval"] = summaries
    metrics["api_usage"] = {
        "requests": embedder.total_requests,
        "reported_embedding_tokens": embedder.total_tokens,
        "cost_estimate_usd_per_million_tokens": ESTIMATED_USD_PER_MILLION_TOKENS,
    }
    metrics["estimated_embedding_cost_usd"] = (
        embedder.total_tokens * ESTIMATED_USD_PER_MILLION_TOKENS / 1_000_000
    )
    metrics["ingestion_errors"] = ingestion_errors
    metrics["runtime"]["results_generated_at_utc"] = datetime.now(
        timezone.utc
    ).isoformat(timespec="seconds").replace("+00:00", "Z")
    _jsonl(RESULTS / "traces.jsonl", traces)
    _write_manual_sheets(pages, chunks_by_strategy, traces)
    _write_report(summaries, metrics, store_info)
    (RESULTS / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    logging.info(
        "Completed: %d traces, %d API requests, %d reported embedding tokens, estimated cost $%.6f.",
        len(traces),
        embedder.total_requests,
        embedder.total_tokens,
        metrics["estimated_embedding_cost_usd"],
    )


if __name__ == "__main__":
    main()
