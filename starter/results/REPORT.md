# Day 06 retrieval experiment

Retrieval chunks and embeddings come from the bundled-corpus run; corrected evidence labels and metrics were recalculated from the saved traces without new embedding/API calls.

## Reproducibility

- Embeddings: OpenRouter `openai/text-embedding-3-small`, 1536 dimensions; same raw text treatment for queries and documents.
- Vector store: PostgreSQL 16.16 (Debian 16.16-1.pgdg12+1) + pgvector 0.8.7, `rag_chunks`, cosine distance, exact sequential search (no ANN index).
- Source extraction: UTF-8 JSONL records prepared from the Project Gutenberg plain-text source; NFC normalization; no OCR.
- A: 500 characters, overlap 50. B: section/paragraph grouping, max 900 characters. C: identical chunks and vectors to B; exact `section` filtering only when the benchmark question supplies it.
- `section` is read from uppercase / marked headings in source order and carried over unit boundaries. Other metadata: `document_id` and source URL are strings; `page` is the stable corpus unit ID; source line range and chunk character offsets are integers. No section is inferred when a source heading is unavailable.
- Query filters are benchmark context supplied in advance, not inferred from gold locations. Filter values are checked against extracted source headings before search.
- Re-ingest replaces a document's rows per strategy in one transaction; each run repeats an upsert and checks row counts to verify idempotency. Changed content removes obsolete chunks.
- Runtime: Python 3.11.5; run started at 2026-11-14T04:45:03Z and results were prepared at 2026-11-14T04:45:54Z (UTC).
- Libraries: psycopg 3.3.8, pypdf 6.21.4, python-dotenv 1.2.4. No random seed is needed: the benchmark split and sampling order are fixed.
- API cost is an estimate using USD 0.02 / 1M embedding tokens; OpenRouter usage-reported tokens are recorded. Provider billing may differ; see `metrics.json`.

## Retrieval results

| Strategy | Split | Hit@1 | Evidence Hit@3 |
|---|---|---:|---:|
| A | dev | 4/18 (0.222) | 7/18 (0.389) |
| A | test | 3/6 (0.500) | 5/6 (0.833) |
| B | dev | 4/18 (0.222) | 10/18 (0.556) |
| B | test | 3/6 (0.500) | 4/6 (0.667) |
| C | dev | 4/18 (0.222) | 10/18 (0.556) |
| C | test | 3/6 (0.500) | 3/6 (0.500) |

A hit requires the chunk to match a gold source/page and contain the complete gold evidence quote after whitespace normalization. A quote split across chunks is not combined for the primary metric; no human-confirmed equivalent was added to the automatic counts. The per-hit `gold_evidence_matches` labels and quoted gold locations are in `traces.jsonl`.

## Manual sample assessment

Coherence uses offset 0 in the first 10 sorted source-unit IDs, the same source positions for A/B/C. Grounding and metadata use the first five sorted test IDs (T01-T05) and all top-3 hits. See both CSVs for each excerpt, chunk ID, score, metadata, evidence match, and reason.
Ratings are AI-assisted by the assistant, not the author or an independent human annotator. The rubric was applied to the displayed chunks independently of their similarity values and then checked against the source/evidence. Retrieval results had already been visible, so the review is not blind; author review is recommended.

| Strategy | Coherence average /2 (n) | Grounding 0/1/2 (n) | Grounding average /2 (n) | Metadata valid (n/N) | Filter-compliant questions (n/N) |
|---|---:|---:|---:|---:|---:|
| A | 1.20 (10/10) | 1/0/4 | 1.60 (5/5) | 15/15 | 1/3 |
| B | 1.50 (10/10) | 1/1/3 | 1.40 (5/5) | 15/15 | 0/3 |
| C | 1.50 (10/10) | 1/1/3 | 1.40 (5/5) | 15/15 | 3/3 |

Metadata validity is checked against the selected corpus/chunk (not an independent human audit): `source`, `page`, source line range, and source-derived `section`. `page` and line range support tracing to the Gutenberg text; `section` supports exact filtering. The heading-based extractor may inherit an imperfect heading or produce no section, so complete metadata is not proof every label is semantically useful.

## Strategy impact and selection

A→B changes chunking; B→C keeps chunks/vectors and adds exact section filtering only when a question supplies a filter. The selection rule is highest dev Evidence Hit@3, then sample grounding, then lower resource use. This rule is documented after existing test scores were already visible; treat the choice as post-hoc and test as descriptive, not confirmatory.
A→B raises dev Evidence Hit@3 from 7/18 to 10/18 and sample coherence from 1.2/2 to 1.5/2, but drops test Evidence Hit@3 from 5/6 to 4/6 and sample grounding from 1.6/2 to 1.4/2. B uses 168 rather than 219 chunks, fewer document embedding tokens (33,501 vs 36,756), and lower measured ingest time (16.22s vs 20.07s).
B→C leaves chunk count, vectors, coherence, and grounding unchanged; dev Evidence Hit@3 remains 10/18, while test Evidence Hit@3 changes from 4/6 to 3/6. On the three filtered questions in the manual sample, top-3 filter compliance improves from 0/3 to 3/3. C reuses B's document vectors, so this run embeds no additional documents and measured ingest time is 0.50s.
Under the stated dev-first tie-break (dev Evidence Hit@3, sample grounding, then measured resources), C is the choice for this corpus because it ties B on the first two criteria and adds exact filter compliance with no new document-embedding cost. If test Evidence Hit@3 were the primary objective, B is preferable in this sample (4/6 vs 3/6); do not generalize either choice.
T01 is a failure for all strategies. On T02, A retrieves the complete negative-discriminant rule, while B/C retrieve an example and its conclusion without the full general condition; the latter receive grounding 1 and are not exact evidence hits. This corpus has one book, six test questions, and a small manual sample; a one-question change is substantial.

Grounding rates evidence retrieval, not LLM answer accuracy. Exact Evidence Hit@3 is reproducible and does not replace the per-question manual labels.

## Resource measurements

| Strategy | Chunks | Character count | Length min / median / max | Ingest seconds | API tokens (documents) |
|---|---:|---:|---:|---:|---:|
| A | 219 | 93467 | 77 / 500 / 500 | 22.77 | 36756 |
| B | 168 | 85187 | 15 / 595.0 / 899 | 21.87 | 33501 |
| C | 168 | 85187 | 15 / 595.0 / 899 | 0.43 | 0 |

Question embedding tokens (shared across strategies): 1069.
Estimated API cost for documents and queries: $0.001427.

## Files

- `traces.jsonl`: top-3 retrieval evidence for all strategies and questions.
- `metrics.json`: exact counts, lengths, timing, API usage, store version, and repeat-ingest check.
- `manual_coherence.csv` and `manual_grounding.csv`: fixed samples with scores and per-case reasons; review the AI-assisted labels.
- `ingestion_errors.jsonl`: malformed input/extraction errors; empty means none were recorded.

The supplied benchmark contains only 24 questions from one book; the fixed test set is small and the book outline can leak concepts. Treat results as a lab comparison, not generalization evidence.
