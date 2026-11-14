"""Persistent exact cosine search using PostgreSQL and pgvector."""
import json
import math
from typing import Any

import psycopg

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if not database_url:
            raise RuntimeError("DATABASE_URL is missing; the vector store is required.")
        if dimensions < 1:
            raise ValueError("Embedding dimensions must be positive.")
        self.database_url = database_url
        self.dimensions = dimensions

    def create_schema(self) -> None:
        with psycopg.connect(self.database_url, connect_timeout=5) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""
                CREATE TABLE IF NOT EXISTS rag_chunks (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page TEXT NOT NULL,
                    section TEXT,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS rag_chunks_section_idx ON rag_chunks (section)"
            )
            row = connection.execute(
                "SELECT vector_dims(embedding) FROM rag_chunks LIMIT 1"
            ).fetchone()
            if row is not None and row[0] != self.dimensions:
                raise RuntimeError(
                    f"Stored vectors are {row[0]}D, but the selected model requires {self.dimensions}D. "
                    "Use a fresh database/schema or migrate the existing vectors explicitly."
                )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and embedding counts differ.")
        if not chunks:
            raise ValueError(f"No chunks to ingest for strategy {strategy}.")
        document_ids = {chunk.document_id for chunk in chunks}
        if len(document_ids) != 1:
            raise ValueError("Each upsert must contain chunks from exactly one document.")
        rows: list[tuple[Any, ...]] = []
        for chunk, vector in zip(chunks, vectors):
            if len(vector) != self.dimensions:
                raise ValueError(f"Embedding {chunk.chunk_id} has the wrong dimension.")
            if any(not math.isfinite(value) for value in vector):
                raise ValueError(f"Embedding {chunk.chunk_id} contains a non-finite value.")
            section = chunk.metadata.get("section")
            rows.append(
                (
                    strategy,
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.source,
                    chunk.page,
                    section,
                    chunk.text,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    "[" + ",".join(repr(float(value)) for value in vector) + "]",
                )
            )

        statement = """
            INSERT INTO rag_chunks
                (strategy, chunk_id, document_id, source, page, section, text, metadata, embedding)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)
            ON CONFLICT (strategy, chunk_id) DO UPDATE SET
                document_id = EXCLUDED.document_id,
                source = EXCLUDED.source,
                page = EXCLUDED.page,
                section = EXCLUDED.section,
                text = EXCLUDED.text,
                metadata = EXCLUDED.metadata,
                embedding = EXCLUDED.embedding
        """
        with psycopg.connect(self.database_url, connect_timeout=5) as connection:
            connection.execute(
                "DELETE FROM rag_chunks WHERE strategy = %s AND document_id = %s",
                (strategy, next(iter(document_ids))),
            )
            with connection.cursor() as cursor:
                cursor.executemany(statement, rows)

    def count(self, strategy: str) -> int:
        with psycopg.connect(self.database_url, connect_timeout=5) as connection:
            row = connection.execute(
                "SELECT count(*) FROM rag_chunks WHERE strategy = %s",
                (strategy,),
            ).fetchone()
        if row is None:
            raise RuntimeError(f"Unable to count stored chunks for strategy {strategy}.")
        return int(row[0])

    def search(
        self,
        strategy: str,
        query_vector: list[float],
        top_k: int = 3,
        filters: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        if len(query_vector) != self.dimensions:
            raise ValueError("Query embedding dimension does not match the stored model.")
        if top_k < 1:
            raise ValueError("top_k must be positive.")
        if filters and set(filters) != {"section"}:
            raise ValueError("Only exact-match filtering on the source-derived 'section' is supported.")
        if filters and not isinstance(filters["section"], str):
            raise ValueError("The section filter value must be a string.")

        vector_literal = "[" + ",".join(repr(float(value)) for value in query_vector) + "]"
        where = "strategy = %s"
        parameters: list[Any] = [strategy]
        if filters:
            where += " AND section = %s"
            parameters.append(filters["section"])
        parameters.extend([vector_literal, top_k])
        with psycopg.connect(self.database_url, connect_timeout=5) as connection:
            rows = connection.execute(
                f"""
                SELECT chunk_id, document_id, source, page, section, text, metadata,
                       1 - (embedding <=> %s::vector) AS cosine_similarity
                FROM rag_chunks
                WHERE {where}
                ORDER BY embedding <=> %s::vector
                LIMIT %s
                """,
                # The vector is first in SELECT, then strategy/filter, then the ORDER BY vector.
                [vector_literal, *parameters],
            ).fetchall()
        return [
            {
                "chunk_id": row[0],
                "document_id": row[1],
                "source": row[2],
                "page": row[3],
                "section": row[4],
                "text": row[5],
                "metadata": row[6],
                "score": float(row[7]),
            }
            for row in rows
        ]
