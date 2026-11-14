"""OpenRouter embedding adapter; failures are explicit and never use mock vectors."""
import json
import math
import os
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OpenRouterEmbedder:
    model_name = "openai/text-embedding-3-small"
    dimensions = 1536
    endpoint = "https://openrouter.ai/api/v1/embeddings"
    batch_size = 32

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.getenv("OPENROUTER_API_KEY", "")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is missing; no embedding fallback is available.")
        self.total_tokens = 0
        self.total_requests = 0

    def _embed(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model_name, "input": texts}).encode("utf-8")
        request = Request(
            self.endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/",
                "X-Title": "Day 06 Data Ingestion Lab",
            },
            method="POST",
        )
        try:
            with urlopen(request, timeout=90) as response:
                body = json.loads(response.read().decode("utf-8"))
        except HTTPError as error:
            details = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"OpenRouter embedding request failed ({error.code}): {details}") from error
        except (URLError, TimeoutError, json.JSONDecodeError) as error:
            raise RuntimeError(f"OpenRouter embedding request failed: {error}") from error

        data = body.get("data")
        if not isinstance(data, list) or len(data) != len(texts):
            raise RuntimeError("OpenRouter returned an invalid embedding response: item count mismatch.")
        vectors: list[list[float] | None] = [None] * len(texts)
        for item in data:
            index = item.get("index")
            vector = item.get("embedding")
            if not isinstance(index, int) or not 0 <= index < len(texts) or vectors[index] is not None:
                raise RuntimeError("OpenRouter returned invalid or duplicate embedding indexes.")
            if not isinstance(vector, list) or len(vector) != self.dimensions:
                raise RuntimeError(
                    f"Expected {self.dimensions}-dimensional embeddings from {self.model_name}."
                )
            if any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in vector):
                raise RuntimeError("OpenRouter returned a non-finite embedding value.")
            vectors[index] = [float(value) for value in vector]
        if any(vector is None for vector in vectors):
            raise RuntimeError("OpenRouter omitted one or more embedding indexes.")

        usage = body.get("usage", {})
        tokens = usage.get("total_tokens", usage.get("prompt_tokens"))
        if not isinstance(tokens, int) or tokens < 0:
            raise RuntimeError("OpenRouter response omitted a valid embedding token count.")
        self.total_tokens += tokens
        self.total_requests += 1
        return [vector for vector in vectors if vector is not None]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for offset in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed(texts[offset : offset + self.batch_size]))
        return vectors

    def embed_queries(self, questions: list[str]) -> list[list[float]]:
        return self.embed_documents(questions)

    def embed_query(self, question: str) -> list[float]:
        return self._embed([question])[0]
