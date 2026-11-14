"""Gold-source retrieval metrics; subjective grounding/coherence stay manual."""
import re
from typing import Any


def _gold_locations(question: dict[str, Any]) -> list[dict[str, str]]:
    locations = [
        {
            "document_id": question["gold_source_id"],
            "page": question["gold_page"],
            "evidence_quote": question["evidence_quote"],
        }
    ]
    for source in question.get("additional_gold_sources", []):
        locations.append(
            {
                "document_id": source["gold_source_id"],
                "page": source["gold_page"],
                "evidence_quote": source["evidence_quote"],
            }
        )
    return locations


def _normalize_evidence(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _matching_evidence(
    question: dict[str, Any], hit: dict[str, Any]
) -> list[dict[str, str]]:
    normalized_text = _normalize_evidence(hit["text"])
    return [
        location
        for location in _gold_locations(question)
        if (hit["document_id"], hit["page"])
        == (location["document_id"], location["page"])
        and _normalize_evidence(location["evidence_quote"]) in normalized_text
    ]


def evaluate_query(
    question: dict[str, Any],
    hits: list[dict[str, Any]],
    strategy: str,
    applied_filter: dict[str, str] | None,
) -> dict[str, Any]:
    gold_locations = _gold_locations(question)
    annotated_hits = [
        {
            **hit,
            "gold_evidence_matches": _matching_evidence(question, hit),
        }
        for hit in hits
    ]
    expected_filter = question.get("filter")
    if expected_filter:
        expected_section = expected_filter["value"]
        filter_compliant = bool(annotated_hits) and all(
            hit.get("section") == expected_section for hit in annotated_hits[:3]
        )
    else:
        filter_compliant = True
    expected_applied_filter = (
        {"section": expected_filter["value"]} if expected_filter else None
    )
    if strategy == "C":
        filter_compliant = filter_compliant and applied_filter == expected_applied_filter
    return {
        "question_id": question["id"],
        "split": question["split"],
        "strategy": strategy,
        "question": question["question"],
        "expected_answer": question["expected_answer"],
        "filter_requested": expected_filter,
        "filter_applied": applied_filter,
        "filter_compliant": filter_compliant,
        "gold_locations": sorted(
            gold_locations,
            key=lambda item: (item["document_id"], item["page"]),
        ),
        "hit_at_1": bool(annotated_hits)
        and bool(annotated_hits[0]["gold_evidence_matches"]),
        "evidence_hit_at_3": any(
            hit["gold_evidence_matches"] for hit in annotated_hits[:3]
        ),
        "hits": annotated_hits,
    }


def summarize(traces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for strategy in ("A", "B", "C"):
        for split in ("dev", "test"):
            group = [
                trace
                for trace in traces
                if trace["strategy"] == strategy and trace["split"] == split
            ]
            if not group:
                continue
            summaries.append(
                {
                    "strategy": strategy,
                    "split": split,
                    "n": len(group),
                    "hit_at_1_count": sum(trace["hit_at_1"] for trace in group),
                    "hit_at_1": sum(trace["hit_at_1"] for trace in group) / len(group),
                    "evidence_hit_at_3_count": sum(
                        trace["evidence_hit_at_3"] for trace in group
                    ),
                    "evidence_hit_at_3": sum(
                        trace["evidence_hit_at_3"] for trace in group
                    )
                    / len(group),
                }
            )
    return summaries
