"""Retrieval metrics and ground-truth matching by source/page/section."""
from __future__ import annotations

import math
import re
import unicodedata
from typing import Any


def key(item: dict[str, Any]) -> tuple[str, int, str]:
    return item["source"], int(item["page"]), item.get("section", "")


def section_key(item: dict[str, Any]) -> tuple[str, str]:
    return item["source"], item.get("section", "")


def question_key(question: str) -> str:
    """Normalize harmless spelling/formatting differences before eval lookup."""
    value = unicodedata.normalize("NFD", question.lower().replace("đ", "d"))
    value = "".join(character for character in value if unicodedata.category(character) != "Mn")
    return re.sub(r"[^a-z0-9]+", " ", value).strip()


def find_question(question: str, questions: list[dict[str, Any]]) -> dict[str, Any] | None:
    normalized_question = question_key(question)
    return next((item for item in questions if question_key(item.get("question", "")) == normalized_question), None)


def metrics(retrieved: list[dict[str, Any]], relevant: list[dict[str, Any]], k: int = 3) -> dict[str, float]:
    """Evaluate unique retrieved sections; source/page/section ground truth stays compatible."""
    relevant_keys = {section_key(item) for item in relevant}
    unique_retrieved: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for item in retrieved:
        item_key = section_key(item)
        if item_key not in seen:
            unique_retrieved.append(item)
            seen.add(item_key)
    flags = [section_key(item) in relevant_keys for item in unique_retrieved[:k]]
    hits = sum(flags); precision = hits / k; recall = hits / len(relevant_keys) if relevant_keys else 0.0
    first = next((rank for rank, hit in enumerate(flags, 1) if hit), None)
    dcg = sum((1 / math.log2(rank + 1)) for rank, hit in enumerate(flags, 1) if hit)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(k, len(relevant_keys)) + 1))
    return {"precision@3": precision, "recall@3": recall, "hit@3": float(hits > 0), "mrr": 1 / first if first else 0.0,
            "ndcg@3": dcg / ideal if ideal else 0.0, "relevant_retrieved": hits,
            "relevant_ground_truth": len(relevant_keys), "evaluation_unit": "section"}


def evaluate_questions(retriever: Any, questions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for item in questions:
        hits, latency = retriever.search(item["question"], k=3)
        row = {"id": item["id"], "question": item["question"], "query_ms": round(latency, 2)}
        row.update(metrics([hit.chunk for hit in hits], item["ground_truth"]))
        rows.append(row)
    return rows
