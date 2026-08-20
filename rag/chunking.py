"""Chunkers that preserve section/page metadata."""
from __future__ import annotations

import re
from typing import Any


def _record(text: str, page: dict[str, Any], chunk_id: int, chunk_index: int, strategy: str) -> dict[str, Any]:
    return {"chunk_id": chunk_id, "text": text.strip(), "source": page["source"], "page": page["page"],
            "section": page["section"], "section_id": page["section_id"], "chunk_index": chunk_index,
            "page_start": page.get("page_start", page["page"]), "page_end": page.get("page_end", page["page"]),
            "strategy": strategy, "pages": page.get("pages", [page["page"]])}


def fixed_chunks(pages: list[dict[str, Any]], size: int = 700, overlap: int = 120) -> list[dict[str, Any]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size phải dương và overlap phải nhỏ hơn size")
    chunks: list[dict[str, Any]] = []
    for page in pages:
        start = 0
        section_chunk_index = 0
        while start < len(page["text"]):
            value = page["text"][start:start + size].strip()
            if value:
                chunks.append(_record(value, page, len(chunks), section_chunk_index, "fixed"))
                section_chunk_index += 1
            if start + size >= len(page["text"]):
                break
            start += size - overlap
    return chunks


def recursive_chunks(pages: list[dict[str, Any]], size: int = 700, overlap: int = 120) -> list[dict[str, Any]]:
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError("size phải dương và overlap phải nhỏ hơn size")
    chunks: list[dict[str, Any]] = []
    for page in pages:
        units = [x.strip() for x in re.split(r"\n\n+|\n|(?<=[.!?])\s+", page["text"]) if x.strip()]
        current = ""
        section_chunk_index = 0
        for unit in units:
            # Preserve the PDF's logical line breaks for both debug display and RAG context.
            candidate = f"{current}\n{unit}".strip() if current else unit
            if current and len(candidate) > size:
                chunks.append(_record(current, page, len(chunks), section_chunk_index, "recursive"))
                section_chunk_index += 1
                current = (current[-overlap:] + "\n" + unit).strip() if overlap else unit
            else:
                current = candidate
        if current:
            chunks.append(_record(current, page, len(chunks), section_chunk_index, "recursive"))
    return chunks


def make_chunks(pages: list[dict[str, Any]], strategy: str, size: int, overlap: int) -> list[dict[str, Any]]:
    return fixed_chunks(pages, size, overlap) if strategy == "fixed" else recursive_chunks(pages, size, overlap)


def merge_overlapping_chunks(chunks: list[dict[str, Any]]) -> str:
    """Merge fallback chunks without duplicating their shared overlap."""
    merged = ""
    for chunk in chunks:
        text = chunk["text"].strip()
        if not merged:
            merged = text
            continue
        left, right = merged.split(), text.split()
        overlap = 0
        for length in range(min(len(left), len(right)), 0, -1):
            if [item.lower() for item in left[-length:]] == [item.lower() for item in right[:length]]:
                overlap = length
                break
        merged = " ".join(left + right[overlap:])
    return merged
