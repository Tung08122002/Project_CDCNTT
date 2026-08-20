"""FAISS-backed semantic search for multilingual PDF chunks."""
from __future__ import annotations

import re
import time
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from rag.chunking import merge_overlapping_chunks


@dataclass
class SearchHit:
    chunk: dict[str, Any]
    score: float
    rank: int


class SemanticRetriever:
    _GENERIC_SECTION_MARKERS = (
        "khong xac dinh",
        "cong hoa xa hoi",
        "bo mon",
        "de cuong hoc phan",
        "truong khoa",
        "truong bo mon",
        "hieu truong",
    )
    _SECTION_INTENTS = (
        (("dieu kien",), ("hoc truoc", "hoc phan hoc truoc", "tien quyet", "prerequisite")),
        (("muc tieu",), ("muc tieu", "objective")),
        (("chuan dau ra",), ("clo", "chuan dau ra", "learning outcome")),
        (("mo ta",), ("ke thua", "tiep noi", "phat trien", "tom tat noi dung", "continuation")),
        (("danh gia",), ("danh gia", "trong so", "diem chuyen can", "thi ket thuc", "kiem tra", "bai tap lon")),
        (
            ("can bo giang day", "cbgd"),
            (
                "can bo giao duc",
                "can bo giang day",
                "cbgd",
                "giang vien",
                "giao vien",
                "thong tin giang day",
            ),
        ),
        (("danh muc tai lieu", "tai lieu tham khao"), ("giao trinh", "tai lieu tham khao", "sach", "textbook")),
    )
    _STAFF_SECTION_MARKERS = ("cbgd", "can bo giang day")
    _STAFF_TITLE_LINE = re.compile(r"^(?:ts|ths|pgs|gs|dr|prof)\.?$", flags=re.IGNORECASE)

    @classmethod
    def _staff_profile_texts(cls, chunk: dict[str, Any]) -> list[str]:
        """Build focused semantic views for people listed in CBGD sections.

        The displayed chunk is not changed. Extra FAISS vectors simply make a
        full name in a long roster retrieve its original cited section.
        """
        normalized_section = cls._normalize(chunk.get("section", ""))
        if not any(marker in normalized_section for marker in cls._STAFF_SECTION_MARKERS):
            return []

        profiles: list[str] = []
        pending_title = ""
        for raw_line in str(chunk.get("text", "")).splitlines():
            line = raw_line.strip()
            normalized_line = cls._normalize(line)
            if (
                not line
                or normalized_line == normalized_section
                or (normalized_line and normalized_line in normalized_section)
                or re.fullmatch(r"\d+(?:\.\d+)*\.?", line)
            ):
                continue
            if cls._STAFF_TITLE_LINE.fullmatch(line):
                pending_title = line
                continue
            if normalized_line in {"khong", "none", "n a"}:
                continue
            if len(normalized_line.split()) < 2:
                continue
            profiles.append(f"{pending_title} {line}".strip() if pending_title else line)
            pending_title = ""
        return profiles

    def __init__(self, chunks: list[dict[str, Any]], sections: list[dict[str, Any]] | None = None,
                 model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"):
        if not chunks:
            raise ValueError("Không có chunk để lập chỉ mục")
        self.chunks, self.model_name = chunks, model_name
        self.sections_by_id = {
            section["section_id"]: section for section in (sections or []) if "section_id" in section
        }
        self.sources = sorted({chunk["source"] for chunk in chunks})
        self._source_aliases = {source: self._make_source_aliases(source) for source in self.sources}
        self.model = SentenceTransformer(model_name)
        embedding_texts: list[str] = []
        embedding_chunk_ids: list[int] = []
        for chunk_index, chunk in enumerate(chunks):
            embedding_texts.append(f"{chunk.get('section', '')}\n{chunk['text']}")
            embedding_chunk_ids.append(chunk_index)
            for profile in self._staff_profile_texts(chunk):
                embedding_texts.append(
                    f"Teaching staff\nSection: {chunk.get('section', '')}\n"
                    f"Staff member: {profile}\nDocument: {chunk['source']}"
                )
                embedding_chunk_ids.append(chunk_index)
        self._embedding_chunk_ids = np.asarray(embedding_chunk_ids, dtype=np.int32)
        vectors = self.model.encode(
            embedding_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=True
        ).astype("float32")
        self.content_vectors = vectors
        section_texts = [f"Mục: {chunk.get('section', '')}\nTài liệu: {chunk['source']}" for chunk in chunks]
        section_texts = [
            f"Section: {chunks[chunk_index].get('section', '')}\n"
            f"Document: {chunks[chunk_index]['source']}"
            for chunk_index in self._embedding_chunk_ids
        ]
        self.section_vectors = self.model.encode(
            section_texts, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)

    def get_full_section(self, section_id: str) -> dict[str, Any] | None:
        """Return original cleaned section text; never ask an LLM to reconstruct it."""
        section = self.sections_by_id.get(section_id)
        if section:
            return section
        section_chunks = sorted(
            (chunk for chunk in self.chunks if chunk.get("section_id") == section_id),
            key=lambda chunk: chunk.get("chunk_index", 0),
        )
        if not section_chunks:
            return None
        return {
            "section_id": section_id, "section": section_chunks[0]["section"],
            "source": section_chunks[0]["source"], "page": section_chunks[0]["page"],
            "pages": section_chunks[0].get("pages", [section_chunks[0]["page"]]),
            "page_start": section_chunks[0].get("page_start", section_chunks[0]["page"]),
            "page_end": section_chunks[0].get("page_end", section_chunks[0]["page"]),
            "full_text": merge_overlapping_chunks(section_chunks),
        }

    def unique_section_hits(self, hits: list[SearchHit]) -> list[SearchHit]:
        """Keep only the highest scoring retrieval hit for each display section."""
        unique: list[SearchHit] = []
        seen: set[str] = set()
        for hit in hits:
            section_id = hit.chunk.get("section_id", str(hit.chunk["chunk_id"]))
            if section_id not in seen:
                unique.append(hit)
                seen.add(section_id)
        return unique

    @staticmethod
    def _normalize(value: str) -> str:
        # ``đ`` is not decomposed by NFD, so map it before the ASCII cleanup.
        # Without this, aliases such as "Đề cương ..." cannot match a question.
        value = value.lower().replace("đ", "d")
        value = unicodedata.normalize("NFD", value)
        value = "".join(char for char in value if unicodedata.category(char) != "Mn")
        return re.sub(r"[^a-z0-9]+", " ", value).strip()

    def _make_source_aliases(self, source: str) -> set[str]:
        stem = Path(source).stem
        normalized = self._normalize(stem)
        aliases = {normalized}
        stripped = re.sub(r"^de cuong(?: hoc phan)?\s+", "", normalized)
        if len(stripped.split()) >= 2:
            aliases.add(stripped)
        return aliases

    def detect_source(self, question: str) -> str | None:
        normalized_question = f" {self._normalize(question)} "
        matches = [(len(alias.split()), source) for source, aliases in self._source_aliases.items() for alias in aliases if f" {alias} " in normalized_question]
        return max(matches)[1] if matches else None

    @staticmethod
    def _query_variants(question: str) -> list[str]:
        """Split a multi-part question into semantic sub-queries for evidence."""
        variants = [question.strip()]
        stop_words = {
            "doi", "voi", "hay", "trinh", "bay", "trong", "cua", "la", "gi", "neu", "them",
            "nhung", "nao", "va", "duoc", "cho", "cac", "mot", "hoc", "phan", "noi", "ve",
        }
        # Keep conjunctions inside a clause: "học phần học trước ... và mã HP"
        # is one piece of evidence, not two unrelated short semantic queries.
        for part in re.split(r"[;?]|\s*,\s*", question):
            part = part.strip(" .,:;?")
            meaningful_words = [word for word in SemanticRetriever._normalize(part).split() if word not in stop_words]
            if len(part) >= 12 and len(meaningful_words) >= 2 and part not in variants:
                variants.append(part)
        return variants

    def _is_generic_section(self, chunk: dict[str, Any]) -> bool:
        normalized_section = self._normalize(chunk.get("section", ""))
        return any(marker in normalized_section for marker in self._GENERIC_SECTION_MARKERS)

    def _section_intent_boost(self, chunk: dict[str, Any], normalized_question: str) -> float:
        """Align syllabus-style question intents with section metadata.

        This is metadata routing only: it never searches document text by
        keywords and the final relevance score remains embedding based.
        """
        normalized_section = self._normalize(chunk.get("section", ""))
        boost = 0.0
        for section_markers, question_markers in self._SECTION_INTENTS:
            if any(marker in normalized_section for marker in section_markers) and any(
                marker in normalized_question for marker in question_markers
            ):
                boost += 0.16
        return min(boost, 0.32)

    @staticmethod
    def _remove_source_mention(question: str, source: str | None) -> str:
        """Remove an explicit filename/course name before semantic matching.

        It is metadata used to scope the search, not the information the user is
        asking for. Leaving it in the embedding query makes an introductory
        "Tên học phần" section dominate every course-specific question.
        """
        if not source:
            return question
        stem = Path(source).stem
        aliases = [stem, re.sub(r"^đề cương(?: học phần)?\s+", "", stem, flags=re.IGNORECASE)]
        cleaned = question
        for alias in aliases:
            if alias:
                cleaned = re.sub(re.escape(alias), " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" ,:;?-")
        return cleaned if len(cleaned) >= 8 else question

    def search(self, question: str, k: int = 3, preferred_source: str | None = None) -> tuple[list[SearchHit], float]:
        start = time.perf_counter()
        preferred_source = preferred_source or self.detect_source(question)
        semantic_question = self._remove_source_mention(question, preferred_source)
        queries = self._query_variants(semantic_question)
        normalized_question = self._normalize(semantic_question)
        query_vectors = self.model.encode(queries, normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        # FAISS remains the semantic retrieval index. Additional semantic scores
        # for each question component and the section title focus evidence on the
        # requested item rather than a similarly worded neighbouring section.
        _, faiss_ids = self.index.search(query_vectors[:1], len(self._embedding_chunk_ids))
        entry_content_scores = (self.content_vectors @ query_vectors.T).max(axis=1)
        entry_section_scores = (self.section_vectors @ query_vectors.T).max(axis=1)
        content_scores = np.full(len(self.chunks), -np.inf, dtype=np.float32)
        section_scores = np.full(len(self.chunks), -np.inf, dtype=np.float32)
        np.maximum.at(content_scores, self._embedding_chunk_ids, entry_content_scores)
        np.maximum.at(section_scores, self._embedding_chunk_ids, entry_section_scores)
        candidate_ids = list(dict.fromkeys(
            int(self._embedding_chunk_ids[idx]) for idx in faiss_ids[0] if idx >= 0
        ))
        if preferred_source in self.sources:
            # A detected file/course is a scope constraint, not merely a boost.
            candidate_ids = [idx for idx in candidate_ids if self.chunks[idx]["source"] == preferred_source]
        non_generic_ids = [idx for idx in candidate_ids if not self._is_generic_section(self.chunks[idx])]
        if non_generic_ids:
            candidate_ids = non_generic_ids
        ranked = sorted(
            (
                (
                    idx,
                    float(
                        0.75 * content_scores[idx]
                        + 0.25 * section_scores[idx]
                        + self._section_intent_boost(self.chunks[idx], normalized_question)
                    ),
                )
                for idx in candidate_ids
            ),
            key=lambda item: item[1],
            reverse=True,
        )
        # Multiple overlapping chunks from one section otherwise crowd out the
        # other sections needed for multi-part questions and evaluation@3.
        chosen: list[tuple[int, float]] = []
        seen_sections: set[str] = set()
        for idx, score in ranked:
            section_id = self.chunks[idx].get("section_id", str(self.chunks[idx]["chunk_id"]))
            if section_id in seen_sections:
                continue
            chosen.append((idx, score))
            seen_sections.add(section_id)
            if len(chosen) == k:
                break
        hits = [SearchHit(self.chunks[idx], score, rank) for rank, (idx, score) in enumerate(chosen, 1)]
        return hits, (time.perf_counter() - start) * 1000
