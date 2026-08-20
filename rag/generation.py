"""Grounded answer generation with mandatory, verifiable citations."""
from __future__ import annotations

import os
import re
import unicodedata
from typing import Any


_CONTEXT_STOP_WORDS = {
    "cua", "la", "gi", "mot", "so", "ve", "trong", "va", "nhung", "cac", "noi", "dung", "hay",
}


def _normalized_tokens(value: str) -> set[str]:
    value = unicodedata.normalize("NFD", value.lower().replace("đ", "d"))
    value = "".join(character for character in value if unicodedata.category(character) != "Mn")
    return {token for token in re.findall(r"[a-z0-9]+", value) if token not in _CONTEXT_STOP_WORDS}


def _is_multi_part_question(question: str) -> bool:
    return bool(re.search(r"[,;]|\s+và\s+", question, flags=re.IGNORECASE))


def select_context_hits(question: str, hits: list[Any]) -> list[Any]:
    """Use the direct section for a single-topic question, preserving multi-part RAG."""
    if not hits or _is_multi_part_question(question):
        return hits
    question_tokens = _normalized_tokens(question)
    question_numbers = set(re.findall(r"\d+", question))
    direct_hits: list[Any] = []
    for hit in hits:
        section = hit.chunk.get("section", "")
        section_tokens = _normalized_tokens(section)
        section_numbers = set(re.findall(r"\d+", section))
        # Syllabus files normally call this section "Cán bộ giảng dạy học
        # phần", but users also refer to it as CBGD, cán bộ giáo dục, or
        # giảng viên. Use those equivalent labels only to select the direct
        # section context for a focused question.
        teaching_staff_question = (
            "cbgd" in question_tokens
            or {"can", "bo", "giao", "duc"}.issubset(question_tokens)
            or {"can", "bo", "giang", "day"}.issubset(question_tokens)
            or {"giang", "vien"}.issubset(question_tokens)
            or {"giao", "vien"}.issubset(question_tokens)
        )
        if teaching_staff_question and (
            "cbgd" in section_tokens or {"can", "bo", "giang", "day"}.issubset(section_tokens)
        ):
            direct_hits.append(hit)
            continue
        # A name-only question (for example, "TS. Nguyen Thi Lan Phuong")
        # may not include the CBGD label. Once semantic retrieval places a
        # staff section in Top-K, keep that direct roster context when at least
        # two queried name tokens occur in it.
        staff_section = "cbgd" in section_tokens or {"can", "bo", "giang", "day"}.issubset(section_tokens)
        staff_name_tokens = question_tokens.intersection(_normalized_tokens(hit.chunk.get("text", "")))
        if staff_section and len(staff_name_tokens) >= 2:
            direct_hits.append(hit)
            continue
        # Do not confuse "lần thứ 4" with an otherwise similar "lần thứ 3" heading.
        if question_numbers and section_numbers and not question_numbers.intersection(section_numbers):
            continue
        shared = question_tokens.intersection(section_tokens)
        if len(shared) >= 2 and len(shared) / max(1, len(section_tokens)) >= 0.55:
            direct_hits.append(hit)
    return direct_hits[:1] if direct_hits else hits


def build_context(hits: list[Any], question: str | None = None) -> str:
    selected_hits = select_context_hits(question, hits) if question else hits
    return "\n\n".join(
        f"[C{hit.rank} | {hit.chunk['source']} | tr.{', '.join(map(str, hit.chunk.get('pages', [hit.chunk['page']])))} | {hit.chunk['section']}]\n{hit.chunk['text']}"
        for hit in selected_hits
    )


def generate_answer(question: str, hits: list[Any], api_key: str | None = None) -> str:
    api_key = api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return "Chưa có GEMINI_API_KEY nên hệ thống chỉ hiển thị Top-K nguồn. Hãy cấu hình khóa để sinh câu trả lời RAG."
    from google import genai
    prompt = f"""Bạn là trợ lý RAG song ngữ Việt-Anh. Chỉ dùng CONTEXT bên dưới.
Nếu context không đủ, nói rõ không đủ thông tin. Trả lời ngắn gọn nhưng đầy đủ,
và sau mỗi ý thực tế phải trích dẫn đúng định dạng [Tên PDF, tr. X, mục Y].
Không được nêu nguồn không có trong context.
Trình bày mỗi ý trên một dòng riêng; dùng đoạn ngắn hoặc danh sách gạch đầu dòng khi câu trả lời có nhiều ý.
Với câu hỏi chỉ có một chủ đề, trả lời trực tiếp nội dung của đề mục phù hợp, không tóm tắt bối cảnh hay các mục đứng trước nó.

CONTEXT:
{build_context(hits, question)}

CÂU HỎI: {question}"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash"), contents=prompt)
    return response.text
