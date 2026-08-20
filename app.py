from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from rag.chunking import make_chunks
from rag.evaluation import find_question, metrics
from rag.generation import build_context, generate_answer
from rag.pdf_ingest import extract_pdf_pages, validate_corpus
from rag.retrieval import SemanticRetriever

load_dotenv()
PDF_DIR = Path("data/pdfs")


def render_tables(tables: list[dict], *, key_prefix: str) -> None:
    """Render extracted PDF tables as rows and columns instead of flat text."""
    for table_number, table in enumerate(tables, start=1):
        rows = table.get("rows", [])
        if not rows:
            continue
        page = table.get("page")
        st.caption(f"Bảng {table_number}" + (f" · trang {page}" if page else ""))
        st.dataframe(
            pd.DataFrame(rows, columns=table.get("headers") or None),
            hide_index=True,
            width="stretch",
            height="content",
            key=f"{key_prefix}_{table_number}",
        )
st.set_page_config(page_title="PDF RAG có trích dẫn", layout="wide")
st.title("RAG đa ngôn ngữ trên PDF — Semantic Search + FAISS")
st.caption("Mỗi kết quả được truy vết về tên PDF, mục/chương và trang gốc.")

with st.sidebar:
    strategy = st.selectbox("Chunking", ["recursive", "fixed"], format_func=lambda x: "Recursive (khuyến nghị)" if x == "recursive" else "Fixed-size")
    size = st.slider("Kích thước chunk (ký tự)", 250, 1400, 700, 50)
    overlap = st.slider("Overlap (ký tự)", 0, 350, 120, 10)
    top_k = st.slider("Top-K", 1, 8, 3)
    build = st.button("Nạp / lập chỉ mục", type="primary")

if build:
    errors = validate_corpus(PDF_DIR)
    if errors:
        st.error("\n".join(errors))
    else:
        sections = [page for pdf in sorted(PDF_DIR.glob("*.pdf")) for page in extract_pdf_pages(pdf)]
        chunks = make_chunks(sections, strategy, size, overlap)
        with st.spinner("Đang tạo embedding và FAISS index..."):
            st.session_state.retriever = SemanticRetriever(chunks, sections=sections)
            st.session_state.chunks = chunks
        st.success(f"Đã lập chỉ mục {len(chunks)} chunks từ {len(sections)} sections.")

if "chunks" in st.session_state:
    chunks = st.session_state.chunks
    lengths = [len(x["text"]) for x in chunks]
    a, b, c = st.columns(3)
    a.metric("Số chunks", len(chunks)); b.metric("Độ dài TB", f"{sum(lengths)/len(lengths):.0f} ký tự"); c.metric("Chiến lược", chunks[0]["strategy"])
    source_choice = st.selectbox("Ưu tiên học phần", ["Tự nhận diện từ câu hỏi"] + st.session_state.retriever.sources,
        help="Nếu câu hỏi có tên học phần, hệ thống tự ưu tiên PDF tương ứng. Chọn thủ công khi tên học phần trong câu hỏi viết tắt hoặc khác tên file.")
    question = st.text_area("Câu hỏi", placeholder="Ví dụ: So sánh hai khái niệm ở chương nào?", height=90)
    if st.button("Tìm kiếm và trả lời", disabled=not question.strip()):
        detected_source = st.session_state.retriever.detect_source(question)
        preferred_source = source_choice if source_choice != "Tự nhận diện từ câu hỏi" else detected_source
        hits, elapsed = st.session_state.retriever.search(question, k=top_k, preferred_source=preferred_source)
        if preferred_source: st.success(f"Đang ưu tiên kết quả thuộc học phần: **{preferred_source}**")
        else: st.caption("Không nhận diện được học phần cụ thể; đang tìm kiếm trên toàn bộ tài liệu.")
        st.info(f"Thời gian truy vấn: {elapsed:.1f} ms · Context: ~{len(build_context(hits, question)) // 4} token")
        with st.expander("Top-K đoạn liên quan (context trước LLM)", expanded=True):
            shown_table_sections: set[str] = set()
            for hit in hits:
                ch = hit.chunk; pages = ", ".join(map(str, ch.get("pages", [ch["page"]])))
                st.markdown(f"**#{hit.rank} — {ch['source']} · tr. {pages} · {ch['section']}** (độ tương đồng: {hit.score:.4f})")
                st.text(ch["text"], width="stretch")
                section_id = ch.get("section_id")
                if section_id and section_id not in shown_table_sections:
                    full_section = st.session_state.retriever.get_full_section(section_id)
                    if full_section and full_section.get("tables"):
                        st.caption("Bảng thuộc mục này")
                        render_tables(full_section["tables"], key_prefix=f"topk_{section_id}")
                    shown_table_sections.add(section_id)
                st.code({key: ch[key] for key in ("chunk_id", "section_id", "chunk_index", "text", "source", "page_start", "page_end", "section", "strategy")}, language="python")
        st.subheader("Nội dung đầy đủ của mục")
        for section_hit in st.session_state.retriever.unique_section_hits(hits):
            full_section = st.session_state.retriever.get_full_section(section_hit.chunk["section_id"])
            if full_section:
                with st.expander(f"{full_section['section']} · {full_section['source']} · trang {full_section['page_start']}–{full_section['page_end']}", expanded=section_hit.rank == 1):
                    st.text(full_section["full_text"], width="stretch")
                    render_tables(full_section.get("tables", []), key_prefix=f"section_{full_section['section_id']}")
        st.subheader("Câu trả lời có căn cứ")
        st.text(generate_answer(question, hits), width="stretch")
        evaluation_path = next(
            (path for path in (Path("data/evaluation_questions.json"), Path("data/evaluation_questions.template.json")) if path.exists()),
            None,
        )
        st.subheader("Đánh giá Retrieval (đơn vị: section)")
        st.caption("Độ tương đồng cho biết mức phù hợp ngữ nghĩa giữa câu hỏi và chunk; càng cao càng liên quan. Precision@3 là tỷ lệ section đúng trong 3 kết quả đầu. Recall@3 là tỷ lệ toàn bộ section đúng đã được tìm thấy trong 3 kết quả đầu.")
        metric_left, metric_right = st.columns(2)
        if evaluation_path:
            evaluation_items = json.loads(evaluation_path.read_text(encoding="utf-8"))
            matched = find_question(question, evaluation_items)
            if matched:
                result = metrics([hit.chunk for hit in hits], matched.get("ground_truth", []), k=3)
                metric_left.metric("Precision@3", f"{result['precision@3']:.2f}", border=True)
                metric_right.metric("Recall@3", f"{result['recall@3']:.2f}", border=True)
                st.caption(f"Relevant retrieved: {result['relevant_retrieved']} · Relevant ground truth: {result['relevant_ground_truth']}")
            else:
                metric_left.metric("Precision@3", "N/A", border=True)
                metric_right.metric("Recall@3", "N/A", border=True)
                st.caption("Không có ground truth cho câu hỏi này, nên không thể tính metric hợp lệ.")
        else:
            metric_left.metric("Precision@3", "N/A", border=True)
            metric_right.metric("Recall@3", "N/A", border=True)
            st.caption("Chưa có file evaluation_questions.json để đối chiếu ground truth.")
else:
    st.warning("Đặt tối thiểu 3 PDF, mỗi file từ 5 trang, vào data/pdfs/, rồi nhấn “Nạp / lập chỉ mục”.")
