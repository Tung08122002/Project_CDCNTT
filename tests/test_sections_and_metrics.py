import unittest
from types import SimpleNamespace
from unittest.mock import patch

from rag.chunking import merge_overlapping_chunks, recursive_chunks
from rag.evaluation import find_question, metrics
from rag.generation import select_context_hits
from rag.pdf_ingest import (
    _is_numbered_heading,
    _is_styled_heading,
    _line_text_from_spans,
    structure_text,
    table_to_display,
    table_to_text,
)
from rag.retrieval import SemanticRetriever


class SectionAndMetricTests(unittest.TestCase):
    def setUp(self):
        self.section = {
            "source": "AI.pdf", "page": 5, "pages": [5, 6], "page_start": 5, "page_end": 6,
            "section": "3. Mục tiêu", "section_id": "ai__003",
            "text": "Một hai ba bốn năm. Sáu bảy tám chín mười.",
        }

    def test_chunk_metadata_keeps_section_and_page_range(self):
        chunks = recursive_chunks([self.section], size=20, overlap=4)
        self.assertTrue(all(chunk["section_id"] == "ai__003" for chunk in chunks))
        self.assertTrue(all((chunk["page_start"], chunk["page_end"]) == (5, 6) for chunk in chunks))
        self.assertEqual([chunk["chunk_index"] for chunk in chunks], list(range(len(chunks))))

    def test_embedding_model_uses_local_cache_first(self):
        cached_model = object()
        with patch("rag.retrieval.SentenceTransformer", return_value=cached_model) as loader:
            model = SemanticRetriever._load_embedding_model("model-id")
        self.assertIs(model, cached_model)
        loader.assert_called_once_with("model-id", local_files_only=True)

    def test_embedding_model_falls_back_to_download_without_cache(self):
        downloaded_model = object()
        with patch(
            "rag.retrieval.SentenceTransformer", side_effect=[OSError("not cached"), downloaded_model]
        ) as loader:
            model = SemanticRetriever._load_embedding_model("model-id")
        self.assertIs(model, downloaded_model)
        self.assertEqual(
            loader.call_args_list,
            [
                unittest.mock.call("model-id", local_files_only=True),
                unittest.mock.call("model-id"),
            ],
        )

    def test_overlap_merge_does_not_duplicate_text(self):
        merged = merge_overlapping_chunks([{"text": "A B C D E"}, {"text": "D E F G"}])
        self.assertEqual(merged, "A B C D E F G")

    def test_section_precision_and_recall_at_3(self):
        retrieved = [
            {"source": "AI.pdf", "section": "A", "page": 1},
            {"source": "AI.pdf", "section": "B", "page": 2},
            {"source": "AI.pdf", "section": "C", "page": 3},
        ]
        relevant = [{"source": "AI.pdf", "section": "A", "page": 1}, {"source": "AI.pdf", "section": "C", "page": 3}]
        result = metrics(retrieved, relevant)
        self.assertAlmostEqual(result["precision@3"], 2 / 3)
        self.assertEqual(result["recall@3"], 1.0)

    def test_visual_line_wraps_become_sentence_lines(self):
        text = "Sinh viên phát triển kỹ năng\nnghe và nói. Người học thực hành\ntrong lớp.\n- Hoạt động tự học"
        self.assertEqual(
            structure_text(text),
            "Sinh viên phát triển kỹ năng nghe và nói.\nNgười học thực hành trong lớp.\n- Hoạt động tự học",
        )

    def test_table_rows_include_header_value_pairs(self):
        text, headers = table_to_text(
            [["Trọng số", "Bài đánh giá"], ["0.6", "Thi cuối kỳ"]]
        )
        self.assertEqual(headers, ["Trọng số", "Bài đánh giá"])
        self.assertIn("Trọng số: 0.6", text)
        self.assertIn("Bài đánh giá: Thi cuối kỳ", text)

    def test_table_display_keeps_columns_and_rows(self):
        rows, headers = table_to_display(
            [["Trọng số", "Bài đánh giá"], ["0.6", "Thi cuối kỳ"]]
        )
        self.assertEqual(headers, ["Trọng số", "Bài đánh giá"])
        self.assertEqual(rows, [{"Trọng số": "0.6", "Bài đánh giá": "Thi cuối kỳ"}])

    def test_unbold_numbered_and_note_headings_are_recognized(self):
        self.assertTrue(_is_numbered_heading("4.1 Nội dung chi tiết", False))
        self.assertTrue(_is_numbered_heading("Ghi chú", False))

    def test_italic_and_bold_unnumbered_headings_are_recognized(self):
        self.assertTrue(_is_styled_heading("a. Khái niệm kỹ năng số", False, True))
        self.assertTrue(_is_styled_heading("Đặc điểm:", False, True))
        self.assertTrue(_is_styled_heading("CHƯƠNG 1: TỔNG QUAN", True, False))
        self.assertFalse(_is_styled_heading("Nguồn: Tài liệu tham khảo", False, True))

    def test_raw_character_gap_restores_missing_word_space(self):
        span = {
            "size": 14,
            "chars": [
                {"c": "s", "bbox": (0, 0, 7, 10)},
                {"c": "ố", "bbox": (7, 0, 14, 10)},
                {"c": "t", "bbox": (17.5, 0, 22, 10)},
                {"c": "r", "bbox": (22, 0, 28, 10)},
                {"c": "o", "bbox": (28, 0, 35, 10)},
                {"c": "n", "bbox": (35, 0, 42, 10)},
                {"c": "g", "bbox": (42, 0, 49, 10)},
            ],
        }
        self.assertEqual(_line_text_from_spans([span]), "số trong")

    def test_vietnamese_course_name_keeps_d_character_for_source_detection(self):
        self.assertEqual(
            SemanticRetriever._normalize("Đề cương Tiếng Anh thương mại 2"),
            "de cuong tieng anh thuong mai 2",
        )

    def test_source_mention_is_removed_from_semantic_question(self):
        result = SemanticRetriever._remove_source_mention(
            "Mục tiêu của học phần Tiếng Anh thương mại 2 là gì?",
            "Đề cương tiếng anh thương mại 2.pdf",
        )
        self.assertEqual(result, "Mục tiêu của học phần là gì")

    def test_section_metadata_routes_assessment_intent(self):
        retriever = object.__new__(SemanticRetriever)
        boost = retriever._section_intent_boost(
            {"section": "10. Đánh giá học phần"},
            SemanticRetriever._normalize("Ba thành phần đánh giá có trọng số bao nhiêu?"),
        )
        self.assertGreater(boost, 0)

    def test_single_topic_context_keeps_only_direct_heading(self):
        hits = [
            SimpleNamespace(rank=1, chunk={"section": "Một số phát minh của cách mạng Công nghiệp lần thứ 4:"}),
            SimpleNamespace(rank=2, chunk={"section": "Tiến trình Cách mạng Công nghiệp lần thứ 3:"}),
        ]
        selected = select_context_hits("Một số phát minh của cách mạng Công nghiệp lần thứ 4 là gì?", hits)
        self.assertEqual(selected, [hits[0]])

    def test_section_metadata_routes_teaching_staff_intent(self):
        retriever = object.__new__(SemanticRetriever)
        boost = retriever._section_intent_boost(
            {"section": "9. Cán bộ giảng dạy học phần"},
            SemanticRetriever._normalize("Thông tin cán bộ giáo dục (CBGD) là gì?"),
        )
        self.assertGreater(boost, 0)

    def test_context_keeps_teaching_staff_heading_for_cbgd_query(self):
        hits = [
            SimpleNamespace(rank=1, chunk={"section": "9. Cán bộ giảng dạy học phần"}),
            SimpleNamespace(rank=2, chunk={"section": "2. Mục tiêu học phần"}),
        ]
        selected = select_context_hits("Thông tin CBGD của học phần là gì?", hits)
        self.assertEqual(selected, [hits[0]])

    def test_staff_profile_embeddings_keep_full_name_and_title(self):
        profiles = SemanticRetriever._staff_profile_texts(
            {
                "section": "9.1. CBGD cơ hữu:",
                "text": "9.1.\nCBGD cơ hữu:\nTS.\nNguyễn Thị Lan Phương\nThS. Lê Thị Phương Mai",
            }
        )
        self.assertEqual(profiles, ["TS. Nguyễn Thị Lan Phương", "ThS. Lê Thị Phương Mai"])

    def test_context_keeps_staff_section_for_name_only_query(self):
        hits = [
            SimpleNamespace(
                rank=1,
                chunk={
                    "section": "9.1. CBGD cơ hữu",
                    "text": "TS. Nguyễn Thị Lan Phương\nThS. Lê Thị Phương Mai",
                },
            ),
            SimpleNamespace(rank=2, chunk={"section": "2. Mục tiêu", "text": "Mục tiêu học phần"}),
        ]
        selected = select_context_hits("Thông tin TS. Nguyễn Thị Lan Phương", hits)
        self.assertEqual(selected, [hits[0]])

    def test_evaluation_question_lookup_ignores_accents_and_punctuation(self):
        questions = [{"question": "Mục tiêu của học phần là gì?", "ground_truth": []}]
        self.assertIs(find_question("muc tieu cua hoc phan la gi", questions), questions[0])


if __name__ == "__main__":
    unittest.main()
