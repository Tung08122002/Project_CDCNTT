# Hệ thống RAG song ngữ trên PDF có citation

Project triển khai Semantic Search với SentenceTransformer + FAISS, RAG trên PDF theo từng trang, chunking và metadata. Project này **chỉ dùng Semantic Search**.

## 1. Chuẩn bị dữ liệu và chạy

1. Tạo môi trường và cài thư viện:

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   python -m pip install -r requirements.txt
   ```

   Nếu terminal báo thiếu `fitz`, `faiss`, `streamlit` hoặc `sentence_transformers`, thường là do chưa kích hoạt `.venv`; chạy lại lệnh `Activate.ps1` trước khi chạy app.
2. Đặt tối thiểu 3 PDF vào `data/pdfs/`; từng file phải có tối thiểu **5 trang**. Project kiểm tra điều kiện này trước khi lập chỉ mục.
3. Sao chép `data/evaluation_questions.template.json` thành `data/evaluation_questions.json`; thay `source`, `page`, `section` bằng ground truth thật.
4. Sao chép `.env.example` thành `.env`, sau đó điền:

   ```env
   GEMINI_API_KEY=khóa_API_của_bạn
   GEMINI_MODEL=gemini-2.5-flash
   ```

   `.env` đã được đưa vào `.gitignore`; không dán khóa vào source hoặc commit nó lên Git.
5. Chạy `python -m streamlit run app.py`.

Ứng dụng đọc từng trang, làm sạch khoảng trắng/dòng trống và ghép từ bị ngắt bởi `-\n`. Với đề cương học phần, dòng in đậm mở đầu bằng số được xem là đầu mục. Một đầu mục có thể tiếp tục sang trang sau: hệ thống ghép nội dung liên tục cho đến đầu mục kế tiếp, vẫn lưu `page` là trang bắt đầu và `pages` là các trang liên quan.

PDF tables are extracted as searchable header/value rows, for example `Trọng số: 0.6 | Bài đánh giá: Thi cuối kỳ`. Table headers are reused when a table continues on a later page. Numbered headings that lose bold formatting and labels such as `Ghi chú`, `Lưu ý`, or `Note` are also treated as section boundaries, so their following content remains retrievable.

### Using evaluation_questions.template.json

Copy `data/evaluation_questions.template.json` to `data/evaluation_questions.json`. Each record contains a `question`, a `ground_truth` list, and an optional `answer_rubric`.

```json
{
  "question": "Mục tiêu của học phần là gì?",
  "ground_truth": [
    {"source": "Đề cương tiếng anh thương mại 2.pdf", "page": 1, "section": "6. Mục tiêu của học phần"}
  ]
}
```

Keep or adapt the three multi-part questions in the template to match the team corpus. Precision@3 and Recall@3 are calculated only when the UI question exactly matches a record in this file; otherwise they display `N/A`.

### Fixing torchvision import errors

After activating `.venv`, run `python -m pip install -r requirements.txt`. `torchvision` is included in `requirements.txt`. If needed, run `python -m pip install torchvision` directly.

## 2. Pipeline

```text
PDF (từng trang) → clean text + numbered bold section → chunks + metadata
                                                    → multilingual SentenceTransformer → FAISS (cosine/IP)
Question → SentenceTransformer → FAISS Top-K → context → LLM chỉ dùng context → answer [PDF, trang, mục]
```

Mỗi chunk có schema:

```python
{"chunk_id": 17, "text": "...", "source": "document.pdf", "page": 5,
 "section": "...", "strategy": "recursive"}
```

## 3. Chức năng

- Hỏi định nghĩa/giải thích, ví dụ, so sánh hai khái niệm hoặc nội dung theo chương/trang.
- Semantic Search bằng FAISS; xem Top-K trước khi LLM trả lời.
- Khi câu hỏi chứa tên học phần, hệ thống nhận diện tên từ PDF và đưa chunks đúng học phần lên trước. Với tên viết tắt hoặc khác filename, chọn học phần ở ô **Ưu tiên học phần**.
- Hiển thị chunk gồm `chunk_id`, `text`, `source`, `page`, `section`, `strategy`.
- Sau Top-K chunks, giao diện hiển thị **Nội dung đầy đủ của mục** theo `section_id`; phần này được lấy trực tiếp từ text PDF đã trích xuất, không do LLM ghép.
- Hiển thị Precision@3 và Recall@3 khi câu hỏi có ground truth trong `data/evaluation_questions.json`; câu hỏi tự do hiển thị N/A.
- Chọn fixed/recursive chunking; xem số chunks, độ dài trung bình, thời gian query và ước lượng token context.

## 4. Evaluation

Chạy trên UI hoặc:

```powershell
python scripts/evaluate.py --strategy recursive
python scripts/evaluate.py --strategy fixed --size 700 --overlap 120
```

Báo cáo gồm `Precision@3`, `Recall@3`, `Hit@3`, `MRR`, `nDCG@3`, `query_ms`. Không commit API key; project đọc key từ `.env`.
