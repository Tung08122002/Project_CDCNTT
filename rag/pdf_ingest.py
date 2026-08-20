"""Page-aware PDF extraction, cleaning and section detection."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import fitz


def clean_text(text: str) -> str:
    """Repair common PDF extraction artifacts without merging pages."""
    text = text.replace("\u00ad", "")
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r" *\n *", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def structure_text(text: str) -> str:
    """Repair visual PDF line wrapping while preserving meaningful line structure.

    Normal wrapped lines are joined into sentences. New lines are introduced after
    sentence-ending punctuation and preserved for numbered/bulleted items.
    """
    output: list[str] = []
    paragraph: list[str] = []

    def flush_paragraph() -> None:
        if not paragraph:
            return
        joined = " ".join(paragraph).strip()
        joined = re.sub(r"(?<=[.!?])\s+(?=[A-ZÀ-ỴĐ(])", "\n", joined)
        output.extend(line.strip() for line in joined.splitlines() if line.strip())
        paragraph.clear()

    for raw_line in clean_text(text).splitlines():
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            continue
        is_item = bool(re.match(r"^(?:[-•*]|\d+(?:\.\d+)*[.)]?)\s+", line))
        if is_item:
            flush_paragraph()
            output.append(line)
        else:
            paragraph.append(line)
    flush_paragraph()
    return "\n".join(output)


_HEADING = re.compile(r"^\s*(?:\d+(?:\.\d+)+[.)]?|\d+[.)])\s+\S.+$", re.IGNORECASE)
_SIMPLE_NUMBERED_HEADING = re.compile(r"^\s*\d{1,2}\s+\S.+$", re.IGNORECASE)
_AUXILIARY_HEADING = re.compile(r"^\s*(?:ghi\s*chú|lưu\s*ý|chú\s*ý|note|notes|remark|remarks|phụ\s*lục|appendix)\b.*", re.IGNORECASE)
_LETTERED_HEADING = re.compile(r"^\s*(?:[a-zđ]|[*•])\.\s+\S.+$", re.IGNORECASE)
_CAPTION_PREFIX = re.compile(r"^\s*(?:bảng|hình|nguồn|source)\b", re.IGNORECASE)
_CHAPTER_HEADING = re.compile(r"^\s*chương\s+\d+\s*[:.]", re.IGNORECASE)


def _slug(value: str) -> str:
    value = value.lower().replace("đ", "d")
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_") or "document"


def _inside_table(bbox: tuple[float, float, float, float], table_boxes: list[tuple[float, float, float, float]]) -> bool:
    center_x = (bbox[0] + bbox[2]) / 2
    center_y = (bbox[1] + bbox[3]) / 2
    return any(left <= center_x <= right and top <= center_y <= bottom for left, top, right, bottom in table_boxes)


def _line_text_from_spans(spans: list[dict[str, Any]]) -> str:
    """Reconstruct spaces that some PDFs encode only as a glyph-position gap.

    ``baigiangtext_moi.pdf`` contains adjacent characters such as ``sốtrong``
    whose bounding boxes have a normal word-sized gap but no literal space
    character. Reading the raw character coordinates preserves these spaces.
    """
    output: list[str] = []
    previous_right: float | None = None
    previous_size = 0.0
    previous_was_space = True

    for span in spans:
        font_size = float(span.get("size", 0.0))
        for char in span.get("chars", []):
            value = char.get("c", "")
            bbox = char.get("bbox")
            if not value or not bbox:
                continue
            left, _, right, _ = bbox
            if value.isspace():
                if output and not previous_was_space:
                    output.append(" ")
                previous_was_space = True
            else:
                # A word space is approximately a quarter of the font size in
                # this document. Normal character kerning is near zero.
                gap = left - previous_right if previous_right is not None else 0.0
                space_threshold = max(1.0, min(previous_size or font_size, font_size) * 0.18)
                if output and not previous_was_space and gap >= space_threshold:
                    output.append(" ")
                output.append(value)
                previous_was_space = False
            previous_right = right
            previous_size = font_size
    return clean_text("".join(output))


def _split_inline_styled_heading(
    spans: list[dict[str, Any]]
) -> tuple[str, str, bool, bool] | None:
    """Split ``italic/bold title: body`` lines into a title and its body.

    Some lecture PDFs place a styled title and the first sentence on the same
    visual line. Treating the line as plain body causes that topic to be merged
    into the preceding section, so a query reaches unrelated preamble first.
    """
    if len(spans) < 2:
        return None
    first_span = spans[0]
    first_text = "".join(char.get("c", "") for char in first_span.get("chars", [])).strip()
    flags = first_span.get("flags", 0)
    if not first_text.endswith(":") or not (flags & (2 | 16)):
        return None
    heading = _line_text_from_spans([first_span])
    body = _line_text_from_spans(spans[1:])
    if len(heading) < 3 or not body:
        return None
    return heading, body, bool(flags & 16), bool(flags & 2)


def _cell_text(value: str | None) -> str:
    return structure_text(value or "").replace("\n", " ").strip()


def _looks_like_header(row: list[str | None]) -> bool:
    values = [_cell_text(value) for value in row if _cell_text(value)]
    if len(values) < 2:
        return False
    numeric = sum(bool(re.fullmatch(r"[\d.,()%–-]+", value)) for value in values)
    return numeric < len(values) / 2


def table_to_text(rows: list[list[str | None]], previous_headers: list[str] | None = None) -> tuple[str, list[str] | None]:
    """Convert a visual table into searchable header/value rows.

    The first header row is retained and reused for continuation tables on later
    pages, which is common in syllabus assessment tables.
    """
    if not rows:
        return "", previous_headers
    headers = previous_headers
    data_rows = rows
    if _looks_like_header(rows[0]):
        headers = [_cell_text(value) or f"Cột {index + 1}" for index, value in enumerate(rows[0])]
        data_rows = rows[1:]
    if not headers:
        headers = [f"Cột {index + 1}" for index in range(max(len(row) for row in rows))]
    output: list[str] = []
    for row_number, row in enumerate(data_rows, start=1):
        values = [_cell_text(value) for value in row]
        pairs = [f"{headers[index] if index < len(headers) else f'Cột {index + 1}'}: {value}"
                 for index, value in enumerate(values) if value]
        if pairs:
            output.append(f"Bảng – hàng {row_number}: " + " | ".join(pairs))
    return "\n".join(output), headers


def table_to_display(
    rows: list[list[str | None]], previous_headers: list[str] | None = None
) -> tuple[list[dict[str, str]], list[str] | None]:
    """Preserve table cells as rows and columns for the Streamlit display.

    ``table_to_text`` remains the representation embedded for Semantic Search.
    This second representation lets the UI render the same source table without
    flattening every cell into one long line.
    """
    if not rows:
        return [], previous_headers
    headers = previous_headers
    data_rows = rows
    if _looks_like_header(rows[0]):
        headers = [_cell_text(value) or f"Cột {index + 1}" for index, value in enumerate(rows[0])]
        data_rows = rows[1:]
    if not headers:
        headers = [f"Cột {index + 1}" for index in range(max(len(row) for row in rows))]

    display_rows: list[dict[str, str]] = []
    for row in data_rows:
        values = [_cell_text(value) for value in row]
        if any(values):
            display_rows.append(
                {headers[index] if index < len(headers) else f"Cột {index + 1}": value
                 for index, value in enumerate(values)}
            )
    return display_rows, headers


def _page_items(
    page: fitz.Page, table_headers: dict[int, list[str]]
) -> list[tuple[float, str, bool, bool, dict[str, Any] | None]]:
    """Extract text and tables in vertical reading order without table duplicates."""
    detected_tables = list(page.find_tables().tables)
    table_boxes = [tuple(table.bbox) for table in detected_tables]
    items: list[tuple[float, str, bool, bool, dict[str, Any] | None]] = []
    for block in page.get_text("rawdict")["blocks"]:
        for line in block.get("lines", []):
            if _inside_table(tuple(line["bbox"]), table_boxes):
                continue
            spans = line.get("spans", [])
            inline_heading = _split_inline_styled_heading(spans)
            if inline_heading:
                heading, body, is_bold, is_italic = inline_heading
                items.append((line["bbox"][1], heading, is_bold, is_italic, None))
                items.append((line["bbox"][1] + 0.001, body, False, False, None))
                continue
            text = _line_text_from_spans(spans)
            if text:
                styled_chars = sum(
                    sum(not char.get("c", "").isspace() for char in span.get("chars", []))
                    for span in spans
                    if span.get("flags", 0) & 16
                )
                italic_chars = sum(
                    sum(not char.get("c", "").isspace() for char in span.get("chars", []))
                    for span in spans
                    if span.get("flags", 0) & 2
                )
                character_count = max(1, sum(not char.get("c", "").isspace() for span in spans for char in span.get("chars", [])))
                items.append((
                    line["bbox"][1],
                    text,
                    styled_chars >= character_count * 0.7,
                    italic_chars >= character_count * 0.7,
                    None,
                ))
    for table in detected_tables:
        rows = table.extract()
        column_count = max((len(row) for row in rows), default=0)
        previous_headers = table_headers.get(column_count)
        table_text, headers = table_to_text(rows, previous_headers)
        display_rows, display_headers = table_to_display(rows, previous_headers)
        if headers:
            table_headers[len(headers)] = headers
        if table_text:
            items.append((
                table.bbox[1],
                table_text,
                False,
                False,
                {"headers": display_headers or headers or [], "rows": display_rows},
            ))
    return sorted(items, key=lambda item: item[0])


def _is_numbered_heading(text: str, is_bold: bool) -> bool:
    if _AUXILIARY_HEADING.match(text):
        return True
    if not _HEADING.match(text):
        # Some outlines use "1 Mục tiêu" without a dot. Limit this relaxed
        # form to short bold lines so years and numbered body sentences remain text.
        return bool(is_bold and _SIMPLE_NUMBERED_HEADING.match(text) and len(text.strip()) <= 140)
    # Numbered syllabus titles may lose bold formatting during PDF extraction;
    # short numbered lines are accepted while table cells are handled separately.
    return is_bold or len(text.strip()) <= 140


def _is_styled_heading(text: str, is_bold: bool, is_italic: bool) -> bool:
    """Recognize short, emphasized titles that do not start with a number."""
    compact = text.strip()
    if not compact or compact[0].isdigit() or _CAPTION_PREFIX.match(compact) or not (is_bold or is_italic):
        return False
    if _LETTERED_HEADING.match(compact):
        return True
    if compact.endswith(":") and len(compact) <= 180:
        return True
    letters = [character for character in compact if character.isalpha()]
    uppercase_ratio = sum(character.isupper() for character in letters) / len(letters) if letters else 0
    if is_bold and uppercase_ratio >= 0.5 and len(compact) <= 180:
        return True
    # A short bold-italic phrase is generally a title in this teaching text.
    # Restrict the fallback to phrase-like lines so wrapped emphasized prose is
    # not accidentally converted into many artificial sections.
    return (
        is_bold
        and is_italic
        and "." not in compact
        and len(compact) <= 100
        and len(compact.split()) <= 10
    )


def _is_heading(text: str, is_bold: bool, is_italic: bool) -> bool:
    return (
        _is_numbered_heading(text, is_bold)
        or bool(_CHAPTER_HEADING.match(text))
        or _is_styled_heading(text, is_bold, is_italic)
    )


def _is_heading_continuation(current: dict[str, Any], line: str, is_bold: bool, is_italic: bool) -> bool:
    """Join wrapped visual title lines before their following section body."""
    current_heading = current.get("section", "")
    if current.get("has_body") or not is_bold or not (_HEADING.match(current_heading) or _CHAPTER_HEADING.match(current_heading)):
        return False
    compact = line.strip()
    return bool(compact) and len(compact) <= 120 and not re.search(r"[.!?]\s*$", compact)


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """Extract continuous numbered syllabus sections, even across page breaks."""
    path = Path(pdf_path)
    pages: list[dict[str, Any]] = []
    source_id = _slug(path.stem)
    section_order = 0
    section = "Không xác định"
    current: dict[str, Any] | None = None
    table_headers: dict[int, list[str]] = {}

    def finish_current() -> None:
        """Finalize a section while keeping search text separate from its tables."""
        if not current or not current["text"]:
            return
        current["text"] = structure_text("\n".join(current["text"]))
        current["full_text"] = structure_text("\n".join(current["display_text"]))
        current["page_start"] = current["page"]
        current["page_end"] = current["pages"][-1]
        pages.append(current)

    with fitz.open(path) as document:
        for number, page in enumerate(document, start=1):
            items = _page_items(page, table_headers)
            if not items:
                continue
            for _, line, is_bold, is_italic, table_data in items:
                explicit_heading = _is_numbered_heading(line, is_bold) or bool(_CHAPTER_HEADING.match(line))
                styled_heading = _is_styled_heading(line, is_bold, is_italic)
                continuation = bool(
                    current
                    and current["page"] == number
                    and not explicit_heading
                    and (bool(_CHAPTER_HEADING.match(current["section"])) or not styled_heading)
                    and _is_heading_continuation(current, line, is_bold, is_italic)
                )
                if explicit_heading or styled_heading or continuation:
                    if continuation:
                        combined_heading = f"{current['section']} {line}".strip()
                        current["section"] = combined_heading
                        current["text"][0] = combined_heading
                        current["display_text"][0] = combined_heading
                        continue
                    finish_current()
                    section_order += 1
                    section = line
                    current = {"source": path.name, "page": number, "pages": [number], "section": section,
                               "section_id": f"{source_id}__{section_order:03d}", "text": [line],
                               "display_text": [line], "tables": [], "has_body": False}
                else:
                    if current is None:
                        section_order += 1
                        current = {"source": path.name, "page": number, "pages": [number], "section": section,
                                   "section_id": f"{source_id}__{section_order:03d}", "text": [],
                                   "display_text": [], "tables": [], "has_body": False}
                    if number not in current["pages"]:
                        current["pages"].append(number)
                    current["text"].append(line)
                    if table_data is None:
                        current["display_text"].append(line)
                    else:
                        current["tables"].append({**table_data, "page": number})
                        current["display_text"].append("[Bảng được hiển thị bên dưới]")
                    current["has_body"] = True
    finish_current()
    return pages


def validate_corpus(pdf_dir: str | Path, minimum_files: int = 3, minimum_pages: int = 5) -> list[str]:
    files = sorted(Path(pdf_dir).glob("*.pdf"))
    errors: list[str] = []
    if len(files) < minimum_files:
        errors.append(f"Cần ít nhất {minimum_files} PDF, hiện có {len(files)}.")
    for path in files:
        with fitz.open(path) as doc:
            if len(doc) < minimum_pages:
                errors.append(f"{path.name} chỉ có {len(doc)} trang (yêu cầu ≥ {minimum_pages}).")
    return errors


def validate_pdf_file(pdf_path: str | Path, minimum_pages: int = 5) -> str | None:
    """Return a user-facing error when an uploaded PDF is not usable."""
    path = Path(pdf_path)
    try:
        with fitz.open(path) as document:
            if len(document) < minimum_pages:
                return f"{path.name} có {len(document)} trang; cần tối thiểu {minimum_pages} trang."
    except (fitz.FileDataError, OSError) as error:
        return f"Không thể đọc {path.name}: {error}"
    return None


def validate_pdf_bytes(data: bytes, filename: str, minimum_pages: int = 5) -> str | None:
    """Validate an upload before replacing an existing PDF with the same name."""
    try:
        with fitz.open(stream=data, filetype="pdf") as document:
            if len(document) < minimum_pages:
                return f"{filename} có {len(document)} trang; cần tối thiểu {minimum_pages} trang."
    except (fitz.FileDataError, OSError) as error:
        return f"Không thể đọc {filename}: {error}"
    return None
