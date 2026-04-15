import re
from pathlib import Path
from typing import Dict, List

import fitz
from pypdf import PdfReader, PdfWriter

try:
    from rapidocr_onnxruntime import RapidOCR
except Exception:  # pragma: no cover
    RapidOCR = None


_ocr_engine = None


def _extract_text_with_ocr(pdf_path: Path) -> Dict[int, str]:
    global _ocr_engine
    if RapidOCR is None:
        return {}
    if _ocr_engine is None:
        _ocr_engine = RapidOCR()

    ocr_results: Dict[int, str] = {}
    doc = fitz.open(str(pdf_path))
    for idx, page in enumerate(doc, start=1):
        pix = page.get_pixmap(dpi=250)
        img_bytes = pix.tobytes("png")
        result, _ = _ocr_engine(img_bytes)
        if not result:
            ocr_results[idx] = ""
            continue
        lines = [item[1] for item in result if len(item) > 1 and item[1]]
        ocr_results[idx] = "\n".join(lines).strip()
    doc.close()
    return ocr_results


def extract_pdf_pages_text(pdf_path: Path) -> Dict[int, str]:
    reader = PdfReader(str(pdf_path))
    page_text: Dict[int, str] = {}
    for idx, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        page_text[idx] = text.strip()
    # OCR fallback for image-only PDFs
    if any(page_text.values()):
        return page_text

    ocr_text = _extract_text_with_ocr(pdf_path)
    if ocr_text:
        return ocr_text
    return page_text


def save_selected_pages_as_pdf(
    source_pdf: Path,
    page_numbers: List[int],
    output_path: Path,
) -> Path:
    reader = PdfReader(str(source_pdf))
    writer = PdfWriter()
    total_pages = len(reader.pages)
    for page_num in page_numbers:
        if 1 <= page_num <= total_pages:
            writer.add_page(reader.pages[page_num - 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        writer.write(f)
    return output_path


def split_lines(text: str) -> List[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def first_match(pattern: str, text: str, flags: int = re.IGNORECASE) -> str:
    match = re.search(pattern, text, flags=flags)
    return match.group(1).strip() if match else ""


def extract_money_values(text: str) -> List[float]:
    values: List[float] = []
    for match in re.finditer(r"(?:INR|Rs\.?|₹)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{1,2})?|[0-9]+(?:\.[0-9]{1,2})?)", text):
        value = match.group(1).replace(",", "")
        try:
            values.append(float(value))
        except ValueError:
            continue
    return values

