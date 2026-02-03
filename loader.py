import os
import sys
import tempfile
import logging
from pathlib import Path

# PaddleX가 참조하는 구 langchain 경로 호환
if "langchain.docstore.document" not in sys.modules:
    import langchain_core.documents as _lc_docs
    _docmod = type(sys)("langchain.docstore.document")
    _docmod.Document = _lc_docs.Document
    sys.modules["langchain.docstore.document"] = _docmod
    if "langchain.docstore" not in sys.modules:
        _ds = type(sys)("langchain.docstore")
        sys.modules["langchain.docstore"] = _ds
if "langchain.text_splitter" not in sys.modules:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    _tsmod = type(sys)("langchain.text_splitter")
    _tsmod.RecursiveCharacterTextSplitter = RecursiveCharacterTextSplitter
    sys.modules["langchain.text_splitter"] = _tsmod

import fitz  # PyMuPDF
from paddleocr import PaddleOCR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# PaddleOCR 인스턴스 (PaddleOCR 3.x / PaddleX API)
_ocr = None

def _get_ocr():
    global _ocr
    if _ocr is None:
        _ocr = PaddleOCR(
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            lang="korean",
            ocr_version="PP-OCRv3",
        )
    return _ocr

def _extract_text_from_paddle3_result(result_list):
    """PaddleOCR 3.x predict() 결과에서 텍스트만 추출 (rec_texts)."""
    if not result_list:
        return ""
    lines = []
    for res in result_list:
        data = getattr(res, "json", None) or (res if isinstance(res, dict) else {})
        res_inner = (data.get("res") or data) if isinstance(data, dict) else {}
        rec_texts = res_inner.get("rec_texts") or []
        for t in rec_texts:
            if t and str(t).strip():
                lines.append(str(t).strip())
    return "\n".join(lines)

def _ocr_image(ocr, image_path: str) -> str:
    result = ocr.predict(image_path)
    return _extract_text_from_paddle3_result(result)

def _load_pdf_with_paddleocr(file_path: str) -> str:
    ocr = _get_ocr()
    doc = fitz.open(file_path)
    texts = []
    try:
        for i in range(len(doc)):
            page = doc.load_page(i)
            pix = page.get_pixmap(alpha=False, dpi=150)
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                pix.save(f.name)
                try:
                    page_text = _ocr_image(ocr, f.name)
                    if page_text:
                        texts.append(page_text)
                finally:
                    os.unlink(f.name)
    finally:
        doc.close()
    return "\n\n".join(texts)

def _load_image_with_paddleocr(file_path: str) -> str:
    ocr = _get_ocr()
    return _ocr_image(ocr, file_path)

def _load_docx(file_path: str) -> str:
    from docx import Document
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)

def _load_xlsx(file_path: str) -> str:
    import openpyxl
    wb = openpyxl.load_workbook(file_path, read_only=True, data_only=True)
    parts = []
    for sheet in wb.worksheets:
        for row in sheet.iter_rows(values_only=True):
            line = "\t".join(str(c) if c is not None else "" for c in row).strip()
            if line:
                parts.append(line)
    return "\n".join(parts)

def _load_pptx(file_path: str) -> str:
    from pptx import Presentation
    prs = Presentation(file_path)
    parts = []
    for slide in prs.slides:
        for shape in slide.shapes:
            if hasattr(shape, "text") and shape.text.strip():
                parts.append(shape.text.strip())
    return "\n".join(parts)

def load_document(file_path: str) -> str:
    """
    PaddleOCR(이미지/PDF) 또는 기본 라이브러리(DOCX, XLSX, PPTX)로 문서 텍스트 추출.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Loading document: {file_path}")
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _load_pdf_with_paddleocr(file_path)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}:
        return _load_image_with_paddleocr(file_path)
    if suffix == ".docx":
        return _load_docx(file_path)
    if suffix == ".xlsx":
        return _load_xlsx(file_path)
    if suffix == ".pptx":
        return _load_pptx(file_path)

    # 기본: 이미지로 시도 후 실패 시 에러
    try:
        return _load_image_with_paddleocr(file_path)
    except Exception:
        raise ValueError(
            f"Unsupported file type: {suffix}. "
            "Supported: PDF, PNG, JPG, BMP, GIF, WEBP, TIFF, DOCX, XLSX, PPTX."
        )

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        text = load_document(sys.argv[1])
        print("--- Extracted Text (First 500 chars) ---")
        print(text[:500])
