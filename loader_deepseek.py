import os
import sys
import tempfile
import logging
from pathlib import Path
from typing import List
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import fitz  # PyMuPDF

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# DeepSeek 모델 캐시
_model_cache = None
_tokenizer_cache = None


def _load_model():
    """DeepSeek VL2 모델 로드"""
    global _model_cache, _tokenizer_cache

    if _model_cache is None:
        logger.info("DeepSeek VL2 모델 로딩 중...")

        model_name = "deepseek-ai/deepseek-vl2"

        # GPU 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"사용 장치: {device}")

        if device == "cpu":
            logger.warning("⚠️  GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다 (매우 느림).")

        try:
            # Tokenizer 로드
            _tokenizer_cache = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Model 로드 (trust_remote_code=True로 커스텀 모델 코드 사용)
            _model_cache = AutoModel.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            ).to(device)

            _model_cache.eval()
            logger.info("✓ 모델 로딩 완료")

        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            logger.error("다음 명령어로 수동 다운로드를 시도하세요:")
            logger.error(f"  huggingface-cli download {model_name}")
            raise

    return _model_cache, _tokenizer_cache


def _extract_text_from_image(image: Image.Image) -> str:
    """
    DeepSeek VL2로 이미지에서 텍스트 추출
    """
    model, tokenizer = _load_model()
    device = next(model.parameters()).device

    # 이미지 크기 조정 (너무 크면 메모리 초과)
    max_size = 1024
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
        logger.info(f"이미지 리사이징: {new_size}")

    # 이미지를 임시 파일로 저장
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        image.save(tmp.name)
        image_path = tmp.name

    try:
        # OCR 프롬프트 (한글/영어 최적화)
        conversation = [
            {
                "role": "User",
                "content": f"<image>\nExtract all text from this image accurately. The text may contain Korean (한글) and English characters. Please preserve the exact text layout and return only the extracted text without any additional explanation.",
                "images": [image_path]
            },
            {
                "role": "Assistant",
                "content": ""
            }
        ]

        # 입력 준비 (DeepSeek VL2 전용 API)
        prepare_inputs = model.prepare_inputs_for_generation
        inputs = model.build_conversation_input_ids(
            tokenizer=tokenizer,
            conversation=conversation,
            images=[image_path],
            force_batchify=True
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 추론
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2048,
                do_sample=False,
            )

        # 텍스트 디코딩
        text = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)

        return text.strip()

    except Exception as e:
        logger.error(f"OCR 처리 중 오류: {e}")
        raise
    finally:
        # 임시 파일 삭제
        if os.path.exists(image_path):
            os.unlink(image_path)


def _ocr_image(image_path: str) -> str:
    """이미지 파일 OCR"""
    logger.info(f"이미지 OCR: {image_path}")

    try:
        image = Image.open(image_path).convert("RGB")
        return _extract_text_from_image(image)
    except Exception as e:
        logger.error(f"이미지 로드 실패: {e}")
        raise


def _load_pdf_with_deepseek(file_path: str) -> str:
    """PDF 파일 OCR"""
    logger.info(f"PDF 파일 OCR: {file_path}")

    doc = fitz.open(file_path)
    texts = []

    try:
        total_pages = len(doc)
        logger.info(f"총 {total_pages} 페이지")

        for i in range(total_pages):
            logger.info(f"페이지 {i+1}/{total_pages} 처리 중...")
            page = doc.load_page(i)

            # 페이지를 이미지로 변환 (DPI 150)
            pix = page.get_pixmap(alpha=False, dpi=150)

            # PIL Image로 변환
            img_data = pix.tobytes("png")
            image = Image.open(io.BytesIO(img_data))

            # OCR
            page_text = _extract_text_from_image(image)
            if page_text:
                texts.append(page_text)

    finally:
        doc.close()

    return "\n\n".join(texts)


def _load_docx(file_path: str) -> str:
    """DOCX 텍스트 추출"""
    from docx import Document
    doc = Document(file_path)
    return "\n".join(p.text for p in doc.paragraphs)


def _load_xlsx(file_path: str) -> str:
    """XLSX 텍스트 추출"""
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
    """PPTX 텍스트 추출"""
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
    DeepSeek VL2로 문서 텍스트 추출

    Args:
        file_path: 문서 경로

    Returns:
        추출된 텍스트
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}")

    path = Path(file_path)
    suffix = path.suffix.lower()

    logger.info(f"문서 로딩: {file_path}")

    if suffix == ".pdf":
        return _load_pdf_with_deepseek(file_path)
    elif suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tiff", ".tif"}:
        return _ocr_image(file_path)
    elif suffix == ".docx":
        return _load_docx(file_path)
    elif suffix == ".xlsx":
        return _load_xlsx(file_path)
    elif suffix == ".pptx":
        return _load_pptx(file_path)
    else:
        # 기본: 이미지로 시도
        try:
            return _ocr_image(file_path)
        except:
            raise ValueError(
                f"지원하지 않는 파일 형식: {suffix}. "
                "지원 형식: PDF, PNG, JPG, BMP, GIF, WEBP, TIFF, DOCX, XLSX, PPTX."
            )


# PDF 처리를 위한 io import 추가
import io


if __name__ == "__main__":
    if len(sys.argv) > 1:
        text = load_document(sys.argv[1])
        print("--- 추출된 텍스트 (처음 500자) ---")
        print(text[:500])
