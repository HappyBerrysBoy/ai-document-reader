import argparse
import sys
import os
from pathlib import Path

from loader import load_document
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="문서 OCR: PDF/이미지 등에서 텍스트를 추출해 파일로 저장합니다."
    )
    parser.add_argument(
        "file",
        help="문서 경로 (PDF, 이미지, DOCX, XLSX, PPTX)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="저장할 텍스트 파일 경로 (미지정 시 입력파일명_ocr.txt)",
    )
    args = parser.parse_args()

    if not os.path.exists(args.file):
        logger.error(f"Error: File '{args.file}' not found.")
        sys.exit(1)

    try:
        logger.info(f"OCR 실행: {args.file}")
        text = load_document(args.file)

        if not text.strip():
            logger.warning("추출된 텍스트가 없습니다.")

        if args.output:
            out_path = Path(args.output)
        else:
            p = Path(args.file)
            out_path = p.parent / f"{p.stem}_ocr.txt"

        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        logger.info(f"결과 저장: {out_path}")
        print(f"\n저장됨: {out_path}\n")

    except Exception as e:
        logger.error(f"An error occurred: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
