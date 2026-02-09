import argparse
import sys
import os
import time
from pathlib import Path
import logging

from loader_deepseek import load_document

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="DeepSeek OCR 기반 문서 OCR (GPU 가속)"
    )
    parser.add_argument(
        "file",
        help="문서 경로 (PDF, 이미지, DOCX, XLSX, PPTX)",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="저장할 텍스트 파일 경로 (미지정 시 입력파일명_deepseek.txt)",
    )
    parser.add_argument(
        "-m", "--mode",
        default="gundam",
        choices=["tiny", "small", "base", "large", "gundam"],
        help="OCR 모드 (기본값: gundam)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.file):
        logger.error(f"파일을 찾을 수 없습니다: {args.file}")
        sys.exit(1)

    # 시작 시간 기록
    start_time = time.time()

    try:
        logger.info("=" * 60)
        logger.info("DeepSeek OCR 시작")
        logger.info(f"입력 파일: {args.file}")
        logger.info(f"OCR 모드: {args.mode}")
        logger.info(f"공식 권장: .eval().cuda().to(bfloat16)")
        if args.mode == "gundam":
            logger.info(f"Gundam 모드: base=1024, img=640, crop=True (권장)")
        logger.info("=" * 60)

        # OCR 실행
        ocr_start = time.time()
        text = load_document(args.file, mode=args.mode, save_to_file=False)
        ocr_time = time.time() - ocr_start

        if not text.strip():
            logger.warning("⚠️  추출된 텍스트가 없습니다.")

        # 출력 파일 경로
        if args.output:
            out_path = Path(args.output)
        else:
            p = Path(args.file)
            out_path = p.parent / f"{p.stem}_deepseek.txt"

        # 저장
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")

        # 전체 소요 시간
        total_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info(f"✓ OCR 완료")
        logger.info(f"✓ 결과 저장: {out_path}")
        logger.info(f"✓ 추출된 문자 수: {len(text)}")
        logger.info(f"✓ OCR 처리 시간: {ocr_time:.2f}초")
        logger.info(f"✓ 전체 소요 시간: {total_time:.2f}초")
        logger.info("=" * 60)

        # 미리보기
        if text:
            print("\n--- 추출된 텍스트 미리보기 (처음 300자) ---")
            print(text[:300])
            if len(text) > 300:
                print("...")
            print(f"\n전체 내용: {out_path}\n")

    except Exception as e:
        logger.error("=" * 60)
        logger.error(f"❌ 오류 발생: {str(e)}")
        logger.error("=" * 60)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
