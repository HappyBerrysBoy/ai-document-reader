#!/usr/bin/env python3
"""
Qwen3-VL 모델을 사용한 텍스트 요약 도구

사용법:
    python summarize_qwen.py input.txt
    python summarize_qwen.py input.txt -o summary.txt
"""

import argparse
import sys
import os
import time
from pathlib import Path
import logging
import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# Qwen3-VL 모델 캐시
_model_cache = None
_processor_cache = None


def _load_model():
    """Qwen3-VL 모델 로드"""
    global _model_cache, _processor_cache

    if _model_cache is None:
        logger.info("Qwen3-VL 모델 로딩 중...")

        model_name = "Qwen/Qwen3-VL-4B-Instruct"

        # GPU 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"사용 장치: {device}")

        if device == "cpu":
            logger.warning("⚠️  GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다 (매우 느림).")

        try:
            # Processor 로드
            _processor_cache = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True
            )

            # Model 로드
            _model_cache = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
                device_map="auto" if device == "cuda" else None,
            )

            if device == "cpu":
                _model_cache = _model_cache.to(device)

            _model_cache.eval()
            logger.info("✓ 모델 로딩 완료")

        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            logger.error("다음 명령어로 수동 다운로드를 시도하세요:")
            logger.error(f"  huggingface-cli download {model_name}")
            raise

    return _model_cache, _processor_cache


def summarize_text(text: str, max_chars: int = 6000) -> str:
    """
    Qwen3-VL 모델을 사용하여 텍스트 요약 생성

    Args:
        text: 요약할 텍스트
        max_chars: 최대 입력 문자 수

    Returns:
        요약된 텍스트
    """
    model, processor = _load_model()
    device = next(model.parameters()).device

    logger.info("텍스트 요약 생성 중...")

    # 텍스트가 너무 길면 앞부분만 사용 (토큰 제한)
    if len(text) > max_chars:
        text_to_summarize = text[:max_chars] + "..."
        logger.info(f"텍스트가 길어서 처음 {max_chars}자만 요약합니다.")
    else:
        text_to_summarize = text

    # 요약 프롬프트
    prompt = f"""다음 문서의 내용을 한글로 요약해주세요. 주요 내용과 핵심 포인트를 포함하여 3-5개 문단으로 작성해주세요.

문서 내용:
{text_to_summarize}

요약:"""

    try:
        # 메시지 구성
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
        ]

        # 입력 준비
        text_prompt = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            text=[text_prompt],
            return_tensors="pt",
        )
        inputs = inputs.to(device)

        # 요약 생성
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=768,
                do_sample=False,
            )

        # 디코딩 (입력 토큰 제외)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, outputs)
        ]

        summary = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        logger.info("✓ 요약 생성 완료")
        return summary.strip()

    except Exception as e:
        logger.error(f"요약 생성 중 오류: {e}")
        return f"요약 생성 실패: {str(e)}"


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-VL 모델을 사용한 텍스트 요약"
    )
    parser.add_argument(
        "input",
        help="입력 텍스트 파일 경로",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="저장할 요약 파일 경로 (미지정 시 입력파일명_summary.txt)",
    )
    parser.add_argument(
        "--max-chars",
        type=int,
        default=6000,
        help="최대 입력 문자 수 (기본값: 6000)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    if not os.path.exists(args.input):
        logger.error(f"파일을 찾을 수 없습니다: {args.input}")
        sys.exit(1)

    # 시작 시간 기록
    start_time = time.time()

    try:
        logger.info("=" * 60)
        logger.info("Qwen3-VL 텍스트 요약 시작")
        logger.info(f"입력 파일: {args.input}")
        logger.info("=" * 60)

        # 텍스트 파일 읽기
        input_path = Path(args.input)
        text = input_path.read_text(encoding="utf-8")

        if not text.strip():
            logger.warning("⚠️  입력 텍스트가 비어있습니다.")
            sys.exit(1)

        logger.info(f"입력 텍스트 길이: {len(text)} 자")

        # 요약 생성
        summary = summarize_text(text, max_chars=args.max_chars)

        # 출력 파일 경로
        if args.output:
            out_path = Path(args.output)
        else:
            out_path = input_path.parent / f"{input_path.stem}_summary.txt"

        # 저장
        out_path = out_path.resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(summary, encoding="utf-8")

        # 전체 소요 시간
        total_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info(f"✓ 요약 완료")
        logger.info(f"✓ 결과 저장: {out_path}")
        logger.info(f"✓ 요약 길이: {len(summary)} 자")
        logger.info(f"✓ 소요 시간: {total_time:.2f}초")
        logger.info("=" * 60)

        # 미리보기
        print("\n--- 요약 미리보기 ---")
        print(summary)
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
