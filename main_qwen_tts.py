import argparse
import sys
import os
import time
from pathlib import Path
import logging

from loader_qwen_tts import (
    synthesize_with_voice_cloning,
    synthesize_with_voice_design,
    synthesize_basic,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s: %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Qwen3-TTS: 음성 합성 (Voice Cloning, Voice Design, GPU 가속)"
    )
    parser.add_argument(
        "-t", "--text",
        required=True,
        help="합성할 텍스트",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        help="출력 오디오 파일 경로 (미지정 시 output_tts.wav)",
    )
    parser.add_argument(
        "--mode",
        choices=["clone", "design", "basic"],
        default="basic",
        help="모드 선택: clone(음성복제), design(음성설계), basic(기본)",
    )

    # Voice Cloning 옵션
    parser.add_argument(
        "--ref_audio",
        help="[Voice Clone] 참조 음성 파일 경로 (WAV, 최소 3초 권장)",
    )
    parser.add_argument(
        "--ref_text",
        help="[Voice Clone] 참조 음성의 원본 텍스트",
    )

    # Voice Design 옵션
    parser.add_argument(
        "--voice_design",
        help="[Voice Design] 음성 특성 설명 (예: '젊은 여성의 밝고 활기찬 목소리')",
    )

    # 공통 옵션
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help="말하기 속도 (기본값 1.0, 범위: 0.5~2.0)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="상세 로그 출력",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # 출력 파일 경로 설정
    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = Path("output_tts.wav").resolve()

    # 시작 시간 기록
    start_time = time.time()

    try:
        logger.info("=" * 60)
        logger.info("Qwen3-TTS 음성 합성 시작")
        logger.info(f"모드: {args.mode}")
        logger.info(f"텍스트: {args.text[:100]}...")
        logger.info("=" * 60)

        # 모드별 처리
        tts_start = time.time()

        if args.mode == "clone":
            # Voice Cloning
            if not args.ref_audio or not args.ref_text:
                logger.error("❌ Voice Clone 모드는 --ref_audio, --ref_text가 필요합니다.")
                sys.exit(1)

            if not os.path.exists(args.ref_audio):
                logger.error(f"❌ 참조 음성 파일을 찾을 수 없습니다: {args.ref_audio}")
                sys.exit(1)

            output_file = synthesize_with_voice_cloning(
                text=args.text,
                ref_audio_path=args.ref_audio,
                ref_text=args.ref_text,
                output_path=str(output_path),
                speed=args.speed,
            )

        elif args.mode == "design":
            # Voice Design
            if not args.voice_design:
                logger.error("❌ Voice Design 모드는 --voice_design이 필요합니다.")
                sys.exit(1)

            output_file = synthesize_with_voice_design(
                text=args.text,
                voice_description=args.voice_design,
                output_path=str(output_path),
                speed=args.speed,
            )

        else:
            # Basic
            output_file = synthesize_basic(
                text=args.text,
                output_path=str(output_path),
                speed=args.speed,
            )

        tts_time = time.time() - tts_start
        total_time = time.time() - start_time

        logger.info("=" * 60)
        logger.info(f"✓ 음성 합성 완료")
        logger.info(f"✓ 출력 파일: {output_file}")
        logger.info(f"✓ TTS 처리 시간: {tts_time:.2f}초")
        logger.info(f"✓ 전체 소요 시간: {total_time:.2f}초")
        logger.info("=" * 60)

        print(f"\n음성 파일이 생성되었습니다: {output_file}\n")

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
