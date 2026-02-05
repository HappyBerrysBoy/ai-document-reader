import os
import sys
import logging
from pathlib import Path
import torch
import soundfile as sf
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Qwen3-TTS 모델 캐시
_model_cache = None
_tokenizer_cache = None


def _load_model(model_type="custom_voice"):
    """
    Qwen3-TTS 모델 로드

    Args:
        model_type: "custom_voice", "voice_design", "base" 중 선택
    """
    global _model_cache, _tokenizer_cache

    if _model_cache is None or _model_cache.get("type") != model_type:
        logger.info(f"Qwen3-TTS 모델 로딩 중 (타입: {model_type})...")

        # 모델 이름 매핑
        model_names = {
            "custom_voice": "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
            "voice_design": "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
            "base": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        }

        model_name = model_names.get(model_type, model_names["custom_voice"])
        tokenizer_name = "Qwen/Qwen3-TTS-Tokenizer-12Hz"

        # GPU 확인
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"사용 장치: {device}")

        if device == "cpu":
            logger.warning("⚠️  GPU를 사용할 수 없습니다. CPU 모드로 실행됩니다 (매우 느림).")

        try:
            from qwen_tts import QwenTTS

            # 모델 로드
            model = QwenTTS(
                model_name=model_name,
                tokenizer_name=tokenizer_name,
                device=device,
                torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
            )

            _model_cache = {
                "model": model,
                "type": model_type,
                "device": device,
            }

            logger.info("✓ 모델 로딩 완료")

        except Exception as e:
            logger.error(f"❌ 모델 로딩 실패: {e}")
            logger.error("다음 명령어로 수동 다운로드를 시도하세요:")
            logger.error(f"  huggingface-cli download {model_name}")
            logger.error(f"  huggingface-cli download {tokenizer_name}")
            raise

    return _model_cache["model"]


def synthesize_with_voice_cloning(
    text: str,
    ref_audio_path: str,
    ref_text: str,
    output_path: str = "output_tts.wav",
    speed: float = 1.0,
) -> str:
    """
    Voice Cloning으로 음성 합성

    Args:
        text: 합성할 텍스트
        ref_audio_path: 참조 음성 파일 경로 (WAV, 최소 3초 권장)
        ref_text: 참조 음성의 원본 텍스트
        output_path: 출력 파일 경로
        speed: 말하기 속도 (기본값 1.0)

    Returns:
        출력 파일 경로
    """
    logger.info(f"Voice Cloning TTS 시작")
    logger.info(f"텍스트: {text[:100]}...")
    logger.info(f"참조 음성: {ref_audio_path}")

    if not os.path.exists(ref_audio_path):
        raise FileNotFoundError(f"참조 음성 파일을 찾을 수 없습니다: {ref_audio_path}")

    # 모델 로드
    model = _load_model(model_type="custom_voice")

    try:
        # 음성 합성
        logger.info("음성 합성 중...")
        audio_output = model.synthesize(
            text=text,
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            speed=speed,
        )

        # 오디오 저장
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # audio_output이 numpy array인 경우
        if isinstance(audio_output, np.ndarray):
            sf.write(str(output_path), audio_output, samplerate=12000)
        # audio_output이 torch tensor인 경우
        elif isinstance(audio_output, torch.Tensor):
            audio_np = audio_output.cpu().numpy()
            sf.write(str(output_path), audio_np, samplerate=12000)
        else:
            # 기타 형식인 경우 모델의 save 메서드 사용
            audio_output.save(str(output_path))

        logger.info(f"✓ 음성 합성 완료: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"음성 합성 중 오류: {e}")
        raise


def synthesize_with_voice_design(
    text: str,
    voice_description: str,
    output_path: str = "output_tts.wav",
    speed: float = 1.0,
) -> str:
    """
    Voice Design으로 음성 합성 (텍스트로 음성 특성 설명)

    Args:
        text: 합성할 텍스트
        voice_description: 음성 특성 설명 (예: "젊은 여성의 밝고 활기찬 목소리")
        output_path: 출력 파일 경로
        speed: 말하기 속도 (기본값 1.0)

    Returns:
        출력 파일 경로
    """
    logger.info(f"Voice Design TTS 시작")
    logger.info(f"텍스트: {text[:100]}...")
    logger.info(f"음성 특성: {voice_description}")

    # 모델 로드
    model = _load_model(model_type="voice_design")

    try:
        # 음성 합성
        logger.info("음성 합성 중...")
        audio_output = model.synthesize(
            text=text,
            voice_design=voice_description,
            speed=speed,
        )

        # 오디오 저장
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(audio_output, np.ndarray):
            sf.write(str(output_path), audio_output, samplerate=12000)
        elif isinstance(audio_output, torch.Tensor):
            audio_np = audio_output.cpu().numpy()
            sf.write(str(output_path), audio_np, samplerate=12000)
        else:
            audio_output.save(str(output_path))

        logger.info(f"✓ 음성 합성 완료: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"음성 합성 중 오류: {e}")
        raise


def synthesize_basic(
    text: str,
    output_path: str = "output_tts.wav",
    speed: float = 1.0,
) -> str:
    """
    기본 음성 합성

    Args:
        text: 합성할 텍스트
        output_path: 출력 파일 경로
        speed: 말하기 속도 (기본값 1.0)

    Returns:
        출력 파일 경로
    """
    logger.info(f"기본 TTS 시작")
    logger.info(f"텍스트: {text[:100]}...")

    # 모델 로드
    model = _load_model(model_type="base")

    try:
        # 음성 합성
        logger.info("음성 합성 중...")
        audio_output = model.synthesize(
            text=text,
            speed=speed,
        )

        # 오디오 저장
        output_path = Path(output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(audio_output, np.ndarray):
            sf.write(str(output_path), audio_output, samplerate=12000)
        elif isinstance(audio_output, torch.Tensor):
            audio_np = audio_output.cpu().numpy()
            sf.write(str(output_path), audio_np, samplerate=12000)
        else:
            audio_output.save(str(output_path))

        logger.info(f"✓ 음성 합성 완료: {output_path}")
        return str(output_path)

    except Exception as e:
        logger.error(f"음성 합성 중 오류: {e}")
        raise


if __name__ == "__main__":
    # 테스트
    if len(sys.argv) > 1:
        test_text = sys.argv[1]
        output = synthesize_basic(test_text, "test_output.wav")
        print(f"음성 파일 생성: {output}")
