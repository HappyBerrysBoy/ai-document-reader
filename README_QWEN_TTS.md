# Qwen3-TTS 음성 합성 가이드

Qwen3-TTS는 Alibaba Cloud의 Qwen 팀에서 개발한 오픈소스 Text-to-Speech 모델입니다.
3초 음성 샘플로 Voice Cloning이 가능하며, 한글/영어를 포함한 10개 언어를 지원합니다.

## 주요 기능

- **Voice Cloning**: 3초 음성 샘플로 목소리 복제
- **Voice Design**: 텍스트로 원하는 음성 특성 설명 (예: "젊은 여성의 밝고 활기찬 목소리")
- **다국어 지원**: 한국어, 영어, 일본어, 중국어 등 10개 언어
- **GPU 가속**: CUDA 지원으로 빠른 음성 합성
- **스트리밍 생성**: 실시간 음성 생성 (지연시간 97ms)

## 1. 환경 설정

### 1.1 Conda 환경 생성

```bash
bash setup_qwen_tts_env.sh
conda activate qwen-tts
```

### 1.2 패키지 설치

```bash
bash install_qwen_tts.sh
```

**설치되는 주요 패키지:**
- PyTorch (CUDA 12.1)
- Transformers
- qwen-tts
- Flash Attention (선택, 2-3배 속도 향상)
- soundfile, librosa, scipy (오디오 처리)

## 2. 사용 방법

### 2.1 기본 음성 합성

가장 간단한 사용 방법입니다.

```bash
python main_qwen_tts.py \
  --text "안녕하세요, Qwen3-TTS를 테스트합니다." \
  --mode basic \
  --output basic_output.wav
```

### 2.2 Voice Cloning (음성 복제)

**3초 이상의 깨끗한 음성 샘플**이 필요합니다.

```bash
python main_qwen_tts.py \
  --text "오늘 날씨가 정말 좋네요. 산책하기 딱 좋은 날씨입니다." \
  --mode clone \
  --ref_audio my_voice_sample.wav \
  --ref_text "이것은 나의 음성 샘플입니다. 이 목소리로 복제됩니다." \
  --output cloned_voice.wav
```

**중요한 팁:**
- 참조 음성은 **조용한 환경**에서 녹음
- **최소 3초** 길이 권장
- 배경 소음이 없는 **고품질** 오디오 사용
- 참조 음성의 **텍스트를 정확히 입력** (ref_text)

**음성 샘플 녹음 방법:**
```bash
# 맥에서 간단한 녹음
# QuickTime Player > File > New Audio Recording

# 리눅스에서 녹음 (arecord 사용)
arecord -d 5 -f cd -t wav my_voice_sample.wav
```

### 2.3 Voice Design (음성 설계)

음성 샘플 없이 **텍스트 설명만으로** 원하는 음성을 생성합니다.

```bash
python main_qwen_tts.py \
  --text "안녕하세요, 저는 AI 비서입니다." \
  --mode design \
  --voice_design "젊은 여성의 밝고 친근한 목소리, 약간 높은 톤" \
  --output designed_voice.wav
```

**Voice Design 예시:**
- "중년 남성의 차분하고 신뢰감 있는 목소리"
- "어린 소녀의 귀엽고 발랄한 목소리"
- "전문 아나운서의 명료하고 정확한 발음"
- "노년 남성의 따뜻하고 지혜로운 목소리"

### 2.4 말하기 속도 조절

```bash
python main_qwen_tts.py \
  --text "빠르게 말하는 테스트입니다." \
  --mode basic \
  --speed 1.5 \
  --output fast_speech.wav

# 천천히 말하기
python main_qwen_tts.py \
  --text "천천히 말하는 테스트입니다." \
  --mode basic \
  --speed 0.7 \
  --output slow_speech.wav
```

- `speed=1.0`: 기본 속도
- `speed=0.5~0.9`: 느리게
- `speed=1.1~2.0`: 빠르게

## 3. OCR + TTS 통합 사용

OCR로 추출한 텍스트를 음성으로 변환:

```bash
# Step 1: OCR로 텍스트 추출
python main_qwen.py document.pdf -o extracted_text.txt

# Step 2: 추출된 텍스트를 음성으로 변환
TEXT=$(cat extracted_text.txt)
python main_qwen_tts.py \
  --text "$TEXT" \
  --mode clone \
  --ref_audio my_voice.wav \
  --ref_text "참조 음성 텍스트" \
  --output document_audio.wav
```

## 4. 고급 사용법

### 4.1 긴 텍스트 처리

긴 텍스트는 자동으로 분할 처리됩니다. 하지만 수동으로 단락을 나누면 더 자연스럽습니다.

```python
# Python 스크립트 예시
from loader_qwen_tts import synthesize_with_voice_cloning
import soundfile as sf
import numpy as np

# 긴 텍스트를 단락으로 분할
paragraphs = [
    "첫 번째 단락입니다.",
    "두 번째 단락입니다.",
    "세 번째 단락입니다.",
]

audio_chunks = []
for para in paragraphs:
    audio = synthesize_with_voice_cloning(
        text=para,
        ref_audio_path="voice.wav",
        ref_text="참조 텍스트",
        output_path=f"temp_{len(audio_chunks)}.wav"
    )
    data, sr = sf.read(audio)
    audio_chunks.append(data)

# 음성 조각들을 하나로 합치기
combined = np.concatenate(audio_chunks)
sf.write("combined_output.wav", combined, 12000)
```

### 4.2 배치 처리

여러 파일을 한 번에 처리:

```bash
#!/bin/bash
# batch_tts.sh

for text_file in texts/*.txt; do
    filename=$(basename "$text_file" .txt)
    python main_qwen_tts.py \
        --text "$(cat $text_file)" \
        --mode clone \
        --ref_audio voice_sample.wav \
        --ref_text "참조 텍스트" \
        --output "outputs/${filename}.wav"
done
```

## 5. 지원 언어

Qwen3-TTS는 다음 10개 언어를 지원합니다:

- 한국어 (Korean)
- 영어 (English)
- 중국어 (Chinese)
- 일본어 (Japanese)
- 독일어 (German)
- 프랑스어 (French)
- 러시아어 (Russian)
- 포르투갈어 (Portuguese)
- 스페인어 (Spanish)
- 이탈리아어 (Italian)

다국어 텍스트 예시:
```bash
python main_qwen_tts.py \
  --text "Hello, 안녕하세요, こんにちは, 你好" \
  --mode basic \
  --output multilingual.wav
```

## 6. 성능 최적화

### 6.1 GPU 확인

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### 6.2 Flash Attention 설치

2-3배 속도 향상을 위해 Flash Attention 설치 (선택):

```bash
pip install flash-attn --no-build-isolation
```

### 6.3 모델 크기 선택

메모리가 제한적인 경우 0.6B 모델 사용:

```python
# loader_qwen_tts.py에서 모델 이름 변경
model_names = {
    "custom_voice": "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",  # 더 가벼움
    ...
}
```

## 7. 문제 해결

### 7.1 GPU 메모리 부족

```bash
# CPU 모드로 강제 실행
CUDA_VISIBLE_DEVICES="" python main_qwen_tts.py --text "테스트" --mode basic
```

### 7.2 음성 품질이 낮음

**Voice Cloning 품질 개선:**
- 더 긴 참조 음성 사용 (3초 → 5초)
- 배경 소음 제거
- 더 명확한 발음
- 참조 텍스트를 정확히 입력

**Voice Design 품질 개선:**
- 더 구체적인 설명 사용
- 다양한 설명 시도

### 7.3 설치 오류

```bash
# 패키지 재설치
conda activate qwen-tts
pip uninstall qwen-tts transformers -y
pip install qwen-tts transformers --upgrade
```

## 8. 모델 정보

### 사용 가능한 모델

| 모델 | 크기 | 용도 | 메모리 |
|------|------|------|--------|
| Qwen3-TTS-12Hz-1.7B-CustomVoice | 1.7B | Voice Cloning | ~6GB |
| Qwen3-TTS-12Hz-1.7B-VoiceDesign | 1.7B | Voice Design | ~6GB |
| Qwen3-TTS-12Hz-1.7B-Base | 1.7B | 기본 합성 | ~6GB |
| Qwen3-TTS-12Hz-0.6B-CustomVoice | 0.6B | Voice Cloning (경량) | ~3GB |
| Qwen3-TTS-12Hz-0.6B-Base | 0.6B | 기본 합성 (경량) | ~3GB |

### 샘플링 레이트
- 12kHz (고품질)
- 25kHz 모델도 출시 예정

## 9. 참고 자료

- **GitHub**: [QwenLM/Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS)
- **HuggingFace**: [Qwen3-TTS Collection](https://huggingface.co/collections/Qwen/qwen3-tts)
- **공식 블로그**: [Qwen3-TTS 소개](https://qwen.ai/blog?id=qwen3tts-0115)
- **Demo**: [HuggingFace Space](https://huggingface.co/spaces/Qwen/Qwen3-TTS)

## 10. 라이선스

Qwen3-TTS는 오픈소스 라이선스로 제공됩니다.
상업적 사용 전에 [공식 라이선스](https://github.com/QwenLM/Qwen3-TTS/blob/main/LICENSE)를 확인하세요.

---

## 빠른 시작 체크리스트

- [ ] Conda 환경 생성 (`bash setup_qwen_tts_env.sh`)
- [ ] 패키지 설치 (`bash install_qwen_tts.sh`)
- [ ] GPU 확인
- [ ] 기본 TTS 테스트
- [ ] 음성 샘플 녹음 (3초 이상)
- [ ] Voice Cloning 테스트
- [ ] OCR + TTS 통합 사용

문제가 있으면 `--verbose` 옵션으로 상세 로그를 확인하세요!
