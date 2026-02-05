#!/bin/bash
# Qwen3-TTS 패키지 설치
set -e

echo "========================================="
echo "Qwen3-TTS 패키지 설치"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "qwen-tts" ]]; then
    echo "❌ qwen-tts 환경이 활성화되지 않았습니다."
    echo "conda activate qwen-tts"
    exit 1
fi

# PyTorch GPU 설치 (CUDA 12.1)
echo "Step 1: PyTorch GPU 설치"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Transformers 및 기본 패키지
echo "Step 2: Transformers 및 기본 패키지 설치"
pip install transformers accelerate sentencepiece protobuf tiktoken
echo ""

# Qwen3-TTS 전용 패키지
echo "Step 3: Qwen3-TTS 패키지 설치"
pip install qwen-tts einops
echo ""

# Flash Attention (선택, 2-3배 속도 향상)
echo "Step 4: Flash Attention 설치 (선택사항, 성능 향상)"
pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention 설치 실패 (선택사항이므로 계속 진행)"
echo ""

# 오디오 처리 라이브러리
echo "Step 5: 오디오 처리 라이브러리 설치"
pip install soundfile librosa scipy
echo ""

# 기타 의존성
echo "Step 6: 기타 패키지 설치"
pip install tqdm numpy pandas
echo ""

# GPU 확인
echo "Step 7: GPU 확인"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
echo ""

echo "========================================="
echo "✓ 설치 완료!"
echo "========================================="
echo ""
echo "실행 방법:"
echo "  # 텍스트에서 음성 생성 (Voice Cloning)"
echo "  python main_qwen_tts.py --text \"안녕하세요, 반갑습니다.\" --ref_audio voice_sample.wav --ref_text \"음성 샘플의 원본 텍스트\""
echo ""
echo "  # Voice Design (텍스트로 음성 특성 설명)"
echo "  python main_qwen_tts.py --text \"안녕하세요\" --voice_design \"젊은 여성의 밝고 활기찬 목소리\""
echo ""
