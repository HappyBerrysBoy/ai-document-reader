#!/bin/bash
# Qwen3-VL OCR 패키지 설치
set -e

echo "========================================="
echo "Qwen3-VL OCR 패키지 설치 (RTX 3080 최적화)"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "qwen-ocr" ]]; then
    echo "❌ qwen-ocr 환경이 활성화되지 않았습니다."
    echo "conda activate qwen-ocr"
    exit 1
fi

# PyTorch GPU 설치
echo "Step 1: PyTorch GPU 설치"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Transformers 및 기본 패키지 (최신 버전)
echo "Step 2: Transformers 및 기본 패키지 설치"
pip install transformers>=4.50.0 accelerate sentencepiece protobuf tiktoken
echo ""

# Qwen3-VL 관련 패키지
echo "Step 3: Qwen3-VL 관련 패키지 설치"
pip install einops qwen-vl-utils
echo ""

# Flash Attention (선택, 성능 향상)
echo "Step 4: Flash Attention 설치 (선택사항, 성능 향상)"
pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention 설치 실패 (선택사항이므로 계속 진행)"
echo ""

# 이미지 처리
echo "Step 5: 이미지 처리 라이브러리 설치"
pip install Pillow pymupdf pdf2image
echo ""

# 기타 의존성
echo "Step 6: 기타 패키지 설치"
pip install pandas openpyxl python-pptx python-docx tqdm
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
echo "  python main_qwen.py ./pdfs/your_file.png"
echo ""
echo "모델 정보:"
echo "  - 기본: Qwen3-VL-4B (빠른 속도, ~4GB VRAM 사용)"
echo "  - 더 높은 품질: loader_qwen.py에서 Qwen3-VL-8B-Instruct로 변경 가능 (느림)"
echo ""
echo "성능 비교 (29페이지 PDF 기준):"
echo "  - Qwen3-VL-4B: 빠름 (권장)"
echo "  - Qwen3-VL-8B: 느림 (Qwen2-VL-7B보다 더 느림)"
echo "  - DeepSeek OCR: 가장 빠름 (OCR 특화)"
