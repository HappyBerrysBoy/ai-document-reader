#!/bin/bash
# Qwen3-Omni OCR 패키지 설치
set -e

echo "========================================="
echo "Qwen3-Omni OCR 패키지 설치"
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

# Transformers 및 기본 패키지
echo "Step 2: Transformers 설치"
pip install transformers accelerate sentencepiece protobuf tiktoken
echo ""

# Qwen 전용 패키지
echo "Step 3: Qwen 관련 패키지 설치"
pip install qwen-vl-utils einops
echo ""

# 이미지 처리
echo "Step 4: 이미지 처리 라이브러리 설치"
pip install Pillow pymupdf pdf2image
echo ""

# 기타 의존성
echo "Step 5: 기타 패키지 설치"
pip install pandas openpyxl python-pptx python-docx
echo ""

# GPU 확인
echo "Step 6: GPU 확인"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
echo ""

echo "========================================="
echo "✓ 설치 완료!"
echo "========================================="
echo ""
echo "실행 방법:"
echo "  python main_qwen.py ./pdfs/your_file.png"
