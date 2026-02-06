#!/bin/bash
# DeepSeek OCR 패키지 설치
set -e

echo "========================================="
echo "DeepSeek OCR 패키지 설치"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "deepseek-ocr" ]]; then
    echo "❌ deepseek-ocr 환경이 활성화되지 않았습니다."
    echo "conda activate deepseek-ocr"
    exit 1
fi

# PyTorch GPU 설치 (CUDA 11.8)
echo "Step 1: PyTorch GPU 설치"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
echo ""

# Transformers 및 기본 패키지
echo "Step 2: Transformers 및 DeepSeek OCR 의존성 설치"
pip install transformers accelerate sentencepiece protobuf
pip install timm einops
pip install addict matplotlib requests
echo ""

# Flash Attention (선택, GPU 성능 향상)
echo "Step 3: Flash Attention 설치 (선택사항, 시간 소요)"
pip install flash-attn --no-build-isolation || echo "⚠️  Flash Attention 설치 실패 (선택사항이므로 계속 진행)"
echo ""

# 이미지 처리
echo "Step 4: 이미지 처리 라이브러리 설치"
pip install Pillow pymupdf pdf2image
echo ""

# 기타 의존성
echo "Step 5: 기타 패키지 설치"
pip install pandas openpyxl python-pptx python-docx tqdm
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
echo "  python main_deepseek.py ./pdfs/your_file.png"
