#!/bin/bash
# DeepSeek OCR 패키지 설치 (A100 최적화)
set -e

echo "========================================="
echo "DeepSeek OCR 패키지 설치 (A100 최적화)"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "deepseek-ocr" ]]; then
    echo "❌ deepseek-ocr 환경이 활성화되지 않았습니다."
    echo "conda activate deepseek-ocr"
    exit 1
fi

# PyTorch GPU 설치 (CUDA 12.1)
echo "Step 1: PyTorch GPU 설치 (CUDA 12.1)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Transformers 및 기본 패키지
echo "Step 2: Transformers 및 DeepSeek OCR 의존성 설치"
pip install "transformers>=4.37.0,<4.46.0" accelerate sentencepiece protobuf
pip install timm einops
pip install addict easydict matplotlib requests
echo ""

# Flash Attention 2 (A100 최적화, 권장)
echo "Step 3: Flash Attention 2 설치 (A100 권장, 2-3배 속도 향상)"
# 방법 1: 사전 빌드된 wheel 다운로드 시도
pip install flash-attn --no-build-isolation 2>/dev/null || \
# 방법 2: 환경 변수 설정 후 재시도
TMPDIR=/tmp pip install flash-attn --no-build-isolation 2>/dev/null || \
# 방법 3: 실패해도 계속 진행 (Flash Attention 없이도 동작 가능)
echo "⚠️  Flash Attention 2 설치 실패 (선택사항이므로 계속 진행)"
echo ""

# BitsAndBytes (양자화 지원)
echo "Step 4: BitsAndBytes 설치 (8bit/4bit 양자화 지원)"
pip install bitsandbytes
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
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB' if torch.cuda.is_available() else '')"
echo ""

echo "========================================="
echo "✓ 설치 완료!"
echo "========================================="
echo ""
echo "실행 방법:"
echo "  python main_deepseek.py ./pdfs/your_file.pdf"
echo ""
echo "A100 최적화 모델 정보:"
echo "  - DeepSeek OCR: OCR 전용 모델, 매우 빠름"
echo "  - Flash Attention 2: 2-3배 속도 향상"
echo "  - A100 40GB/80GB: 최고 성능 발휘"
echo ""
echo "성능 최적화:"
echo "  - Flash Attention 2 활성화 (자동)"
echo "  - Mixed Precision (bfloat16) 사용"
echo "  - 배치 처리 최적화"
