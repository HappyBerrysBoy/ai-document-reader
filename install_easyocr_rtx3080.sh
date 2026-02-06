#!/bin/bash
# EasyOCR 설치 (RTX 3080)
set -e

echo "========================================="
echo "EasyOCR 설치 (RTX 3080)"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "easyocr" ]]; then
    echo "❌ easyocr 환경이 활성화되지 않았습니다."
    echo "conda activate easyocr"
    exit 1
fi

# PyTorch GPU 설치 (CUDA 11.8)
echo "Step 1: PyTorch GPU 설치 (CUDA 11.8)"
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu118
echo ""

# EasyOCR 설치
echo "Step 2: EasyOCR 설치"
pip install easyocr
echo ""

# PDF 처리
echo "Step 3: PDF 처리 라이브러리 설치"
pip install pymupdf pdf2image pillow
echo ""

# 기타 의존성
echo "Step 4: 기타 패키지 설치"
pip install pandas openpyxl python-pptx python-docx tqdm
echo ""

# GPU 확인
echo "Step 5: GPU 확인"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)"
echo ""

echo "========================================="
echo "✓ 설치 완료!"
echo "========================================="
echo ""
echo "EasyOCR 특징:"
echo "  - GPU 가속 지원"
echo "  - 80개 이상 언어 지원 (한글, 영어 포함)"
echo "  - 자동 모델 다운로드"
echo ""
echo "실행 방법:"
echo "  python main_easyocr_rtx3080.py ./pdfs/your_file.pdf"
echo ""
