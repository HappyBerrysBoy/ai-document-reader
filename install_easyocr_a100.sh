#!/bin/bash
# EasyOCR 설치 (A100 최적화)
set -e

echo "========================================="
echo "EasyOCR 설치 (A100 최적화)"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "easyocr" ]]; then
    echo "❌ easyocr 환경이 활성화되지 않았습니다."
    echo "conda activate easyocr"
    exit 1
fi

# PyTorch GPU 설치 (CUDA 12.1)
echo "Step 1: PyTorch GPU 설치 (CUDA 12.1)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB' if torch.cuda.is_available() else '')"
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
echo "  python main_easyocr_a100.py ./pdfs/your_file.pdf"
echo ""
