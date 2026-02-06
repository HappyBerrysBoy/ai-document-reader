#!/bin/bash
# Tesseract OCR 설치 (A100 최적화)
set -e

echo "========================================="
echo "Tesseract OCR 설치 (A100)"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "tesseract-ocr" ]]; then
    echo "❌ tesseract-ocr 환경이 활성화되지 않았습니다."
    echo "conda activate tesseract-ocr"
    exit 1
fi

# PyTorch GPU 설치 (CUDA 12.1)
echo "Step 1: PyTorch GPU 설치 (CUDA 12.1)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Tesseract OCR 설치
echo "Step 2: Tesseract OCR 및 Python 바인딩 설치"
pip install pytesseract pillow
echo ""

# PDF 처리
echo "Step 3: PDF 처리 라이브러리 설치"
pip install pymupdf pdf2image
echo ""

# 기타 의존성
echo "Step 4: 기타 패키지 설치"
pip install pandas openpyxl python-pptx python-docx tqdm
echo ""

# GPU 확인
echo "Step 5: GPU 확인"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
echo ""

echo "========================================="
echo "✓ 설치 완료!"
echo "========================================="
echo ""
echo "⚠️  중요: 시스템에 Tesseract 바이너리를 설치해야 합니다:"
echo "  Ubuntu/Debian: sudo apt-get install tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng"
echo "  macOS: brew install tesseract tesseract-lang"
echo ""
echo "실행 방법:"
echo "  python main_tesseract_a100.py ./pdfs/your_file.pdf"
echo ""
