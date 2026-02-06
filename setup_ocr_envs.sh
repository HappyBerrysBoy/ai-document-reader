#!/bin/bash
# OCR 전용 모델들 Conda 환경 생성

set -e

echo "========================================="
echo "OCR 전용 모델 Conda 환경 생성"
echo "========================================="
echo ""

# 1. Tesseract OCR 환경
echo "1. tesseract-ocr 환경 생성 중..."
conda create -n tesseract-ocr python=3.11 -y
echo "✓ tesseract-ocr 환경 생성 완료"
echo ""

# 2. EasyOCR 환경
echo "2. easyocr 환경 생성 중..."
conda create -n easyocr python=3.11 -y
echo "✓ easyocr 환경 생성 완료"
echo ""

# 3. docTR 환경
echo "3. doctr-ocr 환경 생성 중..."
conda create -n doctr-ocr python=3.11 -y
echo "✓ doctr-ocr 환경 생성 완료"
echo ""

# 4. Calamari OCR 환경
echo "4. calamari-ocr 환경 생성 중..."
conda create -n calamari-ocr python=3.11 -y
echo "✓ calamari-ocr 환경 생성 완료"
echo ""

echo "========================================="
echo "✓ 모든 환경 생성 완료!"
echo "========================================="
echo ""
echo "다음 단계:"
echo "  1. Tesseract: bash install_tesseract_a100.sh (또는 rtx3080)"
echo "  2. EasyOCR: bash install_easyocr_a100.sh (또는 rtx3080)"
echo "  3. docTR: bash install_doctr_a100.sh (또는 rtx3080)"
echo "  4. Calamari: bash install_calamari_a100.sh (또는 rtx3080)"
echo ""
