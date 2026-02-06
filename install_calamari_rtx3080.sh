#!/bin/bash
# Calamari OCR 설치 (RTX 3080)
set -e

echo "========================================="
echo "Calamari OCR 설치 (RTX 3080)"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "calamari-ocr" ]]; then
    echo "❌ calamari-ocr 환경이 활성화되지 않았습니다."
    echo "conda activate calamari-ocr"
    exit 1
fi

# TensorFlow GPU 설치
echo "Step 1: TensorFlow GPU 설치"
pip install tensorflow[and-cuda]
echo ""

# Calamari OCR 설치
echo "Step 2: Calamari OCR 설치"
pip install calamari-ocr
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
python -c "import tensorflow as tf; print('GPU available:', len(tf.config.list_physical_devices('GPU')) > 0); print('GPUs:', tf.config.list_physical_devices('GPU'))"
echo ""

echo "========================================="
echo "✓ 설치 완료!"
echo "========================================="
echo ""
echo "Calamari OCR 특징:"
echo "  - 고품질 OCR 엔진"
echo "  - 역사적 문서/필기체 특화"
echo "  - TensorFlow 기반 GPU 가속"
echo "  - 사용자 정의 모델 학습 가능"
echo ""
echo "실행 방법:"
echo "  python main_calamari_rtx3080.py ./pdfs/your_file.pdf"
echo ""
