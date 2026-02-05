#!/bin/bash
# CPU 안정 버전 설치 (WSL 최적화)
set -e

echo "========================================="
echo "CPU 안정 버전 설치 (WSL 최적화)"
echo "========================================="
echo ""

# 환경 확인
if [[ "$CONDA_DEFAULT_ENV" != "ocr-gpu" ]]; then
    echo "❌ 오류: ocr-gpu 환경이 활성화되지 않았습니다."
    echo "다음 명령어를 먼저 실행하세요:"
    echo "  conda activate ocr-gpu"
    exit 1
fi

# 기존 제거
echo "Step 1: 기존 PaddlePaddle 제거"
pip uninstall -y paddlepaddle paddlepaddle-gpu 2>/dev/null || true
echo ""

# CPU 버전 설치
echo "Step 2: PaddlePaddle CPU 버전 설치"
pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
echo ""

# numpy 맞추기
echo "Step 3: numpy 버전 맞추기"
pip install "numpy<2.0,>=1.19" --force-reinstall
echo ""

# 나머지 패키지
echo "Step 4: 의존성 설치"
pip install paddleocr==2.7.3
pip install pymupdf==1.23.8
pip install Pillow==10.2.0
pip install pandas openpyxl python-pptx python-docx
pip install langchain langchain-community langchain-ollama langchain-text-splitters
echo ""

# 확인
echo "Step 5: 설치 확인"
python -c "import paddle; print('PaddlePaddle 버전:', paddle.__version__); print('설치 완료!')"
echo ""

# 환경 변수
echo "Step 6: 환경 변수 설정"
cat > .env.cpu << 'EOF'
export DOC_OCR_FORCE_CPU=1
export DOC_OCR_DISABLE_MKLDNN=1
EOF
echo ""

echo "========================================="
echo "✓ CPU 버전 설치 완료!"
echo "========================================="
echo ""
echo "실행 방법:"
echo "  source .env.cpu"
echo "  python main_gpu.py ./pdfs/your_file.png"
echo ""
echo "또는:"
echo "  DOC_OCR_FORCE_CPU=1 python main_gpu.py ./pdfs/your_file.png"
