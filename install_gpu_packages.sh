#!/bin/bash
# GPU 패키지 설치 스크립트
set -e

echo "========================================="
echo "GPU 패키지 설치 시작"
echo "========================================="
echo ""

# 환경 확인
if [[ "$CONDA_DEFAULT_ENV" != "ocr-gpu" ]]; then
    echo "❌ 오류: ocr-gpu 환경이 활성화되지 않았습니다."
    echo "다음 명령어를 먼저 실행하세요:"
    echo "  conda activate ocr-gpu"
    exit 1
fi

echo "✓ 현재 환경: $CONDA_DEFAULT_ENV"
echo "✓ Python 버전: $(python --version)"
echo ""

# WSL CUDA 경로 설정
echo "Step 1: WSL CUDA 환경 변수 설정"
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
echo "✓ LD_LIBRARY_PATH 설정 완료"
echo ""

# 기존 PaddlePaddle 제거
echo "Step 2: 기존 PaddlePaddle 제거"
pip uninstall -y paddlepaddle paddlepaddle-gpu 2>/dev/null || true
echo "✓ 기존 패키지 제거 완료"
echo ""

# PaddlePaddle GPU 설치 (CUDA 12.0 - CUDA 13.0과 호환)
echo "Step 3: PaddlePaddle GPU 설치"
echo "CUDA 13.0은 하위 호환되므로 CUDA 12.0 버전 사용"
python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
echo "✓ PaddlePaddle GPU 설치 완료"
echo ""

# numpy 버전 호환성 맞추기
echo "Step 4: numpy 버전 맞추기 (ABI 호환성)"
pip install "numpy<2.0,>=1.19" --force-reinstall
echo "✓ numpy 설치 완료"
echo ""

# 나머지 패키지 설치
echo "Step 5: 의존성 패키지 설치"
pip install paddleocr==2.7.3
pip install pymupdf==1.23.8
pip install Pillow==10.2.0
pip install pandas openpyxl python-pptx python-docx
pip install langchain langchain-community langchain-ollama langchain-text-splitters
echo "✓ 의존성 패키지 설치 완료"
echo ""

# GPU 확인
echo "Step 6: GPU 작동 확인"
python -c "
import paddle
print('PaddlePaddle 버전:', paddle.__version__)
print('CUDA 지원:', paddle.device.is_compiled_with_cuda())
if paddle.device.is_compiled_with_cuda():
    print('GPU 개수:', paddle.device.cuda.device_count())
    if paddle.device.cuda.device_count() > 0:
        print('✓ GPU 정상 감지!')
    else:
        print('⚠️  GPU가 감지되지 않습니다.')
else:
    print('⚠️  CUDA가 지원되지 않습니다.')
"
echo ""

# 모델 캐시 삭제
echo "Step 7: 이전 모델 캐시 삭제"
rm -rf ~/.paddleocr/ 2>/dev/null || true
rm -rf ~/.paddlex/ 2>/dev/null || true
echo "✓ 캐시 삭제 완료"
echo ""

# 환경 변수 설정 파일 생성
echo "Step 8: 환경 변수 설정 파일 생성"
cat > .env.gpu << 'EOF'
# WSL GPU 환경 변수
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export DOC_OCR_DISABLE_MKLDNN=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
EOF
echo "✓ .env.gpu 파일 생성 완료"
echo ""

echo "========================================="
echo "설치 완료!"
echo "========================================="
echo ""
echo "실행 방법:"
echo "  1. 환경 변수 로드: source .env.gpu"
echo "  2. OCR 실행: python main.py ./pdfs/your_file.png"
echo ""
echo "또는 한 줄로 실행:"
echo "  ./run_gpu.sh main.py ./pdfs/your_file.png"
