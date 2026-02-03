#!/bin/bash
# GPU 서버용 PaddlePaddle 설치 스크립트

set -e

echo "=== PaddlePaddle GPU 설치 스크립트 ==="
echo ""

# CUDA 버전 확인
echo "CUDA 버전 확인 중..."
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi를 찾을 수 없습니다. NVIDIA 드라이버가 설치되어 있는지 확인하세요."
    exit 1
fi

CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}' | cut -d'.' -f1,2)
echo "✓ 감지된 CUDA 버전: $CUDA_VERSION"
echo ""

# Python 버전 확인
PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}' | cut -d'.' -f1,2)
echo "✓ Python 버전: $PYTHON_VERSION"
echo ""

# Python 3.13 경고
if [[ "$PYTHON_VERSION" == "3.13" ]]; then
    echo "⚠️  경고: Python 3.13은 PaddlePaddle에서 공식 지원하지 않습니다."
    echo "   Python 3.10-3.12 환경을 만드는 것을 권장합니다:"
    echo "   conda create -n doc-ocr python=3.12"
    echo "   conda activate doc-ocr"
    echo ""
    read -p "계속하시겠습니까? (y/N): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# 기존 PaddlePaddle 제거
echo "기존 PaddlePaddle 제거 중..."
pip uninstall -y paddlepaddle paddlepaddle-gpu 2>/dev/null || true
echo ""

# CUDA 버전에 따른 설치
echo "PaddlePaddle GPU 설치 중..."
if [[ "$CUDA_VERSION" == "11.7" ]]; then
    echo "CUDA 11.7용 PaddlePaddle 설치..."
    python -m pip install paddlepaddle-gpu==2.6.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
elif [[ "$CUDA_VERSION" == "11.8" ]]; then
    echo "CUDA 11.8용 PaddlePaddle 설치..."
    python -m pip install paddlepaddle-gpu==2.6.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
elif [[ "$CUDA_VERSION" == "12.0" ]]; then
    echo "CUDA 12.0용 PaddlePaddle 설치..."
    python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
elif [[ "$CUDA_VERSION" =~ ^12\.[1-9]$ ]] || [[ "$CUDA_VERSION" =~ ^13\. ]]; then
    echo "CUDA 12.3+/13.0용 PaddlePaddle 3.0 베타 설치..."
    python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
else
    echo "❌ 지원되지 않는 CUDA 버전: $CUDA_VERSION"
    echo "   지원 버전: 11.7, 11.8, 12.0, 12.x, 13.x"
    echo ""
    echo "수동 설치를 시도하세요:"
    echo "  CUDA 11.8: python -m pip install paddlepaddle-gpu==2.6.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html"
    exit 1
fi

echo ""
echo "=== 설치 확인 ==="
python -c "import paddle; print('✓ PaddlePaddle 버전:', paddle.__version__); print('✓ CUDA 지원:', paddle.device.is_compiled_with_cuda()); print('✓ GPU 개수:', paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0)"

echo ""
echo "=== 나머지 의존성 설치 ==="
pip install -r requirements.txt

echo ""
echo "✅ 설치 완료!"
echo ""
echo "테스트 실행:"
echo "  python main.py ./pdfs/your_file.pdf"
