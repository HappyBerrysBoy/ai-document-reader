#!/bin/bash
# Flash Attention 2 대체 설치 방법 (사전 빌드된 wheel 사용)
set -e

echo "========================================="
echo "Flash Attention 2 설치 (대체 방법)"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "deepseek-ocr" ]] && [[ "$CONDA_DEFAULT_ENV" != "qwen-ocr" ]]; then
    echo "❌ deepseek-ocr 또는 qwen-ocr 환경이 활성화되지 않았습니다."
    echo "conda activate deepseek-ocr  또는  conda activate qwen-ocr"
    exit 1
fi

echo "현재 환경: $CONDA_DEFAULT_ENV"
echo ""

# PyTorch 및 CUDA 버전 확인
echo "Step 1: PyTorch 및 CUDA 버전 확인"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
echo ""

# 방법 1: 사전 빌드된 wheel 다운로드 (PyTorch 2.5.1 + CUDA 12.1 기준)
echo "Step 2: 사전 빌드된 Flash Attention wheel 다운로드 시도"
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu121torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl 2>/dev/null && {
    echo "✓ Flash Attention 2 설치 성공 (사전 빌드된 wheel)"
    exit 0
}

echo "⚠️  사전 빌드된 wheel 다운로드 실패"
echo ""

# 방법 2: 최신 릴리즈에서 다운로드
echo "Step 3: 최신 릴리즈에서 다운로드 시도"
pip install flash-attn --no-build-isolation --no-cache-dir 2>/dev/null && {
    echo "✓ Flash Attention 2 설치 성공"
    exit 0
}

echo "⚠️  설치 실패"
echo ""

# 방법 3: 소스 빌드 (충분한 메모리와 시간 필요)
echo "Step 4: 소스에서 빌드 시도 (시간이 오래 걸릴 수 있음)"
echo "빌드 환경 설정..."

# CUDA 경로 확인
if [ -d "/usr/local/cuda" ]; then
    export PATH=/usr/local/cuda/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
    echo "CUDA 경로 설정 완료: /usr/local/cuda"
elif [ -d "/usr/local/cuda-12.1" ]; then
    export PATH=/usr/local/cuda-12.1/bin:$PATH
    export LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64:$LD_LIBRARY_PATH
    echo "CUDA 경로 설정 완료: /usr/local/cuda-12.1"
else
    echo "⚠️  CUDA 경로를 찾을 수 없습니다."
fi

# 컴파일러 확인
which nvcc >/dev/null 2>&1 && {
    echo "nvcc 버전: $(nvcc --version | grep release)"
} || {
    echo "⚠️  nvcc를 찾을 수 없습니다. CUDA Toolkit이 설치되어 있는지 확인하세요."
}

echo ""
echo "소스 빌드 시작 (5-10분 소요 예상)..."
MAX_JOBS=4 pip install flash-attn --no-build-isolation 2>&1 | tail -20 && {
    echo "✓ Flash Attention 2 설치 성공 (소스 빌드)"
    exit 0
}

echo ""
echo "========================================="
echo "❌ Flash Attention 2 설치 실패"
echo "========================================="
echo ""
echo "Flash Attention 없이도 모델은 정상 작동합니다."
echo "단, A100에서 최대 성능(2-3배 속도)을 내려면 Flash Attention이 필요합니다."
echo ""
echo "해결 방법:"
echo "1. CUDA Toolkit 설치 확인: nvcc --version"
echo "2. 디스크 공간 확인: df -h"
echo "3. 메모리 확인: free -h"
echo "4. PyTorch CUDA 버전 확인: python -c 'import torch; print(torch.version.cuda)'"
echo ""
echo "또는 Flash Attention 없이 계속 사용하셔도 됩니다."
echo ""
exit 1
