#!/bin/bash
# CUDA 라이브러리 경로 문제 해결 스크립트

echo "=== CUDA 라이브러리 경로 설정 ==="
echo ""

# CUDA 설치 경로 찾기
CUDA_PATHS=(
    "/usr/local/cuda"
    "/usr/local/cuda-13.0"
    "/usr/local/cuda-12.0"
    "/usr/local/cuda-11.8"
    "/usr/lib/x86_64-linux-gnu"
)

FOUND_CUDA=""
for path in "${CUDA_PATHS[@]}"; do
    if [ -d "$path" ]; then
        if [ -f "$path/lib64/libcuda.so" ] || [ -f "$path/libcuda.so.1" ]; then
            FOUND_CUDA="$path"
            echo "✓ CUDA 발견: $path"
            break
        fi
    fi
done

if [ -z "$FOUND_CUDA" ]; then
    echo "⚠️  CUDA 경로를 자동으로 찾지 못했습니다."
    echo "   다음 명령어로 수동 확인:"
    echo "   find /usr -name 'libcuda.so*' 2>/dev/null"
    echo ""
else
    # LD_LIBRARY_PATH 설정
    if [ -d "$FOUND_CUDA/lib64" ]; then
        export LD_LIBRARY_PATH="$FOUND_CUDA/lib64:$LD_LIBRARY_PATH"
        echo "✓ LD_LIBRARY_PATH 설정: $FOUND_CUDA/lib64"
    elif [ -d "$FOUND_CUDA" ]; then
        export LD_LIBRARY_PATH="$FOUND_CUDA:$LD_LIBRARY_PATH"
        echo "✓ LD_LIBRARY_PATH 설정: $FOUND_CUDA"
    fi
fi

echo ""
echo "=== 환경 변수 설정 (현재 세션) ==="
echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\""
echo ""

# .bashrc에 영구 추가 (선택)
read -p "이 설정을 ~/.bashrc에 영구 저장하시겠습니까? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    if ! grep -q "LD_LIBRARY_PATH.*cuda" ~/.bashrc; then
        echo "" >> ~/.bashrc
        echo "# CUDA Library Path for PaddlePaddle" >> ~/.bashrc
        echo "export LD_LIBRARY_PATH=\"$LD_LIBRARY_PATH\"" >> ~/.bashrc
        echo "✓ ~/.bashrc에 저장되었습니다."
        echo "  다음 로그인부터 자동 적용됩니다."
    else
        echo "✓ ~/.bashrc에 이미 설정되어 있습니다."
    fi
fi

echo ""
echo "=== GPU 테스트 ==="
python -c "import paddle; print('CUDA compiled:', paddle.device.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0)"

echo ""
echo "완료! 이제 다음 명령어로 실행하세요:"
echo "  python main.py ./pdfs/your_file.png"
