#!/bin/bash
# Docker GPU 실행 스크립트

# Docker 이미지 빌드
build_image() {
    echo "Docker 이미지 빌드 중..."
    docker build -t ocr-gpu:latest -f Dockerfile.gpu .
}

# Docker 실행
run_ocr() {
    if [ $# -eq 0 ]; then
        echo "사용법: $0 <파일경로>"
        echo "예시: $0 ./pdfs/document.pdf"
        exit 1
    fi

    FILE_PATH=$(realpath "$1")
    FILE_DIR=$(dirname "$FILE_PATH")
    FILE_NAME=$(basename "$FILE_PATH")

    docker run --gpus all --rm \
        -v "$FILE_DIR:/data" \
        ocr-gpu:latest \
        python main_gpu.py "/data/$FILE_NAME"
}

# 빌드가 필요한 경우
if ! docker images | grep -q "ocr-gpu"; then
    build_image
fi

# OCR 실행
run_ocr "$@"
