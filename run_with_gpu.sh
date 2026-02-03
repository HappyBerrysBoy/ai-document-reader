#!/bin/bash
# GPU 환경에서 OCR 실행 스크립트

# CUDA 라이브러리 경로 설정
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda-13.0/lib64:/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

# PaddlePaddle 모델 체크 비활성화 (네트워크 느린 경우)
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# Python 실행
python "$@"
