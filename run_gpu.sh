#!/bin/bash
# GPU 모드 실행 스크립트

# WSL CUDA 경로 설정
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# PaddlePaddle 최적화 설정
export DOC_OCR_DISABLE_MKLDNN=1
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True

# GPU 강제 사용 (CPU 폴백 비활성화)
export DOC_OCR_FORCE_GPU=1

# Python 실행
python "$@"
