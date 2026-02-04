#!/bin/bash
# GPU 환경 완전 새로 설정 - WSL Ubuntu 24.04 + CUDA 13.0 + RTX 3080
set -e

echo "========================================="
echo "GPU OCR 환경 설정 시작"
echo "WSL Ubuntu 24.04 + CUDA 13.0 + RTX 3080"
echo "========================================="
echo ""

# 1. Conda 환경 생성 (Python 3.11 - 안정성과 호환성 최고)
echo "Step 1: Conda 환경 생성 (Python 3.11)"
conda create -n ocr-gpu python=3.11 -y
echo "✓ Conda 환경 생성 완료"
echo ""

echo "Step 2: 환경 활성화"
echo "다음 명령어를 실행하세요:"
echo "  conda activate ocr-gpu"
echo ""
echo "활성화 후 다음 스크립트를 실행하세요:"
echo "  ./install_gpu_packages.sh"
