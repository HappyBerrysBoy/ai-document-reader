#!/bin/bash
# Qwen3-Omni OCR 환경 설정
set -e

echo "========================================="
echo "Qwen3-Omni OCR 환경 설정"
echo "========================================="
echo ""

# Conda 환경 생성
echo "Step 1: Conda 환경 생성 (Python 3.11)"
conda create -n qwen-ocr python=3.11 -y
echo ""

echo "환경 생성 완료!"
echo ""
echo "다음 명령어로 활성화하세요:"
echo "  conda activate qwen-ocr"
echo ""
echo "활성화 후 실행:"
echo "  ./install_qwen.sh"
