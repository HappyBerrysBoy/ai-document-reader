#!/bin/bash
# Qwen3-TTS 전용 Conda 환경 생성

echo "========================================="
echo "Qwen3-TTS Conda 환경 생성"
echo "========================================="
echo ""
echo "환경 이름: qwen-tts"
echo "Python 버전: 3.11"
echo ""

conda create -n qwen-tts python=3.11 -y

echo ""
echo "========================================="
echo "✓ Conda 환경 생성 완료!"
echo "========================================="
echo ""
echo "다음 단계:"
echo "  1. conda activate qwen-tts"
echo "  2. bash install_qwen_tts.sh"
echo ""
