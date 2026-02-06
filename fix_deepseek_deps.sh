#!/bin/bash
# DeepSeek OCR 누락된 의존성 설치 (빠른 수정)

echo "========================================="
echo "DeepSeek OCR 누락 패키지 설치"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "deepseek-ocr" ]]; then
    echo "❌ deepseek-ocr 환경이 활성화되지 않았습니다."
    echo "conda activate deepseek-ocr"
    exit 1
fi

echo "현재 환경: $CONDA_DEFAULT_ENV"
echo ""

# 누락된 패키지 설치
echo "Step 1: 누락된 패키지 설치 중..."
pip install easydict ninja

# Transformers 버전 다운그레이드 (DeepSeek OCR 호환성)
echo ""
echo "Step 2: Transformers 라이브러리 버전 조정 중..."
echo "현재 최신 transformers 버전은 DeepSeek OCR와 호환되지 않습니다."
echo "호환 가능한 버전(4.37.0~4.45.x)으로 다운그레이드합니다..."
pip install "transformers>=4.37.0,<4.46.0" --force-reinstall

echo ""
echo "========================================="
echo "✓ 설치 완료!"
echo "========================================="
echo ""
echo "이제 다음 명령어로 OCR을 실행하세요:"
echo "  python main_deepseek_a100.py ./pdfs/your_file.pdf"
echo ""

# Flash Attention 재시도 (선택사항)
echo "Flash Attention 2 설치를 시도하시겠습니까? (y/N)"
read -r response
if [[ "$response" =~ ^[Yy]$ ]]; then
    echo "Flash Attention 2 설치 중..."
    pip install flash-attn --no-build-isolation || echo "⚠️  실패했지만 계속 진행 가능합니다."
fi

echo ""
echo "완료!"
