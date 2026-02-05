#!/bin/bash
# Qwen3-VL OCR 패키지 설치 (A100 최적화)
set -e

echo "========================================="
echo "Qwen3-VL OCR 패키지 설치 (A100 최적화)"
echo "========================================="
echo ""

if [[ "$CONDA_DEFAULT_ENV" != "qwen-ocr" ]]; then
    echo "❌ qwen-ocr 환경이 활성화되지 않았습니다."
    echo "conda activate qwen-ocr"
    exit 1
fi

# PyTorch GPU 설치 (CUDA 12.1)
echo "Step 1: PyTorch GPU 설치 (CUDA 12.1)"
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
echo ""

# Transformers 및 기본 패키지 (최신 버전)
echo "Step 2: Transformers 및 기본 패키지 설치"
pip install transformers>=4.50.0 accelerate sentencepiece protobuf tiktoken
echo ""

# Qwen3-VL 관련 패키지
echo "Step 3: Qwen3-VL 관련 패키지 설치"
pip install einops qwen-vl-utils
echo ""

# Flash Attention 2 (A100 최적화, 필수)
echo "Step 4: Flash Attention 2 설치 (A100 필수)"
pip install flash-attn --no-build-isolation
echo ""

# BitsAndBytes (양자화 지원)
echo "Step 5: BitsAndBytes 설치 (8bit/4bit 양자화 지원)"
pip install bitsandbytes
echo ""

# 이미지 처리
echo "Step 6: 이미지 처리 라이브러리 설치"
pip install Pillow pymupdf pdf2image
echo ""

# 기타 의존성
echo "Step 7: 기타 패키지 설치"
pip install pandas openpyxl python-pptx python-docx tqdm
echo ""

# GPU 확인
echo "Step 8: GPU 확인"
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'); print('GPU Memory:', round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 1), 'GB' if torch.cuda.is_available() else '')"
echo ""

echo "========================================="
echo "✓ 설치 완료!"
echo "========================================="
echo ""
echo "실행 방법:"
echo "  python main_qwen.py ./pdfs/your_file.pdf"
echo ""
echo "A100 최적화 모델 정보:"
echo "  - 기본: Qwen3-VL-8B (빠르고 고품질)"
echo "  - 최고 성능: loader_qwen.py에서 Qwen3-VL-32B-Instruct로 변경 가능"
echo "  - Flash Attention 2: 2-3배 속도 향상"
echo "  - A100 40GB/80GB: 최대 32B 모델까지 지원"
echo ""
echo "성능 최적화:"
echo "  - Flash Attention 2 활성화 (자동)"
echo "  - Mixed Precision (bfloat16) 사용"
echo "  - Tensor Parallelism 지원 (다중 GPU)"
