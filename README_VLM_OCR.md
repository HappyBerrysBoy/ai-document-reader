# Vision Language Models OCR 가이드

DeepSeek VL2와 Qwen2-VL을 사용한 GPU 가속 OCR 시스템 (한글/영어 최적화)

## 지원 모델

### 1. DeepSeek VL2
- 모델: `deepseek-ai/deepseek-vl2`
- 특징: 고정밀 텍스트 인식
- GPU 메모리: ~8GB
- 환경: `deepseek-ocr`

### 2. Qwen2-VL-7B
- 모델: `Qwen/Qwen2-VL-7B-Instruct`
- 특징: 한글/영어 멀티언어 우수
- GPU 메모리: ~14GB
- 환경: `qwen-ocr`

## 빠른 시작

### DeepSeek VL2 설치 및 사용

```bash
# 1. 환경 생성
./setup_deepseek_env.sh
conda activate deepseek-ocr

# 2. 패키지 설치
./install_deepseek.sh

# 3. OCR 실행
python main_deepseek.py ./pdfs/document.pdf
python main_deepseek.py ./images/scan.png -o result.txt
```

### Qwen2-VL 설치 및 사용

```bash
# 1. 환경 생성
./setup_qwen_env.sh
conda activate qwen-ocr

# 2. 패키지 설치
./install_qwen.sh

# 3. OCR 실행
python main_qwen.py ./pdfs/document.pdf
python main_qwen.py ./images/scan.png -o result.txt
```

## 상세 설치 가이드

### DeepSeek VL2

#### Step 1: 환경 생성
```bash
chmod +x setup_deepseek_env.sh
./setup_deepseek_env.sh
```

#### Step 2: 환경 활성화
```bash
conda activate deepseek-ocr
```

#### Step 3: 패키지 설치
```bash
chmod +x install_deepseek.sh
./install_deepseek.sh
```

설치 내용:
- PyTorch 2.x (CUDA 12.1)
- Transformers
- DeepSeek VL2 모델 (자동 다운로드)

#### Step 4: GPU 확인
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
```

### Qwen2-VL

#### Step 1-3: DeepSeek와 동일
```bash
./setup_qwen_env.sh
conda activate qwen-ocr
./install_qwen.sh
```

## 사용 예시

### 기본 사용

```bash
# DeepSeek VL2
conda activate deepseek-ocr
python main_deepseek.py ./documents/report.pdf

# Qwen2-VL
conda activate qwen-ocr
python main_qwen.py ./documents/report.pdf
```

### 출력 파일 지정

```bash
# 결과를 특정 파일로 저장
python main_deepseek.py ./images/scan.png -o output.txt
python main_qwen.py ./images/scan.png -o output.txt
```

### 상세 로그

```bash
# 상세 로그 출력
python main_deepseek.py ./files/doc.pdf --verbose
python main_qwen.py ./files/doc.pdf --verbose
```

### 배치 처리

```bash
# 여러 파일 처리
for file in ./pdfs/*.pdf; do
    python main_qwen.py "$file"
done
```

## 지원 형식

- **이미지**: PNG, JPG, JPEG, BMP, GIF, WEBP, TIFF
- **문서**: PDF, DOCX, XLSX, PPTX

## 한글/영어 최적화

두 모델 모두 한글과 영어를 동시에 인식하도록 프롬프트가 최적화되어 있습니다:

```python
# 내장 프롬프트 (자동 적용)
"""이 이미지에서 모든 텍스트를 정확하게 추출해주세요. 한글과 영어가 포함되어 있을 수 있습니다.
Extract all text from this image accurately. The text contains Korean and English characters.
텍스트의 레이아웃을 그대로 유지하고, 추출된 텍스트만 반환하세요."""
```

## 성능 비교 (RTX 3080 기준)

| 모델 | 속도 (이미지당) | GPU 메모리 | 한글 정확도 | 영어 정확도 |
|------|----------------|-----------|------------|------------|
| DeepSeek VL2 | ~2-3초 | 8GB | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Qwen2-VL-7B | ~3-4초 | 14GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| PaddleOCR (참고) | ~0.5-1초 | 2GB | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |

## GPU 메모리 관리

### 메모리 부족 시

1. **이미지 크기 자동 조정**
   - DeepSeek: 최대 1024px
   - Qwen: 최대 1280px

2. **배치 크기 조정**
   ```python
   # loader_qwen.py 또는 loader_deepseek.py에서
   max_size = 960  # 줄이면 메모리 절약
   ```

3. **모델 양자화** (고급)
   ```python
   # 4-bit 양자화로 메모리 절약
   from transformers import BitsAndBytesConfig

   quantization_config = BitsAndBytesConfig(
       load_in_4bit=True,
       bnb_4bit_compute_dtype=torch.float16
   )
   ```

## GPU 모니터링

```bash
# GPU 사용률 실시간 확인
watch -n 1 nvidia-smi

# 또는
nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.used,memory.total --format=csv -l 1
```

## 문제 해결

### GPU가 인식되지 않음

```bash
# CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"

# CUDA 버전 확인
nvidia-smi
```

### Out of Memory 오류

1. 이미지 크기 줄이기
2. 배치 크기 줄이기
3. 모델 양자화 사용
4. GPU 메모리가 작다면 DeepSeek VL2 사용 (8GB)

### 모델 다운로드 실패

```bash
# HuggingFace CLI로 수동 다운로드
pip install huggingface-cli

# DeepSeek
huggingface-cli download deepseek-ai/deepseek-vl2

# Qwen
huggingface-cli download Qwen/Qwen2-VL-7B-Instruct
```

### WSL에서 느린 속도

WSL + GPU는 네이티브 Linux보다 느릴 수 있습니다:
- WSL에서 CUDA 드라이버 최신 버전 확인
- Windows NVIDIA 드라이버 업데이트
- 네이티브 Linux 사용 권장

## 파일 구조

```
document-ocr/
├── setup_deepseek_env.sh      # DeepSeek 환경 생성
├── install_deepseek.sh         # DeepSeek 패키지 설치
├── loader_deepseek.py          # DeepSeek OCR 로더
├── main_deepseek.py            # DeepSeek 메인 프로그램
├── setup_qwen_env.sh           # Qwen 환경 생성
├── install_qwen.sh             # Qwen 패키지 설치
├── loader_qwen.py              # Qwen OCR 로더
├── main_qwen.py                # Qwen 메인 프로그램
└── README_VLM_OCR.md           # 이 문서
```

## 추천 사용 시나리오

### DeepSeek VL2 추천
- GPU 메모리가 제한적 (8GB 이하)
- 빠른 처리 속도 필요
- 영어 문서가 많은 경우

### Qwen2-VL 추천
- GPU 메모리 여유 (14GB 이상)
- 한글 인식 정확도가 중요한 경우
- 복잡한 레이아웃의 문서

## 환경 전환

```bash
# DeepSeek → Qwen 전환
conda deactivate
conda activate qwen-ocr

# Qwen → DeepSeek 전환
conda deactivate
conda activate deepseek-ocr
```

## 라이선스 및 주의사항

- DeepSeek VL2: MIT License
- Qwen2-VL: Apache 2.0 License
- 상업적 사용 가능
- GPU 가속을 위해 CUDA 12.1+ 필요

## 추가 리소스

- [DeepSeek VL2 논문](https://github.com/deepseek-ai/DeepSeek-VL2)
- [Qwen2-VL 문서](https://github.com/QwenLM/Qwen2-VL)
- [Transformers 문서](https://huggingface.co/docs/transformers)
