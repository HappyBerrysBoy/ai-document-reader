# GPU별 최적화 가이드

RTX 3080과 A100 GPU별로 최적화된 설치 및 실행 방법을 안내합니다.

## 빠른 선택 가이드

| GPU 모델 | 메모리 | 권장 설정 | 설치 스크립트 |
|---------|--------|---------|-------------|
| **RTX 3080** | 10GB | Qwen3-VL-4B / DeepSeek OCR | `install_qwen.sh` / `install_deepseek.sh` |
| **A100** | 40GB/80GB | Qwen3-VL-8B/32B / DeepSeek OCR | `install_qwen_a100.sh` / `install_deepseek_a100.sh` |

---

## RTX 3080 (10GB) 설정

### Qwen3-VL (RTX 3080)

```bash
# 1. Conda 환경 생성 (공통)
bash setup_qwen_env.sh
conda activate qwen-ocr

# 2. RTX 3080 최적화 패키지 설치
bash install_qwen.sh

# 3. 실행
python main_qwen.py document.pdf -o output.txt
```

**사용 모델**: Qwen3-VL-4B (기본)
**메모리 사용**: ~4GB
**속도**: 빠름

### DeepSeek OCR (RTX 3080)

```bash
# 1. Conda 환경 생성 (공통)
bash setup_deepseek_env.sh
conda activate deepseek-ocr

# 2. RTX 3080 최적화 패키지 설치
bash install_deepseek.sh

# 3. 실행
python main_deepseek.py document.pdf -o output.txt
```

**사용 모델**: DeepSeek-OCR
**메모리 사용**: ~6GB
**속도**: 가장 빠름 (OCR 특화)

---

## A100 (40GB/80GB) 설정

### Qwen3-VL (A100)

```bash
# 1. Conda 환경 생성 (RTX 3080과 동일)
bash setup_qwen_env.sh
conda activate qwen-ocr

# 2. A100 최적화 패키지 설치
bash install_qwen_a100.sh

# 3. main_qwen.py 수정
# 파일 상단의 import 문을 수정:
```

**main_qwen.py 수정:**
```python
# 기존 (RTX 3080)
from loader_qwen import load_document

# 변경 (A100)
from loader_qwen_a100 import load_document
```

```bash
# 4. 실행
python main_qwen.py document.pdf -o output.txt
```

**사용 모델**: Qwen3-VL-8B (기본), 32B까지 가능
**메모리 사용**: ~7GB (8B), ~24GB (32B)
**속도**: 매우 빠름 + 고품질
**최적화**: Flash Attention 2, bfloat16, 고해상도 (DPI 200)

### DeepSeek OCR (A100)

```bash
# 1. Conda 환경 생성 (RTX 3080과 동일)
bash setup_deepseek_env.sh
conda activate deepseek-ocr

# 2. A100 최적화 패키지 설치
bash install_deepseek_a100.sh

# 3. main_deepseek.py 수정
# 파일 상단의 import 문을 수정:
```

**main_deepseek.py 수정:**
```python
# 기존 (RTX 3080)
from loader_deepseek import load_document

# 변경 (A100)
from loader_deepseek_a100 import load_document
```

```bash
# 4. 실행
python main_deepseek.py document.pdf -o output.txt
```

**사용 모델**: DeepSeek-OCR
**메모리 사용**: ~6GB
**속도**: 최고속 + 고품질
**최적화**: Flash Attention 2, bfloat16, 고해상도 (DPI 200)

---

## 주요 차이점

### RTX 3080 버전

**최적화 목표**: 메모리 효율 + 속도

| 항목 | 설정 |
|------|------|
| 모델 크기 | Qwen3-VL-4B |
| 이미지 해상도 | 1280px (Qwen), 1024px (DeepSeek) |
| PDF DPI | 150 |
| 최대 토큰 | 2048 |
| Flash Attention | 선택사항 |
| 데이터 타입 | bfloat16 |

**29페이지 PDF 예상 시간**: 몇 분

### A100 버전

**최적화 목표**: 최고 속도 + 최고 품질

| 항목 | 설정 |
|------|------|
| 모델 크기 | Qwen3-VL-8B (32B 가능) |
| 이미지 해상도 | 2048px (Qwen), 1536px (DeepSeek) |
| PDF DPI | 200 |
| 최대 토큰 | 4096 |
| Flash Attention | 필수 |
| 데이터 타입 | bfloat16 (최적화) |

**29페이지 PDF 예상 시간**: 훨씬 빠름 (RTX 3080 대비 2-3배)

---

## 파일 구조

```
document-ocr/
├── setup_qwen_env.sh           # Qwen Conda 환경 생성 (공통)
├── setup_deepseek_env.sh       # DeepSeek Conda 환경 생성 (공통)
│
├── install_qwen.sh             # RTX 3080용 Qwen 패키지 설치
├── install_qwen_a100.sh        # A100용 Qwen 패키지 설치 (Flash Attention 2 필수)
│
├── install_deepseek.sh         # RTX 3080용 DeepSeek 패키지 설치
├── install_deepseek_a100.sh    # A100용 DeepSeek 패키지 설치 (Flash Attention 2 필수)
│
├── loader_qwen.py              # RTX 3080용 Qwen 로더
├── loader_qwen_a100.py         # A100용 Qwen 로더 (고해상도, Flash Attention 2)
│
├── loader_deepseek.py          # RTX 3080용 DeepSeek 로더
├── loader_deepseek_a100.py     # A100용 DeepSeek 로더 (고해상도, Flash Attention 2)
│
├── main_qwen.py                # Qwen 실행 스크립트 (import만 변경)
└── main_deepseek.py            # DeepSeek 실행 스크립트 (import만 변경)
```

---

## 모델 변경 방법

### RTX 3080: 더 높은 품질 원하면

**loader_qwen.py** (28번째 줄):
```python
# 기본 (빠름)
model_name = "Qwen/Qwen3-VL-4B-Instruct"

# 변경 (느리지만 고품질)
model_name = "Qwen/Qwen3-VL-8B-Instruct"
```

### A100: 최고 품질 원하면

**loader_qwen_a100.py** (26번째 줄):
```python
# 기본 (빠르고 고품질)
model_name = "Qwen/Qwen3-VL-8B-Instruct"

# 변경 (최고 품질, A100 40GB 이상 권장)
model_name = "Qwen/Qwen3-VL-32B-Instruct"
```

---

## GPU 확인

현재 GPU 확인:
```bash
python -c "import torch; print(torch.cuda.get_device_name(0)); print(f'{torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')"
```

**출력 예시:**
- RTX 3080: `NVIDIA GeForce RTX 3080 / 10.0 GB`
- A100: `NVIDIA A100-SXM4-40GB / 40.0 GB`

---

## 문제 해결

### RTX 3080에서 메모리 부족

1. **Qwen3-VL-4B 사용** (기본 설정)
2. 또는 **DeepSeek OCR 사용** (더 가벼움)

### A100에서 Flash Attention 2 설치 실패

```bash
# 재설치 시도
pip install flash-attn --no-build-isolation --force-reinstall

# 실패 시 Flash Attention 없이 실행 가능 (자동 fallback)
```

### A100인데 속도가 느림

1. **Flash Attention 2 설치 확인**
2. **A100 전용 install script 사용 확인**
3. **A100 전용 loader 파일 사용 확인** (import 문 확인)

---

## 추천 구성

| 목적 | GPU | 모델 | 설정 |
|-----|-----|------|-----|
| **빠른 속도** | RTX 3080 | DeepSeek OCR | 기본 설정 |
| **균형** | RTX 3080 | Qwen3-VL-4B | 기본 설정 |
| **고품질 + 빠름** | A100 | Qwen3-VL-8B | A100 전용 설정 |
| **최고 품질** | A100 | Qwen3-VL-32B | A100 전용 설정 + 모델 변경 |

---

## 요약

1. **Conda 환경**: RTX 3080과 A100 **동일** (`setup_*_env.sh` 공통 사용)
2. **패키지 설치**: GPU별 **다름** (`install_*_a100.sh` vs `install_*.sh`)
3. **Loader 파일**: GPU별 **다름** (`loader_*_a100.py` vs `loader_*.py`)
4. **Main 스크립트**: **동일** (import 문만 수정)

GPU에 맞는 설치 스크립트와 loader 파일을 사용하는 것이 핵심입니다!
