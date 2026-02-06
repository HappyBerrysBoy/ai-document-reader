# OCR 모델 비교 및 사용 가이드

## 설치된 OCR 모델들

### 1. **Tesseract OCR**
- **특징**: 가장 오래되고 안정적인 오픈소스 OCR
- **장점**:
  - 빠른 속도
  - 100개 이상 언어 지원
  - 시스템 레벨 설치
- **단점**:
  - 정확도가 다른 딥러닝 모델보다 낮을 수 있음
  - 필기체 인식 약함
- **적합한 용도**: 깨끗한 인쇄물, 빠른 처리 필요 시

### 2. **EasyOCR** ⭐ 추천
- **특징**: 딥러닝 기반 OCR, 사용하기 쉬움
- **장점**:
  - 높은 정확도
  - 80개 이상 언어 지원 (한글 우수)
  - GPU 가속 지원
  - 설치 간단
- **단점**:
  - 모델 크기가 큼
  - Tesseract보다 느림
- **적합한 용도**: 일반적인 OCR 작업, 한글 문서

### 3. **docTR (Document Text Recognition)**
- **특징**: 최신 딥러닝 OCR, 문서 레이아웃 분석
- **장점**:
  - 문서 구조 이해 (테이블, 단락 등)
  - 높은 정확도
  - GPU 가속 지원
  - 한글 지원
- **단점**:
  - 메모리 사용량 높음
  - 속도가 느림
- **적합한 용도**: 복잡한 문서 레이아웃, 테이블 포함 문서

### 4. **Calamari OCR**
- **특징**: 역사적 문서 및 필기체 특화
- **장점**:
  - 필기체 인식 우수
  - 오래된 문서/역사 자료 처리 강점
  - 사용자 정의 모델 학습 가능
  - TensorFlow 기반 GPU 가속
- **단점**:
  - 일반 인쇄물은 다른 모델이 더 나을 수 있음
  - 모델 다운로드 필요
- **적합한 용도**: 필기체, 역사 자료, 손글씨

---

## 빠른 시작

### 1. Conda 환경 생성
```bash
bash setup_ocr_envs.sh
```

### 2. 모델별 설치

#### A100 GPU (40GB/80GB)
```bash
# EasyOCR (추천)
conda activate easyocr
bash install_easyocr_a100.sh

# Tesseract
conda activate tesseract-ocr
bash install_tesseract_a100.sh
sudo apt-get install tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng

# docTR
conda activate doctr-ocr
bash install_doctr_a100.sh

# Calamari
conda activate calamari-ocr
bash install_calamari_a100.sh
```

#### RTX 3080 (10GB)
```bash
# EasyOCR (추천)
conda activate easyocr
bash install_easyocr_rtx3080.sh

# Tesseract
conda activate tesseract-ocr
bash install_tesseract_rtx3080.sh
sudo apt-get install tesseract-ocr tesseract-ocr-kor tesseract-ocr-eng

# docTR
conda activate doctr-ocr
bash install_doctr_rtx3080.sh

# Calamari
conda activate calamari-ocr
bash install_calamari_rtx3080.sh
```

### 3. 실행 방법

```bash
# EasyOCR
conda activate easyocr
python main_easyocr_a100.py ./pdfs/your_file.pdf

# Tesseract
conda activate tesseract-ocr
python main_tesseract_a100.py ./pdfs/your_file.pdf

# docTR
conda activate doctr-ocr
python main_doctr_a100.py ./pdfs/your_file.pdf

# Calamari
conda activate calamari-ocr
python main_calamari_a100.py ./pdfs/your_file.pdf
```

---

## 성능 비교 (참고용)

| 모델 | 속도 (A100) | 정확도 (인쇄물) | 정확도 (필기체) | GPU 메모리 | 한글 지원 |
|------|-------------|-----------------|-----------------|------------|-----------|
| Tesseract | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ | N/A (CPU) | ⭐⭐⭐⭐ |
| EasyOCR | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ~2GB | ⭐⭐⭐⭐⭐ |
| docTR | ⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ | ~3GB | ⭐⭐⭐⭐ |
| Calamari | ⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ~2GB | ⭐⭐ |

---

## Vision Language Models (VLM) vs 순수 OCR

### VLM (DeepSeek, Qwen)
- **장점**: 문맥 이해, 질문 응답, 이미지 설명
- **단점**: 느림, GPU 메모리 많이 필요 (8GB+)
- **용도**: 복잡한 문서 이해, AI 기반 분석

### 순수 OCR (위 4가지 모델)
- **장점**: 빠름, 가벼움, 단순 텍스트 추출
- **단점**: 문맥 이해 없음, 텍스트만 추출
- **용도**: 대량 문서 처리, 빠른 텍스트 추출

---

## 추천 사용 시나리오

| 시나리오 | 추천 모델 |
|---------|----------|
| 일반 문서 (한글/영어) | **EasyOCR** |
| 대량 문서 빠른 처리 | **Tesseract** |
| 복잡한 레이아웃 (테이블 등) | **docTR** |
| 필기체/손글씨/역사 자료 | **Calamari** |
| 문서 내용 이해/요약 필요 | **DeepSeek/Qwen (VLM)** |

---

## 전체 프로젝트 구조

```
document-ocr/
├── setup_ocr_envs.sh              # 환경 생성
│
├── install_tesseract_a100.sh      # Tesseract 설치 (A100)
├── install_tesseract_rtx3080.sh   # Tesseract 설치 (RTX 3080)
├── loader_tesseract_a100.py       # Tesseract 로더 (A100)
├── main_tesseract_a100.py         # Tesseract 실행 (A100)
│
├── install_easyocr_a100.sh        # EasyOCR 설치 (A100)
├── install_easyocr_rtx3080.sh     # EasyOCR 설치 (RTX 3080)
├── loader_easyocr_a100.py         # EasyOCR 로더 (A100)
├── main_easyocr_a100.py           # EasyOCR 실행 (A100)
│
├── install_doctr_a100.sh          # docTR 설치 (A100)
├── install_doctr_rtx3080.sh       # docTR 설치 (RTX 3080)
├── loader_doctr_a100.py           # docTR 로더 (A100)
├── main_doctr_a100.py             # docTR 실행 (A100)
│
├── install_calamari_a100.sh       # Calamari 설치 (A100)
├── install_calamari_rtx3080.sh    # Calamari 설치 (RTX 3080)
├── loader_calamari_a100.py        # Calamari 로더 (A100)
└── main_calamari_a100.py          # Calamari 실행 (A100)
```
