# GPU OCR 완전 새 설치 가이드

WSL Ubuntu 24.04 + CUDA 13.0 + RTX 3080 환경에 최적화된 GPU OCR 시스템

## 시스템 요구사항

- **OS**: WSL Ubuntu 24.04
- **GPU**: NVIDIA RTX 3080
- **CUDA**: 13.0
- **Python**: 3.11 (conda 환경)

## 설치 방법

### 1단계: 스크립트 실행 권한 설정

```bash
chmod +x setup_gpu_env.sh
chmod +x install_gpu_packages.sh
chmod +x run_gpu.sh
```

### 2단계: Conda 환경 생성

```bash
./setup_gpu_env.sh
```

### 3단계: Conda 환경 활성화

```bash
conda activate ocr-gpu
```

### 4단계: GPU 패키지 설치

```bash
./install_gpu_packages.sh
```

이 스크립트는 자동으로:
- PaddlePaddle GPU 버전 설치 (CUDA 11.8, CUDA 13.0 호환)
- PaddleOCR 및 의존성 설치
- GPU 작동 확인
- 환경 변수 설정 파일 생성

### 5단계: GPU 테스트

```bash
python -c "import paddle; print('CUDA:', paddle.device.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count())"
```

출력 예시:
```
CUDA: True
GPU count: 1
```

## 실행 방법

### 방법 1: 간편 실행 스크립트 (권장)

```bash
./run_gpu.sh main_gpu.py ./pdfs/your_file.png
```

### 방법 2: 환경 변수 로드 후 실행

```bash
source .env.gpu
python main_gpu.py ./pdfs/your_file.pdf
```

### 방법 3: 직접 실행

```bash
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export DOC_OCR_DISABLE_MKLDNN=1
python main_gpu.py ./pdfs/your_file.png -o output.txt
```

## 사용 예시

### 기본 사용

```bash
# 이미지 OCR
./run_gpu.sh main_gpu.py ./images/document.png

# PDF OCR
./run_gpu.sh main_gpu.py ./pdfs/report.pdf

# 출력 파일 지정
./run_gpu.sh main_gpu.py ./images/scan.jpg -o result.txt

# 다른 언어 (영어)
./run_gpu.sh main_gpu.py ./docs/english.pdf --lang en

# 상세 로그
./run_gpu.sh main_gpu.py ./files/test.png --verbose
```

### 지원 형식

- **이미지**: PNG, JPG, JPEG, BMP, GIF, WEBP, TIFF
- **문서**: PDF, DOCX, XLSX, PPTX

### 지원 언어

- `korean`: 한글 + 영어 (기본)
- `ch`: 중국어 + 영어 + 일본어
- `en`: 영어
- `fr`: 프랑스어
- `de`: 독일어
- `japan`: 일본어
- 기타 80+ 언어

## GPU 성능 모니터링

다른 터미널에서 GPU 사용률 확인:

```bash
watch -n 1 nvidia-smi
```

## 문제 해결

### GPU가 감지되지 않는 경우

1. **CUDA 라이브러리 경로 확인**
   ```bash
   ls -la /usr/lib/wsl/lib/libcuda*
   ```

2. **LD_LIBRARY_PATH 확인**
   ```bash
   echo $LD_LIBRARY_PATH
   ```

   `/usr/lib/wsl/lib`이 포함되어 있어야 합니다.

3. **환경 변수 수동 설정**
   ```bash
   export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
   ```

### Segmentation Fault 발생 시

1. **모델 캐시 삭제**
   ```bash
   rm -rf ~/.paddleocr/
   rm -rf ~/.paddlex/
   ```

2. **PaddlePaddle 재설치**
   ```bash
   conda activate ocr-gpu
   pip uninstall -y paddlepaddle paddlepaddle-gpu
   python -m pip install paddlepaddle-gpu==2.6.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
   ```

### 메모리 부족 오류

이미지 크기가 자동으로 제한되지만, 필요시 수동 조정:

[loader_gpu.py](loader_gpu.py)에서 `det_limit_side_len` 값 조정:
```python
det_limit_side_len=960,  # 기본값, 줄이면 메모리 사용 감소
```

### WSL GPU 드라이버 문제

Windows에서 최신 NVIDIA 드라이버 설치:
https://www.nvidia.com/Download/index.aspx

## 파일 설명

- `setup_gpu_env.sh`: Conda 환경 생성 스크립트
- `install_gpu_packages.sh`: GPU 패키지 설치 스크립트
- `run_gpu.sh`: GPU 모드 실행 스크립트
- `main_gpu.py`: GPU 최적화 메인 프로그램
- `loader_gpu.py`: GPU 최적화 문서 로더
- `requirements-gpu.txt`: GPU 환경 의존성 목록
- `.env.gpu`: GPU 환경 변수 설정 파일 (자동 생성)

## 성능 비교

RTX 3080 기준:

- **CPU 모드**: 이미지당 ~5-10초
- **GPU 모드**: 이미지당 ~0.5-1초

약 **5-10배 성능 향상**

## 주의사항

1. **conda 환경 활성화 필수**
   ```bash
   conda activate ocr-gpu
   ```

2. **WSL 환경에서는 항상 LD_LIBRARY_PATH 설정 필요**
   ```bash
   export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
   ```

3. **첫 실행 시 모델 다운로드로 시간 소요** (~1-2분)

## 환경 변수 참고

- `LD_LIBRARY_PATH`: CUDA 라이브러리 경로
- `DOC_OCR_DISABLE_MKLDNN`: WSL에서 MKLDNN 비활성화
- `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`: 모델 소스 체크 건너뛰기 (속도 향상)

## 추가 최적화

`.bashrc`에 환경 변수 영구 설정:

```bash
echo 'export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export DOC_OCR_DISABLE_MKLDNN=1' >> ~/.bashrc
source ~/.bashrc
```

## 지원

문제 발생 시 다음 정보와 함께 이슈 제기:

```bash
# 시스템 정보
uname -a
nvidia-smi
python --version
pip list | grep paddle

# 에러 로그
./run_gpu.sh main_gpu.py your_file.png --verbose
```
