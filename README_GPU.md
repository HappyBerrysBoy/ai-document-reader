# GPU 설정 가이드

## 빠른 설치 (권장)

GPU 서버에서 자동 설치 스크립트 사용:

```bash
# 스크립트 실행 (CUDA 버전 자동 감지 후 설치)
chmod +x install_gpu.sh
./install_gpu.sh
```

## 수동 설치

### 1. CUDA 버전 확인
```bash
nvidia-smi  # CUDA Version 확인
nvcc --version  # CUDA toolkit 버전 확인 (설치되어 있다면)
```

### 2. Python 버전 확인 및 환경 설정

**중요**: PaddlePaddle은 Python 3.8-3.12를 지원합니다. Python 3.13은 지원하지 않습니다!

Python 3.13인 경우 새 환경 생성:
```bash
# conda 사용
conda create -n doc-ocr python=3.12
conda activate doc-ocr

# 또는 pyenv 사용
pyenv install 3.12.7
pyenv virtualenv 3.12.7 doc-ocr
pyenv activate doc-ocr
```

### 3. PaddlePaddle GPU 버전 설치

**중요**: CUDA 버전에 따라 설치 명령어가 다릅니다!

#### Step 1: 기존 PaddlePaddle 제거
```bash
pip uninstall paddlepaddle paddlepaddle-gpu -y
```

#### Step 2: CUDA 버전에 맞춰 설치

**CUDA 11.8 (가장 일반적)**
```bash
python -m pip install paddlepaddle-gpu==2.6.1.post118 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

**CUDA 11.7**
```bash
python -m pip install paddlepaddle-gpu==2.6.1.post117 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

**CUDA 12.0**
```bash
python -m pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
```

**CUDA 12.3 / 13.0+ (최신 버전, PaddlePaddle 3.0 베타)**
```bash
# CUDA 12.3+ 지원 (CUDA 13.0도 호환)
python -m pip install paddlepaddle-gpu==3.0.0b2 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/

# 또는 직접 whl 파일 지정
pip install https://paddle-wheel.bj.bcebos.com/3.0.0-beta2/linux/linux-gpu-cuda12.3-cudnn9.0-mkl-gcc12.2-avx/paddlepaddle_gpu-3.0.0b2-cp310-cp310-linux_x86_64.whl
```

**CUDA 11.2 (구 버전)**
```bash
python -m pip install paddlepaddle-gpu==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple
```

공식 설치 가이드: https://www.paddlepaddle.org.cn/install/quick

### 3. GPU 작동 확인
```bash
python -c "import paddle; print('CUDA available:', paddle.device.is_compiled_with_cuda()); print('GPU count:', paddle.device.cuda.device_count() if paddle.device.is_compiled_with_cuda() else 0)"
```

출력 예시:
```
CUDA available: True
GPU count: 1
```

### 4. OCR 실행 및 GPU 사용 확인
```bash
# 다른 터미널에서 GPU 모니터링
watch -n 1 nvidia-smi

# OCR 실행
python main.py your_document.pdf
```

로그에서 다음과 같이 표시되어야 합니다:
```
INFO:loader:GPU detected: Using CUDA (device count: 1)
```

## Mac/CPU 환경에서 설치

```bash
pip install paddlepaddle
```

또는 `requirements.txt`에서:
```
# paddlepaddle-gpu  # GPU 버전 주석 처리
paddlepaddle  # CPU 버전 활성화
```

## 문제 해결

### GPU가 인식되지 않는 경우
1. PaddlePaddle이 CPU 버전인지 확인
2. CUDA 버전과 PaddlePaddle-GPU 버전 호환성 확인
3. NVIDIA 드라이버 버전 확인

### GPU 메모리 부족
환경 변수로 배치 사이즈 조정 (필요시):
```bash
export FLAGS_fraction_of_gpu_memory_to_use=0.5
python main.py your_document.pdf
```
