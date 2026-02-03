# GPU 설정 가이드

## GPU 서버에서 설치 (RTX 3080 등)

### 1. CUDA 버전 확인
```bash
nvidia-smi
nvcc --version
```

### 2. PaddlePaddle GPU 버전 설치

#### CUDA 11.8 (일반적인 경우)
```bash
pip uninstall paddlepaddle -y
pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### CUDA 12.0+
```bash
pip uninstall paddlepaddle -y
pip install paddlepaddle-gpu -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 특정 CUDA 버전 지정
```bash
# CUDA 11.7
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu117/

# CUDA 12.3
python -m pip install paddlepaddle-gpu==3.0.0b1 -i https://www.paddlepaddle.org.cn/packages/stable/cu123/
```

자세한 설치 가이드: https://www.paddlepaddle.org.cn/install/quick

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
