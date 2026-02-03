# WSL에서 설치 및 실행 가이드

## 문제점

WSL(Windows Subsystem for Linux) 환경에서는 PaddlePaddle GPU가 Segmentation Fault를 일으킬 수 있습니다. 이는 WSL의 CUDA 지원 한계와 PaddlePaddle의 호환성 문제 때문입니다.

## 해결 방법: CPU 모드 사용 (권장)

WSL에서는 CPU 모드가 안정적입니다.

### 1. CPU 버전 설치

```bash
# Python 3.12 환경
conda create -n doc-ocr python=3.12 -y
conda activate doc-ocr

# 기존 제거
pip uninstall -y paddlepaddle paddlepaddle-gpu

# CPU 버전 설치 (안정적)
pip install paddlepaddle==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 의존성 설치
cd ~/ai/ai-document-reader
pip install -r requirements.txt
pip install Pillow

# 환경 변수 설정 (WSL 최적화)
export DOC_OCR_FORCE_CPU=1
export DOC_OCR_DISABLE_MKLDNN=1
```

### 2. 실행

```bash
# CPU 모드로 실행
DOC_OCR_FORCE_CPU=1 python main.py ./pdfs/screen1.png
```

### 3. 영구 설정 (~/.bashrc에 추가)

```bash
echo 'export DOC_OCR_FORCE_CPU=1' >> ~/.bashrc
echo 'export DOC_OCR_DISABLE_MKLDNN=1' >> ~/.bashrc
source ~/.bashrc
```

## GPU를 사용하고 싶다면

WSL에서 GPU를 사용하려면 다음 조건이 필요합니다:

### 필수 요구사항
1. **WSL 2** (WSL 1은 GPU 미지원)
2. **Windows 11** 또는 Windows 10 (21H2 이상)
3. **NVIDIA 드라이버** (Windows용, 최신 버전)
4. **CUDA Toolkit** WSL 전용 버전 설치

### WSL GPU 설정 (고급)

```bash
# WSL 버전 확인
wsl --version

# CUDA WSL 전용 설치 (Ubuntu 기준)
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-12-0

# LD_LIBRARY_PATH 설정
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH

# PaddlePaddle GPU 설치
pip install paddlepaddle-gpu==2.6.1 -i https://pypi.tuna.tsinghua.edu.cn/simple

# GPU 확인
python -c "import paddle; print('CUDA:', paddle.device.is_compiled_with_cuda())"
```

하지만 여전히 불안정할 수 있으므로, **네이티브 Linux나 Windows에서 직접 실행하는 것을 권장**합니다.

## 성능 비교

- **CPU (WSL)**: 안정적, 속도는 느림 (이미지당 ~5-10초)
- **GPU (네이티브 Linux)**: 빠름 (이미지당 ~1-2초), 안정적
- **GPU (WSL)**: 불안정, Segmentation Fault 발생 가능

## 네이티브 Linux 서버에서 실행

WSL 대신 네이티브 Linux 서버(또는 Docker)에서 실행하면 GPU를 안정적으로 사용할 수 있습니다.

Docker 사용 예시:
```bash
# NVIDIA Docker 런타임 사용
docker run --gpus all -it -v $(pwd):/workspace nvidia/cuda:12.0.0-base-ubuntu22.04 bash

# 컨테이너 내부에서 설치
cd /workspace
pip install paddlepaddle-gpu==2.6.1.post120 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html
pip install -r requirements.txt
python main.py ./pdfs/screen1.png
```

## 요약

- **WSL 사용자**: `DOC_OCR_FORCE_CPU=1`로 CPU 모드 사용 (안정적)
- **GPU 필요 시**: 네이티브 Linux 서버 또는 Docker 사용
- **성능 필요 시**: Windows에서 직접 PaddlePaddle GPU 설치 시도
