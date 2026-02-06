# Flash Attention 2 설치 문제 해결 가이드

## 왜 Flash Attention이 필요한가?

- **A100 GPU**: 2-3배 속도 향상
- **메모리 효율**: 더 큰 배치 처리 가능
- **고해상도**: 더 높은 DPI로 OCR 처리 가능

## Flash Attention 없이 사용하기

**중요**: Flash Attention 없이도 모델은 정상 작동합니다. 단지 속도가 느릴 뿐입니다.

현재 코드는 Flash Attention이 없으면 자동으로 일반 attention으로 대체됩니다:
```
⚠️  Flash Attention 2 미설치 - 기본 attention 사용
✓ A100 최적화 활성화: bfloat16 (Flash Attention 2 없음)
```

## 설치 방법

### 방법 1: Ninja 빌드 시스템 먼저 설치 (가장 권장)

```bash
# Conda 환경 활성화
conda activate deepseek-ocr  # 또는 qwen-ocr

# Ninja 설치
pip install ninja

# 다시 Flash Attention 설치 시도
pip install flash-attn --no-build-isolation
```

### 방법 2: 대체 설치 스크립트 사용

```bash
conda activate deepseek-ocr  # 또는 qwen-ocr
bash install_flash_attn_alternative.sh
```

### 방법 3: 특정 버전 사전 빌드 wheel 다운로드

PyTorch 2.5.1 + CUDA 12.1 + Python 3.11 조합인 경우:

```bash
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu121torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
```

다른 조합의 경우 [Flash Attention 릴리즈 페이지](https://github.com/Dao-AILab/flash-attention/releases)에서 찾아보세요.

### 방법 4: 환경 변수 설정 후 재시도

```bash
# CUDA 경로 설정
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 임시 디렉토리 변경 (디스크 공간 충분한 곳으로)
export TMPDIR=/tmp

# 병렬 작업 수 제한 (메모리 부족 방지)
MAX_JOBS=4 pip install flash-attn --no-build-isolation
```

## 실패 원인 진단

### 1. CUDA 컴파일러 확인

```bash
nvcc --version
```

출력 없으면 CUDA Toolkit 설치 필요:
```bash
# Ubuntu/Debian
sudo apt-get install nvidia-cuda-toolkit

# 또는 NVIDIA 공식 사이트에서 다운로드
```

### 2. 디스크 공간 확인

```bash
df -h /tmp
df -h ~
```

최소 10GB 이상 필요. 부족하면 임시 디렉토리 변경:
```bash
export TMPDIR=/path/to/large/disk
```

### 3. 메모리 확인

```bash
free -h
```

컴파일 시 최소 16GB RAM 권장. 부족하면 스왑 메모리 증설 또는 `MAX_JOBS` 줄이기:
```bash
MAX_JOBS=2 pip install flash-attn --no-build-isolation
```

### 4. PyTorch/CUDA 버전 확인

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"
```

Flash Attention은 다음을 요구합니다:
- PyTorch >= 2.0
- CUDA >= 11.8
- Python 3.8-3.12

## 성능 비교 (참고용)

| 설정 | A100 속도 | 메모리 사용 |
|------|-----------|------------|
| Flash Attention 2 + bfloat16 | **100%** (기준) | **100%** (기준) |
| bfloat16만 (Flash Attn 없음) | ~40-50% | ~150% |
| float32 (최적화 없음) | ~20-30% | ~200% |

## 결론

Flash Attention 설치가 계속 실패한다면:

1. **그냥 사용하기**: 현재 코드는 Flash Attention 없이도 작동합니다. 속도가 느릴 뿐입니다.

2. **Ninja 먼저 설치**: `pip install ninja` 후 재시도

3. **사전 빌드 wheel**: 맞는 버전을 GitHub 릴리즈에서 다운로드

4. **시스템 업데이트**: CUDA Toolkit, GCC 컴파일러 등 확인

5. **포기하고 진행**: OCR 정확도에는 영향 없고 속도만 느려질 뿐입니다.
