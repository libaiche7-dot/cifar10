# Environment Setup

Recorded on 2026-11-13. Every number below was read from the working machine, not from memory — reproduce them exactly before trusting a new box.

## Machine

| Item | Value |
|------|-------|
| Laptop | Lenovo Legion R9000P |
| CPU | AMD Ryzen 9 |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU, 8151 MiB |
| Compute capability | 12.0 (Blackwell, `sm_120`) |
| GPU driver | 621.24 |
| OS | Windows 11 |

## Software stack

| Component | Version | Location |
|-----------|---------|----------|
| Conda | 25.11.1 | `D:\Anaconda` (base) |
| Python | 3.11.18 | inside `cv-basics` |
| Environment | `cv-basics` | `D:\envs\cv-basics` |
| PyTorch | 2.15.0+cu134 | |
| torchvision | 0.29.2+cu134 | |
| CUDA runtime | 13.4 | bundled in the wheel |
| NumPy | 2.4.6 | |
| TensorBoard | 2.21.0 | |

No standalone CUDA Toolkit is required. The PyTorch wheel ships its own CUDA runtime; only the driver has to be new enough.

## Install from scratch

```powershell
# 1. Put new conda environments on D: instead of C:
conda config --prepend envs_dirs D:\envs

# 2. Create the environment
conda create -n cv-basics python=3.11 -y
conda activate cv-basics

# 3. PyTorch with CUDA 13.4 (must be >= cu128 for Blackwell)
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu134

# 4. Remaining dependencies (fast mirror, no proxy needed)
python -m pip install numpy pyyaml tqdm tensorboard scikit-learn matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple
```

## Verification

Three checks, in increasing strictness. All three must pass.

```powershell
# 1. Versions and device identity
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), torch.cuda.get_device_capability(0))"
# expected: 2.15.0+cu134 13.4 True (12, 0)

# 2. A real kernel launch -- is_available() alone can lie
python -c "import torch; x=torch.randn(2048,2048,device='cuda'); y=x@x; torch.cuda.synchronize(); print('matmul ok', y.mean().item())"

# 3. Driver and VRAM
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
```

## Pitfalls hit while building this

**Blackwell needs CUDA 12.8 or newer.** Wheels built against cu118/cu121 have no kernels for `sm_120`. `torch.cuda.is_available()` may still return `True`, and the failure only appears at the first kernel launch: `no kernel image is available for execution on the device`.

**`pip install torch` on Windows installs the CPU build.** PyPI's Windows wheels carry no CUDA. The `--index-url https://download.pytorch.org/whl/cuXXX` flag is mandatory.

**pip ignores the Windows system proxy.** Browsers read the WinINET setting; pip does not. Set the environment variables in the current shell:

```powershell
$env:HTTP_PROXY="http://127.0.0.1:7890"
$env:HTTPS_PROXY="http://127.0.0.1:7890"
```

**Proxy rule mode may bypass the download host.** With the client in rule mode, `download.pytorch.org` was routed direct and died at the TLS handshake while Google worked fine. Switching to global mode fixed it.

**Never install into `base`.** Activate `cv-basics` first; the prompt must read `(cv-basics)`.

**Environments default to C:.** `conda config --prepend envs_dirs D:\envs` keeps the multi-GB environments off the system drive. pip's cache still lands in `C:\Users\<user>\AppData\Local\pip\Cache`; reclaim it with `python -m pip cache purge`.

**TensorBoard prints "TensorFlow installation not found".** Harmless — TensorBoard runs standalone with a reduced feature set.

**Line endings.** `.gitattributes` pins `* text=auto eol=lf`, so files stay LF even on Windows and shell scripts will not break on a Linux server.

## Reproducing on a Linux server / AutoDL

Same versions, three differences:

1. Install Miniconda, then `conda create -n cv-basics python=3.11`.
2. Use the Linux wheel index — the command is identical, pip resolves the platform automatically.
3. Set `data.root` in `configs/*.yaml` to the server data disk (e.g. `/root/autodl-tmp/data`) and raise `data.num_workers` from 0-2 to 8. Windows spawns dataloader workers differently and often fails above 2.
