"""
Post-install torch verifier.
Called by install.js after torch.js to confirm the installed torch
actually loads on this machine. If CUDA DLL fails (Windows), auto-
falls back to CPU-only torch so the app is always runnable.
"""
import subprocess
import sys

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CPU_PKGS  = ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"]

def install_cpu():
    print("🔄 Installing CPU-only torch (fallback)...")
    r = subprocess.run([
        "uv", "pip", "install", *CPU_PKGS,
        "--index-url", CPU_INDEX,
        "--force-reinstall", "--no-deps"
    ])
    if r.returncode == 0:
        print("✅ CPU-only torch installed — GPU acceleration disabled.")
    else:
        print("❌ CPU torch install failed.")
        sys.exit(1)

# ── Detect hardware info before trying torch ─────────────────
import platform as _plt
import subprocess as _sp

def _nvcc_version():
    try:
        out = _sp.check_output(["nvcc", "--version"], stderr=_sp.DEVNULL).decode()
        for line in out.splitlines():
            if "release" in line:
                return line.strip()
    except Exception:
        return None

def _nvidia_smi():
    try:
        out = _sp.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                                  "--format=csv,noheader"], stderr=_sp.DEVNULL).decode().strip()
        return out
    except Exception:
        return None

print("=" * 56)
print("  Torch Verification & Hardware Report")
print("=" * 56)
print(f"  OS      : {_plt.system()} {_plt.release()} ({_plt.machine()})")

smi = _nvidia_smi()
if smi:
    print(f"  GPU     : {smi}")
    nvcc = _nvcc_version()
    print(f"  NVCC    : {nvcc or 'not found in PATH'}")
else:
    print("  GPU     : No NVIDIA GPU detected (or nvidia-smi not in PATH)")
print("-" * 56)

# ── Try importing torch ───────────────────────────────────────
try:
    import torch
    version = torch.__version__
    cuda_ok = torch.cuda.is_available()
    if cuda_ok:
        dev_name = torch.cuda.get_device_name(0)
        vram     = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        device   = f"cuda — {dev_name} ({vram} MB VRAM)"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps (Apple Silicon)"
    else:
        device = "cpu"

    print(f"  torch   : {version}")
    print(f"  device  : {device}")
    print("=" * 56)
    print("✅ torch loaded successfully — install complete!")

except OSError as e:
    print(f"  torch   : LOAD FAILED")
    print(f"  error   : {e}")
    print("=" * 56)
    print("⚠️  CUDA DLL incompatible with this system.")
    install_cpu()
    # Re-verify after fallback
    try:
        import importlib
        import torch as _t  # fresh import after reinstall
        print(f"✅ Fallback torch {_t.__version__} loaded — running on CPU.")
    except Exception:
        print("⚠️  Torch still unavailable. App will use PIL/OpenCV only.")
