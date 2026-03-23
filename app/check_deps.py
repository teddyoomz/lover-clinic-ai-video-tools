"""
Smart dependency checker — called by start.js before every launch.

1. Compares requirements.txt MD5 hash with stored .req_hash
   → runs uv pip install only when something changed
2. Torch smoke test — if CUDA DLL fails, auto-installs CPU torch
"""
import hashlib
import pathlib
import subprocess
import sys

CPU_INDEX = "https://download.pytorch.org/whl/cpu"
CPU_PKGS  = ["torch==2.7.0", "torchvision==0.22.0", "torchaudio==2.7.0"]

# ── 1. Requirements hash check ────────────────────────────────
req   = pathlib.Path("requirements.txt")
hfile = pathlib.Path("../.req_hash")

current = hashlib.md5(req.read_bytes()).hexdigest()
stored  = hfile.read_text().strip() if hfile.exists() else ""

if current != stored:
    print("📦 requirements.txt changed — updating dependencies...")
    r = subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"])
    if r.returncode != 0:
        print("❌ Dependency install failed.")
        sys.exit(1)
    hfile.write_text(current)
    print("✅ Dependencies updated.")
else:
    print("✅ Dependencies up to date — skipping install.")

# ── 2. Torch smoke test ───────────────────────────────────────
print("🔍 Verifying torch...")
try:
    import torch
    if torch.cuda.is_available():
        dev = f"cuda ({torch.cuda.get_device_name(0)})"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        dev = "mps (Apple Silicon)"
    else:
        dev = "cpu"
    print(f"✅ torch {torch.__version__} | {dev}")

except OSError as e:
    print(f"⚠️  CUDA DLL error: {e}")
    print("🔄 Reinstalling CPU-only torch...")
    r = subprocess.run([
        "uv", "pip", "install", *CPU_PKGS,
        "--index-url", CPU_INDEX,
        "--force-reinstall", "--no-deps"
    ])
    if r.returncode == 0:
        hfile.write_text("")        # force dep re-check next start
        print("✅ CPU-only torch installed. Relaunching cleanly...")
    else:
        print("❌ Could not reinstall torch. App will run without GPU.")

except ImportError:
    print("⚠️  torch not installed — will install via verify_torch.py on next install.")
