"""
Smart dependency checker.
1. Runs uv pip install only when requirements.txt hash changed.
2. Verifies torch loads correctly — if CUDA DLL fails on Windows,
   auto-reinstalls CPU-only torch so the app still starts.
"""
import hashlib
import pathlib
import subprocess
import sys

# ── 1. Requirements hash check ────────────────────────────────
req   = pathlib.Path("requirements.txt")
hfile = pathlib.Path("../.req_hash")

current = hashlib.md5(req.read_bytes()).hexdigest()
stored  = hfile.read_text().strip() if hfile.exists() else ""

if current != stored:
    print("📦 requirements.txt changed — updating dependencies...")
    result = subprocess.run(["uv", "pip", "install", "-r", "requirements.txt"])
    if result.returncode != 0:
        print("❌ Dependency install failed. Check output above.")
        sys.exit(1)
    hfile.write_text(current)
    print("✅ Dependencies updated.")
else:
    print("✅ Dependencies up to date — skipping install.")

# ── 2. Torch smoke test ───────────────────────────────────────
print("🔍 Verifying torch...")
try:
    import torch  # noqa: F401
    print(f"✅ torch {torch.__version__} loaded OK")
except OSError as e:
    print(f"⚠️  torch CUDA DLL error: {e}")
    print("🔄 Reinstalling CPU-only torch...")
    result = subprocess.run([
        "uv", "pip", "install",
        "torch", "torchvision",
        "--index-url", "https://download.pytorch.org/whl/cpu",
        "--force-reinstall"
    ])
    if result.returncode == 0:
        # Reset hash so next start re-checks deps
        hfile.write_text("")
        print("✅ CPU-only torch installed. Restart will be clean.")
    else:
        print("❌ Could not reinstall torch. The app will run without GPU.")
except ImportError as e:
    print(f"⚠️  torch not installed yet: {e} — will install on first use.")
