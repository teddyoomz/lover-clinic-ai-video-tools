"""
Lover Clinic AI Tools — Smart Setup System
==========================================
Unified intelligence layer for install / update / fix / start.

Usage:
  python smart.py install       # full install with GPU auto-detect
  python smart.py update        # smart git-aware dep update
  python smart.py fix           # diagnose & auto-repair issues
  python smart.py start         # pre-launch dep+torch check
  python smart.py check         # print hardware & health report
"""

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────
APP_DIR    = Path(__file__).parent
ROOT_DIR   = APP_DIR.parent
REQ_FILE   = APP_DIR / "requirements.txt"
HASH_FILE  = ROOT_DIR / ".req_hash"
STATE_FILE = ROOT_DIR / ".smart_state.json"
LOG_DIR    = APP_DIR / "logs"
LOG_FILE   = LOG_DIR / "smart.log"

# ── Torch versions ────────────────────────────────────────────
TORCH_VER     = "2.7.0"
TORCHVISION   = "0.22.0"
TORCHAUDIO    = "2.7.0"
CPU_INDEX     = "https://download.pytorch.org/whl/cpu"
CUDA_INDEX    = "https://download.pytorch.org/whl/cu128"
ROCM_INDEX    = "https://download.pytorch.org/whl/rocm6.3"
CPU_PKGS      = [f"torch=={TORCH_VER}", f"torchvision=={TORCHVISION}", f"torchaudio=={TORCHAUDIO}"]

# ── Colours (ANSI) ────────────────────────────────────────────
R = "\033[91m"; G = "\033[92m"; Y = "\033[93m"
B = "\033[94m"; W = "\033[97m"; DIM = "\033[2m"; X = "\033[0m"

def c(colour, text): return f"{colour}{text}{X}"


# ══════════════════════════════════════════════════════════════
class SmartSetup:

    def __init__(self):
        LOG_DIR.mkdir(exist_ok=True)
        self.state = self._load_state()
        self.sys_info = None          # lazy
        self._cleanup_logs(max_age_hours=1.0)

    # ── State persistence ─────────────────────────────────────
    def _load_state(self):
        if STATE_FILE.exists():
            try: return json.loads(STATE_FILE.read_text())
            except: pass
        return {}

    def _save_state(self, **kw):
        self.state.update(kw)
        self.state["updated_at"] = datetime.now().isoformat()
        STATE_FILE.write_text(json.dumps(self.state, indent=2))

    # ── Logging ───────────────────────────────────────────────
    def _cleanup_logs(self, max_age_hours: float = 1.0):
        """Delete smart.log if it's older than max_age_hours, to prevent log bloat."""
        import time
        if LOG_FILE.exists():
            try:
                age = time.time() - LOG_FILE.stat().st_mtime
                if age > max_age_hours * 3600:
                    LOG_FILE.unlink()
            except Exception:
                pass

    def _log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        # strip ANSI for file
        import re
        clean = re.sub(r"\033\[[0-9;]*m", "", line)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(clean + "\n")

    def p(self, msg: str):
        """Print + log (safe for any terminal encoding)."""
        try:
            print(msg)
        except (UnicodeEncodeError, UnicodeDecodeError):
            # Fallback for terminals with limited encodings (e.g. Windows cp874)
            import re
            clean = re.sub(r"[^\x00-\x7F]+", "?", msg)
            print(clean)
        self._log(msg)

    # ── Shell helpers ─────────────────────────────────────────
    def _run(self, cmd, cwd=None, capture=False, timeout=None):
        return subprocess.run(
            cmd, cwd=cwd or APP_DIR,
            capture_output=capture, text=capture,
            timeout=timeout,
        )

    # ── Torch test (subprocess — safe from DLL crashes) ───────
    def _torch_status(self):
        """Returns (ok, version, device, error_msg).
        device will be 'cpu' for CPU-only builds even if GPU exists."""
        script = (
            "import torch; "
            "d='cuda:'+torch.cuda.get_device_name(0) if torch.cuda.is_available() "
            "else ('mps' if hasattr(torch.backends,'mps') and torch.backends.mps.is_available() else 'cpu'); "
            "print(torch.__version__+':'+d)"
        )
        try:
            r = self._run([sys.executable, "-c", script], capture=True, timeout=30)
            if r.returncode == 0:
                out = r.stdout.strip()
                parts = out.split(":", 1)
                ver = parts[0]
                dev = parts[1] if len(parts) > 1 else "cpu"
                return True, ver, dev, None
            return False, None, "cpu", (r.stderr or r.stdout)[:300]
        except Exception as e:
            return False, None, "cpu", str(e)

    def _gpu_torch_needed(self):
        """Returns True if a GPU exists and we should be using GPU torch.
        Used to detect the case where torch works but is CPU-only despite having GPU."""
        hw = self._detect_hardware()
        return hw["gpu_type"] in ("nvidia", "amd")

    def _torch_is_cpu_build(self, ver: str) -> bool:
        """Returns True if the torch version string indicates a CPU-only build (e.g. '2.7.0+cpu')."""
        return ver is not None and "+cpu" in ver

    # ── Requirements hash ─────────────────────────────────────
    def _req_hash(self):
        return hashlib.md5(REQ_FILE.read_bytes()).hexdigest()

    def _deps_current(self):
        stored = HASH_FILE.read_text().strip() if HASH_FILE.exists() else ""
        return self._req_hash() == stored

    # ── Deep import health check ───────────────────────────────
    _CRITICAL_IMPORTS = {
        "gradio":    "import gradio",
        "cv2":       "import cv2",
        "PIL":       "from PIL import Image",
        "numpy":     "import numpy",
        "basicsr":   "from basicsr.archs.rrdbnet_arch import RRDBNet",
        "realesrgan":"from realesrgan import RealESRGANer",
        "realesrgan.srvgg": "from realesrgan.archs.srvgg_arch import SRVGGNetCompact",
        "gfpgan":    "from gfpgan import GFPGANer",
        "rembg":     "from rembg import remove",
        "yt_dlp":    "import yt_dlp",
    }

    def _check_imports(self):
        """Subprocess-safe import test for all critical packages.
        Returns dict  {name: {"ok": bool, "error": str|None}}"""
        import json as _json
        lines = [
            "import sys, json",
            "results = {}",
        ]
        for name, stmt in self._CRITICAL_IMPORTS.items():
            safe_stmt = stmt.replace("\\", "\\\\").replace('"', '\\"')
            safe_name = name.replace(".", "_")
            lines.append(
                f'try:\n    exec("{safe_stmt}")\n    results["{name}"] = {{"ok":True}}'
                f'\nexcept Exception as e:\n    results["{name}"] = {{"ok":False,"error":str(e)[:200]}}'
            )
        lines.append("print(json.dumps(results))")
        script = "\n".join(lines)
        try:
            r = self._run([sys.executable, "-c", script], capture=True, timeout=60)
            if r.returncode == 0 and r.stdout.strip().startswith("{"):
                return _json.loads(r.stdout.strip())
        except Exception as e:
            self.p(c(Y, f"  ⚠️  Import check failed to run: {e}"))
        return {}

    # ── Targeted package repair ────────────────────────────────
    def _repair_imports(self, failed: list):
        """Given a list of failed import keys, run targeted repairs."""
        repaired = set()

        # --- pydantic pin (often breaks after gradio update) ---
        if not any(k in ("gradio",) for k in failed):
            pass  # only if gradio import ok
        self.p(c(Y, "  📌 Pinning pydantic==2.10.6 ..."))
        self._run(["uv", "pip", "install", "pydantic==2.10.6", "--force-reinstall"])

        # --- basicsr / realesrgan / gfpgan cluster ---
        # NOTE: These packages often fail because torch is CPU-only (fs.link side-effect).
        # Always ensure GPU torch is installed first before reinstalling AI packages.
        # CRITICAL: After reinstalling AI packages, ALWAYS re-verify torch — uv's
        # --force-reinstall resolves torch as a dependency and may silently downgrade
        # CUDA torch back to CPU torch from the default PyPI index.
        ai_cluster = {"basicsr", "realesrgan", "realesrgan.srvgg", "gfpgan"}
        if ai_cluster.intersection(failed):
            # Step A: ensure GPU torch BEFORE reinstalling AI packages
            ok, ver, dev, _ = self._torch_status()
            if ok and (self._torch_is_cpu_build(ver) or dev == "cpu") and self._gpu_torch_needed():
                self.p(c(Y, "  🚨 CPU torch with GPU hardware — reinstalling GPU torch first..."))
                self._install_torch(force=True)

            # Step B: reinstall AI packages (may silently downgrade torch!)
            self.p(c(Y, "  🔧 Reinstalling basicsr / realesrgan / gfpgan / facexlib ..."))
            self._run(["uv", "pip", "install",
                       "basicsr", "realesrgan", "gfpgan", "facexlib",
                       "--force-reinstall"])
            repaired |= ai_cluster

            # Step C: ALWAYS re-verify torch after AI package install — it may have
            # been downgraded to CPU by uv's dependency resolver.
            if self._gpu_torch_needed():
                ok2, ver2, dev2, _ = self._torch_status()
                if ok2 and (self._torch_is_cpu_build(ver2) or dev2 == "cpu"):
                    self.p(c(Y, "  🚨 torch was downgraded to CPU during AI package install — restoring GPU torch..."))
                    self._install_torch(force=True)

        # --- rembg / onnxruntime ---
        if "rembg" in failed:
            self.p(c(Y, "  🔧 Reinstalling rembg + onnxruntime ..."))
            hw = self._detect_hardware()
            ort_pkg = "onnxruntime-gpu" if hw["gpu_type"] == "nvidia" else "onnxruntime"
            self._run(["uv", "pip", "install", "rembg", ort_pkg, "--force-reinstall"])
            repaired.add("rembg")

        # --- yt_dlp ---
        if "yt_dlp" in failed:
            self.p(c(Y, "  🔧 Reinstalling yt-dlp ..."))
            self._run(["uv", "pip", "install", "yt-dlp", "--force-reinstall"])
            repaired.add("yt_dlp")

        # --- cv2 ---
        if "cv2" in failed:
            self.p(c(Y, "  🔧 Reinstalling opencv-python-headless ..."))
            self._run(["uv", "pip", "install", "opencv-python-headless", "--force-reinstall"])
            repaired.add("cv2")

        # --- gradio ---
        if "gradio" in failed:
            self.p(c(Y, "  🔧 Reinstalling gradio ..."))
            self._run(["uv", "pip", "install", "gradio", "--force-reinstall"])
            self._run(["uv", "pip", "install", "pydantic==2.10.6", "--force-reinstall"])
            repaired.add("gradio")

        return repaired

    # ── ffmpeg availability ────────────────────────────────────
    def _check_ffmpeg(self):
        """Returns True if a working ffmpeg binary is found."""
        import shutil as _sh
        for exe in (_sh.which("ffmpeg"), r"C:\ffmpeg\bin\ffmpeg.exe"):
            if exe and os.path.isfile(exe):
                try:
                    r = subprocess.run([exe, "-version"],
                                       capture_output=True, timeout=5)
                    if r.returncode == 0:
                        return True
                except Exception:
                    pass
        # imageio-ffmpeg bundled binary
        try:
            r = self._run([sys.executable, "-c",
                           "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"],
                          capture=True, timeout=15)
            if r.returncode == 0 and r.stdout.strip():
                return True
        except Exception:
            pass
        return False

    def _install_ffmpeg(self):
        """Try to install ffmpeg via conda (cross-platform)."""
        import shutil as _sh
        if _sh.which("conda"):
            self.p(c(Y, "  🔧 Installing ffmpeg via conda ..."))
            r = self._run(["conda", "install", "-y", "-c", "conda-forge", "ffmpeg"])
            return r.returncode == 0
        # fallback: imageio-ffmpeg (bundled Python ffmpeg)
        self.p(c(Y, "  🔧 Installing imageio-ffmpeg (bundled ffmpeg) ..."))
        r = self._run(["uv", "pip", "install", "imageio-ffmpeg", "--force-reinstall"])
        return r.returncode == 0

    # ── GPU / hardware detection ──────────────────────────────
    def _detect_hardware(self):
        if self.sys_info:
            return self.sys_info

        info = {
            "os": f"{platform.system()} {platform.release()} ({platform.machine()})",
            "gpu_type": "cpu",
            "gpu_name": "No GPU / CPU only",
            "vram": "",
            "driver": "",
        }

        # NVIDIA
        try:
            out = subprocess.check_output(
                ["nvidia-smi",
                 "--query-gpu=name,memory.total,driver_version",
                 "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=6
            ).decode().strip()
            if out:
                parts = [p.strip() for p in out.split(",")]
                info.update({
                    "gpu_type": "nvidia",
                    "gpu_name": parts[0],
                    "vram":     parts[1] if len(parts) > 1 else "?",
                    "driver":   parts[2] if len(parts) > 2 else "?",
                })
                self.sys_info = info
                return info
        except Exception:
            pass

        # AMD
        try:
            r = subprocess.run(
                ["rocm-smi", "--showproductname"],
                capture_output=True, text=True, timeout=5
            )
            if r.returncode == 0 and "GPU" in r.stdout:
                info.update({"gpu_type": "amd", "gpu_name": "AMD GPU (ROCm detected)"})
                self.sys_info = info
                return info
        except Exception:
            pass

        # Apple Silicon
        if platform.system() == "Darwin" and platform.machine() == "arm64":
            info.update({"gpu_type": "apple", "gpu_name": "Apple Silicon (MPS)"})

        self.sys_info = info
        return info

    # ── Install torch for this machine ────────────────────────
    def _install_torch(self, force=False):
        if not force:
            ok, ver, dev, err = self._torch_status()
            if ok:
                self.p(c(G, f"  ✅ torch {ver} already working ({dev}) — skipping"))
                return True

        hw = self._detect_hardware()
        gtype = hw["gpu_type"]
        os_name = platform.system()

        self.p(c(B, f"  🔍 GPU detected : {hw['gpu_name']}"))

        if gtype == "nvidia" and os_name == "Windows":
            self.p(c(Y, "  🚀 Installing NVIDIA CUDA 12.8 torch (Windows)..."))
            r = self._run(["uv","pip","install", *CPU_PKGS,
                           "--index-url", CUDA_INDEX,
                           "--force-reinstall"])

        elif gtype == "nvidia" and os_name == "Linux":
            self.p(c(Y, "  🚀 Installing NVIDIA CUDA 12.8 torch (Linux)..."))
            r = self._run(["uv","pip","install", *CPU_PKGS,
                           "--index-url", CUDA_INDEX, "--force-reinstall"])

        elif gtype == "amd" and os_name == "Windows":
            self.p(c(Y, "  🚀 Installing AMD DirectML torch (Windows)..."))
            r = self._run(["uv","pip","install",
                           "torch","torch-directml","torchaudio","torchvision",
                           "--force-reinstall"])

        elif gtype == "amd" and os_name == "Linux":
            self.p(c(Y, "  🚀 Installing AMD ROCm 6.3 torch (Linux)..."))
            r = self._run(["uv","pip","install", *CPU_PKGS,
                           "--index-url", ROCM_INDEX,
                           "--force-reinstall"])

        elif gtype == "apple":
            self.p(c(Y, "  🚀 Installing Apple Silicon torch (MPS at runtime)..."))
            r = self._run(["uv","pip","install", *CPU_PKGS,
                           "--index-url", CPU_INDEX,
                           "--force-reinstall"])

        else:
            self.p(c(Y, "  🚀 Installing CPU-only torch..."))
            r = self._run(["uv","pip","install", *CPU_PKGS,
                           "--index-url", CPU_INDEX,
                           "--force-reinstall"])

        if r.returncode != 0:
            self.p(c(R, "  ❌ torch install failed"))
            return False

        # Verify it actually loads
        ok, ver, dev, err = self._torch_status()
        if ok:
            self.p(c(G, f"  ✅ torch {ver} verified — device: {dev}"))
            self._save_state(torch_version=ver, torch_device=dev)
            return True

        # CUDA DLL failed → diagnose driver version, then CPU fallback
        self.p(c(R, f"  ❌ CUDA torch failed to load: {err[:200]}"))

        # Check if this is a driver version problem
        driver_ver = ""
        try:
            drv_out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                stderr=subprocess.DEVNULL, timeout=6
            ).decode().strip()
            driver_ver = drv_out.split("\n")[0].strip()
        except Exception:
            pass

        if driver_ver:
            try:
                major = int(driver_ver.split(".")[0])
                if major < 528:
                    self.p(c(R, f"  ⚠️  NVIDIA Driver {driver_ver} is TOO OLD for CUDA 12.8!"))
                    self.p(c(Y, "  ➡  Please update NVIDIA Driver to version 528+ from:"))
                    self.p(c(Y, "     https://www.nvidia.com/Download/index.aspx"))
                    self.p(c(Y, "     After updating, run 'Fix' from the Pinokio menu."))
                else:
                    self.p(c(Y, f"  Driver {driver_ver} looks OK — may be a DLL conflict"))
                    self.p(c(Y, "  Try: update NVIDIA driver to latest, then run Fix"))
            except (ValueError, IndexError):
                self.p(c(Y, f"  NVIDIA Driver: {driver_ver} — if CUDA fails, update driver"))

        self.p(c(Y, "  🔄 Auto-fallback: installing CPU-only torch (GPU features disabled)..."))
        r = self._run(["uv","pip","install", *CPU_PKGS,
                       "--index-url", CPU_INDEX,
                       "--force-reinstall"])
        if r.returncode == 0:
            ok, ver, dev, err = self._torch_status()
            if ok:
                self.p(c(G, f"  ✅ CPU fallback torch {ver} loaded ({dev})"))
                self._save_state(torch_version=ver, torch_device="cpu (fallback)")
                return True

        self.p(c(R, "  ❌ Could not install any working torch — app will use CPU PIL/cv2 only"))
        return False

    # ── Install pip deps ──────────────────────────────────────
    def _install_deps(self, force=False):
        if not force and self._deps_current():
            self.p(c(G, "  ✅ Python deps up to date (hash match)"))
            return True

        self.p(c(Y, "  📦 Installing Python dependencies..."))
        r = self._run(["uv","pip","install","-r","requirements.txt"])
        if r.returncode != 0:
            self.p(c(R, "  ❌ pip install failed"))
            return False
        self._run(["uv","pip","install","pydantic==2.10.6"])

        # For NVIDIA GPUs: swap onnxruntime (CPU) → onnxruntime-gpu so that
        # rembg background removal runs on GPU from first install, not just CPU.
        hw = self._detect_hardware()
        if hw["gpu_type"] == "nvidia":
            self.p(c(Y, "  🎮 NVIDIA detected — installing onnxruntime-gpu (replaces CPU onnxruntime)..."))
            self._run(["uv", "pip", "install", "onnxruntime-gpu", "--force-reinstall"])

        # Guard: requirements.txt AI packages (basicsr, realesrgan, etc.) may pull in
        # CPU torch as a dependency. Re-ensure GPU torch BEFORE writing the hash.
        # Only write hash if torch is verified correct — prevents false "deps ok" on
        # machines where torch restore failed.
        torch_ok = True
        if self._gpu_torch_needed():
            ok_d, ver_d, dev_d, _ = self._torch_status()
            if ok_d and (self._torch_is_cpu_build(ver_d) or dev_d == "cpu"):
                self.p(c(Y, "  🚨 deps install downgraded torch to CPU — restoring GPU torch..."))
                torch_ok = self._install_torch(force=True)

        HASH_FILE.write_text(self._req_hash())
        self.p(c(G, "  ✅ Python deps installed"))
        return torch_ok

    # ── Banner ────────────────────────────────────────────────
    def _banner(self, title):
        self.p(c(R, "\n  ╔══════════════════════════════════════════════╗"))
        self.p(c(R, f"  ║  🔥 Lover Clinic — Smart {title:<20}║"))
        self.p(c(R, "  ╚══════════════════════════════════════════════╝"))

    # ══════════════════════════════════════════════════════════
    # PUBLIC MODES
    # ══════════════════════════════════════════════════════════

    def install(self):
        self._banner("Install")
        hw = self._detect_hardware()
        self.p(c(W, f"\n  OS  : {hw['os']}"))
        self.p(c(W, f"  GPU : {hw['gpu_name']}"))
        if hw["vram"]:   self.p(c(W, f"  VRAM: {hw['vram']}"))
        if hw["driver"]: self.p(c(W, f"  Drv : {hw['driver']}"))
        self.p("")

        # Always force-install deps on fresh install — .req_hash may survive
        # a reset (env deleted) giving a false "up to date" result.
        ok  = self._install_deps(force=True)
        ok &= self._install_torch()

        self.p(c(G if ok else Y, "\n  🔥 Install complete!\n"))
        return 0 if ok else 1

    # ──────────────────────────────────────────────────────────
    def update(self):
        self._banner("Update")

        # 1. Git pull
        self.p(c(B, "\n  📡 Pulling from GitHub..."))
        r = self._run(["git","pull"], cwd=ROOT_DIR, capture=True)
        pull_out = (r.stdout + r.stderr).strip()
        self.p(f"  {pull_out}")

        already_latest = "Already up to date" in pull_out

        # 2. Detect changed files
        changed = []
        if not already_latest:
            diff = self._run(
                ["git","diff","HEAD@{1}","HEAD","--name-only"],
                cwd=ROOT_DIR, capture=True
            )
            changed = [f.strip() for f in diff.stdout.strip().splitlines() if f.strip()]
            if changed:
                self.p(c(W, f"  📝 Changed files: {', '.join(changed)}"))

        # 3. Smart decision
        req_changed   = any("requirements.txt" in f for f in changed)
        torch_changed = any("torch.js" in f or "verify_torch" in f
                            or "smart.py" in f for f in changed)
        deps_stale    = not self._deps_current()

        if req_changed or deps_stale:
            reason = "requirements.txt updated" if req_changed else "local hash mismatch"
            self.p(c(Y, f"\n  🔄 Updating deps ({reason})..."))
            self._install_deps(force=True)

        if torch_changed:
            self.p(c(Y, "\n  🔄 torch config changed — reinstalling..."))
            self._install_torch(force=True)

        if not req_changed and not torch_changed and not deps_stale:
            if already_latest:
                self.p(c(G, "\n  ✅ Already up to date — nothing to do"))
            else:
                self.p(c(G, "\n  ✅ Code updated — no dep changes needed"))

        self.p(c(G, "\n  🔥 Update complete!\n"))
        return 0

    # ──────────────────────────────────────────────────────────
    def fix(self):
        self._banner("Fix")

        # Detect hardware
        hw = self._detect_hardware()
        self.p(c(W, f"\n  OS  : {hw['os']}"))
        self.p(c(W, f"  GPU : {hw['gpu_name']}"))
        self.p("")

        issues_found = False

        # ── 1. Deps hash ──────────────────────────────────────
        self.p("  🔍 Checking dependency hash...")
        if self._deps_current():
            self.p(c(G, "  ✅ Deps hash OK"))
        else:
            self.p(c(Y, "  ⚠️  Deps hash mismatch — reinstalling..."))
            self._install_deps(force=True)
            issues_found = True

        # ── 2. pydantic pin (must be 2.10.6) ─────────────────
        self.p("  🔍 Pinning pydantic==2.10.6 ...")
        self._run(["uv", "pip", "install", "pydantic==2.10.6", "--quiet"])

        # ── 3. torch ──────────────────────────────────────────
        self.p("  🔍 Checking torch...")
        ok, ver, dev, err = self._torch_status()
        if not ok:
            self.p(c(R, f"  ❌ torch BROKEN: {err[:120]}"))
            self.p(c(Y, "  🔧 Reinstalling torch for this machine..."))
            self._install_torch(force=True)
            issues_found = True
        elif self._gpu_torch_needed() and (self._torch_is_cpu_build(ver) or dev == "cpu"):
            self.p(c(Y, f"  ⚠️  torch {ver} is CPU-only despite GPU hardware — reinstalling GPU torch..."))
            self._install_torch(force=True)
            issues_found = True
        else:
            self.p(c(G, f"  ✅ torch {ver} ({dev})"))

        # ── 4. Deep AI package import check ──────────────────
        self.p("  🔍 Testing all AI package imports...")
        results = self._check_imports()
        failed = [name for name, r in results.items() if not r.get("ok")]
        passed = [name for name, r in results.items() if r.get("ok")]

        for name in passed:
            self.p(c(G, f"  ✅ {name}"))
        for name in failed:
            err_msg = (results[name].get("error") or "")[:80]
            self.p(c(R, f"  ❌ {name} — {err_msg}"))

        if failed:
            self.p(c(Y, f"\n  🔧 Repairing {len(failed)} broken package(s)...\n"))
            self._repair_imports(failed)
            issues_found = True

        # ── 5. ffmpeg ─────────────────────────────────────────
        self.p("  🔍 Checking ffmpeg...")
        if self._check_ffmpeg():
            self.p(c(G, "  ✅ ffmpeg OK"))
        else:
            self.p(c(Y, "  ⚠️  ffmpeg not found — installing..."))
            self._install_ffmpeg()
            issues_found = True

        # ── 6. Final verification ─────────────────────────────
        self.p(c(B, "\n  🔍 Final verification...\n"))
        ok, ver, dev, _ = self._torch_status()
        results2 = self._check_imports()
        still_failed = [n for n, r in results2.items() if not r.get("ok")]

        if ok and not still_failed:
            msg = "✅ All systems healthy!" if not issues_found else "✅ All issues fixed!"
            self.p(c(G, f"\n  {msg}  torch {ver} on {dev}\n"))
            return 0
        else:
            if not ok:
                self.p(c(R, f"  ❌ torch still broken"))
            for n in still_failed:
                self.p(c(R, f"  ❌ {n} still failing"))
            self.p(c(Y, "\n  ⚠️  Some issues could not be auto-fixed."))
            self.p(c(Y, "     Try: Reset → Re-install to start fresh.\n"))
            return 1

    # ──────────────────────────────────────────────────────────
    def _gradio_ok(self):
        """Quick sanity check — can we actually import gradio?"""
        r = self._run([sys.executable, "-c", "import gradio"],
                      capture=True, timeout=15)
        return r.returncode == 0

    def start(self):
        """Pre-launch health check with auto-heal — minimal, fast output."""
        heal_done = False

        # ── 1. Deps hash ──────────────────────────────────────
        if not self._deps_current() or not self._gradio_ok():
            self.p("📦 Deps missing or changed — installing...")
            if not self._install_deps(force=True):
                self.p(c(R, "❌ Dep install failed — run Fix from the menu"))
                sys.exit(1)
            heal_done = True
        else:
            self.p(c(G, "✅ Deps OK"))

        # ── 2. pydantic pin ───────────────────────────────────
        self._run(["uv", "pip", "install", "pydantic==2.10.6", "--quiet"])

        # ── 3. Torch ──────────────────────────────────────────
        ok, ver, dev, err = self._torch_status()
        if ok:
            # Torch loads fine — but is it the RIGHT build for this hardware?
            gpu_needed = self._gpu_torch_needed()
            is_cpu_build = self._torch_is_cpu_build(ver)
            running_on_cpu = (dev == "cpu")

            if gpu_needed and (is_cpu_build or running_on_cpu):
                # fs.link may have replaced CUDA torch with CPU version from shared cache
                self.p(c(Y, f"⚠️  torch {ver} is CPU-only but GPU hardware detected — reinstalling GPU torch..."))
                self._install_torch(force=True)
                # Re-check after reinstall
                ok2, ver2, dev2, err2 = self._torch_status()
                if ok2:
                    self.p(c(G, f"✅ torch {ver2} ({dev2})"))
                else:
                    self.p(c(Y, f"⚠️  GPU torch failed to load ({err2[:80]}) — continuing with CPU fallback"))
                heal_done = True
            else:
                self.p(c(G, f"✅ torch {ver} ({dev})"))
        else:
            self.p(c(Y, f"⚠️  torch broken — auto-fixing..."))
            self._install_torch(force=True)
            heal_done = True

        # ── 4. Quick import smoke-test (critical packages only) ─
        QUICK_CHECKS = {
            "gradio":     "import gradio",
            "cv2":        "import cv2",
            "basicsr":    "from basicsr.archs.rrdbnet_arch import RRDBNet",
            "realesrgan": "from realesrgan import RealESRGANer",
            "realesrgan.srvgg": "from realesrgan.archs.srvgg_arch import SRVGGNetCompact",
        }
        # Temporarily override _CRITICAL_IMPORTS for quick scan
        _orig = self._CRITICAL_IMPORTS
        self.__class__._CRITICAL_IMPORTS = QUICK_CHECKS
        results = self._check_imports()
        self.__class__._CRITICAL_IMPORTS = _orig

        failed = [n for n, r in results.items() if not r.get("ok")]
        if failed:
            self.p(c(Y, f"⚠️  Import issues: {', '.join(failed)} — auto-repairing..."))
            self._repair_imports(failed)
            heal_done = True
        else:
            self.p(c(G, "✅ Core AI packages OK"))

        # ── 5. Final torch guard — repair_imports may have downgraded torch ──
        # uv --force-reinstall resolves torch as dependency and can install CPU
        # version from default PyPI, overwriting the CUDA torch we set up above.
        if self._gpu_torch_needed():
            ok_f, ver_f, dev_f, _ = self._torch_status()
            if ok_f and (self._torch_is_cpu_build(ver_f) or dev_f == "cpu"):
                self.p(c(Y, f"⚠️  torch downgraded to CPU ({ver_f}) — restoring GPU torch..."))
                self._install_torch(force=True)
                heal_done = True
            # else: torch is fine

        if heal_done:
            self.p(c(G, "✅ Auto-heal complete — launching app..."))
        else:
            self.p(c(G, "✅ All checks passed — launching app..."))

    # ──────────────────────────────────────────────────────────
    def torch_reinstall(self):
        """Force-reinstall GPU torch (run after fs.link which may overwrite with CPU version)."""
        self._banner("Torch Reinstall")
        hw = self._detect_hardware()
        self.p(c(W, f"\n  GPU : {hw['gpu_name']}"))
        self.p("")
        ok = self._install_torch(force=True)
        self.p(c(G if ok else Y, "\n  🔥 Torch reinstall complete!\n"))
        return 0 if ok else 1

    # ──────────────────────────────────────────────────────────
    def check(self):
        """Print hardware + health report."""
        self._banner("Health Check")
        hw = self._detect_hardware()

        self.p(c(W, f"\n  OS     : {hw['os']}"))
        self.p(c(W, f"  GPU    : {hw['gpu_name']}"))
        if hw["vram"]:   self.p(c(W, f"  VRAM   : {hw['vram']}"))
        if hw["driver"]: self.p(c(W, f"  Driver : {hw['driver']}"))

        ok, ver, dev, err = self._torch_status()
        status = c(G, f"✅ {ver} ({dev})") if ok else c(R, f"❌ {err[:80]}")
        self.p(f"  torch  : {status}")

        deps_ok = self._deps_current()
        self.p(f"  deps   : {c(G,'✅ current') if deps_ok else c(Y,'⚠️  stale')}")

        st = self.state
        if st.get("updated_at"):
            self.p(c(DIM, f"  last   : {st['updated_at']}"))
        self.p("")
        return 0


# ── Entry point ───────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="smart.py")
    parser.add_argument("mode",
        choices=["install","update","fix","start","check","torch"])
    args = parser.parse_args()

    s = SmartSetup()
    modes = {
        "install": s.install,
        "torch":   s.torch_reinstall,
        "update":  s.update,
        "fix":     s.fix,
        "start":   s.start,
        "check":   s.check,
    }
    sys.exit(modes[args.mode]() or 0)
