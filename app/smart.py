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
    def _log(self, msg: str):
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        # strip ANSI for file
        import re
        clean = re.sub(r"\033\[[0-9;]*m", "", line)
        with open(LOG_FILE, "a", encoding="utf-8") as f:
            f.write(clean + "\n")

    def p(self, msg: str):
        """Print + log."""
        print(msg)
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
        """Returns (ok, version, device, error_msg)."""
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

    # ── Requirements hash ─────────────────────────────────────
    def _req_hash(self):
        return hashlib.md5(REQ_FILE.read_bytes()).hexdigest()

    def _deps_current(self):
        stored = HASH_FILE.read_text().strip() if HASH_FILE.exists() else ""
        return self._req_hash() == stored

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
                           "--force-reinstall", "--no-deps"])

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
                           "--force-reinstall", "--no-deps"])

        elif gtype == "apple":
            self.p(c(Y, "  🚀 Installing Apple Silicon torch (MPS at runtime)..."))
            r = self._run(["uv","pip","install", *CPU_PKGS,
                           "--index-url", CPU_INDEX,
                           "--force-reinstall", "--no-deps"])

        else:
            self.p(c(Y, "  🚀 Installing CPU-only torch..."))
            r = self._run(["uv","pip","install", *CPU_PKGS,
                           "--index-url", CPU_INDEX,
                           "--force-reinstall", "--no-deps"])

        if r.returncode != 0:
            self.p(c(R, "  ❌ torch install failed"))
            return False

        # Verify it actually loads
        ok, ver, dev, err = self._torch_status()
        if ok:
            self.p(c(G, f"  ✅ torch {ver} verified — device: {dev}"))
            self._save_state(torch_version=ver, torch_device=dev)
            return True

        # CUDA DLL failed → CPU fallback
        self.p(c(Y, f"  ⚠️  torch load failed: {err[:120]}"))
        self.p(c(Y, "  🔄 Auto-fallback: installing CPU-only torch..."))
        r = self._run(["uv","pip","install", *CPU_PKGS,
                       "--index-url", CPU_INDEX,
                       "--force-reinstall", "--no-deps"])
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
        HASH_FILE.write_text(self._req_hash())
        self.p(c(G, "  ✅ Python deps installed"))
        return True

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

        ok  = self._install_deps()
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
        issues = []

        # Detect hardware
        hw = self._detect_hardware()
        self.p(c(W, f"\n  OS  : {hw['os']}"))
        self.p(c(W, f"  GPU : {hw['gpu_name']}"))
        self.p("")

        # Check torch
        self.p("  🔍 Checking torch...")
        ok, ver, dev, err = self._torch_status()
        if ok:
            self.p(c(G, f"  ✅ torch {ver} ({dev})"))
        else:
            self.p(c(R, f"  ❌ torch BROKEN: {err[:120]}"))
            issues.append("torch")

        # Check deps
        self.p("  🔍 Checking dependencies...")
        if self._deps_current():
            self.p(c(G, "  ✅ Python deps OK"))
        else:
            self.p(c(Y, "  ⚠️  Deps hash mismatch"))
            issues.append("deps")

        if not issues:
            self.p(c(G, "\n  ✅ Everything looks healthy — no fixes needed!\n"))
            return 0

        self.p(c(Y, f"\n  🔧 Fixing {len(issues)} issue(s)...\n"))

        if "deps" in issues:
            self._install_deps(force=True)
        if "torch" in issues:
            self._install_torch(force=True)

        # Re-check
        ok, ver, dev, _ = self._torch_status()
        if ok:
            self.p(c(G, f"\n  ✅ Fix complete — torch {ver} running on {dev}\n"))
        else:
            self.p(c(R, "\n  ⚠️  Some issues remain — check logs/smart.log\n"))
        return 0

    # ──────────────────────────────────────────────────────────
    def start(self):
        """Fast pre-launch check — minimal output."""
        # Deps
        if not self._deps_current():
            self.p("📦 Deps changed — updating...")
            if not self._install_deps(force=True):
                sys.exit(1)
        else:
            self.p(c(G, "✅ Deps OK"))

        # Torch
        ok, ver, dev, err = self._torch_status()
        if ok:
            self.p(c(G, f"✅ torch {ver} ({dev})"))
        else:
            self.p(c(Y, f"⚠️  torch issue: {err[:80]}"))
            self.p("🔄 Auto-fixing torch...")
            self._install_torch(force=True)

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
        choices=["install","update","fix","start","check"])
    args = parser.parse_args()

    s = SmartSetup()
    modes = {
        "install": s.install,
        "update":  s.update,
        "fix":     s.fix,
        "start":   s.start,
        "check":   s.check,
    }
    sys.exit(modes[args.mode]() or 0)
