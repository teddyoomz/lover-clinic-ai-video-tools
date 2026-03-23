# Codebase Map — Lover Clinic AI Video Tools

> Quick-nav for Claude. Read this before touching any file to avoid redundant searches.
> **Keep this file updated every time code structure changes.**

---

## Project Root

```
lover-clinic-ai-video-tools/
├── app/
│   ├── app.py              ← Main Gradio app (844 lines). ALL UI + AI logic here.
│   ├── smart.py            ← Unified intelligence layer: install/update/fix/start/check
│   ├── requirements.txt    ← Python deps (MD5 hash tracked in ../.req_hash)
│   ├── verify_torch.py     ← Legacy: post-install torch verifier (superseded by smart.py)
│   ├── check_deps.py       ← Legacy: hash check + smoke test (superseded by smart.py)
│   ├── static/
│   │   ├── icon.png        ← App icon (loaded as base64 in app.py:66)
│   │   └── logo.png        ← Lover Clinic wordmark (loaded as base64 in app.py:67)
│   └── logs/
│       ├── app.log         ← Runtime error log (gitignored, written by app.py)
│       └── smart.log       ← Smart setup log (written by smart.py)
├── install.js              ← python smart.py install → fs.link
├── start.js                ← python smart.py start → python app.py --port {{port}}
├── update.js               ← python smart.py update
├── fix.js                  ← python smart.py fix
├── reset.js                ← deletes app/env
├── pinokio.js              ← dynamic sidebar UI (auto-boot via default:true on Start)
├── pinokio.json            ← metadata (title, description, icon)
├── icon.png                ← branding icon (pinokio sidebar)
├── README.md               ← user-facing docs
├── CODEBASE.md             ← this file
├── .smart_state.json       ← smart.py state (torch version, last run — gitignored)
├── .req_hash               ← MD5 of requirements.txt (gitignored)
└── .gitignore
```

---

## app/app.py — Line Index (844 lines)

| Lines | Section | What it does |
|-------|---------|--------------|
| 1–28 | Imports | stdlib + PIL, cv2, numpy, gradio, traceback |
| 29–30 | **Logging setup** | `logger = logging.getLogger("lover-clinic")` → `app/logs/app.log` |
| 33–46 | **Torchvision patch** | `sys.modules` shim for `functional_tensor` (removed in torchvision ≥ 0.16) |
| 52–68 | **Base64 image helpers** | `_b64_img()`, `_ICON_SRC`, `_LOGO_SRC` — avoids `/file=` path issues in Gradio |
| 74 | Model caches | `_realesrgan_cache{}` — lazy per-key dict |
| 79–90 | `get_device()` | Returns `"cuda"` / `"mps"` / `"cpu"` (catches OSError for CUDA DLL failures) |
| 91–127 | `get_realesrgan(scale, model_type)` | Lazy-loads RealESRGAN. Keys: `"4_general"`, `"2_general"`, `"4_anime"` |
| 130–139 | `get_rembg_session(model_name)` | Lazy-loads rembg ONNX session |
| 141–157 | `upscale_photo()` | Real-ESRGAN photo upscale → returns `(PIL.Image, status_str)` |
| 161–237 | `upscale_video()` | cv2 frame extract → Real-ESRGAN per frame → ffmpeg/cv2 reassemble |
| 241–279 | `remove_background()` | rembg + BiRefNet. BG options: Transparent / White / Black / Red / Custom |
| 283–316 | `enhance_image()` | GFPGAN v1.3 + optional Real-ESRGAN bg pass |
| 324–346 | `resize_image()` | PIL resize, aspect-ratio lock, 4 resample filters |
| 350–362 | `crop_image()` | PIL crop with bounds clamping |
| 366–396 | `convert_format()` | PIL save to JPEG/PNG/WEBP/BMP/TIFF, quality slider |
| 404–556 | `CSS` constant | Full Red/Black/White/Fire theme |
| 557–577 | `HEADER_HTML` | f-string with base64 `_ICON_SRC`/`_LOGO_SRC` |
| 578–587 | `WARN_GPU` | GPU warning info-box HTML |
| 588–819 | `build_app()` | Gradio UI — 7 tabs wired to functions above |
| 820–844 | `__main__` | argparse `--port`, `allowed_paths`, `favicon_path`, `app.launch()` |

---

## app/smart.py — Line Index (447 lines)

| Lines | Section | What it does |
|-------|---------|--------------|
| 1–44 | Config | Paths, torch versions, ANSI colours |
| 50–111 | `SmartSetup.__init__` / helpers | State load, logging, `_run()`, `_torch_status()` (subprocess-safe) |
| 113–119 | `_req_hash()` / `_deps_current()` | MD5 hash of `requirements.txt` vs stored `.req_hash` |
| 122–173 | `_detect_hardware()` | nvidia-smi / rocm-smi / Apple arm64 detection → dict |
| 176–249 | `_install_torch()` | GPU-aware torch install: CUDA12.8 / DirectML / ROCm6.3 / CPU + auto-fallback |
| 252–265 | `_install_deps()` | `uv pip install -r requirements.txt` (skip if hash matches) |
| 277–290 | `install()` mode | Hardware report + `_install_deps` + `_install_torch` |
| 293–337 | `update()` mode | git pull + diff → only reinstall changed components |
| 340–384 | `fix()` mode | Diagnose torch + deps → auto-repair broken items |
| 387–404 | `start()` mode | Fast pre-launch: dep hash check + torch smoke → auto-fix if broken |
| 407–428 | `check()` mode | Print full hardware + health report |
| 432–447 | Entry point | `argparse` → dispatch to mode |

State file: `.smart_state.json` (torch_version, torch_device, updated_at)
Log file: `app/logs/smart.log`

---

## Gradio Tab → Function Map

| Tab label | Handler function | Line |
|-----------|-----------------|------|
| 🔬 AI Upscale Photo | `upscale_photo` | 141 |
| 🎬 AI Upscale Video | `upscale_video` | 161 |
| ✂️ AI Remove BG | `remove_background` | 241 |
| ✨ AI Enhance Image | `enhance_image` | 283 |
| 📐 Resize Image | `resize_image` | 324 |
| ✂️ Crop Image | `crop_image` | 350 |
| 🔄 Convert Format | `convert_format` | 366 |

---

## Model URLs (hard-coded in get_realesrgan / enhance_image)

| Key | Model file | Line |
|-----|-----------|------|
| `4_general` | `RealESRGAN_x4plus.pth` | ~105 |
| `2_general` | `RealESRGAN_x2plus.pth` | ~99 |
| `4_anime` | `realesr-animevideov3.pth` | ~93 |
| GFPGAN | `GFPGANv1.3.pth` | ~295 |

---

## Pinokio Scripts — Quick Ref

| File | Delegates to | Notes |
|------|-------------|-------|
| `install.js` | `python smart.py install` | GPU detect → deps → torch → `fs.link` |
| `start.js` | `python smart.py start` → `python app.py --port {{port}}` | `daemon:true`, URL captured by regex |
| `update.js` | `python smart.py update` | git pull + smart dep diff |
| `fix.js` | `python smart.py fix` | Diagnose + auto-repair torch/deps |
| `reset.js` | deletes `app/env` | Pinokio built-in `fs.delete` or shell `rm -rf` |
| `pinokio.js` | — | `default:true` on Start → **auto-boot** when installed & idle |

### pinokio.js State Machine

```
Not installed           → default: Install  (auto-runs install.js)
Installing              → default: Installing spinner
Installed + idle        → default: Start    (auto-runs start.js = SMART BOOT)
Starting (no URL yet)   → default: Starting spinner
Running + URL ready     → default: Open Web UI  (auto-opens browser)
Updating                → default: Updating spinner
Resetting               → default: Resetting spinner
Fixing                  → default: Fixing spinner
```

---

## Known Issues & Fixes Applied

| Issue | Fix | File:Line |
|-------|-----|-----------|
| `torchvision.transforms.functional_tensor` missing (≥0.16) | `sys.modules` shim | `app.py:39` |
| CUDA DLL WinError 127 | Subprocess torch test + OSError catch in `get_device()` | `smart.py:93`, `app.py:79` |
| Gradio 6.0 warning: theme/css in Blocks() | Moved to `launch()` | `app.py:820` |
| ffmpeg conda install fails (gdk-pixbuf post-link) | Removed step — Pinokio bundles ffmpeg | `install.js` |
| `fs.link` wrong param (`drive:` → `venv:`) | Fixed | `install.js` |
| Logo/icon not displaying (`/file=` broken in Gradio) | Base64 data URI embed | `app.py:52` |

---

## Logs Location

| Log | Path | Purpose |
|-----|------|---------|
| App runtime errors | `app/logs/app.log` | Full Python tracebacks per function |
| Smart setup log | `app/logs/smart.log` | Install/update/fix/start history |
| Pinokio install log | `logs/api/install.js/latest` | smart.py install output |
| Pinokio start log | `logs/api/start.js/latest` | server stdout |

---

## Dependencies (requirements.txt)

```
gradio>=4.44.0              UI framework
pillow>=10.0.0              Image I/O
numpy<2.0.0                 pinned — basicsr requires <2
opencv-python-headless      video + image processing
basicsr                     Real-ESRGAN backbone
facexlib                    face detection (GFPGAN dep)
gfpgan                      face enhancement
realesrgan                  photo/video upscale
rembg                       background removal (BiRefNet/U2Net)
devicetorch                 auto GPU/CPU torch selector
ffmpeg-python               ffmpeg Python bindings
pydantic==2.10.6            pinned — Gradio 4.x compat
requests                    HTTP downloads
```
