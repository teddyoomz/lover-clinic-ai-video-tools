# Codebase Map — Lover Clinic AI Video Tools

> Quick-nav for Claude. Read this before touching any file to avoid redundant searches.
> **Keep this file updated every time code structure changes.**

---

## Project Root

```
lover-clinic-ai-video-tools/
├── app/
│   ├── app.py              ← Single Gradio app (~915 lines). ALL logic lives here.
│   ├── requirements.txt    ← Python deps
│   ├── static/
│   │   └── icon.png        ← App icon served via allowed_paths → /file=static/icon.png
│   └── logs/app.log        ← Runtime error log (gitignored, written by app.py)
├── install.js              ← pip install + torch.js + fs.link
├── start.js                ← git pull → python app.py --port {{port}}
├── update.js               ← uv pip install -r requirements.txt (re-deps)
├── reset.js                ← deletes app/env
├── torch.js                ← cross-platform PyTorch installer (do not touch)
├── pinokio.js              ← dynamic sidebar UI
├── pinokio.json            ← metadata (title, description, icon)
├── icon.png                ← branding icon
├── README.md               ← user-facing docs
├── CODEBASE.md             ← this file
└── .gitignore
```

---

## app/app.py — Line Index

| Lines | Section | What it does |
|-------|---------|--------------|
| 1–13 | Imports | stdlib + PIL, cv2, numpy, gradio |
| 15–30 | **Logging setup** | Writes to `app/logs/app.log` + stderr. `logger = logging.getLogger("lover-clinic")` |
| 32–45 | **Torchvision patch** | `sys.modules` shim for `functional_tensor` (removed in torchvision ≥ 0.16) |
| 47–53 | Model caches | `_realesrgan_cache{}`, `_gfpgan_cache{}`, `_rembg_sessions{}` — lazy, keyed dicts |
| 56–65 | `get_device()` | Returns `"cuda"` / `"mps"` / `"cpu"` |
| 68–104 | `get_realesrgan(scale, model_type)` | Lazy-loads RealESRGAN. Keys: `"2_general"`, `"4_general"`, `"4_anime"` |
| 107–111 | `get_rembg_session(model_name)` | Lazy-loads rembg ONNX session |
| 114–135 | `upscale_photo()` | Real-ESRGAN photo upscale → returns `(PIL.Image, status_str)` |
| 138–215 | `upscale_video()` | cv2 frame extract → Real-ESRGAN per frame → ffmpeg/cv2 reassemble |
| 218–257 | `remove_background()` | rembg + BiRefNet. BG options: Transparent / White / Black / Red / Custom |
| 260–294 | `enhance_image()` | GFPGAN v1.3 + optional Real-ESRGAN bg pass |
| 297–324 | `resize_image()` | PIL resize, aspect-ratio lock, 4 resample filters |
| 327–340 | `crop_image()` | PIL crop with bounds clamping |
| 343–374 | `convert_format()` | PIL save to JPEG/PNG/WEBP/BMP/TIFF, quality slider |
| 377–600 | `CSS` constant | Full Red/Black/White/Fire theme + header redesign styles |
| 601–615 | `HEADER_HTML` | Icon + brand name + CLINIC + subtitle + tech pills |
| 617–622 | `WARN_GPU` | GPU warning info-box HTML |
| 625–890 | `build_app()` | Gradio UI — 7 tabs wired to functions above |
| 892–915 | `__main__` | argparse `--port`, `allowed_paths`, `favicon_path`, `app.launch()` |

---

## Gradio Tab → Function Map

| Tab label | Handler function | Line |
|-----------|-----------------|------|
| 🔬 AI Upscale Photo | `upscale_photo` | 118 |
| 🎬 AI Upscale Video | `upscale_video` | 138 |
| ✂️ AI Remove BG | `remove_background` | 218 |
| ✨ AI Enhance Image | `enhance_image` | 260 |
| 📐 Resize Image | `resize_image` | 301 |
| ✂️ Crop Image | `crop_image` | 327 |
| 🔄 Convert Format | `convert_format` | 343 |

---

## Model URLs (hard-coded in get_realesrgan)

| Key | Model file | Line |
|-----|-----------|------|
| `4_general` | `RealESRGAN_x4plus.pth` | 90 |
| `2_general` | `RealESRGAN_x2plus.pth` | 84 |
| `4_anime` | `realesr-animevideov3.pth` | 78 |
| GFPGAN | `GFPGANv1.3.pth` | 273 |

---

## Pinokio Scripts — Quick Ref

| File | Key params | Notes |
|------|-----------|-------|
| `install.js` | `venv:"env" path:"app"` | step 1: pip; step 2: torch.js; step 3: fs.link venv:"app/env" |
| `start.js` | `daemon:true` | git pull first, then `python app.py --port {{port}}`, captures `/(http:\/\/[0-9.:]+)/`, sets `local.url` via `input.event[1]` |
| `update.js` | `venv:"env" path:"app"` | re-runs pip install only |
| `reset.js` | — | deletes `app/env` |
| `pinokio.js` | — | checks `info.exists("app/env")`, `info.running(...)`, `info.local("start.js").url` |

---

## Known Issues & Fixes Applied

| Issue | Fix | File:Line |
|-------|-----|-----------|
| `torchvision.transforms.functional_tensor` missing (≥0.16) | `sys.modules` shim | `app.py:37` |
| Gradio 6.0 warning: theme/css in Blocks() | Moved to `launch()` | `app.py:810` |
| ffmpeg conda install fails (gdk-pixbuf post-link) | Removed step — Pinokio bundles ffmpeg | `install.js` |
| `fs.link` wrong param (`drive:` → `venv:`) | Fixed | `install.js` |

---

## Logs Location

| Log | Path | Purpose |
|-----|------|---------|
| App runtime errors | `app/logs/app.log` | Full Python tracebacks per function |
| Pinokio install log | `logs/api/install.js/latest` | conda/pip output |
| Pinokio start log | `logs/api/start.js/latest` | server stdout |
| Pinokio torch log | `logs/api/torch.js/latest` | PyTorch install output |

---

## Dependencies (requirements.txt)

```
gradio>=4.44.0      UI framework
pillow>=10.0.0      Image I/O
numpy<2.0.0         pinned — basicsr requires <2
opencv-python-headless  video + image processing
basicsr             Real-ESRGAN backbone
facexlib            face detection (GFPGAN dep)
gfpgan              face enhancement
realesrgan          photo/video upscale
rembg               background removal (BiRefNet/U2Net)
devicetorch         auto GPU/CPU torch selector
ffmpeg-python       ffmpeg Python bindings
pydantic==2.10.6    pinned — Gradio 4.x compat
requests            HTTP downloads
```
