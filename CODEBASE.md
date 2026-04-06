# CODEBASE MAP — Lover Clinic AI Video Tools

> **อ่านไฟล์นี้ก่อนเสมอ** ก่อนค้นหา code ใด ๆ ในโปรเจ็กต์นี้
> GitHub: https://github.com/teddyoomz/lover-clinic-ai-video-tools
> Last updated: 2026-04-06

---

## 1. Project Structure

```
lover-clinic-ai-video-tools/           ← project root
│
├── CODEBASE.md                        ← THIS FILE — read first
├── CLAUDE.md                          ← Pinokio launcher dev guide (Claude instructions)
├── README.md                          ← user documentation
├── GRADIO_OVERRIDE_KNOWLEDGE.md       ← Gradio 6 override tricks (tab overflow, etc.)
├── ENVIRONMENT                        ← system environment description
│
├── pinokio.json                       ← Pinokio metadata (title, icon, version)
├── pinokio.js                         ← Pinokio launcher UI (dynamic menu)
├── install.js                         ← install: smart.py install → fs.link → smart.py torch → start.js
├── start.js                           ← start: smart.py start → app.py → open browser
├── update.js                          ← update: git pull + smart.py update
├── reset.js                           ← reset: delete app/env
├── fix.js                             ← fix: conda nodejs + smart.py fix
├── torch.js                           ← shared Pinokio torch installer helper
│
├── icon.png                           ← app icon for Pinokio
├── lc_settings.json                   ← [gitignored] persisted user settings
├── .req_hash                          ← [gitignored] requirements.txt hash for smart.py
├── .smart_state.json                  ← [gitignored] smart.py runtime state
│
├── app/                               ← ALL Python app logic
│   ├── app.py                         ← ★ MAIN APP — all UI and processing (~3650 lines)
│   ├── smart.py                       ← smart installer/fixer (~805 lines)
│   ├── requirements.txt               ← Python dependencies
│   ├── check_deps.py                  ← dependency checker helper
│   ├── verify_torch.py                ← torch verification helper
│   ├── env/                           ← [gitignored] Python venv (uv)
│   ├── gfpgan/                        ← [gitignored] auto-downloaded GFPGAN weights
│   ├── logs/                          ← [gitignored] runtime logs (app.log)
│   └── static/
│       ├── icon.png                   ← embedded in header as base64
│       └── logo.png                   ← embedded in header as base64
│
├── models/                            ← [gitignored] cached model weights
├── output/                            ← [gitignored] user output files
├── cache/                             ← [gitignored] misc cache
└── logs/                              ← [gitignored] Pinokio script logs
```

---

## 2. app/app.py — Section Map (~3650 lines)

### Lines 1–16 · Imports
```
gradio, os, sys, json, tempfile, shutil, subprocess, logging, traceback,
platform, pathlib.Path, datetime, PIL.Image, numpy, cv2, io
```

### Lines 18–33 · Logging Setup
- Log file: `app/logs/app.log` · Logger name: `"lover-clinic"`

### Lines 35–49 · Compatibility Patch
- Patches `torchvision.transforms.functional_tensor` (removed in torchvision >= 0.16)
- Required by `basicsr` / `realesrgan`

### Lines 51–71 · Base64 Image Helpers
- `_b64_img(filename)` → loads from `app/static/` as base64 data-URI
- `_ICON_SRC`, `_LOGO_SRC` → pre-loaded at startup

### Lines 73–83 · Project Paths
- `_PROJ_ROOT` = project root (above app/)
- `_OUTPUT_ROOT` = `_PROJ_ROOT/output/`

### Lines 85–205 · Settings Persistence
| Symbol | Purpose |
|--------|---------|
| `_SETTINGS_FILE` | `_PROJ_ROOT/lc_settings.json` |
| `_SETTINGS_DEFAULTS` | Default values for all saveable controls |
| `_load_settings()` | Load JSON + merge with defaults + migrate old labels |
| `_save_settings(cfg)` | Write JSON file |
| `_make_saver(key)` | Returns a 1-arg `.change()` handler that saves one key |

### Lines 208–264 · Startup Checks
| Function | Purpose |
|----------|---------|
| `_startup_device_check()` | Logs GPU info at startup (CUDA/MPS/CPU) |
| `_cleanup_old_logs(max_age_hours)` | Deletes old logs to prevent bloat |

### Lines 266–398 · AI Model Loaders
| Function | Description |
|----------|-------------|
| `get_device()` | Returns `"cuda"` / `"mps"` / `"cpu"` |
| `get_realesrgan(scale, model_type)` | Lazy-loads RealESRGAN (cached in `_realesrgan_cache`) |
| `get_rembg_session(model_name)` | Lazy-loads rembg session (cached in `_rembg_cache`) |
| `get_gfpganer(upscale_factor, enhance_bg)` | Lazy-loads GFPGANer |

### Lines 400–675 · AI Processing Functions
| Function | Description |
|----------|-------------|
| `upscale_photo(image, scale, model_type, output_dir, fmt, progress)` | Real-ESRGAN photo upscale |
| `_free_upsampler_vram(upsampler)` | Free CUDA memory after upscale |
| `_find_ffmpeg()` | Finds ffmpeg binary (hardcoded path first, then which, then imageio-ffmpeg) |
| `upscale_video(video_path, scale, model_type, output_dir, progress)` | Real-ESRGAN per-frame → H.264 |
| `remove_background(image, model_choice, bg_option, custom_bg, output_dir, fmt, progress)` | rembg BiRefNet/U2Net |
| `enhance_image(image, upscale_factor, enhance_bg, output_dir, fmt, progress)` | GFPGAN face enhance |

**rembg model mapping:**
```python
"BiRefNet — General (Best)"  → "birefnet-general"
"BiRefNet — Portrait"        → "birefnet-portrait"
"U2Net"                      → "u2net"
"RMBG 1.4"                   → "isnet-general-use"
```

### Lines 677–977 · Utility Functions + Crop Tool
| Function | Purpose |
|----------|---------|
| `_output_subdir(name)` | Returns `output/{name}/` path |
| `_open_folder(path)` | Opens Explorer + PowerShell AppActivate |
| `_make_cropper_html(img)` | Full HTML + CSS for interactive crop canvas |
| `_save_dir_row(subdir, label)` | Renders output folder textbox + Open button |
| `_save_image(img, save_dir, prefix, fmt)` | Saves PIL image with timestamp |

### Lines 979–1100 · Image Processing Functions
| Function | Description |
|----------|-------------|
| `resize_image(image, w, h, ar, filter, output_dir, fmt)` | PIL resize |
| `crop_image(image, coords_str, output_dir, fmt)` | Crop with rotation/flip |
| `convert_format(image, out_format, quality, output_dir)` | PIL format convert |

**crop_image coords_str format:** `"x1,y1,x2,y2|r:90|fh:1|fv:0"`

### Lines 1102–1640 · WATERMARK REMOVER (Smart Auto: SAM2 + Edge + LaMa)
| Function | Description |
|----------|-------------|
| `_load_lama()` | Lazy-load SimpleLama inpainting model |
| `_editor_to_mask(editor_data)` | Extract BGR image + binary mask from ImageEditor |
| `_inpaint_lama(image_bgr, mask, lama_model)` | Single-frame LaMa inpainting |
| `_load_sam2()` | Lazy-load SAM2 VideoPredictor (`facebook/sam2.1-hiera-tiny`), returns None if unavailable |
| `_propagate_mask_sam2(predictor, mask, video_path, total_frames, progress)` | SAM2 mask propagation → returns `(masks, jpeg_dir)` — JPEG dir kept for edge reuse |
| `_validate_sam2_masks(masks, original_mask, min_frames, stuck_threshold)` | Dual validation: centroid std (<20px=stuck) + mask size ratio (>3x=background) |
| `_track_watermark_edges(first_frame, mask, video_path, total_frames, progress, frames_dir)` | Edge-based template matching; `frames_dir` reuses SAM2 JPEGs if available |
| `_extract_first_frame(video_path)` | Extract frame 1 for ImageEditor preview |
| `remove_watermark_image(editor_data, output_dir, progress)` | Manual mask + LaMa (image) |
| `remove_watermark_video(video_path, editor_data, wm_mode, output_dir, progress)` | Video watermark removal (hybrid Smart Auto) |

**Video watermark modes:**
- **"อยู่กับที่ (เร็ว)"** — Fixed: same mask every frame, LaMa per-frame (fast)
- **"เคลื่อนที่ได้ (Smart Auto)"** — Hybrid SAM2 + Edge tracking with auto-selection

**Smart Auto hybrid pipeline:**
1. **SAM2 first** — Extract all frames as JPEGs → `SAM2VideoPredictor` propagates masks. Best for opaque logos/objects with pixel-perfect segmentation.
2. **Validate (dual check):**
   - Centroid std deviation across frames — if < 20px → SAM2 is "stuck" (background noise jitters ~5-10px, real movement >> 20px)
   - Mask size ratio — if SAM2 median mask > 3x user-drawn mask → tracking large background region, not small watermark
3. **Auto-fallback to Edge tracking** — Reuses SAM2's extracted JPEGs (no re-decode). Canny edge templates + `cv2.matchTemplate(TM_CCOEFF_NORMED)` with `BORDER_CONSTANT` padding.
4. **Transition guard** — When edge tracker detects watermark jumped position (distance > 30% of template size), masks BOTH old and new positions on that frame → eliminates 1-frame flash.
5. **Inpaint** — Per-frame tracked masks → LaMa inpainting → ffmpeg H.264 + audio from original (`-map 1:a?`).

**When each method wins:**
- **SAM2**: Opaque logos, channel bugs, solid objects that move — pixel-perfect masks
- **Edge tracking**: Semi-transparent watermarks, text overlays — SAM2 tracks background instead

### Lines 1443–1582 · LIGHTBOX_JS
- Injected via `launch(js=...)` — intercepts Gradio fullscreen button
- Shows custom lightbox popup (image/video) instead of native fullscreen
- Also patches Gradio tab overflow (getBoundingClientRect override → all tabs visible)

### Lines 1583–2025 · CSS_STYLE + CROP_INIT_JS
- `CSS_STYLE` — Full design system: `--accent #dc2626`, dark theme, responsive breakpoints
- `CROP_INIT_JS` — Interactive crop canvas JS (rotation, flip, zoom, ratio lock, grid overlays)

**Key JS state:** `rotation`, `flipH/V`, `baseScale`, `zoomFactor`, `lockedRatio`, `gridMode`, `sx/sy/ex/ey`

### Lines 2026–2160 · Header & Constants
| Function | Description |
|----------|-------------|
| `_get_gpu_badge_html()` | Returns GPU badge pill (NVIDIA/AMD/MPS/CPU) |
| `_header_html()` | Flat dark topbar: icon + logo + subtitle + badges |
| `WARN_GPU` | Warning HTML for video upscale tab |

### Lines 2162–2735 · CROP_INIT_JS (continued)
Large JS string with crop tool logic — see "Crop Tool" section in original CODEBASE.md for details.

### Lines 2737–3093 · _build_download_tab(cfg)
Video downloader using yt-dlp.

| Inner function | Purpose |
|----------------|---------|
| `_ffmpeg_bin()` | Returns imageio-ffmpeg binary path |
| `_base_ydl_opts()` | Base yt-dlp options (ffmpeg + JS runtime) |
| `_fetch(url)` | Extract video info → quality dropdown |
| `_download(url, selected, quality_map, save_dir)` | Streaming download with progress bar |
| `_pause_toggle()` | Toggle pause state |
| `_stop_download()` | Stop download |

**Quality labels:** 🏆 Best auto, 📹 4K/2K/1080p/720p/etc., 🎵 Audio MP3

### Lines 3095–3432 · build_app() — Main UI
- `cfg = _load_settings()` called once at startup
- All controls use `cfg[key]` as value, `.change(_make_saver(key))` for persistence

**Tab → function → settings keys:**
| Tab | Function | Settings keys |
|-----|----------|---------------|
| 🔬 AI เพิ่มความชัด (ภาพ) | `upscale_photo` | `up_scale, up_model, up_fmt, up_out_dir` |
| 🎬 AI เพิ่มความชัด (วีดีโอ) | `upscale_video` | `vid_scale, vid_model, vid_out_dir` |
| ✂️ AI ลบพื้นหลัง | `remove_background` | `bg_model, bg_option, bg_fmt, bg_out_dir` |
| 🪄 AI ฟื้นฟูภาพ | `enhance_image` | `enh_scale, enh_bg, enh_fmt, enh_out_dir` |
| 📐 ปรับขนาดภาพ | `resize_image` | `res_w, res_h, res_ar, res_filter, res_fmt, res_out_dir` |
| ✂️ ครอปภาพ | `crop_image` | `crop_fmt, crop_out_dir` |
| 🔄 แปลงรูปแบบ | `convert_format` | `conv_fmt, conv_q, conv_out_dir` |
| 🔇 AI ลบ Watermark (ภาพ) | `remove_watermark_image` | `wm_out_dir` |
| 🔇 AI ลบ Watermark (วีดีโอ) | `remove_watermark_video` | `wm_out_dir` |
| ⬇️ ดาวน์โหลดวีดีโอ | `_build_download_tab()` | `dl_out_dir` |

---

## 3. app/smart.py — Section Map (~805 lines)

### Class: SmartSetup

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__()` | 52–56 | Load state, cleanup old logs |
| `_load_state()` / `_save_state()` | 59–68 | JSON state persistence |
| `_log()` / `p()` | 82–100 | Print + log (handles encoding issues) |
| `_run(cmd, cwd, capture, timeout, env)` | 103–108 | subprocess.run wrapper |
| `_torch_status()` | 111–130 | Returns (ok, version, device, error) via subprocess |
| `_gpu_torch_needed()` | 132–136 | True if GPU hardware exists |
| `_torch_is_cpu_build(ver)` | 138–140 | True if `+cpu` in version string |
| `_req_hash()` / `_deps_current()` | 143–148 | MD5 hash of requirements.txt |
| `_CRITICAL_IMPORTS` | 151–164 | Dict of packages to import-test |
| `_check_imports()` | 166–189 | Subprocess import test for all critical packages |
| `_repair_imports(failed)` | 192–258 | Targeted reinstall per failed package |
| `_check_ffmpeg()` / `_install_ffmpeg()` | 261–294 | ffmpeg availability check + conda install |
| `_detect_hardware()` | 297–348 | Detect GPU: nvidia/amd/apple/cpu |
| `_install_torch(force)` | 351–455 | Install correct torch wheel per GPU + fallback |
| `_install_deps(force)` | 458–498 | `uv pip install -r requirements.txt` + SAM-2 + torch |

### Public Modes (entry points)

| Mode | Method | What it does |
|------|--------|--------------|
| `install` | `install()` | Full install: deps + torch |
| `update` | `update()` | Git pull → smart dep update |
| `fix` | `fix()` | Deep health check: deps hash, pydantic pin, torch, imports, torch guard |
| `start` | `start()` | Pre-launch: dep hash check, torch smoke test, quick import check, auto-heal |
| `torch` | `torch_reinstall()` | Force GPU torch reinstall (used after fs.link) |
| `check` | `check()` | Print hardware + health report |

### SAM2 Install (in `_install_deps`)
- Installed with `uv pip install sam2` (lowercase package name)
- Windows: `SAM2_BUILD_CUDA=0` env var to skip CUDA extension compilation
- Also in `QUICK_CHECKS` + `_repair_imports()` for auto-heal at startup
- Non-critical: if install fails, Smart Auto falls back to edge tracking only

---

## 4. Pinokio Launcher Files

### pinokio.js — Dynamic Menu States
```
not installed    → [Install]
installing       → [Installing… spinner]
fixing           → [Fixing… spinner]
installed+idle   → [Start, Update, Fix, Re-install, Reset]
starting (no url)→ [Starting… spinner]
running (url set)→ [Terminal]   ← URL opens in system browser
updating         → [Updating… spinner]
resetting        → [Resetting… spinner]
```

### install.js Flow
```
1. shell.run: conda install nodejs -y         (yt-dlp JS runtime)
2. shell.run: python smart.py install         (GPU detect + pip + SAM-2 + torch)
3. fs.link: app/env                           (symlink venv)
4. shell.run: python smart.py torch           (fix torch after symlink)
5. script.start: start.js                     (auto-launch)
```

### start.js Flow
```
1. shell.run: python smart.py start           (pre-launch check + auto-heal)
2. shell.run: python app.py --port {{port}}
   on: event="/(http:\/\/[0-9.:]+)/"  done:true
3. local.set: url = input.event[1]
4. shell.run: python -c "webbrowser.open('{{local.url}}')"
```

---

## 5. Key Dependencies

| Package | Used For |
|---------|---------|
| `gradio >= 6.0.0, < 7.0.0` | Web UI framework |
| `realesrgan` + `basicsr` + `gfpgan` | Photo/video upscale + face enhance |
| `rembg` + `onnxruntime` | Background removal |
| `simple-lama-inpainting` | Watermark inpainting (single-frame) |
| `sam2` (installed by smart.py) | SAM2 VideoPredictor — used in Smart Auto hybrid for opaque logo tracking |
| `opencv-python-headless` | Video frame I/O + image processing |
| `pillow` + `numpy < 2.0.0` | Image I/O and transforms |
| `transformers >= 4.45.0, < 4.50.0` + `timm` | Florence-2 (legacy, may be removable) |
| `devicetorch` | Cross-platform GPU detection |
| `ffmpeg-python` + `imageio-ffmpeg` | ffmpeg bindings + portable binary |
| `pydantic == 2.10.6` | Pinned for Gradio compatibility |
| `yt-dlp` | Video download from 1000+ sites |

---

## 6. Known Issues & Workarounds

| Issue | Workaround in code |
|-------|-------------------|
| Conda ffmpeg crashes with 0xC0000135 | `_find_ffmpeg()` tries hardcoded paths first |
| torchvision >= 0.16 removed `functional_tensor` | Compat patch at startup (lines 35–49) |
| `<script>` in `gr.HTML` never executes | JS via `.then(fn=None, js=...)` or `launch(js=...)` |
| React controlled inputs ignore `.value=` | Native setter + dispatchEvent pattern |
| `fs.link` overwrites CUDA torch with shared CPU | install.js re-runs `smart.py torch` after |
| Gradio tab "..." overflow hides tabs | Override `getBoundingClientRect` → width:9999 |
| sam2 Windows CUDA compilation fails | `SAM2_BUILD_CUDA=0` env var skips it |
| SAM2 can't track semi-transparent watermarks | Tracks background instead → Smart Auto auto-detects and falls back to edge tracking |
| uv/pip may overwrite CUDA torch during dep install | `_install_torch()` called AFTER `_install_deps()` |

---

## 7. Files NOT to Edit

| Path | Reason |
|------|--------|
| `lc_settings.json` | Auto-generated at runtime, gitignored |
| `app/env/` | Python venv, gitignored |
| `output/` | User output files, gitignored |
| `app/logs/` | Runtime logs, gitignored |
| `pinokio.json` → `version` field | Pinokio schema version, must stay as-is |
| `.req_hash` / `.smart_state.json` | Auto-generated by smart.py |
