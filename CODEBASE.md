# CODEBASE MAP — Lover Clinic AI Video Tools

> **อ่านไฟล์นี้ก่อนเสมอ** ก่อนค้นหา code ใด ๆ ในโปรเจ็กต์นี้
> GitHub: https://github.com/teddyoomz/lover-clinic-ai-video-tools

---

## 1. Project Structure

```
lover-clinic-ai-video-tools/        ← project root (_PROJ_ROOT)
│
├── CODEBASE.md                     ← this file
├── README.md                       ← user documentation
├── pinokio.json                    ← Pinokio metadata (title, icon, version)
├── pinokio.js                      ← Pinokio launcher UI (dynamic menu)
├── install.js                      ← install: smart.py install → fs.link → smart.py torch → start.js
├── start.js                        ← start: smart.py start → app.py → open browser
├── update.js                       ← update: git pull + re-install
├── reset.js                        ← reset: delete app/env
├── fix.js                          ← fix: re-run smart.py install without full reset
├── torch.js                        ← shared Pinokio torch installer helper
├── lc_settings.json                ← [gitignored] persisted user settings (auto-generated)
│
├── app/                            ← all Python app logic lives here
│   ├── app.py                      ← ★ MAIN APP — all UI and processing (1821 lines)
│   ├── smart.py                    ← smart installer: detects GPU, installs correct torch
│   ├── requirements.txt            ← Python dependencies
│   ├── check_deps.py               ← dependency checker helper
│   ├── verify_torch.py             ← torch verification helper
│   ├── logs/
│   │   └── app.log                 ← [gitignored] runtime log
│   └── static/
│       ├── icon.png                ← app icon (embedded as base64 in header)
│       ├── logo.png                ← app logo (embedded as base64 in header)
│       ├── cropper.min.js          ← Cropper.js (downloaded, currently unused)
│       └── cropper.min.css         ← Cropper.js styles (downloaded, currently unused)
│
└── output/                         ← [gitignored] all saved output files
    ├── photo/  video/  remove_bg/  enhance/  resize/  crop/  convert/
```

---

## 2. app/app.py — Section Map (~2800 lines)

### Lines 1–16 · Imports
```
gradio, os, sys, json, tempfile, shutil, subprocess, logging, traceback,
platform, pathlib.Path, datetime, PIL.Image, numpy, cv2, io
```

### Lines 18–33 · Logging Setup
- Log file: `app/logs/app.log`
- Logger name: `"lover-clinic"`

### Lines 35–49 · Compatibility Patch
- Patches `torchvision.transforms.functional_tensor` (removed in torchvision ≥ 0.16)
- Required by `basicsr` / `realesrgan`

### Lines 51–71 · Base64 Image Helpers
- `_b64_img(filename)` → loads from `app/static/` as base64 data-URI
- `_ICON_SRC`, `_LOGO_SRC` → pre-loaded at startup

### Lines 73–83 · Project Paths
- `_PROJ_ROOT` = `app/../..` (project root, above app/)
- `_OUTPUT_ROOT` = `_PROJ_ROOT/output/`
- `DEFAULT_OUT` = string version

### Lines 85–168 · Settings Persistence
| Symbol | Purpose |
|--------|---------|
| `_SETTINGS_FILE` | `_PROJ_ROOT/lc_settings.json` |
| `_SETTINGS_DEFAULTS` | Default values for all saveable controls |
| `_load_settings()` | Load JSON + merge with defaults |
| `_save_settings(cfg)` | Write JSON file |
| `_make_saver(key)` | Returns a 1-arg Gradio `.change()` handler that saves one key |

### Lines 175–235 · AI Model Loaders
| Function | Description |
|----------|-------------|
| `get_device()` | Returns `"cuda"` / `"mps"` / `"cpu"` |
| `get_realesrgan(scale, model_type)` | Lazy-loads RealESRGAN (cached in `_realesrgan_cache`) |
| `get_rembg_session(model_name)` | Lazy-loads rembg session (cached in `_rembg_cache`) |

### Lines 237–371 · AI Processing Functions
| Function | Description |
|----------|-------------|
| `upscale_photo(image, scale, model_type, output_dir, fmt, progress)` | Real-ESRGAN photo upscale |
| `upscale_video(video_path, scale, model_type, output_dir, progress)` | Real-ESRGAN frame-by-frame → H.264 via ffmpeg |

**Video ffmpeg notes:**
- `_find_ffmpeg()` tries `C:\ffmpeg\bin\ffmpeg.exe` first (avoids broken Conda ffmpeg 0xC0000135)
- Two-step: OpenCV mp4v raw → ffmpeg H.264/yuv420p for browser compatibility

### Lines 373–457 · More AI Functions
| Function | Description |
|----------|-------------|
| `remove_background(image, model_choice, bg_option, custom_bg, output_dir, fmt, progress)` | rembg BiRefNet/U2Net |
| `enhance_image(image, upscale_factor, enhance_bg, output_dir, fmt, progress)` | GFPGAN + optional Real-ESRGAN BG |

**rembg model name mapping:**
```python
"BiRefNet — General (Best)"  → "birefnet-general"
"BiRefNet — Portrait"        → "birefnet-portrait"
"U2Net"                      → "u2net"
"RMBG 1.4"                   → "isnet-general-use"
```

### Lines 459–487 · Utility Functions
| Function | Purpose |
|----------|---------|
| `_output_subdir(name)` | Returns `output/{name}/` path, creates if missing |
| `_open_folder(path)` | Opens Explorer at path + PowerShell AppActivate to bring window to front |

### Lines ~488–860 · Crop Tool HTML Generator
- `_make_cropper_html(img)` → returns HTML string for the interactive crop canvas widget
- Embeds image as base64 PNG inside `<img id="lc-crop-img" data-w="…" data-h="…">`
- **No `<script>` tags** — JS is injected separately via `.then(fn=None, js=CROP_INIT_JS)`
- All `.lc-*` CSS is embedded inside the HTML `<style>` block (f-string, `{{` = literal `{`)
- Toolbar layout (3 rows, each with a label column + scrollable content):
  - **สัดส่วน**: Ratio pill-group (Free/1:1/4:3/3:4/16:9/9:16/4:5/5:4/3:2/2:3) + Swap / Center / Clear
  - **เครื่องมือ**: Rotate ↺L / ↻R / 180° + Flip H / V + Reset + Grid ภาพ / Grid Crop + Zoom −/val/+/Fit
  - **Social**: Chips — IG Feed, Square, Story/Reel, YouTube, FB Cover, LinkedIn, Pinterest, OG Image
- Button sizes auto-scale to container width via JS `_fitToolbar()` (beats Gradio CSS via `style.setProperty(..., 'important')`)

### Lines 657–786 · Tool Processing Functions
| Function | Description |
|----------|-------------|
| `_save_dir_row(subdir, label)` | Renders output folder textbox + Open Folder button |
| `_save_image(img, save_dir, prefix, fmt)` | Saves PIL image with `prefix_YYYYMMDD_HHMMSS.ext` |
| `resize_image(image, w, h, ar, filter, output_dir, fmt)` | PIL resize |
| `crop_image(image, coords_str, output_dir, fmt)` | Crop with rotation/flip support |
| `convert_format(image, out_format, quality, output_dir)` | PIL format convert |

**`crop_image` coords_str format:**
```
"x1,y1,x2,y2|r:90|fh:1|fv:0"
              │    │   └── flip vertical (0/1)
              │    └─────── flip horizontal (0/1)
              └──────────── rotation CW degrees (0/90/180/270)

Coordinates are in DISPLAY image space (after rotation+flip applied).
Python applies: rotate(-r, expand=True) → FLIP_LEFT_RIGHT → FLIP_TOP_BOTTOM → crop
```

### Lines 810–1104 · CSS String
- CSS custom properties: `--accent #dc2626`, `--bg-base/card/raised`, `--tx-head/body/muted/faint`
- `--r-xs/sm/md/lg/pill`, `--ease cubic-bezier(.34,1.56,.64,1)`, `--dur 0.22s`
- Design system matches Lover Clinic OPD (`lover-clinic-app.vercel.app`)
- Responsive breakpoints: `@media (max-width: 1024px)` tablet, `@media (max-width: 640px)` mobile
- Crop tool inner CSS (`.lc-*` classes) embedded inside `_make_cropper_html()`

### Lines 1105–1197 · Header & Constants
| Symbol | Description |
|--------|-------------|
| `_header_html()` | Flat dark topbar: icon + logo + subtitle + feature badge pills |
| `WARN_GPU` | HTML warning box on video upscale tab |

### Lines ~1700–2200 · CROP_INIT_JS (JavaScript)
Large JS string, injected via `.then(fn=None, js=CROP_INIT_JS)`.

**State variables:**
```javascript
rotation    // 0 | 90 | 180 | 270  (degrees CW)
flipH, flipV// bool
baseScale   // fit-to-container scale factor (recalculated on rotation)
zoomFactor  // user zoom multiplier — persisted to localStorage('lc_crop_zoom')
lockedRatio // null | {w, h}
gridMode    // 0=off 1=thirds 2=golden-ratio 3=diagonal+cross  (crop box overlay)
imgGridMode // same cycle — full canvas overlay
sx,sy,ex,ey // selection corners in canvas pixels
hasSel, isDown, dragMode  // 'draw' | 'move' | 'resize-<handle>'
_rafId      // requestAnimationFrame handle — throttles redraw to 1/frame
```

**Key JS functions:**
| Function | Purpose |
|----------|---------|
| `computeCanvasDims()` | `{w,h}` after rotation (90/270 swap origW↔origH) × baseScale × zoomFactor |
| `displayDims()` | Logical pixel size of displayed image after rotation |
| `drawTransformed()` | Draws image with rotation + flip via canvas context transforms |
| `redraw()` | Full repaint: image + imgGrid + dim overlay + selection rect + grid + 8 handles |
| `scheduleRedraw()` | RAF-throttled wrapper for redraw — prevents flicker during drag |
| `updateCoords()` | Selection → image coords + encode transforms → `window._lcCropCoords` + hidden textarea |
| `scheduleCoords()` | 60ms debounced wrapper for updateCoords |
| `applyZoom(factor)` | Resize canvas + scale selection proportionally + save to localStorage |
| `applyTransform(type)` | `rot-l`/`rot-r`/`rot-180`/`flip-h`/`flip-v`/`xform-reset` + recalculates baseScale |
| `setRatio(str)` | Lock aspect ratio from ratio buttons |
| `centerAndApplyRatio(w,h)` | Center + fit selection box to given ratio |
| `_fitToolbar()` | Scales toolbar buttons to container width via `style.setProperty('important')` |

**Coordinate flow:**
```
Canvas drag → mouseup
  → updateCoords()
      → sc = 1 / (baseScale * zoomFactor)
      → x0,y0,x1,y1 in display image space
      → coords = "x0,y0,x1,y1|r:rotation|fh:…|fv:…"
      → window._lcCropCoords = coords
      → React native setter → hidden #lc-crop-coords textarea

crop_btn.click  (js= intercept)
  → reads window._lcCropCoords
  → passes to Python crop_image()
```

**React native setter pattern (required for Gradio controlled inputs):**
```javascript
var proto = el.tagName==='TEXTAREA'
  ? window.HTMLTextAreaElement.prototype
  : window.HTMLInputElement.prototype;
Object.getOwnPropertyDescriptor(proto,'value').set.call(el, value);
el.dispatchEvent(new Event('input', {bubbles:true}));
el.dispatchEvent(new Event('change', {bubbles:true}));
```

### Lines ~1608–1808 · _build_download_tab(cfg)
Called inside `build_app()` from the Download Video tab context.

| Inner function | Purpose |
|----------------|---------|
| `_ffmpeg_bin()` | Returns `imageio-ffmpeg` binary path (or `None`) |
| `_base_ydl_opts()` | Builds base yt-dlp options dict (ffmpeg path + JS runtime detection) |
| `_fetch(url)` | Extracts video info, returns quality dropdown choices + `quality_map` state |
| `_download(url, selected, quality_map, save_dir)` | Streaming generator: downloads in background thread, yields real-time progress bar |
| `_pause_toggle()` | Toggles pause state in shared `_current["state"]` dict |
| `_stop_download()` | Sets `stop=True` → raises `_StopDownload(BaseException)` in progress hook |

**Quality labels:** 🏆 Best auto, 📹 4K/2K/1080p/720p/etc., 🎵 Audio MP3
**Progress bar format:** `⬇️  [████████░░░░░░░░░░░░] 40%\n📦 12.3 MB / 30.5 MB   🚀 2.1 MB/s   ⏱ ETA 8s`
**Output dir:** `output/download/` (persisted via `dl_out_dir` setting)

### Lines ~1810–2165 · build_app()
- `cfg = _load_settings()` called once at startup
- All saveable controls use `cfg[key]` as `value=`
- All saveable controls have `.change(_make_saver(key), inputs=[ctrl])` wired

**Tab → function → settings keys:**
| Tab (Thai) | Function | Settings keys |
|------------|----------|---------------|
| 🔬 AI เพิ่มความชัด (ภาพ) | `upscale_photo` | `up_scale, up_model, up_fmt, up_out_dir` |
| 🎬 AI เพิ่มความชัด (วีดีโอ) | `upscale_video` | `vid_scale, vid_model, vid_out_dir` |
| ✂️ AI ลบพื้นหลัง | `remove_background` | `bg_model, bg_option, bg_fmt, bg_out_dir` |
| 🪄 AI ฟื้นฟูภาพ | `enhance_image` | `enh_scale, enh_bg, enh_fmt, enh_out_dir` |
| 📐 ปรับขนาดภาพ | `resize_image` | `res_w, res_h, res_ar, res_filter, res_fmt, res_out_dir` |
| ✂️ ครอปภาพ | `crop_image` | `crop_fmt, crop_out_dir` |
| 🔄 แปลงรูปแบบ | `convert_format` | `conv_fmt, conv_q, conv_out_dir` |
| ⬇️ ดาวน์โหลดวีดีโอ | `_build_download_tab()` | `dl_out_dir` |

---

## 3. Pinokio Launcher Files

### pinokio.js — Dynamic Menu States
```
not installed    → [Install]
installing       → [Installing… spinner]
fixing           → [Fixing… spinner]
installed+idle   → [Start, Update, Fix, Re-install, Reset]
starting (no url)→ [Starting… spinner]
running (url set)→ [Terminal]   ← URL opens in system browser (not Pinokio iframe)
updating         → [Updating… spinner]
resetting        → [Resetting… spinner]
```

### install.js Flow
```
1. shell.run: python smart.py install    GPU detection + pip install
2. fs.link: app/env                      symlink venv for disk savings
3. shell.run: python smart.py torch      force-reinstall correct GPU torch post-symlink
4. script.start: start.js               auto-launch
```

### start.js Flow
```
1. shell.run: python smart.py start      pre-launch hash check + torch smoke test
2. shell.run: python app.py --port {{port}}
   on: event="/(http:\/\/[0-9.:]+)/"  done:true
3. local.set: url = input.event[1]
4. shell.run: python -c "webbrowser.open('{{local.url}}')"
```

### smart.py Modes
```
python smart.py install   → detect GPU, uv pip install, correct torch wheel
python smart.py start     → hash-based dep check, torch smoke test, auto-fix
python smart.py torch     → force-reinstall correct torch for detected GPU
```

---

## 4. Key Dependencies

| Package | Used For |
|---------|---------|
| `gradio >= 6.0.0, < 7.0.0` | Web UI framework |
| `realesrgan` | Photo/video upscaling |
| `gfpgan` + `basicsr` | Face enhancement + backend |
| `rembg` | Background removal |
| `onnxruntime` | Required by rembg |
| `opencv-python-headless` | Video frame extraction + mp4v writer |
| `pillow` | All image I/O and transforms |
| `numpy < 2.0.0` | Pinned for compatibility |
| `devicetorch` | Cross-platform GPU detection |
| `ffmpeg-python` | ffmpeg Python bindings |
| `pydantic == 2.10.6` | Pinned for Gradio compatibility |
| `yt-dlp` | Video download (YouTube, Facebook, TikTok, 1000+ sites) |
| `imageio-ffmpeg` | Portable ffmpeg binary for yt-dlp merge/convert |

**External binary required:** `ffmpeg`
Preferred: `C:\ffmpeg\bin\ffmpeg.exe` — avoids broken Conda ffmpeg (DLL error 0xC0000135)

---

## 5. Known Issues & Workarounds

| Issue | Workaround in code |
|-------|-------------------|
| Conda ffmpeg crashes with 0xC0000135 | `_find_ffmpeg()` tries hardcoded paths first |
| torchvision ≥ 0.16 removed `functional_tensor` | Compat patch at startup |
| `<script>` in `gr.HTML` never executes | Crop JS via `.then(fn=None, js=CROP_INIT_JS)` |
| React controlled inputs ignore `.value=` | Native setter + dispatchEvent pattern |
| `gr.Video(interactive=False)` won't update in Gradio 6 | Removed `interactive=False` from vid_out |
| `fs.link` overwrites CUDA torch with shared CPU | install.js re-runs `smart.py torch` after fs.link |
| Gradio CSS overrides button styles after page load | JS `style.setProperty(..., 'important')` wins over dynamic stylesheets |
| Gradio tab "..." overflow hides tabs | Override `getBoundingClientRect` on `[role=tablist]` → returns width:9999 → all tabs visible (see `GRADIO_OVERRIDE_KNOWLEDGE.md`) |
| Canvas crop flicker during drag | `requestAnimationFrame` throttle via `scheduleRedraw()` |
| Image falls offscreen after rotation | Recalculate `baseScale` from `displayDims()` inside `applyTransform()` |

---

## 6. Files NOT to Edit

| Path | Reason |
|------|--------|
| `lc_settings.json` | Auto-generated at runtime, gitignored |
| `app/env/` | Python venv, gitignored |
| `output/` | User output files, gitignored |
| `app/logs/` | Runtime logs, gitignored |
| `pinokio.json` → `version` field | Pinokio schema version, must stay as-is |
| `app/static/cropper.min.*` | Downloaded library, not actively used |
